#include "rd_kernel.h"
#include "uint_util.hcu"

#include <stdio.h>

/*
 * Utility function to initialize U and V
 */
__host__
void initializeConcentrations(unsigned int width, unsigned int height, float *U, float *V) {
	float *_U = new float[width*height];
	float *_V = new float[width*height];

	int k = 0;
	int i, j;

	for (i = 0; i < width * height; ++i) {
		_U[k] = 1.0f;
		_V[k++] = 0.0f;
	}

	for (i = (0.48f)*height; i < (0.52f)*height; ++i) {
		for (j = (0.48f)*width; j < (0.52f)*width; ++j) {
			_U[ (i * width + j) ] = 0.5f;
			_V[ (i * width + j) ] = 0.25f;
		}
	}

	// Now perturb the entire grid. Bound the values by [0,1]
	for (k = 0; k < width * height; ++k) {
		if ( _U[k] < 1.0f ) {
			float rRand = 0.02f*(float)rand() / RAND_MAX - 0.01f;
			_U[k] += rRand * _U[k];
		}
		if ( _V[k] < 1.0f ) {
			float rRand = 0.02f*(float)rand() / RAND_MAX - 0.01f;
			_V[k] += rRand * _V[k];
		}
	}

	// Upload initial state U and V to the GPU
	cudaMemcpy( U, _U, width*height*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( V, _V, width*height*sizeof(float), cudaMemcpyHostToDevice );

	delete[] _U;
	delete[] _V;
}

__device__
float laplacian(float* p_array, const uint2 co, const uint2 dim, const float dx) {
	uint2 right_co, left_co, up_co, down_co;
	
	right_co = make_uint2(co.x+1, co.y);
	left_co = make_uint2(co.x-1, co.y);
	down_co = make_uint2(co.x, co.y+1);
	up_co = make_uint2(co.x, co.y-1);

	if(right_co.x >= dim.x) {
		 right_co.x -= dim.x;
	}
	if(left_co.x == ((uint)0)-1) {
		left_co.x += dim.x;
	}
	if(up_co.y == ((uint)0)-1) {
		up_co.y += dim.y;
	}
	if(down_co.y >= dim.y) {
		down_co.y -= dim.y;
	}


	float right = p_array[co_to_idx(right_co, dim)];
	float left = p_array[co_to_idx(left_co, dim)];
	float up = p_array[co_to_idx(up_co, dim)];
	float down = p_array[co_to_idx(down_co, dim)];
	float p = p_array[co_to_idx(co, dim)];

	return (left + right + up + down - 4.0f*p) / dx*dx;
}

/*
 * Kernel for the reaction-diffusion model
 * This kernel is responsible for updating 'U' and 'V'
 */
__global__
void rd_kernel(const unsigned int width, const unsigned int height,
					const float dt, const float dx, const float Du, const float Dv,
					const float F, const float k, float *U, float *V,
					float *newU, float *newV) {
	// extern __shared__ float U_V_Results[];
	// float *newU = &U_V_Results[0];
	// float *newV = &U_V_Results[width*height];

	// Coordinate of the current pixel (for this thread)
	const uint2 co = make_uint2( blockIdx.x*blockDim.x + threadIdx.x,
						 blockIdx.y*blockDim.y + threadIdx.y );
	
	// Linear index of the current pixel
	const unsigned int idx = co.y*width + co.x;

	// for(int i = 0; i < width/TILE_WIDTH; i++) {
		float u = U[idx];
		float v = V[idx];
		uint2 dim = make_uint2(width, height);

		newU[idx] += dt*(Du * laplacian(U, co, dim, dx) - u * v*v + F*(1.0f-u));
		newV[idx] += dt*(Dv * laplacian(V, co, dim, dx) + u * v*v - (F+k)*v);

		// __syncthreads();
		U[idx] = newU[idx];
		V[idx] = newV[idx];
	// }
}




/*
 * Wrapper for the reaction-diffusion kernel. 
 * Called every frame by 'display'
 * 'result_devPtr' is a floating buffer used for visualization.
 * Make sure whatever needs visualization goes there.
 */
extern "C" __host__
void rd(unsigned int width, unsigned int height, float *result_devPtr) {
	// Create buffers for 'U' and 'V' at first pass
	static float *U, *V;
	static float *newU, *newV;
	static bool first_pass = true;

	if (first_pass){
		// Allocate device memory for U and V
		cudaMalloc((void**)&U, width*height*sizeof(float));
		cudaMalloc((void**)&V, width*height*sizeof(float));
		cudaMalloc((void**)&newU, width*height*sizeof(float));
		cudaMalloc((void**)&newV, width*height*sizeof(float));
 
		// Check for Cuda errors
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		// Initialize U and V on the CPU and upload to the GPU
		initializeConcentrations( width, height, U, V );
		cudaMemcpy( newU, U, width*height*sizeof(float), cudaMemcpyDeviceToDevice );
		cudaMemcpy( newV, V, width*height*sizeof(float), cudaMemcpyDeviceToDevice );
		// initializeConcentrations( width, height, newU, newV );

		// Make sure we never get in here again...
		first_pass = false;
	}

	// Kernel block dimensions
	const dim3 blockDim(16,16);

	// Verify input image dimensions
	if (width%blockDim.x || height%blockDim.y) {
		printf("\nImage width and height must be a multiple of the block dimensions\n");
		exit(1);
	}

	// Experiment with different settings of these constants
	const float dt = 0.5f;
	const float dx = 2.0f;
	const float Du = 0.0004f*((width*height)/100.0f);
	const float Dv = 0.0002f*((width*height)/100.0f);
	const float F = 0.012f; 
	const float k = 0.052f;

	// Invoke kernel (update U and V)
	rd_kernel<<< dim3(width/blockDim.x, height/blockDim.y), blockDim>>>( width, height, dt, dx, Du, Dv, F, k, U, V, newU, newV);

	// Check for errors
	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'rd_kernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	// cudaMemcpy( U, newU, width*height*sizeof(float), cudaMemcpyDeviceToDevice );
	// cudaMemcpy( V, newV, width*height*sizeof(float), cudaMemcpyDeviceToDevice );

	// For visualization we use a 'float1' image. You can use either 'U' or 'V'.
	cudaMemcpy( result_devPtr, U, width*height*sizeof(float), cudaMemcpyDeviceToDevice );
}
