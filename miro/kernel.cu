#include "kernel.h"
#include <stdio.h>



__global__
void kernel_add(unsigned int width, unsigned int height, float *U) {
	// Coordinate of the current pixel (for this thread)
	const uint2 co = make_uint2( blockIdx.x*blockDim.x + threadIdx.x,
						 blockIdx.y*blockDim.y + threadIdx.y );
	
	// Linear index of the current pixel
	const unsigned int idx = co.y*width + co.x;

	U[idx] += 1.0f;
}


extern "C" __host__
float* kernel(unsigned int width, unsigned int height, float *result_devPtr) {
	static float *U;
	cudaMalloc((void**)&U, width*height*sizeof(float));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	//initialize data
	float *_U = new float[width*height];
	for(int i=0; i<width*height; i++) {
		_U[i] = 0.0f;
	}
	cudaMemcpy( U, _U, width*height*sizeof(float), cudaMemcpyHostToDevice );
	delete[] _U;

	// Kernel block dimensions
	const dim3 blockDim(16,16);

	// Verify input image dimensions
	if (width%blockDim.x || height%blockDim.y) {
		printf("\nImage width and height must be a multiple of the block dimensions\n");
		exit(1);
	}

	// Invoke kernel (update U and V)
	kernel_add<<<dim3(width/blockDim.x, height/blockDim.y), blockDim>>>( width, height, U);

	// Check for errors
	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'rd_kernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	float* result = new float[width*height];
	cudaMemcpy(result, U, width*height*sizeof(float), cudaMemcpyDeviceToHost );

	return result;
	// cudaMemcpy(result_devPtr, U, width*height*sizeof(float), cudaMemcpyDeviceToDevice );
}

