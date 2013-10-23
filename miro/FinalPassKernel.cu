#include "FinalPassKernel.h"
#include <stdio.h>
// #include "cuda_profiler_api.h"

#define E (2.7182818284f)

using namespace std;


texture <float4> scatteringPositions_tex;
texture <float4> scatteringFlux_tex;
texture <float> scatteringR2_tex;



__global__
void finalPassKernel(const int height, const int width, /*const HitInfo* dev_scatteringMPs,*/ const int scatteringMPsSize, const Vector3 hiP, Vector3 *scatteringMPsFlux,
                     const float sigmaTR, const float Fdt, const float zr, const float zv) {
	
	// const HitInfo hi = hiArray[hiIndex];

/*
	const float my = 1.3;
	const float sigmaS = 2.6f		* translucentMaterialScale;
	const float sigmaA = 0.0041f 	* translucentMaterialScale;
	const float sigmaT = sigmaS + sigmaA;
	// const float alpha = sigmaS / sigmaT;
	const float sigmaTR = sqrt(3.0f*sigmaA*sigmaT);
	const float lu = 1.0f/sigmaT;
	const float Fdr =  -1.440f/(my*my) + 0.710f*my + 0.668f + 0.0636f*my;
	const float Fdt = 1.0f - Fdr;
	const float A = (1 + Fdr) / (1 - Fdr);
	const float zr = lu;
	const float zv = lu*(1.0f + 4.0f/(3.0f*A));
*/

/*
	//PRECOMPUTED VALUES
	// const float my = 1.3;
	// const float sigmaS = 156;
	// const float sigmaA = 0.246;
	// const float sigmaT = 156.246;
	const float sigmaTR = 10.73822834549535922654;
	// const float lu = 0.00640016384419441138;
	// const float Fdr = 0.82160899408284023669;
	const float Fdt = 0.17839100591715976331;
	// const float A = 10.21132755386080971017;
	const float zr = 0.00640016384419441138;
	const float zv = 0.00723585849283319265;

*/


	// MULTIPLE SCATTER
	const int scatteringMPsIndex = blockIdx.x*blockDim.x + threadIdx.x;
	// const HitInfo sHI = dev_scatteringMPs[scatteringMPsIndex];
	const float4 scatteringP_f = tex1Dfetch(scatteringPositions_tex, scatteringMPsIndex);
	const float4 scatteringFlux_f = tex1Dfetch(scatteringFlux_tex, scatteringMPsIndex);
	const float scatteringR2 = tex1Dfetch(scatteringR2_tex, scatteringMPsIndex);

	const Vector3 scatteringP = Vector3(scatteringP_f.x, scatteringP_f.y, scatteringP_f.z);
	const Vector3 scatteringFlux = Vector3(scatteringFlux_f.x, scatteringFlux_f.y, scatteringFlux_f.z);


	// localScatteringMPs[threadIdx.x] = dev_scatteringMPs[scatteringMPsIndex];
	// __syncthreads();
	
	const float r2 = (hiP - scatteringP).length2();
	const float dr = sqrtf(r2+zr*zr);
	const float dv = sqrtf(r2+zv*zv);
	const float C1 = zr * (sigmaTR + 1.0f/dr);
	const float C2 = zv * (sigmaTR + 1.0f/dv);

	const float dMoOverAlphaPhi = 1.0f/(4.0f*PI) * (C1*(powf(E,-sigmaTR*dr)/dr*dr) + C2*(powf(E,-sigmaTR*dv)/dv*dv));
	const Vector3 MoP = Fdt * dMoOverAlphaPhi * scatteringFlux * scatteringR2 * PI;

	scatteringMPsFlux[scatteringMPsIndex] = MoP;
	// hi.flux += MoP;	
}










extern "C" __host__
HitInfo* finalPass(const int width, const int height, const HitInfo* scatteringMPs, const int scatteringMPsSize, HitInfo* measureHIArray, const float translucentMaterialScale) {
	// cudaProfilerStart();



	//split scatteringMPs into 3 arrays of position, flux and r2. to be able to use textures
	float4 scatteringPositions[scatteringMPsSize];
	float4 scatteringFlux[scatteringMPsSize];
	float scatteringR2[scatteringMPsSize];
	for(int i = 0; i<scatteringMPsSize; i++) {
		HitInfo sHI = scatteringMPs[i];
		scatteringPositions[i] = make_float4(sHI.P.x, sHI.P.y, sHI.P.z, 0.0f);
		scatteringFlux[i] = make_float4(sHI.flux.x, sHI.flux.y, sHI.flux.z, 0.0f);
		scatteringR2[i] = scatteringMPs[i].r2;
	}

	static float4 *dev_scatteringPositions, *dev_scatteringFlux;
	static float *dev_scatteringR2;
	cudaMalloc((void**)&dev_scatteringPositions, scatteringMPsSize*sizeof(float4));
	cudaMalloc((void**)&dev_scatteringFlux, scatteringMPsSize*sizeof(float4));
	cudaMalloc((void**)&dev_scatteringR2, scatteringMPsSize*sizeof(float));

	std::cout << "allocated" << std::endl;

	cudaMemcpy( dev_scatteringPositions, scatteringPositions, scatteringMPsSize*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_scatteringFlux, scatteringFlux, scatteringMPsSize*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_scatteringR2, scatteringR2, scatteringMPsSize*sizeof(float), cudaMemcpyHostToDevice );

	std::cout << "Memcpy'ed" << std::endl;

	cudaBindTexture( NULL, scatteringPositions_tex, dev_scatteringPositions, scatteringMPsSize*sizeof(float4));
	cudaBindTexture( NULL, scatteringFlux_tex, dev_scatteringFlux, scatteringMPsSize*sizeof(float4));
	cudaBindTexture( NULL, scatteringR2_tex, dev_scatteringR2, scatteringMPsSize*sizeof(float));

	std::cout << "Textures bound" << std::endl;

	// static HitInfo *dev_scatteringMPs;//, *dev_eyeMPs;
	static Vector3 *scatteringMPsFlux;
	// cudaMalloc((void**)&dev_scatteringMPs, scatteringMPsSize*sizeof(HitInfo));
	cudaMalloc((void**)&scatteringMPsFlux, scatteringMPsSize*sizeof(Vector3));
	// cudaMalloc((void**)&dev_eyeMPs, width*height*sizeof(HitInfo));
	

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	// cudaMemcpy( dev_scatteringMPs, scatteringMPs, scatteringMPsSize*sizeof(HitInfo), cudaMemcpyHostToDevice );
	// cudaMemcpy( dev_eyeMPs, measureHIArray, width*height*sizeof(HitInfo), cudaMemcpyHostToDevice );

	Vector3 *perPixelFlux = new Vector3[scatteringMPsSize];


	//VALUES FOR DIPOLE DIFFUSION MULTIPLE SCATTER
	const float my = 1.3;
	const float sigmaS = 2.6f		* translucentMaterialScale;
	const float sigmaA = 0.0041f 	* translucentMaterialScale;
	const float sigmaT = sigmaS + sigmaA;
	// const float alpha = sigmaS / sigmaT;
	const float sigmaTR = sqrt(3.0f*sigmaA*sigmaT);
	const float lu = 1.0f/sigmaT;
	const float Fdr =  -1.440f/(my*my) + 0.710f*my + 0.668f + 0.0636f*my;
	const float Fdt = 1.0f - Fdr;
	const float A = (1 + Fdr) / (1 - Fdr);
	const float zr = lu;
	const float zv = lu*(1.0f + 4.0f/(3.0f*A));

	std::cout << "scatteringMPsSize: " << scatteringMPsSize << " total size: " << scatteringMPsSize*sizeof(HitInfo) << std::endl;

	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			HitInfo hi = measureHIArray[j*width + i];
			if(hi.material == NULL) continue;

			dim3 dimBlock(64);
			dim3 dimGrid = scatteringMPsSize/dimBlock.x;

			finalPassKernel<<<dimGrid, dimBlock>>>(height, width, /*dev_scatteringMPs,*/ scatteringMPsSize, hi.P, scatteringMPsFlux, sigmaTR, Fdt, zr, zv);

			err = cudaGetLastError();
			if( err != cudaSuccess ) {
				printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}
			// cout << "after kernel" << endl;
			cudaMemcpy(perPixelFlux, scatteringMPsFlux, scatteringMPsSize*sizeof(Vector3), cudaMemcpyDeviceToHost );
			Vector3 flux;
			for(int _i = 0; _i < scatteringMPsSize; _i++) {
				flux += perPixelFlux[_i];
				// std::cout << perPixelFlux[_i];
			}
			// cout << "after sum" << endl;
			// cout << measureHIArray[j*width+i].flux << " ";
			measureHIArray[j*width+i].flux = flux;
			// cout << measureHIArray[j*width+i].flux << endl;

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ) {
				printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}
		}
		printf("Kernel Progress: %.3f%%\r", (j)/float(height) *100.0f);
		fflush(stdout);
	}



	// HitInfo* result = new HitInfo[width*height];
	// cudaMemcpy(measureHIArray, dev_eyeMPs, width*height*sizeof(HitInfo), cudaMemcpyDeviceToHost );
	
	



	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
	// cudaDeviceReset();
	// cudaProfilerStop();

	cudaFree(scatteringMPsFlux);
	// cudaFree(dev_scatteringMPs);

	cudaUnbindTexture(scatteringPositions_tex);
	cudaUnbindTexture(scatteringFlux_tex);
	cudaUnbindTexture(scatteringR2_tex);
	
	return measureHIArray;
}

