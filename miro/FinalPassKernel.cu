#include "FinalPassKernel.h"
#include <stdio.h>

#define E (2.7182818284f)

using namespace std;


__global__
void finalPassKernel(const int height, const int width, const HitInfo* dev_scatteringMPs, const int scatteringMPsSize, const HitInfo* hiArray) {
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	// HitInfo* hi = dev_eyeMPs[j*width + i];
	HitInfo hi = hiArray[j];

	// if(hi == NULL) return;

	const float my = 1.3;
	const float sigmaS = 2.6f		* 60; // TranslucentMaterialScale;
	const float sigmaA = 0.0041f 	* 60; //TranslucentMaterialScale;
	const float sigmaT = sigmaS + sigmaA;
	// const float alpha = sigmaS / sigmaT;
	const float sigmaTR = sqrt(3.0f*sigmaA*sigmaT);
	const float lu = 1.0f/sigmaT;
	const float Fdr =  -1.440f/(my*my) + 0.710f*my + 0.668f + 0.0636f*my;
	const float Fdt = 1.0f - Fdr;
	const float A = (1 + Fdr) / (1 - Fdr);
	const float zr = lu;
	const float zv = lu*(1.0f + 4.0f/(3.0f*A));

	// const int scatteringMPsIndex = blockIdx.x*blockDim.x + threadIdx.x;
	const HitInfo sHI = dev_scatteringMPs[i];
	
	const float r2 = (hi.P - sHI.P).length2();
	const float dr = sqrt(r2+zr*zr);
	const float dv = sqrt(r2+zv*zv);
	const float C1 = zr * (sigmaTR + 1.0f/dr);
	const float C2 = zv * (sigmaTR + 1.0f/dv);

	const float dMoOverAlphaPhi = 1.0f/(4.0f*PI) * (C1*(pow(E,-sigmaTR*dr)/dr*dr) + C2*(pow(E,-sigmaTR*dv)/dv*dv));
	const Vector3 MoP = Fdt * dMoOverAlphaPhi * sHI.flux * sHI.r2 * PI;

	// scatteringMPsFlux[i] = MoP;
	hi.flux += MoP;


	
}

extern "C" __host__
HitInfo* finalPass(Image* img, HitInfo* scatteringMPs, int scatteringMPsSize, HitInfo* measureHIArray, Camera* cam) {
	int width = img->width();
	int height = img->height();

	static HitInfo *dev_scatteringMPs, *dev_eyeMPs;
	// static Vector3 *scatteringMPsFlux;
	cudaMalloc((void**)&dev_scatteringMPs, scatteringMPsSize*sizeof(HitInfo));
	// cudaMalloc((void**)&scatteringMPsFlux, width*height*scatteringMPsSize*sizeof(Vector3));
	cudaMalloc((void**)&dev_eyeMPs, width*height*sizeof(HitInfo));
	

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	cudaMemcpy( dev_scatteringMPs, scatteringMPs, scatteringMPsSize*sizeof(HitInfo), cudaMemcpyHostToDevice );
	err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("\nCuda error detected when copying scatteringMPs: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	cudaMemcpy( dev_eyeMPs, measureHIArray, width*height*sizeof(HitInfo), cudaMemcpyHostToDevice );
	err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("\nCuda error detected when copying measureHIArray: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
	// Vector3 *fluxArray = new Vector3[width*height];

	// Kernel block dimensions
	// const dim3 blockDim(8,8);

	// for (int j = 0; j < img->height(); ++j) {
	// 	for (int i = 0; i < img->width(); ++i) {
			// HitInfo hi = measureHIArray[j*width + i];
			// if(hi.t == 0.0f) continue;

			const dim3 dimBlock(16, 16);
			const dim3 dimGrid(scatteringMPsSize/dimBlock.x, width*height/dimBlock.y);

			finalPassKernel<<<dimGrid, dimBlock>>>(height, width, dev_scatteringMPs, scatteringMPsSize, dev_eyeMPs);
			// cout << "after kernel" << endl;
			// cudaMemcpy(fluxArray, scatteringMPsFlux, width*height*scatteringMPsSize*sizeof(Vector3), cudaMemcpyDeviceToHost );
			// Vector3 flux;
			// for(int _j = 0; _j < width*height; _j++) {
			// 	for(int _i = 1; _i < scatteringMPsSize; _i++) {
			// 		measureHIArray[_j].flux += fluxArray[_i][_j];
			// 		// fluxArray[0][_j] += fluxArray[_i][_j];
			// 		// std::cout << fluxArray[_i];
			// 	}
			// }
			// cout << "after sum" << endl;
			// cout << measureHIArray[j*width+i].flux << " ";
			// measureHIArray[j*width+i].flux = flux;
			// cout << measureHIArray[j*width+i].flux << endl;



			const char* error = cudaGetErrorString(cudaPeekAtLastError());
			printf("%s\n", error);
			error = cudaGetErrorString(cudaThreadSynchronize());
			printf("%s\n", error);




			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ) {
				printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}
	// 	}
	// }



	// HitInfo* result = new HitInfo[width*height];
	cudaMemcpy(measureHIArray, dev_eyeMPs, width*height*sizeof(HitInfo), cudaMemcpyDeviceToHost );
	
	



	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected when copying result back: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return measureHIArray;
}

