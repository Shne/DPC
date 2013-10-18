#include "FinalPassKernel.h"
#include <stdio.h>
#include "cuda_profiler_api.h"

#define E (2.7182818284f)

using namespace std;


__global__
void finalPassKernel(const int height, const int width, const HitInfo* dev_scatteringMPs, const int scatteringMPsSize, HitInfo* hiArray, const int index) {
	
	// int j = blockIdx.x*blockDim.x + threadIdx.x;
	// int i = blockIdx.y*blockDim.y + threadIdx.y;
	// HitInfo hi = hiArray[j*width+i];

	HitInfo hi = hiArray[index];

	// if(hi == NULL) return;

	// const float my = 1.3;
	// const float sigmaS = 2.6f		* 60; // TranslucentMaterialScale;
	// const float sigmaA = 0.0041f 	* 60; //TranslucentMaterialScale;
	// const float sigmaT = sigmaS + sigmaA;
	// // const float alpha = sigmaS / sigmaT;
	// const float sigmaTR = sqrt(3.0f*sigmaA*sigmaT);
	// const float lu = 1.0f/sigmaT;
	// const float Fdr =  -1.440f/(my*my) + 0.710f*my + 0.668f + 0.0636f*my;
	// const float Fdt = 1.0f - Fdr;
	// const float A = (1 + Fdr) / (1 - Fdr);
	// const float zr = lu;
	// const float zv = lu*(1.0f + 4.0f/(3.0f*A));


	//PRECOMPUTED VALUES
	// const float my = 1.3;
	// const float sigmaS = 156;
	// const float sigmaA = 0.246;
	// const float sigmaT = 156.246;
	const float sigmaTR = 10.73822834549535922654;
	// const float lu = 0.00640016384419441138;
	// const float Fdr =  0.82160899408284023669;
	const float Fdt = 0.17839100591715976331;
	// const float A = 10.21132755386080971017;
	const float zr = 0.00640016384419441138;
	const float zv = 0.00723585849283319265;



	// MULTIPLE SCATTER
	// int scatteringMPsSize = scatteringMPs.size();
	// for(int i=0; i<scatteringMPsSize; i++) {
	// 	HitInfo* sHI = scatteringMPs[i];
	const int scatteringMPsIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(scatteringMPsIndex >= scatteringMPsSize) {return;}
	const HitInfo sHI = dev_scatteringMPs[scatteringMPsIndex];
	// const HitInfo sHI = dev_scatteringMPs[index];

	
	const float r2 = (hi.P - sHI.P).length2();
	const float dr = sqrt(r2+zr*zr);
	const float dv = sqrt(r2+zv*zv);
	const float C1 = zr * (sigmaTR + 1.0f/dr);
	const float C2 = zv * (sigmaTR + 1.0f/dv);

	const float dMoOverAlphaPhi = 1.0f/(4.0f*PI) * (C1*(pow(E,-sigmaTR*dr)/dr*dr) + C2*(pow(E,-sigmaTR*dv)/dv*dv));
	const Vector3 MoP = Fdt * dMoOverAlphaPhi * sHI.flux * sHI.r2 * PI;

	if(sHI.flux.x > 0.0f) {
		int *hats = (int*)0xffffffff;
		*hats = 12;
	}

	// scatteringMPsFlux[scatteringMPsIndex] = MoP;
	hiArray[index].flux += MoP;
}

extern "C" __host__
HitInfo* finalPass(const Image* img, const HitInfo* scatteringMPs, const int scatteringMPsSize, HitInfo* measureHIArray, const Camera* cam) {
	// cudaProfilerStart();

	int width = img->width();
	int height = img->height();

	static HitInfo *dev_scatteringMPs, *dev_eyeMPs;
	// Vector3 *scatteringMPsFlux;
	cudaMalloc((void**)&dev_scatteringMPs, scatteringMPsSize*sizeof(HitInfo));
	// cudaMalloc((void**)&scatteringMPsFlux, scatteringMPsSize*sizeof(Vector3));
	cudaMalloc((void**)&dev_eyeMPs, width*height*sizeof(HitInfo));
	

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	for(int derp=0;derp<scatteringMPsSize;derp++){
		if(scatteringMPs[derp].flux.x > 0.0f) {
			cout << "HURR" << endl;
		}
	}

	cudaMemcpy( dev_scatteringMPs, scatteringMPs, scatteringMPsSize*sizeof(HitInfo), cudaMemcpyHostToDevice );
	// Check for errors
	err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	cudaMemcpy( dev_eyeMPs, measureHIArray, width*height*sizeof(HitInfo), cudaMemcpyHostToDevice );
	// Check for errors
	err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	// Vector3 *perPixelFlux = new Vector3[scatteringMPsSize];

	for (int j = 0; j < img->height(); ++j) {
		for (int i = 0; i < img->width(); ++i) {
			HitInfo hi = measureHIArray[j*width + i];
			if(hi.material == NULL) continue;

			dim3 dimBlock(64);
			dim3 dimGrid = scatteringMPsSize/dimBlock.x;

			finalPassKernel<<<dimGrid, dimBlock>>>(height, width, dev_scatteringMPs, scatteringMPsSize, dev_eyeMPs, j*width + i);

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ) {
				printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}
		}
	}


	// for (int i = 0; i < scatteringMPsSize; i++) {
	// 	dim3 dimBlock(16,16);
	// 	dim3 dimGrid(width/dimBlock.x, height/dimBlock.y);

	// 	finalPassKernel<<<dimGrid, dimBlock>>>(height, width, dev_scatteringMPs, scatteringMPsSize, dev_eyeMPs, i);
	// 	// cout << "after kernel" << endl;
	// 	// cudaMemcpy(perPixelFlux, scatteringMPsFlux, scatteringMPsSize*sizeof(Vector3), cudaMemcpyDeviceToHost );
	// 	// Vector3 flux;
	// 	// for(int _i = 0; _i < scatteringMPsSize; _i++) {
	// 	// 	flux += perPixelFlux[_i];
	// 	// 	// std::cout << perPixelFlux[_i];
	// 	// }
	// 	// cout << "after sum" << endl;
	// 	// cout << measureHIArray[j*width+i].flux << " ";
	// 	// measureHIArray[j*width+i].flux = flux;
	// 	// cout << measureHIArray[j*width+i].flux << endl;

	// 	// Check for errors
	// 	err = cudaGetLastError();
	// 	if( err != cudaSuccess ) {
	// 		printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
	// 		exit(1);
	// 	}
	// }



	// HitInfo* result = new HitInfo[width*height];
	cudaMemcpy(measureHIArray, dev_eyeMPs, width*height*sizeof(HitInfo), cudaMemcpyDeviceToHost );
	


	cudaFree(dev_eyeMPs);
	cudaFree(dev_scatteringMPs);


	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
	// cudaProfilerStop();
	return measureHIArray;
}

