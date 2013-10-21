#include "PhotonPassKernel.h"
#include <stdio.h>
// #include <cmath>
// #include "cuda_profiler_api.h"

#define ALPHA (0.7f)

using namespace std;


__global__
void photonEyePassKernel(HitInfo* dev_hiArray, const int* dev_hiIndexArray, const int hiIndexListSize, const Vector3 photonPosition, const Vector3 flux) {
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index >= hiIndexListSize) return;
	const int hiIndex = dev_hiIndexArray[index];
	HitInfo hi = dev_hiArray[hiIndex];

	const float distance2 = (hi.P - photonPosition).length2();
		
	if(distance2 < hi.r2) {
		const float g = (hi.photons*ALPHA+ALPHA) 
				   / (hi.photons*ALPHA+1.0f);
		hi.r2 = hi.r2*g;
		hi.photons++;
		hi.flux += flux * (1.0f/PI) * g;
	}
	dev_hiArray[hiIndex] = hi;	
}

extern "C" __host__
HitInfo* photonEyePass(const int hiArraySize, HitInfo* hiArray, std::list<int> hiIndexList, const Vector3 photonPosition, const Vector3 flux) {
	int hiIndexListSize = hiIndexList.size();
	if(hiIndexListSize == 0) {
		return hiArray;
	}
	std::cout << hiIndexListSize << " ";

	static HitInfo *dev_hiArray;
	static int *dev_hiIndexArray;
	if(cudaMalloc((void**)&dev_hiArray, hiArraySize*sizeof(HitInfo)) != cudaSuccess) {
		std::cout << "HURR" << std::endl;
	};
	cudaMalloc((void**)&dev_hiIndexArray, hiIndexListSize*sizeof(int));
	

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	int hiIndexArray[hiIndexListSize];
	int i = 0;
	for(std::list<int>::const_iterator iter = hiIndexList.begin(); iter != hiIndexList.end(); iter++) {
		hiIndexArray[i++] = (*iter);
	}

	cudaMemcpy( dev_hiArray, hiArray, hiArraySize*sizeof(HitInfo), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_hiIndexArray, hiIndexArray, hiIndexListSize*sizeof(int), cudaMemcpyHostToDevice );
	err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("\nCuda error detected in 'photonEyePassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}



	dim3 dimBlock(128);
	int herp = hiIndexListSize >= dimBlock.x ? hiIndexListSize : dimBlock.x;
	dim3 dimGrid = herp/dimBlock.x;	
	photonEyePassKernel<<<dimGrid, dimBlock>>>(dev_hiArray, dev_hiIndexArray, hiIndexListSize, photonPosition, flux);
	err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("\nCuda error detected in 'photonEyePassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	cudaMemcpy( hiArray, dev_hiArray, hiArraySize*sizeof(HitInfo), cudaMemcpyDeviceToHost );
	err = cudaGetLastError();
	if( err != cudaSuccess ) {
		printf("\nCuda error detected in 'photonEyePassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
	
	cudaFree(dev_hiArray);
	cudaFree(dev_hiIndexArray);

	return hiArray;
}













extern "C" __host__
HitInfo* photonScatterPass(const int hiArraySize, HitInfo* hiArray, const std::list<int> hiList, const Vector3 photonPosition, const Vector3 flux) {

}