#include "FinalPassKernel.h"
#include <stdio.h>

#define E (2.7182818284f)


__global__
void finalPassKernel(int height, int width, HitInfo* dev_scatteringMPs, HitInfo* hi) {
	
	// int i = blockIdx.x*blockDim.x + threadIdx.x;
	// int j = blockIdx.y*blockDim.y + threadIdx.y;
	// HitInfo* hi = dev_eyeMPs[j*width + i];

	if(hi == NULL) return;

	const float my = 1.3;
	const float sigmaS = 2.6		* 60; // TranslucentMaterialScale;
	const float sigmaA = 0.0041 	* 60; //TranslucentMaterialScale;
	const float sigmaT = sigmaS + sigmaA;
	// const float alpha = sigmaS / sigmaT;
	const float sigmaTR = sqrt(3.0*sigmaA*sigmaT);
	const float lu = 1.0/sigmaT;
	const float Fdr =  -1.440/(my*my) + 0.710*my + 0.668 + 0.0636*my;
	const float Fdt = 1.0 - Fdr;
	const float A = (1 + Fdr) / (1 - Fdr);
	const float zr = lu;
	const float zv = lu*(1.0 + 4.0/(3.0*A));

	// MULTIPLE SCATTER
	// int scatteringMPsSize = scatteringMPs.size();
	// for(int i=0; i<scatteringMPsSize; i++) {
	// 	HitInfo* sHI = scatteringMPs[i];
	HitInfo sHI = dev_scatteringMPs[blockIdx.x*blockDim.x + threadIdx.x];

	
	float r2 = (hi->P - sHI.P).length2();
	float dr = sqrt(r2+zr*zr);
	float dv = sqrt(r2+zv*zv);
	float C1 = zr * (sigmaTR + 1.0f/dr);
	float C2 = zv * (sigmaTR + 1.0f/dv);

	float dMoOverAlphaPhi = 1.0f/(4.0f*PI) * (C1*(pow(E,-sigmaTR*dr)/dr*dr) + C2*(pow(E,-sigmaTR*dv)/dv*dv));
	Vector3 MoP = Fdt * dMoOverAlphaPhi * sHI.flux * sHI.r2 * PI;
	hi->flux += MoP;


	
}

extern "C" __host__
HitInfo* finalPass(Image* img, HitInfo* scatteringMPs, int scatteringMPsSize, HitInfo* measureHIArray, Camera* cam) {
	int width = img->width();
	int height = img->height();

	static HitInfo *dev_scatteringMPs, *dev_eyeMPs;
	cudaMalloc((void**)&dev_scatteringMPs, scatteringMPsSize*sizeof(HitInfo));
	cudaMalloc((void**)&dev_eyeMPs, width*height*sizeof(HitInfo));
	// cudaMalloc((void**)&dev_hi, sizeof(HitInfo*));
	// cudaMalloc((void**)&dev_eyeMPs, eyeMPs->size()*sizeof(HitInfo*));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	cudaMemcpy( dev_scatteringMPs, scatteringMPs, scatteringMPsSize*sizeof(HitInfo), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_eyeMPs, measureHIArray, width*height*sizeof(HitInfo), cudaMemcpyHostToDevice );

	// Kernel block dimensions
	// const dim3 blockDim(16,16);

	for (int j = 0; j < img->height(); ++j) {
		for (int i = 0; i < img->width(); ++i) {
			HitInfo hi = measureHIArray[j*img->width() + i];
			if(hi.t == 0.0f) continue;
			// finalPassKernel<<<dim3(width/blockDim.x, height/blockDim.y), blockDim>>>(height, width, dev_scatteringMPs, hi);
			finalPassKernel<<<64, 256>>>(height, width, dev_scatteringMPs, &hi);

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ) {
				printf("\nCuda error detected in 'finalPassKernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}
		}
	}



	HitInfo* result = new HitInfo[width*height];
	cudaMemcpy(result, dev_eyeMPs, width*height*sizeof(HitInfo), cudaMemcpyDeviceToHost );

	return result;
}

