#ifndef _KERNEL_H_
#define _KERNEL_H_

extern "C" 
float* kernel( unsigned int width, unsigned int height, float *result_devPtr );


// __global__
// void kernel_add(float x, float y, float *U);

#endif /* _KERNEL_H_ */
