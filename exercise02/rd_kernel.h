#ifndef _RD_KERNEL_H_
#define _RD_KERNEL_H_

extern "C" 
void rd( unsigned int width, unsigned int height, float *result_devPtr );

static const int TILE_WIDTH = 8;

#endif /* _RD_KERNEL_H_ */
