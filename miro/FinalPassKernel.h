#ifndef _FINAL_PASS_KERNEL_H_
#define _FINAL_PASS_KERNEL_H_

#include "Image.h"
#include <vector>
// #include "Miro.h"
#include "Ray.h"

extern "C" 
HitInfo* finalPass(Image* img, HitInfo* scatteringMPs, int scatteringMPsSize, HitInfo* measureHIArray, Camera* cam);


#endif /* _FINAL_PASS_KERNEL_H_ */