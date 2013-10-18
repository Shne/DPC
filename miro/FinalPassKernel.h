#ifndef _FINAL_PASS_KERNEL_H_
#define _FINAL_PASS_KERNEL_H_

#include "Image.h"
#include <vector>
// #include "Miro.h"
#include "Ray.h"

extern "C" 
HitInfo* finalPass(const Image* img, const HitInfo* scatteringMPs, const int scatteringMPsSize, HitInfo* measureHIArray, const Camera* cam);


#endif /* _FINAL_PASS_KERNEL_H_ */