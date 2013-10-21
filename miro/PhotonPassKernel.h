#ifndef _PHOTON_PASS_KERNEL_H_
#define _PHOTON_PASS_KERNEL_H_

// #include "Image.h"
// #include <vector>
// #include "Miro.h"
#include <list>
#include "Vector3.cu"
#include "Ray.h"

extern "C" 
HitInfo* photonEyePass(const int hiArraySize, HitInfo* hiArray, const std::list<int> hiIndexList, const Vector3 photonPosition, const Vector3 flux);

extern "C"
HitInfo* photonScatterPass(const int hiArraySize, HitInfo* hiArray, const std::list<int> hiIndexList, const Vector3 photonPosition, const Vector3 flux);


#endif /* _PHOTON_PASS_KERNEL_H_ */