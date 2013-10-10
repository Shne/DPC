#ifndef _HashGrid_H_
#define _HashGrid_H_


#include "Vector3.h"
#include "Ray.h"
#include <list>

class HashGrid {

public: 
	void initializeGrid(Vector3 minCorner, Vector3 maxCorner, int inputHashSize, float initialRadius);

	unsigned int doHash(const int x, const int y, const int z) const;

	void addHitPoint(HitInfo* hi);

	std::list<HitInfo*> lookup(Vector3 position);
	std::list<HitInfo*> lookup(unsigned int index);

protected:
	double scale;
	float initialRadius;
	int hashSize;
	Vector3 maxBVHCorner, minBVHCorner;
	std::list<HitInfo*>** hash;
};

#endif