#ifndef _HashGrid_H_
#define _HashGrid_H_


#include "Vector3.cu"
#include "Ray.h"
#include <list>

class HashGrid {

public: 
	void initializeGrid(Vector3 minCorner, Vector3 maxCorner, int inputHashSize, float initialRadius);

	unsigned int doHash(const int x, const int y, const int z) const;

	void addHitPoint(Vector3 P, int index);

	std::list<int> lookup(Vector3 position);
	std::list<int> lookup(unsigned int index);

protected:
	double scale;
	float initialRadius;
	int hashSize;
	Vector3 maxBVHCorner, minBVHCorner;
	std::list<int>** hash;
};

#endif