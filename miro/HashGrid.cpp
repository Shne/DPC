#include "HashGrid.h"
#include <list>

void
HashGrid::initializeGrid(Vector3 minCorner, Vector3 maxCorner, int inputHashSize, float inputInitialRadius) {
	initialRadius = inputInitialRadius;
	maxBVHCorner = maxCorner;// + initialRadius;
	minBVHCorner = minCorner;// - initialRadius;
	scale = 1.0/(2.0*initialRadius);

	hashSize = inputHashSize;

	hash = new std::list<int>*[hashSize];
	for(unsigned int i=0; i<hashSize; i++) {
		hash[i] = new std::list<int>();
	}
}

unsigned int
HashGrid::doHash(const int x, const int y, const int z) const {
	return (unsigned int)((x*73856093)^(y*19349663)^(z*83492791)) % hashSize;
};


void
HashGrid::addHitPoint(Vector3 P, int index){
	// Vector3 BMin = ((hi->P - initialRadius) - minBVHCorner) * scale;
	// Vector3 BMax = ((hi->P + initialRadius) - minBVHCorner) * scale;
	Vector3 BMin = P - initialRadius;
	Vector3 BMax = P + initialRadius;
	// for (int iz = abs(int(BMin.z)); iz <= abs(int(BMax.z)); iz++) {
	// 	for (int iy = abs(int(BMin.y)); iy <= abs(int(BMax.y)); iy++) {
	// 		for (int ix = abs(int(BMin.x)); ix <= abs(int(BMax.x)); ix++) {

	// int intBMinz = static_cast<int>(BMin.z);
	// int intBMiny = static_cast<int>(BMin.y);
	// int intBMinx = static_cast<int>(BMin.x);
	// int intBMaxz = static_cast<int>(BMax.z);
	// int intBMaxy = static_cast<int>(BMax.y);
	// int intBMaxx = static_cast<int>(BMax.x);


	for (int iz = static_cast<int>(BMin.z); iz <= static_cast<int>(BMax.z); iz++) {
		if(iz < minBVHCorner.z) continue;
		for (int iy = static_cast<int>(BMin.y); iy <= static_cast<int>(BMax.y); iy++) {
			if(iy < minBVHCorner.y) continue;
			for (int ix = static_cast<int>(BMin.x); ix <= static_cast<int>(BMax.x); ix++) {
				if(ix < minBVHCorner.x) continue;
				unsigned int hv = doHash(ix,iy,iz); 
				hash[hv]->push_front(index);
			}
		}
	}
}

std::list<int>
HashGrid::lookup(Vector3 position) {
	unsigned int index = doHash(position.x, position.y, position.z);
	return lookup(index);
}

std::list<int>
HashGrid::lookup(unsigned int index) {
	return (*hash[index]);
}