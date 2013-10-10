#ifndef CSE168_BVH_H_INCLUDED
#define CSE168_BVH_H_INCLUDED

#include "Miro.h"
#include "Object.h"

class BVH
{
public:
	BVH();

	void build(Objects* objs, int depth);

	bool intersect(HitInfo& result, const Ray& ray,
						float tMin = 0.0f, float tMax = MIRO_TMAX);

	void calcCorners(Vector3 corners[2], Objects* objs);

	float getCostOfSplit(Objects* parent, Objects* left, Objects* right);

	static unsigned int totalNodes;
	static unsigned int leafNodes;
	static unsigned int rayBVIntersections;
	Vector3 corners[2];

protected:
	Objects* m_objects;
	BVH* leftBVH;
	BVH* rightBVH;
	
	int static const maxDepth = 100;
	bool isLeaf;
};

#endif // CSE168_BVH_H_INCLUDED
