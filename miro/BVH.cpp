#include <algorithm>
#include "BVH.h"
#include "Ray.h"
#include "Console.h"

using namespace std;

uint BVH::totalNodes;
uint BVH::leafNodes;
uint BVH::rayBVIntersections;

BVH::BVH() :
	leftBVH(0), rightBVH(0) 
	{}

void
BVH::calcCorners(Vector3 corners[2], Objects * objs) {
	Vector3* minCorner = new Vector3(infinity);
	Vector3* maxCorner = new Vector3(-infinity);

	for(int i = 0; i < objs->size(); i++ ) {
		objs->at(i)->calcBVCoords();
		Vector3 objectMinCorner = objs->at(i)->minCoords();
		Vector3 objectMaxCorner = objs->at(i)->maxCoords();

		for(int j=0; j<3; j++) {
			if(objectMinCorner[j] < (*minCorner)[j]) 
				(*minCorner)[j] = objectMinCorner[j];
			if(objectMaxCorner[j] > (*maxCorner)[j]) 
				(*maxCorner)[j] = objectMaxCorner[j];
		}
	}

	if(objs->size() == 0) {
		delete minCorner;
		delete maxCorner;
		minCorner = new Vector3(0.0);
		maxCorner = new Vector3(0.0);
	}

	corners[0] = (*minCorner);
	corners[1] = (*maxCorner);
}

float
BVH::getCostOfSplit(Objects* parent, Objects* left, Objects* right) {
	Vector3 leftCorners[2], rightCorners[2], parentCorners[2];
	calcCorners(leftCorners, left);
	calcCorners(rightCorners, right);
	calcCorners(parentCorners, parent);

	Vector3 leftInnerVector = leftCorners[1] - leftCorners[0];
	Vector3 rightInnerVector = rightCorners[1] - rightCorners[0];
	Vector3 parentInnerVector = parentCorners[1] - parentCorners[0];

	float leftVolume = leftInnerVector.x * leftInnerVector.y * leftInnerVector.z;
	float rightVolume = rightInnerVector.x * rightInnerVector.y * rightInnerVector.z;
	float parentVolume = parentInnerVector.x * parentInnerVector.y * parentInnerVector.z;

	float costOfTraversal = 5.0;
	float cost = costOfTraversal + (leftVolume/parentVolume)*left->size() + (rightVolume/parentVolume)*right->size();
	return cost;
}


void
BVH::build(Objects* objs, int depth)
{
	totalNodes++;
	isLeaf = false;
	calcCorners(corners, objs); //the corners of this BV

	if(objs->size() < 3 || depth > maxDepth) {
		leafNodes++;
		isLeaf = true;
		m_objects = objs;
		return;
	}
	
	Vector3 innerCrossingVector = (corners[1] - corners[0]);
	int splits = 7;

	float prevBestCost = objs->size(); //no split
	Objects* splitCandLeft = new Objects;
	Objects* splitCandRight = new Objects;

	for(int axis = 0; axis < 3; axis++) {
		for(int j=1; j<=splits; j++) {
			float splitCoord = corners[0][axis] + j*innerCrossingVector[axis]/((float)splits+1);
			
			Objects* left = new Objects;
			Objects* right = new Objects;
			for(int i = 0; i < objs->size(); i++) {
				(*objs)[i]->calcCenterCoords();
				Vector3 objCenter = (*objs)[i]->centerCoords();

				if(objCenter[axis] < splitCoord) {
					left->push_back((*objs)[i]);
				} else {
					right->push_back((*objs)[i]);
				}
			}
			float cost = getCostOfSplit(objs,left,right);

			if(cost < prevBestCost) {
				splitCandLeft->swap((*left));
				splitCandRight->swap((*right));
				prevBestCost = cost;
			}
			delete left;
			delete right;
		}
	}

	if(prevBestCost == objs->size()) {
		delete splitCandLeft;
		delete splitCandRight;
		leafNodes++;
		isLeaf = true;
		m_objects = objs;
		return;
	}

	leftBVH = new BVH();
	rightBVH = new BVH();
	leftBVH->build(splitCandLeft, depth+1);
	rightBVH->build(splitCandRight, depth+1);
}


bool
BVH::intersect(HitInfo& minHit, const Ray& ray, float tMin, float tMax)
{
	rayBVIntersections++;
	bool hit = false;

	// This is the implementation we did based on the slides on slab-test it doesn't work properly.
	// Trying to find out why, we found a different implementation we then adapted to our code, below.
	// float tx1 = (corners[1].x - ray.o.x) / ray.d.x;
	// float tx2 = (corners[0].x - ray.o.x) / ray.d.x;
	// float ty1 = (corners[1].y - ray.o.y) / ray.d.y;
	// float ty2 = (corners[0].y - ray.o.y) / ray.d.y;
	// float tz1 = (corners[1].z - ray.o.z) / ray.d.z;
	// float tz2 = (corners[0].z - ray.o.z) / ray.d.z;

	// float tmin = max( max( min(tx1,tx2), min(ty1,ty2)), min(tz1, tz2));
	// float tmax = min( min( max(tx1,tx2), max(ty1,ty2)), max(tz1, tz2));
	// if(!((tmin < tmax) && (tmin > 0 || tmax > 0)))
	// 	return false; //this bounding box didn't hit
	

	//non-optimised version from http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-box-intersection/	
	float localtmin = (corners[0].x - ray.o.x) / ray.d.x;
	float localtmax = (corners[1].x - ray.o.x) / ray.d.x;
	if (localtmin > localtmax) swap(localtmin, localtmax);
	float tymin = (corners[0].y - ray.o.y) / ray.d.y;
	float tymax = (corners[1].y - ray.o.y) / ray.d.y;
	if (tymin > tymax) swap(tymin, tymax);
	if ((localtmin > tymax) || (tymin > localtmax))
		return false;
	if (tymin > localtmin) localtmin = tymin;
	if (tymax < localtmax) localtmax = tymax;
	float tzmin = (corners[0].z - ray.o.z) / ray.d.z;
	float tzmax = (corners[1].z - ray.o.z) / ray.d.z;
	if (tzmin > tzmax) swap(tzmin, tzmax);
	if ((localtmin > tzmax) || (tzmin > localtmax))
		return false;
	if (tzmin > localtmin) localtmin = tzmin;
	if (tzmax < localtmax) localtmax = tzmax;
	if ((localtmin > tMax) || (localtmax < tMin)) return false;
	// if (tMin < localtmin) tMin = localtmin;
	// if (tMax > localtmax) tMax = localtmax;


	if(isLeaf) {
		for (int i = 0; i < m_objects->size(); i++) {
			if ((*m_objects)[i]->intersect(minHit, ray, tMin, tMax)) {
				hit = true;
				minHit.ray = ray;
				tMax = minHit.t;
			}
		}
	} else {
		if (leftBVH->intersect(minHit, ray, tMin, tMax)) {
			hit = true;
			tMax = minHit.t;
		}
		if (rightBVH->intersect(minHit, ray, tMin, tMax)) {
			hit = true;
			// tMax = minHit.t;
		}
	}
	// if(hit) {
	// 	cout << minHit.P.x << '\n';
	// }
	return hit;
}
