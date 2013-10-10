#include "Plane.h"
#include "Ray.h"



Plane::Plane(Vector3 position, Vector3 normal) :
	m_position(position), m_normal(normal)
{

}


Plane::~Plane() {}

void
Plane::calcBVCoords() {
}

void
Plane::calcCenterCoords() {
}


void
Plane::renderGL()
{
}


bool
Plane::intersect(HitInfo& result, const Ray& r,float tMin, float tMax)
{
	float nDotl = dot(m_normal, r.d);
	if(nDotl == 0.0f) {
		return false;
	}

	float d = dot((m_position - r.o), m_normal) / dot(m_normal, r.d);

	if(d < tMin || d > tMax) {
		return false;
	}
	result.t = d;
	result.P = r.o + r.d*d;
	return true;
}
