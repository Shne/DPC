#ifndef CSE168_MATERIAL_H_INCLUDED
#define CSE168_MATERIAL_H_INCLUDED

#include "Miro.h"
#include "Vector3.cu"

class Material
{
public:
	Material();
	virtual ~Material();

	virtual void preCalc() {}

	virtual bool traceFurther(HitInfo &hitInfo, const Ray &ray, Ray &furtherRay, Scene& scene, const int depth = 0) const {return false;}
	
	virtual Vector3 shade(const Ray& ray, const HitInfo& hit,
						  Scene& scene, int depth = maxDepth) const;

	virtual const bool isTranslucent() const {return false;}

protected:
	int static const maxDepth = 6;
};

#endif // CSE168_MATERIAL_H_INCLUDED
