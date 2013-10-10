#include "SpecularReflection.h"
#include "Ray.h"
#include "Scene.h"
#include "Lambert.h"

SpecularReflection::SpecularReflection(const Vector3 & ks, const Vector3 & kd, const Vector3 & ka) :
	m_ks(ks), m_kd(kd), m_ka(ka)
{
	lambert = new Lambert(m_kd, m_ka);
}

SpecularReflection::~SpecularReflection()
{
}

bool
SpecularReflection::traceFurther(HitInfo &hit, const Ray &ray, Ray &furtherRay, Scene& scene, const int depth) const {
	furtherRay = Ray(hit.P ,ray.d - 2*dot(ray.d, hit.N)*hit.N);
	hit.ray = furtherRay;
	return scene.trace(hit, furtherRay, 0.001f, MIRO_TMAX, depth);
}

Vector3
SpecularReflection::shade(const Ray& ray, const HitInfo& hit, Scene& scene, int depth) const
{
	Vector3 L = Vector3(0.0f, 0.0f, 0.0f);
	

	HitInfo hitInfo;
	Ray furtherRay;
	// if(depth < maxDepth && (scene.trace(hitInfo, reflectedRay, 0.01f) || scene.hasEnvMap())) {
	if(traceFurther(hitInfo, ray, furtherRay, scene, 0) || scene.hasEnvMap()) {
		L = m_ks * hitInfo.material->shade(furtherRay, hitInfo, scene, depth+1);	
	} else {
		// Material* lambert = new Lambert(m_kd, m_ka);
		L = /*m_ks **/ lambert->shade(ray, hit, scene);
		// delete lambert;
		// L += m_ka;
	}
	
	// printf("Specular light: %f, %f, %f\n",L.x, L.y, L.z);
	return L;
}
