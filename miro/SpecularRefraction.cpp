#include "SpecularRefraction.h"
#include "Ray.h"
#include "Scene.h"
#include "Lambert.h"

SpecularRefraction::SpecularRefraction(const Vector3 & ks, const Vector3 & ka) :
	m_ks(ks), m_ka(ka)
{
	lambert = new Lambert(m_ks, m_ka);
}

SpecularRefraction::~SpecularRefraction()
{
}

bool
SpecularRefraction::traceFurther(HitInfo &hit, const Ray &ray, Ray &furtherRay, Scene& scene, const int depth) const {
	float refractionIndex = 2.0;
	float refractionIndexFraction;

	bool goingIn = dot(ray.d,hit.N) < 0;
	refractionIndexFraction = goingIn ? (1.0/refractionIndex) : (refractionIndex/1.0);
	Vector3 normal = goingIn ? hit.N : -hit.N;

	// our first attempt from slides
	// Ray furtherRay = Ray(hit.P, (-refractionIndexFraction) * (ray.d - dot(ray.d, normal) * normal) - sqrt(1.0-pow(refractionIndexFraction,2) * (1.0-pow(dot(ray.d, normal),2))) * normal);

	// the following math is adapted from http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
	const float cosI = dot(normal, ray.d);
	const float sinT2 = refractionIndexFraction * refractionIndexFraction * (1.0 - cosI * cosI);

	// Ray furtherRay;
	if (sinT2 > 1.0) {
		furtherRay = Ray(hit.P ,ray.d - 2*dot(ray.d, hit.N)*hit.N); //actually reflected
		// Material* lambert = new Lambert(m_ks, m_ka);
		// return m_ks * lambert->shade(ray, hit, scene);
	} else {
		furtherRay = Ray(hit.P, refractionIndexFraction * ray.d - (refractionIndexFraction + sqrt(1.0 - sinT2)) * normal);
	}

	hit.ray = furtherRay;
	return scene.trace(hit, furtherRay, 0.001f, MIRO_TMAX, depth);
}

Vector3
SpecularRefraction::shade(const Ray& ray, const HitInfo& hit, Scene& scene) const
{	
	Vector3 L = Vector3(0.0f, 0.0f, 0.0f);
	
	HitInfo hitInfo;
	Ray furtherRay;
	// if(traceFurther(hitInfo, ray, furtherRay, scene, 0) || scene.hasEnvMap()) {
	// 	L = m_ks * hitInfo.material->shade(furtherRay, hitInfo, scene);
	// } else {
		// Material* lambert = new Lambert(m_ks, m_ka);
		L = m_ks * lambert->shade(ray, hit, scene);
	// }
	
	return L;
}
