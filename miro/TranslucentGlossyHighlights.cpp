#include <cmath>
#include "stdlib.h"
#include "TranslucentGlossyHighlights.h"
#include "Ray.h"
#include "Scene.h"

TranslucentGlossyHighlights::TranslucentGlossyHighlights(const Vector3 & ks, const Vector3 & kd, const Vector3 & ka, const float & shininess) :
	m_ks(ks), m_kd(kd), m_ka(ka), m_shininess(shininess)
{
}

TranslucentGlossyHighlights::~TranslucentGlossyHighlights()
{
}


Vector3
TranslucentGlossyHighlights::shade(const Ray& ray, const HitInfo& hit, Scene& scene, int depth) const
{
	//Specular / Glossy
	Vector3 Ls = Vector3(0.0f);

	const Lights* lights = scene.lights();
	for(int i=0; i<lights->size(); i++) {
		PointLight* light = lights->at(i);
		Vector3 lightDir = (light->position() - hit.P).normalize();

		Vector3 reflectedDir = (ray.d - 2*dot(ray.d, hit.N)*hit.N).normalize();

		//self-shadow
		HitInfo hitInfo;
		Ray lightRay = Ray(hit.P, lightDir);
		bool selfShadow = scene.trace(hitInfo, lightRay, 0.01f, infinity, 0);
		if(!selfShadow) {
			float power = pow(std::max(0.0f, dot(lightDir, reflectedDir)), m_shininess);
			Ls += power * light->color() * (light->wattage()/100) / PI;
		}
		
	}
	return Ls;
}
