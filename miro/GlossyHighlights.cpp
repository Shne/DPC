#include <cmath>
#include "stdlib.h"
#include "GlossyHighlights.h"
#include "Ray.h"
#include "Scene.h"
#include "Lambert.h"

GlossyHighlights::GlossyHighlights(const Vector3 & ks, const Vector3 & kd, const Vector3 & ka, const float & shininess) :
	m_ks(ks), m_kd(kd), m_ka(ka), m_shininess(shininess)
{
	lambert = new Lambert(m_kd, m_ka);
}

GlossyHighlights::~GlossyHighlights()
{
}


Vector3
GlossyHighlights::shade(const Ray& ray, const HitInfo& hit, Scene& scene, int depth) const
{
	//DIFFUSE
	const Vector3 Ld = lambert->shade(ray, hit, scene);

	//Specular / Glossy
	Vector3 Ls = Vector3(0.0f);

	const Lights* lights = scene.lights();
	for(int i=0; i<lights->size(); i++) {
		PointLight* light = lights->at(i);
		Vector3 lightDir = (light->position() - hit.P).normalize();

		// the inverse-squared falloff
		// float falloff = lightDir.length2();

		Vector3 reflectedRay = (ray.d - 2*dot(ray.d, hit.N)*hit.N).normalize();
		float power = pow(std::max(0.0f, dot(lightDir, reflectedRay)), m_shininess);

		Ls += power * light->color() * (light->wattage()/100) / PI;
	}
	return m_ka + Ls;
	// return Ld + Ls;
}
