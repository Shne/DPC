#include "Lambert.h"
#include "Ray.h"
#include "Scene.h"




Lambert::Lambert(const Vector3 & kd, const Vector3 & ka, const int ibl_samples) :
	m_kd(kd), m_ka(ka), ibl_samples(ibl_samples) {}

Lambert::~Lambert() {}

Vector3
Lambert::shade(const Ray& ray, const HitInfo& hit, Scene& scene, int depth) const {
	Vector3 L = Vector3(0.0f, 0.0f, 0.0f);
	
	const Vector3 viewDir = -ray.d; // d is a unit vector
	
	const Lights *lightlist = scene.lights();
	
	// loop over all of the lights
	Lights::const_iterator lightIter;
	for (lightIter = lightlist->begin(); lightIter != lightlist->end(); lightIter++)
	{
		PointLight* pLight = *lightIter;
	
		Vector3 l = pLight->position() - hit.P;
		
		// the inverse-squared falloff
		float falloff = l.length2();
		
		// normalize the light direction
		l /= sqrt(falloff);

		// get the diffuse component
		float nDotL = dot(hit.N, l);
		Vector3 result = m_kd * pLight->color();
		
		L += std::max(0.0f, nDotL/falloff * pLight->wattage() / PI) * result;
	}


	if(scene.hasEnvMap()) {
		//trace rays to environment and receive some color from it
		for(int i=0; i<ibl_samples; i++) {
			// Vector3 random = Vector3(2.0 * (float)rand()/(float)RAND_MAX -1.0,
			//                          2.0 * (float)rand()/(float)RAND_MAX -1.0,
			//                          2.0 * (float)rand()/(float)RAND_MAX -1.0);
			// Vector3 d = (hit.N + random).normalize();

			//Generate random ray in the general direction of the normal. adapted from smallpt
			float r1=2*PI*scene.hal(0,i);
			float r2=scene.hal(1,i);
			float r2s=sqrt(r2);
			Vector3 w = hit.N;
			Vector3 u = ((fabs(w.x)>.1 ? Vector3(0.0,1.0,0.0) : Vector3(1.0,0.0,0.0) ) % w).normalize();
			Vector3 v = w % u;
			Vector3 d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).normalize(); 

			Ray r = Ray(hit.P, d);
			HitInfo hitInfo;

			if(!scene.trace(hitInfo, r, 0.01f)) {
				Vector3 tracedColor = hitInfo.material->shade(r, hitInfo, scene).normalize();

				L += /*(m_kd * Vector3(1.0f)) * */ m_kd * tracedColor / ibl_samples; 
			}
		}
	}
	
	// add the ambient component
	L += m_ka;
	
	return L;
}

