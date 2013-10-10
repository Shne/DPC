#include "Lambert.h"

class TranslucentLambert : public Lambert {

public:
	TranslucentLambert(const Vector3 & kd = Vector3(1),
			const Vector3 & ka = Vector3(0),
			const int ibl_samples = 16):
		Lambert(kd, ka, ibl_samples) {};
	~TranslucentLambert() {};

	const bool isTranslucent() const {return true;}

	virtual Vector3 shade(const Ray& ray, const HitInfo& hit,
						  Scene& scene, int depth = maxDepth) const {return m_ka;}
};
