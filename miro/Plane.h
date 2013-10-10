#include "Object.h"

class Plane : public Object
{
public:
    Plane(Vector3 position, Vector3 normal);
    virtual ~Plane();

    virtual void calcBVCoords();
    virtual void calcCenterCoords();

    virtual void renderGL();
    virtual bool intersect(HitInfo& result, const Ray& ray,
                           float tMin = 0.0f, float tMax = MIRO_TMAX);

    inline void setPosition(Vector3 p) {m_position = p;}
    inline void setNormal(Vector3 n) {m_normal = n;}
    
protected:
    Vector3 m_position, m_normal;

};
