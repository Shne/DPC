#ifndef CSE168_OBJECT_H_INCLUDED
#define CSE168_OBJECT_H_INCLUDED

#include <vector>
#include "Miro.h"
#include "Vector3.cu"
// #include "Material.h"

class Object
{
public:
    Object() {}
    virtual ~Object() {}

    // void setMaterial(const Material* m) {m_material = m;}

    virtual void renderGL() {}
    virtual void preCalc() {}

    virtual Vector3 maxCoords() const { return m_cachedMaxCoords; }
    virtual Vector3 minCoords() const { return m_cachedMinCoords; }
    virtual Vector3 centerCoords() const { return m_cachedCenterCoords; }
    virtual void calcBVCoords() {}
    virtual void calcCenterCoords() {}


    virtual bool intersect(HitInfo& result, const Ray& ray,
                           float tMin = 0.0f, float tMax = MIRO_TMAX) = 0;

protected:
    // const Material* m_material;
    Vector3 m_cachedMinCoords, m_cachedMaxCoords, m_cachedCenterCoords;
};

typedef std::vector<Object*> Objects;

#endif // CSE168_OBJECT_H_INCLUDED
