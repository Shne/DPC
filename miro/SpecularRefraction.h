#include "Material.h"

class SpecularRefraction : public Material
{
public:
    SpecularRefraction(const Vector3 & ks = Vector3(1),
            const Vector3 & ka = Vector3(0));
    virtual ~SpecularRefraction();

    const Vector3 & ks() const {return m_ks;}
    const Vector3 & ka() const {return m_ka;}

    void setks(const Vector3 & ks) {m_ks = ks;}
    void setKa(const Vector3 & ka) {m_ka = ka;}

    virtual void preCalc() {}

    virtual bool traceFurther(HitInfo &hitInfo, const Ray &ray, Ray &furtherRay, Scene& scene, const int depth) const;
    
    virtual Vector3 shade(const Ray& ray, const HitInfo& hit,
                          Scene& scene/*, int depth = maxDepth*/) const;

protected:
    Vector3 m_ks;
    Vector3 m_ka;
    Material* lambert;
};
