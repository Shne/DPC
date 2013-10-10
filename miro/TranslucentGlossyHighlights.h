#include "Material.h"

class TranslucentGlossyHighlights : public Material
{
public:
    TranslucentGlossyHighlights(const Vector3 & ks = Vector3(1), const Vector3 & kd = Vector3(1), 
            const Vector3 & ka = Vector3(0), const float & shininess = infinity);
    virtual ~TranslucentGlossyHighlights();

    const Vector3 & ks() const {return m_ks;}
    const Vector3 & ka() const {return m_ka;}

    void setKs(const Vector3 & ks) {m_ks = ks;}
    void setKa(const Vector3 & ka) {m_ka = ka;}

    void setShininess(const float shininess) { m_shininess = shininess; }
    float getShininess() const { return m_shininess; }

    virtual void preCalc() {}

    const bool isTranslucent() const {return true;}
    
    virtual Vector3 shade(const Ray& ray, const HitInfo& hit,
                          Scene& scene, int depth = maxDepth) const;

protected:
    Vector3 m_ks;
    Vector3 m_kd;
    Vector3 m_ka;
    float m_shininess;
};
