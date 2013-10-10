#include "Material.h"

class EnvMapMaterial : public Material
{
public:
    EnvMapMaterial(const char * filename, int width, int height);
    virtual ~EnvMapMaterial();

    virtual void preCalc() {}

    // virtual bool traceFurther(HitInfo &hitInfo, const Ray &ray, Ray &furtherRay, Scene& scene, const int depth) const {}
    
    virtual Vector3 shade(const Ray& ray, const HitInfo& hit,
                          Scene& scene, int depth = maxDepth) const;

protected:
    const int mapHeight, mapWidth;
    const Vector3* m_envMap;
};
