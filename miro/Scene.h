#ifndef CSE168_SCENE_H_INCLUDED
#define CSE168_SCENE_H_INCLUDED

#include "Miro.h"
#include "Object.h"
#include "PointLight.h"
#include "BVH.h"
#include "EnvMapMaterial.h"
#include <list>
#include "HashGrid.h"

class Camera;
class Image;

class Scene
{
public:
    void addObject(Object* pObj)        {m_objects.push_back(pObj);}
    const Objects* objects() const      {return &m_objects;}

    void addLight(PointLight* pObj)     {m_lights.push_back(pObj);}
    const Lights* lights() const        {return &m_lights;}

    void preCalc();
    void openGL(Camera *cam);

    void raytraceImage(Camera *cam, Image *img);
    void photonmapImage(Camera *cam, Image *img);
    bool trace(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = MIRO_TMAX, int depth = 0);

    float calcShadow(HitInfo& hitInfo);

    void loadEnvMap(const char * filename, int width, int height);
    void setInitialHitPointRadius(float r) { initialHitPointRadius = r;}
    void setPhotonsPerLight(unsigned int i) { photonsPerLight = i;}
    void addMeshTrianglesToScene(TriangleMesh * mesh);
    void setTranslucentMaterialScale(float scale) {TranslucentMaterialScale = scale;}
    void setScatterHitpointRadius(float radius) {scatterHitpointRadius = radius;}

    inline bool hasEnvMap() {return m_envMapMaterial != 0;}

    Vector3 expose(Vector3& pixelColor, float exposure);
    void gammaCorrect(Vector3 & v);

    Vector3 calcPixelColor(const Ray ray);

    //adapted from smallppm.cpp
    virtual float hal(const int b, int j) const;
    inline int rev(const int i, const int p) const { return i==0 ? i : p-i; }


protected:
    int static const primes[];
    Objects m_objects;
    BVH m_bvh;
    Lights m_lights;
    EnvMapMaterial* m_envMapMaterial;
    std::list<TriangleMesh*> m_triangleMeshes;
    std::list<HashGrid*> m_hashGrids;
    float initialHitPointRadius;
    unsigned int photonsPerLight;
    unsigned int static totalRays;
    float srgbEncode(float c);
    int static const maxDepth = 6;
    float TranslucentMaterialScale; //60
    float scatterHitpointRadius; // = 0.1;
};

extern Scene * g_scene;

#endif // CSE168_SCENE_H_INCLUDED
