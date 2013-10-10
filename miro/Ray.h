#ifndef CSE168_RAY_H_INCLUDED
#define CSE168_RAY_H_INCLUDED

#include "Vector3.h"
#include "Material.h"

class Ray
{
public:
    Vector3 o,      //!< Origin of ray
            d;      //!< Direction of ray

    Ray() : o(), d(Vector3(0.0f,0.0f,1.0f))
    {
        // empty
    }

    Ray(const Vector3& o, const Vector3& d) : o(o), d(d)
    {
        // empty
    }
};


//! Contains information about a ray hit with a surface.
/*!
    HitInfos are used by object intersection routines. They are useful in
    order to return more than just the hit distance.
*/
class HitInfo
{
public:
    float t;                            //!< The hit distance
    Vector3 P;                          //!< The hit point
    Vector3 N;                          //!< Shading normal vector
    const Material* material;           //!< Material of the intersected object
    float r2; //the gradually smaller radius^2 the point collects photons from
    float A; //represented area
    unsigned int photons;
    int pixel_index;
    Vector3 flux;
    Ray ray;

    //! Default constructor.
    explicit HitInfo(float t = 0.0f,
                     const Vector3& P = Vector3(),
                     const Vector3& N = Vector3(0.0f, 1.0f, 0.0f)) :
        t(t), P(P), N(N), material(0), r2(0), A(0), photons(0), pixel_index(0), flux(Vector3(0.0f)), ray(Ray())
    {
        // empty
    }
};

#endif // CSE168_RAY_H_INCLUDED
