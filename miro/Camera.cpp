#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "Miro.h"
#include "Camera.h"
#include "Image.h"
#include "Scene.h"
#include "Console.h" 
#include "OpenGL.h"



Camera * g_camera = 0;

static bool firstRayTrace = true; 

const float HalfDegToRad = DegToRad/2.0f;


Camera::Camera() :
    m_bgColor(0,0,0),
    m_renderer(RENDER_OPENGL),
    m_eye(0,0,0),
    m_viewDir(0,0,-1),
    m_up(0,1,0),
    m_lookAt(FLT_MAX, FLT_MAX, FLT_MAX),
    m_fov((45.)*(PI/180.)),
    m_lensRadius(0.0),
    m_focalPlaneDistance(6.0),
    m_focalPlane(m_eye+m_viewDir*m_focalPlaneDistance, m_viewDir),
    m_noOfDoFSamples(1),
    m_right(cross(m_viewDir, m_up).normalize()),
    m_exposure(0.0f)
{
    calcLookAt();
}


Camera::~Camera()
{

}


void
Camera::click(Scene* pScene, Image* pImage)
{
    calcLookAt();
    static bool firstRayTrace = false;

    if (m_renderer == RENDER_OPENGL)
    {
        glDrawBuffer(GL_BACK);
        pScene->openGL(this);
        firstRayTrace = true;
    }
    else if (m_renderer == RENDER_RAYTRACE)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDrawBuffer(GL_FRONT);
        if (firstRayTrace)
        {
            pImage->clear(bgColor());
            // pScene->raytraceImage(this, g_image);
            pScene->photonmapImage(this, g_image, 64);
            firstRayTrace = false;
        }
        
        g_image->draw();
    }
}


void
Camera::calcLookAt()
{
    // this is true when a "lookat" is not used in the config file
    if (m_lookAt.x != FLT_MAX)
    {
        setLookAt(m_lookAt);
        m_lookAt.set(FLT_MAX, FLT_MAX, FLT_MAX);
    }
}


void
Camera::drawGL()
{
    // set up the screen with our camera parameters
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov(), g_image->width()/(float)g_image->height(),
                   0.01, 10000);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    Vector3 vCenter = eye() + viewDir();
    gluLookAt(eye().x, eye().y, eye().z,
              vCenter.x, vCenter.y, vCenter.z,
              up().x, up().y, up().z);
}


Ray*
Camera::eyeRays(int x, int y, int imageWidth, int imageHeight)
{
    // first compute the camera coordinate system 
    // ------------------------------------------

    // wDir = e - (e+m_viewDir) = -m_vView
    const Vector3 wDir = Vector3(-m_viewDir).normalize(); 
    const Vector3 uDir = cross(m_up, wDir).normalize(); 
    const Vector3 vDir = cross(wDir, uDir);    



    // next find the corners of the image plane in camera space
    // --------------------------------------------------------

    const float aspectRatio = (float)imageWidth/(float)imageHeight; 


    const float top     = tan(m_fov*HalfDegToRad); 
    const float right   = aspectRatio*top; 

    const float bottom  = -top; 
    const float left    = -right; 



    // transform x and y into camera space 
    // -----------------------------------

    const float imPlaneUPos = left   + (right - left)*(((float)x+0.5f)/(float)imageWidth); 
    const float imPlaneVPos = bottom + (top - bottom)*(((float)y+0.5f)/(float)imageHeight); 

    Ray* ray = new Ray(m_eye, (imPlaneUPos*uDir + imPlaneVPos*vDir - wDir).normalize());

    //if it's a pinhole camera, return the ray as is
    if(m_lensRadius == 0 || m_noOfDoFSamples == 1) {
        // std::cout << ray->d.x;
        return ray;
    }


    HitInfo hi;
    m_focalPlane.intersect(hi, (*ray));
    Vector3 focalPlaneIntersectionPoint = hi.P;

    float lensSampleX, lensSampleY = 0.0;

    Ray* rays = new Ray[m_noOfDoFSamples];
    // Ray* rays = static_cast<Ray*> (::operator new (sizeof(Ray[m_noOfDoFSamples])));

    for(int i=0; i<m_noOfDoFSamples; i++) {
        do {
            lensSampleX = (2.0 * ((float)rand()/(float)RAND_MAX) - 1.0);
            lensSampleY = (2.0 * ((float)rand()/(float)RAND_MAX) - 1.0);
        } while(!in_aperture(lensSampleX,lensSampleY));

        Vector3 newOrigin = m_eye + m_right*lensSampleX*m_lensRadius + m_up*lensSampleY*m_lensRadius;

        rays[i] = Ray(newOrigin, focalPlaneIntersectionPoint - newOrigin);
    }
    // return &rays[0];
    return rays;
}

bool
Camera::in_aperture(float x, float y) {
    //CIRCLE
    return (x*x + y*y) < 1;

    //HEXAGON adapted from http://stackoverflow.com/questions/5193331/is-a-point-inside-regular-hexagon
    float a = 0.25 * sqrt(3.0);
    return (a*fabs(x) + 0.25*fabs(y) < a);
}