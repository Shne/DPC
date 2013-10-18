#include <math.h>
#include "Miro.h"
#include "Scene.h"
#include "Camera.h"
#include "Image.h"
#include "Console.h"

#include "PointLight.h"
#include "Sphere.h"
#include "TriangleMesh.h"
#include "Triangle.h"
#include "Lambert.h"
#include "MiroWindow.h"

#include "BuildScenes.h"

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_gl_interop.h>

// void
// makeSpiralScene()
// {
// 	g_camera = new Camera;
// 	g_scene = new Scene;
// 	g_image = new Image;

// 	g_image->resize(512, 512);
	
// 	// set up the camera
// 	g_camera->setBGColor(Vector3(1.0f, 1.0f, 1.0f));
// 	g_camera->setEye(Vector3(-5, 2, 3));
// 	g_camera->setLookAt(Vector3(0, 0, 0));
// 	g_camera->setUp(Vector3(0, 1, 0));
// 	g_camera->setFOV(45);

// 	// create and place a point light source
// 	PointLight * light = new PointLight;
// 	light->setPosition(Vector3(-3, 15, 3));
// 	light->setColor(Vector3(1, 1, 1));
// 	light->setWattage(1000);
// 	g_scene->addLight(light);

// 	// create a spiral of spheres
// 	Material* mat = new Lambert(Vector3(1.0f, 0.0f, 0.0f));
// 	const int maxI = 200;
// 	const float a = 0.15f;
// 	for (int i = 1; i < maxI; ++i)
// 	{
// 		float t = i/float(maxI);
// 		float theta = 4*PI*t;
// 		float r = a*theta;
// 		float x = r*cos(theta);
// 		float y = r*sin(theta);
// 		float z = 2*(2*PI*a - r);
// 		Sphere * sphere = new Sphere;
// 		sphere->setCenter(Vector3(x,y,z));
// 		sphere->setRadius(r/10);
// 		// sphere->setMaterial(mat);
// 		g_scene->addObject(sphere);
// 	}
	
// 	// let objects do pre-calculations if needed
// 	g_scene->preCalc();
// }



int
main(int argc, char*argv[])
{
	// create a scene
	// makeSpiralScene();
	makeTeapotScene();
	// makeDragonSmoothScene();
	// makeSphereScene();
	// makeSphereSmoothScene();
	// makeTeapotHiResScene();
	// makeBunny1Scene();
	// makeBunny20Scene();
	// makeSponzaScene();
	// makeCubeScene();
	// makeSmallLightScene();

	g_scene->photonmapImage(g_camera, g_image);

	// g_camera->setRenderer(Camera::RENDER_RAYTRACE);
	// g_camera->click(g_scene, g_image);
	char str[1024];
    sprintf(str, "miro_%ld.ppm", time(0));
    g_image->writePPM(str);

	// MiroWindow miro(&argc, argv);
	// miro.mainLoop();

	return 0; // never executed
}

