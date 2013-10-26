#include "BuildScenes.h"
#include <math.h>
#include "Miro.h"
#include "Scene.h"
#include "Camera.h"
#include "Image.h"

#include "PointLight.h"
#include "TriangleMesh.h"
#include "Triangle.h"
#include "Lambert.h"
#include "TranslucentLambert.h"

#include "SpecularReflection.h"
#include "SpecularRefraction.h"
#include "GlossyHighlights.h"
#include "TranslucentGlossyHighlights.h"

// #include "EnvMapMaterial.h"

#define SIENNA Vector3(0.62745098f, 0.321568627f, 0.176470588f)
#define SADDLE_BROWN Vector3(0.545098039f, 0.270588235f, 0.074509804f)
#define BROWN Vector3(0.647058824f, 0.164705882f, 0.164705882f)
#define MARBLE_WHITE Vector3(0.933333333, 0.917647059, 0.968627451)

// local helper function declarations
namespace
{
// void g_scene->addMeshTrianglesToScene(TriangleMesh * mesh, Material * material);
inline Matrix4x4 translate(float x, float y, float z);
inline Matrix4x4 scale(float x, float y, float z);
inline Matrix4x4 rotate(float angle, float x, float y, float z);
} // namespace


void
makeTeapotScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	// g_image->resize(512, 512);
	g_image->resize(256, 256);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.0f));
	g_camera->setEye(Vector3(0, 3, 6));
	g_camera->setLookAt(Vector3(0, 0, 0));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	g_camera->setExposure(2.2f);

	// g_camera->setLensRadius(.1);
	// g_camera->setFocalPlaneDistance(8.0);
	// g_camera->setNoOfDoFSamples(16);

	// create and place a point light source
	PointLight * light = new PointLight;
	// light->setPosition(Vector3(10, 10, 10));
	// light->setPosition(Vector3(-5, 6, 1));
	light->setPosition(Vector3(0.0, 4.0, -1.5)); //(-180, 320, 220)
	light->setColor(Vector3(1.0f, 1.0f, 1.0f));
	light->setWattage(700);
	g_scene->addLight(light);

	g_scene->setInitialHitPointRadius(0.08f); //was 0.08
	g_scene->setPhotonsPerLight(pow(2,19));
	g_scene->setTranslucentMaterialScale(40); //before area scaling, this was 40
	g_scene->setScatterHitpointRadius(0.08f); //previously 0.08 //default is 0.1

	// g_scene->loadEnvMap("stpeters_probe.pfm", 1500, 1500);
	// g_scene->loadEnvMap("rnl_probe.pfm", 900, 900);
	// g_scene->loadEnvMap("grace_probe.pfm", 1000, 1000);
	// g_scene->loadEnvMap("kitchen_probe.pfm", 640, 640);
	// g_scene->loadEnvMap("campus_probe.pfm", 640, 640);

	// Material* teapotMaterial = new Lambert(BROWN, Vector3(.01f), 128);
	// Material* teapotMaterial = new TranslucentLambert(BROWN, MARBLE_WHITE, 0);
	Material* teapotMaterial = new TranslucentGlossyHighlights(MARBLE_WHITE*.5, MARBLE_WHITE, MARBLE_WHITE * 0.01f, 50.0f);
	// Material* teapotMaterial = new TranslucentLambert(BROWN, Vector3(1.,2.,3.), 128);
	
	// Material* floorMaterial = new Lambert(Vector3(.4f), Vector3(.001f), 32);

	// Material* teapotMaterial = new SpecularReflection(Vector3(0.6f), Vector3(0.3f, 0.5f, 0.7f), Vector3(0.3f, 0.5f, 0.7f)*0.1f);
	// Material* floorMaterial = new SpecularReflection(Vector3(0.9f), Vector3(0.3f, 0.5f, 0.7f), Vector3(.05f));

	// Material* teapotMaterial = new SpecularRefraction(Vector3(0.8f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* teapotMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(.01f,.02f,.03f), Vector3(0.01f, 0.02f, 0.03f), 50.0f);
	// Material* floorMaterial = new GlossyHighlights(Vector3(0.5f), SADDLE_BROWN, SADDLE_BROWN * 0.01f, 50.0f);

	TriangleMesh * teapot = new TriangleMesh(teapotMaterial);
	teapot->load("../teapot.obj");
	teapot->setMpPerTri(80); //80
	teapot->setProp(1);
	g_scene->addMeshTrianglesToScene(teapot);
	
	// create the floor triangle
	// TriangleMesh * floor = new TriangleMesh(floorMaterial);
	// floor->createSingleTriangle();
	// floor->setV1(Vector3(-6, -.1, -6));
	// floor->setV2(Vector3(  0, -.1,  6));
	// floor->setV3(Vector3( 6, -.1, -6));
	// floor->setN1(Vector3(0, 1, 0));
	// floor->setN2(Vector3(0, 1, 0));
	// floor->setN3(Vector3(0, 1, 0));
	
	// Triangle* t = new Triangle;
	// t->setIndex(0);
	// t->setMesh(floor);
	// t->setMaterial(floorMaterial); 
	// g_scene->addMeshTrianglesToScene(floor);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}

void
makeDragonSmoothScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	// g_image->resize(100, 100);
	
	// set up the camera
	// g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.2f)); // dark blue
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.0f)); //black
	g_camera->setEye(Vector3(0, 3, 3));
	g_camera->setLookAt(Vector3(0, .5, 0));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	g_camera->setExposure(2.2f);

	g_scene->setInitialHitPointRadius(0.1);
	g_scene->setPhotonsPerLight(pow(2,22));
	g_scene->setTranslucentMaterialScale(40); //perhaps higher value of this.
	g_scene->setScatterHitpointRadius(0.1f);

	// g_camera->setLensRadius(.1);
	// g_camera->setFocalPlaneDistance(8.0);
	// g_camera->setNoOfDoFSamples(16);

	// create and place a point light source
	PointLight * light = new PointLight;
	// light->setPosition(Vector3(10, 10, 10));
	// light->setPosition(Vector3(8.0, 8.0f, -8.0f));
	light->setPosition(Vector3(0.0, 8.0f, -6.0f));
	light->setColor(Vector3(1.0f, 1.0f, 1.0f));
	light->setWattage(300);
	g_scene->addLight(light);

	// g_scene->loadEnvMap("stpeters_probe.pfm", 1500, 1500);
	// g_scene->loadEnvMap("rnl_probe.pfm", 900, 900);
	// g_scene->loadEnvMap("grace_probe.pfm", 1000, 1000);
	// g_scene->loadEnvMap("kitchen_probe.pfm", 640, 640);
	// g_scene->loadEnvMap("campus_probe.pfm", 640, 640);

	// Material* dragonMaterial = new Lambert(Vector3(.8f), Vector3(.01f), 1024);
	// Material* floorMaterial = new Lambert(SADDLE_BROWN, Vector3(.01f), 64);

	// Material* dragonMaterial = new TranslucentLambert(BROWN, Vector3(.01f,.02f,.03f), 128);
	// Material* dragonMaterial = new TranslucentLambert(BROWN, MARBLE_WHITE, 0);
	Material* dragonMaterial = new TranslucentGlossyHighlights(MARBLE_WHITE, MARBLE_WHITE, MARBLE_WHITE, 50.0f);

	// Material* dragonMaterial = new SpecularReflection(Vector3(0.6f), Vector3(0.3f, 0.5f, 0.7f), Vector3(0.3f, 0.5f, 0.7f)*0.1f);
	// Material* floorMaterial = new SpecularReflection(Vector3(0.9f), Vector3(0.3f, 0.5f, 0.7f), Vector3(.05f));

	// Material* dragonMaterial = new SpecularRefraction(Vector3(0.8f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* dragonMaterial = new GlossyHighlights(Vector3(0.9f), Vector3(0.05f, 0.1f, 0.2f), Vector3(0.005f, 0.01f, 0.02f), 50.0f);
	// Material* floorMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.5f), Vector3(0.01f), 50.0f);

	TriangleMesh * dragon = new TriangleMesh(dragonMaterial);
	dragon->load("../dragon_smooth.obj");
	dragon->setMpPerTri(1);
	dragon->setProp(.5);
	g_scene->addMeshTrianglesToScene(dragon);
	
	// create the floor triangle
	// TriangleMesh * floor = new TriangleMesh(floorMaterial);
	// floor->createSingleTriangle();
	// floor->setV1(Vector3(-3, 0, -4));
	// floor->setV2(Vector3(  0, 0,  4));
	// floor->setV3(Vector3( 3, 0, -4));
	// floor->setN1(Vector3(0, 1, 0));
	// floor->setN2(Vector3(0, 1, 0));
	// floor->setN3(Vector3(0, 1, 0));
	
	// Triangle* t = new Triangle;
	// t->setIndex(0);
	// t->setMesh(floor);
	// t->setMaterial(floorMaterial); 
	// g_scene->addMeshTrianglesToScene(floor);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}

void
makeTeapotHiResScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	// g_image->resize(120, 120);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.0f));
	g_camera->setEye(Vector3(360, 200, 0));
	g_camera->setLookAt(Vector3(0, 60, 0));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	g_camera->setExposure(2.2f);

	g_scene->setInitialHitPointRadius(3.0); 
	g_scene->setPhotonsPerLight(pow(2,25));
	g_scene->setTranslucentMaterialScale(0.7); //higher=more absorbtion		w/o areascaling: 1.0
	g_scene->setScatterHitpointRadius(0.5); //higher=more intense			w/o areascaling: 0.5

	// create and place a point light source
	PointLight * light = new PointLight;
	// light->setPosition(Vector3(-180, 320, 220));
	light->setPosition(Vector3(-320, 320, 0));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(4200);
	g_scene->addLight(light);

	// Material* material = new Lambert(Vector3(1.0f));

	// Material* teapotMaterial = new SpecularReflection(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularReflection(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* teapotMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* teapotMaterial = new TranslucentLambert(BROWN, MARBLE_WHITE, 0);
	Material* teapotMaterial = new TranslucentGlossyHighlights(MARBLE_WHITE, MARBLE_WHITE, MARBLE_WHITE, 250.0f);


	TriangleMesh * teapot = new TriangleMesh(teapotMaterial);
	teapot->load("teapot_hires.obj");
	teapot->setMpPerTri(1);
	teapot->setProp(1.0);
	g_scene->addMeshTrianglesToScene(teapot);
	
	// create the floor triangle
	// TriangleMesh * floor = new TriangleMesh(floorMaterial);
	// floor->createSingleTriangle();
	// floor->setV1(Vector3(-10, 0, -10));
	// floor->setV2(Vector3(  0, 0,  10));
	// floor->setV3(Vector3( 10, 0, -10));
	// floor->setN1(Vector3(0, 1, 0));
	// floor->setN2(Vector3(0, 1, 0));
	// floor->setN3(Vector3(0, 1, 0));
	
	// Triangle* t = new Triangle;
	// t->setIndex(0);
	// t->setMesh(floor);
	// t->setMaterial(floorMaterial); 
	// g_scene->addMeshTrianglesToScene(floor);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}


void
makeSphereScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.2f));
	g_camera->setEye(Vector3(0, 3, 6));
	g_camera->setLookAt(Vector3(0, 0, 0));
	// g_camera->setLookAt(Vector3(10, 10, 10));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	// g_camera->setLensRadius(.1);
	// g_camera->setFocalPlaneDistance(8.0);
	// g_camera->setNoOfDoFSamples(128);

	// create and place a point light source
	PointLight * light = new PointLight;
	light->setPosition(Vector3(10, 10, 10));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(700);
	g_scene->addLight(light);

	// Material* material = new Lambert(Vector3(1.0f));

	// Material* sphereMaterial = new SpecularReflection(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularReflection(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	//Material* sphereMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f));
	//Material* floorMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	Material* sphereMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f));
	Material* floorMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	TriangleMesh * sphere = new TriangleMesh(sphereMaterial);
	sphere->load("sphere.obj");
	g_scene->addMeshTrianglesToScene(sphere);
	
	// create the floor triangle
	TriangleMesh * floor = new TriangleMesh(floorMaterial);
	floor->createSingleTriangle();
	floor->setV1(Vector3(-10, -2, -10));
	floor->setV2(Vector3(  0, -2,  10));
	floor->setV3(Vector3( 10, -2, -10));
	floor->setN1(Vector3(0, 1, 0));
	floor->setN2(Vector3(0, 1, 0));
	floor->setN3(Vector3(0, 1, 0));
	
	Triangle* t = new Triangle;
	t->setIndex(0);
	t->setMesh(floor);
	// t->setMaterial(floorMaterial); 
	g_scene->addMeshTrianglesToScene(floor);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}


void
makeSmallLightScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.2f));
	g_camera->setEye(Vector3(0, 3, 6));
	g_camera->setLookAt(Vector3(0, 0, 0));
	// g_camera->setLookAt(Vector3(10, 10, 10));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	g_camera->setExposure(200.0);

	g_camera->setLensRadius(.2);
	g_camera->setFocalPlaneDistance(12.0);
	g_camera->setNoOfDoFSamples(1024);

	// create and place a point light source
	PointLight * light = new PointLight;
	light->setPosition(Vector3(0, 3, -6));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(700);
	g_scene->addLight(light);

	// Material* material = new Lambert(Vector3(1.0f));

	Material* floorMaterial = new SpecularReflection(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	//Material* floorMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* floorMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));
	
	// create the floor triangle
	TriangleMesh * floor = new TriangleMesh(floorMaterial);
	floor->createSingleTriangle();
	floor->setV1(Vector3(-0.006, -0.008, -0.01));
	floor->setV2(Vector3(  0, -0.008,  0.06));
	floor->setV3(Vector3( 0.006, -0.008, -0.01));
	floor->setN1(Vector3(0, 1, 0));
	floor->setN2(Vector3(0, 1, 0));
	floor->setN3(Vector3(0, 1, 0));
	
	Triangle* t = new Triangle;
	t->setIndex(0);
	t->setMesh(floor);
	// t->setMaterial(floorMaterial); 
	g_scene->addMeshTrianglesToScene(floor);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}

void
makeSphereSmoothScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.2f));
	g_camera->setEye(Vector3(0, 1, 3));
	g_camera->setLookAt(Vector3(0, 0, 0));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	// create and place a point light source
	PointLight * light = new PointLight;
	light->setPosition(Vector3(4, 10, 4));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(400);
	g_scene->addLight(light);

	g_scene->loadEnvMap("stpeters_probe.pfm", 1500, 1500);

	// Material* material = new Lambert(Vector3(1.0f));

	Material* sphereMaterial = new SpecularReflection(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularReflection(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	//Material* sphereMaterial = new SpecularRefraction(Vector3(0.7f), Vector3(0.3f, 0.5f, 1.0f));
	//Material* floorMaterial = new SpecularRefraction(Vector3(0.7f), Vector3(1.0f, 0.3f, 0.0f));

	// Material* sphereMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f), 500.0f);
	Material* floorMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));


	Matrix4x4 xform;
	xform.setIdentity();
	// xform *= scale(0.3, 2.0, 0.7);
	xform *= translate(0, .4, .7);
	// xform *= rotate(25, .3, .1, .6);


	TriangleMesh * sphere = new TriangleMesh(sphereMaterial);
	// TriangleMesh * sphere2 = new TriangleMesh;
	sphere->load("sphere_smooth.obj");
	// sphere2->load("sphere_smooth.obj", xform);
	g_scene->addMeshTrianglesToScene(sphere);
	// g_scene->addMeshTrianglesToScene(sphere2, sphereMaterial);
	
	// create the floor triangle
	TriangleMesh * floor = new TriangleMesh(floorMaterial);
	floor->createSingleTriangle();
	floor->setV1(Vector3(-5, -0.5, -5));
	floor->setV2(Vector3(  0, -0.5,  5));
	floor->setV3(Vector3( 5, -0.5, -5));
	floor->setN1(Vector3(0, 1, 0));
	floor->setN2(Vector3(0, 1, 0));
	floor->setN3(Vector3(0, 1, 0));
	
	Triangle* t = new Triangle;
	t->setIndex(0);
	t->setMesh(floor);
	// t->setMaterial(floorMaterial); 
	g_scene->addMeshTrianglesToScene(floor);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}


void
makeBunny1Scene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	// g_image->resize(180, 180);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.0f));
	g_camera->setEye(Vector3(0, 5, 4));
	g_camera->setLookAt(Vector3(-.5, 1, 0));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	// create and place a point light source
	PointLight * light = new PointLight;
	light->setPosition(Vector3(-1., 3., -3.));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(300);
	g_scene->addLight(light);

	g_scene->setInitialHitPointRadius(0.2);
	g_scene->setPhotonsPerLight(pow(2,20));
	g_scene->setTranslucentMaterialScale(12);
	// g_scene->setScatterHitpointRadius(0.06);

	// g_scene->loadEnvMap("stpeters_probe.pfm", 1500, 1500);
	// g_scene->loadEnvMap("rnl_probe.pfm", 900, 900);
	// g_scene->loadEnvMap("grace_probe.pfm", 1000, 1000);
	// g_scene->loadEnvMap("campus_probe.pfm", 640, 640);

	// Material* bunnyMaterial = new Lambert(Vector3(1.0f));

	// Material* bunnyMaterial = new TranslucentLambert(BROWN, MARBLE_WHITE, 0);
	Material* bunnyMaterial = new TranslucentGlossyHighlights(MARBLE_WHITE, MARBLE_WHITE, MARBLE_WHITE, 500.0f);
	// Material* bunnyMaterial = new TranslucentLambert(BROWN, Vector3(.1f,.2f,.3f), 128);

	// Material* bunnyMaterial = new SpecularReflection(Vector3(0.6f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularReflection(Vector3(0.8f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* bunnyMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f));
	// Material* floorMaterial = new SpecularRefraction(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* bunnyMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.3f, 0.5f, 0.7f), 50.0f);
	// Material* floorMaterial = new GlossyHighlights(Vector3(0.5f), Vector3(0.5f, 0.5f, 0.5f));

	TriangleMesh * bunny = new TriangleMesh(bunnyMaterial);
	bunny->load("bunny.obj");
	bunny->setMpPerTri(1);
	bunny->setProp(1);
	g_scene->addMeshTrianglesToScene(bunny);
	
	// create the floor triangle
	// TriangleMesh * floor = new TriangleMesh(floorMaterial);
	// floor->createSingleTriangle();
	// floor->setV1(Vector3(-100, 0, -100));
	// floor->setV2(Vector3(   0, 0,  100));
	// floor->setV3(Vector3( 100, 0, -100));
	// floor->setN1(Vector3(0, 1, 0));
	// floor->setN2(Vector3(0, 1, 0));
	// floor->setN3(Vector3(0, 1, 0));
	
	// Triangle* t = new Triangle;
	// t->setIndex(0);
	// t->setMesh(floor);
	// t->setMaterial(floorMaterial); 
	// g_scene->addMeshTrianglesToScene(floor);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}



void
makeBunny20Scene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.2f));
	g_camera->setEye(Vector3(0, 5, 15));
	g_camera->setLookAt(Vector3(0, 0, 0));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(45);

	// create and place a point light source
	PointLight * light = new PointLight;
	light->setPosition(Vector3(10, 20, 10));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(1000);
	g_scene->addLight(light);

	// g_scene->loadEnvMap("stpeters_probe.pfm", 1500, 1500);
	// g_scene->loadEnvMap("rnl_probe.pfm", 900, 900);
	// g_scene->loadEnvMap("grace_probe.pfm", 1000, 1000);
	g_scene->loadEnvMap("campus_probe.pfm", 640, 640);

	TriangleMesh * mesh;
	// Material* material = new Lambert(Vector3(1.0f));
	Material* bunnyMaterial = new SpecularReflection(Vector3(0.6f), Vector3(0.3f, 0.5f, 0.7f));
	Material* floorMaterial = new SpecularReflection(Vector3(0.8f), Vector3(0.5f, 0.5f, 0.5f));
	Matrix4x4 xform;
	Matrix4x4 xform2;
	xform2 *= rotate(110, 0, 1, 0);
	xform2 *= scale(.6, 1, 1.1);


	// bunny 1
	xform.setIdentity();
	xform *= scale(0.3, 2.0, 0.7);
	xform *= translate(-1, .4, .3);
	xform *= rotate(25, .3, .1, .6);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 2
	xform.setIdentity();
	xform *= scale(.6, 1.2, .9);
	xform *= translate(7.6, .8, .6);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 3
	xform.setIdentity();
	xform *= translate(.7, 0, -2);
	xform *= rotate(120, 0, .6, 1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 4
	xform.setIdentity();
	xform *= translate(3.6, 3, -1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 5
	xform.setIdentity();
	xform *= translate(-2.4, 2, 3);
	xform *= scale(1, .8, 2);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 6
	xform.setIdentity();
	xform *= translate(5.5, -.5, 1);
	xform *= scale(1, 2, 1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 7
	xform.setIdentity();
	xform *= rotate(15, 0, 0, 1);
	xform *= translate(-4, -.5, -6);
	xform *= scale(1, 2, 1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 8
	xform.setIdentity();
	xform *= rotate(60, 0, 1, 0);
	xform *= translate(5, .1, 3);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 9
	xform.setIdentity();
	xform *= translate(-3, .4, 6);
	xform *= rotate(-30, 0, 1, 0);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 10
	xform.setIdentity();
	xform *= translate(3, 0.5, -2);
	xform *= rotate(180, 0, 1, 0);
	xform *= scale(1.5, 1.5, 1.5);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 11
	xform = xform2;
	xform *= scale(0.3, 2.0, 0.7);
	xform *= translate(-1, .4, .3);
	xform *= rotate(25, .3, .1, .6);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 12
	xform = xform2;
	xform *= scale(.6, 1.2, .9);
	xform *= translate(7.6, .8, .6);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 13
	xform = xform2;
	xform *= translate(.7, 0, -2);
	xform *= rotate(120, 0, .6, 1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 14
	xform = xform2;
	xform *= translate(3.6, 3, -1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 15
	xform = xform2;
	xform *= translate(-2.4, 2, 3);
	xform *= scale(1, .8, 2);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 16
	xform = xform2;
	xform *= translate(5.5, -.5, 1);
	xform *= scale(1, 2, 1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 17
	xform = xform2;
	xform *= rotate(15, 0, 0, 1);
	xform *= translate(-4, -.5, -6);
	xform *= scale(1, 2, 1);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 18
	xform = xform2;
	xform *= rotate(60, 0, 1, 0);
	xform *= translate(5, .1, 3);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 19
	xform = xform2;
	xform *= translate(-3, .4, 6);
	xform *= rotate(-30, 0, 1, 0);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);

	// bunny 20
	xform = xform2;
	xform *= translate(3, 0.5, -2);
	xform *= rotate(180, 0, 1, 0);
	xform *= scale(1.5, 1.5, 1.5);
	mesh = new TriangleMesh(bunnyMaterial);
	mesh->load("bunny.obj", xform);
	g_scene->addMeshTrianglesToScene(mesh);


	// create the floor triangle
	mesh = new TriangleMesh(floorMaterial);
	mesh->createSingleTriangle();
	mesh->setV1(Vector3(-100, 0, -100));
	mesh->setV2(Vector3(   0, 0,  100));
	mesh->setV3(Vector3( 100, 0, -100));
	mesh->setN1(Vector3(0, 1, 0));
	mesh->setN2(Vector3(0, 1, 0));
	mesh->setN3(Vector3(0, 1, 0));
	
	Triangle* t = new Triangle;
	t->setIndex(0);
	t->setMesh(mesh);
	// t->setMaterial(floorMaterial); 
	g_scene->addMeshTrianglesToScene(mesh);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}


void
makeSponzaScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.2f));
	g_camera->setEye(Vector3(8, 1.5, 1));
	g_camera->setLookAt(Vector3(0, 2.5, -1));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(55);

	g_camera->setLensRadius(.1);
	g_camera->setFocalPlaneDistance(11.0);
	g_camera->setNoOfDoFSamples(8);

	// create and place a point light source
	PointLight * light = new PointLight;
	light->setPosition(Vector3(0, 10.0, 0));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(200);
	g_scene->addLight(light);

	// g_scene->loadEnvMap("stpeters_probe.pfm", 1500, 1500);

	// Material* sceneMaterial = new Lambert(Vector3(1.0f));

	// Material* sceneMaterial = new SpecularReflection(Vector3(1.0f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* sceneMaterial = new SpecularRefraction(Vector3(1.0f), Vector3(0.5f, 0.5f, 0.5f));

	Material* sceneMaterial = new GlossyHighlights(Vector3(1.0f), Vector3(0.6f), Vector3(0.1f), 50.0f);

	TriangleMesh * mesh = new TriangleMesh(sceneMaterial);
	mesh->load("sponza.obj");
	g_scene->addMeshTrianglesToScene(mesh);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}


void
makeCubeScene()
{
	g_camera = new Camera;
	g_scene = new Scene;
	g_image = new Image;

	g_image->resize(512, 512);
	
	// set up the camera
	g_camera->setBGColor(Vector3(0.0f, 0.0f, 0.2f));
	g_camera->setEye(Vector3(5, 1, -2));
	g_camera->setLookAt(Vector3(0, -1, -1.5));
	g_camera->setUp(Vector3(0, 1, 0));
	g_camera->setFOV(55);

	// create and place a point light source
	PointLight * light = new PointLight;
	light->setPosition(Vector3(0, 10.0, 0));
	light->setColor(Vector3(1, 1, 1));
	light->setWattage(200);
	g_scene->addLight(light);

	g_scene->loadEnvMap("stpeters_probe.pfm", 1500, 1500);

	Material* sceneMaterial = new Lambert(Vector3(1.0f));

	// Material* sceneMaterial = new SpecularReflection(Vector3(1.0f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* sceneMaterial = new SpecularRefraction(Vector3(1.0f), Vector3(0.5f, 0.5f, 0.5f));

	// Material* sceneMaterial = new GlossyHighlights(Vector3(1.0f), Vector3(0.5f, 0.5f, 0.5f), 50.0f);

	TriangleMesh * mesh = new TriangleMesh(sceneMaterial);
	mesh->load("cube.obj");
	g_scene->addMeshTrianglesToScene(mesh);
	
	// let objects do pre-calculations if needed
	g_scene->preCalc();
}


// local helper function definitions
namespace
{

inline Matrix4x4
translate(float x, float y, float z)
{
	Matrix4x4 m;
	m.setColumn4(Vector4(x, y, z, 1));
	return m;
}


inline Matrix4x4
scale(float x, float y, float z)
{
	Matrix4x4 m;
	m.m11 = x;
	m.m22 = y;
	m.m33 = z;
	return m;
}

// angle is in degrees
inline Matrix4x4
rotate(float angle, float x, float y, float z)
{
	float rad = angle*(PI/180.);
	
	float x2 = x*x;
	float y2 = y*y;
	float z2 = z*z;
	float c = cos(rad);
	float cinv = 1-c;
	float s = sin(rad);
	float xy = x*y;
	float xz = x*z;
	float yz = y*z;
	float xs = x*s;
	float ys = y*s;
	float zs = z*s;
	float xzcinv = xz*cinv;
	float xycinv = xy*cinv;
	float yzcinv = yz*cinv;
	
	Matrix4x4 m;
	m.set(x2 + c*(1-x2), xy*cinv+zs, xzcinv - ys, 0,
		  xycinv - zs, y2 + c*(1-y2), yzcinv + xs, 0,
		  xzcinv + ys, yzcinv - xs, z2 + c*(1-z2), 0,
		  0, 0, 0, 1);
	return m;
}

} // namespace


