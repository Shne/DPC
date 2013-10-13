#include <cmath>
#include "Miro.h"
#include "Scene.h"
#include "Camera.h"
#include "Image.h"
#include "Console.h"
#include <time.h>
#include "Triangle.h"
#include "TriangleMesh.h"
#include <iostream>
#include <map>

#include "Clock.h"
#include "HashGrid.h"

#include "SpecularReflection.h"
#include "SpecularRefraction.h"
#include "GlossyHighlights.h"

// #include "kernel.h"
#include "FinalPassKernel.h"

using namespace std;

#define ALPHA (0.7f)
#define E (2.7182818284f)
#define RAND ((float)rand()/(float)RAND_MAX)

Scene * g_scene = 0;
EnvMapMaterial* m_envMapMaterial = 0;
unsigned int Scene::totalRays;
int const Scene::maxDepth;

const int Scene::primes[61] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
							  101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,
							  193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283};


void
Scene::openGL(Camera *cam)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	cam->drawGL();

	// draw objects
	for (size_t i = 0; i < m_objects.size(); ++i)
		m_objects[i]->renderGL();

	glutSwapBuffers();
}

void
Scene::preCalc()
{
	BVH::totalNodes = 0;
	BVH::leafNodes = 0;

	Objects::iterator it;
	for (it = m_objects.begin(); it != m_objects.end(); it++)
	{
		Object* pObject = *it;
		pObject->preCalc();
	}
	Lights::iterator lit;
	for (lit = m_lights.begin(); lit != m_lights.end(); lit++)
	{
		PointLight* pLight = *lit;
		pLight->preCalc();
	}

	struct timespec start, finish;
	float elapsed_time;
	clock_gettime(CLOCK_MONOTONIC, &start);

	m_bvh.build(&m_objects, (int)0);

	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_time = (finish.tv_sec - start.tv_sec);
	elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("Build Time:                     %f secs\n",elapsed_time);
	printf("Total BVH Nodes:                %u\n", BVH::totalNodes);
	printf("Leaf BVH Nodes:                 %u\n", BVH::leafNodes);
}

//tone mapping
//adapted from http://freespace.virgin.net/hugo.elias/graphics/x_posure.htm
Vector3
Scene::expose(Vector3& pixelColor, float exposure) {
	float red = 1.0f - exp(-pixelColor.x * exposure);
	float green = 1.0f - exp(-pixelColor.y * exposure);
	float blue = 1.0f - exp(-pixelColor.z * exposure);
	return Vector3(red, green, blue);
}

void
Scene::gammaCorrect(Vector3 & v) {
	v.x = srgbEncode(v.x);
	v.y = srgbEncode(v.y);
	v.z = srgbEncode(v.z);
}

//adapted from http://www.codermind.com/articles/Raytracer-in-C++-Part-II-Specularity-post-processing.html
float
Scene::srgbEncode(float c) {
	if (c <= 0.0031308f) {
		return 12.92f * c; 
	} else {
		return 1.055f * powf(c, 0.4166667f) - 0.055f; // Inverse gamma 2.4
	}
}


void
Scene::photonmapImage(Camera *cam, Image *img) {
	// uint w = img->width();
	// uint h = img->height();
	// float* result = kernel(w, h, NULL);

	// cout << "kernel done" << endl;

	// for(int j = 0; j < h; ++j) {
	// 	for(int i = 0; i < w; ++i) {
	// 		// cout << "HURR" << endl;
	// 		// cout << result[i*w+j] << endl;
	// 		img->setPixel(i, j, Vector3(result[i*w+j]));
	// 	}
	// 	img->drawScanline(j);
	// 	glFinish();
	// }
	// return;


	/***************************
	 *        ORIGINAL         *
	 ***************************/
	Clock clock = Clock();
	clock.start();

	int imgSize = img->width()*img->height();
	int hashSize = imgSize;
	// int hashSize = pow(2,12);

	HashGrid eye_mp_hg = HashGrid();
	eye_mp_hg.initializeGrid(m_bvh.corners[0], m_bvh.corners[1], hashSize, initialHitPointRadius);

	// m_hashGrids.push_back(&eye_mp_hg);


	// EYE PASS, for all pixels
	clock.start();

	HitInfo measureHIArray[imgSize];

	for (int j = 0; j < img->height(); ++j) {
		for (int i = 0; i < img->width(); ++i) {
			Ray* rays = cam->eyeRays(i, j, img->width(), img->height());
			totalRays++;
			HitInfo hitInfo = HitInfo();
			hitInfo.ray = rays[0];
			hitInfo.r2 = initialHitPointRadius*initialHitPointRadius;
			hitInfo.pixel_index = j*img->width() + i;
			bool hit = trace(hitInfo, rays[0]);

			// if(hit || hasEnvMap()) {
			// 	measureHIArray[j*img->width() + i] = hitInfo;
			// 	eye_mp_hg.addHitPoint(hitInfo);
			// } else {
			// 	measureHIArray[j*img->width() + i] = NULL;
			// }
			measureHIArray[j*img->width() + i] = hitInfo;
			if(hit) eye_mp_hg.addHitPoint(hitInfo);
		}
	}
	std::cout << "Eye pass done!                  " << clock.stop() << "\n";
	


	// OBJECT PASS, distributing more irradiance samples across translucent objects
	// to be used in calculating the BSSRDF in the final pass.
	HashGrid scatteringMPs_hg;
	HitInfo* scatteringMPs;
	int scatteringMPsSize;
	// std::vector<HitInfo> scatteringMPs;

	// for each trianglemesh
	for(std::list<TriangleMesh*>::iterator tMeshIter=m_triangleMeshes.begin(); tMeshIter!=m_triangleMeshes.end(); ++tMeshIter) {
		TriangleMesh* tMesh = (*tMeshIter);
		// if its material is translucent
		// if(!tMesh->material()->isTranslucent()) continue;

		// have it generate an even distribution of measurement points and put it in the hashgrid
		// HashGrid* scatteringMPs_hg = tMesh->calculateEvenlyDistributedMPs(m_bvh.corners[0], m_bvh.corners[1], 1024);
		scatteringMPs_hg = HashGrid();
		scatteringMPs = tMesh->calculateMPs(m_bvh.corners[0], m_bvh.corners[1], scatteringMPs_hg, scatterHitpointRadius);
		scatteringMPsSize = tMesh->mpPerTri() * tMesh->numTris();
		//	add the hashgrid to a vector of hashgrids, including the eye_mp_hg
		// m_hashGrids.push_back(scatteringMPs_hg);
	}
	
	




	// PHOTON PASS, for all lights
	clock.start();	
	for(int i = 0; i<m_lights.size(); i++) {
		int m = 1000*i;
		Vector3 flux = Vector3(m_lights[i]->wattage()) * (PI*4.0);
		
		#pragma omp parallel for num_threads(4) schedule(dynamic)
		for(int j = 0; j<photonsPerLight; j++) {
			float p = 2. * PI * hal(0,m+j),
				  t = 2. * acos(sqrt(1.-hal(1,m+j))),
					st = sin(t);

			Ray r;
			r.d = Vector3(cos(p)*st, cos(t), sin(p)*st).normalize();
			r.o = m_lights[i]->position();

			HitInfo photonHI;
			if(trace(photonHI, r)) {

				//for each hashgrid in vector of hashgrids
				std::list<HitInfo*> hiList = eye_mp_hg.lookup(photonHI.P);
				
				for(std::list<HitInfo*>::iterator hiIter = hiList.begin(); hiIter != hiList.end(); ++hiIter) {
					HitInfo* measureHI = (*hiIter);
					if(measureHI == 0) continue;
					if(measureHI->material == m_envMapMaterial) continue;

					float distance2 = (measureHI->P - photonHI.P).length2();
					
					if(distance2 < measureHI->r2) {
						float g = (measureHI->photons*ALPHA+ALPHA) 
								   / (measureHI->photons*ALPHA+1.0);
						measureHI->r2 = measureHI->r2*g;
						measureHI->photons++;
						measureHI->flux += flux * (1./PI) * g;
					}
				}

				std::list<HitInfo*> scatteringHiList = scatteringMPs_hg.lookup(photonHI.P);
				for(std::list<HitInfo*>::iterator sHiIter = scatteringHiList.begin(); sHiIter != scatteringHiList.end(); ++sHiIter) {
					HitInfo* scatteringHI = (*sHiIter);
					if(scatteringHI == 0) continue;
					float distance2 = (scatteringHI->P - photonHI.P).length2();
					
					if(distance2 < scatteringHI->r2) {
						// float g = (scatteringHI->photons*ALPHA+ALPHA) 
						// 		   / (scatteringHI->photons*ALPHA+1.0);
						// scatteringHI->r2 = scatteringHI->r2*g; 
						// scatteringHI->photons++;
						scatteringHI->flux += flux;// * (1./PI) ;//* g;
					}
				}
				// }
			}
		}
	}
	std::cout << "Photon pass done!               " << clock.stop() << "\n";


	finalPass(img, scatteringMPs, scatteringMPsSize, measureHIArray, cam);
	





	// SHADING
	Vector3 pixelColor;
	Vector3 marbleWhite = Vector3(0.933333333, 0.917647059, 0.968627451);
	
	Vector3 shadeResult = hi->material->shade(hi->ray, (*hi), *this, 0);
	pixelColor = shadeResult + marbleWhite * hi->flux * (1.0/(photonsPerLight));
	
	if(cam->exposure() != 0.0) {
		pixelColor = expose(pixelColor, cam->exposure()); //tone mapping
		gammaCorrect(pixelColor);
	}

	img->setPixel(i, j, pixelColor);


	return;












	// FINAL PASS, for each pixel
	clock.start();
	int finished_lines = 0;

	//values for multiple subsurface scattering
	float my = 1.3;
	float sigmaS = 2.6		* TranslucentMaterialScale;
	float sigmaA = 0.0041 	* TranslucentMaterialScale;
	float sigmaT = sigmaS + sigmaA;
	float alpha = sigmaS / sigmaT;
	float sigmaTR = sqrt(3.0*sigmaA*sigmaT);
	float lu = 1.0/sigmaT;
	float Fdr =  -1.440/(my*my) + 0.710*my + 0.668 + 0.0636*my;
	float Fdt = 1.0 - Fdr;
	float A = (1 + Fdr) / (1 - Fdr);
	float zr = lu;
	float zv = lu*(1.0 + 4.0/(3.0*A));


	#pragma omp parallel for num_threads(4) schedule(dynamic, 10)
	for (int j = 0; j < img->height(); ++j) {
		for (int i = 0; i < img->width(); ++i) {
			HitInfo* hi = measureHIArray[j*img->width() + i];
			if(hi == NULL) continue;

			/*
			// SINGLE SCATTER
			float so = RAND/sigmaT;
			const float cosI = dot(hi->N, hi->ray.d);
			const float sinT2 = my * my * (1.0 - cosI * cosI);
			Vector3 refractedRay = my * hi->ray.d - (my + sqrt(1.0 - sinT2)) * hi->N;
			Vector3 samplePoint = hi->P + so * refractedRay;

			Vector3 irradiance = Vector3(0.0);
			std::vector<HitInfo*> sampleHiList = scatteringMPs_hg->lookup(samplePoint);
			for(auto sHiIter = sampleHiList.begin(); sHiIter != sampleHiList.end(); ++sHiIter) {
				HitInfo* sHI = (*sHiIter);
				irradiance += sHI->flux * sHI->r2;
			}
			// float siprime = 

			// Vector3 radiance = sigmaS * pow(E,-)
			// hi->flux += 
			*/


			// MULTIPLE SCATTER
			int scatteringMPsSize = scatteringMPs.size();
			for(int i=0; i<scatteringMPsSize; i++) {
				HitInfo* sHI = scatteringMPs[i];
				
				float r2 = (hi->P - sHI->P).length2();
				float dr = sqrt(r2+zr*zr);
				float dv = sqrt(r2+zv*zv);
				float C1 = zr * (sigmaTR + 1.0/dr);
				float C2 = zv * (sigmaTR + 1.0/dv);

				float dMoOverAlphaPhi = 1.0/(4.0*PI) * (C1*(pow(E,-sigmaTR*dr)/dr*dr) + C2*(pow(E,-sigmaTR*dv)/dv*dv));
				Vector3 MoP = Fdt * dMoOverAlphaPhi * sHI->flux * sHI->r2*PI;
				hi->flux += MoP;
			}
		

			// SHADING
			Vector3 pixelColor;
			Vector3 marbleWhite = Vector3(0.933333333, 0.917647059, 0.968627451);
			// if(hi->material == NULL) continue;
			if(hi->material == m_envMapMaterial) {
				pixelColor = hi->material->shade(hi->ray, (*hi), *this, 0);
			} else {
				Vector3 shadeResult = hi->material->shade(hi->ray, (*hi), *this, 0);
				// pixelColor = shadeResult 
				// 					   * hi->flux 
				// 					   * (1.0/(PI * hi->r2 * photonsPerLight));
				// 				;// + (shadeResult/100.0);
				pixelColor = shadeResult + marbleWhite * hi->flux * (1.0/(photonsPerLight));
			}
			
			if(cam->exposure() != 0.0) {
				pixelColor = expose(pixelColor, cam->exposure()); //tone mapping
				gammaCorrect(pixelColor);
			}

			img->setPixel(i, j, pixelColor);
		}
		finished_lines++;
		img->drawScanline(j);
		glFinish();
		printf("Rendering Progress: %.3f%%\r", finished_lines/float(img->height())*100.0f);
		fflush(stdout);
	}
	std::cout << "Final pass done!                " << clock.stop() << "\n";








	float elapsed_time = clock.stop();

	std::cout << "Image Resolution:               " << img->width() << ", " << img->height() << "\n";
	std::cout << "Total number of rays:           " << totalRays << "\n";
	std::cout << "Render Time:                    " << elapsed_time << " secs \n";
	std::cout << "Ray-Triangle intersections:     " << Triangle::triangleIntersectionsDone << "\n";
	std::cout << "Ray-B.Volume intersections:     " << BVH::rayBVIntersections << "\n";
	std::cout << "Rendering Progress:             100.000%\n";
	std::cout << "done Photon Mapping!\n";
}


void
Scene::raytraceImage(Camera *cam, Image *img)
{	
	Scene::totalRays = 0;
	Triangle::triangleIntersectionsDone = 0;

	//timing code adapted from http://stackoverflow.com/questions/2962785/c-using-clock-to-measure-time-in-multi-threaded-programs
	struct timespec start, finish;
	float elapsed_time;
	clock_gettime(CLOCK_MONOTONIC, &start);

	const int noOfDoFSamples = cam->noOfDoFSamples();

	int finished_lines = 0;


	// RAYTRACING
	// loop over all pixels in the image
	#pragma omp parallel for num_threads(4) schedule(dynamic, 10)
	for (int j = 0; j < img->height(); ++j)
	{
		for (int i = 0; i < img->width(); ++i)
		{
			Vector3 pixelColor = Vector3(0.0f);
			Ray* rays = cam->eyeRays(i, j, img->width(), img->height());
			for(int l=0; l<noOfDoFSamples; l++) {
				// std::cout << rays[l].d.x;
				pixelColor += calcPixelColor(rays[l]);
			}
			pixelColor /= (float)noOfDoFSamples;
			if(cam->exposure() != 0.0) {
				pixelColor = expose(pixelColor, cam->exposure()); //tone mapping
				gammaCorrect(pixelColor);
			}
			img->setPixel(i, j, pixelColor);
			if(noOfDoFSamples > 1) {
				delete[] rays;
			}
		}
		finished_lines++;
		img->drawScanline(j);
		glFinish();
		printf("Rendering Progress: %.3f%%\r", finished_lines/float(img->height())*100.0f);
		fflush(stdout);
	}

	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed_time = (finish.tv_sec - start.tv_sec);
	elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;



	printf("Image Resolution:               %dx%d\n", img->width(), img->height());
	printf("Total number of rays:           %u\n", totalRays);
	printf("Render Time:                    %f secs\n",elapsed_time);
	printf("Ray-Triangle intersections:     %u\n", Triangle::triangleIntersectionsDone);
	printf("Ray-B.Volume intersections:     %u\n", BVH::rayBVIntersections);
	// printf("Rendering Progress:             100.000%\n");
	debug("done Raytracing!\n");
}

Vector3
Scene::calcPixelColor(const Ray ray) {
	totalRays++;
	HitInfo hitInfo;
	if (trace(hitInfo, ray) || hasEnvMap()) {
		Vector3 shadeResult = hitInfo.material->shade(ray, hitInfo, *this, 0);
		return shadeResult * calcShadow(hitInfo);
	} else {
		return Vector3(0.0f);
	}
}

bool
Scene::trace(HitInfo& minHit, const Ray& ray, float tMin, float tMax, int depth) {
	if(depth > maxDepth) return false;

	minHit.t = tMax;
	bool intersect = m_bvh.intersect(minHit, ray, tMin, tMax);
	if(intersect) {
		Ray furtherRay;
		minHit.material->traceFurther(minHit, ray, furtherRay, *this, depth+1);
		return true;
	}
	if(m_envMapMaterial != 0) {
		minHit.material = m_envMapMaterial;
	}
	minHit.t = 0.0f;
	return false;
}


void 
Scene::loadEnvMap(const char * filename, int width, int height) {
	m_envMapMaterial = new EnvMapMaterial(filename, width, height);
}


void
Scene::addMeshTrianglesToScene(TriangleMesh * mesh)
{
	m_triangleMeshes.push_back(mesh);
	for (int i = 0; i < mesh->numTris(); ++i)
	{
		Triangle* t = new Triangle;
		t->setIndex(i);
		t->setMesh(mesh);
		// t->setMaterial(material); 
		g_scene->addObject(t);
	}
}


float
Scene::calcShadow(HitInfo& hitInfo) {
	for(int i=0; i<m_lights.size(); i++) {
		HitInfo minHit;
		Vector3 direction = (m_lights[i]->position() - hitInfo.P).normalize();
		Ray ray = Ray(hitInfo.P, direction);
		if(trace(minHit, ray, 0.0001f)) {
			return 0.2f;
		} else {
			return 1.0f;
		}
	}
}

//adapted from smallppm.cpp
float
Scene::hal(const int b, int j) const { // Halton sequence with reverse permutation
	const int p = primes[b];
	float h = 0.0,
		  f = 1.0 / (float)p,
		  fct = f;

	while (j > 0) {
		h += rev(j % p, p) * fct;
		j /= p;
		fct *= f;
	}
	return h;
}
