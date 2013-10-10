#include "EnvMapMaterial.h"
#include "Ray.h"
#include "PFMLoader.h"


EnvMapMaterial::EnvMapMaterial(const char * filename, int width, int height) :
	mapWidth(width), mapHeight(height)
{
	m_envMap = readPFMImage(filename, &width, &height);
}

EnvMapMaterial::~EnvMapMaterial()
{
}

Vector3
EnvMapMaterial::shade(const Ray& ray, const HitInfo& hit, Scene& scene, int depth) const
{
	//calculation based on http://www.pauldebevec.com/RNL/Source/angmap.cal
	float norm = 1/sqrt(ray.d.y*ray.d.y + ray.d.x*ray.d.x + ray.d.z*ray.d.z);
	
	float DDy = ray.d.y*norm;
	float DDx = ray.d.x*norm;
	float DDz = ray.d.z*norm;

	float r = 0.159154943*acos(DDz)/sqrt(DDx*DDx + DDy*DDy);

	float sb_u = 0.5 + DDx * r;
	float sb_v = 0.5 + DDy * r;

	int u = (int)floor(sb_u * mapWidth);
	int v = (int)floor(sb_v * mapHeight);

	int coord = v * mapWidth + u;
	if(coord > mapWidth*mapHeight) {
		coord -= mapWidth*mapHeight;
	}

	return m_envMap[coord];
}
