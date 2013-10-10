#include "Material.h"



Material::Material()
{
}

Material::~Material()
{
}

Vector3
Material::shade(const Ray&, const HitInfo&, Scene&, int depth) const
{
	return Vector3(1.0f, 1.0f, 1.0f);
}


