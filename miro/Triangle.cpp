#include "Triangle.h"
#include "TriangleMesh.h"
#include "Ray.h"

uint Triangle::triangleIntersectionsDone;

Triangle::Triangle(TriangleMesh * m, unsigned int i) :
	m_mesh(m), m_index(i)
{
	hasBVCoords = false;
	hasCenterCoords = false;
}


Triangle::~Triangle() {}

void
Triangle::calcBVCoords() {
	if(hasBVCoords) return; //To avoid memory leak and repeating calculations

	TriangleMesh::TupleI3 vti3 = m_mesh->vIndices()[m_index];
	Vector3 verts[3] = {
		m_mesh->vertices()[vti3.x],
		m_mesh->vertices()[vti3.y],
		m_mesh->vertices()[vti3.z]
	};

	m_cachedMaxCoords = (*new Vector3(-infinity));
	m_cachedMinCoords = (*new Vector3(infinity));

	for(int i = 0; i<3; i++) {
		//min
		if(verts[i].x < m_cachedMinCoords.x) m_cachedMinCoords.x = verts[i].x;
		if(verts[i].y < m_cachedMinCoords.y) m_cachedMinCoords.y = verts[i].y;
		if(verts[i].z < m_cachedMinCoords.z) m_cachedMinCoords.z = verts[i].z;
		//max
		if(verts[i].x > m_cachedMaxCoords.x) m_cachedMaxCoords.x = verts[i].x;
		if(verts[i].y > m_cachedMaxCoords.y) m_cachedMaxCoords.y = verts[i].y;
		if(verts[i].z > m_cachedMaxCoords.z) m_cachedMaxCoords.z = verts[i].z;
	}
	hasBVCoords = true;
}

void
Triangle::calcCenterCoords() {
	if(hasCenterCoords) return;

	TriangleMesh::TupleI3 vti3 = m_mesh->vIndices()[m_index];
	Vector3 verts[3] = {
		m_mesh->vertices()[vti3.x],
		m_mesh->vertices()[vti3.y],
		m_mesh->vertices()[vti3.z]
	};

	// m_cachedCenterCoords = verts[0] + (verts[1]-verts[0])/3.0 + (verts[2]-verts[0])/3.0;
	m_cachedCenterCoords = (verts[0] + verts[1] + verts[2])/3.0;
}


void
Triangle::renderGL()
{
	TriangleMesh::TupleI3 ti3 = m_mesh->vIndices()[m_index];
	const Vector3 & v0 = m_mesh->vertices()[ti3.x]; //vertex a of triangle
	const Vector3 & v1 = m_mesh->vertices()[ti3.y]; //vertex b of triangle
	const Vector3 & v2 = m_mesh->vertices()[ti3.z]; //vertex c of triangle

	glBegin(GL_TRIANGLES);
		glVertex3f(v0.x, v0.y, v0.z);
		glVertex3f(v1.x, v1.y, v1.z);
		glVertex3f(v2.x, v2.y, v2.z);
	glEnd();
}


//Using signed volume
bool
Triangle::intersect(HitInfo& result, const Ray& r,float tMin, float tMax)
{
	triangleIntersectionsDone++;

	TriangleMesh::TupleI3 vti3 = m_mesh->vIndices()[m_index];
	Vector3 a = m_mesh->vertices()[vti3.x]; //vertex a of triangle
	Vector3 b = m_mesh->vertices()[vti3.y]; //vertex b of triangle
	Vector3 c = m_mesh->vertices()[vti3.z]; //vertex c of triangle

	Vector3 q = r.d+r.o;
	Vector3 o = r.o;
	Vector3 d = r.d;

	Vector3 q_o = q-o;
	Vector3 a_o = a-o;
	Vector3 b_o = b-o;
	Vector3 c_o = c-o;

	float Va = dot(q_o, cross(c_o, b_o)) / 6.0;
	float Vb = dot(q_o, cross(a_o, c_o)) / 6.0;
	float Vc = dot(q_o, cross(b_o, a_o)) / 6.0;

	bool hit = (Va>0 && Vb>0 && Vc>0) || (Va<0 && Vb<0 && Vc<0);

	if(!hit) {
		return false;
	}

	float sum = Va + Vb + Vc;
	float alpha = Va / sum;
	float beta = Vb / sum;
	float gamma = Vc / sum;

	Vector3 P = alpha*a + beta*b + gamma*c;

	float t = dot((P - o), d);

	if(t<tMin || t>tMax) {
		return false;
	}

	TriangleMesh::TupleI3 nti3 = m_mesh->nIndices()[m_index];
	Vector3 nb = m_mesh->normals()[nti3.y];
	Vector3 na = m_mesh->normals()[nti3.x];
	Vector3 nc = m_mesh->normals()[nti3.z];

	Vector3 N = (alpha*na + beta*nb + gamma*nc).normalize();

	result.t = t;
	result.P = P;
	result.N = N;
	// result.material = m_material;
	result.material = m_mesh->material();

	return true;
}
