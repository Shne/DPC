#include "TriangleMesh.h"
#include "Triangle.h"
#include "Scene.h"

#define RAND ((float)rand()/(float)RAND_MAX)

TriangleMesh::TriangleMesh(Material* m) :
	m_normals(0),
	m_vertices(0),
	m_texCoords(0),
	m_normalIndices(0),
	m_vertexIndices(0),
	m_texCoordIndices(0),
	m_material(m)
{

}

TriangleMesh::~TriangleMesh()
{
	delete [] m_normals;
	delete [] m_vertices;
	delete [] m_texCoords;
	delete [] m_normalIndices;
	delete [] m_vertexIndices;
	delete [] m_texCoordIndices;
}

// HashGrid* 
HitInfo*
// std::vector<HitInfo>
TriangleMesh::calculateMPs(Vector3 minCorner, Vector3 maxCorner, HashGrid* hg, float hitPointRadius) {
	int hashSize = m_numTris;
	hg->initializeGrid(minCorner, maxCorner, hashSize, hitPointRadius);

	// HitInfo* mpArray[m_numTris];
	// std::vector<HitInfo> mpVector;
	HitInfo* mpArray[m_numTris*m_mpPerTri];
	// mpVector.resize(m_numTris);

	for(int i=0; i<m_numTris; i++) {
		TupleI3 vti3 = m_vertexIndices[i];
		Vector3 verts[3] = {
			m_vertices[vti3.x],
			m_vertices[vti3.y],
			m_vertices[vti3.z]
		};

		//Area of triangle
		Vector3 A = verts[0];
		Vector3 B = verts[1];
		Vector3 C = verts[2];
		Vector3 AB = B - A;
		Vector3 AC = C - A;
		float Area = cross(AB,AC).length() / 2.0;

		for(int j=0; j<m_mpPerTri; j++) {
			// a way of limiting the number of scatter samples below number of triangles
			float random = RAND;
			if(random > m_prop) continue;

			HitInfo mp = HitInfo();

			float R = RAND;
			float S = RAND;
			if(R + S >= 1.0) {
				R = 1.0 - R;
				S = 1.0 - S;
			}
			Vector3 randomPoint = A + R*AB + S*AC;


			// //point on triangle
			// TupleI3 vti3 = m_vertexIndices[i];
			// Vector3 verts[3] = {
			// 	m_vertices[vti3.x],
			// 	m_vertices[vti3.y],
			// 	m_vertices[vti3.z]
			// };
			// Vector3 middlePoint = (verts[0] + verts[1] + verts[2]) / 3.0;


			
			//interpolated normal for point
			TupleI3 nti3 = m_normalIndices[i];
			Vector3 norms[3] = {
				m_normals[nti3.x],
				m_normals[nti3.y],
				m_normals[nti3.z]
			};

			//using cramer'r rule
			//found at http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
			Vector3 v0 = B - A;
			Vector3 v1 = C - A;
			Vector3 v2 = randomPoint - A;
			float d00 = dot(v0, v0);
			float d01 = dot(v0, v1);
			float d11 = dot(v1, v1);
			float d20 = dot(v2, v0);
			float d21 = dot(v2, v1);
			float denom = d00 * d11 - d01 * d01;
			float alpha = (d11 * d20 - d01 * d21) / denom;
			float beta = (d00 * d21 - d01 * d20) / denom;
			float gamma = 1.0f - alpha - beta;


			Vector3 normal = alpha*norms[0] + beta*norms[1] + gamma*norms[2];
			// Vector3 normal = (norms[0] + norms[1] + norms[2]).normalize();
			
		
			mp.r2 = hitPointRadius*hitPointRadius;
			// mp->r2 = (Area/PI) * hitPointRadius*hitPointRadius; //using hitpoint radius as a scaling factor instead and set radius based on triangle size.
			// mp->P = middlePoint;
			mp.P = randomPoint;
			mp.N = normal;
			// mp->A = Area;
			mpArray[i*j] = mp;
			// mpVector.push_back(mp);
			hg->addHitPoint(&mp);
		}
	}
	std::cout << "Number of scatter samples:      " << mpVector.size() << "\n";
	// return hg;
	// return mpVector;
	return mpArray;
}

/*
HashGrid* 
TriangleMesh::calculateEvenlyDistributedMPs(Vector3 minCorner, Vector3 maxCorner, int noOfMPs) {
	HashGrid* hg = new HashGrid();
	int hashSize = 1024;
	float hitPointRadius = 5.0f;
	hg->initializeGrid(minCorner, maxCorner, hashSize, hitPointRadius);
	float forceScaling = 0.1;


	//Turk says it is important to completely evenly randomly distribute the points
	//binary tree of partial sums described in an article I can't get
	//So I just choose random triangles and place randomly on them
	//** I have later found that the hierarchical approach used in fast_bssrdf.pdf
	//** doesn't care whether the points are uniformly distributed.
	//** Since I can't find the articly describing how to use a "binary tree of partial sums"
	//** to evenly distribute the inital random points, and because of the difficulty in
	//** finding neighbour triangles in the current TriangleMesh implementation
	//** I will (for now at least) just put a measurepoint in the middle of every triangle.
	HitInfo* hiArray[noOfMPs];
	Vector3 forceArray[noOfMPs];

	for(int i=0; i<noOfMPs; i++) {
		HitInfo* mp = new HitInfo();

		//random triangle
		int triangleIndex = rand() % m_numTris;	
		TupleI3 vti3 = m_vertexIndices[triangleIndex];
		Vector3 verts[3] = {
			m_vertices[vti3.x],
			m_vertices[vti3.y],
			m_vertices[vti3.z]
		};

		//random place on triangle. random barycentric coords
		Vector3 A = verts[0];
		Vector3 B = verts[1];
		Vector3 C = verts[2];
		Vector3 AB = B - A;
		Vector3 AC = C - A;
		float R = ((float)rand()/(float)RAND_MAX);
		float S = ((float)rand()/(float)RAND_MAX);
		if(R + S >= 1.0) {
			R = 1.0 - R;
			S = 1.0 - S;
		}
		Vector3 randomPoint = A + R*AB + S*AC;
		Vector3 normal = cross(A,B).normalize();

		mp->P = randomPoint;
		mp->N = normal;
		hiArray[i] = mp;
		hg->addHitPoint(mp);
	}

	
	// from Generating Textures on Arbitrary Surfaces Using Reaction-Diffusion
	// Greg Turk - University of North Carolina at Chapel Hill
	// loop k times
	int k = 20;
	for(int j=0; j<k; j++) {
		// for each point P on surface
		for(int i=0; i<noOfMPs; i++) {
			Vector3 planeP = hiArray[i]->P;
			Vector3 planeN = hiArray[i]->N;
			// determine nearby points to P
			std::forward_list<HitInfo*> hiList = hg->lookup(planeP);
			// map these nearby points onto the plane containing the polygon of planeP
			for(auto hiIter=hiList.begin(); hiIter!=hiList.end(); ++hiIter) {
				Vector3 nearP = (*hiIter)->P;
				Vector3 projectedPoint = nearP - dot(nearP - planeP, planeN) * planeN //this is wrong. needs to fold around edges
				// compute and store the repulsive forces that the mapped points exert on P
				// for now, just try simply the distance ~jhk
				Vector3 distance = planeP - projectedPoint;
				forceArray[i] += forceScaling*distance;
				
				// r = 2 * sqrt(a/n)
				// u = area of surface
				// n = number of points on surface
				
			}
		}
		// for each point P on surface
		for(int i=0; i<noOfMPs; i++) {
			// compute the new position of P based on the repulsive forces
			HitPoint* hi = hiArray[i];
			hi->P += forceArray[i]; //should also fold around edges
		}
	}

	return hg;
}
*/