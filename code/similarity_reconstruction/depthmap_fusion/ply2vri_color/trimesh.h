// trimesh.h
//
// A class for manipulating triangle meshes.
// by Brett Allen
//
// *** For license information, please see license.txt. ***

#ifndef _TRIMESH_H_
#define _TRIMESH_H_

#include <vector>
using namespace std;
#include "vec.h"
#include "mat.h"

typedef vector<int> intVec;

class HBBox;

const int VM_WIREFRAME = 0;
const int VM_FLAT      = 1;
const int VM_ROUND     = 2;

class TriMesh {
protected:
	int getPointIndex(Vec3d pt);
	void addTri(int v0, int v1, int v2);
	HBBox *calcHBBRecursive(int maxSize, intVec &tris);

public:
	typedef vector<Vec3d> ptVec;
	ptVec  m_points;
	ptVec  m_triNormals;
	ptVec  m_vertexNormals;
	ptVec  m_colors;
	intVec m_tris;
	vector<double> m_conf;
	double xClip;
	Vec3d solidColor;
	double alpha;
	bool showColor;

        // added by zc
        int closest_tri_idx;

	HBBox  *m_hbb;
	
	TriMesh();
	TriMesh(TriMesh *t);
	~TriMesh();

	bool loadRaw(istream &in);
	bool loadTxt(istream &in);
	bool loadPly(char *fn, bool color = false);
        bool savePly(const char *fn);
	void makeCylinder(double len, double rad, int lenSegs, int radSegs, double eRad = 0, double zMult = 1.0);
	void calcNormals();

	void copyPoints(TriMesh *source);

	void scale(double xs, double ys, double zs);
	void translate(double xt, double yt, double zt);
	void transform(Mat4d &m);

	void render(int viewMode = VM_WIREFRAME);

	void calcTriBounds(double *bb, intVec &tris);
	void calcPointBounds(double *bb, int *pts, int numPts);
	void calcPointBounds(double *bb);
	void dumpBounds();

	void calcHBB(int maxSize);

	Vec3d rayOrig, rayDir;
	double tPos, tNeg;
	bool hitPos, hitNeg;
	bool calcRayIntersection(Vec3d orig, Vec3d dir);
	void calcRayIntersection(intVec &tris);
	void calcRayIntersection(HBBox *hbb);
	double closestDist;
	Vec3d closestPt, closestBary;
	int closestTri[3];
	bool calcClosestPoint(Vec3d orig, double closest, Vec3d normalMatch = Vec3d());
	bool calcClosestPoint(intVec &tris);
	bool calcClosestPoint(HBBox *hbb);
};

#endif
