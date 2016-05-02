// ply2vri.cpp
//
// Scanline converts a triangle mesh in a ply file to a voxel grid.
//
// by Brett Allen
// 2002
//
// *** For license information, please see license.txt. ***

#include <fstream>
#include <iomanip>
#include <limits.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <string>
#include "trimesh.h"
#include "vrip/OccGridRLE.h"

#define CLOSEST_TRI_IDX

using namespace std;

const double EPSILON = 0.0000001;

// defaults
int rampSize = 5;
double resolution = 0.002;
bool hardEdges = true;
bool useConf = false;
int stripEdges = 0;
int algorithm = 0;
OccGridRLE *templateVRI = NULL;

// A simple class for holding expanded voxel data:
class VoxelGrid {
public:
	int max[3];
	int numPts;
	unsigned short *rawData;
#ifdef CLOSEST_TRI_IDX
        int *triIndices;
#endif

        VoxelGrid():rawData(NULL)
  #ifdef CLOSEST_TRI_IDX
          , triIndices(NULL)
  #endif
        {}
        ~VoxelGrid() { clearData(); }
	bool initData(int x, int y, int z) {
                clearData();
		max[0] = x;
		max[1] = y;
		max[2] = z;
		numPts = max[0] * max[1] * max[2];
                rawData = new unsigned short[4 * numPts];
		if (rawData == NULL)
			return false;

                memset(rawData, 0, 4 * max[0] * max[1] * max[2] * sizeof(unsigned short));

#ifdef CLOSEST_TRI_IDX
                triIndices = new int[4 * numPts];
                if (triIndices == NULL) return false;
                for (int i = 0; i < 4 * numPts; ++i) triIndices[i] = -1; // no closest triangle
#endif
		return true;
	}

        // added by zc
        bool clearData()
        {
            if (rawData) { delete [] rawData;  rawData = NULL; }
#ifdef CLOSEST_TRI_IDX
            if (triIndices) { delete [] triIndices; triIndices = NULL; }
#endif
        }

	unsigned short &dist(int x, int y, int z) {
                return rawData[4 * ((z*max[1] + y)*max[0] + x) + 0];
	}

	unsigned short &conf(int x, int y, int z) {
                return rawData[4 * ((z*max[1] + y)*max[0] + x) + 1];
	}

        void set_color(int x, int y, int z, unsigned char r, unsigned char g, unsigned char b) {
                rawData[4 * ((z*max[1] + y)*max[0] + x) + 2] = g << 8 | b;
                rawData[4 * ((z*max[1] + y)*max[0] + x) + 3] = r;
        }

#ifdef CLOSEST_TRI_IDX
        int &closest_tri_idx(int x, int y, int z) {
            return triIndices[ ((z*max[1] + y)*max[0] + x) * 4 + 0 ];
        }
        void set_closest_tri_intersection(int x, int y, int z, const int *vert_idx) {
            triIndices[ ((z*max[1] + y)*max[0] + x) * 4 + 1 ] = vert_idx[0];
            triIndices[ ((z*max[1] + y)*max[0] + x) * 4 + 2 ] = vert_idx[1];
            triIndices[ ((z*max[1] + y)*max[0] + x) * 4 + 3 ] = vert_idx[2];
        }
        Vec3i closest_tri_point_idx(int x, int y, int z) {
            return Vec3i(triIndices[ ((z*max[1] + y)*max[0] + x) * 4 + 1 ],
                    triIndices[ ((z*max[1] + y)*max[0] + x) * 4 + 2 ],
                    triIndices[ ((z*max[1] + y)*max[0] + x) * 4 + 3 ]);
        }
#endif

	void removeLowConf(unsigned short minConf) {
		int x, y, z;
		for (z=0; z < max[2]; z++) {
			for (y=0; y < max[1]; y++) {
				for (x=0; x < max[0]; x++) {
					if (conf(x, y, z) <= minConf) {
						conf(x, y, z) = 0;
						dist(x, y, z) = USHRT_MAX;
					}
				}
			}
		}
	}

	void savePPM(char *fname) {
		ofstream out(fname);
		if (!out.good()) {
			cout << "can't open " << fname << endl;
			return;
		}

		out << "P3" << endl << max[0] << " " << (max[1]*max[2]) << endl << 255 << endl;
		int x, y, z;
		for (z=0; z < max[2]; z++) {
			for (y=0; y < max[1]; y++) {
				for (x=0; x < max[0]; x++) {
					if (conf(x, y, z) == 0 && dist(x, y, z) == USHRT_MAX)
						out << "128 64 0" << endl;
					else
						out << (dist(x, y, z) >> 8) << " " << (dist(x, y, z) >> 8) << " " << (dist(x, y, z) >> 8) << endl;
				}
			}
		}
		out.close();
	}

	void saveRaw(char *fname, double scale = 1, Vec3d trans = Vec3d(0, 0, 0)) {
		FILE *f = fopen(fname, "wb");
		if (!f) {
			cout << "can't open " << fname << endl;
			return;
		}

		int x, y, z;
		x = 0;
		// version
		fwrite(&x, sizeof(int), 1, f);

		fwrite(&scale, sizeof(double), 1, f);
		fwrite(&trans[0], sizeof(double), 1, f);
		fwrite(&trans[1], sizeof(double), 1, f);
		fwrite(&trans[2], sizeof(double), 1, f);

		fwrite(&max[0], sizeof(int), 1, f);
		fwrite(&max[1], sizeof(int), 1, f);
		fwrite(&max[2], sizeof(int), 1, f);

		for (z=0; z < max[2]; z++) {
			for (y=0; y < max[1]; y++) {
				for (x=0; x < max[0]; x++) {
                                        fwrite(rawData + (((z*max[1] + y) * max[0] + x) * 2 /*not valid any more*/ ), sizeof(unsigned short), 1, f);
				}
			}
		}
		fclose(f);
	}

	void saveVRI(char *fname, double scale = 1, Vec3d trans = Vec3d(0, 0, 0)) {
		OccGridRLE rle;
		rle.init(max[0], max[1], max[2], 1000000);

		rle.resolution = scale;
		rle.origin[0] = trans[0];
		rle.origin[1] = trans[1];
		rle.origin[2] = trans[2];
		int y, z;
		for (z = 0; z < max[2]; z++) {
			for (y = 0; y < max[1]; y++) {
                                //cout << "z, y: " << z << " " << y << endl;
                                rle.putScanline((OccElement *)(rawData + ((z*max[1] + y) * max[0] * 4)), y, z);
			}
		}
		rle.write(fname);
	}

#ifdef CLOSEST_TRI_IDX
        void SaveClosestTriangles(const char *fname, double scale = 1, Vec3d trans = Vec3d(0, 0, 0)) {
            FILE* f = fopen(fname, "wb");
            if (!f) {
                    cout << "can't open " << fname << endl;
                    return;
            }
            int x, y, z;
            x = 0;
            // version
            fwrite(&x, sizeof(int), 1, f);

            fwrite(&scale, sizeof(double), 1, f);
            fwrite(&trans[0], sizeof(double), 1, f);
            fwrite(&trans[1], sizeof(double), 1, f);
            fwrite(&trans[2], sizeof(double), 1, f);

            fwrite(&max[0], sizeof(int), 1, f);
            fwrite(&max[1], sizeof(int), 1, f);
            fwrite(&max[2], sizeof(int), 1, f);

            int vox_pos[3] = {0};
            for (z=0; z < max[2]; z++) {
                for (y=0; y < max[1]; y++) {
                    for (x=0; x < max[0]; x++) {
                        if ( triIndices[ (((z*max[1] + y) * max[0] + x) * 1) ] >= 0 ) {
                            vox_pos[0] = x; vox_pos[1] = y; vox_pos[2] = z;
                            fwrite(vox_pos, sizeof(int), 3, f);
                            fwrite(triIndices + (((z*max[1] + y) * max[0] + x) * 4), sizeof(int), 4, f);
                        }
                    }
                }
            }
            fclose(f);
        }
#endif

        // added by zc
        void DumpGridPoints(const Vec3d &world_min_pt, const Vec3d &world_max_pt, const double resolution, const Vec3d& trans, const int rampsize, const char *fname);
        void SetClosestTrimeshColor(const Vec3d& world_min_pt, const Vec3d& world_max_pt, const double resolution, const Vec3d &trans, const int rampsize, const TriMesh &depthmap_tm, TriMesh* closesttri_tm);
};


struct EdgeInfo {
	int vert;
	int count;
	Vec3d normal;

	EdgeInfo() {
		count = 0;
		normal = Vec3d(0, 0, 0);
	}
};


// This class is just a list of every edge in a mesh.  We'll
// use it to find out if an edge appears twice.
class EdgeList {
public:
	int numVerts;
	typedef vector<EdgeInfo> edgeVec;
	edgeVec *edges;

	EdgeList() {
		edges = NULL;
	}

	void init(int nv) {
		numVerts = nv;
		if (edges)
			delete []edges;
		edges = new edgeVec[numVerts];
	}

	EdgeInfo *findEdge(int v0, int v1) {
		if (v1 < v0)
			swap(v1, v0);

		int i;
		for (i=0; i < edges[v0].size(); i++)
			if (edges[v0][i].vert == v1)
				return &edges[v0][i];
		return NULL;
	}

	void addEdge(int v0, int v1, Vec3d normal) {
		if (v1 < v0)
			swap(v1, v0);

		EdgeInfo newEI;
		bool isNew = false;
		EdgeInfo *ei = findEdge(v0, v1);
		if (ei == NULL) {
			ei = &newEI;
			isNew = true;
		}

		ei->vert = v1;
		ei->count++;
		ei->normal += normal;

		if (isNew)
			edges[v0].push_back(*ei);
	}

	/*
	int containsEdge(int v0, int v1) {
		if (v1 < v0)
			swap(v1, v0);
		int i;
		for (i=0; i < edges[v0].size(); i++)
			if (edges[v0][i] == v1)
				return true;
		return false;
	}
	*/

	void buildFromTriMesh(TriMesh &tm) {
		init(tm.m_points.size());
		int i;

		for (i=0; i < tm.m_tris.size(); i += 3) {
			addEdge(tm.m_tris[i+0], tm.m_tris[i+1], tm.m_triNormals[i/3]);
			addEdge(tm.m_tris[i+1], tm.m_tris[i+2], tm.m_triNormals[i/3]);
			addEdge(tm.m_tris[i+2], tm.m_tris[i+0], tm.m_triNormals[i/3]);
		}

/*		int j;
		for (i=0; i < numVerts; i++) {
			if (edges[i].size() > 0) {
				for (j=0; j < edges[i].size(); j++) {
					edges[i][j].normal = 0.5 * (tm.m_vertexNormals[i] + tm.m_vertexNormals[j]);
				}
			}
		}*/
	}

	void markVerts(bool *verts) {
		memset(verts, 0, sizeof(bool) * numVerts);
		int i, j;
		for (i=0; i < numVerts; i++) {
			if (edges[i].size() > 0) {
				for (j=0; j < edges[i].size(); j++) {
					if (edges[i][j].count == 1) {
						verts[i] = true;
						verts[edges[i][j].vert] = true;
					}
				}
			}
		}
	}
};

bool *vertList;
EdgeList edgeList;


static double dMax(double a, double b) {
	if (a > b)
		return a;
	return b;
}

static double dMin(double a, double b) {
	if (a < b)
		return a;
	return b;
}

void updateFromClosest(TriMesh &tm, Vec3d &voxelPt, VoxelGrid &vg) {
	double dist = tm.closestDist;
	unsigned short conf = USHRT_MAX;
        // added by zc
        Vec3d grid_point_color;
        // tm.closestTri: the three vert indices of the closest triangle

	// if closest point is a vertex, check if we need to flip the sign
	if (tm.closestTri[1] == -1) {
                Vec3d delta = tm.closestPt - voxelPt;
                // modified by zc
                // Vec3d delta = voxelPt - tm.closestPt;
		delta.normalize();
		double dotProd = delta * tm.m_vertexNormals[tm.closestTri[0]];
		if (dotProd < 0)
			dist *= -1;

		// check if this vertex adjoins a hole
		if (hardEdges && vertList[tm.closestTri[0]])
			conf = 0;

                if (!tm.m_colors.empty())
                        grid_point_color = tm.m_colors[tm.closestTri[0]];
	}
	// if closest point is an edge, check if we need to flip the sign
	else if (tm.closestTri[2] == - 1) {
                Vec3d delta = tm.closestPt - voxelPt;
                // modified by zc
                // Vec3d delta = voxelPt - tm.closestPt;
		delta.normalize();

		EdgeInfo *ei = edgeList.findEdge(tm.closestTri[0], tm.closestTri[1]);
		if (!ei) {
			cout << "error: unknown edge!!" << endl;
			return;
		}
		double dotProd = delta * ei->normal;
		if (dotProd < 0)
			dist *= -1;

		// check if this edge adjoins a hole
		if (hardEdges && ei->count == 1)
			conf = 0;

                // pt = (verts[dVert0] + (verts[dVert1] - verts[dVert0]) * param);
                // bary[0] = param;
                if (!tm.m_colors.empty())
                        grid_point_color = tm.m_colors[tm.closestTri[0]] +
                                ((tm.m_colors[tm.closestTri[1]] - tm.m_colors[tm.closestTri[0]]) * tm.closestBary[0]);

                if (voxelPt == Vec3d(105, 29, 16))
                {
                    cout << endl;
                    cout << "voxelPt: " << voxelPt << endl;
                    cout << "closestPt: " << tm.closestPt << endl;
                    cout << "delta: " << delta << endl;
                    cout << "ei normal: " << ei->normal << endl;
                    cout << "dotProd: " << dotProd << endl;
                    cout << "dist: " << dist << endl;
                    cout << "conf: " << conf << endl;
                    cout << "closest_tri: " << tm.closest_tri_idx << endl;
                    cout << "closest_tri_pt: " << tm.closestTri[0] << " " << tm.closestTri[1] << " " << tm.closestTri[2] << endl;
                }
	}
        else {
            // added by zc, compute grid_point_color when the closest point is on the triangle
            if (!tm.m_colors.empty())
                    grid_point_color = tm.m_colors[tm.closestTri[0]] * tm.closestBary[0] +
                            tm.m_colors[tm.closestTri[1]] * tm.closestBary[1] +
                            tm.m_colors[tm.closestTri[2]] * tm.closestBary[2];
        }

	dist = dist / (rampSize * 2);
        // dist = (-dist + 0.5) * USHRT_MAX;
        // modified by zc (sign flipped)
        dist = (dist + 0.5) * USHRT_MAX;
	if (dist < 0)
		dist = 0;
	if (dist > USHRT_MAX)
		dist = USHRT_MAX;

        vg.dist(voxelPt[0], voxelPt[1], voxelPt[2]) = (unsigned short)round(dist);
	vg.conf(voxelPt[0], voxelPt[1], voxelPt[2]) = conf;

        // update closest tri, added by zc
        vg.closest_tri_idx(voxelPt[0], voxelPt[1], voxelPt[2]) = tm.closest_tri_idx;
        vg.set_closest_tri_intersection(voxelPt[0], voxelPt[1], voxelPt[2], tm.closestTri);

        if (!tm.m_colors.empty())
        {
            Vec3d grid_point_color_255 = grid_point_color * 255;
            uchar r = (uchar)grid_point_color_255[0];
            uchar g = (uchar)grid_point_color_255[1];
            uchar b = (uchar)grid_point_color_255[2];
            vg.set_color(voxelPt[0], voxelPt[1], voxelPt[2], r, g, b);
//            cout << "dist: " << (unsigned short)round(dist) << endl;
//            cout << "conf: " << conf << endl;
             //cout << "rgb: " << (int)r << " " << (int)g << " " << (int)b << endl;
        }
//        cout << "dist: " << (unsigned short)(dist) << endl;
//        cout << "conf: " << conf << endl;
}

void processVoxel(TriMesh &tm, int x, int y, int z, VoxelGrid &vg) {
	Vec3d voxelPt = Vec3d(x, y, z);
	tm.closestDist = dMin(rampSize, fabs(((double)vg.dist(x, y, z) / USHRT_MAX) - 0.5) * (rampSize * 2));
	
	if (tm.calcClosestPoint(voxelPt, tm.closestDist)) {
		updateFromClosest(tm, voxelPt, vg);
	}
	else
		vg.conf(x, y, z) = 0;
}

void processTriangle(TriMesh &tm, int tri, VoxelGrid &vg, bool markOnly = false) {
// scan convert a triangle into the voxel grid

	double triBB[6];
	int bb[6];
	Vec3d delta;
	double conf;
	int vertIndices[3];
	int i;

	for (i=0; i < 3; i++)
		vertIndices[i] = tm.m_tris[tri + i];

	// ignore triangles with low confidence
	if (useConf && tm.m_conf.size() > 0) {
		conf = tm.m_conf[vertIndices[0]] + tm.m_conf[vertIndices[1]] + tm.m_conf[vertIndices[2]];
		if (conf < 0.01)
			return;
	}

	tm.calcPointBounds(triBB, vertIndices, 3);

	Vec3d verts[3];
	Vec3d norm;

	norm = tm.m_triNormals[tri / 3];

	for (i=0; i < 3; i++) {
		bb[i*2+0] = dMax(0, ceil(triBB[i*2+0]) - rampSize);
		bb[i*2+1] = dMin(vg.max[i]-1, floor(triBB[i*2+1]) + rampSize);

		verts[i] = tm.m_points[vertIndices[i]];
	}

	double baryDenom = ((verts[1] - verts[0]) ^ (verts[2] - verts[0])) * norm;

	vector<int> pts;
	pts.push_back(vertIndices[0]);
	pts.push_back(vertIndices[1]);
	pts.push_back(vertIndices[2]);

	int x, y, z;
	for (z = bb[4]; z <= bb[5]; z++) {
		for (y = bb[2]; y <= bb[3]; y++) {
			for (x = bb[0]; x <= bb[1]; x++) {
				if (markOnly) {
					vg.conf(x, y, z) = 1;
					continue;
				}

				Vec3d voxelPt(x, y, z);

				tm.rayOrig = voxelPt;
				tm.closestDist = dMin(rampSize, fabs(((double)vg.dist(x, y, z) / USHRT_MAX) - 0.5) * (rampSize * 2));
				
				if (tm.calcClosestPoint(pts)) {
                                    // tm.closestDist is updated if calcClosestPoint returns true
                                    tm.closest_tri_idx = tri/3;
                                    updateFromClosest(tm, voxelPt, vg);
				}
			}
		}
	}
}

int main(int argc, char *argv[]) {
	TriMesh tm;
	double scale;
	Vec3d trans;
	char input[1024], output[1024];
	int i;

	if (argc < 2 || (argc == 2 && argv[1][0] == '-')) {
		cout << "usage: ply2vri [options] <input> [<output>]" << endl
			<< "options are as follows:" << endl
			<< "  -r     voxel size (defaults is -r" << resolution << ")" << endl
			<< "  -l     ramp length (defaults is -l" << rampSize << ")" << endl
			<< "  -h     hard edges (default is -h" << (hardEdges?"1":"0") << "):" << endl
			<< "             0 = report true distance to mesh around holes" << endl
			<< "             1 = create a sharp cutoff around holes" << endl
			<< "  -s     # of triangles to strip from the edges (default is -s" << stripEdges << ")" << endl
			<< "  -a     algorithm to use (default is -a" << algorithm << "):" << endl
			<< "             0 = iterate over triangles" << endl
			<< "             1 = iterate over voxels" << endl
			<< "  -v     name of a vri file to use as a template and copy space carving" << endl
			<< "           information from (this will override some parameters)" << endl
			<< endl
			<< "input is a ply file containing triangles or triangle strips" << endl 
			<< "ouput is the vri file to save (if the extension is ppm," << endl
			<< "  a ppm file will be saved instead)" << endl;
		return 0;
	}

	strcpy(input, "in.ply");
	output[0] = 0;

	// process command-line parameters
	for (i = 1; i < argc-1; i++) {
		if (argv[i][0] == '-') {
			switch (argv[i][1]) {
			case 'r':
				resolution = atof(argv[i]+2);
				break;
			case 'l':
				rampSize = atoi(argv[i]+2);
				break;
			case 'h':
				hardEdges = (argv[i][2] == '1');
				break;
			case 's':
				stripEdges = atoi(argv[i]+2);
				break;
			case 'a':
				algorithm = atoi(argv[i]+2);
				break;
			case 'v':
				templateVRI = new OccGridRLE(1, 1, 1, 1000000);
				if (!templateVRI->read(argv[i]+2)) {
					delete templateVRI;
					templateVRI = NULL;
					cout << "can't load template VRI, '" << argv[i]+2 << "'" << endl;
				}
				else {
					resolution = templateVRI->resolution;
					trans[0] = -templateVRI->origin[0];
					trans[1] = -templateVRI->origin[1];
					trans[2] = -templateVRI->origin[2];
				}
				break;
			}
		}
	}

	// establish input and output file names
	if (argc < 3 || argv[argc-2][0] == '-') {
		strncpy(input, argv[argc-1], 1023);
		strncpy(output, argv[argc-1], 1023);
		output[strlen(output)-3] = 'v';
		output[strlen(output)-2] = 'r';
		output[strlen(output)-1] = 'i';
	}
	else {
		strncpy(input, argv[argc-2], 1023);
		strncpy(output, argv[argc-1], 1023);
	}

	// report parameters to user
	if (templateVRI)
		cout << "using a template VRI file" << endl;
	cout << "input is '" << input << "'" << endl;
	cout << "output is '" << output << "'" << endl;
	cout << "resolution is " << resolution << endl;
	cout << "ramp length is " << rampSize << endl;
	cout << "hard edges are " << (hardEdges?"enabled":"disabled") << endl;
	cout << "will strip " << stripEdges << " edge(s)" << endl;
	cout << "algorithm is " << algorithm << endl;
	cout << endl;

        if (!tm.loadPly(input, true))
		return 0;
        cout << "loaded " << (tm.m_tris.size()/3) << " triangles" << endl;
        if (tm.m_colors.empty())
        {
            cout << "no color information read, adding color information for testing" << endl;
            Vec3d color(1, 0, 0);
            tm.m_colors.assign(tm.m_points.size(), color);
            cout << "finished adding color information for testing" << endl;
        }


        cout << "computing point bounds" << endl;
	// scale and translate the triangle mesh so that 1 unit == 1 voxel
        double bb[6];
        tm.calcPointBounds(bb);
        cout << "finished point bounds" << endl;
	if (!templateVRI)
		trans = Vec3d(-bb[0] + resolution * rampSize, -bb[2] + resolution * rampSize, -bb[4] + resolution * rampSize);
        // added by zc
        // clamp to final grid points
        Vec3d final_offset = Vec3d(0, 0, 0); // defaults to 0, 0, 0, voxel2World offset in final grid
        trans = floor(((-trans) - final_offset)/resolution) * resolution;
        trans = -trans;

        tm.translate(trans[0], trans[1], trans[2]);
        cout << "finished translating" << endl;

	VoxelGrid vg;
	if (templateVRI) {
		if (!vg.initData(templateVRI->xdim, templateVRI->ydim, templateVRI->zdim)) {
			cout << "couldn't allocate " << (vg.max[0]*vg.max[1]*vg.max[2]*2*sizeof(short)) << " bytes of voxel data; aborting" << endl;
			return 0;
		}
	}
	else {
		if (!vg.initData(
                            // +1 added by zc to compensate for clamping current grid points to final grid points
                        ceil((bb[1]-bb[0]) / resolution) + rampSize * 2 + 1,
                        ceil((bb[3]-bb[2]) / resolution) + rampSize * 2 + 1,
                        ceil((bb[5]-bb[4]) / resolution) + rampSize * 2 + 1)) {
			cout << "couldn't allocate " << (vg.max[0]*vg.max[1]*vg.max[2]*2*sizeof(short)) << " bytes of voxel data; aborting" << endl;
			return 0;
		}
	}

	cout << "grid size: " << vg.max[0] << " " << vg.max[1] << " " << vg.max[2] << " (" <<
		(vg.max[0]*vg.max[1]*vg.max[2]) << " voxels)" << endl;

	scale = 1.0 / resolution;
	tm.scale(scale, scale, scale);
	tm.calcNormals();

	// detect edges of holes:
	vertList = NULL;
	bool doStrip = (stripEdges > 0);
        cout << "strip Edges: " << stripEdges << endl;
	do {
                printf("doing strip\n");
		edgeList.buildFromTriMesh(tm);
		if (vertList)
			delete []vertList;
                // if the vert is on the boundary, mark it as true
		vertList = new bool[edgeList.numVerts];
		edgeList.markVerts(vertList);

		if (stripEdges > 0) {
                        cout << "cur_stripEdge " << stripEdges << endl;
			// remove all vertices that are on an edge
			vector<Vec3d> oldPoints = tm.m_points;
			vector<Vec3d> oldVNormals = tm.m_vertexNormals;
			vector<double> oldConf = tm.m_conf;
                        vector<Vec3d> oldColors = tm.m_colors;  // [0, 1]

			vector<int> vertMap;
			tm.m_points.clear();
			tm.m_vertexNormals.clear();
			tm.m_conf.clear();
                        tm.m_colors.clear();

                        cout << "oldpointsize: " << oldPoints.size() << endl;

			for (i=0; i < oldPoints.size(); i++) {
				if (!vertList[i]) {
                                       //cout << "reserve " << i << "th vert" << endl;
					vertMap.push_back(tm.m_points.size());

					tm.m_points.push_back(oldPoints[i]);
					tm.m_vertexNormals.push_back(oldVNormals[i]);
                                        if (oldConf.empty())
                                        {
                                            tm.m_conf.push_back(1.0);
                                        }
                                        else
                                        {
                                            tm.m_conf.push_back(oldConf[i]);
                                        }
                                        tm.m_colors.push_back(oldColors[i]);
				}
				else
                                {
                                       //cout << "remove " << i << "th vert" << endl;
					vertMap.push_back(-1);
                                }
			}

			// remove all triangles that are on an edge
			vector<int> oldTris;
			vector<Vec3d> oldTNormals = tm.m_triNormals;
			oldTris = tm.m_tris;
			tm.m_tris.clear();
			tm.m_triNormals.clear();
			for (i=0; i < oldTris.size(); i += 3) {
				if (vertList[oldTris[i + 0]] || vertList[oldTris[i + 1]] || vertList[oldTris[i + 2]]) {
					continue;
				}
				tm.m_tris.push_back(vertMap[oldTris[i + 0]]);
				tm.m_tris.push_back(vertMap[oldTris[i + 1]]);
				tm.m_tris.push_back(vertMap[oldTris[i + 2]]);

				tm.m_triNormals.push_back(oldTNormals[i/3]);
			}
		}

		stripEdges--;
	} while (stripEdges > -1);

        if (doStrip) {
		cout << "stripped down to " << tm.m_tris.size()/3 << " triangles" << endl;		
	}

	int tenPercent;

	switch (algorithm) {
	case 0:
		// per-triangle algorithm
		tenPercent = (tm.m_tris.size() / 3 / 10) * 3;
		cout << "scan converting..." << endl << " |          |\n  ";
		for (i=0; i < tm.m_tris.size(); i += 3) {
			if (tenPercent > 0 && i % tenPercent == 0) {
				cout << "*";
				cout.flush();
			}
			processTriangle(tm, i, vg);
		}
		cout << endl;
		break;

	case 1:
		// per-voxel algorithm
		cout << "calculating hbb" << endl;
		tm.calcHBB(100);
		// mark voxels to examine
		cout << "marking" << endl;
		for (i=0; i < tm.m_tris.size(); i += 3) {
			processTriangle(tm, i, vg, true);
		}
		tenPercent = vg.max[2] / 10;
		cout << "scan converting..." << endl << " |          |\n  ";
		int x, y, z;
		for (z=0; z < vg.max[2]; z++) {
			for (y=0; y < vg.max[1]; y++) {
				for (x=0; x < vg.max[0]; x++) {
					if 	(vg.conf(x, y, z) == 1)
						processVoxel(tm, x ,y, z, vg);
				}
			}
			if (tenPercent > 0 && z % tenPercent == 0) {
				cout << "*";
				cout.flush();
			}
		}
		cout << endl;
		break;
	}
	
	// turn zero confidence voxels into special empty voxels
	vg.removeLowConf(0);

	if (templateVRI) {
		cout << "copying space carving information from VRI" << endl;
		int x, y, z;
		for (z=0; z < vg.max[2]; z++) {
			for (y=0; y < vg.max[1]; y++) {
				OccElement *curScanline = templateVRI->getScanline(y, z);
				for (x=0; x < vg.max[0]; x++) {
					if (curScanline[x].value == 0 && curScanline[x].totalWeight == 0) {
						vg.conf(x, y, z) = 0;
						vg.dist(x, y, z) = 0;
					}
				}
			}
		}
	}

    //    Vec3d ext(0.6, 0.6, 0.6);
    //    Vec3d world_pt_center(1230.3, 3699.2, 116.6);
    //    Vec3d world_min_pt = world_pt_center - ext;
    //    Vec3d world_max_pt = world_pt_center + ext;
    //    vg.DumpGridPoints(world_min_pt, world_max_pt,  resolution, -trans, rampSize, (std::string(output) + "_gridinfo.ply").c_str());
    //    TriMesh closest_tm;
    //    vg.SetClosestTrimeshColor(world_min_pt, world_max_pt, resolution, -trans, rampSize, tm, &closest_tm);
    //    closest_tm.savePly((std::string(output) + "closest_tm.ply").c_str());

	if (strcmp(output + strlen(output) - 3, "ppm") == 0) {
		cout << "saving ppm '" << output << "'" << endl;
		vg.savePPM(output);
	}
	else if (strcmp(output + strlen(output) - 3, "raw") == 0) {
		cout << "saving raw '" << output << "'" << endl;
		vg.saveRaw(output, resolution, -trans);
	}
	else {
		cout << "saving vri '" << output << "'" << endl;
		vg.saveVRI(output, resolution, -trans);

		// for testing
//		*(argv[2] + strlen(argv[2]) - 3) = 'r';
//		*(argv[2] + strlen(argv[2]) - 2) = 'a';
//		*(argv[2] + strlen(argv[2]) - 1) = 'w';
//		cout << "saving raw '" << argv[2] << "'" << endl;
//		vg.saveRaw(argv[2], resolution, -trans);
	}
	return 0;
}


void VoxelGrid::DumpGridPoints(
        const Vec3d &world_min_pt,
        const Vec3d &world_max_pt,
        const double resolution, const Vec3d& trans, const int rampsize,
        const char *fname)
{
    const float map[8][4] = {
        {0,0,0,114},{0,0,1,185},{1,0,0,114},{1,0,1,174},
        {0,1,0,114},{0,1,1,185},{1,1,0,114},{1,1,1,0}
    };
    float sum = 0;
    for (int32_t i=0; i<8; i++)
        sum += map[i][3];

    float weights[8]; // relative weights
    float cumsum[8];  // cumulative weights
    cumsum[0] = 0;
    for (int32_t i=0; i<7; i++)
    {
        weights[i]  = sum/map[i][3];
        cumsum[i+1] = cumsum[i] + map[i][3]/sum;
    }

    TriMesh tm;
    std::ofstream ofs( (std::string(fname) + "_tsdf_info.txt").c_str() );
    for (int z=0; z < max[2]; z++) {
            for (int y=0; y < max[1]; y++) {
                    for (int x=0; x < max[0]; x++) {
                            unsigned short d = dist(x, y, z);
                            unsigned short voxel_conf  = conf(x, y, z);
                            if (voxel_conf > 0)
                            {
                                float scaled_val = d / 65535.0;
                                // find bin
                                int32_t i;
                                for (i=0; i<7; i++)
                                    if (scaled_val < cumsum[i+1])
                                        break;
                                // compute red/green/blue values
                                float   w = 1.0 - (scaled_val-cumsum[i])*weights[i];
                                uint8_t r = (uint8_t)((w*map[i][0]+(1.0-w)*map[i+1][0]) * 255.0);
                                uint8_t g = (uint8_t)((w*map[i][1]+(1.0-w)*map[i+1][1]) * 255.0);
                                uint8_t b = (uint8_t)((w*map[i][2]+(1.0-w)*map[i+1][2]) * 255.0);

                                Vec3d world_coord = resolution * Vec3d(x, y, z) + trans;

                                if (world_coord[0] >= world_min_pt[0] && world_coord[0] <= world_max_pt[0] &&
                                        world_coord[1] >= world_min_pt[1] && world_coord[1] <= world_max_pt[1] &&
                                        world_coord[2] >= world_min_pt[2] && world_coord[2] <= world_max_pt[2])
                                {
                                    tm.m_points.push_back(world_coord);
                                    tm.m_colors.push_back(Vec3d((float)r/255.0, (float)g/255.0, (float)b/255.0));
                                    tm.m_conf.push_back(voxel_conf);
                                    Vec3i nearest_pts = closest_tri_point_idx(x, y, z);
                                    ofs << x << " " << y << " " << z << " "
                                                  << std::right << std::setw(10)  << world_coord[0] << " "
                                                  << std::right << std::setw(10)  << world_coord[1] << " "
                                                  << std::right << std::setw(10)  << world_coord[2] << " "
                                                  << std::right << std::setw(10)  << d << " "
                                                  << std::right << std::setw(10) << (scaled_val - 0.5) * rampsize * resolution << " "
                                                  << std::right << std::setw(10) << voxel_conf/65535.0 << " | "
                                                  << std::right << std::setw(10) << closest_tri_idx(x, y, z ) <<  " | "
                                                  << std::right << std::setw(10) << nearest_pts[0] << " "
                                                  << std::right << std::setw(10) << nearest_pts[1] << " "
                                                  << std::right << std::setw(10) << nearest_pts[2] << " "
                                                  << std::endl ;
                                }  // if world_coord
                            }  // if conf
                    }  // for x
            }  // for y
    }  // for z
    tm.savePly(fname);
}

void VoxelGrid::SetClosestTrimeshColor(
        const Vec3d &world_min_pt,
        const Vec3d &world_max_pt,
        const double resolution, const Vec3d& trans, const int rampsize,
        const TriMesh& depthmap_tm,
        TriMesh* closesttri_tm)
{
    for (int z=0; z < max[2]; z++) {
            for (int y=0; y < max[1]; y++) {
                    for (int x=0; x < max[0]; x++) {
                            unsigned short d = dist(x, y, z);
                            unsigned short voxel_conf  = conf(x, y, z);
                            if (voxel_conf > 0)
                            {
                                float scaled_val = d / 65535.0;
                                Vec3d world_coord = resolution * Vec3d(x, y, z) + trans;
                                if (world_coord[0] >= world_min_pt[0] && world_coord[0] <= world_max_pt[0] &&
                                        world_coord[1] >= world_min_pt[1] && world_coord[1] <= world_max_pt[1] &&
                                        world_coord[2] >= world_min_pt[2] && world_coord[2] <= world_max_pt[2])
                                {
                                    int closest_tri = closest_tri_idx(x, y, z);
                                    cout << closest_tri << endl;
                                    // Vec3i closest_tri_points = closest_tri_point_idx(x, y, z);
                                    Vec3i closest_tri_points(depthmap_tm.m_tris[3 * closest_tri + 0], depthmap_tm.m_tris[3 * closest_tri + 1], depthmap_tm.m_tris[3 * closest_tri + 2]);
                                    closesttri_tm->m_points.push_back(depthmap_tm.m_points[closest_tri_points[0]]);
                                    closesttri_tm->m_points.push_back(depthmap_tm.m_points[closest_tri_points[1]]);
                                    closesttri_tm->m_points.push_back(depthmap_tm.m_points[closest_tri_points[2]]);

                                    closesttri_tm->m_colors.push_back(Vec3d(1.0, 0.0, 0.0));
                                    closesttri_tm->m_colors.push_back(Vec3d(1.0, 0.0, 0.0));
                                    closesttri_tm->m_colors.push_back(Vec3d(1.0, 0.0, 0.0));

                                    closesttri_tm->m_tris.push_back(closesttri_tm->m_points.size() - 3);
                                    closesttri_tm->m_tris.push_back(closesttri_tm->m_points.size() - 2);
                                    closesttri_tm->m_tris.push_back(closesttri_tm->m_points.size() - 1);
                                }  // if world_coord
                            }  // if conf
                    }  // for x
            }  // for y
    }  // for z
    closesttri_tm->scale(resolution, resolution, resolution);
    closesttri_tm->translate(trans[0], trans[1], trans[2]);
}
