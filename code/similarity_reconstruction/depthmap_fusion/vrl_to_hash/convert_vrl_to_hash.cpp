/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "convert_vrl_to_hash.h"
#include <string>
#include <climits>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "vrl_grid_representation/OccGridRLE.h"
#include "tsdf_representation/tsdf_hash.h"

using std::string;
using namespace std;

bool cpu_tsdf::ConvertRLEGridToHashMap(OccGridRLE& rle_grid, int original_ramp_size, cpu_tsdf::TSDFHashing* hash_grid, bool init_hash)
{
    // load the dimensions of the grid
    const int xdim = rle_grid.xdim;
    const int ydim = rle_grid.ydim;
    const int zdim = rle_grid.zdim;
    const float voxel_length = rle_grid.resolution;
    const float delta = voxel_length/2.0;
    Eigen::Vector3f offset(rle_grid.origin[0], rle_grid.origin[1], rle_grid.origin[2]);
    //Eigen::Vector3f offset(0, 0, 0);
    //offset = offset - Eigen::Vector3f::Ones() * delta ;
    const int ramp_size = original_ramp_size;
    if (init_hash)
    {
        hash_grid->Init(voxel_length, offset, ramp_size*voxel_length, -ramp_size*voxel_length);
    }
    // iterate over the voxels and store the results
    int ix, iy, iz;
    for (iz = 0; iz < rle_grid.zdim; iz++)
    { // sweep in only one direction
        for (iy = 0; iy < rle_grid.ydim; iy++)
        {
            OccElement *line = rle_grid.getScanline(iy, iz);
            for (ix = 0; ix < rle_grid.xdim; ix++) {
                unsigned short value = line[ix].value;
                unsigned short weight = line[ix].totalWeight;
#ifdef OCCELEM_COLOR
                unsigned int color = line[ix].color;
#else
                unsigned int color = 0xffffffff;
#endif
                if (weight > 0)
                {
                    // printf("cur_color: %d\n", color);
                    cv::Vec3b cur_color;
                    cur_color[2] = color & 0xff;
                    cur_color[1] = (color & 0xff00) >> 8;
                    cur_color[0] = (color & 0xff0000) >> 16;
                    // printf("cur_color:rgb: %d %d %d\n", cur_color[0], cur_color[1], cur_color[2]);
                    // float float_dist_world = float((int)value - USHRT_MAX/2) / USHRT_MAX * 2 * float(ramp_size) * voxel_length;
                    float float_dist_world = (((float)value/(float)USHRT_MAX) - 0.5) * 2 * float(ramp_size) * voxel_length;
                    float float_weight = float(weight)/USHRT_MAX;
                    hash_grid->SetTSDFValue(cv::Vec3i(ix, iy, iz), float_dist_world, float_weight, cur_color);  // note -float_dist is set here
//                    if (ix == 105 && iy == 29 && iz == 16)
//                    {
//                        printf("cur conversion debug points: %d %d %d\nvalue: %d\n fvalue: %f\n", ix, iy, iz, value, float_dist_world);
//                    }
                }
                else
                {
                    continue;
                }
            }
        }
    }
    return true;
}

bool cpu_tsdf::ReadFromVRIFile(const std::string& filepath, int original_ramp_size, cpu_tsdf::TSDFHashing* hash_grid, bool init_hash /* = true */)
{
    OccGridRLE rle_grid(1, 1, 1, CHUNK_SIZE);
    if (!rle_grid.read(const_cast<char*>(filepath.c_str()))) return false;
    cout << "reading VRI file finished" << endl;
    return ConvertRLEGridToHashMap(rle_grid, original_ramp_size, hash_grid, init_hash);
}
