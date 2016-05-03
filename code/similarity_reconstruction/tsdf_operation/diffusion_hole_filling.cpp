/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "diffusion_hole_filling.h"
#include <memory>
#include <vector>
#include <cstddef>
#include <opencv2/opencv.hpp>
//#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_smooth.h"

// can't be used to classify hole grid points because the model is not a close surface
//bool IsOnHoleBoundary(cpu_tsdf::TSDFHashing* tsdf_volume, const cv::Vec3i& voxel_coord)
//{
//    float d,w;
//    cv::Vec3b color;
//    if (!tsdf_volume->RetriveData(voxel_coord, &d, &w, &color)) return false;

//    std::vector<float> d_list;
//    std::vector<float> w_list;
//    std::vector<cv::Vec3b> color_list;
//    tsdf_volume->GetNeighborPointData(voxel_coord, 1, d_list, w_list, color_list);
    
//    bool invalid_neighbor = false;
//    bool valid_opposite = false;
//    for (int i = 0; i < d_list.size(); ++i)
//    {
//        if (w_list[i] == 0) invalid_neighbor = true;
//        else if (w_list[i] > 0 && d_list[i] * d <= 0) valid_opposite = true;
//    }
//    return (invalid_neighbor && valid_opposite);
//}

//void ComputeDiffusionSeeds(cpu_tsdf::TSDFHashing* tsdf_volume, int voxel_from_hole_boundary,
//        cpu_tsdf::TSDFHashing* diffusion_seeds_tsdf)
//{
//    fprintf(stderr, "begin computing diffusion seeds\n");
//    diffusion_seeds_tsdf->Clear();
//    diffusion_seeds_tsdf->offset(tsdf_volume->offset());
//    diffusion_seeds_tsdf->voxel_length(tsdf_volume->voxel_length());
//    int hole_cnt = 0;
//    cpu_tsdf::TSDFHashing debug_tsdf(tsdf_volume->offset(), tsdf_volume->voxel_length());
//    for (cpu_tsdf::TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
//    {
//        float cur_d, cur_w;
//        cv::Vec3b cur_color;
//        cv::Vec3i voxel_coord = citr.VoxelCoord();
//        citr->RetriveData(&cur_d, &cur_w, &cur_color);
//        if (IsOnHoleBoundary(tsdf_volume, voxel_coord))
//        {
//            debug_tsdf.SetTSDFValue(voxel_coord, cur_d, cur_w, cur_color);

//            hole_cnt++ ;
//            diffusion_seeds_tsdf->CopyVoxelDataFrom(*tsdf_volume, voxel_coord, voxel_from_hole_boundary);
//            fprintf(stderr, "added %ith hole\n", hole_cnt);
//        }  // end if
//    }
//    debug_tsdf.OutputTSDFGrid("debug_hole_pts.ply");
//    fprintf(stderr, "finished computing diffusion seeds\n");
//}

void cpu_tsdf::DiffusionHoleFilling(TSDFHashing* tsdf_volume,
        int iterations,
        TSDFHashing* new_tsdf_volume)
{
    // Iteratively smoothing
    // each time swapping tsdf_volume/new_tsdf_volume as input/output
    // as the SmoothTSDFFunction*() functions doesn't deal with self assignment.
    fprintf(stderr, "begin diffusion smoothing..\n");
    TSDFHashing* ptr_diffusion_tsdf_volume_cur = tsdf_volume;
    TSDFHashing* ptr_diffusion_tsdf_volume_last = new_tsdf_volume;
    for (int i = 0; i < iterations; ++i)
    {
        fprintf(stderr, "iteration %d..\n", i);
        SmoothTSDFFunctionOnlyExtension(ptr_diffusion_tsdf_volume_cur, 1, ptr_diffusion_tsdf_volume_last);
        fprintf(stderr, "swapping\n");
        std::swap(ptr_diffusion_tsdf_volume_cur, ptr_diffusion_tsdf_volume_last);
    }
    fprintf(stderr, "generating results\n");
    if (ptr_diffusion_tsdf_volume_cur != new_tsdf_volume)
    {
        (*new_tsdf_volume) = *ptr_diffusion_tsdf_volume_cur;
    }
    fprintf(stderr, "finished diffusion smoothing..\n");
    return;
}

