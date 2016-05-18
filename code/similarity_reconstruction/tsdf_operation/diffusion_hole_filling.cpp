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

