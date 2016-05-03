#include "tsdf_smooth.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>
//#include <pcl/point_types.h>
#include <Eigen/Eigen>

#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_representation/voxel_hashmap.h"
#include "tsdf_representation/voxel_data.h"

namespace cpu_tsdf
{
void SmoothTSDFFunction(const TSDFHashing* tsdf_volume, int neighborhood, TSDFHashing* new_tsdf_volume)
{
    fprintf(stderr, "begin smoothing..\n");
    new_tsdf_volume->Clear();
    float max_pos_dist, max_neg_dist;
    tsdf_volume->getDepthTruncationLimits(max_pos_dist, max_neg_dist);
    new_tsdf_volume->Init(tsdf_volume->voxel_length(), tsdf_volume->offset(),
                          max_pos_dist, max_neg_dist);
    int total_size = (2*neighborhood+1);
    total_size = total_size * total_size * total_size;
    std::vector<float> d_list;
    std::vector<float> w_list;
    std::vector<cv::Vec3b> color_list;
    for (TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        citr->RetriveData(&d, &w, &color);
        // get data from neighbor points, if the voxel doesn't exist, corresponding weight is set to 0
        tsdf_volume->GetNeighborPointData(cur_voxel_coord, neighborhood, d_list, w_list, color_list);
        float total_weight = 0;
        float weighted_sum_dist = 0;
        cv::Vec3f total_color;
        for(int i = 0; i < d_list.size(); ++i)
        {
            total_weight += w_list[i];
            weighted_sum_dist += w_list[i] * d_list[i];
            total_color += static_cast<cv::Vec3f>(color_list[i])*w_list[i];
        }
        if(total_weight > 0)
        {
            // take average
            float final_d = weighted_sum_dist/total_weight;
            float final_w = total_weight/total_size;
            cv::Vec3b final_color(total_color/total_weight);
            (new_tsdf_volume)->SetTSDFValue(cur_voxel_coord, final_d, final_w, final_color);
        }
    }  // end for
    fprintf(stderr, "finished\n");
}

void SmoothTSDFFunctionOnlyExtension(const TSDFHashing* tsdf_volume, int neighborhood, TSDFHashing* new_tsdf_volume)
{
    fprintf(stderr, "begin smoothing..\n");
    new_tsdf_volume->Clear();
    float max_pos_dist, max_neg_dist;
    tsdf_volume->getDepthTruncationLimits(max_pos_dist, max_neg_dist);
    new_tsdf_volume->Init(tsdf_volume->voxel_length(), tsdf_volume->offset(),
                          max_pos_dist, max_neg_dist);
    int total_size = (2*neighborhood+1);
    total_size = total_size * total_size * total_size;
    std::vector<float> d_list;
    std::vector<float> w_list;
    std::vector<cv::Vec3b> color_list;
    std::vector<VoxelData::VoxelState> state_list;
    for (TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        VoxelData::VoxelState st;
        citr->RetriveData(&d, &w, &color, &st);
        if (st == VoxelData::FILLED)
        {
            // if the voxel is filled, its TSDF value remains unchanged
            (new_tsdf_volume)->SetTSDFValue(cur_voxel_coord, d, w, color, st);
        }
        else
        {
            tsdf_volume->GetNeighborPointData(cur_voxel_coord, neighborhood, d_list, w_list, color_list, state_list);
            float total_weight = 0;
            float weighted_sum_dist = 0;
            float total_cnt = 0;
            cv::Vec3f total_color;
            for(int i = 0; i < d_list.size(); ++i)
            {
                if (state_list[i] != VoxelData::EMPTY)
                {
                    // compute the sum of data of non-empty voxels
                    total_weight += w_list[i];
                    total_cnt ++;
                    weighted_sum_dist +=  w_list[i] * d_list[i];
                    total_color += static_cast<cv::Vec3f>(color_list[i] * w_list[i]);
                }
            }
            if(total_weight > 0)
            {
                //weighted average
                float final_d = weighted_sum_dist/total_weight;
                float final_w = total_weight/total_cnt;
                cv::Vec3b final_color(total_color/total_weight);
                (new_tsdf_volume)->SetTSDFValue(cur_voxel_coord, final_d, final_w, final_color, VoxelData::MODIFIABLE);
            }
        }  // end else
    }  // end for
    fprintf(stderr, "finished\n");
}

void SmoothTSDFFunctionOnlyExtensionGaussian(const TSDFHashing* tsdf_volume, int neighborhood,
                                             const std::vector<float>& kernel, TSDFHashing* new_tsdf_volume)
{
    fprintf(stderr, "begin smoothing..\n");
    new_tsdf_volume->Clear();
    float max_pos_dist, max_neg_dist;
    tsdf_volume->getDepthTruncationLimits(max_pos_dist, max_neg_dist);
    new_tsdf_volume->Init(tsdf_volume->voxel_length(), tsdf_volume->offset(),
                          max_pos_dist, max_neg_dist);
    int total_size = (2*neighborhood+1);
    total_size = total_size * total_size * total_size;
    std::vector<float> d_list;
    std::vector<float> w_list;
    std::vector<cv::Vec3b> color_list;
    std::vector<VoxelData::VoxelState> state_list;
    float max_kernel_sum = std::accumulate(kernel.begin(), kernel.end(), 0.f);
    for (TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        VoxelData::VoxelState st;
        citr->RetriveData(&d, &w, &color, &st);
        if (st == VoxelData::FILLED)
        {
            (new_tsdf_volume)->SetTSDFValue(cur_voxel_coord, d, w, color, st);
        }
        else
        {
            tsdf_volume->GetNeighborPointData(cur_voxel_coord, neighborhood, d_list, w_list, color_list, state_list);
            float total_weight = 0;
            float weighted_sum_dist = 0;
            cv::Vec3f total_color;
            for(int i = 0; i < d_list.size(); ++i)
            {
                if (state_list[i] != VoxelData::EMPTY)
                {
                    total_weight += w_list[i] * kernel[i];
                    weighted_sum_dist += w_list[i] * kernel[i] * d_list[i];
                    total_color += static_cast<cv::Vec3f>(color_list[i])*w_list[i]*kernel[i];
                }
            }
            if(total_weight > 0)
            {
                float final_d = weighted_sum_dist/total_weight;
                float final_w = total_weight/max_kernel_sum;
                cv::Vec3b final_color(total_color/total_weight);
                (new_tsdf_volume)->SetTSDFValue(cur_voxel_coord, final_d, final_w, final_color, VoxelData::MODIFIABLE);
            }
        }
    }  // end for
    fprintf(stderr, "finished\n");
}


void SmoothTSDFFunctionOnlyExtensionGaussian(const TSDFHashing* tsdf_volume, int neighborhood,
                                             double sigma, TSDFHashing* new_tsdf_volume)
{
    // create 3D gaussian kernel with parameter sigma (cov_matrix: sigma * I(3), mean: 0)
    int total_size0 = (2*neighborhood+1);
    int total_size1 = total_size0 * total_size0 * total_size0;
    std::vector<float> g_kernel(total_size1, 0);
    cv::Vec3i center(neighborhood, neighborhood, neighborhood);
    double total_weight_sum = 0;
    int curidx = 0;
    for (int ix = 0; ix < total_size0; ++ix, ++curidx)
        for (int iy = 0; iy < total_size0; ++iy, ++curidx)
            for (int iz = 0; iz < total_size0; ++iz, ++curidx)
            {
                cv::Vec3f distv(ix - center[0], iy - center[0], iz - center[0]);
                double dist = cv::norm(distv);
                g_kernel[curidx] = std::exp(-sigma*dist);
                total_weight_sum += g_kernel[curidx];
            }
    for (int i = 0; i < g_kernel.size(); ++i)
    {
        g_kernel[i] /= total_weight_sum;
    }
    // display the kernel
    curidx = 0;
    for (int ix = 0; ix < total_size0; ++ix, ++curidx)
    {
        fprintf(stderr, "x: %d ", ix);
        for (int iy = 0; iy < total_size0; ++iy, ++curidx)
        {
            for (int iz = 0; iz < total_size0; ++iz, ++curidx)
            {
                fprintf(stderr, "%f ", g_kernel[curidx]);
            }
            fprintf(stderr, "\n");
        }
    }
    SmoothTSDFFunctionOnlyExtensionGaussian(tsdf_volume, neighborhood, g_kernel, new_tsdf_volume);
}

}
