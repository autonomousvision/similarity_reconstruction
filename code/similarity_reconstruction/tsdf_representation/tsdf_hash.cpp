/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "tsdf_hash.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_map>
#include <Eigen/Eigen>
#include <pcl/console/time.h>
#include <pcl/common/io.h>
#include <pcl/io/ply_io.h>
#include <opencv2/opencv.hpp>

#include "common/fisheye_camera/RectifiedCameraPair.h"
#include "voxel_hashmap.h"
#include "voxel_data.h"

using std::vector;
using std::string;
using std::endl;

template <typename T> 
inline int possgn(T val) 
{
    return (T(0) <= val) - (val < T(0));
}

inline Eigen::Vector3f Clamp3DPoint(const Eigen::Vector3f& voxel_coord)
{
    Eigen::Vector3i floored_voxel(floor(voxel_coord(0)), floor(voxel_coord(1)), floor(voxel_coord(2)));
    Eigen::Vector3f clamped_voxel = voxel_coord;
    const float clamp_thresh = 0.9999;
    clamped_voxel[0] = (voxel_coord[0] - floored_voxel[0] > clamp_thresh) ? floored_voxel[0] + 1 : voxel_coord[0];
    clamped_voxel[1] = (voxel_coord[1] - floored_voxel[1] > clamp_thresh) ? floored_voxel[1] + 1 : voxel_coord[1];
    clamped_voxel[2] = (voxel_coord[2] - floored_voxel[2] > clamp_thresh) ? floored_voxel[2] + 1 : voxel_coord[2];
    return clamped_voxel;
}

namespace cpu_tsdf {

inline cv::Vec3i RoundVec3d(const cv::Vec3d& v)
{
    return cv::Vec3i(cvRound(v[0]), cvRound(v[1]), cvRound(v[2]));
}

void TSDFHashing::CopyHashParametersFrom(const TSDFHashing &tsdf)
{
    this->Init(tsdf.voxel_length_, tsdf.offset_, tsdf.max_dist_pos_, tsdf.max_dist_neg_);
}

void TSDFHashing::Init(float voxel_length, const Eigen::Vector3f& offset, float max_dist_pos, float max_dist_neg)
{
    offset_ = offset;
    voxel_length_ = voxel_length;
    max_dist_pos_ = max_dist_pos;
    max_dist_neg_ = max_dist_neg;
    // dist_neg_inflection_point_ = max_dist_neg;
    neighbor_adding_limit_ = ceil(std::max(fabs(max_dist_pos_/voxel_length_/(float)VoxelHashMap::kBrickSideLength), 
            fabs(max_dist_neg_/voxel_length_/(float)VoxelHashMap::kBrickSideLength)));
    voxel_hash_map_.Clear();
    std::cout << "voxel_length: " << voxel_length_ << std::endl;
    std::cout << "max_dist_pos_: " << max_dist_pos_ << std::endl;
    std::cout << "max_dist_neg_: " << max_dist_neg_ << std::endl;
    std::cout << "neighbor_adding_limit_: " << neighbor_adding_limit_ << std::endl;
}

bool TSDFHashing::integrateCloud_Spherical_Queue (const cv::Mat& depth,
                                                  const cv::Mat& confidence,
                                                  const cv::Mat& image,
                                                  const cv::Mat& semantic_label_mat,
                                                  const RectifiedCameraPair& cam_info,
                                                  float neg_dist_full_weight_delta,
                                                  float neg_weight_thresh,
                                                  float neg_weight_dist_thresh)
{
    fprintf(stderr, "enqueue modified voxels\n");
    update_hashset_type update_hashset;
    for(int y = 0; y < depth.rows; y++)
    {
        for(int x = 0; x < depth.cols; x++)
        {
            unsigned short quant_depth = depth.at<unsigned short>(y, x);
            if(quant_depth == 0) continue;
            EnqueueModifiedBricks(x, y, cam_info, quant_depth, update_hashset);
        }
    }
    fprintf(stderr, "queue size: %lu\n", update_hashset.size());
    ////////////////////////////////////////////////////////////
    voxel_hash_map_.DisplayHashMapInfo();
    ///////////////////////////////////////////////////////////
    fprintf(stderr, "update modified bricks\n");
    for(update_hashset_type::iterator itr = update_hashset.begin(); itr != update_hashset.end(); ++itr)
    {
        UpdateBrick(*itr, cam_info,
                    depth, confidence, image, semantic_label_mat,
                    neg_dist_full_weight_delta,
                    neg_weight_thresh,
                    neg_weight_dist_thresh);
    }
    fprintf(stderr, "finished updating\n");
    voxel_hash_map_.DisplayHashMapInfo();
    return true;
}

bool TSDFHashing::integrateCloud_Spherical_Queue_DoubleImCoord (const cv::Mat& depth,
                                                  const cv::Mat& confidence,
                                                  const cv::Mat& image,
                                                  const cv::Mat& semantic_label_mat,
                                                  const RectifiedCameraPair& cam_info,
                                                  const double stepsize, float neg_dist_full_weight_delta, float neg_weight_thresh, float neg_weight_dist_thresh)
{
    fprintf(stderr, "enqueue modified voxels\n");
    update_hashset_type update_hashset;
    for(double y = 0; y < depth.rows; y+=stepsize)
    {
        for(double x = 0; x < depth.cols; x+=stepsize)
        {
            unsigned short quant_depth = depth.at<unsigned short>((int)y, (int)x);
            if(quant_depth == 0) continue;
            EnqueueModifiedBricksDoubleImCoord(x, y, cam_info, quant_depth, update_hashset);
        }
    }
    fprintf(stderr, "queue size: %lu\n", update_hashset.size());
    ////////////////////////////////////////////////////////////
    voxel_hash_map_.DisplayHashMapInfo();
    ///////////////////////////////////////////////////////////
    fprintf(stderr, "update modified bricks\n");
    for(update_hashset_type::iterator itr = update_hashset.begin(); itr != update_hashset.end(); ++itr)
    {
        UpdateBrick(*itr, cam_info,
                    depth, confidence, image, semantic_label_mat,
                    neg_dist_full_weight_delta,
                    neg_weight_thresh,
                    neg_weight_dist_thresh);
    }
    fprintf(stderr, "finished updating\n");
    voxel_hash_map_.DisplayHashMapInfo();
    return true;
}

float TSDFHashing::ComputeTSDFWeight(float diff_observed_dist_cur_dist,
                                     float neg_dist_full_weight_delta,
                                     float neg_weight_thresh,
                                     float neg_weight_dist_thresh)
{
    const float neg_dist_full_weight_threshold = neg_dist_full_weight_delta;
    const float neg_weight_thresh1 = neg_weight_thresh;

    if (diff_observed_dist_cur_dist >= neg_dist_full_weight_threshold) return 1.0;
    else if (diff_observed_dist_cur_dist >= neg_weight_dist_thresh) return neg_weight_thresh1 + (1.0 - neg_weight_thresh1) * ((diff_observed_dist_cur_dist - neg_weight_dist_thresh) / (neg_dist_full_weight_threshold - neg_weight_dist_thresh));
    else if (diff_observed_dist_cur_dist > max_dist_neg_ ) return neg_weight_thresh1 * ((diff_observed_dist_cur_dist - max_dist_neg_) / (neg_weight_dist_thresh - max_dist_neg_));
    else return 0.0;
}

void TSDFHashing::EnqueueModifiedBricks(int imx, int imy, const RectifiedCameraPair& cam_info,
                                        unsigned short quant_depth,
                                        update_hashset_type& update_hashset)
{
    assert(quant_depth > 0);
    cv::Vec3d voxel_point_d = cam_info.RectifiedImagePointToVoxel3DPoint(imx, imy, quant_depth);
    cv::Vec3i voxel_point = RoundVec3d(voxel_point_d);
    int delta = neighbor_adding_limit_ * VoxelHashMap::kBrickSideLength;
    for (int ix = voxel_point[0] - delta; ix <= voxel_point[0] + delta; ix += VoxelHashMap::kBrickSideLength)
        for (int iy = voxel_point[1] - delta; iy <= voxel_point[1] + delta; iy += VoxelHashMap::kBrickSideLength)
            for (int iz = voxel_point[2] - delta; iz <= voxel_point[2] + delta; iz += VoxelHashMap::kBrickSideLength)
            {
                update_hashset.insert(VoxelHashMap::BrickPosition(ix, iy, iz));
            }
    return;
}

void TSDFHashing::EnqueueModifiedBricksDoubleImCoord(double imx, double imy, const RectifiedCameraPair& cam_info,
                                        unsigned short quant_depth,
                                        update_hashset_type& update_hashset)
{
    assert(quant_depth > 0);
    cv::Vec3d voxel_point_d = cam_info.RectifiedImagePointToVoxel3DPointDouble(imx, imy, quant_depth);
    cv::Vec3i voxel_point = RoundVec3d(voxel_point_d);
    int delta = neighbor_adding_limit_ * VoxelHashMap::kBrickSideLength;
    for (int ix = voxel_point[0] - delta; ix <= voxel_point[0] + delta; ix += VoxelHashMap::kBrickSideLength)
        for (int iy = voxel_point[1] - delta; iy <= voxel_point[1] + delta; iy += VoxelHashMap::kBrickSideLength)
            for (int iz = voxel_point[2] - delta; iz <= voxel_point[2] + delta; iz += VoxelHashMap::kBrickSideLength)
            {
                update_hashset.insert(VoxelHashMap::BrickPosition(ix, iy, iz));
            }
    return;
}

bool TSDFHashing::UpdateBrick(const VoxelHashMap::BrickPosition& bpos, const RectifiedCameraPair& cam_info,
                              const cv::Mat& depth, const cv::Mat& confidence, const cv::Mat& image, const cv::Mat& semantic_label,
                              float neg_dist_full_weight_delta,
                              float neg_weight_thresh,
                              float neg_weight_dist_thresh)
{
    VoxelHashMap::BrickData& bdata = voxel_hash_map_.RetriveBrickDataWithAllocation(bpos);
    const int sidelength = VoxelHashMap::kBrickSideLength;
    cv::Vec3i base_voxel_pos(bpos[0], bpos[1], bpos[2]);
    for (int ix = 0; ix < sidelength; ++ix)
        for (int iy = 0; iy < sidelength; ++iy)
            for (int iz = 0; iz < sidelength; ++iz)
            {
                float length;
                int cur_imx, cur_imy;
                cv::Vec3i offset(ix, iy, iz);
                cv::Vec3i cur_voxel_pos = base_voxel_pos + offset;
                if(!cam_info.Voxel3DPointToImageCoord(static_cast<cv::Vec3d>(cur_voxel_pos), &cur_imx, &cur_imy, &length))
                {
                    continue;
                }
                float observed_length = depth.at<ushort>(cur_imy, cur_imx) * cam_info.DepthImageScaling();
                float w_inc = confidence.at<ushort>(cur_imy, cur_imx) / 65535.0;
                if(observed_length == 0.0 || w_inc == 0.0) continue;
                float d_inc = observed_length - length;

                float w_dist = this->ComputeTSDFWeight(d_inc,
                                                       neg_dist_full_weight_delta,
                                                       neg_weight_thresh,
                                                       neg_weight_dist_thresh);
                if (w_dist == 0.0 || d_inc > max_dist_pos_ || d_inc < max_dist_neg_) continue;
                //std::cout << "w_dist: " << w_dist << " d_inc:" << d_inc << " " << max_dist_neg_ / 20.0 << std::endl;
                w_inc = w_dist * w_inc;
                cv::Vec3b cur_color = image.at<cv::Vec3b>(cur_imy, cur_imx);
                int semantic_labelv = -1;
                if (!semantic_label.empty()) {
                    semantic_labelv = semantic_label.at<ushort>(cur_imy, cur_imx);
                  }
                cv::Vec3b rgb_color(cur_color[2], cur_color[1], cur_color[0]);
                voxel_hash_map_.AddObservation(bdata, offset, d_inc, w_inc, rgb_color, semantic_labelv,
                        max_dist_pos_,
                        max_dist_neg_);
            }
    return true;
}

bool TSDFHashing::AddBrickUpdateList(const cv::Vec3i& voxel_point, 
        update_hashset_type* brick_update_hashset) const
{
    //const int voxel_neighbor_limit = neighbor_adding_limit_ * VoxelHashMap::kBrickSideLength;
    const int delta = neighbor_adding_limit_ * VoxelHashMap::kBrickSideLength;
    for (int ix = voxel_point[0] - delta; ix <= voxel_point[0] + delta; ix += VoxelHashMap::kBrickSideLength)
        for (int iy = voxel_point[1] - delta; iy <= voxel_point[1] + delta; iy += VoxelHashMap::kBrickSideLength)
            for (int iz = voxel_point[2] - delta; iz <= voxel_point[2] + delta; iz += VoxelHashMap::kBrickSideLength)
            {
                brick_update_hashset->insert(VoxelHashMap::BrickPosition(ix, iy, iz));
            }
    return true;
}

bool TSDFHashing::RetriveDataFromWorldCoord(const Eigen::Vector3f& world_coord, float* d, float* w /*=NULL*/, cv::Vec3b* pcolor /*=NULL*/) const
{
    if (*w) *w = 0;
    Eigen::Vector3f voxel_coord = Clamp3DPoint(World2Voxel(world_coord));
    Eigen::Vector3i floored_voxel(floor(voxel_coord(0)), floor(voxel_coord(1)), floor(voxel_coord(2)));
    float final_d = 0;
    float total_linear_w = 0;
    float total_point_w = 0;
    Eigen::Vector3f total_color(0, 0, 0);
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
            {
                Eigen::Vector3i cur_voxel(floored_voxel(0) + ix, floored_voxel(1) + iy, floored_voxel(2) + iz);
                float cur_d = -1;
                float cur_w = 0;
                cv::Vec3b color;
                if (!this->RetriveData(cur_voxel, &cur_d, &cur_w, &color) || cur_w == 0)
                {
                    //return false;
                    continue;
                }
                assert(cur_w > 0);
                float linear_w = (1.0 - fabs(voxel_coord(0) - cur_voxel(0))) *
                    (1.0 - fabs(voxel_coord(1) - cur_voxel(1))) *
                    (1.0 - fabs(voxel_coord(2) - cur_voxel(2)));
                total_linear_w += linear_w;
                final_d += linear_w * cur_d;
                total_color += linear_w * Eigen::Vector3f(color[0], color[1], color[2]);
                total_point_w += linear_w * cur_w;
            }
    // const float total_thresh = 0.5;
    const float total_thresh = 0.01;
    if (total_linear_w > total_thresh)
    {
        *d = final_d / total_linear_w;
        if (w)
            *w = total_point_w / total_linear_w;
        if (pcolor)
        {
            *pcolor = utility::EigenVectorToCvVector3((total_color / total_linear_w).eval());
        }
    }
    return total_linear_w > total_thresh;
}

bool TSDFHashing::RetriveDataFromWorldCoord_NearestNeighbor(const Eigen::Vector3f& world_coord, float* d, float* w /*=NULL*/, cv::Vec3b* pcolor /*=NULL*/) const
{
    Eigen::Vector3f voxel_coord = Clamp3DPoint(World2Voxel(world_coord));
    Eigen::Vector3i rounded_voxel(round(voxel_coord(0)), round(voxel_coord(1)), round(voxel_coord(2)));
    float cur_d = 0;
    float cur_w = 0;
    cv::Vec3b color;
    if (!this->RetriveData(rounded_voxel, &cur_d, &cur_w, &color) || cur_w == 0)
    {
        return false;
    }
    else
    {
        *d = cur_d;
        if (w) *w = cur_w;
        if(pcolor) *pcolor = color;
        return true;
    }
}

bool TSDFHashing::RetriveGradientFromWorldCoord(const Eigen::Vector3f& world_coord, Eigen::Vector3f* grad, Eigen::Vector3f* wgrad) const
{
    Eigen::Vector3f voxel_coord = Clamp3DPoint(World2Voxel(world_coord));
    Eigen::Vector3i floored_voxel(floor(voxel_coord(0)), floor(voxel_coord(1)), floor(voxel_coord(2)));
    float weight[3][2] = {0};  // [x, y, z][+, -]
    float total_grad[3][2] = {0};  // [x, y, z][+, -]
    float total_weight[3][2] = {0};
    float cur_weight[3] = {0};
    float darr[8] = {0};
    bool flag = false;
    (*grad)(0) = (*grad)(1) = (*grad)(2) = 0.0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
            {
                Eigen::Vector3i cur_voxel(floored_voxel(0) + ix, floored_voxel(1) + iy, floored_voxel(2) + iz);
                float cur_d = -1;
                float cur_w = 0;
                cv::Vec3b color;
                if (!this->RetriveData(cur_voxel, &cur_d, &cur_w, &color) || cur_w == 0)
                {
                    //continue;
                    return false;
                }
                flag = true;
                assert(cur_w > 0);
                cur_weight[0] = (1.0 - fabs(voxel_coord(1) - cur_voxel(1))) * 
                    (1.0 - fabs(voxel_coord(2) - cur_voxel(2)));
                cur_weight[1] = (1.0 - fabs(voxel_coord(0) - cur_voxel(0))) * 
                    (1.0 - fabs(voxel_coord(2) - cur_voxel(2)));
                cur_weight[2] = (1.0 - fabs(voxel_coord(0) - cur_voxel(0))) * 
                    (1.0 - fabs(voxel_coord(1) - cur_voxel(1)));
                // add positive/negative weight
                (-possgn((voxel_coord(0) - cur_voxel(0))) > 0) ? 
                    (weight[0][0] += cur_weight[0], total_grad[0][0] += cur_weight[0] * cur_d, total_weight[0][0] += cur_weight[0] * cur_w) :
                    (weight[0][1] += cur_weight[0], total_grad[0][1] += cur_weight[0] * cur_d, total_weight[0][1] += cur_weight[0] * cur_w);
                (-possgn((voxel_coord(1) - cur_voxel(1))) > 0 ) ? 
                    (weight[1][0] += cur_weight[1], total_grad[1][0] += cur_weight[1] * cur_d, total_weight[1][0] += cur_weight[1] * cur_w) :
                    (weight[1][1] += cur_weight[1], total_grad[1][1] += cur_weight[1] * cur_d, total_weight[1][1] += cur_weight[1] * cur_w);
                (-possgn((voxel_coord(2) - cur_voxel(2))) > 0) ? 
                    (weight[2][0] += cur_weight[2], total_grad[2][0] += cur_weight[2] * cur_d, total_weight[2][0] += cur_weight[2] * cur_w) :
                    (weight[2][1] += cur_weight[2], total_grad[2][1] += cur_weight[2] * cur_d, total_weight[2][1] += cur_weight[2] * cur_w);
                darr[iz + iy * 2 + ix * 4] = cur_d;
            }
    if (!flag) return false;  // empty cube.
    (*grad)(0) = (weight[0][0] > 0 && weight[0][1] > 0) ? 
        total_grad[0][0]/weight[0][0] - total_grad[0][1]/weight[0][1] : 0.0;
    (*grad)(1) = (weight[1][0] > 0 && weight[1][1] > 0) ?
        total_grad[1][0]/weight[1][0] - total_grad[1][1]/weight[1][1] : 0.0;
    (*grad)(2) = (weight[2][0] > 0 && weight[2][1] > 0) ?
        total_grad[2][0]/weight[2][0] - total_grad[2][1]/weight[2][1] : 0.0;
    (*grad) = (*grad) / voxel_length_;

    if (wgrad)
    {
        (*wgrad)(0) = (weight[0][0] > 0 && weight[0][1] > 0) ?
                    total_weight[0][0]/weight[0][0] - total_weight[0][1]/weight[0][1] : 0.0;
        (*wgrad)(1) = (weight[1][0] > 0 && weight[1][1] > 0) ?
                    total_weight[1][0]/weight[1][0] - total_weight[1][1]/weight[1][1] : 0.0;
        (*wgrad)(2) = (weight[2][0] > 0 && weight[2][1] > 0) ?
                    total_weight[2][0]/weight[2][0] - total_weight[2][1]/weight[2][1] : 0.0;
        (*wgrad) = (*wgrad) / voxel_length_;
    }
    return true;
}

bool TSDFHashing::RetriveAbsGradientFromWorldCoord(const Eigen::Vector3f& world_coord, Eigen::Vector3f* grad) const
{
    Eigen::Vector3f voxel_coord = Clamp3DPoint(World2Voxel(world_coord));
    Eigen::Vector3i floored_voxel(floor(voxel_coord(0)), floor(voxel_coord(1)), floor(voxel_coord(2)));
    float weight[3][2] = {0};  // [x, y, z][+, -]
    float total_grad[3][2] = {0};  // [x, y, z][+, -]
    float cur_weight[3] = {0};
    float darr[8] = {0};
    bool flag = false;
    (*grad)(0) = (*grad)(1) = (*grad)(2) = 0.0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
            {
                Eigen::Vector3i cur_voxel(floored_voxel(0) + ix, floored_voxel(1) + iy, floored_voxel(2) + iz);
                float cur_d = -1;
                float cur_w = 0;
                cv::Vec3b color;
                if (!this->RetriveData(cur_voxel, &cur_d, &cur_w, &color) || cur_w == 0)
                {
                    continue;
                }
                flag = true;
                assert(cur_w > 0);
                cur_weight[0] = (1.0 - fabs(voxel_coord(1) - cur_voxel(1))) * 
                    (1.0 - fabs(voxel_coord(2) - cur_voxel(2)));
                cur_weight[1] = (1.0 - fabs(voxel_coord(0) - cur_voxel(0))) * 
                    (1.0 - fabs(voxel_coord(2) - cur_voxel(2)));
                cur_weight[2] = (1.0 - fabs(voxel_coord(0) - cur_voxel(0))) * 
                    (1.0 - fabs(voxel_coord(1) - cur_voxel(1)));
                // add positive/negative weight
                (-possgn((voxel_coord(0) - cur_voxel(0))) > 0) ? 
                    (weight[0][0] += cur_weight[0], total_grad[0][0] += cur_weight[0] * fabs(cur_d)) : 
                    (weight[0][1] += cur_weight[0], total_grad[0][1] += cur_weight[0] * fabs(cur_d));
                (-possgn((voxel_coord(1) - cur_voxel(1))) > 0 ) ? 
                    (weight[1][0] += cur_weight[1], total_grad[1][0] += cur_weight[1] * fabs(cur_d)) : 
                    (weight[1][1] += cur_weight[1], total_grad[1][1] += cur_weight[1] * fabs(cur_d));
                (-possgn((voxel_coord(2) - cur_voxel(2))) > 0) ? 
                    (weight[2][0] += cur_weight[2], total_grad[2][0] += cur_weight[2] * fabs(cur_d)) : 
                    (weight[2][1] += cur_weight[2], total_grad[2][1] += cur_weight[2] * fabs(cur_d));
                darr[iz + iy * 2 + ix * 4] = cur_d;
            }
    if (!flag) return false;  // empty cube.
    (*grad)(0) = (weight[0][0] > 0 && weight[0][1] > 0) ? 
        total_grad[0][0]/weight[0][0] - total_grad[0][1]/weight[0][1] : 0.0;
    (*grad)(1) = (weight[1][0] > 0 && weight[1][1] > 0) ?
        total_grad[1][0]/weight[1][0] - total_grad[1][1]/weight[1][1] : 0.0;
    (*grad)(2) = (weight[2][0] > 0 && weight[2][1] > 0) ?
        total_grad[2][0]/weight[2][0] - total_grad[2][1]/weight[2][1] : 0.0;
    (*grad) = (*grad) / voxel_length_;
    return true;
}

bool TSDFHashing::RetriveData(const Eigen::Vector3i& voxel_coord, 
                              float* d, float* w, cv::Vec3b* color) const
{
   return voxel_hash_map_.RetriveData(cv::Vec3d(voxel_coord(0), voxel_coord(1), voxel_coord(2)),
                                      d, w, color);
}

bool TSDFHashing::RetriveData(const Eigen::Vector3i& voxel_coord, 
                              float* d, float* w, cv::Vec3b* color, VoxelData::VoxelState* st) const
{
   return voxel_hash_map_.RetriveData(cv::Vec3d(voxel_coord(0), voxel_coord(1), voxel_coord(2)),
                                      d, w, color, st);
}

void TSDFHashing::GetNeighborPointData(const cv::Vec3i& voxel_coord, int neighborhood, 
        std::vector<float>& d, std::vector<float>& w, std::vector<cv::Vec3b>& colors, std::vector<VoxelData::VoxelState>& states) const
{
    using std::vector;
    int total_size = (2*neighborhood+1);
    total_size = total_size * total_size * total_size;
    d.assign(total_size, -1) ;
    w.assign(total_size, 0);
    states.assign(total_size, VoxelData::EMPTY);
    colors.resize(total_size);
    int cnt = 0;
    for(int ix = voxel_coord[0] - neighborhood; ix <= voxel_coord[0] + neighborhood; ++ix)
        for(int iy = voxel_coord[1] - neighborhood; iy <= voxel_coord[1] + neighborhood; ++iy)
            for(int iz = voxel_coord[2] - neighborhood; iz <= voxel_coord[2] + neighborhood; ++iz)
            {
                float cur_d = -1;
                float cur_w = 0;
                cv::Vec3b color;
                VoxelData::VoxelState st;
                this->RetriveData(Eigen::Vector3i(ix, iy, iz), &cur_d, &cur_w, &color, &st);
                d[cnt] = cur_d;
                w[cnt] = cur_w;
                colors[cnt] = color;
                states[cnt] = st;
                cnt++;
            }
}

void TSDFHashing::GetNeighborPointData(const cv::Vec3i& voxel_coord, int neighborhood, 
        std::vector<float>& d, std::vector<float>& w, std::vector<cv::Vec3b>& colors) const
{
    using std::vector;
    int total_size = (2*neighborhood+1);
    total_size = total_size * total_size * total_size;
    d.assign(total_size, -1) ;
    w.assign(total_size, 0);
    colors.resize(total_size);
    int cnt = 0;
    for(int ix = voxel_coord[0] - neighborhood; ix <= voxel_coord[0] + neighborhood; ++ix)
        for(int iy = voxel_coord[1] - neighborhood; iy <= voxel_coord[1] + neighborhood; ++iy)
            for(int iz = voxel_coord[2] - neighborhood; iz <= voxel_coord[2] + neighborhood; ++iz)
            {
                float cur_d = -1;
                float cur_w = 0;
                cv::Vec3b color;
                this->RetriveData(Eigen::Vector3i(ix, iy, iz), &cur_d, &cur_w, &color);
                d[cnt] = cur_d;
                w[cnt] = cur_w;
                colors[cnt] = color;
                cnt++;
            }
}

void TSDFHashing::OutputTSDFGrid(const std::string& filepath, const Eigen::Vector3f* bb_min_pt_world, const Eigen::Vector3f* bb_max_pt_world,
                                 const RectifiedCameraPair* caminfo, const cv::Mat* depthmap, const float* depth_scale) const
{
    Eigen::Vector3f min_pt_world;
    Eigen::Vector3f max_pt_world;
    this->getBoundingBoxInWorldCoord(min_pt_world, max_pt_world);
    if (bb_min_pt_world)
    {
        min_pt_world = *bb_min_pt_world;
    }
    if (bb_max_pt_world)
    {
        max_pt_world = *bb_max_pt_world;
    }
    const float min_val = max_dist_neg_;
    const float max_val = max_dist_pos_;
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

    pcl::PointCloud<pcl::PointXYZRGB> point_vis;
    Eigen::Vector3f origin = getVoxelOriginInWorldCoord();

    std::ofstream ofs(filepath + "_tsdf_info.txt");
    for(const_iterator citr = begin(); citr != end(); ++citr)
    {
        cv::Vec3i pos = citr.VoxelCoord();
        float dist;
        float weight;
        cv::Vec3b color;
        Eigen::Vector4f pos_neg_mean_weights;
        citr->RetriveDataDebug(&dist, &weight, &color, &pos_neg_mean_weights);
        if(weight == 0.0f) continue;
        assert(dist >= min_val);
        assert(dist <= max_val);
        float scaled_val = std::min<float>(std::max<float>(float(dist - min_val)/(max_val - min_val), 0.0f),1.0f);
        // find bin
        int32_t i;
        for (i=0; i<7; i++)
            if (scaled_val<cumsum[i+1])
                break;

        // compute red/green/blue values
        float   w = 1.0-(scaled_val-cumsum[i])*weights[i];
        uint8_t r = (uint8_t)((w*map[i][0]+(1.0-w)*map[i+1][0]) * 255.0);
        uint8_t g = (uint8_t)((w*map[i][1]+(1.0-w)*map[i+1][1]) * 255.0);
        uint8_t b = (uint8_t)((w*map[i][2]+(1.0-w)*map[i+1][2]) * 255.0);

        pcl::PointXYZRGB point;
        point.x = pos[0]*voxel_length_ + origin(0);
        point.y = pos[1]*voxel_length_ + origin(1);
        point.z = pos[2]*voxel_length_ + origin(2);
        point.r = r;
        point.g = g;
        point.b = b;
       // if((fabs(point.x - (-9.60))>0.16 || fabs(point.y - (-7.20))>0.16|| fabs(point.z - 1.35)>0.16)) continue;
        if (point.x >= min_pt_world[0] && point.x <= max_pt_world[0] &&
                point.y >= min_pt_world[1] && point.y <= max_pt_world[1] &&
                point.z >= min_pt_world[2] && point.z <= max_pt_world[2])
        {
            point_vis.push_back(point);
            ofs << pos[0] << " " << pos[1] << " " << pos[2] << " "
                          << std::right << std::setw(10)  << point.x << " "
                          << std::right << std::setw(10)  << point.y << " "
                          << std::right << std::setw(10)  << point.z << " "
                          << std::right << std::setw(10) << dist << " " << std::right << std::setw(10) << weight ;
            ofs << " | " <<  std::right << std::setw(10) << pos_neg_mean_weights[0] << " "
                << std::right << std::setw(10) << pos_neg_mean_weights[1] << " "
                << std::right << std::setw(10) << pos_neg_mean_weights[2] << " "
                << std::right << std::setw(10) << pos_neg_mean_weights[3] << " | mean_pos/w, mean_neg/w" << endl;
            int imx = -1;
            int imy = -1;
            float voxel_length = -1;
            float observe_length = -1;
            if (
            caminfo->Voxel3DPointToImageCoord(cv::Vec3d(pos), &imx, &imy, &voxel_length)
                    )
            {
                observe_length= float((*depthmap).at<unsigned short>(imy, imx)) * (*depth_scale);
            }
            ofs << imx << " " << imy << " " << voxel_length << " " << observe_length << " | imx/y, vlen, depthlen" << endl;
        }
    }
    pcl::io::savePLYFile (filepath, point_vis);
}

void TSDFHashing::OutputTSDFGrid(
        const std::string& filepath, const Eigen::Vector3f* bb_min_pt_world, const Eigen::Vector3f* bb_max_pt_world) const
{
    using namespace std;
    Eigen::Vector3f min_pt_world;
    Eigen::Vector3f max_pt_world;
    this->getBoundingBoxInWorldCoord(min_pt_world, max_pt_world);
    if (bb_min_pt_world)
    {
        min_pt_world = *bb_min_pt_world;
    }
    if (bb_max_pt_world)
    {
        max_pt_world = *bb_max_pt_world;
    }
    const float min_val = max_dist_neg_;
    const float max_val = max_dist_pos_;
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

    pcl::PointCloud<pcl::PointXYZRGB> point_vis;
    Eigen::Vector3f origin = getVoxelOriginInWorldCoord();

    std::ofstream ofs(filepath + "_tsdf_info.txt");
    for(const_iterator citr = begin(); citr != end(); ++citr)
    {
        cv::Vec3i pos = citr.VoxelCoord();
        float dist;
        float weight;
        cv::Vec3b color;
        Eigen::Vector4f pos_neg_mean_weights;
        citr->RetriveDataDebug(&dist, &weight, &color, &pos_neg_mean_weights);

        if(weight == 0.0f) continue;
        assert(dist >= min_val);
        assert(dist <= max_val);
        float scaled_val = std::min<float>(std::max<float>(float(dist - min_val)/(max_val - min_val), 0.0f),1.0f);
        // find bin
        int32_t i;
        for (i=0; i<7; i++)
            if (scaled_val<cumsum[i+1])
                break;

        // compute red/green/blue values
        float   w = 1.0-(scaled_val-cumsum[i])*weights[i];
        uint8_t r = (uint8_t)((w*map[i][0]+(1.0-w)*map[i+1][0]) * 255.0);
        uint8_t g = (uint8_t)((w*map[i][1]+(1.0-w)*map[i+1][1]) * 255.0);
        uint8_t b = (uint8_t)((w*map[i][2]+(1.0-w)*map[i+1][2]) * 255.0);

        pcl::PointXYZRGB point;
        point.x = pos[0]*voxel_length_ + origin(0);
        point.y = pos[1]*voxel_length_ + origin(1);
        point.z = pos[2]*voxel_length_ + origin(2);
        point.r = r;
        point.g = g;
        point.b = b;
        if (point.x >= min_pt_world[0] && point.x <= max_pt_world[0] &&
                point.y >= min_pt_world[1] && point.y <= max_pt_world[1] &&
                point.z >= min_pt_world[2] && point.z <= max_pt_world[2])
        {
            point_vis.push_back(point);
            ofs << pos[0] << " " << pos[1] << " " << pos[2] << " "
                          << std::right << std::setw(10)  << point.x << " "
                          << std::right << std::setw(10)  << point.y << " "
                          << std::right << std::setw(10)  << point.z << " "
                          << std::right << std::setw(10) << dist << " " << std::right << std::setw(10) << weight ;
            ofs << " | " <<  std::right << std::setw(10) << pos_neg_mean_weights[0] << " "
                << std::right << std::setw(10) << pos_neg_mean_weights[1] << " "
                << std::right << std::setw(10) << pos_neg_mean_weights[2] << " "
                << std::right << std::setw(10) << pos_neg_mean_weights[3] << " | mean_pos/w, mean_neg/w" << endl;
        }
    }
    pcl::io::savePLYFile (filepath, point_vis);
}

bool ScaleTSDFWeight(TSDFHashing *tsdf, const float scaling_factor)
{
    for (TSDFHashing::iterator itr = tsdf->begin(); itr != tsdf->end(); ++itr)
    {
        float dist;
        float w;
        cv::Vec3b color;
        if (itr->RetriveData(&dist, &w, &color))
        {
            itr->SetWeight(scaling_factor * w);
        }
    }
    return true;

}

bool SaveTSDFPPM(const TSDFHashing *tsdf, const cv::Vec3i min_pt, const cv::Vec3i max_pt, const std::string &outfname)
{
    using namespace std;
    std::ofstream out(outfname);
    if (!out.good()) {
        cout << "can't open " << outfname << endl;
        return false;
    }

    cv::Vec3i max = max_pt - min_pt;
    float max_pos_dist, max_neg_dist;
    tsdf->getDepthTruncationLimits(max_pos_dist, max_neg_dist);
    float max_dist = max_pos_dist - max_neg_dist;

    out << "P3" << endl << max[0] << " " << (max[1]*max[2]) << endl << 255 << endl;
    int x, y, z;
    for (z=0; z < max[2]; z++) {
        //for (y=max[1] - 1; y >= 0; y++) {
        for (y=0; y < max[1]; y++) {
            //for (x=0; x < max[0]; x++) {
            for (x=max[0] - 1; x >= 0; x--) { // conform with cloudcompare/meshlab
                cv::Vec3i cur_pt = cv::Vec3i(x, y, z) + min_pt;
                float d;
                float w;
                cv::Vec3b color;
                if (!tsdf->RetriveData(cur_pt, &d, &w, &color) || w == 0)
                {
                    out << "128 64 0" << endl;
                }
                else
                {
                    ushort int_dist = round((d - max_neg_dist)/(max_dist) * 65535.0);
                    out << (int_dist >> 8) << " " << (int_dist >> 8) << " " << (int_dist >> 8) << endl;
                }
            }
        }
    }
    out.close();

    using namespace std;
    using namespace cv;
    cv::Mat img = cv::imread(outfname);
    for (int i = 0; i < max[2]; ++i)
    {
        cv::Vec3i cur_pt = cv::Vec3i(0, 0, i) + min_pt;
        cv::Vec3f world_pt = tsdf->Voxel2World(cv::Vec3f(cur_pt));
        char tmpchar[30];
        sprintf(tmpchar, "%.02f", world_pt[2]);
        string text = tmpchar;
        int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = 0.26;
        int thickness = 1;
        int baseline= 0;
        baseline += thickness;

        // center the text
        Point textOrg(0, (max[1]*i) + 6);

        // then put the text itself
        putText(img, text, textOrg, fontFace, fontScale,
                Scalar(255, 255, 128), thickness, 8);
    }
    cv::imwrite(outfname + ".png", img);
    return true;
}

}
