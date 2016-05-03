/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <Eigen/Eigen>
#include <string>
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "tsdf_hash_utilities/utility.h"

namespace tsdf_utility {

// In canonical position
// [-0.5, 0.5]
class TSDFGridInfo
{
public:
    TSDFGridInfo(const cpu_tsdf::TSDFHashing& tsdf_model,
            const Eigen::Vector3i obb_boundingbox_voxel_size,
            const float vmin_mesh_weight = 0);

    Eigen::Vector3i boundingbox_size() const;
    void boundingbox_size(const Eigen::Vector3i &value);

    Eigen::Vector3f offset() const;
    void offset(const Eigen::Vector3f &value);

    Eigen::Vector3f voxel_lengths() const;
    void voxel_lengths(Eigen::Vector3f value);
    float max_dist_pos() const;
    void max_dist_pos(float value);

    float max_dist_neg() const;
    void max_dist_neg(float value);

    float min_model_weight() const;
    void min_model_weight(float value);

private:
    void InitFromVoxelBBSize(const cpu_tsdf::TSDFHashing &tsdf_model, const float vmin_mesh_weight);
    Eigen::Vector3i boundingbox_size_;
    Eigen::Vector3f offset_;
    Eigen::Vector3f voxel_lengths_;
    float max_dist_pos_;
    float max_dist_neg_;
    float min_model_weight_;
};

struct OptimizationParams {
    OptimizationParams()
        : save_path(), sample_size(51, 51, 51), canonical_obb(),
          lambda_average_scale(1000), lambda_observation(0), lambda_regularization(50), lambda_outlier(0),
          min_meshing_weight(0), max_dist_pos(1.2), max_dist_neg(-1.0),
          pc_num(0), pc_max_iter(25), opt_max_iter(3),
          noise_observation_thresh(3), noise_connected_component_thresh(-1) {}

    inline float VoxelLength() const {
        return canonical_obb.SamplingDeltas(sample_size)[0];
    }

    inline Eigen::Vector3f Offset() const {
        return canonical_obb.Offset();
    }

    std::string save_path;

    Eigen::Vector3i sample_size;
    tsdf_utility::OrientedBoundingBox canonical_obb;

    float lambda_average_scale;
    float lambda_observation;
    float lambda_regularization;
    float lambda_outlier;

    float min_meshing_weight;
    float max_dist_pos;
    float max_dist_neg;

    int pc_num;
    int pc_max_iter;
    int opt_max_iter;

    float  noise_observation_thresh;
    int noise_connected_component_thresh;
};

cpu_tsdf::PCAOptions OptParams2PCAOptions(const OptimizationParams& params);
OptimizationParams PCAOptions2OptParams(const cpu_tsdf::PCAOptions& options);

}
