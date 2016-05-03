// utilities for hashmap
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include "tsdf_representation/tsdf_hash.h"

namespace cpu_tsdf
{
// WrapAngle to [0, 2*pi);
inline double WrapAngleRange2PI(double angle) {
    angle = fmod(angle, 2.0 * M_PI);
    if (angle < 0.0)
        angle += 2.0 * M_PI;
    return angle;
}

// WrapAngle to [-pi, pi)
inline double WrapAngleRangePIs(double angle) {
    return ::atan2(::sin ( angle ), ::cos ( angle ));
}

template<typename T, typename Flag>
inline void EraseElementsAccordingToFlags(const std::vector<Flag>& should_remove, std::vector<T>* obj_vector) {
    typename std::vector<T>::iterator itr = obj_vector->begin();
    int index = 0;
    for (typename std::vector<Flag>::const_iterator fitr = should_remove.begin(); fitr != should_remove.end(); ++fitr, ++index) {
        if (*fitr) continue;
        *itr++ = (*obj_vector)[index];
    }
    obj_vector->erase(itr, obj_vector->end());
}

inline int Subscript3DToIndex(const Eigen::Vector3i& lengths, const Eigen::Vector3i& sub)
{
    return (sub[0] * lengths[1] + sub[1]) * lengths[2] + sub[2];
}

inline Eigen::Vector3i IndexToSubscript3D(const Eigen::Vector3i& lengths, const int index)
{
    Eigen::Vector3i sub;
    sub[2] = index % lengths[2];
    int temp = (index / lengths[2]);
    sub[1] =  temp % lengths[1];
    sub[0] =  temp / lengths[1];
    return sub;
}

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

class PCAOptions
{
public:
    Eigen::Vector3i boundingbox_size;
    float min_model_weight;
    std::string save_path;
    float lambda_scale_diff;
    float lambda_observation;
    float lambda_reg_rot;
    float lambda_reg_scale;
    float lambda_reg_trans;
    float lambda_reg_zscale;
    float lambda_outlier;
    float robust_loss_outlier_threshold;
//    cpu_tsdf::TSDFHashing tsdf_for_parameters;

//    float voxel_length() { return tsdf_for_parameters.voxel_length(); }
//    Eigen::Vector3f offset() { return tsdf_for_parameters.offset(); }
//    float max_dist_pos() { return tsdf_for_parameters.max_dist_pos(); }
//    float max_dist_neg() { return tsdf_for_parameters.max_dist_neg(); }
//    float dist_neg_inflection_point() { return tsdf_for_parameters.dist_neg_inflection_point(); }
//    float neg_inflection_weight() { return tsdf_for_parameters.neg_inflection_weight(); }


    float voxel_length;
    Eigen::Vector3f offset;
    float max_dist_pos;
    float max_dist_neg;
//    float dist_neg_inflection_point;
//    float neg_inflection_weight;


    //float ratio_original_voxel_length_to_unit_cube_vlength;
};

struct OrientedBoundingBox
{
    Eigen::Matrix3f bb_orientation;
    Eigen::Vector3f bb_sidelengths;
    Eigen::Vector3f bb_offset;
    Eigen::Vector3f voxel_lengths;
    // bool is_trainining_sample;
    inline Eigen::Vector3f BoxCenter() const
    {
        return bb_offset + (bb_orientation.col(0) * bb_sidelengths[0] + bb_orientation.col(1) * bb_sidelengths[1] + bb_orientation.col(2) * bb_sidelengths[2])/2.0;
    }
    inline Eigen::Vector3f BoxBottomCenter() const
    {
        return bb_offset + bb_orientation * Eigen::Vector3f(bb_sidelengths[0], bb_sidelengths[1], 0)/2.0;
    }
    inline void Clear()
    {
        bb_sidelengths.setZero();
        //voxel_lengths.setZero();
        bb_offset.setZero();
        bb_orientation.setZero();
    }
    void Extension(const Eigen::Vector3f& extension_each_side);

    void Display() const;

    OrientedBoundingBox Rescale(const Eigen::Vector3f& scales) const;
    OrientedBoundingBox RescaleNoBottom(const Eigen::Vector3f &scales) const;


    inline void ComputeVoxelLengths(const Eigen::Vector3f& voxel_side_lengths)
    {
        voxel_lengths = bb_sidelengths.cwiseQuotient((voxel_side_lengths).cast<float>() - Eigen::Vector3f::Ones());
        Eigen::Vector3f test = (voxel_lengths.cwiseProduct((voxel_side_lengths).cast<float>() - Eigen::Vector3f::Ones()) - bb_sidelengths);
        CHECK_LE(test.norm(), 1e-4);
    }

    //OrientedBoundingBox JitterSample(float transx, float transy, float angle, const Eigen::Vector3f& scales);


//    inline Eigen::Vector3f BoxCenter2Offset(const Eigen::Vector3f& center)
//    {
//        return center - (bb_orientation.col(0) * bb_sidelengths[0] + bb_orientation.col(1) * bb_sidelengths[1] + bb_orientation.col(2) * bb_sidelengths[2])/2.0;
//    }
};

inline bool VerticeInOBB(const Eigen::Vector3f& point, const OrientedBoundingBox& obb)
{
    const Eigen::Vector3f& world_coord = point;
    Eigen::Vector3f obb_coord = obb.bb_orientation.transpose() * (world_coord - obb.bb_offset);
    const Eigen::Vector3f& box_lengths = obb.bb_sidelengths;
    if (! (obb_coord[0] >= 0 && obb_coord[0] <= box_lengths[0] &&
           obb_coord[1] >= 0 && obb_coord[1] <= box_lengths[1] &&
           obb_coord[2] >= 0 && obb_coord[2] <= box_lengths[2]))
    {
        return false;
    }
    return true;
}

inline OrientedBoundingBox ComputeOBBFromCenter(
        const Eigen::Vector3f& center,
        const Eigen::Matrix3f& orient,
        const Eigen::Vector3f& side_lengths,
        const Eigen::Vector3i& voxel_side_lengths)
{
    OrientedBoundingBox obb;
    obb.bb_offset = center - orient * side_lengths/2.0;
    obb.bb_orientation = orient;
    obb.bb_sidelengths = side_lengths;
    obb.voxel_lengths = obb.bb_sidelengths.cwiseQuotient((voxel_side_lengths).cast<float>() - Eigen::Vector3f::Ones());
    return obb;
}

inline OrientedBoundingBox ComputeOBBFromBottomCenter(
        const Eigen::Vector2f& center_xy,
        const float bottom_z,
        const Eigen::Matrix3f& orient,
        const Eigen::Vector3f& side_lengths,
        const Eigen::Vector3i& voxel_side_lengths)
{
    Eigen::Vector3f center(center_xy[0], center_xy[1], bottom_z + side_lengths[2]/2.0);
    return ComputeOBBFromCenter(center, orient, side_lengths, voxel_side_lengths);
}

bool ComputeOrientedBoundingboxVertices(const Eigen::Matrix3f &voxel_world_rotation,
	const Eigen::Vector3f &offset, const Eigen::Vector3f &world_side_lengths,
	Eigen::Vector3f* box_vertices);

bool AffineToOrientedBB(const Eigen::Affine3f& trans, cpu_tsdf::OrientedBoundingBox* obb);

bool AffinesToOrientedBBs(const std::vector<Eigen::Affine3f>& trans, std::vector<cpu_tsdf::OrientedBoundingBox>* obbs);

void OBBToAffine(const OrientedBoundingBox &obb, Eigen::Affine3f *transform);

void OBBsToAffines(const std::vector<OrientedBoundingBox> &obbs, std::vector<Eigen::Affine3f> *transforms);

Eigen::SparseMatrix<float, Eigen::ColMajor> SparseVectorsToEigenMat(const std::vector<Eigen::SparseVector<float>>& samples);

std::vector<Eigen::SparseVector<float>> EigenMatToSparseVectors(const Eigen::SparseMatrix<float, Eigen::ColMajor> & samples);



inline void ConvertDataMatrixToDataVectors(const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat, std::vector<Eigen::SparseVector<float> > *data_vec)
{
    *data_vec = EigenMatToSparseVectors(data_mat);
}

inline void ConvertDataVectorsToDataMat(const std::vector<Eigen::SparseVector<float> > &data_vec, Eigen::SparseMatrix<float, Eigen::ColMajor> *data_mat)
{
    *data_mat = SparseVectorsToEigenMat(data_vec);
}

//bool ExtractSamplesFromAffineTransform(
//        const TSDFHashing &scene_tsdf,
//        const std::vector<Eigen::Affine3f> &affine_transforms,
//        const TSDFGridInfo &options,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights);
//
//bool ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf,
//                                         const Eigen::Affine3f &affine_transform,
//                                         const TSDFGridInfo &options,
//                                         Eigen::SparseVector<float> *sample,
//                                         Eigen::SparseVector<float> *weight);
//
bool ConvertDataVectorToTSDFWithWeightAndWorldPos(const Eigen::SparseVector<float> &tsdf_data_vec,
        const Eigen::SparseVector<float> &tsdf_weight_vec,
        const TSDFGridInfo& tsdf_info,
        TSDFHashing *tsdf,
        std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f> > *idx_worldpos);

bool ConvertDataVectorToTSDFWithWeight(const Eigen::SparseVector<float>& tsdf_data_vec,
        const Eigen::SparseVector<float>& tsdf_weight_vec,
        const TSDFGridInfo& tsdf_info,
        cpu_tsdf::TSDFHashing* tsdf);

bool ConvertDataVectorToTSDFNoWeight(
        const Eigen::SparseVector<float> &tsdf_data_vec,
        const cpu_tsdf::TSDFGridInfo &options,
        TSDFHashing *tsdf);

bool ConvertDataVectorToTSDFNoWeight(
        const Eigen::SparseVector<float> &tsdf_data_vec,
        const PCAOptions &options,
        TSDFHashing *tsdf);

bool ConvertDataVectorsToTSDFsNoWeight(
        const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec,
        const TSDFGridInfo &options,
        std::vector<TSDFHashing::Ptr> *tsdfs);

bool ConvertDataVectorsToTSDFsNoWeight(
        const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec,
        const PCAOptions &options,
        std::vector<TSDFHashing::Ptr> *tsdfs);

bool ConvertDataVectorsToTSDFsWithWeight(
        const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec,
        const std::vector<Eigen::SparseVector<float> > &tsdf_weight_vec,
        const PCAOptions &options,
        std::vector<TSDFHashing::Ptr> *tsdfs);

bool ConvertDataVectorsToTSDFsWithWeight(
        const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec,
        const std::vector<Eigen::SparseVector<float> > &tsdf_weight_vec,
        const TSDFGridInfo &options,
        std::vector<TSDFHashing::Ptr> *tsdfs);


bool ConvertDataVectorsToTSDFsWithWeight(
        const std::vector<Eigen::SparseVector<float> > &tsdf_data,
        Eigen::SparseMatrix<float, Eigen::ColMajor> &tsdf_weights,
        const PCAOptions &options,
        std::vector<TSDFHashing::Ptr> *tsdfs);

bool ConvertDataVectorsToTSDFsWithWeight(
        const std::vector<Eigen::SparseVector<float> > &tsdf_data,
        Eigen::SparseMatrix<float, Eigen::ColMajor> &tsdf_weights,
        const TSDFGridInfo &options,
        std::vector<TSDFHashing::Ptr> *tsdfs);


bool ConvertDataMatrixToTSDFs(
        const float voxel_length,
        const Eigen::Vector3f &offset,
        const float max_dist_pos, const float max_dist_neg,
        const Eigen::Vector3i &voxel_bounding_box_size,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat,
        std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models);

inline bool ConvertDataMatrixToTSDFs(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat,
        const cpu_tsdf::TSDFGridInfo& grid_info,
        std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models
        )
{
    return ConvertDataMatrixToTSDFs(grid_info.voxel_lengths()[0], grid_info.offset(), grid_info.max_dist_pos(), grid_info.max_dist_neg(),
            grid_info.boundingbox_size(), data_mat, weight_mat, projected_tsdf_models);
}

bool ConvertDataMatrixToTSDFsNoWeight(
        const float voxel_length,
        const Eigen::Vector3f &offset,
        const float max_dist_pos, const float max_dist_neg,
        const Eigen::Vector3i &voxel_bounding_box_size,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat,
        std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models);

/*
 * deprecated
*/

/* deprecated*/
//bool ExtractSamplesFromAffineTransform(
//        const TSDFHashing &scene_tsdf,
//        const std::vector<Eigen::Affine3f> &affine_transforms,
//        const PCAOptions &options,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights);
//
///* deprecated*/
//bool ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf,
//                                         const Eigen::Affine3f &affine_transform,
//                                         const PCAOptions &options,
//                                         Eigen::SparseVector<float> *sample,
//                                         Eigen::SparseVector<float> *weight);

/* deprecated*/
bool ConvertDataVectorToTSDFWithWeightAndWorldPos(const Eigen::SparseVector<float> &tsdf_data_vec,
        const Eigen::SparseVector<float> &tsdf_weight_vec,
        const PCAOptions &options,
        TSDFHashing *tsdf,
        std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f> > *idx_worldpos);

/* deprecated*/
bool ConvertDataVectorToTSDFWithWeight(
        const Eigen::SparseVector<float>& tsdf_data_vec,
        const Eigen::SparseVector<float>& tsdf_weight_vec,
        const PCAOptions& options,
        cpu_tsdf::TSDFHashing* tsdf);

void MaskImageSidesAsZero(const int side_width, cv::Mat* image);

void GetClusterSampleIdx(const std::vector<int> &sample_cluster_idx, const std::vector<double>& outlier_gamma,
                         const int model_number, std::vector<std::vector<int>>* cluster_sample_idx);
}  // namespace cpu_tsdf
