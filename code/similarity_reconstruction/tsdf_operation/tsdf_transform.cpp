/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "tsdf_transform.h"
#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <cmath>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "common/utilities/common_utility.h"
#include "common/utilities/eigen_utility.h"
#include "../tsdf_representation/tsdf_hash.h"
#include "glog/logging.h"
#include "tsdf_hash_utilities/utility.h"
// #include "tsdf_operation/tsdf_io.h"
// #include "tsdf_operation/tsdf_pca.h"

using namespace utility;
using namespace std;
using namespace cv;
namespace bfs = boost::filesystem;

namespace cpu_tsdf {
bool TransformTSDF(const TSDFHashing& tsdf_origin,
                   const Eigen::Affine3f& transform,
                   TSDFHashing* tsdf_translated,
                   const float* pvoxel_length,
                   const Eigen::Vector3f* pscene_offset)
{
    Eigen::Matrix3f rotation;
    Eigen::Vector3f scaling;
    Eigen::Vector3f trans;
    utility::EigenAffine3fDecomposition(transform, &rotation, &scaling, &trans);
    //transform.computeRotationScaling(&rotation, &scaling);
    //float scale = scaling(0, 0);
    //Eigen::Vector3f trans = transform.translation();

    //cout << "Eigen Rotation Matrix:\n " << rotation << endl;
    //cout << "Eigen Scaling Matrix:\n " << scaling << endl;
    //cout << "Eigen Translation:\n " << trans << endl;
    return TransformTSDF(tsdf_origin, EigenMatToCvMatx<float, 3, 3>(rotation),
                         EigenVectorToCvVector3(trans), EigenVectorToCvVector3(scaling), tsdf_translated, pvoxel_length, pscene_offset);

}

bool TransformTSDF(const TSDFHashing& tsdf_origin,
                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const cv::Vec3f& scale, TSDFHashing* tsdf_translated,
                   const float* pvoxel_length, const Eigen::Vector3f *pscene_offset)
{
    //cout << "begin transform TSDF. " << endl;
    //cout << "rotation: " << (cv::Mat)rotation << endl;
    //cout << "trans: " << (cv::Mat)trans << endl;
    //cout << "scale: " << scale << endl;

    cv::Matx33f scale_mat = cv::Matx33f::eye();
    scale_mat(0, 0) = scale[0];
    scale_mat(1, 1) = scale[1];
    scale_mat(2, 2) = scale[2];
    cv::Matx33f scaled_rot = rotation * scale_mat;
    float voxel_length = pvoxel_length ? *pvoxel_length:tsdf_origin.voxel_length();
    Eigen::Vector3f offset = pscene_offset ? *pscene_offset : tsdf_origin.offset();
    // cv::Vec3f cvoffset = EigenVectorToCvVector3(offset);
    // tsdf_translated->offset(CvVectorToEigenVector3(scaled_rot * cvoffset + trans));
    float max_dist_pos, max_dist_neg;
    tsdf_origin.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    // tsdf_translated->Init(voxel_length, CvVectorToEigenVector3(scaled_rot * cvoffset + trans), max_dist_pos, max_dist_neg);
    tsdf_translated->Init(voxel_length, offset, max_dist_pos, max_dist_neg);

    TSDFHashing::update_hashset_type brick_update_hashset;
    for (TSDFHashing::const_iterator citr = tsdf_origin.begin(); citr != tsdf_origin.end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        citr->RetriveData(&d, &w, &color);
        if (w > 0)
        {
            cv::Vec3f world_coord_origin = tsdf_origin.Voxel2World(cv::Vec3f(cur_voxel_coord));
            cv::Vec3f world_coord_origin_transformed =
                    scaled_rot * (world_coord_origin) + trans;
            cv::Vec3i voxel_coord_transformed = cv::Vec3i(tsdf_translated->World2Voxel(world_coord_origin_transformed));
            tsdf_translated->AddBrickUpdateList(voxel_coord_transformed, &brick_update_hashset);
        }  // end if
    }  // end for
    // cout << "update list size: " << brick_update_hashset.size() << endl;

    struct TSDFTranslatedVoxelUpdater
    {
        TSDFTranslatedVoxelUpdater(const TSDFHashing& tsdf_origin, const TSDFHashing& tsdf_template,
                                   const cv::Matx33f& inverse_scaled_rot, const cv::Vec3f& inverse_trans)
            : tsdf_origin(tsdf_origin), tsdf_template(tsdf_template), inverse_scaled_rot(inverse_scaled_rot),
              inverse_trans(inverse_trans) {}
        bool operator () (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
        {
            // for every voxel in its new location, retrive its value in the original tsdf
            cv::Vec3f world_coord_transformed = tsdf_template.Voxel2World(cv::Vec3f(cur_voxel_coord));
            cv::Vec3f world_coord_untransformed = inverse_scaled_rot * world_coord_transformed + inverse_trans;
            float cur_d;
            float cur_w;
            cv::Vec3b cur_color;
            if (!tsdf_origin.RetriveDataFromWorldCoord(world_coord_untransformed, &cur_d, &cur_w, &cur_color)) return false;
            *d = cur_d;
            *w = cur_w;
            *color = cur_color;
            return true;
        }
        const TSDFHashing& tsdf_origin;
        const TSDFHashing& tsdf_template;
        const cv::Matx33f& inverse_scaled_rot;
        const cv::Vec3f inverse_trans;
    };
    // cv::Matx33f inverse_scaled_rot = rotation.t() * (1.0 / scale);
    cv::Matx33f inverse_scale_mat = cv::Matx33f::eye();
    inverse_scale_mat(0, 0) = 1.0/scale[0];
    inverse_scale_mat(1, 1) = 1.0/scale[1];
    inverse_scale_mat(2, 2) = 1.0/scale[2];
    cv::Matx33f inverse_scaled_rot = inverse_scale_mat * rotation.t();
    cv::Vec3f inverse_trans = inverse_scaled_rot * (-trans);
    TSDFTranslatedVoxelUpdater updater(tsdf_origin, *tsdf_translated, inverse_scaled_rot, inverse_trans);
    tsdf_translated->UpdateBricksInQueue(brick_update_hashset,
                                         updater);
    //cout << "finished transform TSDF. " << endl;
    return true;
}

bool TransformTSDF(const TSDFHashing& tsdf_origin,
                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const float scale, TSDFHashing* tsdf_translated)
{
    //cout << "begin transform TSDF. " << endl;
    //cout << "rotation: " << (cv::Mat)rotation << endl;
    //cout << "trans: " << (cv::Mat)trans << endl;
    //cout << "scale: " << scale << endl;

    cv::Matx33f scaled_rot = rotation * scale;
    float voxel_length = tsdf_origin.voxel_length();
    Eigen::Vector3f offset = tsdf_origin.offset();
    cv::Vec3f cvoffset = EigenVectorToCvVector3(offset);
    float max_dist_pos, max_dist_neg;
    tsdf_origin.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    tsdf_translated->Init(voxel_length, CvVectorToEigenVector3(scaled_rot * cvoffset + trans), max_dist_pos, max_dist_neg);
    TSDFHashing::update_hashset_type brick_update_hashset;
    for (TSDFHashing::const_iterator citr = tsdf_origin.begin(); citr != tsdf_origin.end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        citr->RetriveData(&d, &w, &color);
        if (w > 0)
        {
            cv::Vec3f world_coord_origin = tsdf_origin.Voxel2World(cv::Vec3f(cur_voxel_coord));
            cv::Vec3f world_coord_origin_transformed =
                    scaled_rot * (world_coord_origin) + trans;
            cv::Vec3i voxel_coord_transformed = cv::Vec3i(tsdf_translated->World2Voxel(world_coord_origin_transformed));
            tsdf_translated->AddBrickUpdateList(voxel_coord_transformed, &brick_update_hashset);
        }  // end if
    }  // end for
    // cout << "update list size: " << brick_update_hashset.size() << endl;

    struct TSDFTranslatedVoxelUpdater
    {
        TSDFTranslatedVoxelUpdater(const TSDFHashing& tsdf_origin, const TSDFHashing& tsdf_template,
                                   const cv::Matx33f& inverse_scaled_rot, const cv::Vec3f& inverse_trans)
            : tsdf_origin(tsdf_origin), tsdf_template(tsdf_template), inverse_scaled_rot(inverse_scaled_rot),
              inverse_trans(inverse_trans) {}
        bool operator () (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
        {
            // for every voxel in its new location, retrive its value in the original tsdf
            cv::Vec3f world_coord_transformed = tsdf_template.Voxel2World(cv::Vec3f(cur_voxel_coord));
            cv::Vec3f world_coord_untransformed = inverse_scaled_rot * world_coord_transformed + inverse_trans;
            float cur_d;
            float cur_w;
            cv::Vec3b cur_color;
            if (!tsdf_origin.RetriveDataFromWorldCoord(world_coord_untransformed, &cur_d, &cur_w, &cur_color)) return false;
            *d = cur_d;
            *w = cur_w;
            *color = cur_color;
            return true;
        }
        const TSDFHashing& tsdf_origin;
        const TSDFHashing& tsdf_template;
        const cv::Matx33f& inverse_scaled_rot;
        const cv::Vec3f inverse_trans;
    };
    cv::Matx33f inverse_scaled_rot = rotation.t() * (1.0/ scale);
    cv::Vec3f inverse_trans = inverse_scaled_rot * (-trans);
    TSDFTranslatedVoxelUpdater updater(tsdf_origin, *tsdf_translated, inverse_scaled_rot, inverse_trans);
    tsdf_translated->UpdateBricksInQueue(brick_update_hashset,
                                         updater);
    // cout << "finished transform TSDF. " << endl;
    return true;
}

bool TransformTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
                    const std::vector<Eigen::Affine3f> &affine_transforms,
                    std::vector<cpu_tsdf::TSDFHashing::Ptr> *transformed_tsdf_models,
                    const float *voxel_length, const Eigen::Vector3f* scene_offset)
{
    transformed_tsdf_models->resize(tsdf_models.size());
    for (int i = 0; i < tsdf_models.size(); ++i)
    {
        (*transformed_tsdf_models)[i].reset(new cpu_tsdf::TSDFHashing);
        TransformTSDF(*(tsdf_models[i]), affine_transforms[i], (*transformed_tsdf_models)[i].get(), voxel_length, scene_offset);
    }
}
}
