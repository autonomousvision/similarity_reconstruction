/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_hash_utilities/utility.h"

namespace cpu_tsdf {
class TSDFHashing;
}

namespace cpu_tsdf {
bool TransformTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models,
                    const std::vector<Eigen::Affine3f>& affine_transforms,
                    std::vector<cpu_tsdf::TSDFHashing::Ptr>* transformed_tsdf_models,
                    const float* voxel_length, const Eigen::Vector3f *scene_offset = NULL);
/**
 * @brief TransformTSDF
 * Apply an affine transformation to a TSDF
 * @param tsdf_origin Input TSDF
 * @param rotation Affine transform parameter
 * @param trans Affine transform parameter
 * @param scale Affine transform parameter
 * @param tsdf_translated Output
 * @return
 */
//bool TransformTSDF(const TSDFHashing& tsdf_origin,
//                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const float scale, TSDFHashing* tsdf_translated);
bool TransformTSDF(const TSDFHashing& tsdf_origin,
                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const cv::Vec3f &scale,
                   TSDFHashing* tsdf_translated,
                   const float *pvoxel_length = NULL, const Eigen::Vector3f *pscene_offset = NULL);

//bool TransformTSDF(const TSDFHashing& tsdf_origin,
//                   const Eigen::Affine3f& transform, TSDFHashing* tsdf_translated);
bool TransformTSDF(const TSDFHashing& tsdf_origin,
                   const Eigen::Affine3f& transform, TSDFHashing* tsdf_translated,
                   const float* pvoxel_length = NULL, const Eigen::Vector3f *pscene_offset = NULL);

}
