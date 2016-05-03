#pragma once
/**
  * 1. AlignTSDF(): Compute the affine transformation (rotation, translation, scale) to align tsdf_model to tsdf_template.
  * The squared TSDF value difference is minimized using the ceres solver
  * 2. TransformTSDF(): Apply an affine transformation to a TSDF.
  */
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_utility.h"

namespace cpu_tsdf
{

// for writing out results during iterations in alignment
void WriteResultsForICCV(
        TSDFHashing::ConstPtr original_scene,
        const std::vector<Eigen::Affine3f> &affine_transforms,
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& reconstructed_sample_weights,
        const std::vector<int> &model_assign_idx,
        const PCAOptions& pca_options,
        const std::string &output_dir, const std::string &file_prefix);

/**
 * @brief AlignTSDF
 * Compute the affine transformation (rotation, translation, scale) to align tsdf_model to tsdf_template.
 * tsdf_model = tsdf_template * sR + T (template aligned to model)
 * rotation/trans contains the initial estimation of the transform
 * @param tsdf_model
 * @param tsdf_template
 * @param min_model_weight the minimum weight to consider the squared TSDF difference.
 * If a voxel in tsdf_model has smaller weight than min_model_weight, the TSDF difference on this voxel is not considered.
 * @param rotation
 * @param trans
 * @param scale
 * @return the final cost
 */
double AlignTSDF(const TSDFHashing& tsdf_model, 
                 const TSDFHashing& tsdf_template, const float min_model_weight, cv::Matx33f& rotation, cv::Vec3f& trans, float& scale);

double AlignTSDF(const TSDFHashing& tsdf_model, 
                 const TSDFHashing& tsdf_template, const float min_model_weight, Eigen::Affine3f* transform);

//bool TransformTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models,
//                    const std::vector<Eigen::Affine3f>& affine_transforms,
//                    std::vector<cpu_tsdf::TSDFHashing::Ptr>* transformed_tsdf_models,
//                    const float* voxel_length, const Eigen::Vector3f *scene_offset = NULL);
///**
// * @brief TransformTSDF
// * Apply an affine transformation to a TSDF
// * @param tsdf_origin Input TSDF
// * @param rotation Affine transform parameter
// * @param trans Affine transform parameter
// * @param scale Affine transform parameter
// * @param tsdf_translated Output
// * @return
// */
////bool TransformTSDF(const TSDFHashing& tsdf_origin,
////                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const float scale, TSDFHashing* tsdf_translated);
//bool TransformTSDF(const TSDFHashing& tsdf_origin,
//                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const cv::Vec3f &scale,
//                   TSDFHashing* tsdf_translated,
//                   const float *pvoxel_length = NULL, const Eigen::Vector3f *pscene_offset = NULL);

////bool TransformTSDF(const TSDFHashing& tsdf_origin,
////                   const Eigen::Affine3f& transform, TSDFHashing* tsdf_translated);
//bool TransformTSDF(const TSDFHashing& tsdf_origin,
//                   const Eigen::Affine3f& transform, TSDFHashing* tsdf_translated,
//                   const float* pvoxel_length = NULL, const Eigen::Vector3f *pscene_offset = NULL);

/**
 * @brief InitialAlign Compute an initial alignment between template and model
 * However currently it simply centralize the tsdf_model and tsdf_template to the origin of the world coordinate
 * (by modifying the voxel/world coordinate conversion part of the tsdf_model/tsdf_template)
 * @param tsdf_model
 * @param tsdf_template
 * @param rotation
 * @param trans
 * @param scale
 * @return
 */
bool InitialAlign(TSDFHashing& tsdf_model, TSDFHashing& tsdf_template, cv::Matx33f& rotation,
                  cv::Vec3f& trans, float& scale);

bool OptimizeTransformAndScalesImpl(const TSDFHashing& scene_tsdf,
                                    const std::vector<const TSDFHashing*> model_reconstructed_samples,
                                    const std::vector<int>& sample_model_assignment,
                                    const std::vector<double> &outlier_gammas,
                                    PCAOptions &pca_options,
                                    const std::string &save_path,
                                    std::vector<Eigen::Affine3f>* transforms, /*#samples, input and output*/
                                    std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
                                    );

//bool OptimizeTransformAndScalesImplRobustLoss(
//        const TSDFHashing& scene_tsdf,
//        const std::vector<const TSDFHashing*> model_reconstructed_samples,
//        const std::vector<int>& sample_model_assignment,
//        const PCAOptions& pca_options,
//        const std::string& save_path,
//        std::vector<Eigen::Affine3f>* transforms, /*#samples, input and output*/
//        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
//        );

bool OptimizeTransformAndScalesImplRobustLoss1(
        const TSDFHashing& scene_tsdf,
        const std::vector<const TSDFHashing*> model_reconstructed_samples,
        const std::vector<int>& sample_model_assignment,
        const std::vector<double>& gammas, // outliers
        tsdf_utility::OptimizationParams& params,
        std::vector<tsdf_utility::OrientedBoundingBox>* obbs, /*#samples, input and output*/
        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
        );

bool OptimizeTransformAndScalesImplRobustLoss(
        const TSDFHashing& scene_tsdf,
        const std::vector<const TSDFHashing*> model_reconstructed_samples,
        const std::vector<int>& sample_model_assignment,
        const std::vector<double>& gammas, // outliers
        PCAOptions& pca_options,
        const std::string& save_path,
        std::vector<Eigen::Affine3f>* transforms, /*#samples, input and output*/
        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
        );

bool OptimizeTransformAndScalesImpl_Debug(
        const TSDFHashing& scene_tsdf,
        const std::vector<const TSDFHashing*> model_reconstructed_samples,
        const std::vector<int>& sample_model_assignment,
        const float min_model_weight,
        std::vector<Eigen::Affine3f>* transforms, /*#samples, input and output*/
        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
        );


void ComputeTargetZScaleOneCluster(const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
        const std::vector<double> &cur_zscale,
        const TSDFGridInfo& grid_info,
        std::vector<double> *target_scale,
        const float percentile = 0.05
        );


void ComputeTargetZScaleOneCluster(std::vector<const cpu_tsdf::TSDFHashing*> tsdfs, const std::vector<double> &current_zscale,
        std::vector<double> *target_scale,
        const Eigen::Vector3i& boundingbox_size,
        const float percentile = 0.05
        );

}
