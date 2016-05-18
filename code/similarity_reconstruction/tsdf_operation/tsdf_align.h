/*
 * Aligning multiple TSDFs together
 * Chen Zhou (zhouch@pku.edu.cn)
 */
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

bool OptimizeTransformAndScalesImplRobustLoss1(
        const TSDFHashing& scene_tsdf,
        const std::vector<const TSDFHashing*> model_reconstructed_samples,
        const std::vector<int>& sample_model_assignment,
        const std::vector<double>& gammas, // outliers
        tsdf_utility::OptimizationParams& params,
        std::vector<tsdf_utility::OrientedBoundingBox>* obbs, /*#samples, input and output*/
        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
        );

}
