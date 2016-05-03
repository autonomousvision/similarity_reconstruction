#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include "tsdf_representation/tsdf_hash.h"
#include "common/utilities/common_utility.h"
#include "common/utilities/eigen_utility.h"
#include "tsdf_hash_utilities/utility.h"
// #include "tsdf_feature_generate.h"
#include "tsdf_operation/tsdf_pca.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "tsdf_operation/tsdf_utility.h"

namespace cpu_tsdf
{
class TSDFHashing;
}

using namespace std;

namespace tsdf_optimization {

// Compute average scales for each model
void ComputeModelAverageScales(
        const std::vector<tsdf_utility::OrientedBoundingBox>& obbs,
        const std::vector<int>& sample_model_assign,
        const int model_num,
        std::vector<Eigen::Vector3f> *model_average_scales
        );

void InitializeOptimization(const cpu_tsdf::TSDFHashing &scene_tsdf,
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs, const std::vector<float> &obb_scores,
        const std::vector<int> sample_model_assign,
        tsdf_utility::OptimizationParams& params,
        std::vector<Eigen::SparseVector<float> > *model_means,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        std::vector<Eigen::Vector3f> *model_average_scales,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights);

bool JointClusteringAndModelLearning(const cpu_tsdf::TSDFHashing &scene_tsdf,
                                     tsdf_utility::OptimizationParams& params,
                                     std::vector<Eigen::SparseVector<float> > *model_means,
                                     std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
                                     std::vector<Eigen::VectorXf> *projected_coeffs,
                                     std::vector<int> *sample_model_assign,
                                     std::vector<double> *outlier_gammas,
                                     std::vector<tsdf_utility::OrientedBoundingBox> *obbs,
                                     std::vector<Eigen::Vector3f> *model_average_scales,
                                     Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights
                                     );

bool OptimizeTransformAndScale(const cpu_tsdf::TSDFHashing &scene_tsdf,
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const std::vector<int> &model_assign_idx,
        const std::vector<double> & outlier_gammas,
        tsdf_utility::OptimizationParams& params,
        std::vector<tsdf_utility::OrientedBoundingBox> *obbs,
        std::vector<Eigen::Vector3f> *model_average_scales, Eigen::SparseMatrix<float, Eigen::ColMajor> &recon_weights,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights
        );

bool OptimizePCACoeffAndCluster(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &reconstructed_sample_weights,
        const std::vector<Eigen::SparseVector<float>> &cluster_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>> &cluster_bases,
        const std::vector<tsdf_utility::OrientedBoundingBox>& obbs,
        const tsdf_utility::OptimizationParams& params,
        std::vector<int> *cluster_assignment,
        std::vector<double> *outlier_gammas,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        std::vector<float> *cluster_assignment_error,
        std::vector<Eigen::Vector3f>* cluster_average_scales
        );

bool WriteSampleClusters(const std::vector<int> &sample_model_assign,
                         const std::vector<Eigen::Vector3f> &model_average_scales,
                         const std::vector<double>& outlier_gammas,
                         const std::vector<float>& cluster_assignment_error, const string &save_path);

bool WeightedPCAProjectionMultipleSamples(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights, const Eigen::SparseMatrix<float, Eigen::ColMajor> &reconstructed_sample_weights,
        const Eigen::SparseVector<float> &mean_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
        Eigen::MatrixXf *projected_coeffs,
        float *squared_errors);

bool WeightedPCAProjectionOneSample(const Eigen::SparseVector<float> &sample,
        const Eigen::SparseVector<float> &weight, const Eigen::SparseVector<float> &reconstruct_weight,
        const Eigen::SparseVector<float> &mean_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
        Eigen::VectorXf *projected_coeff,
        float *squared_error);
}
