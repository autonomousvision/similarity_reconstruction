/*
 * Performing PCA on TSDF samples
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include "tsdf_representation/tsdf_hash.h"
#include "common/utilities/common_utility.h"
#include "common/utilities/eigen_utility.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_utility.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
// #include "tsdf_feature_generate.h"

namespace cpu_tsdf
{
class TSDFHashing;
}

namespace cpu_tsdf {

struct OptimizationOptions
{
    int PCA_component_num;
    int PCA_max_iter;
    int opt_max_iter;
    float  noise_clean_counter_thresh;
    int compo_thresh;
};

void ReconstructTSDFsFromPCAOriginPos(
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& recon_sample_weight,
        const std::vector<int> &sample_model_assign,
        const std::vector<tsdf_utility::OrientedBoundingBox>& obbs,
        const cpu_tsdf::TSDFGridInfo& grid_info,
        const float voxel_length,
        const Eigen::Vector3f& scene_tsdf_offset,
        std::vector<cpu_tsdf::TSDFHashing::Ptr>* reconstructed_samples_original_pos);

void ReconstructTSDFsFromPCAOriginPos(
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& recon_sample_mat,
        const std::vector<int> &sample_model_assign,
        const std::vector<Eigen::Affine3f>& affine_transforms,
        const cpu_tsdf::TSDFGridInfo& grid_info,
        const float voxel_length,
        const Eigen::Vector3f& scene_offset,
        std::vector<cpu_tsdf::TSDFHashing::Ptr>* reconstructed_samples_original_pos);

bool PCAReconstructionResult(
        const std::vector<Eigen::SparseVector<float>>& model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>& model_bases,
        const std::vector<Eigen::VectorXf>& projected_coeffs, /*#sample*/
        const std::vector<int>& model_assign_idx,
        std::vector<Eigen::SparseVector<float>>* reconstructed_samples
        );

bool PCAReconstructionResult(
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const std::vector<int> &model_assign_idx,
        std::vector<Eigen::SparseVector<float> > *reconstructed_samples);

bool OptimizeModelAndCoeff(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const std::vector<int> &model_assign_idx,
        const std::vector<double> &outlier_gammas,
        const int component_num, const int max_iter,
        std::vector<Eigen::SparseVector<float> > *model_means,
        std::vector<Eigen::SparseVector<float> > *model_mean_weight,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        tsdf_utility::OptimizationParams& params);

bool WeightedPCADeflationOrthogonal(const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
                const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
                const int component_num, const int max_iter,
        Eigen::SparseVector<float> *mean_mat,
        Eigen::SparseVector<float>* mean_weight,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
        Eigen::MatrixXf *coeff_mat,
        const tsdf_utility::OptimizationParams& options);

/**
 * @brief TSDFWeightedPCADeflationOrthogonal
 * Do weighted PCA using deflation, forcing principal components to be orthogonal to each other during optimization
 * @param tsdf_models
 * @param component_num
 * @param max_iter
 * @param mean_tsdf
 * @param mean_mat
 * @param base_mat
 * @param coeff_mat
 * @param weight_mat
 * @param boundingbox_size
 * @param save_filepath
 * @return
 */
bool TSDFWeightedPCADeflationOrthogonal_Old(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
                                        TSDFHashing *mean_tsdf, Eigen::SparseVector<float>* mean_mat,
                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf* coeff_mat,
                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
                                        Eigen::Vector3i *boundingbox_size, const std::string* save_filepath);

void WPCASaveMeanTSDF(
        const Eigen::SparseVector<float>& mean_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
        const Eigen::Vector3i& boundingbox_size,
        const float voxel_length,
        const Eigen::Vector3f& offset,
        const float max_dist_pos,
        const float max_dist_neg,
        const std::string& save_filepath);

void WPCASaveMeanTSDF(const Eigen::SparseVector<float>& mean_mat,
                      const cpu_tsdf::TSDFHashing& mean_tsdf,
                      const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
                      const Eigen::Vector3i& boundingbox_size,
                      const std::string& save_filepath);

// #samples: N, #dim: D, #principal_components: K
// BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;

bool TSDFWeightedPCA_NewWrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
                               cpu_tsdf::TSDFHashing *mean_tsdf, std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models,
                               const std::string *save_filepath);

/**
 * @brief TSDFWeightedPCA
 * Do weighted PCA. A wrapper function.
 * @param tsdf_models input models
 * @param component_num
 * @param max_iter maximum iterations
 * @param mean_tsdf output, the mean_tsdf
 * @param projected_tsdf_models output, the PCA projected tsdf models
 * @param save_filepath for debugging, saving file path
 * @return
 */
bool TSDFWeightedPCA_Wrapper(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
                     TSDFHashing *mean_tsdf, std::vector<TSDFHashing::Ptr>* projected_tsdf_models, const std::string *save_filepath);

/**
 * @brief TSDFWeightedPCA
 * Do weighted PCA using the original EM method
 * @param tsdf_models
 * @param component_num
 * @param max_iter
 * @param mean_tsdf Output, mean TSDF
 * @param mean_mat Output, mean TSDF in matrix
 * @param base_mat Output, base matrix
 * @param coeff_mat Output, coefficient matrix
 * @param weight_mat Output, the weight matrix (i.e. the voxel weights in tsdf_models)
 * @param boundingbox_size Output, the bounding box contains all the models
 * @param save_filepath Input, for debugging
 * @return
 */
bool TSDFWeightedPCA_Wrapper(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
                     TSDFHashing *mean_tsdf, Eigen::SparseVector<float>* mean_mat,
                     Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf* coeff_mat,
                     Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
                     Eigen::Vector3i *boundingbox_size, const std::string* save_filepath);

/**
 * @brief TSDFWeightedPCADeflation
 * Do weighted PCA using deflation
 * @param tsdf_models
 * @param component_num
 * @param max_iter
 * @param mean_tsdf
 * @param mean_mat
 * @param base_mat
 * @param coeff_mat
 * @param weight_mat
 * @param boundingbox_size
 * @param save_filepath
 * @return
 */
bool TSDFWeightedPCADeflation(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
                              TSDFHashing *mean_tsdf, Eigen::SparseVector<float>* mean_mat,
                              Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf* coeff_mat,
                              Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
                              Eigen::Vector3i *boundingbox_size, const std::string* save_filepath);



// void GetClusterSampleIdx(const std::vector<int> &model_assign_idx, const int model_number, std::vector<std::vector<int>>* cluster_sample_idx);
}
