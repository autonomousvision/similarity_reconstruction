#pragma once
/**
  * Functions for joint TSDF align and doing WPCA on TSDFs.
  * For notation used in PCA functions, refer to
  * Weighted and robust learning of subspace representations, Pattern Recognition 2007
  * http://ac.els-cdn.com/S0031320306003876/1-s2.0-S0031320306003876-main.pdf?_tid=124f2502-5afa-11e4-bbd1-00000aacb360&acdnat=1414099262_dd5fdb6db7f187a02e93533f488872f1
  * and
  * http://cs.brown.edu/~black/rpca.html
  */

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include "tsdf_representation/tsdf_hash.h"
#include "common/utility/common_utility.h"
#include "common/utility/eigen_utility.h"
#include "utility/utility.h"
#include "tsdf_feature_generate.h"
#include "tsdf_operation/tsdf_pca.h"

namespace cpu_tsdf
{
class TSDFHashing;
}

namespace cpu_tsdf
{

//struct OptimizationOptions
//{
//    int PCA_component_num;
//    int PCA_max_iter;
//    int opt_max_iter;
//    float  noise_clean_counter_thresh;
//    int compo_thresh;
//};

//void WriteResultsForICCV(
//        TSDFHashing::ConstPtr original_scene,
//        const std::vector<Eigen::Affine3f> &affine_transforms,
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& reconstructed_sample_weights,
//        const std::vector<int> &model_assign_idx,
//        const PCAOptions& pca_options,
//        const std::string &output_dir, const std::string &file_prefix);

//void ReconstructTSDFsFromPCAOriginPos(
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& recon_sample_mat,
//        const std::vector<int> &sample_model_assign,
//        const std::vector<Eigen::Affine3f>& affine_transforms,
//        const cpu_tsdf::TSDFGridInfo& grid_info,
//        const float voxel_length,
//        const Eigen::Vector3f& scene_offset,
//        std::vector<cpu_tsdf::TSDFHashing::Ptr>* reconstructed_samples_original_pos);

//void InitialClusteringAndModelLearningInitialDebugging2(const TSDFHashing &scene_tsdf,
//                                                        const std::vector<OrientedBoundingBox> &detected_boxes,
//                                                        PCAOptions *pca_options,
//                                                        const OptimizationOptions &optimize_options,
//                                                        std::vector<Eigen::SparseVector<float> > *model_means,
//                                                        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
//                                                        std::vector<Eigen::VectorXf> *projected_coeffs,
//                                                        std::vector<int> *sample_model_assign,
//                                                        std::vector<Eigen::Affine3f> *affine_transforms,
//                                                        std::vector<Eigen::Vector3f> *model_average_scales,
//                                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *reconstructed_sample_weights);

void InitialClusteringAndModelLearningInitialDebugging2(
        const TSDFHashing &scene_tsdf,
        const std::vector<OrientedBoundingBox> &detected_boxes,
        PCAOptions *pca_options,
        const OptimizationOptions &optimize_options,
        std::vector<Eigen::SparseVector<float> > *model_means,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        std::vector<int> *sample_model_assign,
        std::vector<Eigen::Affine3f> *affine_transforms,
        std::vector<Eigen::Vector3f> *model_average_scales,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights);
/**
 * @brief JointTSDFAlign Iteratively optimizing the transformation parameters and averaging
 * @param tsdf_models: Input, the tsdf_models to align, modified (transformed) during the process
 * @param transform: Output, the transform
 * @param tsdf_template: Output, the learned template
 * @param poutput_plyfilename: for debugging, filename for saving intermediate results
 * @return
 */
bool JointTSDFAlign(std::vector<cpu_tsdf::TSDFHashing::Ptr>* tsdf_models, std::vector<Eigen::Affine3f>* transform,
                    cpu_tsdf::TSDFHashing* tsdf_template, const std::string* poutput_plyfilename = NULL);

/**
 * @brief AverageTSDFsIntersection: Computing the average TSDF on the intersection of tsdf_models
 * @param tsdf_models
 * @param tsdf_template
 * @return
 */
bool AverageTSDFsIntersection(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models, cpu_tsdf::TSDFHashing* tsdf_template);

/**
 * @brief AverageTSDFsUnion: Computing the average TSDF on the union of tsdf_models
 * @param tsdf_models
 * @param tsdf_template
 * @return
 */
bool AverageTSDFsUnion(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models, cpu_tsdf::TSDFHashing* tsdf_template);

//// #samples: N, #dim: D, #principal_components: K
//// BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;

//bool TSDFWeightedPCA_NewWrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
//                               cpu_tsdf::TSDFHashing *mean_tsdf, std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models,
//                               const std::string *save_filepath);

///**
// * @brief TSDFWeightedPCA
// * Do weighted PCA. A wrapper function.
// * @param tsdf_models input models
// * @param component_num
// * @param max_iter maximum iterations
// * @param mean_tsdf output, the mean_tsdf
// * @param projected_tsdf_models output, the PCA projected tsdf models
// * @param save_filepath for debugging, saving file path
// * @return
// */
//bool TSDFWeightedPCA_Wrapper(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
//                     TSDFHashing *mean_tsdf, std::vector<TSDFHashing::Ptr>* projected_tsdf_models, const std::string *save_filepath);

///**
// * @brief TSDFWeightedPCA
// * Do weighted PCA using the original EM method
// * @param tsdf_models
// * @param component_num
// * @param max_iter
// * @param mean_tsdf Output, mean TSDF
// * @param mean_mat Output, mean TSDF in matrix
// * @param base_mat Output, base matrix
// * @param coeff_mat Output, coefficient matrix
// * @param weight_mat Output, the weight matrix (i.e. the voxel weights in tsdf_models)
// * @param boundingbox_size Output, the bounding box contains all the models
// * @param save_filepath Input, for debugging
// * @return
// */
//bool TSDFWeightedPCA_Wrapper(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
//                     TSDFHashing *mean_tsdf, Eigen::SparseVector<float>* mean_mat,
//                     Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf* coeff_mat,
//                     Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
//                     Eigen::Vector3i *boundingbox_size, const std::string* save_filepath);

///**
// * @brief TSDFWeightedPCADeflation
// * Do weighted PCA using deflation
// * @param tsdf_models
// * @param component_num
// * @param max_iter
// * @param mean_tsdf
// * @param mean_mat
// * @param base_mat
// * @param coeff_mat
// * @param weight_mat
// * @param boundingbox_size
// * @param save_filepath
// * @return
// */
//bool TSDFWeightedPCADeflation(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
//                              TSDFHashing *mean_tsdf, Eigen::SparseVector<float>* mean_mat,
//                              Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf* coeff_mat,
//                              Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
//                              Eigen::Vector3i *boundingbox_size, const std::string* save_filepath);

///**
// * @brief TSDFWeightedPCADeflationOrthogonal
// * Do weighted PCA using deflation, forcing principal components to be orthogonal to each other during optimization
// * @param tsdf_models
// * @param component_num
// * @param max_iter
// * @param mean_tsdf
// * @param mean_mat
// * @param base_mat
// * @param coeff_mat
// * @param weight_mat
// * @param boundingbox_size
// * @param save_filepath
// * @return
// */
//bool TSDFWeightedPCADeflationOrthogonal_Old(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
//                                        TSDFHashing *mean_tsdf, Eigen::SparseVector<float>* mean_mat,
//                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf* coeff_mat,
//                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
//                                        Eigen::Vector3i *boundingbox_size, const std::string* save_filepath);
//bool TSDFWeightedPCADeflationOrthogonal(const std::vector<TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
//                                        TSDFHashing *mean_tsdf, Eigen::SparseVector<float>* mean_mat,
//                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf* coeff_mat,
//                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
//                                        Eigen::Vector3i *boundingbox_size, const std::string* save_filepath);

//void ConvertOneOBBToAffineTransform(const cpu_tsdf::OrientedBoundingBox& obb,
//        Eigen::Affine3f *transform);

//void ConvertOrientedBoundingboxToAffineTransforms(const std::vector<cpu_tsdf::OrientedBoundingBox>& obbs,
//        std::vector<Eigen::Affine3f> *transforms
//        );

//bool ExtractOneSampleFromAffineTransform(const TSDFHashing& scene_tsdf,
//        const Eigen::Affine3f &affine_transform,
//        const PCAOptions& options,
//        Eigen::SparseVector<float>* sample,
//        Eigen::SparseVector<float>* weight
//        );

//bool ExtractSamplesFromAffineTransform(const TSDFHashing& scene_tsdf,
//        const std::vector<Eigen::Affine3f> &affine_transforms, /*size: #samples*/
//        const PCAOptions& options,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* weights
//        );

void InitialClusteringAndModelLearningInitialDebugging(
        const cpu_tsdf::TSDFHashing& scene_tsdf,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& detected_boxes,
        PCAOptions* pca_options,
        std::vector<Eigen::SparseVector<float>>* model_means,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>* model_bases,
        std::vector<Eigen::VectorXf>* projected_coeffs,
        std::vector<int>* sample_model_assign,
        std::vector<Eigen::Affine3f>* affine_transforms,
        std::vector<Eigen::Vector3f>* model_average_scales
        );

//void InitialClusteringAndModelLearning(
//        const cpu_tsdf::TSDFHashing& scene_tsdf,
//        const std::vector<cpu_tsdf::OrientedBoundingBox>& detected_boxes,
//        const PCAOptions* pca_options,
//        std::vector<Eigen::SparseVector<float>>* model_means,
//        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>* model_bases,
//        std::vector<Eigen::VectorXf>* projected_coeffs,
//        std::vector<int>* sample_model_assign,
//        std::vector<Eigen::Affine3d>* affine_transforms,
//        std::vector<Eigen::Vector3f>* model_average_scales
//        )
//{
//}

//bool ConvertDataVectorToTSDFNoWeight(
//        const Eigen::SparseVector<float>& tsdf_data_vec,
//        const PCAOptions& options,
//        cpu_tsdf::TSDFHashing* tsdf);

//bool ConvertDataVectorsToTSDFsNoWeight(
//        const std::vector<Eigen::SparseVector<float>>& tsdf_data_vec,
//        const PCAOptions& options,
//        std::vector<cpu_tsdf::TSDFHashing::Ptr>* tsdfs
//        );

//bool ConvertDataVectorToTSDFWithWeight(
//        const Eigen::SparseVector<float>& tsdf_data_vec,
//        const Eigen::SparseVector<float>& tsdf_weight_vec,
//        const PCAOptions& options,
//        cpu_tsdf::TSDFHashing* tsdf);

//bool ConvertDataVectorsToTSDFsWithWeight(
//        const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec,
//        const std::vector<Eigen::SparseVector<float> > &tsdf_weight_vec,
//        const PCAOptions &options,
//        std::vector<TSDFHashing::Ptr> *tsdfs);

//bool ConvertDataVectorsToTSDFsWithWeight(
//        const std::vector<Eigen::SparseVector<float> > &tsdf_data,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> &tsdf_weights,
//        const PCAOptions &options,
//        std::vector<TSDFHashing::Ptr> *tsdfs);

bool JointClusteringAndModelLearning(const cpu_tsdf::TSDFHashing& scene_tsdf,
        std::vector<Eigen::SparseVector<float>>* model_means,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>* model_bases,
        std::vector<Eigen::VectorXf>* projected_coeffs,
        std::vector<int>* sample_model_assign,
        std::vector<double>* outlier_gammas,
        std::vector<Eigen::Affine3f>* affine_transforms,
        std::vector<Eigen::Vector3f>* model_average_scales, Eigen::SparseMatrix<float, Eigen::ColMajor> *reconstructed_sample_weights,
        PCAOptions &options,
        const OptimizationOptions& optimize_options
        );

//bool WeightedPCADeflationOrthogonal(const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
//        const int component_num, const int max_iter,
//        Eigen::SparseVector<float> *mean_mat, Eigen::SparseVector<float> *mean_weight,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
//        Eigen::MatrixXf *coeff_mat,
//        const PCAOptions& options);

bool WeightedPCAProjectionOneSample(const Eigen::SparseVector<float> &sample,
        const Eigen::SparseVector<float> &weight,
        const Eigen::SparseVector<float>& mean_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& base_mat,
        Eigen::VectorXf* projected_coeff, float* squared_error);

bool WeightedPCAProjectionMultipleSamples(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
        const Eigen::SparseVector<float>& mean_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& base_mat,
        Eigen::MatrixXf* projected_coeffs, float* squared_errors
        );

bool OptimizePCACoeffAndCluster(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const std::vector<Eigen::SparseVector<float>> &cluster_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>> &cluster_bases,
        std::vector<Eigen::Affine3f>& sample_transforms,
        const PCAOptions& pca_options,
        std::vector<int> *cluster_assignment, std::vector<double> *outlier_gammas,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        std::vector<float> *cluster_assignment_error,
        std::vector<Eigen::Vector3f>* cluster_average_scales
        );

//bool OptimizeModelAndCoeff(const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
//        const std::vector<int>& model_assign_idx,
//        const int component_num, const int max_iter,
//        std::vector<Eigen::SparseVector<float>> * model_means,
//        std::vector<Eigen::SparseVector<float> > *model_mean_weight,
//        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>> *model_bases,
//        std::vector<Eigen::VectorXf> * projected_coeffs,
//        PCAOptions &options
//        );

//bool PCAReconstructionResult(
//        const std::vector<Eigen::SparseVector<float>>& model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>& model_bases,
//        const std::vector<Eigen::VectorXf>& projected_coeffs, /*#sample*/
//        const std::vector<int>& model_assign_idx,
//        std::vector<Eigen::SparseVector<float>>* reconstructed_samples
//        );

//bool OptimizeTransformAndScale(const TSDFHashing& scene_tsdf,
//        const std::vector<Eigen::SparseVector<float>>& model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>& model_bases,
//        const std::vector<Eigen::VectorXf>& projected_coeffs,
//        const std::vector<int>& model_assign_idx,
//        std::vector<Eigen::Affine3f> *affine_transforms, /*size: #samples*/
//        std::vector<Eigen::Vector3f>* model_average_scales, /*size: #models*/
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights,
//        const PCAOptions& options
//        );

bool OptimizeTransformAndScale(const TSDFHashing &scene_tsdf,
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        /*const */Eigen::SparseMatrix<float, Eigen::ColMajor>& reconstructed_sample_weights,
        const std::vector<int> &model_assign_idx,
        std::vector<Eigen::Affine3f> *affine_transforms,
        std::vector<Eigen::Vector3f> *model_average_scales,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights,
        PCAOptions &options);

///**
// * @brief ConvertDataMatrixToTSDFs
// * convert matrices to TSDFs. Used for debugging/output results
// * only the voxels with weights are assigned TSDF values
// * @param voxel_length
// * @param offset
// * @param max_dist_pos
// * @param max_dis_neg
// * @param voxel_bounding_box_size
// * @param data_mat
// * @param weight_mat
// * @param projected_tsdf_models
// * @return
// */
//bool ConvertDataMatrixToTSDFs(const float voxel_length,
//                              const Eigen::Vector3f& offset,
//                              const float max_dist_pos,
//                              const float max_dis_neg,
//                              const Eigen::Vector3i& voxel_bounding_box_size,
//                              const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat,
//                              const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
//                              std::vector<TSDFHashing::Ptr> *projected_tsdf_models
//                              );


///**
// * @brief ConvertDataMatrixToTSDFsNoWeight
// * convert matrices to TSDFs. Used for debugging/output results
// * all the voxels are assigned TSDF values
// * @param voxel_length
// * @param offset
// * @param max_dist_pos
// * @param max_dist_neg
// * @param voxel_bounding_box_size
// * @param data_mat
// * @param projected_tsdf_models
// * @return
// */
//bool ConvertDataMatrixToTSDFsNoWeight(const float voxel_length,
//                                      const Eigen::Vector3f &offset,
//                                      const float max_dist_pos, const float max_dist_neg,
//                                      const Eigen::Vector3i &voxel_bounding_box_size,
//                                      const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat,
//                                      std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models);

//bool WriteAffineTransformsAndTSDFs(const cpu_tsdf::TSDFHashing& scene_tsdf,
//                           const std::vector<Eigen::Affine3f>& affine_transforms, const PCAOptions &options);

bool WriteSampleClusters(const std::vector<int>& sample_model_assign,
                         const std::vector<Eigen::Vector3f>& model_average_scales,
                         const std::vector<double>& outlier_gammas, const std::vector<float> &cluster_assignment_error,
                         const std::string &save_path);

//void WPCASaveMeanTSDF(const Eigen::SparseVector<float>& mean_mat,
//                             const cpu_tsdf::TSDFHashing& mean_tsdf,
//                             const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
//                             const Eigen::Vector3i& boundingbox_size,
//                             const string& save_filepath);

//void WPCASaveMeanTSDF(
//        const Eigen::SparseVector<float>& mean_mat,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
//        const Eigen::Vector3i& boundingbox_size,
//        const float voxel_length,
//        const Eigen::Vector3f& offset,
//        const float max_dist_pos,
//        const float max_dist_neg,
//        const std::string& save_filepath);

//void WPCASaveMeanTSDF(const Eigen::SparseVector<float>& mean_mat,
//                      const cpu_tsdf::TSDFHashing& mean_tsdf,
//                      const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
//                      const Eigen::Vector3i& boundingbox_size,
//                      const std::string& save_filepath);

//void GetClusterSampleIdx(const std::vector<int> &model_assign_idx, const int model_number, std::vector<std::vector<int>>* cluster_sample_idx);

void ComputeModelAverageScales(
        const std::vector<int>& sample_model_assign,
        const std::vector<Eigen::Affine3f>& affine_transforms,
        const int model_num,
        std::vector<Eigen::Vector3f> *model_average_scales
        );

//bool PCAReconstructionResult(
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const std::vector<int> &model_assign_idx,
//        std::vector<Eigen::SparseVector<float> > *reconstructed_samples);

//void ConvertDataMatrixToDataVectors(const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat,
//        std::vector<Eigen::SparseVector<float> > *data_vec);

//void ConvertDataVectorsToDataMat(
//        const std::vector<Eigen::SparseVector<float>>& data_vec,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* data_mat
//        );

//bool CleanNoiseInSamples(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
//        const std::vector<int>& model_assign_idx,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* pweights, Eigen::SparseMatrix<float, Eigen::ColMajor> *valid_obs_weight_mat,
//        float counter_thresh, float pos_trunc, float neg_trunc, const PCAOptions &options);

//bool CleanNoiseInSamplesOneCluster(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* weights, Eigen::SparseVector<float> *valid_obs_positions,
//        float counter_thresh, float pos_trunc, float neg_trunc, const PCAOptions &options);

void ComputeWeightMat(const Eigen::SparseMatrix<float, Eigen::ColMajor> recon_sample_mat,
        const std::vector<int>& sample_cluster_assign,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights
        );

//bool PCAReconstructionResult(
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const std::vector<int> &model_assign_idx,
//        std::vector<Eigen::SparseVector<float> > *reconstructed_samples);

enum {
    OUTLIER_CLUSTER_INDEX = 100
};

}

