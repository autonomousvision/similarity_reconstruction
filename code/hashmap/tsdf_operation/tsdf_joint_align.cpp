#include "tsdf_joint_align.h"
#include <vector>
#include <string>
#include <unordered_set>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
//#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include "tsdf_operation/tsdf_align.h"
#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_representation/voxel_hashmap.h"
#include "common/utility/common_utility.h"
#include "common/utility/pcl_utility.h"
#include "common/utility/eigen_utility.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "utility/utility.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_clean.h"
#include "tsdf_operation/tsdf_slice.h"
#include "tsdf_operation/tsdf_transform.h"

using Eigen::ColMajor;
using Eigen::RowMajor;
using std::cout;
using std::endl;
using std::cerr;
using utility::min_vec3;
using utility::max_vec3;
using utility::flattenVertices;
using utility::CvVectorToEigenVector3;
using utility::EigenVectorToCvVector3;
namespace bfs = boost::filesystem;
using namespace std;

//////////////////////////////////////////////
namespace cpu_tsdf
{
//void WriteResultsForICCV(
//        TSDFHashing::ConstPtr original_scene,
//        const std::vector<Eigen::Affine3f> &affine_transforms,
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& reconstructed_sample_weights,
//        const std::vector<int> &model_assign_idx,
//        const PCAOptions& pca_options,
//        const string &output_dir, const string &file_prefix)
//{
//    PCAOptions my_option = pca_options;
//    static int call_count = 0;
//    char call_count_str[10] = {'\0'};
//    sprintf(call_count_str, "%d", call_count);

//    bfs::path write_dir(output_dir);
//    bfs::create_directory(write_dir);
//    bfs::path write_prefix = write_dir/file_prefix;

//    // save original scene mesh file
//    const float mesh_min_weight = my_option.min_model_weight;
//    cpu_tsdf::WriteTSDFModel(original_scene,
//                             (write_prefix.replace_extension("original_scene.ply")).string(),
//                             false, true, mesh_min_weight);
//    // save all the bounding boxes and cropped tsdfs
//    my_option.save_path = (write_prefix.replace_extension("obbs.ply")).string();
//    WriteAffineTransformsAndTSDFs(*original_scene, affine_transforms, my_option);
//    // save reconstructed samples
//    cpu_tsdf::TSDFGridInfo grid_info(*original_scene, my_option.boundingbox_size, mesh_min_weight);
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_original_pos;
//    cpu_tsdf::ReconstructTSDFsFromPCAOriginPos(
//            model_means,
//            model_bases,
//            projected_coeffs,
//            reconstructed_sample_weights,
//            model_assign_idx,
//            affine_transforms,
//            grid_info,
//            original_scene->voxel_length(),
//            original_scene->offset(),
//            &reconstructed_samples_original_pos);
//    cpu_tsdf::WriteTSDFModels(reconstructed_samples_original_pos,
//                              write_prefix.replace_extension("_frecon_tsdf.ply").string(),
//                              false, true, mesh_min_weight);
//    // save merged scene model
//    cpu_tsdf::TSDFHashing::Ptr copied_scene(new cpu_tsdf::TSDFHashing);
//    *copied_scene = *original_scene;
//    cpu_tsdf::MergeTSDFs(reconstructed_samples_original_pos, copied_scene.get());
//    cpu_tsdf::CleanTSDF(copied_scene, 100);
//    cpu_tsdf::WriteTSDFModel(copied_scene,
//                             (write_prefix.replace_extension("_merged_tsdf.ply")).string(),
//                             true, true, mesh_min_weight);
//}

//void ReconstructTSDFsFromPCAOriginPos(
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& recon_sample_weight,
//        const std::vector<int> &sample_model_assign,
//        const std::vector<Eigen::Affine3f>& affine_transforms,
//        const cpu_tsdf::TSDFGridInfo& grid_info,
//        const float voxel_length,
//        const Eigen::Vector3f& scene_tsdf_offset,
//        std::vector<cpu_tsdf::TSDFHashing::Ptr>* reconstructed_samples_original_pos)
//{
//    std::vector<Eigen::SparseVector<float>> recon_samples;
//    PCAReconstructionResult(model_means, model_bases, projected_coeffs, sample_model_assign, &recon_samples);
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
//    ConvertDataVectorsToTSDFsWithWeight(recon_samples,
//                                        cpu_tsdf::EigenMatToSparseVectors( recon_sample_weight), grid_info, &recon_tsdfs);
//    //cpu_tsdf::WriteTSDFModels(recon_tsdfs, "/home/dell/p01.ply", false, true, 0);
//    //    cpu_tsdf::WriteTSDFsFromMatWithWeight(
//    //                cpu_tsdf::SparseVectorsToEigenMat(recon_samples),
//    //                recon_sample_weight, grid_info, "/home/dell/p1.ply");
//    //ConvertDataVectorsToTSDFsNoWeight(recon_samples, grid_info, &recon_tsdfs);
//    TransformTSDFs(recon_tsdfs, affine_transforms, reconstructed_samples_original_pos, &voxel_length, &scene_tsdf_offset);
//    //cpu_tsdf::WriteTSDFModels(*reconstructed_samples_original_pos, "/home/dell/p2.ply", false, true, 0);
//}

//// Declarations for (PCA related) functions only used in this file
//void OrthogonalizeVector(const Eigen::SparseMatrix<float, Eigen::ColMajor>& base_mat, const int current_components, Eigen::VectorXf* base_vec, float *norm);

//bool InitializeDataMatrixFromTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
//                                   Eigen::SparseMatrix<float, Eigen::ColMajor> *centrlized_data_mat,
//                                   Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
//                                   Eigen::SparseVector<float> *mean_mat,
//                                   cpu_tsdf::TSDFHashing* tsdf_template,
//                                   Eigen::Vector3i* bounding_box);

//void ComputeCoeffInEStepOneVec(const Eigen::SparseMatrix<float, Eigen::ColMajor>& centralized_data_mat,
//                               const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
//                               const Eigen::VectorXf &base_vec,
//                               Eigen::VectorXf* coeff_vec);

//void ComputeBasisInMStepOneVec(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
//                               const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
//                               const Eigen::VectorXf &coeff_vec,
//                               Eigen::VectorXf *base_vec);

//void ComputeCoeffInEStep(const Eigen::SparseMatrix<float, Eigen::ColMajor>& centralized_data_mat,
//                         const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
//                         const Eigen::SparseMatrix<float, Eigen::ColMajor>& base_mat,
//                         Eigen::MatrixXf* coeff_mat);

//void ComputeBasisInMStep(const Eigen::SparseMatrix<float, Eigen::RowMajor>& centralized_data_mat_row_major,
//                         const Eigen::SparseMatrix<float, Eigen::RowMajor>& weight_mat_row_major,
//                         const Eigen::MatrixXf& coeff_mat,
//                         Eigen::SparseMatrix<float, Eigen::RowMajor>* base_mat_row_major
//                         );

//void InitialEstimate(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat,
//                     const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat,
//                     const int component_num,
//                     Eigen::SparseMatrix<float, Eigen::RowMajor> *base_mat_row_major, Eigen::MatrixXf *coeff_mat
//                     );

//bool WeightedPCAProjectionOneSample(
//        const Eigen::SparseVector<float> &sample,
//        const Eigen::SparseVector<float> &weight,
//        const Eigen::SparseVector<float> &mean_mat,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
//        Eigen::VectorXf *projected_coeff,
//        float *squared_error)
//{
//    using namespace Eigen;
//    Eigen::SparseVector<float> centralized_sample = sample - mean_mat;
//    Eigen::VectorXf dense_weight = weight;
//    if (base_mat.rows() * base_mat.cols() > 0)
//    {
////        Eigen::MatrixXf base_trans_weight = (base_mat.transpose() * weight).eval();
////        //    *projected_coeff =
////        //            (base_trans_weight * base_trans_weight.transpose()).jacobiSvd(ComputeThinU | ComputeThinV).
////        //            solve(Eigen::MatrixXf((base_mat.transpose() * centralized_sample).eval()));
////        *projected_coeff =
////                (base_trans_weight * base_trans_weight.transpose()).jacobiSvd(ComputeThinU | ComputeThinV).
////                solve(Eigen::MatrixXf((base_trans_weight * (weight.transpose() * centralized_sample)).eval()));
////        *squared_error =
////                (weight.cwiseProduct(centralized_sample - base_mat * projected_coeff->sparseView())).
////                norm();
//        // ||w^T(d - Bc)||_2
//        Eigen::MatrixXf A = (base_mat.transpose() * dense_weight.asDiagonal() * base_mat).eval();
//        *projected_coeff =
//                (A).jacobiSvd(ComputeThinU | ComputeThinV).
//                solve(Eigen::MatrixXf((base_mat.transpose() * dense_weight.asDiagonal() * centralized_sample).eval()));
//        Eigen::VectorXf diff = Eigen::VectorXf(centralized_sample) - (base_mat * (*projected_coeff));
//        *squared_error = diff.transpose() * dense_weight.asDiagonal() * diff;
//    }
//    else
//    {
//        projected_coeff->resize(0);
//        *squared_error = (centralized_sample.transpose() * dense_weight.asDiagonal() * centralized_sample).eval().coeff(0, 0);
//    }
//    return true;
//}

//bool WeightedPCAProjectionMultipleSamples(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
//        const Eigen::SparseVector<float> &mean_mat,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
//        Eigen::MatrixXf *projected_coeffs,
//        float *squared_errors)
//{
//    int sample_num = samples.cols();
//    int base_num = base_mat.cols();
//    if (base_mat.rows() * base_mat.cols() > 0)
//    {
//        projected_coeffs->resize(base_num, sample_num);
//    }
//    else
//    {
//        projected_coeffs->resize(0, 0);
//    }
//    for (int i = 0; i < sample_num; ++i)
//    {
//        Eigen::VectorXf current_coeff;
//        WeightedPCAProjectionOneSample(samples.col(i), weights.col(i), mean_mat, base_mat, &current_coeff, &(squared_errors[i]));
//        if (current_coeff.size() > 0)
//        {
//            projected_coeffs->col(i) = current_coeff;
//        }
//    }
//    return true;
//}

//bool OptimizePCACoeffAndCluster(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
//        const std::vector<Eigen::SparseVector<float>> &cluster_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>> &cluster_bases,
//        std::vector<Eigen::Affine3f>& sample_transforms,
//        const PCAOptions& pca_options,
//        std::vector<int> *cluster_assignment,
//        std::vector<double> *outlier_gammas,
//        std::vector<Eigen::VectorXf> *projected_coeffs,
//        std::vector<float> *cluster_assignment_error,
//        std::vector<Eigen::Vector3f>* cluster_average_scales
//        )
//{
//    using namespace std;
//    const int sample_num = samples.cols();
//    // const int feature_dim = samples.rows();
//    const int cluster_num = cluster_means.size();

//    std::vector<Eigen::VectorXf> min_cluster_coeffs(sample_num);
//    std::vector<int> min_cluster_idx(sample_num);
//    std::vector<float> min_cluster_error(sample_num, FLT_MAX);
//    Eigen::MatrixXf cur_projected_coeffs;
//    vector<float> squared_errors(sample_num);
//    for (int cluster_i = 0; cluster_i < cluster_num; ++cluster_i)
//    {
//        WeightedPCAProjectionMultipleSamples(samples, weights,
//                                             (cluster_means[cluster_i]), (cluster_bases[cluster_i]),
//                                             &cur_projected_coeffs, &(squared_errors[0]));
////        WeightedPCAProjectionMultipleSamples(samples, weights,
////                                             (samples.col(0)), (cluster_bases[cluster_i]),
////                                             &cur_projected_coeffs, &(squared_errors[0]));
//        for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//        {
//            Eigen::Matrix3f rot;
//            Eigen::Vector3f trans, scale;
//            utility::EigenAffine3fDecomposition(sample_transforms[sample_i], &rot, &scale, &trans);
//            squared_errors[sample_i] += (pca_options.lambda_scale_diff * ((scale - (*cluster_average_scales)[cluster_i]).squaredNorm()));
//            if (squared_errors[sample_i] < min_cluster_error[sample_i])
//            {
//                min_cluster_idx[sample_i] = cluster_i;
//                min_cluster_error[sample_i] = squared_errors[sample_i];
//                if (cur_projected_coeffs.size() > 0)
//                    min_cluster_coeffs[sample_i] = cur_projected_coeffs.col(sample_i);
//            }
//        }
//    }
//    *cluster_assignment = std::move(min_cluster_idx);
//    *projected_coeffs = std::move(min_cluster_coeffs);
//    *cluster_assignment_error = std::move(min_cluster_error);

//    // compute outlier_gammas
//    for (int i = 0; i < (*cluster_assignment).size(); ++i)
//    {
//        Eigen::Matrix3f rot;
//        Eigen::Vector3f trans, scale;
//        utility::EigenAffine3fDecomposition(sample_transforms[i], &rot, &scale, &trans);
//        double err_i = sqrt((*cluster_assignment_error)[i]) - pca_options.lambda_outlier;
//        cout << "outlier: " << i << " " << err_i << endl;
//        cout << "outlier detail: " << i << " " << (pca_options.lambda_scale_diff * ((scale - (*cluster_average_scales)[(*cluster_assignment)[i]]).squaredNorm())) << " " << squared_errors[i] - (pca_options.lambda_scale_diff * ((scale - (*cluster_average_scales)[(*cluster_assignment)[i]]).squaredNorm())) << endl;
//        err_i = err_i >= 0 ? err_i:0;
//        (*outlier_gammas)[i] = err_i;
//    }
//    //char ch;
//    //cin >> ch;
//    ComputeModelAverageScales(*cluster_assignment, sample_transforms, cluster_num, cluster_average_scales);
//    return true;
//}

//void GetClusterSampleIdx(const std::vector<int> &model_assign_idx, const int model_number, std::vector<std::vector<int>>* cluster_sample_idx)
//{
//    const int sample_number = model_assign_idx.size();
//    (*cluster_sample_idx).resize(model_number);
//    for (int i = 0; i < sample_number; ++i)
//    {
//        (*cluster_sample_idx)[model_assign_idx[i]].push_back(i);
//    }
//    return;
//}

//bool CleanNoiseInSamplesOneCluster(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* weights,
//        Eigen::SparseVector<float>* valid_obs_positions,
//        float counter_thresh, float pos_trunc, float neg_trunc, const PCAOptions& options)
//{
//    const float abs_neg_trunc = fabs(neg_trunc);
//    const float abs_pos_trunc = fabs(pos_trunc);
//    const float weight_thresh = 0.00;
//    if (samples.size() == 0 || weights->size() == 0) return false;
//    Eigen::VectorXf obs_counter = Eigen::VectorXf::Zero(weights->rows());
//    cout << "go through mat" << endl;
//    for (int k=0; k<weights->outerSize(); ++k)
//      for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator it(*weights,k); it; ++it)
//      {
//          if (it.value() > weight_thresh)
//          {
//              float cur_dist = samples.coeff(it.row(), it.col());
//              //obs_counter(it.row()) += 1 - cur_dist/(cur_dist >= 0 ? abs_pos_trunc : abs_neg_trunc);
//              obs_counter(it.row()) += 1;
//          }
//      }
//    cout << "begin setting zeros" << endl;
//    // second time
//    valid_obs_positions->resize(samples.rows());
//    for (int k=0; k<weights->outerSize(); ++k)
//    {
//      for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator it(*weights,k); it; ++it)
//      {
//          if (obs_counter(it.row()) < counter_thresh)
//          {
//              // cout << "clean " << it.row() << " " << it.col() <<endl;
//              it.valueRef() = 0;
//          }
//      }
//    }  // end k
//    for (int r = 0; r < obs_counter.size(); ++r)
//    {
//        if (obs_counter(r) >= counter_thresh)
//        {
//            valid_obs_positions->coeffRef(r) = 1.0 * cpu_tsdf::TSDFHashing::getVoxelMaxWeight();
//        }
//    }
//    cpu_tsdf::Write3DArrayMatlab(obs_counter, options.boundingbox_size, "obs_counter", options.save_path);
//    cout << "finished setting zeros" << endl;
////    char ch;
////    cin >> ch;
//    weights->prune(weight_thresh, 1);
//    return true;
//}

//bool CleanNoiseInSamples(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
//        const std::vector<int>& model_assign_idx,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* pweights,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* valid_obs_weight_mat,
//        float counter_thresh, float pos_trunc, float neg_trunc, const PCAOptions& options)
//{
//    cout << "begin clean noise" << endl;
//    Eigen::SparseMatrix<float, Eigen::ColMajor>& weights = *pweights;
//    valid_obs_weight_mat->resize(pweights->rows(), pweights->cols());
//    // const int sample_number = samples.cols();
//    const int feature_dim = samples.rows();
//    const int model_number = *(std::max_element(model_assign_idx.begin(), model_assign_idx.end())) + 1;
//    vector<vector<int>> cluster_sample_idx;
//    GetClusterSampleIdx(model_assign_idx, model_number, &cluster_sample_idx);
//    for (int i = 0; i < model_number; ++i)
//    {
//        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_samples(feature_dim, cluster_sample_idx[i].size());
//        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_weights(feature_dim, cluster_sample_idx[i].size());
//        Eigen::SparseVector<float> cur_valid_obs(feature_dim);
//        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
//        {
//            cur_samples.col(j) = samples.col(cluster_sample_idx[i][j]);
//            cur_weights.col(j) = weights.col(cluster_sample_idx[i][j]);
//        }
////        char ch;
////        cin >> ch;
//        cout << "clean for cluster " << i << endl;
//        CleanNoiseInSamplesOneCluster(
//                    cur_samples, &cur_weights, &cur_valid_obs, counter_thresh,
//                    pos_trunc, neg_trunc, options);
//        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
//        {
//            weights.col(cluster_sample_idx[i][j]) = cur_weights.col(j);
//            valid_obs_weight_mat->col(cluster_sample_idx[i][j]) = cur_valid_obs;
//        }
//    }  // end for i
//    weights.prune(0, 1);
//    cout << "finished clean noise" << endl;
//    return true;
//}


//bool OptimizeModelAndCoeff(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
//        const std::vector<int> &model_assign_idx,
//        const int component_num, const int max_iter,
//        std::vector<Eigen::SparseVector<float> > *model_means,
//        std::vector<Eigen::SparseVector<float> > *model_mean_weight,
//        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
//        std::vector<Eigen::VectorXf> *projected_coeffs,
//        PCAOptions &options)
//{
//    using namespace std;
//    const int sample_number = samples.cols();
//    const int feature_dim = samples.rows();
//    const int model_number = *(std::max_element(model_assign_idx.begin(), model_assign_idx.end())) + 1;
//    vector<vector<int>> cluster_sample_idx;
//    GetClusterSampleIdx(model_assign_idx, model_number, &cluster_sample_idx);
//    model_means->resize(model_number);
//    model_bases->resize(model_number);
//    projected_coeffs->resize(sample_number);
//    model_mean_weight->resize(model_number);
//    for (int i = 0; i < model_number; ++i)
//    {
//        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_samples(feature_dim, cluster_sample_idx[i].size());
//        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_weights(feature_dim, cluster_sample_idx[i].size());
//        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
//        {
//            cur_samples.col(j) = samples.col(cluster_sample_idx[i][j]);
//            cur_weights.col(j) = weights.col(cluster_sample_idx[i][j]);
//        }
//        Eigen::MatrixXf current_projected_coeffs;
//        const std::string prev_save_path = options.save_path;
//        bfs::path bfs_prefix(options.save_path);
//        options.save_path = (bfs_prefix.parent_path()/bfs_prefix.stem()).string() + "_cluster_" + boost::lexical_cast<string>(i) + ".ply";
//        WeightedPCADeflationOrthogonal(cur_samples, cur_weights, component_num, max_iter,
//                                       &((*model_means)[i]),
//                                       &((*model_mean_weight)[i]),
//                                       &((*model_bases)[i]),
//                                       &current_projected_coeffs,
//                                       options);
//        options.save_path = prev_save_path;
//        if (component_num == 0) continue;
//        assert(current_projected_coeffs.cols() == cluster_sample_idx[i].size());
//        for (int j = 0; j < current_projected_coeffs.cols(); ++j)
//        {
//            (*projected_coeffs)[cluster_sample_idx[i][j]] = current_projected_coeffs.col(j);
//        }
//    }  // end for i
//    return true;
//}

bool OptimizeTransformAndScale(
        const TSDFHashing &scene_tsdf,
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        /*const */Eigen::SparseMatrix<float, Eigen::ColMajor>& reconstructed_sample_weights,
        const std::vector<int> &model_assign_idx,
        const std::vector<double> & outlier_gammas,
        std::vector<Eigen::Affine3f> *affine_transforms,
        std::vector<Eigen::Vector3f> *model_average_scales,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights,
        PCAOptions &options)
{
    const int sample_num = model_assign_idx.size();
    const int model_num = model_means.size();
    // 1. model to model_reconstructed_samples
    std::vector<Eigen::SparseVector<float>> reconstructed_samples;
    PCAReconstructionResult(model_means, model_bases, projected_coeffs, model_assign_idx, &reconstructed_samples);
    //utility::WriteEigenMatrix(Eigen::MatrixXf(reconstructed_samples[0]), options.save_path + "_debug_house_model_tobealigned.txt");
    cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, options.boundingbox_size, options.min_model_weight);
    ExtractSamplesFromAffineTransform(
            scene_tsdf,
            *affine_transforms, /*size: #samples*/
            grid_info,
            samples,
            weights
            );
    std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_tsdf(sample_num);
    ConvertDataVectorsToTSDFsWithWeight(reconstructed_samples, *weights, options, &reconstructed_samples_tsdf);
    // ConvertDataVectorsToTSDFsWithWeight(reconstructed_samples, reconstructed_sample_weights, options, &reconstructed_samples_tsdf);
    cpu_tsdf::WriteTSDFModels(reconstructed_samples_tsdf, options.save_path + "_debug_house_model_tobealigned1.txt", false, true, 0);
    std::vector<const TSDFHashing*> ptr_reconstructed_samples_tsdf(sample_num);
    for (int i = 0; i < sample_num; ++i) ptr_reconstructed_samples_tsdf[i] = reconstructed_samples_tsdf[i].get();
    // 2. optimize
#if 1
    OptimizeTransformAndScalesImplRobustLoss(
                scene_tsdf,
                ptr_reconstructed_samples_tsdf,
                model_assign_idx,
                outlier_gammas,
                options,
                options.save_path,
                affine_transforms, /*#samples, input and output*/
                model_average_scales /*#model/clusters, input and output*/
                );
//    OptimizeTransformAndScalesImplRobustLoss(
//                scene_tsdf,
//                ptr_reconstructed_samples_tsdf,
//                model_assign_idx,
//                options,
//                options.save_path,
//                affine_transforms, /*#samples, input and output*/
//                model_average_scales /*#model/clusters, input and output*/
//                );
#endif
    // 3. get sample data vectors and weight vectors after optimization
    //ExtractSamplesFromAffineTransform(
    //        scene_tsdf,
    //        *affine_transforms, /*size: #samples*/
    //        grid_info,
    //        samples,
    //        weights
    //        );
    ExtractSamplesFromAffineTransform(
            scene_tsdf,
            *affine_transforms, /*size: #samples*/
            grid_info,
            samples,
            weights
            );
    return true;
}

void ComputeWeightMat(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> recon_sample_mat,
        const std::vector<int>& sample_cluster_assign,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights
        )
{
    reconstructed_sample_weights->resize(recon_sample_mat.rows(), recon_sample_mat.cols());
    reconstructed_sample_weights->setZero();
    //float factor = 1;
    for (int k=0; k<(recon_sample_mat).outerSize(); ++k)
    {
//        if (sample_cluster_assign[k] == 1)
//        {
//            factor = 1000;
//        }
//        else
//        {
//            factor = 1;
//        }
      for (Eigen::SparseMatrix<float>::InnerIterator it((recon_sample_mat),k); it; ++it)
      {
          if (it.value() != 0)
          {
              reconstructed_sample_weights->coeffRef(it.row(), it.col()) = cpu_tsdf::TSDFHashing::getVoxelMaxWeight();
          }
      }
    }
}

//bool JointClusteringAndModelLearningOld(const TSDFHashing &scene_tsdf,
//                                     std::vector<Eigen::SparseVector<float> > *model_means,
//                                     std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
//                                     std::vector<Eigen::VectorXf> *projected_coeffs,
//                                     std::vector<int> *sample_model_assign,
//                                     std::vector<double> *outlier_gammas,
//                                     std::vector<Eigen::Affine3f> *affine_transforms,
//                                     std::vector<Eigen::Vector3f> *model_average_scales,
//                                     Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights,
//                                     PCAOptions &options, const OptimizationOptions &optimize_options)
//{
//    const std::string options_init_save_path = options.save_path;
//    for (int i = 0; i < optimize_options.opt_max_iter; ++i)
//    {
//        bfs::path prefix(options_init_save_path);
//        bfs::path write_dir(prefix.parent_path()/(std::string("iteration_") + boost::lexical_cast<string>(i)));
//        bfs::create_directories(write_dir);
//
//        bfs::path write_dir_block1(write_dir/"block1_alignment");
//        bfs::create_directories(write_dir_block1);
//        string cur_save_path = (write_dir_block1/prefix.stem()).string();
//        options.save_path = (write_dir_block1/prefix.stem()).string() + "_TransformScale";
//
//        // block 1
//        Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
//        Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
//        float scene_voxel_length = (scene_tsdf.voxel_length());
//        OptimizeTransformAndScale(scene_tsdf,
//                                  *model_means, *model_bases, *projected_coeffs, *reconstructed_sample_weights,
//                                  *sample_model_assign,
//                                  *outlier_gammas,
//                                  affine_transforms,
//                                  model_average_scales,
//                                  &samples,
//                                  &weights,
//                                  options);
//        options.save_path = cur_save_path + "_TransformScale_EndSave";
//        WriteAffineTransformsAndTSDFs(scene_tsdf, *affine_transforms, options);
//        {
//            cpu_tsdf::TSDFHashing::Ptr current_scene(new TSDFHashing);
//            bfs::path parent_dir = bfs::path(options.save_path).parent_path();
//            bfs::path iccv_save_dir = parent_dir/"iccv_result";
//            *current_scene = scene_tsdf;
//            WriteResultsForICCV(current_scene, *affine_transforms,
//                                *model_means, *model_bases, *projected_coeffs,
//                                *reconstructed_sample_weights, *sample_model_assign, options, iccv_save_dir.string(), "res.ply");
//
//            // TSDFGridInfo tsdf_info(scene_tsdf, options.boundingbox_size, 0);
//            //cpu_tsdf::WriteTSDFsFromMatsWithWeight_Matlab(samples, weights, tsdf_info,
//            //                                               options.save_path + "_TransformScale.mat");
//            //vector<cpu_tsdf::OrientedBoundingBox> obbs;
//            //cpu_tsdf::AffinesToOrientedBBs(*affine_transforms, &obbs);
//            //cpu_tsdf::WriteOrientedBoundingBoxes(options.save_path + "_cur_obb_info.txt", obbs, *sample_model_assign);
//        }
//
//        // block 2
//        bfs::path write_dir_block2(write_dir/"block2_pca");
//        bfs::create_directories(write_dir_block2);
//        options.save_path = (write_dir_block2/prefix.stem()).string() + "_ModelCoeffPCA";
//
//        Eigen::SparseMatrix<float, Eigen::ColMajor> valid_obs_weight_mat;
//        float pos_trunc, neg_trunc;
//        scene_tsdf.getDepthTruncationLimits(pos_trunc, neg_trunc);
//        //weights is modified
//        CleanNoiseInSamples(
//                    samples, *sample_model_assign, &weights, &valid_obs_weight_mat,
//                    optimize_options.noise_clean_counter_thresh, pos_trunc, neg_trunc, options);
//        // clean isolated parts
//        CleanTSDFSampleMatrix(scene_tsdf, samples, options.boundingbox_size, optimize_options.compo_thresh, &weights, -1, 2);
//        std::vector<Eigen::SparseVector<float>> model_mean_weights;
//        OptimizeModelAndCoeff(samples, weights, *sample_model_assign, *outlier_gammas,
//                              optimize_options.PCA_component_num, optimize_options.PCA_max_iter,
//                              model_means, &model_mean_weights, model_bases, projected_coeffs,
//                              options);
//        // recompute reconstructed model's weights
//        std::vector<Eigen::SparseVector<float>> recon_samples;
//        PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
//        Eigen::SparseMatrix<float, Eigen::ColMajor> recon_sample_mat = cpu_tsdf::SparseVectorsToEigenMat(recon_samples);
//        *reconstructed_sample_weights = valid_obs_weight_mat;
//        // ComputeWeightMat(recon_sample_mat, *sample_model_assign, reconstructed_sample_weights);
//        // clean noise in reconstructed samples
//        {
////            options.save_path = cur_save_path + "_ModelCoeffPCA_tsdf_denoise.mat";
////            CleanNoiseInSamples(recon_sample_mat, *sample_model_assign, &valid_obs_weight_mat, reconstructed_sample_weights,
////                                optimize_options.noise_clean_counter_thresh, pos_trunc, neg_trunc, options);
//            CleanTSDFSampleMatrix(scene_tsdf, recon_sample_mat,
//                                  options.boundingbox_size, optimize_options.compo_thresh,
//                                  reconstructed_sample_weights, -1, 2);
//        }
//        // save
//        {
//            cpu_tsdf::TSDFHashing::Ptr current_scene(new TSDFHashing);
//            bfs::path parent_dir = bfs::path(options.save_path).parent_path();
//            bfs::path iccv_save_dir = parent_dir/"iccv_result";
//            *current_scene = scene_tsdf;
//            WriteResultsForICCV(current_scene, *affine_transforms,
//                                *model_means, *model_bases, *projected_coeffs,
//                                *reconstructed_sample_weights, *sample_model_assign, options, iccv_save_dir.string(), "res.ply");
//
//            // options.save_path = cur_save_path;
//            TSDFGridInfo tsdf_info(scene_tsdf, options.boundingbox_size, 0);
//            cpu_tsdf::WriteTSDFsFromMatWithWeight(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//                                                  options.save_path + "_ModelCoeffPCA_reconstructed_withweight.ply");
//            cpu_tsdf::WriteTSDFsFromMatsWithWeight_Matlab(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//                                                           options.save_path + "_ModelCoeffPCA_reconstructed_withweight.mat");
//            cpu_tsdf::WriteTSDFsFromMatNoWeight(recon_sample_mat, tsdf_info,
//                                                  options.save_path + "_ModelCoeffPCA_reconstructed_noweight.ply");
//
//        }
//       // block 3
//       if (1)
//       {
//           bfs::path write_dir_block3(write_dir/"block3_cluster");
//           bfs::create_directories(write_dir_block3);
//           string cur_save_path = (write_dir_block3/prefix.stem()).string();
//           options.save_path = (write_dir_block3/prefix.stem()).string() + "_Cluster";
//
//           cout << "do cluster" << endl;
//           std::vector<float> cluster_assignment_error;
//           OptimizePCACoeffAndCluster(samples, weights,
//                                      *model_means, *model_bases,
//                                      *affine_transforms,
//                                      options,
//                                      sample_model_assign,
//                                      projected_coeffs,
//                                      &cluster_assignment_error,
//                                      model_average_scales);
//           WriteSampleClusters(*sample_model_assign,  *model_average_scales, options.save_path + "_cluster.txt");
//           {
//               std::vector<Eigen::SparseVector<float>> recon_samples;
//               options.save_path = cur_save_path + "_Cluster_recon";
//               PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
//               std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
//               // ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, options, &recon_tsdfs);
//               ConvertDataVectorsToTSDFsNoWeight(recon_samples, options, &recon_tsdfs);
//               std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_recon_tsdfs;
//               WriteTSDFModels(recon_tsdfs, options.save_path + "_canonical.ply", false, true, options.min_model_weight);
//               TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
//               WriteTSDFModels(transformed_recon_tsdfs, options.save_path + "_transformed.ply", false, true, options.min_model_weight);
//
//               options.save_path = cur_save_path + "_Cluster_recon_with_weights";
//               ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, options, &recon_tsdfs);
//               WriteTSDFModels(recon_tsdfs, options.save_path + "_canonical.ply", false, true, options.min_model_weight);
//               transformed_recon_tsdfs.clear();
//               TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
//               WriteTSDFModels(transformed_recon_tsdfs, options.save_path + "_transformed.ply", false, true, options.min_model_weight);
//               //options.save_path = options_old_save_path;
//           }
//       }
//
//       // put the not-well reconstructed samples
//
//
//
//if (1)
//{
//#if 1
//       options.save_path = cur_save_path;
//        using namespace std;
//        cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, options.boundingbox_size, options.min_model_weight);
//        std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_original_pos;
//        Eigen::SparseMatrix<float, Eigen::ColMajor> original_weighmat = *reconstructed_sample_weights;
//        float kkfactors[1]={0.0};
//        for (int kki = 0; kki < 1; ++kki)
//        {
//            float kkfactor = kkfactors[kki];
////            for (int kk = 0; kk < reconstructed_sample_weights->cols(); ++kk)
////            {
////                if ((*sample_model_assign)[kk] == 1)
////                {
////                    reconstructed_sample_weights->col(kk) = original_weighmat.col(kk) * kkfactor;
////                }
////            }
//            cpu_tsdf::ReconstructTSDFsFromPCAOriginPos(
//                        *model_means,
//                        *model_bases,
//                        *projected_coeffs,
//                        *reconstructed_sample_weights,
//                        *sample_model_assign,
//                        *affine_transforms,
//                        grid_info,
//                        scene_tsdf.voxel_length(),
//                        scene_tsdf.offset(),
//                        &reconstructed_samples_original_pos);
//            for (int i = 0; i < reconstructed_samples_original_pos.size(); ++i)
//            {
//                cout << "cleaning " << i <<"th model" << endl;
//                cpu_tsdf::CleanTSDF(reconstructed_samples_original_pos[i], optimize_options.compo_thresh);
//            }
//            cpu_tsdf::WriteTSDFModels(reconstructed_samples_original_pos,
//                                      (bfs::path(options.save_path).replace_extension
//                                       (string("_recon_single_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i)  + "_" + utility::double2str(kkfactor)+ ".ply")).string(),
//                                      false, true, options.min_model_weight);
//            cout << "begin merging... " << endl;
//            cpu_tsdf::TSDFHashing::Ptr cur_scene(new TSDFHashing);
//            *cur_scene = scene_tsdf;
////            cpu_tsdf::WriteTSDFModel(cur_scene,
////                                     (bfs::path(options.save_path).replace_extension
////                                      (string("_beforemerge_beforeclean_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
////                                     false, true, options.min_model_weight);
//            cpu_tsdf::CleanTSDF(cur_scene, 100);
////            cpu_tsdf::WriteTSDFModel(cur_scene,
////                                     (bfs::path(options.save_path).replace_extension
////                                      (string("_beforemerge_1stclean_100_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
////                                     false, true, options.min_model_weight);
//
//             // substitute
////                    for (int ki = 0; ki < affine_transforms->size(); ++ki)
////                    {
////                        cpu_tsdf::OrientedBoundingBox obb;
////                        cpu_tsdf::AffineToOrientedBB((*affine_transforms)[ki], &obb);
////                        cpu_tsdf::CleanTSDFPart(cur_scene.get(), obb);
////                        //cpu_tsdf::WriteTSDFModel(cur_scene, string("/home/dell/scene_tsdf1_") + utility::int2str(ki) + ".ply", false, true, 0);
////                        cpu_tsdf::MergeTSDFNearestNeighbor(*(reconstructed_samples_original_pos[ki]), cur_scene.get());
////                        cpu_tsdf::WriteTSDFModel(cur_scene, string("/home/dell/scene_tsdf2_") + utility::int2str(ki) + ".ply", false, true, 0);
////                    }
//
//            std::vector<Eigen::Affine3f> car_affines;
//            for (int i = 0; i < affine_transforms->size(); ++i)
//            {
//                //if ((*sample_model_assign)[i] == 1)
//                {
//                    car_affines.push_back((*affine_transforms)[i]);
//                }
//            }
//            cpu_tsdf::ScaleTSDFParts(cur_scene.get(), car_affines, kkfactor);
//            cpu_tsdf::MergeTSDFs(reconstructed_samples_original_pos, cur_scene.get());
//            cur_scene->DisplayInfo();
////            cpu_tsdf::WriteTSDFModel(cur_scene,
////                                     (bfs::path(options.save_path).replace_extension
////                                      (string("_aftermerge_1stclean_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
////                                     false, true, options.min_model_weight);
//            cpu_tsdf::CleanTSDF(cur_scene, 100);
//            cpu_tsdf::WriteTSDFModel(cur_scene,
//                                     (bfs::path(options.save_path).replace_extension
//                                      (string("_raftermerge_2ndclean_100") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
//                                     true, true, options.min_model_weight);
//        } // for kki
//#endif
//}
//    }  // optimization iteration
//    options.save_path = options_init_save_path;
//    return true;
//}

void GetNonOutlierSamples(const std::vector<int>& sample_model_assign,
                          const std::vector<Eigen::Affine3f>& affine_transforms,
                          std::vector<int>* non_outlier_sample_idx,
                          std::vector<int>* nonout_sample_model_assign,
                          std::vector<Eigen::Affine3f>* nonout_affine_transforms)
{
    non_outlier_sample_idx->clear();
    for (int cur_idx = 0; cur_idx < sample_model_assign.size(); ++cur_idx)
    {
        if (sample_model_assign[cur_idx] != OUTLIER_CLUSTER_INDEX)
            non_outlier_sample_idx->push_back(cur_idx);
    }
    nonout_sample_model_assign->clear();
    std::copy_if(sample_model_assign.begin(), sample_model_assign.end(), std::back_inserter(*nonout_sample_model_assign), [](const int val) {return val != OUTLIER_CLUSTER_INDEX;});
    nonout_affine_transforms->clear();
    for (int i = 0; i < non_outlier_sample_idx->size(); ++i)
    {
        nonout_affine_transforms->push_back(affine_transforms[(*non_outlier_sample_idx)[i]]);
    }
}

void GetNonOutlierSamplesWithGamma(
        const std::vector<int>& sample_model_assign,
        const std::vector<Eigen::Affine3f>& affine_transforms,
        const std::vector<Eigen::VectorXf>& projected_coeffs,
        const std::vector<float>& gammas, // non zero gamma indicates outliers
        std::vector<int>* non_outlier_sample_idx,
        std::vector<int>* nonout_sample_model_assign,
        std::vector<Eigen::Affine3f>* nonout_affine_transforms,
        std::vector<Eigen::VectorXf> *nonout_projected_coeffs
        )
{
    non_outlier_sample_idx->clear();
    nonout_sample_model_assign->clear();
    nonout_affine_transforms->clear();
    for (int cur_idx = 0; cur_idx < sample_model_assign.size(); ++cur_idx)
    {
        if (gammas[cur_idx] > 0)
        {
            non_outlier_sample_idx->push_back(cur_idx);
            nonout_sample_model_assign->push_back(sample_model_assign[cur_idx]);
            nonout_affine_transforms->push_back(affine_transforms[cur_idx]);
            nonout_projected_coeffs->push_back(projected_coeffs[cur_idx]);
        }
    }
}

void RecomposeSamplesWithOutliers(const std::vector<Eigen::Affine3f>& nonout_affines,
                                  const std::vector<Eigen::VectorXf>& nonout_proj_coeffs,
                                  const std::vector<int>& nonout_sample_idx,
                                  const std::vector<Eigen::Affine3f>& origin_affines,
                                  const std::vector<Eigen::VectorXf>& origin_proj,
                                  std::vector<Eigen::Affine3f>* composed_affines,
                                  std::vector<Eigen::VectorXf>* composed_proj
                                  )
{
    const int tot_sample = origin_affines.size();
    int nonout_sample_idx_cnt = 0;
    for (int  i = 0; i < tot_sample; ++i)
    {
        if (i == nonout_sample_idx[nonout_sample_idx_cnt])
        {
            composed_affines->push_back(nonout_affines[i]);
            composed_proj->push_back(nonout_proj_coeffs[i]);
            nonout_sample_idx_cnt++;
        }
        else
        {
            composed_affines->push_back(origin_affines[i]);
            composed_proj->push_back(origin_proj[i]);
        }
    }
}

//void OptimizeAlignmentAndModels(
//        const cpu_tsdf::TSDFHashing &scene_tsdf,
//        std::vector<Eigen::SparseVector<float>> *model_means,
//        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>> *model_bases,
//        std::vector<Eigen::VectorXf> *projected_coeffs,
//        std::vector<int> *sample_model_assign_no_outlier,
//        std::vector<Eigen::Affine3f> *affine_transforms,
//        std::vector<Eigen::Vector3f> *model_average_scales,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* weights,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights,
//        PCAOptions &options, const OptimizationOptions &optimize_options
//        )
//{
//    string cur_save_path = options.save_path;
//    // block 1
//    float scene_voxel_length = (scene_tsdf.voxel_length());
//    OptimizeTransformAndScale(scene_tsdf,
//                              *model_means, *model_bases, *projected_coeffs, *reconstructed_sample_weights,
//                              *sample_model_assign_no_outlier,
//                              affine_transforms,
//                              model_average_scales,
//                              samples,
//                              weights,
//                              options);
//    options.save_path = cur_save_path + "_TransformScale_EndSave";
//    WriteAffineTransformsAndTSDFs(scene_tsdf, *affine_transforms, options);
//    {
//        TSDFGridInfo tsdf_info(scene_tsdf, options.boundingbox_size, 0);
//        cpu_tsdf::WriteTSDFsFromMatsWithWeight_Matlab(*samples, *weights, tsdf_info,
//                                                       options.save_path + "_TransformScale.mat");
//        vector<cpu_tsdf::OrientedBoundingBox> obbs;
//        cpu_tsdf::AffinesToOrientedBBs(*affine_transforms, &obbs);
//        cpu_tsdf::WriteOrientedBoundingBoxes(options.save_path + "_cur_obb_info.txt", obbs, *sample_model_assign_no_outlier);
//    }
//
//    // block 2
//    Eigen::SparseMatrix<float, Eigen::ColMajor> valid_obs_weight_mat;
//    options.save_path = cur_save_path + "_ModelCoeffPCA";
//    float pos_trunc, neg_trunc;
//    scene_tsdf.getDepthTruncationLimits(pos_trunc, neg_trunc);
//    //weights is modified
//    CleanNoiseInSamples(
//                *samples, *sample_model_assign_no_outlier, weights, &valid_obs_weight_mat,
//                optimize_options.noise_clean_counter_thresh, pos_trunc, neg_trunc, options);
//    // clean isolated parts
//    CleanTSDFSampleMatrix(scene_tsdf, *samples, options.boundingbox_size, optimize_options.compo_thresh, weights, -1, 2);
//    std::vector<Eigen::SparseVector<float>> model_mean_weights;
//    OptimizeModelAndCoeff(*samples, *weights, *sample_model_assign_no_outlier,
//                          optimize_options.PCA_component_num, optimize_options.PCA_max_iter,
//                          model_means, &model_mean_weights, model_bases, projected_coeffs,
//                          options);
//    // recompute reconstructed model's weights
//    std::vector<Eigen::SparseVector<float>> recon_samples;
//    PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign_no_outlier, &recon_samples);
//    Eigen::SparseMatrix<float, Eigen::ColMajor> recon_sample_mat = cpu_tsdf::SparseVectorsToEigenMat(recon_samples);
//    *reconstructed_sample_weights = valid_obs_weight_mat;
//    // ComputeWeightMat(recon_sample_mat, *sample_model_assign, reconstructed_sample_weights);
//    // clean noise in reconstructed samples
//    {
//        options.save_path = cur_save_path + "_ModelCoeffPCA_tsdf_denoise.mat";
////            CleanNoiseInSamples(recon_sample_mat, *sample_model_assign, &valid_obs_weight_mat, reconstructed_sample_weights,
////                                optimize_options.noise_clean_counter_thresh, pos_trunc, neg_trunc, options);
//        CleanTSDFSampleMatrix(scene_tsdf, recon_sample_mat,
//                              options.boundingbox_size, optimize_options.compo_thresh,
//                              reconstructed_sample_weights, -1, 2);
//    }
//    // save
//    {
//        options.save_path = cur_save_path;
//        TSDFGridInfo tsdf_info(scene_tsdf, options.boundingbox_size, 0);
//        cpu_tsdf::WriteTSDFsFromMatWithWeight(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//                                              options.save_path + "_ModelCoeffPCA_reconstructed_withweight.ply");
//        cpu_tsdf::WriteTSDFsFromMatsWithWeight_Matlab(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//                                                       options.save_path + "_ModelCoeffPCA_reconstructed_withweight.mat");
//        cpu_tsdf::WriteTSDFsFromMatNoWeight(recon_sample_mat, tsdf_info,
//                                              options.save_path + "_ModelCoeffPCA_reconstructed_noweight.ply");
//
//    }
//}



// model_bases: number: M
//bool JointClusteringAndModelLearning(const TSDFHashing &scene_tsdf,
//                                     std::vector<Eigen::SparseVector<float> > *model_means,
//                                     std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
//                                     std::vector<Eigen::VectorXf> *projected_coeffs,
//                                     std::vector<int> *sample_model_assign,
//                                     std::vector<double> *outlier_gammas,
//                                     std::vector<Eigen::Affine3f> *affine_transforms,
//                                     std::vector<Eigen::Vector3f> *model_average_scales,
//                                     Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights,
//                                     PCAOptions &options, const OptimizationOptions &optimize_options)
//{
//    const std::string options_init_save_path = options.save_path;
//    for (int i = 0; i < optimize_options.opt_max_iter; ++i)
//    {
//        bfs::path prefix(options_init_save_path);
//        bfs::path write_dir(prefix.parent_path()/(std::string("iteration_") + boost::lexical_cast<string>(i)));
//        bfs::create_directories(write_dir);

//        bfs::path write_dir_block1(write_dir/"block1_alignment");
//        bfs::create_directories(write_dir_block1);
//        string cur_save_path = (write_dir_block1/prefix.stem()).string();
//        options.save_path = (write_dir_block1/prefix.stem()).string() + "_TransformScale";

//        // block 1
//        Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
//        Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
//        float scene_voxel_length = (scene_tsdf.voxel_length());
//        OptimizeTransformAndScale(scene_tsdf,
//                                  *model_means, *model_bases, *projected_coeffs, *reconstructed_sample_weights,
//                                  *sample_model_assign,
//                                  *outlier_gammas,
//                                  affine_transforms,
//                                  model_average_scales,
//                                  &samples,
//                                  &weights,
//                                  options);
//        options.save_path = cur_save_path + "_TransformScale_EndSave";
//        // WriteAffineTransformsAndTSDFs(scene_tsdf, *affine_transforms, options);
//        {
//           // cpu_tsdf::TSDFHashing::Ptr current_scene(new TSDFHashing);
//           // bfs::path parent_dir = bfs::path(options.save_path).parent_path();
//           // bfs::path iccv_save_dir = parent_dir/"iccv_result";
//           // *current_scene = scene_tsdf;
//           // WriteResultsForICCV(current_scene, *affine_transforms,
//           //                     *model_means, *model_bases, *projected_coeffs,
//           //                     *reconstructed_sample_weights, *sample_model_assign, options, iccv_save_dir.string(), "res.ply");
//            TSDFGridInfo tsdf_info(scene_tsdf, options.boundingbox_size, 0);
//            cpu_tsdf::WriteTSDFsFromMatWithWeight_Matlab(samples, weights, tsdf_info,
//                                                           options.save_path + "_TransformScale.mat");
//            //vector<cpu_tsdf::OrientedBoundingBox> obbs;
//            //cpu_tsdf::AffinesToOrientedBBs(*affine_transforms, &obbs);
//            //cpu_tsdf::WriteOrientedBoundingBoxes(options.save_path + "_cur_obb_info.txt", obbs, *sample_model_assign);
//        }

//        // block 2
//        bfs::path write_dir_block2(write_dir/"block2_pca");
//        bfs::create_directories(write_dir_block2);
//        cur_save_path =  (write_dir_block2/prefix.stem()).string();
//        options.save_path = (write_dir_block2/prefix.stem()).string() + "_ModelCoeffPCA";

//        Eigen::SparseMatrix<float, Eigen::ColMajor> valid_obs_weight_mat;
//        float pos_trunc, neg_trunc;
//        scene_tsdf.getDepthTruncationLimits(pos_trunc, neg_trunc);
//        // weights is modified
//        // weights: cleaned weight; valid_obs_weight_mat: 0-max_weight binary matrix
//        // not considering outliers for observation counting
//        // only consider observations appeared in multiple samples
//        CleanNoiseInSamples(
//                    samples, *sample_model_assign, *outlier_gammas, &weights, &valid_obs_weight_mat,
//                    optimize_options.noise_clean_counter_thresh, pos_trunc, neg_trunc, options);
//        // clean isolated parts
//        // clean noise also for outliers, as the outliers are also extracted from affine transforms in the previous step
//        CleanTSDFSampleMatrix(scene_tsdf, samples, options.boundingbox_size, optimize_options.compo_thresh, &weights, -1, 2);
//        std::vector<Eigen::SparseVector<float>> model_mean_weights;
//        cpu_tsdf::WriteMatrixMatlab(options.save_path + "_ModelCoeffPCAData.mat", "valid_weights", (valid_obs_weight_mat));
//        cpu_tsdf::WriteMatrixMatlab(options.save_path + "_ModelCoeffPCAData.mat", "data", (samples));
//        cpu_tsdf::WriteMatrixMatlab(options.save_path + "_ModelCoeffPCAData.mat", "weight", (weights));
//        OptimizeModelAndCoeff(samples, weights, *sample_model_assign,
//                             *outlier_gammas,
//                              optimize_options.PCA_component_num, optimize_options.PCA_max_iter,
//                              model_means, &model_mean_weights, model_bases, projected_coeffs,
//                              options);
//        // recompute reconstructed model's weights
//        std::vector<Eigen::SparseVector<float>> recon_samples;
//        PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
//        Eigen::SparseMatrix<float, Eigen::ColMajor> recon_sample_mat = cpu_tsdf::SparseVectorsToEigenMat(recon_samples);
//        *reconstructed_sample_weights = valid_obs_weight_mat;
//        // ComputeWeightMat(recon_sample_mat, *sample_model_assign, reconstructed_sample_weights);
//        // clean noise in reconstructed samples
//        {
////            options.save_path = cur_save_path + "_ModelCoeffPCA_tsdf_denoise.mat";
////            CleanNoiseInSamples(recon_sample_mat, *sample_model_assign, &valid_obs_weight_mat, reconstructed_sample_weights,
////                                optimize_options.noise_clean_counter_thresh, pos_trunc, neg_trunc, options);
//            //CleanTSDFSampleMatrix(scene_tsdf, recon_sample_mat,
//            //                      options.boundingbox_size, optimize_options.compo_thresh,
//            //                      reconstructed_sample_weights, -1, 2);
//        }
//        // save
//        {
//            cpu_tsdf::TSDFHashing::Ptr current_scene(new TSDFHashing);
//            bfs::path parent_dir = bfs::path(options.save_path).parent_path();
//            bfs::path iccv_save_dir = parent_dir/"iccv_result";
//            *current_scene = scene_tsdf;
//            WriteResultsForICCV(current_scene, *affine_transforms,
//                                *model_means, *model_bases, *projected_coeffs,
//                                *reconstructed_sample_weights, *sample_model_assign, options, iccv_save_dir.string(), "res.ply");

//            options.save_path = cur_save_path;
//            TSDFGridInfo tsdf_info(scene_tsdf, options.boundingbox_size, 0);
//            cpu_tsdf::WriteTSDFsFromMatWithWeight(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//                                                  options.save_path + "_ModelCoeffPCA_reconstructed_withweight.ply");
//            cpu_tsdf::WriteTSDFsFromMatWithWeight_Matlab(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//                                                           options.save_path + "_ModelCoeffPCA_reconstructed_withweight.mat");
//            //cpu_tsdf::WriteTSDFsFromMatsWithWeight_Matlab(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//            //                                               options.save_path + "_ModelCoeffPCA_reconstructed_withweight.mat");
//            cpu_tsdf::WriteTSDFsFromMatNoWeight(recon_sample_mat, tsdf_info,
//                                                  options.save_path + "_ModelCoeffPCA_reconstructed_noweight.ply");

//        }
//       // block 3
//       if (1)
//       {
//           bfs::path write_dir_block3(write_dir/"block3_cluster");
//           bfs::create_directories(write_dir_block3);
//           cur_save_path = (write_dir_block3/prefix.stem()).string();
//           options.save_path = (write_dir_block3/prefix.stem()).string() + "_Cluster";

//           cout << "do cluster" << endl;
//           std::vector<float> cluster_assignment_error;
//           OptimizePCACoeffAndCluster(samples, weights,
//                                      *model_means, *model_bases,
//                                      *affine_transforms,
//                                      options,
//                                      sample_model_assign,
//                                      outlier_gammas,
//                                      projected_coeffs,
//                                      &cluster_assignment_error,
//                                      model_average_scales);
//           WriteSampleClusters(*sample_model_assign,  *model_average_scales, *outlier_gammas, cluster_assignment_error, options.save_path + "_cluster.txt");
//           {
//               std::vector<Eigen::SparseVector<float>> recon_samples;
//               options.save_path = cur_save_path + "_Cluster_recon";
//               PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
//               std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
//               // ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, options, &recon_tsdfs);
//               ConvertDataVectorsToTSDFsNoWeight(recon_samples, options, &recon_tsdfs);
//               std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_recon_tsdfs;
//               WriteTSDFModels(recon_tsdfs, options.save_path + "_canonical.ply", false, true, options.min_model_weight);
//               TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
//               WriteTSDFModels(transformed_recon_tsdfs, options.save_path + "_transformed.ply", false, true, options.min_model_weight);

//               options.save_path = cur_save_path + "_Cluster_recon_with_weights";
//               ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, options, &recon_tsdfs);
//               WriteTSDFModels(recon_tsdfs, options.save_path + "_canonical.ply", false, true, options.min_model_weight);
//               transformed_recon_tsdfs.clear();
//               TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
//               WriteTSDFModels(transformed_recon_tsdfs, options.save_path + "_transformed.ply", false, true, options.min_model_weight);
//               //options.save_path = options_old_save_path;
//           }
//       }

//       // put the not-well reconstructed samples



//if (1)
//{
//#if 1
//       options.save_path = cur_save_path;
//        using namespace std;
//        cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, options.boundingbox_size, options.min_model_weight);
//        std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_original_pos;
//        Eigen::SparseMatrix<float, Eigen::ColMajor> original_weighmat = *reconstructed_sample_weights;
//        float kkfactors[1]={0.5};
//        for (int kki = 0; kki < 1; ++kki)
//        {
//            float kkfactor = kkfactors[kki];
////            for (int kk = 0; kk < reconstructed_sample_weights->cols(); ++kk)
////            {
////                if ((*sample_model_assign)[kk] == 1)
////                {
////                    reconstructed_sample_weights->col(kk) = original_weighmat.col(kk) * kkfactor;
////                }
////            }
//            cpu_tsdf::ReconstructTSDFsFromPCAOriginPos(
//                        *model_means,
//                        *model_bases,
//                        *projected_coeffs,
//                        *reconstructed_sample_weights,
//                        *sample_model_assign,
//                        *affine_transforms,
//                        grid_info,
//                        scene_tsdf.voxel_length(),
//                        scene_tsdf.offset(),
//                        &reconstructed_samples_original_pos);
//            for (int i = 0; i < reconstructed_samples_original_pos.size(); ++i)
//            {
//                cout << "cleaning " << i <<"th model" << endl;
//                cpu_tsdf::CleanTSDF(reconstructed_samples_original_pos[i], optimize_options.compo_thresh);
//            }
//            cpu_tsdf::WriteTSDFModels(reconstructed_samples_original_pos,
//                                      (bfs::path(options.save_path).replace_extension
//                                       (string("_recon_single_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i)  + "_" + utility::double2str(kkfactor)+ ".ply")).string(),
//                                      false, true, options.min_model_weight, *outlier_gammas);
//            cout << "begin merging... " << endl;
//            cpu_tsdf::TSDFHashing::Ptr cur_scene(new TSDFHashing);
//            *cur_scene = scene_tsdf;
////            cpu_tsdf::WriteTSDFModel(cur_scene,
////                                     (bfs::path(options.save_path).replace_extension
////                                      (string("_beforemerge_beforeclean_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
////                                     false, true, options.min_model_weight);
//            cpu_tsdf::CleanTSDF(cur_scene, 100);
////            cpu_tsdf::WriteTSDFModel(cur_scene,
////                                     (bfs::path(options.save_path).replace_extension
////                                      (string("_beforemerge_1stclean_100_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
////                                     false, true, options.min_model_weight);

//             // substitute
////                    for (int ki = 0; ki < affine_transforms->size(); ++ki)
////                    {
////                        cpu_tsdf::OrientedBoundingBox obb;
////                        cpu_tsdf::AffineToOrientedBB((*affine_transforms)[ki], &obb);
////                        cpu_tsdf::CleanTSDFPart(cur_scene.get(), obb);
////                        //cpu_tsdf::WriteTSDFModel(cur_scene, string("/home/dell/scene_tsdf1_") + utility::int2str(ki) + ".ply", false, true, 0);
////                        cpu_tsdf::MergeTSDFNearestNeighbor(*(reconstructed_samples_original_pos[ki]), cur_scene.get());
////                        cpu_tsdf::WriteTSDFModel(cur_scene, string("/home/dell/scene_tsdf2_") + utility::int2str(ki) + ".ply", false, true, 0);
////                    }

//            std::vector<Eigen::Affine3f> car_affines;
//            for (int i = 0; i < affine_transforms->size(); ++i)
//            {
//                //if ((*sample_model_assign)[i] == 1)
//                {
//                    car_affines.push_back((*affine_transforms)[i]);
//                }
//            }
//            cpu_tsdf::ScaleTSDFParts(cur_scene.get(), car_affines, kkfactor);
//            for (int i = 0; i < reconstructed_samples_original_pos.size(); ++i)
//            {
//                if ((*outlier_gammas)[i] > 1e-5)
//                {
//                    reconstructed_samples_original_pos[i].reset();
//                }
//            }
//            reconstructed_samples_original_pos.erase(std::remove_if(reconstructed_samples_original_pos.begin(), reconstructed_samples_original_pos.end(), [](const  cpu_tsdf::TSDFHashing::Ptr& ptr){
//                return !bool(ptr);
//            }), reconstructed_samples_original_pos.end());
//            cpu_tsdf::MergeTSDFs(reconstructed_samples_original_pos, cur_scene.get());
//            cur_scene->DisplayInfo();
////            cpu_tsdf::WriteTSDFModel(cur_scene,
////                                     (bfs::path(options.save_path).replace_extension
////                                      (string("_aftermerge_1stclean_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
////                                     false, true, options.min_model_weight);
//            cpu_tsdf::CleanTSDF(cur_scene, 100);
//            cpu_tsdf::WriteTSDFModel(cur_scene,
//                                     (bfs::path(options.save_path).replace_extension
//                                      (string("_raftermerge_2ndclean_100") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
//                                     true, true, options.min_model_weight);
//        } // for kki
//#endif
//}
//    }  // optimization iteration
//    options.save_path = options_init_save_path;
//    return true;
//}

//bool ExtractSamplesFromAffineTransform(
//        const TSDFHashing &scene_tsdf,
//        const std::vector<Eigen::Affine3f> &affine_transforms,
//        const PCAOptions &options,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
//{
//    const int sample_num = affine_transforms.size();
//    const int feature_dim = options.boundingbox_size[0] * options.boundingbox_size[1] * options.boundingbox_size[2];
//    samples->resize(feature_dim, sample_num);
//    samples->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    weights->resize(feature_dim, sample_num);
//    weights->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    for (int i = 0; i < sample_num; ++i)
//    {
//        Eigen::SparseVector<float> sample(feature_dim);
//        Eigen::SparseVector<float> weight(feature_dim);
//        ExtractOneSampleFromAffineTransform(scene_tsdf, affine_transforms[i], options,
//                                            &sample,
//                                            &weight);
//        Eigen::SparseVector<float>::InnerIterator it_s(sample);
//        for (Eigen::SparseVector<float>::InnerIterator it(weight); it && it_s; ++it, ++it_s)
//        {
//            CHECK(it_s.index() == it.index());
//            samples->insert(it.index(), i) = it_s.value();
//            weights->insert(it.index(), i) = it.value();
//        }

//        ///////////////////////////////////////////////
////        Eigen::Matrix3f test_r;
////        Eigen::Vector3f test_scale;
////        Eigen::Vector3f test_trans;
////        utility::EigenAffine3fDecomposition(
////                    affine_transforms[i],
////                    &test_r,
////                    &test_scale,
////                    &test_trans);
////        cpu_tsdf::TSDFHashing::Ptr cur_tsdf(new cpu_tsdf::TSDFHashing);
////        ConvertDataVectorToTSDFWithWeight(
////        sample,
////        weight,
////        options,
////        cur_tsdf.get());
////        bfs::path output_path(options.save_path);
////        string save_path = (output_path.parent_path()/output_path.stem()).string() + "_check_affine_conversion_" + boost::lexical_cast<string>(i) + ".ply";
////        cpu_tsdf::WriteTSDFModel(cur_tsdf, save_path, false, true, options.min_model_weight);
//        ///////////////////////////////////////////////
//    }
//    return true;
//}

//bool ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf,
//                                         const Eigen::Affine3f &affine_transform,
//                                         const PCAOptions &options,
//                                         Eigen::SparseVector<float> *sample,
//                                         Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset;
//    for (int x = 0; x < options.boundingbox_size[0]; ++x)
//        for (int y = 0; y < options.boundingbox_size[1]; ++y)
//            for (int z = 0; z < options.boundingbox_size[2]; ++z)
//            {
//                Eigen::Vector3f current_world_point = offset;
//                current_world_point[0] += options.voxel_length *  x;
//                current_world_point[1] += options.voxel_length *  y;
//                current_world_point[2] += options.voxel_length *  z;
//                Eigen::Vector3f transformed_world_point = affine_transform * current_world_point;
//                float cur_d, cur_w;
//                if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) && /*cur_w > 0*/ cur_w > options.min_model_weight)
//                {
//                    int current_index = z +
//                            (y + x * options.boundingbox_size[1]) * options.boundingbox_size[2];
//                    sample->coeffRef(current_index) = cur_d;
//                    weight->coeffRef(current_index) = cur_w;
//                }
//            }
//    return true;
//}

//void ConvertOrientedBoundingboxToAffineTransforms(const std::vector<OrientedBoundingBox> &obbs, std::vector<Eigen::Affine3f> *transforms)
//{
//    transforms->resize(obbs.size());
//    for (int i = 0; i < obbs.size(); ++i)
//    {
//        ConvertOneOBBToAffineTransform(obbs[i], &((*transforms)[i]));
//    }
//}



//void ConvertOneOBBToAffineTransform(const OrientedBoundingBox &obb, Eigen::Affine3f *transform)
//{
//    Eigen::Vector3f scale3d = obb.bb_sidelengths;
//    Eigen::Matrix3f rotation = obb.bb_orientation;
//    Eigen::Vector3f offset = obb.BoxCenter();
//    Eigen::Matrix4f transform_mat = Eigen::Matrix4f::Zero();
//    transform_mat.block<3, 3>(0, 0) = rotation * scale3d.asDiagonal();
//    transform_mat.block<3, 1>(0, 3) = offset;
//    transform_mat.coeffRef(3, 3) = 1;
//    transform->matrix() = transform_mat;
//}

//void ComputeModelAverageScales(
//        const std::vector<int>& sample_model_assign,
//        const std::vector<Eigen::Affine3f>& affine_transforms,
//        const int model_num,
//        std::vector<Eigen::Vector3f> *model_average_scales
//        )
//{
//    const int sample_num = sample_model_assign.size();
//    model_average_scales->resize(model_num);
//    for (int i = 0; i < model_average_scales->size(); ++i)
//    {
//        (*model_average_scales)[i].setZero();
//    }
//    std::vector<int> sample_model_count(model_num, 0);
//    for (int i = 0; i < sample_num; ++i)
//    {
//        Eigen::Matrix3f rot;
//        Eigen::Vector3f scale, trans;
//        utility::EigenAffine3fDecomposition(affine_transforms[i], &rot, &scale, &trans);
//        int cur_model = (sample_model_assign)[i];
//        (*model_average_scales)[cur_model] += scale;
//        sample_model_count[cur_model]++;
//    }
//    for (int i = 0; i < model_num; ++i)
//    {
//        (*model_average_scales)[i] /= (float)sample_model_count[i];
//    }
//}

//void InitialClusteringAndModelLearningInitialDebugging2(
//        const TSDFHashing &scene_tsdf,
//        const std::vector<OrientedBoundingBox> &detected_boxes,
//        PCAOptions *pca_options,
//        const OptimizationOptions &optimize_options,
//        std::vector<Eigen::SparseVector<float> > *model_means,
//        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
//        std::vector<Eigen::VectorXf> *projected_coeffs,
//        std::vector<int> *sample_model_assign,
//        std::vector<Eigen::Affine3f> *affine_transforms,
//        std::vector<Eigen::Vector3f> *model_average_scales,
//        Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights)
//{
//    const int sample_num = detected_boxes.size();
//    const int model_num = *(max_element(sample_model_assign->begin(), sample_model_assign->end())) + 1;
//    // pca option
//    pca_options->voxel_length =  1.0/50;
//    int bb_size = round(1/pca_options->voxel_length) + 1;
//    pca_options->boundingbox_size = Eigen::Vector3i(bb_size, bb_size, bb_size);
//    pca_options->offset = Eigen::Vector3f(-0.5, -0.5, -0.5);
//    scene_tsdf.getDepthTruncationLimits((pca_options->max_dist_pos), (pca_options->max_dist_neg));
//    const int feat_dim = pca_options->boundingbox_size[0] * pca_options->boundingbox_size[1] * pca_options->boundingbox_size[2];

//    // affine_transforms
//    OBBsToAffines(detected_boxes, affine_transforms);

//    // model_average_scales
//    ComputeModelAverageScales(*sample_model_assign, *affine_transforms, model_num, model_average_scales);

//    // sample_model_assign
//    // leave it as is
//    // for (int i = 0; i < model_average_scales->size(); ++i) (*model_average_scales)[i] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);

//    // initialize models
//    //    std::vector<Eigen::SparseVector<float>>* model_means,
//    //    std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>* model_bases,
//    //    std::vector<Eigen::VectorXf>* projected_coeffs,
//    Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
//    Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
//    cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, pca_options->boundingbox_size, pca_options->min_model_weight);
//    grid_info.max_dist_pos(20 * 0.2);
//    grid_info.max_dist_neg(-20 * 0.2);
//    ExtractSamplesFromAffineTransform(
//                scene_tsdf,
//                *affine_transforms,
//                grid_info,
//                &samples,
//                &weights);

//    ////////////////////////////////////////////////////////save
//    const std::string old_save_path = pca_options->save_path;
//    bfs::path prefix(pca_options->save_path);
//    pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_initial_affine_vis.ply";
//    // WriteAffineTransformsAndTSDFs(scene_tsdf, *affine_transforms, *pca_options);
//    pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_initial_samples_vis.ply";
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
//    std::vector<Eigen::SparseVector<float>> vec_samples;
//    ConvertDataMatrixToDataVectors(samples, &vec_samples);
//    // cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, pca_options->boundingbox_size, pca_options->min_model_weight);
//    ConvertDataVectorsToTSDFsWithWeight(vec_samples, weights, grid_info, &recon_tsdfs);
//    // std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_recon_tsdfs;
//    // float scene_voxel_length = (scene_tsdf.voxel_length());
//    // TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
//    WriteTSDFModels(recon_tsdfs, pca_options->save_path, false, true, pca_options->min_model_weight);
//    ////////////////////////////////////////////////////////save
//    ////////////////////////////////////////////////////////
//    /// using mean as initial template
//    std::vector<Eigen::SparseVector<float>> model_mean_weights;
//    std::vector<double> cur_gammas(samples.cols(), 0);
//    OptimizeModelAndCoeff(
//                samples,
//                weights,
//                *sample_model_assign,
//                cur_gammas,
//                0, 0,
//                model_means,
//                &model_mean_weights,
//                model_bases,
//                projected_coeffs,
//                *pca_options
//                );
//    vector<vector<int>> cluster_sample_idx;
//    GetClusterSampleIdx(*sample_model_assign, vector<double>(), model_num, &cluster_sample_idx);
//    reconstructed_sample_weights->resize(feat_dim, sample_num);
//    for (int model_i = 0; model_i < model_num; ++model_i)
//    {
//        //int cur_model_sampleidx = rand()%cluster_sample_idx[model_i].size();
//        // the first sample is set as the mean
//        //int cur_model_sampleidx = cluster_sample_idx[model_i][0];
//        //(*model_means)[model_i] = Eigen::SparseVector<float>((samples).col(cur_model_sampleidx));
//        for (int sample_i = 0; sample_i < cluster_sample_idx[model_i].size(); ++sample_i)
//        {
//            reconstructed_sample_weights->col(cluster_sample_idx[model_i][sample_i]) = model_mean_weights[model_i];
//        }
//    }
//    ////////////////////////////////////////////////////////
//    /// using the first sample as the mean
//    //vector<vector<int>> cluster_sample_idx;
//    //GetClusterSampleIdx(*sample_model_assign, vector<double>(), model_num, &cluster_sample_idx);
//    //model_means->resize(model_num);
//    //for (int model_i = 0; model_i < model_num; ++model_i)
//    //{
//    //    //int cur_model_sampleidx = rand()%cluster_sample_idx[model_i].size();
//    //    // the first sample is set as the mean
//    //    int cur_model_sampleidx = cluster_sample_idx[model_i][0];
//    //    (*model_means)[model_i] = Eigen::SparseVector<float>((samples).col(cur_model_sampleidx));
//    //}
//    //model_bases->resize(model_num);
//    //projected_coeffs->resize(sample_num);
//    //////////////////////////////////////////////////////////
//    //std::vector<Eigen::SparseVector<float>> recon_samples;
//    //PCAReconstructionResult((*model_means), *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
//    //float pos_trunc, neg_trunc;
//    //scene_tsdf.getDepthTruncationLimits(pos_trunc, neg_trunc);
//    //// clean noise that are observed too few times
//    //// should be not making a real difference here
//    //CleanNoiseInSamples(cpu_tsdf::SparseVectorsToEigenMat(recon_samples), *sample_model_assign, std::vector<double>(),
//    //                    &weights, reconstructed_sample_weights, optimize_options.noise_clean_counter_thresh, pos_trunc, neg_trunc, *pca_options);
//    ////// ComputeWeightMat(cpu_tsdf::SparseVectorsToEigenMat(recon_samples), *sample_model_assign, reconstructed_sample_weights);
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////save
//    pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_intial_cluster.txt";
//    std::vector<double> tmp_gammas(sample_model_assign->size(), 0);
//    WriteSampleClusters(*sample_model_assign, *model_average_scales, tmp_gammas, std::vector<float>(), pca_options->save_path);
//    pca_options->save_path = old_save_path;

//    // for iccv result
//    cpu_tsdf::TSDFHashing::Ptr current_scene(new TSDFHashing);
//    bfs::path parent_dir = bfs::path(pca_options->save_path).parent_path();
//    bfs::path iccv_save_dir = parent_dir/"iccv_result_initial";
//    *current_scene = scene_tsdf;
//   // cpu_tsdf::WriteTSDFModel(current_scene,
//   //                           (prefix.parent_path()/prefix.stem()).string() + "_initial_scene_withbin.ply",
//   //                          true, true, pca_options->min_model_weight);
//    //WriteResultsForICCV(current_scene, *affine_transforms,
//    //                    *model_means, *model_bases, *projected_coeffs,
//    //                    *reconstructed_sample_weights, *sample_model_assign, *pca_options, iccv_save_dir.string(), "initial_res.ply");
//    WriteResultsForICCV(current_scene, *affine_transforms,
//                        *model_means, *model_bases, *projected_coeffs,
//                        *reconstructed_sample_weights, *sample_model_assign, *pca_options, iccv_save_dir.string(), "initial_res.ply");
//    cout << "init finished. "<< endl;
//}

//void InitialClusteringAndModelLearningInitialDebugging(
//        const TSDFHashing &scene_tsdf,
//        const std::vector<OrientedBoundingBox> &detected_boxes,
//        PCAOptions *pca_options,
//        std::vector<Eigen::SparseVector<float> > *model_means,
//        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
//        std::vector<Eigen::VectorXf> *projected_coeffs,
//        std::vector<int> *sample_model_assign,
//        std::vector<Eigen::Affine3f> *affine_transforms,
//        std::vector<Eigen::Vector3f> *model_average_scales)
//{
//    const int sample_num = detected_boxes.size();
//    const int model_num = *(max_element(sample_model_assign->begin(), sample_model_assign->end())) + 1;
//    // pca option
//    pca_options->voxel_length =  1.0/50;
//    int bb_size = round(1/pca_options->voxel_length) + 1;
//    pca_options->boundingbox_size = Eigen::Vector3i(bb_size, bb_size, bb_size);
//    pca_options->offset = Eigen::Vector3f(-0.5, -0.5, -0.5);
//    scene_tsdf.getDepthTruncationLimits((pca_options->max_dist_pos), (pca_options->max_dist_neg));

//    // affine_transforms
//    OBBsToAffines(detected_boxes, affine_transforms);

//    // model_average_scales
//    ComputeModelAverageScales(*sample_model_assign, *affine_transforms, model_num, model_average_scales);

//    // sample_model_assign
//    // leave it as is
//    // for (int i = 0; i < model_average_scales->size(); ++i) (*model_average_scales)[i] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);

//    // initialize models
//    //    std::vector<Eigen::SparseVector<float>>* model_means,
//    //    std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>>* model_bases,
//    //    std::vector<Eigen::VectorXf>* projected_coeffs,
//    Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
//    Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
//    cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, pca_options->boundingbox_size, pca_options->min_model_weight);
//    ExtractSamplesFromAffineTransform(
//                scene_tsdf,
//                *affine_transforms,
//                grid_info,
//                &samples,
//                &weights);

//    ////////////////////////////////////////////////////////save
//    const std::string old_save_path = pca_options->save_path;
//    bfs::path prefix(pca_options->save_path);
//    pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_initial_affine_vis.ply";
//    WriteAffineTransformsAndTSDFs(scene_tsdf, *affine_transforms, *pca_options);
//    pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_initial_samples_vis.ply";
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
//    std::vector<Eigen::SparseVector<float>> vec_samples;
//    ConvertDataMatrixToDataVectors(samples, &vec_samples);
//    ConvertDataVectorsToTSDFsNoWeight(vec_samples, *pca_options, &recon_tsdfs);
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_recon_tsdfs;
//    float scene_voxel_length = (scene_tsdf.voxel_length());
//    TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
//    WriteTSDFModels(recon_tsdfs, pca_options->save_path, false, true, pca_options->min_model_weight);
//    ////////////////////////////////////////////////////////save

//    //pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_intial_mean_model.ply";
//    /////////////////////////////////////////////////////////

//    std::vector<Eigen::SparseVector<float>> model_mean_weights;
//    std::vector<double> cur_gammas(samples.cols(), 0);
//    OptimizeModelAndCoeff(
//                samples,
//                weights,
//                *sample_model_assign,
//                cur_gammas,
//                0, 0,
//                model_means,
//                &model_mean_weights,
//                model_bases,
//                projected_coeffs,
//                *pca_options
//                );
//        ////////////////////////////////////////save
////        {
////            pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_initial_model_with_weight.ply";
////            std::vector<Eigen::SparseVector<float>> recon_samples;
////            PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
////            std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
////            Eigen::SparseMatrix<float, Eigen::ColMajor> tmp_weights(model_means->size(), sample_num);
////            for (int i = 0; i < sample_num; ++i)
////            {
////                tmp_weights.col(i) = model_mean_weights[0];
////            }
////            ConvertDataVectorsToTSDFsWithWeight(recon_samples, tmp_weights, *pca_options, &recon_tsdfs);
////            // ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, *pca_options, &recon_tsdfs);
////            // ConvertDataVectorsToTSDFsNoWeight(recon_samples, options, &recon_tsdfs);
////            std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_recon_tsdfs;
////            float scene_voxel_length = (scene_tsdf.voxel_length());
//////            TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
//////            WriteTSDFModels(transformed_recon_tsdfs, pca_options->save_path, false, true, pca_options->min_model_weight);
////            WriteTSDFModels(recon_tsdfs, pca_options->save_path, false, true, pca_options->min_model_weight);
////        }
//        {
//            pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_initial_model_without_weight.ply";
//            std::vector<Eigen::SparseVector<float>> recon_samples;
//            PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
//            std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
//            // ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, *pca_options, &recon_tsdfs);
//            ConvertDataVectorsToTSDFsNoWeight(recon_samples, *pca_options, &recon_tsdfs);
//            std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_recon_tsdfs;
//            float scene_voxel_length = (scene_tsdf.voxel_length());
////            TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
////            WriteTSDFModels(transformed_recon_tsdfs, pca_options->save_path, false, true, pca_options->min_model_weight);
//            WriteTSDFModels(recon_tsdfs, pca_options->save_path, false, true, pca_options->min_model_weight);

//            std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_tsdfs;
//            std::vector<Eigen::Affine3f> scale_trans(samples.cols());
//            for (int i = 0; i < samples.cols(); ++i)
//            {
//                Eigen::Matrix4f cur_trans_mat = Eigen::Matrix4f::Identity();
//                cur_trans_mat.coeffRef(0, 0) = (*model_average_scales)[(*sample_model_assign)[i]][0];
//                cur_trans_mat.coeffRef(1, 1) = (*model_average_scales)[(*sample_model_assign)[i]][1];
//                cur_trans_mat.coeffRef(2, 2) = (*model_average_scales)[(*sample_model_assign)[i]][2];
//                scale_trans[i] = cur_trans_mat;
//            }
//            TransformTSDFs(recon_tsdfs, scale_trans, &transformed_tsdfs, &scene_voxel_length);
//            WriteTSDFModels(transformed_tsdfs, pca_options->save_path + "canonical_size_mean.ply", false, true, pca_options->min_model_weight);
//        }
//        /// ////////////////////////////////////save
//    /////////////////////////////////////////////////////
//    vector<vector<int>> cluster_sample_idx;
//    GetClusterSampleIdx(*sample_model_assign,  vector<double>(), model_num, &cluster_sample_idx);
//    model_means->resize(model_num);
//    for (int model_i = 0; model_i < model_num; ++model_i)
//    {
//        //int cur_model_sampleidx = rand()%cluster_sample_idx[model_i].size();
//        int cur_model_sampleidx = cluster_sample_idx[model_i][0];
//        (*model_means)[model_i] = Eigen::SparseVector<float>((samples).col(cur_model_sampleidx));
//    }
//    model_bases->resize(model_num);
//    projected_coeffs->resize(sample_num);
///////////////////////////////////////////////////////////
//    pca_options->save_path = (prefix.parent_path()/prefix.stem()).string() + "_intial_cluster.txt";
//    std::vector<double> tmp_gammas(sample_model_assign->size(), 0);
//    WriteSampleClusters(*sample_model_assign, *model_average_scales, tmp_gammas, std::vector<float>(), pca_options->save_path);
//    pca_options->save_path = old_save_path;
//}

//bool ConvertDataVectorToTSDFNoWeight(
//        const Eigen::SparseVector<float> &tsdf_data_vec,
//        const PCAOptions &options,
//        TSDFHashing *tsdf)
//{
//    //tsdf->CopyHashParametersFrom(options.tsdf_for_parameters);
//    tsdf->Init(options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg);
//    const int size_yz = options.boundingbox_size[1] * options.boundingbox_size[2];
//    const Eigen::Vector3i voxel_bounding_box_size = options.boundingbox_size;
//    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_data_vec); it; ++it)
//    {
//        int data_dim_idx = it.index();
//        float dist = it.value();
//        cv::Vec3i pos;
//        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
//        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
//        pos[0] = data_dim_idx / size_yz;
//        tsdf->AddObservation(pos, dist, tsdf->getVoxelMaxWeight(), cv::Vec3b(255, 255, 255));
//    }
//    return true;
//}

//bool ConvertDataVectorsToTSDFsNoWeight(
//        const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec,
//        const PCAOptions &options,
//        std::vector<TSDFHashing::Ptr> *tsdfs)
//{
//    tsdfs->resize(tsdf_data_vec.size());
//    for(int i = 0; i < tsdf_data_vec.size(); ++i)
//    {
//        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
//        ConvertDataVectorToTSDFNoWeight(
//                    tsdf_data_vec[i],
//                    options,
//                    ((*tsdfs)[i].get()));
//    }
//    return true;
//}

//bool ConvertDataVectorToTSDFWithWeight(
//        const Eigen::SparseVector<float> &tsdf_data_vec,
//        const Eigen::SparseVector<float> &tsdf_weight_vec,
//        const PCAOptions &options, TSDFHashing *tsdf)
//{
//    //tsdf->CopyHashParametersFrom(options.tsdf_for_parameters);
//    tsdf->Init(options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg);
//    const int size_yz = options.boundingbox_size[1] * options.boundingbox_size[2];
//    const Eigen::Vector3i voxel_bounding_box_size = options.boundingbox_size;
//    // const float ratio = options.ratio_original_voxel_length_to_unit_cube_vlength;
//    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_weight_vec); it; ++it)
//    {
//        int data_dim_idx = it.index();
//        const float weight = it.value();
//        if ( weight < options.min_model_weight ) continue;
//        const float dist = tsdf_data_vec.coeff(data_dim_idx);
//        cv::Vec3i pos;
//        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
//        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
//        pos[0] = data_dim_idx / size_yz;
//        tsdf->AddObservation(pos, dist, weight, cv::Vec3b(255, 255, 255));
//    }
//    //assert(tsdf_weight_vec.nonZeros() == tsdf->vo)
//    tsdf->DisplayInfo();
//    return true;
//}

//bool ConvertDataVectorsToTSDFsWithWeight(
//        const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec,
//        const std::vector<Eigen::SparseVector<float> > &tsdf_weight_vec,
//        const PCAOptions &options,
//        std::vector<TSDFHashing::Ptr> *tsdfs)
//{
//    tsdfs->resize(tsdf_data_vec.size());
//    for(int i = 0; i < tsdf_data_vec.size(); ++i)
//    {
//        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
//        ConvertDataVectorToTSDFWithWeight(
//                    tsdf_data_vec[i],
//                    tsdf_weight_vec[i],
//                    options,
//                    ((*tsdfs)[i].get()));
//    }
//    return true;
//}

//bool ConvertDataVectorsToTSDFsWithWeight(
//        const std::vector<Eigen::SparseVector<float> > &tsdf_data,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> &tsdf_weights,
//        const PCAOptions &options,
//        std::vector<TSDFHashing::Ptr> *tsdfs)
//{
//    tsdfs->resize(tsdf_data.size());
//    for(int i = 0; i < tsdf_data.size(); ++i)
//    {
//        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
//        ConvertDataVectorToTSDFWithWeight(
//                    tsdf_data[i],
//                    tsdf_weights.col(i),
//                    options,
//                    ((*tsdfs)[i].get()));
//    }
//    return true;
//}

//bool PCAReconstructionResult(
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const std::vector<int> &model_assign_idx,
//        std::vector<Eigen::SparseVector<float> > *reconstructed_samples)
//{
//    const int sample_num = model_assign_idx.size();
//    reconstructed_samples->resize(sample_num);
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        int current_model = model_assign_idx[sample_i];
//        (*reconstructed_samples)[sample_i] = (model_means[current_model]);
//        if (model_bases[current_model].rows() * model_bases[current_model].cols() > 0)
//        {
//            // assume projected_coeffs here is not empty
//            (*reconstructed_samples)[sample_i] += (model_bases[current_model]) * projected_coeffs[sample_i].sparseView().eval();
//        }
//    }
//    return true;
//}

//bool WriteAffineTransformsAndTSDFs(const TSDFHashing &scene_tsdf,
//                           const std::vector<Eigen::Affine3f> &affine_transforms,
//                           const PCAOptions& options)
//{
//    const int sample_num = affine_transforms.size();
//    Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
//    Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
//    ExtractSamplesFromAffineTransform(
//            scene_tsdf,
//            affine_transforms,
//            options,
//            &samples,
//            &weights);

//    for (int i = 0; i < sample_num; ++i)
//    {
//        bfs::path prefix(options.save_path);
//        std::string cur_save_path = (prefix.parent_path()/prefix.stem()).string() + "_obb_" + boost::lexical_cast<string>(i) + ".ply";
//        // 1. save obb
//        Eigen::Matrix3f test_r;
//        Eigen::Vector3f test_scale;
//        Eigen::Vector3f test_trans;
//        utility::EigenAffine3fDecomposition(
//                    affine_transforms[i],
//                    &test_r,
//                    &test_scale,
//                    &test_trans);
//        cpu_tsdf::SaveOrientedBoundingbox(test_r, test_trans - (test_r * test_scale.asDiagonal() * Eigen::Vector3f::Ones(3, 1))/2.0f, test_scale, cur_save_path);
//        // 2. save TSDF
//        cpu_tsdf::TSDFHashing::Ptr cur_tsdf(new cpu_tsdf::TSDFHashing);
//        ConvertDataVectorToTSDFWithWeight(
//        samples.col(i),
//        weights.col(i),
//        options,
//        cur_tsdf.get());
//        string save_path = (prefix.parent_path()/prefix.stem()).string() + "_check_affine_" + boost::lexical_cast<string>(i) + ".ply";
//        cpu_tsdf::TSDFHashing::Ptr transformed_tsdf(new cpu_tsdf::TSDFHashing);
//        float voxel_len = (scene_tsdf.voxel_length());
//        cpu_tsdf::TransformTSDF(*cur_tsdf, affine_transforms[i], transformed_tsdf.get(), &voxel_len);
//        cpu_tsdf::WriteTSDFModel(transformed_tsdf, save_path, false, true, options.min_model_weight);
//    }
//    return true;
//}

//bool WriteSampleClusters(const std::vector<int> &sample_model_assign,
//                         const std::vector<Eigen::Vector3f> &model_average_scales,
//                         const std::vector<double>& outlier_gammas,
//                         const std::vector<float>& cluster_assignment_error, const string &save_path)
//{
//    FILE* hf = fopen(save_path.c_str(), "w");
//    for (int i = 0; i < sample_model_assign.size(); ++i)
//    {
//        fprintf(hf, "%8d \t", i);
//        fprintf(hf, "%8d \t", sample_model_assign[i]);
//        fprintf(hf, "gamma: %10.7f\t", outlier_gammas[i]);
//        if (!cluster_assignment_error.empty())
//            fprintf(hf, "assign_error: %10.7f\n", cluster_assignment_error[i]);
//        else
//            fprintf(hf, "\n");
//    }
//    fprintf(hf, "\n");
//    for (int i = 0; i < model_average_scales.size(); ++i)
//    {
//        fprintf(hf, "%dth model, scale-x-y-z: %f %f %f\n", i, model_average_scales[i][0], model_average_scales[i][1], model_average_scales[i][2]);
//    }
//    fclose(hf);
//}

//void ConvertDataMatrixToDataVectors(const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat, std::vector<Eigen::SparseVector<float> > *data_vec)
//{
//    data_vec->resize(data_mat.cols());
//    for (int i = 0; i < data_mat.cols(); ++i)
//    {
//        (*data_vec)[i] = data_mat.col(i);
//    }
//}

//void ConvertDataVectorsToDataMat(const std::vector<Eigen::SparseVector<float> > &data_vec, Eigen::SparseMatrix<float, Eigen::ColMajor> *data_mat)
//{
//    if (data_vec.empty()) return;
//    const int sample_num = data_vec.size();
//    const int feat_dim = data_vec[0].size();
//    for (int i = 0; i < sample_num; ++i)
//    {
//        data_mat->col(i) = data_vec[i];
//    }
//}

}
////////////////////////////////////////////////


//bool cpu_tsdf::JointTSDFAlign(std::vector<cpu_tsdf::TSDFHashing::Ptr>* tsdf_models, std::vector<Eigen::Affine3f>* transform,
//                              cpu_tsdf::TSDFHashing* tsdf_template, const std::string* poutput_plyfilename)
//{
//    std::cout << "Begin joint aligning TSDFs. " << std::endl;
//    if (tsdf_models->empty()) return false;
//    TSDFHashing cur_tsdf_template;
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> temp_tsdf_models(tsdf_models->size());
//    for (int i = 0; i < temp_tsdf_models.size(); ++i)
//    {
//        temp_tsdf_models[i].reset(new cpu_tsdf::TSDFHashing);
//    }
//    std::vector<cpu_tsdf::TSDFHashing::Ptr>* models_pointers[2] = {tsdf_models, &temp_tsdf_models};
//    int cur_model_pointer = 0;
//    std::vector<Eigen::Affine3f> cur_transforms(tsdf_models->size());
//    std::vector<Eigen::Affine3f> accu_transforms(tsdf_models->size());
//    for (int i = 0; i < 8; ++i)
//    {
//        std::cout << "========================================Iteration: " << i << std::endl;
//        AverageTSDFsUnion(*models_pointers[cur_model_pointer], &cur_tsdf_template);
//        for (int j = 0; j < tsdf_models->size(); ++j)
//        {
//            std::cout << "Aligning: " << j << "th model to template. " << std::endl;
//            cur_transforms[j].setIdentity();
//            // optimize transform parameters
//            AlignTSDF(cur_tsdf_template, *((*models_pointers[cur_model_pointer])[j]),
//                      float(tsdf_models->size()) - 0.5, &(cur_transforms[j]));
//            // apply transform
//            TransformTSDF(*((*models_pointers[cur_model_pointer])[j]), cur_transforms[j],
//                          (*models_pointers[(cur_model_pointer+1)%2])[j].get());
//            // accumulate the transform computed each time
//            accu_transforms[j] = cur_transforms[j] * accu_transforms[j];
//            std::cout << "Aligning: " << j << "th model finished. " << std::endl;
//        }
//        cur_model_pointer = (cur_model_pointer+1)%2;
//        // only for saving results
//        if (poutput_plyfilename)
//        {
//            string output_dir = bfs::path(*poutput_plyfilename).remove_filename().string();
//            string current_output_filename =
//                    (bfs::path(output_dir) / (bfs::path(*poutput_plyfilename).stem().string()
//                                              + "_models_iter_" + boost::lexical_cast<string>(i)
//                                              + ".ply")).string();
//            std::cout << "Saving transformed models" << std::endl;
//            std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models = ((*models_pointers[cur_model_pointer]));
//            // current transformed models
//            WriteTSDFModels((*models_pointers[(cur_model_pointer+1)%2]), current_output_filename, true, true, 0.5);
//            std::cout << "Finished saving transformed models" << std::endl;
//
//            std::cout << "Saving learned template" << std::endl;
//            current_output_filename =
//                    (bfs::path(output_dir) / (bfs::path(*poutput_plyfilename).stem().string()
//                                              + "_template_iter_" + boost::lexical_cast<string>(i)
//                                              + ".ply")).string();
//            TSDFHashing::Ptr temp_ptr(new TSDFHashing);
//            *temp_ptr = cur_tsdf_template;
//            WriteTSDFModel(temp_ptr, current_output_filename, true, true, 0.5);
//            std::cout << "Finished Saving learned template" << std::endl;
//        }
//    }
//    *transform = accu_transforms;
//    *tsdf_template = cur_tsdf_template;
//    // set to the latest transformed models
//    tsdf_models->swap((*models_pointers[(cur_model_pointer+1)%2]));
//    std::cout << "End joint aligning TSDFs. " << std::endl;
//    return true;
//}

bool cpu_tsdf::AverageTSDFsIntersection(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models, cpu_tsdf::TSDFHashing* tsdf_template)
{
    std::cout << "Begin averaging TSDFs. " << std::endl;
    if (tsdf_models.empty()) return false;

    cv::Vec3f boundingbox_min, boundingbox_max;
    tsdf_models[0]->getBoundingBoxInWorldCoord(boundingbox_min, boundingbox_max);
    float voxel_length = FLT_MAX;
    float max_dist_pos = FLT_MIN;
    float max_dist_neg = FLT_MAX;
    for (int i = 1; i < tsdf_models.size(); ++i)
    {
        cv::Vec3f cur_min, cur_max;
        tsdf_models[i]->getBoundingBoxInWorldCoord(cur_min, cur_max);
        boundingbox_min = utility::max_vec3(boundingbox_min, cur_min);
        boundingbox_max = utility::min_vec3(boundingbox_max, cur_max);  // get intersection of all the bounding boxes
        voxel_length = std::min(voxel_length, tsdf_models[i]->voxel_length());
        float cur_max_dist_pos, cur_max_dist_neg;
        tsdf_models[i]->getDepthTruncationLimits(cur_max_dist_pos, cur_max_dist_neg);
        max_dist_pos = std::max(max_dist_pos, cur_max_dist_pos);
        max_dist_neg = std::min(max_dist_neg, cur_max_dist_neg);
    }
    // tsdf_template->CopyHashParametersFrom(*(tsdf_models[0]));
    tsdf_template->Init(voxel_length, CvVectorToEigenVector3(boundingbox_min), max_dist_pos, max_dist_neg);

    TSDFHashing::update_hashset_type brick_update_set;
    const int neighbor_limit = 1 * VoxelHashMap::kBrickSideLength;
    for (int i = 0; i < tsdf_models.size(); ++i)
    {
        const TSDFHashing& cur_tsdf = *(tsdf_models[i]);
        for (TSDFHashing::const_iterator citr = cur_tsdf.begin(); citr != cur_tsdf.end(); ++citr)
        {
            cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
            float d, w;
            cv::Vec3b color;
            citr->RetriveData(&d, &w, &color);
            if (w > 0)
            {
                cv::Vec3f world_coord = cur_tsdf.Voxel2World(cv::Vec3f(cur_voxel_coord));
                cv::Vec3i voxel_coord_template = cv::Vec3i(tsdf_template->World2Voxel(world_coord));
                tsdf_template->AddBrickUpdateList(voxel_coord_template, &brick_update_set);
            }  // end if
        }  // end for
    }  // end for
    struct TemplateVoxelUpdater // functor to update every voxel of tsdf_template
    {
        TemplateVoxelUpdater(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models,
                             const TSDFHashing* tsdf_template)
            : tsdf_models(tsdf_models), tsdf_template(tsdf_template) {}
        bool operator() (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
        {
            cv::Vec3f cur_world_coord = tsdf_template->Voxel2World(cv::Vec3f(cur_voxel_coord));
            float final_d = 0;
            for (int i = 0; i < tsdf_models.size(); ++i)
            {
                float cur_d;
                if(!tsdf_models[i]->RetriveDataFromWorldCoord(cur_world_coord, &cur_d))
                {
                    return false;
                }
                final_d += cur_d;
            }
            final_d /= (float)(tsdf_models.size());
            *d = final_d;
            *w = 1;
            *color = cv::Vec3b(255, 255, 255);
            return true;
        }
        const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models;
        const TSDFHashing* tsdf_template;
    };
    TemplateVoxelUpdater tmp_updater(tsdf_models, tsdf_template);
    tsdf_template->UpdateBricksInQueue(brick_update_set, tmp_updater);
    std::cout << "End averaging TSDFs. " << std::endl;
    return true;
}

bool cpu_tsdf::AverageTSDFsUnion(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models, cpu_tsdf::TSDFHashing* tsdf_template)
{
    std::cout << "Begin averaging TSDFs. " << std::endl;
    if (tsdf_models.empty()) return false;

    cv::Vec3f boundingbox_min, boundingbox_max;
    tsdf_models[0]->getBoundingBoxInWorldCoord(boundingbox_min, boundingbox_max);
    float voxel_length = FLT_MAX;
    float max_dist_pos = FLT_MIN;
    float max_dist_neg = FLT_MAX;
    for (int i = 1; i < tsdf_models.size(); ++i)
    {
        cv::Vec3f cur_min, cur_max;
        tsdf_models[i]->getBoundingBoxInWorldCoord(cur_min, cur_max);
        boundingbox_min = min_vec3(boundingbox_min, cur_min);
        boundingbox_max = max_vec3(boundingbox_max, cur_max);  // get intersection of all the bounding boxes
        voxel_length = std::min(voxel_length, tsdf_models[i]->voxel_length());
        float cur_max_dist_pos, cur_max_dist_neg;
        tsdf_models[i]->getDepthTruncationLimits(cur_max_dist_pos, cur_max_dist_neg);
        max_dist_pos = std::max(max_dist_pos, cur_max_dist_pos);
        max_dist_neg = std::min(max_dist_neg, cur_max_dist_neg);
    }
    // tsdf_template->CopyHashParametersFrom(*(tsdf_models[0]));
    tsdf_template->Init(voxel_length, CvVectorToEigenVector3(boundingbox_min), max_dist_pos, max_dist_neg);

    TSDFHashing::update_hashset_type brick_update_set;
    const int neighbor_limit = 1 * VoxelHashMap::kBrickSideLength;
    for (int i = 0; i < tsdf_models.size(); ++i)
    {
        const TSDFHashing& cur_tsdf = *(tsdf_models[i]);
        for (TSDFHashing::const_iterator citr = cur_tsdf.begin(); citr != cur_tsdf.end(); ++citr)
        {
            cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
            float d, w;
            cv::Vec3b color;
            citr->RetriveData(&d, &w, &color);
            if (w > 0)
            {
                cv::Vec3f world_coord = cur_tsdf.Voxel2World(cv::Vec3f(cur_voxel_coord));
                cv::Vec3i voxel_coord_template = cv::Vec3i(tsdf_template->World2Voxel(world_coord));
                tsdf_template->AddBrickUpdateList(voxel_coord_template, &brick_update_set);
            }  // end if
        }  // end for
    }  // end for

    struct TemplateVoxelUpdater // functor to update every voxel of tsdf_template
    {
        TemplateVoxelUpdater(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models,
                             const TSDFHashing* tsdf_template)
            : tsdf_models(tsdf_models), tsdf_template(tsdf_template) {}
        bool operator() (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
        {
            cv::Vec3f cur_world_coord = tsdf_template->Voxel2World(cv::Vec3f(cur_voxel_coord));
            float final_d = 0;
            float final_w = 0;
            for (int i = 0; i < tsdf_models.size(); ++i)
            {
                float cur_d;
                float cur_w;
                if(!tsdf_models[i]->RetriveDataFromWorldCoord(cur_world_coord, &cur_d, &cur_w))
                {
                    continue;
                }
                final_d += (cur_d * cur_w);
                final_w += cur_w;
            }
            if (final_w == 0) return false;
            final_d /= (float)(final_w);
            *d = final_d;
            *w = final_w;
            *color = cv::Vec3b(255, 255, 255);
            return true;
        }
        const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models;
        const TSDFHashing* tsdf_template;
    };
    TemplateVoxelUpdater tmp_updater(tsdf_models, tsdf_template);
    tsdf_template->UpdateBricksInQueue(brick_update_set, tmp_updater);
    std::cout << "End averaging TSDFs. " << std::endl;
    return true;
}

///**
// * @brief cpu_tsdf::InitializeDataMatrixFromTSDFs
// * Compute multiple matrix for PCA from TSDFs
// * @param tsdf_models
// * @param centrlized_data_mat
// * @param weight_mat
// * @param mean_mat D * 1 mean mat
// * @param tsdf_template Mean TSDF
// * @param bounding_box Size of the bounding box
// * @return
// */
//bool cpu_tsdf::InitializeDataMatrixFromTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
//                                             Eigen::SparseMatrix<float, ColMajor> *centrlized_data_mat,
//                                             Eigen::SparseMatrix<float, ColMajor> *weight_mat,
//                                             Eigen::SparseVector<float> *mean_mat,
//                                             cpu_tsdf::TSDFHashing* tsdf_template,
//                                             Eigen::Vector3i* bounding_box)  // also compute the mean
//{
//    std::cout << "Begin TSDFsToMatrix and averaging TSDFs. " << std::endl;
//    if (tsdf_models.empty()) return false;

//    // compute the bounding box of the models
//    cv::Vec3f boundingbox_min, boundingbox_max;
//    tsdf_models[0]->getBoundingBoxInWorldCoord(boundingbox_min, boundingbox_max);
//    float voxel_length = FLT_MAX;
//    float max_dist_pos = FLT_MIN;
//    float max_dist_neg = FLT_MAX;
//    for (int i = 1; i < tsdf_models.size(); ++i)
//    {
//        cv::Vec3f cur_min, cur_max;
//        tsdf_models[i]->getBoundingBoxInWorldCoord(cur_min, cur_max);
//        boundingbox_min = min_vec3(boundingbox_min, cur_min);
//        boundingbox_max = max_vec3(boundingbox_max, cur_max);  // get intersection of all the bounding boxes
//        voxel_length = std::min(voxel_length, tsdf_models[i]->voxel_length());
//        float cur_max_dist_pos, cur_max_dist_neg;
//        tsdf_models[i]->getDepthTruncationLimits(cur_max_dist_pos, cur_max_dist_neg);
//        max_dist_pos = std::max(max_dist_pos, cur_max_dist_pos);
//        max_dist_neg = std::min(max_dist_neg, cur_max_dist_neg);
//    }
//    float mean_sample_voxel_length = voxel_length /** 2*/;
//    // tsdf_template->CopyHashParametersFrom(tsdf_models[0]);
//    tsdf_template->Init(mean_sample_voxel_length, CvVectorToEigenVector3(boundingbox_min), max_dist_pos, max_dist_neg);
//    Eigen::Vector3i voxel_bounding_box_size = CvVectorToEigenVector3((boundingbox_max - boundingbox_min)/mean_sample_voxel_length).cast<int>();
//    const int data_dim = voxel_bounding_box_size[0] * voxel_bounding_box_size[1] * voxel_bounding_box_size[2];
//    const int sample_num = tsdf_models.size();
//    centrlized_data_mat->resize(data_dim, sample_num);
//    weight_mat->resize(data_dim, sample_num);
//    mean_mat->resize(data_dim);
//    mean_mat->setZero();

//    TSDFHashing::update_hashset_type brick_update_set;
//    const int neighbor_limit = 1 * VoxelHashMap::kBrickSideLength;
//    for (int i = 0; i < tsdf_models.size(); ++i)
//    {
//        const TSDFHashing& cur_tsdf = *(tsdf_models[i]);
//        for (TSDFHashing::const_iterator citr = cur_tsdf.begin(); citr != cur_tsdf.end(); ++citr)
//        {
//            cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
//            float d, w;
//            cv::Vec3b color;
//            citr->RetriveData(&d, &w, &color);
//            if (w > 0)
//            {
//                cv::Vec3f world_coord = cur_tsdf.Voxel2World(cv::Vec3f(cur_voxel_coord));
//                cv::Vec3i voxel_coord_template = cv::Vec3i(tsdf_template->World2Voxel(world_coord));
//                tsdf_template->AddBrickUpdateList(voxel_coord_template, &brick_update_set);
//            }  // end if
//        }  // end for
//    }  // end for

//    // compute mean/data/weight matrices
//    struct TemplateVoxelUpdater // functor to update every voxel of tsdf_template
//    {
//        TemplateVoxelUpdater(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models,
//                             const TSDFHashing* tsdf_template, const Eigen::Vector3i& bb_size, Eigen::SparseVector<float>* vmean_mat)
//            : tsdf_models(tsdf_models), tsdf_template(tsdf_template), voxel_bounding_box_size(bb_size),
//              valid_sample(tsdf_models.size(), false), cur_dists(tsdf_models.size()), cur_weights(tsdf_models.size()),
//              mean_mat(vmean_mat) {}
//        bool operator() (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color)
//        {
//            std::fill(valid_sample.begin(), valid_sample.end(), false);
//            std::fill(cur_dists.begin(), cur_dists.end(), 0);
//            std::fill(cur_weights.begin(), cur_weights.end(), 0);

//            cv::Vec3f cur_world_coord = tsdf_template->Voxel2World(cv::Vec3f(cur_voxel_coord));
//            const int data_dim_idx = cur_voxel_coord[2] +
//                    (cur_voxel_coord[1] + cur_voxel_coord[0] * voxel_bounding_box_size[1]) * voxel_bounding_box_size[2];
//            float final_d = 0;
//            float final_w = 0;
//            for (int i = 0; i < tsdf_models.size(); ++i)
//            {
//                float cur_d;
//                float cur_w;
//                if(!tsdf_models[i]->RetriveDataFromWorldCoord(cur_world_coord, &cur_d, &cur_w))
//                {
//                    continue;
//                }
//                valid_sample[i] = true;
//                cur_dists[i] = cur_d;
//                cur_weights[i] = cur_w;

//                final_d += (cur_d * cur_w);
//                final_w += cur_w;
//            }
//            if (final_w < 1e-5) return false;
//            final_d /= (float)(final_w);
//            *d = final_d;
//            *w = final_w;
//            *color = cv::Vec3b(255, 255, 255);
//            for (int sample_dim_idx = 0; sample_dim_idx < valid_sample.size(); ++sample_dim_idx)
//            {
//                if (valid_sample[sample_dim_idx])
//                {
//                    data_list.push_back(Eigen::Triplet<float>(data_dim_idx, sample_dim_idx, cur_dists[sample_dim_idx] - final_d));
//                    weight_list.push_back(Eigen::Triplet<float>(data_dim_idx, sample_dim_idx, cur_weights[sample_dim_idx]));
//                }
//            }
//            mean_mat->coeffRef(data_dim_idx) = final_d;
//            return true;
//        }
//        const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models;
//        const TSDFHashing* tsdf_template;
//        const Eigen::Vector3i voxel_bounding_box_size;
//        std::vector<Eigen::Triplet<float>> data_list;
//        std::vector<Eigen::Triplet<float>> weight_list;
//        std::vector<uchar> valid_sample;
//        std::vector<float> cur_dists;
//        std::vector<float> cur_weights;
//        Eigen::SparseVector<float>* mean_mat;
//    };
//    TemplateVoxelUpdater new_updater(tsdf_models, tsdf_template, voxel_bounding_box_size, mean_mat);
//    tsdf_template->UpdateBricksInQueue(brick_update_set, new_updater);
//    centrlized_data_mat->setFromTriplets(new_updater.data_list.begin(),
//                                         new_updater.data_list.end());
//    weight_mat->setFromTriplets(new_updater.weight_list.begin(),
//                                new_updater.weight_list.end());
//    *bounding_box = voxel_bounding_box_size;
//    std::cout << "End averaging TSDFs. Initial estimate finished" << std::endl;
//    return true;
//}

///**
// * @brief cpu_tsdf::InitialEstimate
// * Do an initial estimate of the basis. May use random initialization or do an inital PCA
// * @param centralized_data_mat_row_major
// * @param weight_mat_row_major
// * @param component_num
// * @param base_mat_row_major Output, estimated base_mat
// * @param coeff_mat Output estimated coefficient mat
// */
//void cpu_tsdf::InitialEstimate(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
//                               const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
//                               const int component_num,
//                               Eigen::SparseMatrix<float, Eigen::RowMajor> *base_mat_row_major, Eigen::MatrixXf *coeff_mat)
//{

//    const int sample_num = centralized_data_mat_row_major.cols();
//    const int data_dim = centralized_data_mat_row_major.rows();
//    base_mat_row_major->resize(data_dim, component_num);
//    base_mat_row_major->setZero();
//    base_mat_row_major->reserve(data_dim * sample_num * 0.7);
//    std::cerr << "making initial estimate: " << std::endl;
//    std::cerr << "date_dim: " << data_dim << std::endl;
//    std::cerr << "sample_num: " << sample_num << std::endl;
//    std::cerr << "data_mat_nnz: " << centralized_data_mat_row_major.nonZeros() << std::endl;

//    //1. random:
//    //  for (int i = 0; i < data_dim; ++i)
//    //    {
//    //      bool flag = false;
//    //      for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(centralized_data_mat_row_major,i); it; ++it)
//    //        {
//    //          flag = true;
//    //          break;
//    //        }
//    //      if (flag)
//    //        {
//    //          for (int j = 0; j < component_num; ++j)
//    //          {
//    //            base_mat_row_major->insert(i, j) = float(rand())/RAND_MAX - 0.5;
//    //          }
//    //        }
//    //      //std::cerr << i << std::endl;
//    //    }
//    //2. pca
//    Eigen::MatrixXf weighted_D = weight_mat_row_major.cwiseProduct(centralized_data_mat_row_major);
//    Eigen::MatrixXf WD_trans_WD = weighted_D.transpose() * weighted_D;
//    Eigen::MatrixXf W_trans_W = (weight_mat_row_major.transpose() * weight_mat_row_major).eval();
//    WD_trans_WD = WD_trans_WD.cwiseQuotient(W_trans_W) * (data_dim/sample_num);
//    Eigen::JacobiSVD<Eigen::MatrixXf> svd(WD_trans_WD ,Eigen::ComputeFullV);
//    double tolerance =
//            std::numeric_limits<float>::epsilon() * std::max(WD_trans_WD.cols(), WD_trans_WD.rows()) *svd.singularValues().array().abs()(0);
//    *coeff_mat =
//            (((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array(), 0)).array().sqrt().matrix().asDiagonal()
//             * svd.matrixV().transpose()).middleRows(0, component_num);
//    ComputeBasisInMStep(centralized_data_mat_row_major, weight_mat_row_major, *coeff_mat, base_mat_row_major);
//    std::cerr << "finished making initial estimate " << std::endl;
//    return;
//}


///**
// * @brief cpu_tsdf::ComputeCoeffInEStep
// * E step, compute coefficient matrix
// * @param centralized_data_mat
// * @param weight_mat
// * @param base_mat
// * @param coeff_mat
// */
//void cpu_tsdf::ComputeCoeffInEStep(const Eigen::SparseMatrix<float, Eigen::ColMajor> &centralized_data_mat,
//                                   const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat,
//                                   const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
//                                   Eigen::MatrixXf *coeff_mat)
//{
//    const int sample_num = centralized_data_mat.cols();
//    const int data_dim = centralized_data_mat.rows();
//    const int component_num = base_mat.cols();
//    Eigen::SparseMatrix<float, Eigen::ColMajor> W_j_U(data_dim, component_num);
//    for (int j = 0; j < sample_num; ++j)
//    {
//        W_j_U = (Eigen::VectorXf(weight_mat.col(j)).asDiagonal()) * base_mat;
//        Eigen::MatrixXf temp1 = Eigen::MatrixXf((base_mat.transpose() * W_j_U).eval());
//        coeff_mat->col(j) =
//                Eigen::VectorXf(
//                    (Eigen::MatrixXf(utility::PseudoInverse(temp1)) *
//                     (W_j_U.transpose() * centralized_data_mat.col(j).eval()))
//                    );
//    }
//}

///**
// * @brief cpu_tsdf::ComputeBasisInMStep
// * M step, compute the bases matrix (slow)
// * @param centralized_data_mat_row_major
// * @param weight_mat_row_major
// * @param coeff_mat
// * @param base_mat_row_major
// */
//void cpu_tsdf::ComputeBasisInMStep(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
//                                   const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
//                                   const Eigen::MatrixXf &coeff_mat, Eigen::SparseMatrix<float, Eigen::RowMajor> *base_mat_row_major)
//{
//    const int sample_num = centralized_data_mat_row_major.cols();
//    const int data_dim = centralized_data_mat_row_major.rows();
//    const int component_num = coeff_mat.rows();
//    Eigen::VectorXf data_row_vec =
//            Eigen::VectorXf::Zero(sample_num, 1);
//    base_mat_row_major->setZero();
//    for (int i = 0; i < data_dim; ++i)
//    {
//        if (i % 100000 == 0)
//            std::cerr << "cur_i: " << i << std::endl;
//        bool flag = false;
//        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itr(centralized_data_mat_row_major, i);
//             itr; ++itr)
//        {
//            flag = true;
//            data_row_vec.coeffRef(itr.col()) = itr.value();
//        }
//        if (flag)
//        {
//            Eigen::MatrixXf A_Wi = coeff_mat * Eigen::MatrixXf(weight_mat_row_major.row(i)).asDiagonal();
//            Eigen::VectorXf row_i_dense =
//                    Eigen::MatrixXf( A_Wi * coeff_mat.transpose() ).
//                    jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).
//                    solve(A_Wi * data_row_vec);
//            for (int k = 0; k < component_num; ++k)
//            {
//                if (abs(row_i_dense[k]) > std::numeric_limits<double>::epsilon())
//                {
//                    base_mat_row_major->insert(i, k) = row_i_dense[k];
//                }
//            }
//            data_row_vec.setZero();
//        }  // end if
//    }  // end for i
//}

//// compute only one row of coeffcient matrix
//void cpu_tsdf::ComputeCoeffInEStepOneVec(const Eigen::SparseMatrix<float, Eigen::ColMajor> &centralized_data_mat,
//                                         const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat,
//                                         const Eigen::VectorXf &base_vec,
//                                         Eigen::VectorXf *coeff_vec)
//{
//    const int sample_num = centralized_data_mat.cols();
//    const int data_dim = centralized_data_mat.rows();
//    Eigen::VectorXf tmpA(data_dim);
//    for (int j = 0; j < sample_num; ++j)
//    {
//        tmpA = weight_mat.col(j).cwiseProduct(base_vec);
//        (*coeff_vec)(j) = centralized_data_mat.col(j).dot(tmpA) / (tmpA.dot(base_vec));
//        if (std::isnan((*coeff_vec)(j))) (*coeff_vec)(j) = 0;
//    }
//}

//// compute only one column of base matrix
//void cpu_tsdf::ComputeBasisInMStepOneVec(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
//                                         const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
//                                         const Eigen::VectorXf &coeff_vec, Eigen::VectorXf *base_vec)
//{
//    const int sample_num = centralized_data_mat_row_major.cols();
//    const int data_dim = centralized_data_mat_row_major.rows();
//    Eigen::VectorXf data_row_vec =
//            Eigen::VectorXf::Zero(sample_num, 1);
//    base_vec->setZero();
//    for (int i = 0; i < data_dim; ++i)
//    {
//        //      if (i % 10000 == 0)
//        //        std::cerr << "cur_i: " << i << std::endl;

//        bool flag = false;
//        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itr(centralized_data_mat_row_major, i);
//             itr; ++itr)
//        {
//            flag = true;
//            data_row_vec.coeffRef(itr.col()) = itr.value();
//        }
//        if (flag)
//        {
//            Eigen::VectorXf tmpB = weight_mat_row_major.row(i).transpose().cwiseProduct(coeff_vec);
//            (*base_vec)(i) = data_row_vec.dot(tmpB) / (tmpB.dot(coeff_vec));
//            if (std::isnan((*base_vec)(i))) (*base_vec)(i) = 0;
//            data_row_vec.setZero();
//        }  // end if
//    }  // end for i
//}

///**
// * @brief cpu_tsdf::OrthogonalizeVector
// * @param base_mat
// * @param current_components use [0, current_components) columns of base_mat for orthogonalization
// * @param base_vec Input vector to be orthogonalized, also Output orthogonalized vector
// * @param norm Output, the vector norm
// */
//void cpu_tsdf::OrthogonalizeVector(const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
//                                   const int current_components,
//                                   Eigen::VectorXf *base_vec, float* norm)
//{
//    for (int i = 0; i < current_components; ++i)
//    {
//        Eigen::VectorXf current_column = base_mat.col(i);
//        *base_vec -= (base_vec->dot(current_column) * current_column);
//    }
//    *norm = base_vec->norm();
//    base_vec->normalize();
//    return;
//}

//bool cpu_tsdf::TSDFWeightedPCA_NewWrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
//                               const int component_num, const int max_iter,
//                               cpu_tsdf::TSDFHashing *mean_tsdf, std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models,
//                               const std::string *save_filepath)
//{
//    Eigen::SparseMatrix<float, Eigen::ColMajor> weight_mat;
//    Eigen::SparseMatrix<float, Eigen::ColMajor> base_mat;
//    Eigen::MatrixXf coeff_mat;
//    Eigen::SparseVector<float> mean_mat;
//    Eigen::Vector3i voxel_bounding_box_size;

//    // 1st: test whether the PCA part are identical
//    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
//    cout << "00. convert TSDFs to sparse matrix" << endl;
//    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, &weight_mat, &mean_mat, mean_tsdf, &voxel_bounding_box_size)) return false;

//    Eigen::SparseVector<float> new_mean_mat;
//    PCAOptions options;
//    options.boundingbox_size = voxel_bounding_box_size;
//    options.lambda_scale_diff = 1.0;
//    tsdf_models[0]->getDepthTruncationLimits(options.max_dist_pos, options.max_dist_neg);
//    options.min_model_weight = 0;
//    options.offset = Eigen::Vector3f(0, 0, 0);
//    options.voxel_length = tsdf_models[0]->voxel_length();
//    if (save_filepath)
//        options.save_path = *save_filepath;
//   // cpu_tsdf::WeightedPCADeflationOrthogonal(centralized_data_mat, weiht_mat, component_num, max_iter, &new_mean_mat, &base_mat, &coeff_mat, options);

//    //  TSDFWeightedPCA(tsdf_models, component_num, max_iter,
//    //                  mean_tsdf, &mean_mat,
//    //                  &base_mat, &coeff_mat, &weight_mat,
//    //                  &voxel_bounding_box_size, save_filepath);
//    //  TSDFWeightedPCADeflation(tsdf_models, component_num, max_iter,
//    //                           mean_tsdf, &mean_mat,
//    //                           &base_mat, &coeff_mat, &weight_mat,
//    //                           &voxel_bounding_box_size, save_filepath);
//    TSDFWeightedPCADeflationOrthogonal_Old(tsdf_models, component_num, max_iter,
//                                       mean_tsdf, &mean_mat,
//                                       &base_mat, &coeff_mat, &weight_mat,
//                                       &voxel_bounding_box_size, save_filepath);

//    if (save_filepath)
//    {
//        string output_file = (*save_filepath).substr(0, save_filepath->length() - 4) + "_boundingbox.txt";
//        ofstream os(output_file);
//        os << voxel_bounding_box_size << endl;
//    }
//    const int data_dim = base_mat.rows();
//    const int sample_num = tsdf_models.size();
//    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
//            mean_mat * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
//            base_mat * coeff_mat.sparseView().eval();
//    const float voxel_length = mean_tsdf->voxel_length();
//    const Eigen::Vector3f offset = mean_tsdf->offset();
//    float max_dist_pos, max_dist_neg;
//    mean_tsdf->getDepthTruncationLimits(max_dist_pos, max_dist_neg);
//    return ConvertDataMatrixToTSDFsNoWeight(voxel_length,
//                                            offset,
//                                            max_dist_pos,
//                                            max_dist_neg,
//                                            voxel_bounding_box_size,
//                                            projected_data_mat,
//                                            projected_tsdf_models
//                                            );
//    //  return ConvertDataMatrixToTSDFs(voxel_length,
//    //                                  offset,
//    //                                  max_dist_pos,
//    //                                  max_dist_neg,
//    //                                  voxel_bounding_box_size,
//    //                                  projected_data_mat,
//    //                                  weight_mat,
//    //                                  projected_tsdf_models
//    //                                  );
//}


///**
// * @brief cpu_tsdf::TSDFWeightedPCA
// * Wrapper function for doing PCA
// * @param tsdf_models
// * @param component_num
// * @param max_iter
// * @param mean_tsdf
// * @param projected_tsdf_models
// * @param save_filepath
// * @return
// */
//bool cpu_tsdf::TSDFWeightedPCA_Wrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
//                               cpu_tsdf::TSDFHashing *mean_tsdf, std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models,
//                               const std::string *save_filepath)
//{
//    Eigen::SparseMatrix<float, Eigen::ColMajor> weight_mat;
//    Eigen::SparseMatrix<float, Eigen::ColMajor> base_mat;
//    Eigen::MatrixXf coeff_mat;
//    Eigen::SparseVector<float> mean_mat;
//    Eigen::Vector3i voxel_bounding_box_size;
//    //  TSDFWeightedPCA(tsdf_models, component_num, max_iter,
//    //                  mean_tsdf, &mean_mat,
//    //                  &base_mat, &coeff_mat, &weight_mat,
//    //                  &voxel_bounding_box_size, save_filepath);
//    //  TSDFWeightedPCADeflation(tsdf_models, component_num, max_iter,
//    //                           mean_tsdf, &mean_mat,
//    //                           &base_mat, &coeff_mat, &weight_mat,
//    //                           &voxel_bounding_box_size, save_filepath);
//    TSDFWeightedPCADeflationOrthogonal_Old(tsdf_models, component_num, max_iter,
//                                       mean_tsdf, &mean_mat,
//                                       &base_mat, &coeff_mat, &weight_mat,
//                                       &voxel_bounding_box_size, save_filepath);

//    if (save_filepath)
//    {
//        string output_file = (*save_filepath).substr(0, save_filepath->length() - 4) + "_boundingbox.txt";
//        ofstream os(output_file);
//        os << voxel_bounding_box_size << endl;
//    }
//    const int data_dim = base_mat.rows();
//    const int sample_num = tsdf_models.size();
//    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
//            mean_mat * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
//            base_mat * coeff_mat.sparseView().eval();
//    const float voxel_length = mean_tsdf->voxel_length();
//    const Eigen::Vector3f offset = mean_tsdf->offset();
//    float max_dist_pos, max_dist_neg;
//    mean_tsdf->getDepthTruncationLimits(max_dist_pos, max_dist_neg);
//    return ConvertDataMatrixToTSDFsNoWeight(voxel_length,
//                                            offset,
//                                            max_dist_pos,
//                                            max_dist_neg,
//                                            voxel_bounding_box_size,
//                                            projected_data_mat,
//                                            projected_tsdf_models
//                                            );
//    //  return ConvertDataMatrixToTSDFs(voxel_length,
//    //                                  offset,
//    //                                  max_dist_pos,
//    //                                  max_dist_neg,
//    //                                  voxel_bounding_box_size,
//    //                                  projected_data_mat,
//    //                                  weight_mat,
//    //                                  projected_tsdf_models
//    //                                  );
//}

//// utility function to save intermediate results
//static void WPCASaveRelatedMatrices(const Eigen::SparseMatrix<float, ColMajor>* centralized_data_mat,
//                                    const Eigen::SparseVector<float>* mean_mat,
//                                    const Eigen::SparseMatrix<float, ColMajor>* weight_mat,
//                                    const Eigen::SparseMatrix<float, ColMajor>* base_mat,
//                                    const Eigen::MatrixXf *coeff_mat,
//                                    const string& save_filepath
//                                    )
//{
//    cout << "save related matrices" << endl;
//    string output_dir = bfs::path(save_filepath).remove_filename().string();
//    std::string output_plyfilename = save_filepath;
//    string output_datamat =
//            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
//                                      + "_output_data_mat.txt")).string();
//    string output_meanmat =
//            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
//                                      + "_output_mean_mat.txt")).string();
//    string output_weightmat =
//            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
//                                      + "_output_weight_mat.txt")).string();
//    string output_initbasemat =
//            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
//                                      + "_output_base_mat.txt")).string();
//    string output_initcoeffmat =
//            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
//                                      + "_output_coeff_mat.txt")).string();
//    if (centralized_data_mat)
//        utility::WriteEigenMatrix(*centralized_data_mat, output_datamat);
//    if (mean_mat)
//        utility::WriteEigenMatrix(Eigen::MatrixXf(*mean_mat), output_meanmat);
//    if (weight_mat)
//        utility::WriteEigenMatrix(*weight_mat, output_weightmat);
//    if (base_mat)
//        utility::WriteEigenMatrix(*base_mat, output_initbasemat);
//    if (coeff_mat)
//        utility::WriteEigenMatrix(*coeff_mat, output_initcoeffmat);
//    cout << "save related matrices finished" << endl;
//}


//void cpu_tsdf::WPCASaveMeanTSDF(
//        const Eigen::SparseVector<float>& mean_mat,
//        const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
//        const Eigen::Vector3i& boundingbox_size,
//        const float voxel_length,
//        const Eigen::Vector3f& offset,
//        const float max_dist_pos,
//        const float max_dist_neg,
//        const string& save_filepath)
//{
//    cout << "save mean tsdf" << endl;
//    const int sample_num = weight_mat.cols();
//    //mean tsdf (in mean_mat)
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
//    Eigen::SparseMatrix<float, Eigen::ColMajor> tweight = (weight_mat * Eigen::VectorXf::Ones(sample_num, 1) / sample_num).sparseView();
//    cpu_tsdf::ConvertDataMatrixToTSDFs(voxel_length,
//                             offset,
//                             max_dist_pos,
//                             max_dist_neg,
//                             boundingbox_size,
//                             mean_mat,
//                             tweight,
//                             &projected_tsdf_models
//                             );
//    string output_dir = bfs::path(save_filepath).remove_filename().string();
//    string output_meanply =
//            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
//                                      + "mean.ply")).string();
//    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_meanply, false, true, 0);
//}

//void cpu_tsdf::WPCASaveMeanTSDF(const Eigen::SparseVector<float>& mean_mat,
//                             const cpu_tsdf::TSDFHashing& mean_tsdf,
//                             const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
//                             const Eigen::Vector3i& boundingbox_size,
//                             const string& save_filepath)
//{
//    cout << "save mean tsdf" << endl;
//    //mean tsdf (in mean_mat)
//    const float voxel_length = mean_tsdf.voxel_length();
//    const Eigen::Vector3f offset = mean_tsdf.offset();
//    float max_dist_pos, max_dist_neg;
//    mean_tsdf.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
//    Eigen::SparseMatrix<float, Eigen::ColMajor> tweight = (weight_mat * Eigen::VectorXf::Ones(weight_mat.cols(), 1)).sparseView();
//    cpu_tsdf::ConvertDataMatrixToTSDFs(voxel_length,
//                             offset,
//                             max_dist_pos,
//                             max_dist_neg,
//                             boundingbox_size,
//                             mean_mat,
//                             tweight,
//                             &projected_tsdf_models
//                             );
//    string output_dir = bfs::path(save_filepath).remove_filename().string();
//    string output_meanply =
//            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
//                                      + "mean.ply")).string();
//    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_meanply, false, true, 0);

//    //assert(*mean_tsdf == *(projected_tsdf_models[0]));
//    //mean tsdf - tsdf version
////    {
////        std::string output_plyfilename = *save_filepath;
////        cout << "meshing pca projected models and template - meantsdf." << endl;
////        cpu_tsdf::TSDFHashing::Ptr ptr_temp_mean_tsdf(new cpu_tsdf::TSDFHashing);
////        *ptr_temp_mean_tsdf = *mean_tsdf;
////        ptr_temp_mean_tsdf->DisplayInfo();
////        {
////            std::cout << "begin marching cubes for model "   << std::endl;
////            cpu_tsdf::MarchingCubesTSDFHashing mc;
////            mc.setMinWeight(0);
////            mc.setInputTSDF (ptr_temp_mean_tsdf);
////            pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh);
////            fprintf(stderr, "perform reconstruction: \n");
////            mc.reconstruct (*mesh);
////            //PCL_INFO ("Entire pipeline took %f ms\n", tt.toc ());
////            flattenVertices(*mesh);
////            //string output_dir = bfs::path(input_model_filenames[i]).remove_filename().string();
////            string output_dir = bfs::path(output_plyfilename).remove_filename().string();
////            string output_modelply =
////                    (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
////                                              + "_tsdf_pca_projected_modelply_"
////                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
////                                              + "_origin_mean_tsdfv" + ".ply")).string();
////            std::cout << "save tsdf file path: " << output_modelply << std::endl;
////            pcl::io::savePLYFileBinary (output_modelply, *mesh);
////        }
////    }
//}

//static void WPCASaveTSDFsFromMats(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& projected_data_mat,
//        const Eigen::Vector3i& boundingbox_size,
//        const float voxel_length,
//        const Eigen::Vector3f& offset,
//        const float max_dist_pos,
//        const float max_dist_neg,
//        const string& save_filepath)
//{
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
//    cpu_tsdf::ConvertDataMatrixToTSDFsNoWeight(voxel_length,
//                                               offset,
//                                               max_dist_pos,
//                                               max_dist_neg,
//                                               boundingbox_size,
//                                               projected_data_mat,
//                                               &projected_tsdf_models
//                                               );
//    string output_dir = bfs::path(save_filepath).remove_filename().string();
//    string output_modelply =
//            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
//                                      + "_recovered_models.ply")).string();
//    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_modelply, false, true, 0);
//}

//static void WPCASaveTSDFsFromMats(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& projected_data_mat,
//        const cpu_tsdf::TSDFHashing& mean_tsdf,
//        const Eigen::Vector3i& boundingbox_size,
//        const string& save_filepath)
//{
//    const float voxel_length = mean_tsdf.voxel_length();
//    const Eigen::Vector3f offset = mean_tsdf.offset();
//    float max_dist_pos, max_dist_neg;
//    mean_tsdf.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
//    cpu_tsdf::ConvertDataMatrixToTSDFsNoWeight(voxel_length,
//                                               offset,
//                                               max_dist_pos,
//                                               max_dist_neg,
//                                               boundingbox_size,
//                                               projected_data_mat,
//                                               &projected_tsdf_models
//                                               );
//    string output_dir = bfs::path(save_filepath).remove_filename().string();
//    string output_modelply =
//            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
//                                      + "_recovered_models.ply")).string();
//    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_modelply, false, true, 0);
//}

//static void WPCASaveUncentralizedTSDFs(
//        const Eigen::SparseMatrix<float, ColMajor>& centralized_data_mat,
//        const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
//        const Eigen::SparseVector<float>& mean_mat,
//        const Eigen::Vector3i& boundingbox_size,
//        const float voxel_length,
//        const Eigen::Vector3f& offset,
//        const float max_dist_pos,
//        const float max_dist_neg,
//        const string& save_filepath)
//{
//    cout << "save uncentralized tsdf" << endl;
//    const int data_dim = centralized_data_mat.rows();
//    const int sample_num = centralized_data_mat.cols();
//    // original tsdf models
//    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
//                     (mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
//                     centralized_data_mat;
//    WPCASaveTSDFsFromMats(projected_data_mat, boundingbox_size,
//                          voxel_length, offset, max_dist_pos, max_dist_neg, save_filepath);
//}

//static void WPCASaveUncentralizedTSDFs(const Eigen::SparseMatrix<float, ColMajor>& centralized_data_mat,
//                                       const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
//                                       const Eigen::SparseVector<float>& mean_mat,
//                                       const cpu_tsdf::TSDFHashing& mean_tsdf,
//                                       const Eigen::Vector3i& boundingbox_size,
//                                       const string& save_filepath)
//{
//    cout << "save uncentralized tsdf" << endl;
//    const int data_dim = centralized_data_mat.rows();
//    const int sample_num = centralized_data_mat.cols();
//    // original tsdf models
//    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
//                     (mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
//                     centralized_data_mat;
//    WPCASaveTSDFsFromMats(projected_data_mat, mean_tsdf, boundingbox_size, save_filepath);
//}

//// utility function to save intermediate results
//static void WPCASaveRecoveredTSDFs(const Eigen::SparseVector<float>& mean_part_mat,
//                                   // these two cannot be const because bug(?) in Eigen
//                                   // If set to const, the production of these two will be computed in some "conservative" way
//                                   // and leads to infinite recursion?
//                                   Eigen::SparseMatrix<float, ColMajor>& base_mat,
//                                   Eigen::MatrixXf& coeff_mat,
//                                   const int component_num,
//                                   const Eigen::Vector3i& boundingbox_size,
//                                   const float voxel_length,
//                                   const Eigen::Vector3f& offset,
//                                   const float max_dist_pos,
//                                   const float max_dist_neg,
//                                   const string& save_filepath
//                                   )
//{
//    cout << "save pca recovered tsdf" << endl;
//    const int data_dim = base_mat.rows();
//    const int sample_num = coeff_mat.cols();
//    // tsdf models
//    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
//                  (mean_part_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
//                     (base_mat).leftCols(component_num) * (coeff_mat).topRows(component_num).sparseView().eval();
//    WPCASaveTSDFsFromMats(projected_data_mat, boundingbox_size,
//                          voxel_length, offset, max_dist_pos, max_dist_neg, save_filepath);
//}

//static void WPCASaveRecoveredTSDFs(const Eigen::SparseVector<float>& mean_part_mat,
//                                   const cpu_tsdf::TSDFHashing& mean_tsdf,
//                                   // these two cannot be const because bug(?) in Eigen
//                                   // If set to const, the production of these two will be computed in some "conservative" way
//                                   // and leads to infinite recursion?
//                                   Eigen::SparseMatrix<float, ColMajor>& base_mat,
//                                   Eigen::MatrixXf& coeff_mat,
//                                   const int component_num,
//                                   const Eigen::Vector3i& boundingbox_size,
//                                   const string& save_filepath
//                                   )
//{
//    cout << "save pca recovered tsdf" << endl;
//    const int data_dim = base_mat.rows();
//    const int sample_num = coeff_mat.cols();
//    // tsdf models
//    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
//                  (mean_part_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
//                     (base_mat).leftCols(component_num) * (coeff_mat).topRows(component_num).sparseView().eval();
//    WPCASaveTSDFsFromMats(projected_data_mat, mean_tsdf, boundingbox_size, save_filepath);
//}

//bool cpu_tsdf::TSDFWeightedPCA_Wrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models, const int component_num,
//                               const int max_iter,
//                               cpu_tsdf::TSDFHashing* mean_tsdf, Eigen::SparseVector<float> *mean_mat,
//                               Eigen::SparseMatrix<float, ColMajor> *base_mat,
//                               Eigen::MatrixXf *coeff_mat,
//                               Eigen::SparseMatrix<float, ColMajor> *weight_mat,
//                               Eigen::Vector3i *boundingbox_size,
//                               const std::string* save_filepath)
//{
//    // #samples: N, #dim: D, #principal_components: K
//    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
//    cout << "begin weighted pca" << endl;
//    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
//    cout << "1. convert TSDFs to sparse matrix" << endl;
//    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, weight_mat, mean_mat, mean_tsdf, boundingbox_size)) return false;
//    const int data_dim = centralized_data_mat.rows();
//    const int sample_num = centralized_data_mat.cols();
//    if (save_filepath)
//    {
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_original" + ".ply")).string();
//        // 1. save the original mats for matlab debug
//        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat, NULL, NULL, output_filename);
//        // 2. save the tsdfs
//        WPCASaveUncentralizedTSDFs(centralized_data_mat, *weight_mat, *mean_mat, *mean_tsdf,
//                                   *boundingbox_size, output_filename);
//        WPCASaveMeanTSDF(*mean_mat, *mean_tsdf, *weight_mat, *boundingbox_size, output_filename);
//    }
//    base_mat->resize(data_dim, sample_num);
//    coeff_mat->resize(component_num, sample_num);
//    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(*weight_mat);
//    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
//    // initialize base_mat U
//    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
//    cout << "data_dim: " << data_dim << endl;
//    cout << "sample_num: " << sample_num << endl;
//    cout << "compo_num: " << component_num << endl;
//    cerr << "2. Initial Estimate" << endl;
//    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
//    *base_mat = base_mat_row_major;
//    /// save
//    if (save_filepath)
//    {
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_init_estimated" + ".ply")).string();
//        // 1. save the initial mats for matlab debug
//        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
//                                base_mat, coeff_mat, output_filename);
//        // 2. save the tsdfs
//        WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
//                               *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
//    }
//    // do EM
//    cerr << "3. Do EM" << endl;
//    for (int i = 0; i < max_iter; ++i)
//    {
//        cout << "Iteration: \nEstep " << i <<  endl;
//        //E step
//        ComputeCoeffInEStep(centralized_data_mat, *weight_mat, *base_mat, coeff_mat);
//        //M step
//        cout << "Mstep " << i <<  endl;
//        ComputeBasisInMStep(centralized_data_mat_row_major, weight_mat_row_major, *coeff_mat, &base_mat_row_major);
//        *base_mat = base_mat_row_major;  // update colmajor base mat;
//        /// save
//        if (save_filepath)
//        {
//            string output_dir = bfs::path(*save_filepath).remove_filename().string();
//            string output_filename =
//                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                              + "_tsdf_wpca"
//                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                              + "_itr_" + boost::lexical_cast<string>(i) + ".ply")).string();
//            WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
//                                    base_mat, coeff_mat, output_filename);
//            WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
//                                   *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
//        }
//    }
//    return true;
//}



//bool cpu_tsdf::TSDFWeightedPCADeflation(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
//                                        const int component_num,
//                                        const int max_iter,
//                                        cpu_tsdf::TSDFHashing *mean_tsdf,
//                                        Eigen::SparseVector<float> *mean_mat,
//                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
//                                        Eigen::MatrixXf *coeff_mat, Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
//                                        Eigen::Vector3i *boundingbox_size, const string *save_filepath)
//{
//    // #samples: N, #dim: D, #principal_components: K
//    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
//    cout << "begin weighted pca" << endl;
//    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
//    cout << "1. convert TSDFs to sparse matrix" << endl;
//    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, weight_mat, mean_mat, mean_tsdf, boundingbox_size)) return false;
//    const int data_dim = centralized_data_mat.rows();
//    const int sample_num = centralized_data_mat.cols();
//    if (save_filepath)
//    {
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca_deflation"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_original" + ".ply")).string();
//        // 1. save the original mats for matlab debug
//        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat, NULL, NULL, output_filename);
//        // 2. save the tsdfs
//        WPCASaveUncentralizedTSDFs(centralized_data_mat, *weight_mat, *mean_mat, *mean_tsdf,
//                                   *boundingbox_size, output_filename);
//        WPCASaveMeanTSDF(*mean_mat, *mean_tsdf, *weight_mat, *boundingbox_size, output_filename);
//    }
//    base_mat->resize(data_dim, sample_num);
//    coeff_mat->resize(component_num, sample_num);
//    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(*weight_mat);
//    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
//    // initialize base_mat U
//    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
//    cout << "data_dim: " << data_dim << endl;
//    cout << "sample_num: " << sample_num << endl;
//    cout << "compo_num: " << component_num << endl;
//    cerr << "2. Initial Estimate" << endl;
//    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
//    *base_mat = base_mat_row_major;
//    if (save_filepath)
//    {
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca_deflation"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_init_estimated" + ".ply")).string();
//        // 1. save the initial mats for matlab debug
//        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
//                                base_mat, coeff_mat, output_filename);
//        // 2. save the tsdfs
//        WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
//                               *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
//    }
//    // do EM
//    cerr << "3. Do EM" << endl;
//    const float thresh = 1e-4;
//    Eigen::VectorXf prev_base(data_dim);
//    Eigen::VectorXf current_base(data_dim);
//    Eigen::VectorXf current_coeff(sample_num);
//    const int iteration_number = std::max(sample_num, 25);
//    for (int k = 0; k < component_num; ++k)
//    {
//        current_base = base_mat->col(k);
//        current_coeff = coeff_mat->row(k);
//        prev_base = current_base;
//        cout << "Computing " << k << "th component." << endl;
//        for (int i = 0; i < iteration_number; ++i)
//        {
//            cout << "Iteration: \nEstep " << i <<  endl;
//            //E step
//            ComputeCoeffInEStepOneVec(centralized_data_mat, *weight_mat, current_base, &current_coeff);
//            //M step
//            cout << "Mstep " << i <<  endl;
//            ComputeBasisInMStepOneVec(centralized_data_mat_row_major, weight_mat_row_major, current_coeff, &current_base);
//            ////////////////////////////////////////////////
//            /// save
//            if (save_filepath)
//            {
//                string output_dir = bfs::path(*save_filepath).remove_filename().string();
//                string output_filename =
//                        (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                                  + "_tsdf_wpca_deflation"
//                                                  + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                                  + "_comp_" + boost::lexical_cast<string>(k)
//                                                  + "_itr_" + boost::lexical_cast<string>(i)
//                                                  + ".ply")).string();
//                Eigen::SparseMatrix<float, ColMajor> temp_base = current_base.sparseView();
//                Eigen::MatrixXf temp_coeff = current_coeff;
//                WPCASaveRelatedMatrices(NULL, NULL, NULL,
//                                        &temp_base, &temp_coeff, output_filename);
//                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
//                        =
//                        (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
//                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval()
//                        + temp_base * temp_coeff.transpose().sparseView().eval();
//                WPCASaveTSDFsFromMats(temp_projected_data_mat, *mean_tsdf, *boundingbox_size, output_filename);
//            }
//            ////////////////////////////////////////////////
//            //test convergence
//            float t = (prev_base.normalized() - current_base.normalized()).cwiseAbs().sum();
//            cerr << "difference.. " << t << endl;
//            if (t < thresh)
//            {
//                cerr << "converge reached.. " << endl;
//                break;
//            }
//            prev_base = current_base;
//        }
//        for (int p = 0; p < data_dim; ++p)
//        {
//            if (abs(current_base[p]) > std::numeric_limits<double>::epsilon())
//            {
//                base_mat->coeffRef(p, k) = current_base[p];
//            }
//        }
//        (*coeff_mat).row(k) = current_coeff;
//        centralized_data_mat -= weight_mat->cwiseProduct(current_base.sparseView() * current_coeff.transpose());
//        centralized_data_mat_row_major = centralized_data_mat;
//        if (save_filepath)
//        {
//            string output_dir = bfs::path(*save_filepath).remove_filename().string();
//            string output_filename =
//                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                              + "_tsdf_wpca_deflation"
//                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                              + "_comp_" + boost::lexical_cast<string>(k) + ".ply")).string();
//            WPCASaveRelatedMatrices(&centralized_data_mat, NULL, NULL,
//                                    base_mat, coeff_mat, output_filename);
//            WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
//                                   *base_mat, *coeff_mat, k + 1, *boundingbox_size, output_filename);
//        }
//    }
//    return true;
//}

//bool cpu_tsdf::WeightedPCADeflationOrthogonal(
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
//        const int component_num, const int max_iter,
//        Eigen::SparseVector<float> *mean_mat,
//        Eigen::SparseVector<float>* mean_weight,
//        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
//        Eigen::MatrixXf *coeff_mat,
//        const PCAOptions& options)
//{
//    const int sample_num = samples.cols();
//    const int data_dim = samples.rows();
//    if (sample_num == 0 || data_dim == 0) return false;
//    // #samples: N, #dim: D, #principal_components: K
//    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
//    cout << "begin weighted pca" << endl;
//    cerr << "1. compute mean mat" << endl;
//    // ((Samples.*Weight)) * ones(N, 1) ./ Weight * ones(N, 1)
//    Eigen::VectorXf onesN = Eigen::MatrixXf::Ones(sample_num, 1);
//    // *mean_mat = (((samples.cwiseProduct(weights)) * onesN).cwiseQuotient( (weights * onesN) )).sparseView();
//    Eigen::SparseVector<float> sum_vec = ((samples.cwiseProduct(weights)) * onesN).sparseView();
//    Eigen::SparseVector<float> sum_weights = (weights * onesN).sparseView();
//    mean_mat->resize(data_dim);
//    mean_mat->reserve(sum_weights.nonZeros());
//    for (Eigen::SparseVector<float>::InnerIterator it(sum_weights); it; ++it)
//    {
//        const float cur_weight = it.value();
//        const int cur_idx = it.index();
//        if (cur_weight > 0)
//        {
//            mean_mat->insert(cur_idx) = sum_vec.coeff(cur_idx) / cur_weight;
//        }
//    }
//    *mean_weight = sum_weights / sample_num;
//    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat = samples - (*mean_mat) * onesN.transpose();
//    if (!options.save_path.empty())
//    {
//        const string* save_filepath = &options.save_path;
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca_deflation_ortho"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_original" + "pp.ply")).string();
//        // 1. save the original mats for matlab debug
//        // WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, &weights, NULL, NULL, output_filename);
//        // 2. save the tsdfs
////        WPCASaveUncentralizedTSDFs(centralized_data_mat, weights, *mean_mat,
////                                   options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
////                                   output_filename);
//        WPCASaveMeanTSDF(*mean_mat, weights,
//                         options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
//                         output_filename);
//    }
//    if (component_num == 0) return true;
//    base_mat->resize(data_dim, sample_num);
//    coeff_mat->resize(component_num, sample_num);
//    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(weights);
//    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
//    // initialize base_mat U
//    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
//    cout << "data_dim: " << data_dim << endl;
//    cout << "sample_num: " << sample_num << endl;
//    cout << "compo_num: " << component_num << endl;
//    cerr << "2. Initial Estimate" << endl;
//    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
//    *base_mat = base_mat_row_major;
//    if (!options.save_path.empty())
//    {
//        const string* save_filepath = &options.save_path;
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca_deflation_ortho"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_init_estimated" + ".ply")).string();
//        // 1. save the initial mats for matlab debug
//        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, &weights,
//                                base_mat, coeff_mat, output_filename);
//        // 2. save the tsdfs
//        WPCASaveRecoveredTSDFs(*mean_mat, *base_mat, *coeff_mat, component_num,
//                               options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
//                               output_filename);
//    }
//    // do EM
//    cerr << "3. Do EM" << endl;
//    const float thresh = 1e-4;
//    Eigen::VectorXf prev_base(data_dim);
//    Eigen::VectorXf current_base(data_dim);
//    Eigen::VectorXf current_coeff(sample_num);
//    const int iteration_number = std::max(sample_num, max_iter);
//    for (int k = 0; k < component_num; ++k)
//    {
//        current_base = base_mat->col(k);
//        current_coeff = coeff_mat->row(k);
//        prev_base = current_base;
//        cout << "Computing " << k << "th component." << endl;
//        for (int i = 0; i < iteration_number; ++i)
//        {
//            cout << "Iteration: \nEstep " << i <<  endl;
//            //E step
//            ComputeCoeffInEStepOneVec(centralized_data_mat, weights, current_base, &current_coeff);
//            //M step
//            cout << "Mstep " << i <<  endl;
//            ComputeBasisInMStepOneVec(centralized_data_mat_row_major, weight_mat_row_major, current_coeff, &current_base);
//            //orthogonalize:
//            float component_norm = 0;
//            OrthogonalizeVector(*base_mat, k, &current_base, &component_norm);
//            current_coeff *= component_norm;
//            ////////////////////////////////////////////////
//            /// save
//            if (!options.save_path.empty() && (i%10) == 0)
//            {
//                const string* save_filepath = &options.save_path;
//                string output_dir = bfs::path(*save_filepath).remove_filename().string();
//                string output_filename =
//                        (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                                  + "_tsdf_wpca_deflation_ortho"
//                                                  + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                                  + "_comp_" + boost::lexical_cast<string>(k)
//                                                  + "_itr_" + boost::lexical_cast<string>(i)
//                                                  + ".ply")).string();
//                Eigen::SparseMatrix<float, ColMajor> temp_base = current_base.sparseView();
//                Eigen::MatrixXf temp_coeff = current_coeff;
//                WPCASaveRelatedMatrices(NULL, NULL, NULL,
//                                        &temp_base, &temp_coeff, output_filename);
//                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
//                        =
//                        (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
//                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval()
//                        + temp_base * temp_coeff.transpose().sparseView().eval();
//                WPCASaveTSDFsFromMats(temp_projected_data_mat,
//                                      options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
//                                      output_filename);
////                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
////                        = (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
////                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval();
////                WPCASaveRecoveredTSDFs(temp_projected_data_mat, *mean_tsdf,
////                                       temp_base, temp_coeff, 1, *boundingbox_size, output_filename);
//            }
//            ////////////////////////////////////////////////
//            //test convergence
//            float t = (prev_base.normalized() - current_base.normalized()).cwiseAbs().sum();
//            cerr << "difference.. " << t << endl;
//            if (t < thresh)
//            {
//                cerr << "converge reached.. " << endl;
//                break;
//            }
//            prev_base = current_base;
//        }  // end for i (iteration)
//        for (int p = 0; p < data_dim; ++p)
//        {
//            if (abs(current_base[p]) > std::numeric_limits<double>::epsilon())
//            {
//                //base_mat->insert(p, k) = current_base[p];
//                base_mat->coeffRef(p, k) = current_base[p];
//            }
//        }
//        (*coeff_mat).row(k) = current_coeff;
//        centralized_data_mat -= weights.cwiseProduct(current_base.sparseView() * current_coeff.transpose());
//        centralized_data_mat_row_major = centralized_data_mat;
//        if (!options.save_path.empty())
//        {
//            const string* save_filepath = &options.save_path;
//            string output_dir = bfs::path(*save_filepath).remove_filename().string();
//            string output_filename =
//                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                              + "_tsdf_wpca_deflation"
//                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                              + "_comp_" + boost::lexical_cast<string>(k) + ".ply")).string();
//            WPCASaveRelatedMatrices(&centralized_data_mat, NULL, NULL,
//                                    base_mat, coeff_mat, output_filename);
//            WPCASaveRecoveredTSDFs(*mean_mat, *base_mat, *coeff_mat, k + 1,
//                                   options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
//                                   output_filename);
//        }
//    }  // end for k
//    return true;
//}

//bool cpu_tsdf::TSDFWeightedPCADeflationOrthogonal_Old(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
//                                                  const int component_num, const int max_iter,
//                                                  cpu_tsdf::TSDFHashing *mean_tsdf,
//                                                  Eigen::SparseVector<float> *mean_mat,
//                                                  Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
//                                                  Eigen::MatrixXf *coeff_mat, Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
//                                                  Eigen::Vector3i *boundingbox_size, const string *save_filepath)
//{
//    // #samples: N, #dim: D, #principal_components: K
//    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
//    cout << "begin weighted pca" << endl;
//    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
//    cout << "1. convert TSDFs to sparse matrix" << endl;
//    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, weight_mat, mean_mat, mean_tsdf, boundingbox_size)) return false;
//    const int data_dim = centralized_data_mat.rows();
//    const int sample_num = centralized_data_mat.cols();
//    if (save_filepath)
//    {
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca_deflation_ortho"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_original" + ".ply")).string();
//        // 1. save the original mats for matlab debug
//        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat, NULL, NULL, output_filename);
//        // 2. save the tsdfs
//        WPCASaveUncentralizedTSDFs(centralized_data_mat, *weight_mat, *mean_mat, *mean_tsdf,
//                                   *boundingbox_size, output_filename);
//        WPCASaveMeanTSDF(*mean_mat, *mean_tsdf, *weight_mat, *boundingbox_size, output_filename);
//    }
//    base_mat->resize(data_dim, sample_num);
//    coeff_mat->resize(component_num, sample_num);
//    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(*weight_mat);
//    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
//    // initialize base_mat U
//    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
//    cout << "data_dim: " << data_dim << endl;
//    cout << "sample_num: " << sample_num << endl;
//    cout << "compo_num: " << component_num << endl;
//    cerr << "2. Initial Estimate" << endl;
//    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
//    *base_mat = base_mat_row_major;
//    if (save_filepath)
//    {
//        string output_dir = bfs::path(*save_filepath).remove_filename().string();
//        string output_filename =
//                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                          + "_tsdf_wpca_deflation_ortho"
//                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                          + "_init_estimated" + ".ply")).string();
//        // 1. save the initial mats for matlab debug
//        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
//                                base_mat, coeff_mat, output_filename);
//        // 2. save the tsdfs
//        WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
//                               *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
//    }
//    // do EM
//    cerr << "3. Do EM" << endl;
//    const float thresh = 1e-4;
//    Eigen::VectorXf prev_base(data_dim);
//    Eigen::VectorXf current_base(data_dim);
//    Eigen::VectorXf current_coeff(sample_num);
//    const int iteration_number = std::max(sample_num, 25);
//    for (int k = 0; k < component_num; ++k)
//    {
//        current_base = base_mat->col(k);
//        current_coeff = coeff_mat->row(k);
//        prev_base = current_base;
//        cout << "Computing " << k << "th component." << endl;
//        for (int i = 0; i < iteration_number; ++i)
//        {
//            cout << "Iteration: \nEstep " << i <<  endl;
//            //E step
//            ComputeCoeffInEStepOneVec(centralized_data_mat, *weight_mat, current_base, &current_coeff);
//            //M step
//            cout << "Mstep " << i <<  endl;
//            ComputeBasisInMStepOneVec(centralized_data_mat_row_major, weight_mat_row_major, current_coeff, &current_base);
//            //orthogonalize:
//            float component_norm = 0;
//            OrthogonalizeVector(*base_mat, k, &current_base, &component_norm);
//            current_coeff *= component_norm;
//            ////////////////////////////////////////////////
//            /// save
//            if (save_filepath)
//            {
//                string output_dir = bfs::path(*save_filepath).remove_filename().string();
//                string output_filename =
//                        (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                                  + "_tsdf_wpca_deflation_ortho"
//                                                  + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                                  + "_comp_" + boost::lexical_cast<string>(k)
//                                                  + "_itr_" + boost::lexical_cast<string>(i)
//                                                  + ".ply")).string();
//                Eigen::SparseMatrix<float, ColMajor> temp_base = current_base.sparseView();
//                Eigen::MatrixXf temp_coeff = current_coeff;
//                WPCASaveRelatedMatrices(NULL, NULL, NULL,
//                                        &temp_base, &temp_coeff, output_filename);
//                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
//                        =
//                        (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
//                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval()
//                        + temp_base * temp_coeff.transpose().sparseView().eval();
//                WPCASaveTSDFsFromMats(temp_projected_data_mat, *mean_tsdf, *boundingbox_size, output_filename);
////                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
////                        = (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
////                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval();
////                WPCASaveRecoveredTSDFs(temp_projected_data_mat, *mean_tsdf,
////                                       temp_base, temp_coeff, 1, *boundingbox_size, output_filename);
//            }
//            ////////////////////////////////////////////////
//            //test convergence
//            float t = (prev_base.normalized() - current_base.normalized()).cwiseAbs().sum();
//            cerr << "difference.. " << t << endl;
//            if (t < thresh)
//            {
//                cerr << "converge reached.. " << endl;
//                break;
//            }
//            prev_base = current_base;
//        }
//        for (int p = 0; p < data_dim; ++p)
//        {
//            if (abs(current_base[p]) > std::numeric_limits<double>::epsilon())
//            {
//                //base_mat->insert(p, k) = current_base[p];
//                base_mat->coeffRef(p, k) = current_base[p];
//            }
//        }
//        (*coeff_mat).row(k) = current_coeff;
//        centralized_data_mat -= weight_mat->cwiseProduct(current_base.sparseView() * current_coeff.transpose());
//        centralized_data_mat_row_major = centralized_data_mat;
//        if (save_filepath)
//        {
//            string output_dir = bfs::path(*save_filepath).remove_filename().string();
//            string output_filename =
//                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
//                                              + "_tsdf_wpca_deflation"
//                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                              + "_comp_" + boost::lexical_cast<string>(k) + ".ply")).string();
//            WPCASaveRelatedMatrices(&centralized_data_mat, NULL, NULL,
//                                    base_mat, coeff_mat, output_filename);
//            WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
//                                   *base_mat, *coeff_mat, k + 1, *boundingbox_size, output_filename);
//        }
//    }
//    return true;
//}







