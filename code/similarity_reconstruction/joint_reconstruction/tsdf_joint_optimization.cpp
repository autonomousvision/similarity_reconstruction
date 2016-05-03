/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "tsdf_joint_optimization.h"
#include <vector>
#include <string>
#include <algorithm>

#include <Eigen/Sparse>

#include <boost/lexical_cast.hpp>
#include "tsdf_operation/tsdf_pca.h"
#include "tsdf_operation/tsdf_slice.h"
#include "tsdf_operation/tsdf_align.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_clean.h"
#include "tsdf_operation/tsdf_transform.h"

using namespace cpu_tsdf;
using namespace std;

namespace tsdf_optimization {

void ComputeModelAverageScales(
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
        const std::vector<int> &sample_model_assign,
        const int model_num,
        std::vector<Eigen::Vector3f> *model_average_scales)
{
    const int sample_num = sample_model_assign.size();
    model_average_scales->resize(model_num);
    for (int i = 0; i < model_average_scales->size(); ++i)
    {
        (*model_average_scales)[i].setZero();
    }
    std::vector<int> sample_model_count(model_num, 0);
    for (int i = 0; i < sample_num; ++i)
    {
        Eigen::Vector3f scale = obbs[i].SideLengths();
        int cur_model = sample_model_assign[i];
        (*model_average_scales)[cur_model] += scale;
        sample_model_count[cur_model]++;
    }
    for (int i = 0; i < model_num; ++i)
    {
        (*model_average_scales)[i] /= (float)sample_model_count[i];
    }
}

void InitializeOptimization(const cpu_tsdf::TSDFHashing &scene_tsdf,
                            const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
                            const std::vector<float>& obb_scores,
                            const std::vector<int> sample_model_assign,
                            tsdf_utility::OptimizationParams& params,
                            std::vector<Eigen::SparseVector<float> > *model_means,
                            std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
                            std::vector<Eigen::VectorXf> *projected_coeffs,
                            std::vector<Eigen::Vector3f> *model_average_scales,
                            Eigen::SparseMatrix<float, Eigen::ColMajor>* reconstructed_sample_weights) {
    const int sample_num = obbs.size();
    const int model_num = *(max_element(sample_model_assign.begin(), sample_model_assign.end())) + 1;
    const int feat_dim = params.sample_size.prod();

    // initial model_average_scales
    ComputeModelAverageScales(obbs, sample_model_assign, model_num, model_average_scales);

    // initialize tsdf samples
    Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
    Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
    cpu_tsdf::ExtractSamplesFromOBBs(scene_tsdf, obbs, params.sample_size, params.min_meshing_weight,
                                     &samples, &weights);

    // using mean as initial model
    std::vector<Eigen::SparseVector<float>> model_mean_weights;
    std::vector<double> cur_gammas(samples.cols(), 0);
    // using the highest scored samples as initial mean
    // detection score threshold > 0.5
    //const float score_thresh = 0.5;
    //for (int i = 0; i < obbs.size(); ++i) {
    //    if (obb_scores[i] < score_thresh) {
    //        cur_gammas[i] = 1.0;  // label as outlier for mean computation
    //    }
    //}
    OptimizeModelAndCoeff(
                samples,
                weights,
                sample_model_assign,
                cur_gammas,
                0, 0,
                model_means,
                &model_mean_weights,
                model_bases,
                projected_coeffs,
                params
                );
    vector<vector<int>> cluster_sample_idx;
    GetClusterSampleIdx(sample_model_assign, vector<double>(), model_num, &cluster_sample_idx);
    // the weights for each reconstructed sample. Initialized to the weight of the mean of each model (cluster)
    reconstructed_sample_weights->resize(feat_dim, sample_num);
    for (int model_i = 0; model_i < model_num; ++model_i)
    {
        for (int sample_i = 0; sample_i < cluster_sample_idx[model_i].size(); ++sample_i)
        {
            reconstructed_sample_weights->col(cluster_sample_idx[model_i][sample_i]) = model_mean_weights[model_i];
        }
    }
    //////////////////////////////////
    cpu_tsdf::TSDFGridInfo tsdf_info(scene_tsdf, params.sample_size, params.min_meshing_weight);
    WriteOBBsAndTSDFs(scene_tsdf, obbs, tsdf_info, params.save_path + "_init.ply");
    std::vector<Eigen::SparseVector<float>> recon_samples;
    PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, sample_model_assign, &recon_samples);
    WriteTSDFsFromMatWithWeight(SparseVectorsToEigenMat(recon_samples), *reconstructed_sample_weights, tsdf_info, params.save_path + "_initmean.ply" );
    WriteTSDFsFromMatWithWeight(SparseVectorsToEigenMat(recon_samples), *reconstructed_sample_weights, tsdf_info, params.save_path + "_initmean.ply" );
    std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_original_pos;
    std::vector<Eigen::Affine3f> temp_affines;
    for (int i =0 ; i < obbs.size(); ++i) { temp_affines.push_back(obbs[i].AffineTransform()); }
    cpu_tsdf::ReconstructTSDFsFromPCAOriginPos(
                            *model_means,
                            *model_bases,
                            *projected_coeffs,
                            *reconstructed_sample_weights,
                            sample_model_assign,
                            temp_affines,
                            tsdf_info,
                            scene_tsdf.voxel_length(),
                            scene_tsdf.offset(),
                            &reconstructed_samples_original_pos);
    WriteTSDFModels(reconstructed_samples_original_pos, params.save_path+ "trans_mean.ply", false, true, params.min_meshing_weight);
    //////////////////////////////////
}

bool JointClusteringAndModelLearning(
        const TSDFHashing &scene_tsdf,
        tsdf_utility::OptimizationParams &params,
        std::vector<Eigen::SparseVector<float> > *model_means,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        std::vector<int> *sample_model_assign,
        std::vector<double> *outlier_gammas,
        std::vector<tsdf_utility::OrientedBoundingBox> *obbs,
        std::vector<Eigen::Vector3f> *model_average_scales,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *reconstructed_sample_weights)
{
    string params_init_save_path = params.save_path;
    for (int i = 0; i < params.opt_max_iter; ++i)
    {
        bfs::path prefix(params.save_path);
        bfs::path write_dir(prefix.parent_path()/(std::string("iteration_") + boost::lexical_cast<string>(i)));
        bfs::create_directories(write_dir);

        bfs::path write_dir_block1(write_dir/"block1_alignment");
        bfs::create_directories(write_dir_block1);
        string cur_save_path = (write_dir_block1/prefix.stem()).string();
        params.save_path = (write_dir_block1/prefix.stem()).string() + "_TransformScale";

        // block 1
        Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
        Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
        OptimizeTransformAndScale(scene_tsdf,
                                  *model_means, *model_bases, *projected_coeffs,
                                  *sample_model_assign,
                                  *outlier_gammas,
                                  params,
                                  obbs,
                                  model_average_scales,
                                  *reconstructed_sample_weights,
                                  &samples,
                                  &weights);
        params.save_path = cur_save_path + "_TransformScale_EndSave";
        //WriteAffineTransformsAndTSDFs(scene_tsdf, *affine_transforms, params);
        {
            tsdf_utility::OutputOBBsAsPly(*obbs, params.save_path);
            TSDFGridInfo tsdf_info(scene_tsdf, params.sample_size, 0);
            cpu_tsdf::WriteTSDFsFromMatWithWeight(samples, *reconstructed_sample_weights, tsdf_info, params.save_path + "_transformscale.ply");
            cpu_tsdf::WriteTSDFsFromMatWithWeight_Matlab(samples, weights, tsdf_info,
                                                         params.save_path + "_TransformScale.mat");
        }

        // block 2
        bfs::path write_dir_block2(write_dir/"block2_pca");
        bfs::create_directories(write_dir_block2);
        cur_save_path =  (write_dir_block2/prefix.stem()).string();
        params.save_path = (write_dir_block2/prefix.stem()).string() + "_ModelCoeffPCA";

        float pos_trunc, neg_trunc;
        scene_tsdf.getDepthTruncationLimits(pos_trunc, neg_trunc);
        Eigen::SparseMatrix<float, Eigen::ColMajor> cleaned_weights = weights;
        Eigen::SparseMatrix<float, Eigen::ColMajor> valid_weights;
        // cleaned_weight; valid_obs_weight_mat: 0-max_weight binary matrix
        // not considering outliers for observation counting
        // only consider observations appeared in multiple samples
        CleanNoiseInSamples(
                    samples, *sample_model_assign, *outlier_gammas, &cleaned_weights, &valid_weights,
                    params.noise_observation_thresh, pos_trunc, neg_trunc);
        // clean isolated parts
        // clean noise also for outliers, as the outliers are also extracted from affine transforms in the previous step
        CleanTSDFSampleMatrix(scene_tsdf, samples, params.sample_size, params.noise_connected_component_thresh, &cleaned_weights, -1, 2);
        std::vector<Eigen::SparseVector<float>> model_mean_weights;
        OptimizeModelAndCoeff(samples, cleaned_weights, *sample_model_assign,
                              *outlier_gammas,
                              params.pc_num, params.pc_max_iter,
                              model_means, &model_mean_weights, model_bases, projected_coeffs,
                              params);
        // recompute reconstructed model's weights
        std::vector<Eigen::SparseVector<float>> recon_samples;
        PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
        Eigen::SparseMatrix<float, Eigen::ColMajor> recon_sample_mat = cpu_tsdf::SparseVectorsToEigenMat(recon_samples);
        CleanTSDFSampleMatrix(scene_tsdf, recon_sample_mat, params.sample_size, params.noise_connected_component_thresh, &valid_weights, -1, 2);
        *reconstructed_sample_weights = valid_weights;
        TSDFGridInfo tsdf_info(scene_tsdf, params.sample_size, 0);
        tsdf_info.offset(Eigen::Vector3f(-0.5,-0.5,-0.5));
        cpu_tsdf::WriteTSDFsFromMatWithWeight(recon_sample_mat, *reconstructed_sample_weights, tsdf_info, params.save_path + "_ModelCoeffPCA_reconweight.ply");
        cpu_tsdf::WriteTSDFsFromMatWithWeight_Matlab(recon_sample_mat, *reconstructed_sample_weights, tsdf_info, params.save_path + "_ModelCoeffPCA_reconweight.mat");

        // block 3
        if (1)
        {
            bfs::path write_dir_block3(write_dir/"block3_cluster");
            bfs::create_directories(write_dir_block3);
            cur_save_path = (write_dir_block3/prefix.stem()).string();
            params.save_path = (write_dir_block3/prefix.stem()).string() + "_Cluster";

            cout << "do cluster" << endl;
            std::vector<float> cluster_assignment_error;
            OptimizePCACoeffAndCluster(samples, weights,
                                       *reconstructed_sample_weights,
                                       *model_means, *model_bases,
                                       *obbs,
                                       params,
                                       sample_model_assign,
                                       outlier_gammas,
                                       projected_coeffs,
                                       &cluster_assignment_error,
                                       model_average_scales);
            WriteSampleClusters(*sample_model_assign,  *model_average_scales, *outlier_gammas, cluster_assignment_error, params.save_path + "_cluster.txt");
            //{
            //    std::vector<Eigen::SparseVector<float>> recon_samples;
            //    params.save_path = cur_save_path + "_Cluster_recon";
            //    PCAReconstructionResult(*model_means, *model_bases, *projected_coeffs, *sample_model_assign, &recon_samples);
            //    std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
            //    // ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, params, &recon_tsdfs);
            //    ConvertDataVectorsToTSDFsNoWeight(recon_samples, params, &recon_tsdfs);
            //    std::vector<cpu_tsdf::TSDFHashing::Ptr> transformed_recon_tsdfs;
            //    WriteTSDFModels(recon_tsdfs, params.save_path + "_canonical.ply", false, true, params.min_model_weight);
            //    TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
            //    WriteTSDFModels(transformed_recon_tsdfs, params.save_path + "_transformed.ply", false, true, params.min_model_weight);

            //    params.save_path = cur_save_path + "_Cluster_recon_with_weights";
            //    ConvertDataVectorsToTSDFsWithWeight(recon_samples, weights, params, &recon_tsdfs);
            //    WriteTSDFModels(recon_tsdfs, params.save_path + "_canonical.ply", false, true, params.min_model_weight);
            //    transformed_recon_tsdfs.clear();
            //    TransformTSDFs(recon_tsdfs, *affine_transforms, &transformed_recon_tsdfs, &scene_voxel_length);
            //    WriteTSDFModels(transformed_recon_tsdfs, params.save_path + "_transformed.ply", false, true, params.min_model_weight);
            //    //params.save_path = params_old_save_path;
            //}
        }

        // put the not-well reconstructed samples



        if (1)
        {
#if 0
            params.save_path = cur_save_path;
            using namespace std;
            cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, params.sample_size, params.min_meshing_weight);
            std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_original_pos;
            Eigen::SparseMatrix<float, Eigen::ColMajor> original_weighmat = *reconstructed_sample_weights;
            float kkfactors[1]={0.5};
            for (int kki = 0; kki < 1; ++kki)
            {
                float kkfactor = kkfactors[kki];

                cpu_tsdf::ReconstructTSDFsFromPCAOriginPos(
                            *model_means,
                            *model_bases,
                            *projected_coeffs,
                            *reconstructed_sample_weights,
                            *sample_model_assign,
                            *obbs,
                            grid_info,
                            scene_tsdf.voxel_length(),
                            scene_tsdf.offset(),
                            &reconstructed_samples_original_pos);
                for (int i = 0; i < reconstructed_samples_original_pos.size(); ++i)
                {
                    cout << "cleaning " << i <<"th model" << endl;
                    cpu_tsdf::CleanTSDF(reconstructed_samples_original_pos[i], params.noise_connected_component_thresh);
                }
                cpu_tsdf::WriteTSDFModels(reconstructed_samples_original_pos,
                                          (bfs::path(params.save_path).replace_extension
                                           (string("_recon_single_tsdf") + "_iter_" + boost::lexical_cast<std::string>(i)  + "_" + utility::double2str(kkfactor)+ ".ply")).string(),
                                          false, true, params.min_meshing_weight, *outlier_gammas);
                cout << "begin merging... " << endl;
                cpu_tsdf::TSDFHashing::Ptr cur_scene(new TSDFHashing);
                *cur_scene = scene_tsdf;
                cpu_tsdf::CleanTSDF(cur_scene, 100);

                std::vector<Eigen::Affine3f> obj_affines;
                for (int i = 0; i < obbs->size(); ++i)
                {
                    //if ((*sample_model_assign)[i] == 1)
                    {
                        obj_affines.push_back((*obbs)[i].AffineTransformOriginAsCenter());
                    }
                }
                cpu_tsdf::ScaleTSDFParts(cur_scene.get(), obj_affines, kkfactor);
                for (int i = 0; i < reconstructed_samples_original_pos.size(); ++i)
                {
                    if ((*outlier_gammas)[i] > 1e-5)
                    {
                        reconstructed_samples_original_pos[i].reset();
                    }
                }
                reconstructed_samples_original_pos.erase(std::remove_if(reconstructed_samples_original_pos.begin(), reconstructed_samples_original_pos.end(), [](const  cpu_tsdf::TSDFHashing::Ptr& ptr){
                    return !bool(ptr);
                }), reconstructed_samples_original_pos.end());
                cpu_tsdf::MergeTSDFs(reconstructed_samples_original_pos, cur_scene.get());
                cur_scene->DisplayInfo();
                cpu_tsdf::CleanTSDF(cur_scene, 100);
                cpu_tsdf::WriteTSDFModel(cur_scene,
                                         (bfs::path(params.save_path).replace_extension
                                          (string("_raftermerge_2ndclean_100") + "_iter_" + boost::lexical_cast<std::string>(i) + "_" + utility::double2str(kkfactor) + ".ply")).string(),
                                         true, true, params.min_meshing_weight);
            } // for kki
#endif
        }
    params.save_path = params_init_save_path;
    }  // optimization iteration
    params.save_path = params_init_save_path;
    return true;
}

bool OptimizeTransformAndScale(
        const TSDFHashing &scene_tsdf,
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const std::vector<int> &model_assign_idx,
        const std::vector<double> &outlier_gammas,
        tsdf_utility::OptimizationParams &params,
        std::vector<tsdf_utility::OrientedBoundingBox> *obbs,
        std::vector<Eigen::Vector3f> *model_average_scales,
        Eigen::SparseMatrix<float, Eigen::ColMajor>& recon_weights,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
{
    const int sample_num = model_assign_idx.size();
    // 1. from PCA components to model_reconstructed_samples
    std::vector<Eigen::SparseVector<float>> reconstructed_samples;
    PCAReconstructionResult(model_means, model_bases, projected_coeffs, model_assign_idx, &reconstructed_samples);
    // extracting the weights of the samples
    //ExtractSamplesFromOBBs(
    //                        scene_tsdf,
    //                        *obbs,
    //                        params.sample_size,
    //                        params.min_meshing_weight,
    //                        samples,
    //                        weights
    //                        );
    std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_sample_tsdfs(sample_num);
    cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, params.sample_size, params.min_meshing_weight);
    // ConvertDataVectorsToTSDFsWithWeight(reconstructed_samples, *weights, grid_info, &reconstructed_sample_tsdfs);
    ConvertDataVectorsToTSDFsWithWeight(reconstructed_samples, recon_weights, grid_info, &reconstructed_sample_tsdfs);
    cpu_tsdf::WriteTSDFModels(reconstructed_sample_tsdfs, params.save_path + "_debug_house_model_tobealigned1.txt", false, true, 0);
    std::vector<const TSDFHashing*> ptr_reconstructed_samples_tsdf(sample_num);
    for (int i = 0; i < sample_num; ++i) ptr_reconstructed_samples_tsdf[i] = reconstructed_sample_tsdfs[i].get();
    // 2. optimize
    // do it for each category seperately gives better result
    //////////////////////////////////////////////////////////////
    const int model_number = *(std::max_element(model_assign_idx.begin(), model_assign_idx.end())) + 1;
    vector<vector<int>> cluster_sample_idx;
    GetClusterSampleIdx(model_assign_idx,  outlier_gammas, model_number, &cluster_sample_idx);
    for (int i = 0; i < model_number; ++i)
    {
        int sample_num_cur_cluster = cluster_sample_idx[i].size();
        std::vector<const TSDFHashing*> recon_samples_tsdf_cur_cluster(sample_num_cur_cluster);
        std::vector<int> model_assign_idx_cur_cluster(sample_num_cur_cluster);
        std::vector<double> outlier_gammas_cur_cluster(sample_num_cur_cluster);
        std::vector<tsdf_utility::OrientedBoundingBox> obbs_cur_cluster(sample_num_cur_cluster);
        std::vector<Eigen::Vector3f> model_average_scale_cur_cluster(1);
        model_average_scale_cur_cluster[0] = (*model_average_scales)[i];
        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
        {
            int cur_sample_idx = cluster_sample_idx[i][j];
            recon_samples_tsdf_cur_cluster[j] = ptr_reconstructed_samples_tsdf[cur_sample_idx];
            model_assign_idx_cur_cluster[j] = 0;
            outlier_gammas_cur_cluster[j] = outlier_gammas[cur_sample_idx];
            obbs_cur_cluster[j] = (*obbs)[cur_sample_idx];
        }
        cpu_tsdf::OptimizeTransformAndScalesImplRobustLoss1(
                                scene_tsdf,
                                recon_samples_tsdf_cur_cluster,
                                model_assign_idx_cur_cluster,
                                outlier_gammas_cur_cluster,
                                params,
                                &obbs_cur_cluster,
                                &model_average_scale_cur_cluster/*#model/clusters, input and output*/
                                );
        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
        {
            int cur_sample_idx = cluster_sample_idx[i][j];
            (*obbs)[cur_sample_idx] = obbs_cur_cluster[j];
            (*model_average_scales)[i] = model_average_scale_cur_cluster[0];
        }
    }  // end for i
    //////////////////////////////////////////////////////////////
    //cpu_tsdf::OptimizeTransformAndScalesImplRobustLoss1(
    //            scene_tsdf,
    //            ptr_reconstructed_samples_tsdf,
    //            model_assign_idx,
    //            outlier_gammas,
    //            params,
    //            obbs,
    //            model_average_scales /*#model/clusters, input and output*/
    //            );
    ////OptimizeTransformAndScalesImplRobustLoss(
    ////            scene_tsdf,
    ////            ptr_reconstructed_samples_tsdf,
    ////            model_assign_idx,
    ////            outlier_gammas,
    ////            options,
    ////            options.save_path,
    ////            affine_transforms, /*#samples, input and output*/
    ////            model_average_scales /*#model/clusters, input and output*/
    ////            );
    // 3. get sample data vectors and weight vectors after optimization
    ExtractSamplesFromOBBs(scene_tsdf,
                           *obbs,
                           params.sample_size,
                           params.min_meshing_weight,
                           samples,
                           weights
                           );
    //ExtractSamplesFromAffineTransform(
    //            scene_tsdf,
    //            *affine_transforms, /*size: #samples*/
    //            grid_info,
    //            samples,
    //            weights
    //            );
    return true;
}

bool OptimizePCACoeffAndCluster(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &reconstructed_sample_weights,
        const std::vector<Eigen::SparseVector<float> > &cluster_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &cluster_bases,
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
        const tsdf_utility::OptimizationParams &params, std::vector<int> *cluster_assignment,
        std::vector<double> *outlier_gammas,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        std::vector<float> *cluster_assignment_error,
        std::vector<Eigen::Vector3f> *cluster_average_scales)
{
    utility::OutputVectorTemplate(params.save_path + "debug_initial_cluster.txt", *cluster_assignment);
    using namespace std;
    const int sample_num = samples.cols();
    const int cluster_num = cluster_means.size();

    std::vector<Eigen::VectorXf> min_cluster_coeffs(sample_num);
    std::vector<int> min_cluster_idx(sample_num);
    std::vector<float> min_cluster_error(sample_num, FLT_MAX);
    std::vector<float> debug_min_cluster_recon_error(sample_num, FLT_MAX);
    std::vector<float> debug_min_cluster_scale_error(sample_num, FLT_MAX);

    Eigen::MatrixXf cur_projected_coeffs;
    vector<float> squared_errors(sample_num);
    for (int cluster_i = 0; cluster_i < cluster_num; ++cluster_i) {
        // all the samples are projected to current cluster
        WeightedPCAProjectionMultipleSamples(samples, weights,
                                             reconstructed_sample_weights,
                                             (cluster_means[cluster_i]), (cluster_bases[cluster_i]),
                                             &cur_projected_coeffs, &(squared_errors[0]));
        // switch cluster
        for (int sample_i = 0; sample_i < sample_num; ++sample_i) {
            float origin_sq_error = squared_errors[sample_i];
            float scale_sq_error = (params.lambda_average_scale* ((obbs[sample_i].SideLengths() - (*cluster_average_scales)[cluster_i]).squaredNorm()));
            squared_errors[sample_i] += scale_sq_error;
            if (squared_errors[sample_i] < min_cluster_error[sample_i]) {
                min_cluster_idx[sample_i] = cluster_i;
                min_cluster_error[sample_i] = squared_errors[sample_i];
                if (cur_projected_coeffs.size() > 0)
                    min_cluster_coeffs[sample_i] = cur_projected_coeffs.col(sample_i);
        debug_min_cluster_recon_error[sample_i] = origin_sq_error;
        debug_min_cluster_scale_error[sample_i] = scale_sq_error;
            }
        }
    }
    *cluster_assignment = std::move(min_cluster_idx);
    *projected_coeffs = std::move(min_cluster_coeffs);
    *cluster_assignment_error = std::move(min_cluster_error);

    utility::OutputVectorTemplate(params.save_path + "opt_initial_cluster.txt", *cluster_assignment);
    utility::OutputVectorTemplate(params.save_path + "cluster_recon_err.txt", debug_min_cluster_recon_error);
    utility::OutputVectorTemplate(params.save_path + "cluster_scale_err.txt", debug_min_cluster_scale_error);
    utility::OutputVectorTemplate(params.save_path + "cluster_assign_err.txt", *(cluster_assignment_error));
    // compute outlier_gammas
    for (int i = 0; i < (*cluster_assignment).size(); ++i)
    {
        double err_i = sqrt((*cluster_assignment_error)[i]) - params.lambda_outlier;
        //cout << "outlier: " << i << " " << err_i << endl;
        //cout << "outlier detail: " << i << " " << (params.lambda_average_scale * ((obbs[i].SideLengths() - (*cluster_average_scales)[(*cluster_assignment)[i]]).squaredNorm())) << " " << squared_errors[i] - (pca_options.lambda_scale_diff * ((scale - (*cluster_average_scales)[(*cluster_assignment)[i]]).squaredNorm())) << endl;
        err_i = err_i >= 0 ? err_i:0;
        (*outlier_gammas)[i] = err_i;
    }
    ComputeModelAverageScales(obbs, *cluster_assignment, cluster_num, cluster_average_scales);
    return true;
}

bool WriteSampleClusters(const std::vector<int> &sample_model_assign, const std::vector<Eigen::Vector3f> &model_average_scales, const std::vector<double> &outlier_gammas, const std::vector<float> &cluster_assignment_error, const string &save_path)
{
    FILE* hf = fopen(save_path.c_str(), "w");
    for (int i = 0; i < sample_model_assign.size(); ++i)
    {
        fprintf(hf, "%8d \t", i);
        fprintf(hf, "%8d \t", sample_model_assign[i]);
        fprintf(hf, "gamma: %10.7f\t", outlier_gammas[i]);
        if (!cluster_assignment_error.empty())
            fprintf(hf, "assign_error: %10.7f\n", cluster_assignment_error[i]);
        else
            fprintf(hf, "\n");
    }
    fprintf(hf, "\n");
    for (int i = 0; i < model_average_scales.size(); ++i)
    {
        fprintf(hf, "%dth model, scale-x-y-z: %f %f %f\n", i, model_average_scales[i][0], model_average_scales[i][1], model_average_scales[i][2]);
    }
    fclose(hf);
}

bool WeightedPCAProjectionMultipleSamples(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
                                          const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
                                          const Eigen::SparseMatrix<float, Eigen::ColMajor> &reconstructed_sample_weights,
                                          const Eigen::SparseVector<float> &mean_mat,
                                          const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
                                          Eigen::MatrixXf *projected_coeffs,
                                          float *squared_errors)
{
    int sample_num = samples.cols();
    int base_num = base_mat.cols();
    if (base_mat.rows() * base_mat.cols() > 0)
    {
        projected_coeffs->resize(base_num, sample_num);
    }
    else
    {
        projected_coeffs->resize(0, 0);
    }
    for (int i = 0; i < sample_num; ++i)
    {
        Eigen::VectorXf current_coeff;
        WeightedPCAProjectionOneSample(samples.col(i), weights.col(i), reconstructed_sample_weights.col(i), mean_mat, base_mat, &current_coeff, &(squared_errors[i]));
        if (current_coeff.size() > 0)
        {
            projected_coeffs->col(i) = current_coeff;
        }
    }
    return true;
}

bool WeightedPCAProjectionOneSample(const Eigen::SparseVector<float> &sample,
                                    const Eigen::SparseVector<float> &weight,
                                    const Eigen::SparseVector<float> &reconstruct_weight,
                                    const Eigen::SparseVector<float> &mean_mat,
                                    const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
                                    Eigen::VectorXf *projected_coeff,
                                    float *squared_error)
{
    using namespace Eigen;
    Eigen::SparseVector<float> centralized_sample = sample - mean_mat;
    Eigen::SparseVector<float> recon_weight_bin = reconstruct_weight;
    // thresholding
    for (SparseVector<float>::InnerIterator it(recon_weight_bin); it; ++it)
    {
        if (it.value() > 1e-5) {
            it.valueRef() = 1;
        }
    }
    Eigen::VectorXf dense_weight = weight.cwiseProduct(recon_weight_bin);
    float dense_weight_nnz = (dense_weight.array() > 0).sum();
    if (base_mat.rows() * base_mat.cols() > 0) {
        // ||w^T(d - Bc)||_2
        Eigen::MatrixXf A = (base_mat.transpose() * dense_weight.asDiagonal() * base_mat).eval();
        *projected_coeff =
                (A).jacobiSvd(ComputeThinU | ComputeThinV).
                solve(Eigen::MatrixXf((base_mat.transpose() * dense_weight.asDiagonal() * centralized_sample).eval()));
        Eigen::VectorXf diff = Eigen::VectorXf(centralized_sample) - (base_mat * (*projected_coeff));
        *squared_error = diff.transpose() * dense_weight.asDiagonal() * diff;
    } else {
        projected_coeff->resize(0);
        *squared_error = (centralized_sample.transpose() * dense_weight.asDiagonal() * centralized_sample).eval().coeff(0, 0);
    }
    // mean error  scaled by the sample size
    //(*squared_error) /= dense_weight_nnz;
    //(*squared_error) *= dense_weight.size();
    return true;
}

}
