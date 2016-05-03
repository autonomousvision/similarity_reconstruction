#include <iostream>
#include <string>
#include <vector>

#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include "detection/detector.h"
#include "detection/detect_sample.h"
#include "tsdf_operation/tsdf_slice.h"
#include "tsdf_operation/tsdf_align.h"
// #include "tsdf_operation/tsdf_joint_align.h"
#include "tsdf_representation/tsdf_hash.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/utilities/pcl_utility.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_slice.h"
#include "tsdf_operation/tsdf_clean.h"
#include "detection/detection_utility.h"
#include "detection/detector.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "tsdf_operation/tsdf_utility.h"
#include "tsdf_joint_optimization.h"

using std::vector;
using std::string;
using namespace std;

int
main (int argc, char** argv)
{
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;
    bpo::options_description opts_desc("Allowed options");
    string input_scene_file;
    string detect_obb_file;
    string output_prefix;
    float min_meshing_weight;
    int pc_num;
    int optimize_max_iter;

    float lambda_average_scale;
    float lambda_observation;
    float lambda_regularization;
    float lambda_outlier;

    float noise_observation_thresh = 3;
    float noise_connected_component_thresh = -1;

    opts_desc.add_options()
            ("help,h", "produce help message")
            ("scene_file", bpo::value<std::string>(&input_scene_file)->required(), "input scene tsdf file")
            ("detect_obb_file", bpo::value<std::string>(&detect_obb_file)->required(), "detected oriented bounging box file")
            ("output_prefix", bpo::value<std::string>(&output_prefix)->required (), "output ply path")
            ("mesh_min_weight", bpo::value<float>(&min_meshing_weight)->default_value(0.0), "minimum weight doing marching cubes")
            ("pca_number", bpo::value<int>(&pc_num)->default_value(0), "principal component number")
            ("optimize_max_iter", bpo::value<int>(&optimize_max_iter)->default_value(3), "max iteration number")
            ("lambda_avg_scale", bpo::value<float>(&lambda_average_scale)->default_value(100), "average scale term weight")
            ("lambda_obs", bpo::value<float>(&lambda_observation)->default_value(0.0), "observation term weight")
            ("lambda_reg", bpo::value<float>(&lambda_regularization)->default_value(50), "regularization term weight")
            ("lambda_outlier", bpo::value<float>(&lambda_outlier)->default_value(50000), "lambda_outlier_thresh")
            ("noise_observation_thresh", bpo::value<float>(&noise_observation_thresh)->default_value(2), "observed voxels less than the threshold will be removed")
            ("noise_connected_component_thresh", bpo::value<float>(&noise_connected_component_thresh)->default_value(-1), "connected component parts smaller than this thresh wil be removed")
            ("logtostderr", bpo::bool_switch(&FLAGS_logtostderr)->default_value(false), "log to std error")
            ("alsologtostderr", bpo::bool_switch(&FLAGS_alsologtostderr)->default_value(true), "also log to std error")
            ;

    bpo::variables_map opts;
    bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
    bpo::notify(opts);
    if(opts.count("help")) {
        cout << opts_desc << endl;
        return EXIT_FAILURE;
    }
    FLAGS_log_dir = output_prefix;
    google::InitGoogleLogging("...");

    cpu_tsdf::TSDFHashing::Ptr scene_tsdf(new cpu_tsdf::TSDFHashing);
    LOG(INFO) << "Reading scene model file.";
    {
        ifstream is(input_scene_file);
        boost::archive::binary_iarchive ia(is);
        ia >> *(scene_tsdf);
    }

    LOG(INFO) << "Reading detected box file";
    tsdf_detection::SampleCollection sample_collection;
    sample_collection.ReadOBBs(detect_obb_file);
    std::vector<tsdf_utility::OrientedBoundingBox> obbs;
    std::vector<int> sample_model_idx;
    std::vector<float> sample_scores;
    sample_collection.GetOBBCollection(&obbs, &sample_model_idx, &sample_scores);

    ///
    //std::vector<tsdf_utility::OrientedBoundingBox> obbs;
    //std::vector<int> sample_model_idx;
    //std::vector<cpu_tsdf::OrientedBoundingBox> old_obbs;
    //cpu_tsdf::ReadOrientedBoundingBoxes(detect_obb_file, &old_obbs, &sample_model_idx);
    //for (int i = 0; i <old_obbs.size(); ++i)
    //    {
    //        cpu_tsdf::OrientedBoundingBox obb;
    //        obb = old_obbs[i];
    //        {
    //            float z_below = 0.2;
    //            Eigen::Vector3f ext_below_ground(0, 0, z_below);
    //            obb.bb_offset = obb.bb_offset - ext_below_ground;
    //            tsdf_detection::ExtendOBBNoBottom(obb, Eigen::Vector3f(1, 1, 2 + z_below));
    //        }
    //        old_obbs[i] = obb;
    //    }
    //std::vector<float> sample_scores(old_obbs.size(), 0);
    //obbs = tsdf_utility::NewOBBsFromOlds(old_obbs);
    ///

    for (auto& obbi : obbs) {
        //obbi = obbi.ExtendSidesByPercent(Eigen::Vector3f(0.1, 0.1, 0));
        obbi = obbi.ExtendSides(Eigen::Vector3f(1, 1, 2.2));
        Eigen::AffineCompact3f temp1 = obbi.transform();
        temp1.translation() -= Eigen::Vector3f(0,0,0.2);
        obbi = tsdf_utility::OrientedBoundingBox(temp1);
    }
    LOG(INFO) << "Read " << obbs.size() << " samples.";
    if (obbs.empty()) return EXIT_FAILURE;

    LOG(INFO) << "Initialize.";
    tsdf_utility::OptimizationParams params;
    params.save_path = output_prefix + "res.ply";
    params.min_meshing_weight = min_meshing_weight;
    params.lambda_average_scale = lambda_average_scale;
    params.lambda_observation = lambda_observation;
    params.lambda_regularization = lambda_regularization;
    params.lambda_outlier = lambda_outlier;
    params.pc_num = pc_num;
    params.pc_max_iter = 25;
    params.opt_max_iter = optimize_max_iter;
    params.noise_connected_component_thresh = noise_connected_component_thresh;
    params.noise_observation_thresh = noise_observation_thresh;
    scene_tsdf->getDepthTruncationLimits((params.max_dist_pos), (params.max_dist_neg));

    std::vector<Eigen::SparseVector<float>> model_means;
    std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor>> model_bases;
    std::vector<Eigen::VectorXf> projected_coeffs;
    std::vector<Eigen::Vector3f> model_average_scales;
    Eigen::SparseMatrix<float, Eigen::ColMajor> reconstructed_sample_weights;

    params.save_path = output_prefix + "_icompnum_" + boost::lexical_cast<string>(0) + ".ply";

    tsdf_optimization::InitializeOptimization(
                *scene_tsdf,
                obbs,
                sample_scores,
                sample_model_idx,
                params,
                &model_means,
                &model_bases,
                &projected_coeffs,
                &model_average_scales,
                &reconstructed_sample_weights
                );

    std::vector<double> outlier_gammas(obbs.size(), 0.0);
    for (int icompnum = 0; icompnum <= pc_num; ++icompnum)
    {
        bfs::path write_dir(bfs::path(output_prefix).parent_path()/(string("comp_num_") + boost::lexical_cast<string>(icompnum)));
        bfs::create_directories(write_dir);
        params.save_path = (write_dir/bfs::path(output_prefix).stem()).string() + "res.ply";
        params.pc_num = icompnum;

        cout << "begin following optimization " << endl;

        tsdf_optimization::JointClusteringAndModelLearning(*scene_tsdf,
                                                           params,
                                                           &model_means,
                                                           &model_bases,
                                                           &projected_coeffs,
                                                           &sample_model_idx,
                                                           &outlier_gammas,
                                                           &obbs,
                                                           &model_average_scales,
                                                           &reconstructed_sample_weights
                                                           );
        cout << "finished following optimization " << icompnum << endl;
    }
    tsdf_utility::OutputOBBsAsPly(obbs, params.save_path + "finalres.ply");
    tsdf_detection::SampleCollection samples;
    samples.AddSamplesFromOBBs(obbs, sample_model_idx);
    samples.WriteOBBs(params.save_path + "finalres.txt");
    cout << "begin fusing reconstructed tsdf samples with original scene tsdf " << endl;
    cpu_tsdf::TSDFGridInfo grid_info(*scene_tsdf, params.sample_size, params.min_meshing_weight);
    std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_original_pos;
    cpu_tsdf::ReconstructTSDFsFromPCAOriginPos(
            model_means,
            model_bases,
            projected_coeffs,
            reconstructed_sample_weights,
            sample_model_idx,
            obbs,
            grid_info,
            scene_tsdf->voxel_length(),
            scene_tsdf->offset(),
            &reconstructed_samples_original_pos);
    cpu_tsdf::WriteTSDFModels(reconstructed_samples_original_pos,
                              (bfs::path(params.save_path).replace_extension(".frecon_tsdf.ply")).string(),
                              false, true, min_meshing_weight, outlier_gammas);
    cpu_tsdf::CleanTSDFs(reconstructed_samples_original_pos, -1, 0, 1);
    cpu_tsdf::WriteTSDFModels(reconstructed_samples_original_pos,
                              (bfs::path(params.save_path).replace_extension(".sample_cleaned_frecon_tsdf.ply")).string(),
                              false, true, min_meshing_weight, outlier_gammas);
    scene_tsdf->DisplayInfo();
    cout << "begin merging... " << endl;
    for (int i = 0; i < reconstructed_samples_original_pos.size(); ++i) {
        if (outlier_gammas[i] > 1e-5) {
            reconstructed_samples_original_pos[i].reset();
        }
    }
    reconstructed_samples_original_pos.erase(std::remove_if(reconstructed_samples_original_pos.begin(), reconstructed_samples_original_pos.end(), [](const  cpu_tsdf::TSDFHashing::Ptr& ptr){
        return !bool(ptr);
    }), reconstructed_samples_original_pos.end());
    std::vector<Eigen::Affine3f> obb_affines;
    for (int i = 0; i < obbs.size(); ++i) {
            obb_affines.push_back(obbs[i].AffineTransformOriginAsCenter());
    }
    // cpu_tsdf::ScaleTSDFParts(scene_tsdf.get(), obb_affines, 0.5);
    cpu_tsdf::MergeTSDFs(reconstructed_samples_original_pos, scene_tsdf.get());
    cpu_tsdf::CleanTSDF(scene_tsdf, 100);
    scene_tsdf->DisplayInfo();
    cpu_tsdf::WriteTSDFModel(scene_tsdf,
                             (bfs::path(params.save_path).replace_extension(".merged_tsdf.ply")).string(),
                             true, true, min_meshing_weight);
    cout << "finished fusing" << endl;
    return 0;
}


