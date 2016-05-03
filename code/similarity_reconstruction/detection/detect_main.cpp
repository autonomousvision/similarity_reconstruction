#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>

#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <boost/format.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/Eigen>
#include <glog/logging.h>

#include "tsdf_operation/tsdf_slice.h"
// #include "tsdf_operation//tsdf_feature_generate.h"
#include "tsdf_representation/tsdf_hash.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/utilities/pcl_utility.h"
#include "common/utilities/eigen_utility.h"
#include "common/utilities/common_utility.h"
#include "common/data_load/urban_reconstruction_data_load.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_operation/tsdf_io.h"
#include "detection/detector.h"
#include "detection/detect_sample.h"
#include "detection/detection_utility.h"
#include "detection/obb_intersection.h"

namespace bpo = boost::program_options;
namespace bfs = boost::filesystem;
using namespace std;
using namespace Eigen;

int
main (int argc, char** argv)
{
  using namespace std;
  bpo::options_description opts_desc("Allowed options");

  string input_scene_filename;
  string model_filename;
  string output_prefix;
  vector<float> detect_deltas;
  vector<float> min_scores_to_keep;
  int total_thread = 8;
  opts_desc.add_options()
          ("help,h", "produce help message")
          ("scene_file", bpo::value<string>(&input_scene_filename)->required (), "input TSDF model")
          ("detector_file", bpo::value<string>(&model_filename)->required(), "detector file path")
          ("output_prefix", bpo::value<string>(&output_prefix)->required (), "output dir path")
          ("total_thread", bpo::value<int>(&total_thread)->default_value(8), "thread number for running detection")
          ("detect_deltas", bpo::value<vector<float>>(&detect_deltas)->multitoken(), "deltas in detection")
          ("min_score_to_keep", bpo::value<vector<float>>(&min_scores_to_keep)->multitoken(), "min detection score threshold")
          ("logtostderr", bpo::bool_switch(&FLAGS_logtostderr)->default_value(false), "log to std error")
          ("alsologtostderr", bpo::bool_switch(&FLAGS_alsologtostderr)->default_value(true), "also log to std error")
    ;
  bpo::variables_map opts;
  bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
  bpo::notify(opts);
  if (opts.count("help"))
  {
      cout << "Wrong argument!" << endl;
      cout << endl;
      cout << opts_desc << endl;
      return EXIT_FAILURE;
  }
  FLAGS_log_dir = output_prefix;
  google::InitGoogleLogging("...");

  LOG(INFO) << "Reading detetor model file.";
  boost::system::error_code ec;
  vector<unique_ptr<tsdf_detection::Detector>> detectors;
  if (bfs::is_directory(bfs::path(model_filename), ec)) {
      // model_filename is a directory containing detectors for different categories
      LOG(INFO) << "detector_file " << model_filename << " is a directory.";
      vector<string> dirlist;
      if (ListDirs(model_filename, "category_", &dirlist)) {
          detectors.resize(dirlist.size());
          for (const auto& dirname : dirlist) {
              int category = 0;
              sscanf(bfs::path(dirname).stem().string().c_str(), "category_%d", &category);
              string filename = dirname + "/trained_svm_model.svm";
              if (bfs::exists(bfs::path(filename), ec)) {
                  LOG(INFO) << "reading: " << filename;
                  detectors[category].reset(new tsdf_detection::Detector(filename));
              }
          }
      }
  } else {
      LOG(INFO) << "detector_file " << model_filename;
      detectors.emplace_back(new tsdf_detection::Detector(model_filename));
  }
  LOG(INFO) << "Read " << detectors.size() << " detector(s).";
  // default minimum detection scores
  if (min_scores_to_keep.size() < detectors.size()) {
      min_scores_to_keep.insert(min_scores_to_keep.end(), detectors.size() - min_scores_to_keep.size(), -0.5);
  }
  CHECK_GT(detectors.size(), 0);

  LOG(INFO) << "Reading TSDF scene. ";
  cpu_tsdf::TSDFHashing::Ptr tsdf_model (new cpu_tsdf::TSDFHashing);
  {
      ifstream is(input_scene_filename);
      boost::archive::binary_iarchive ia(is);
      ia >> *tsdf_model;
  }
  // compute scene boundary
  Eigen::Vector3f min_pt, max_pt;
  tsdf_model->RecomputeBoundingBoxInVoxelCoord();
  tsdf_model->getBoundingBoxInWorldCoord(min_pt, max_pt);
  LOG(INFO) << "scene boundary in voxels: \nmin_pt \n" << min_pt << "\nmax_pt \n" << max_pt;

  // training/detection parameters
  tsdf_detection::DetectionParams params;
  params.save_prefix = output_prefix;
  params.NMS_overlap_threshold = 0.05;
  params.detection_total_thread = total_thread;
  params.do_NMS = true;
  params.hard_negative_mining_iterations = 3;
  params.min_nonempty_voxel_weight = 0;
  params.min_score_to_keep = -0.5;
  params.minimum_occupied_ratio = 0.01;
  params.obb_matching_thresh = 0.3;
  params.do_final_NMS = true;

  std::vector<tsdf_detection::SceneDiscretizeInfo> discretize_infos;
  tsdf_detection::SampleCollection sample_collections;
  for (int category = 0; category < detectors.size(); ++category) {
      params.save_prefix = output_prefix + "/detection_" + utility::int2str(category, 0);
      params.min_score_to_keep = min_scores_to_keep[category];
      LOG(INFO) << "min detection score: " << params.min_score_to_keep;
      bfs::create_directories(bfs::path(params.save_prefix));
      if (!detectors[category]) continue;

      // scene discretization info
      // when doing detection (in negative mining), the detector is placed at discretized positions
      // the SceneDiscretizeInfo converts between discretized positions and positions in the world coordinate
      tsdf_detection::SampleTemplate template_obb = detectors[category]->sample_template();
      float delta_x, delta_y, delta_rotation;
      if (!detect_deltas.empty()) {
          CHECK_EQ(detect_deltas.size(), 3);
          delta_x = detect_deltas[0];
          delta_y = detect_deltas[1];
          delta_rotation = detect_deltas[2]/180.0 * M_PI;
      } else {
          tsdf_detection::StepSizeFromOBB(template_obb.SideLengths(), delta_x, delta_y, delta_rotation);
      }
      tsdf_detection::SceneDiscretizeInfo discretize_info(Vector2f(min_pt[0], max_pt[0]),
                      Vector2f(min_pt[1], max_pt[1]),
                      Vector3f(delta_x, delta_y, delta_rotation));
      discretize_info.DisplayDiscretizeInfo();
      tsdf_detection::Sample temp = tsdf_detection::Sample(template_obb.OBB(), template_obb.sample_size(), *tsdf_model, -1, params.min_nonempty_voxel_weight);
      // params.minimum_occupied_ratio = temp.OccupiedRatio() * 0.3;

      tsdf_detection::SampleCollection cur_samples;
      // cur_samples.ReadOBBs(params.save_prefix + "/obbs.txt");
      tsdf_detection::Detect(*tsdf_model, (*detectors[category]), discretize_info, params, 0, &cur_samples);
      // set object category label
      for (auto& samplei : cur_samples.samples) {
          samplei.category_label(category);
      }
      cur_samples.WriteOBBsToPLY(params.save_prefix + "/obb.ply");
      cur_samples.WriteOBBs(params.save_prefix + "/obbs.txt");
      sample_collections.samples.insert(sample_collections.samples.begin(), cur_samples.samples.begin(), cur_samples.samples.end());
      discretize_infos.push_back(discretize_info);
  }
  sample_collections.WriteOBBs(output_prefix + "/detect_res_all_obb.txt");
  if (params.do_final_NMS) {
      tsdf_detection::NMS(params, &sample_collections);
  }
  // fine tune the positions and scales
  cout << "adjusting positions and scales" << endl;
  for (int i = 0; i < sample_collections.samples.size(); ++i) {
      tsdf_detection::Sample& cur_sample = sample_collections.samples[i];
      int cur_category = cur_sample.category_label();
      tsdf_detection::AdjustOneSamplePos(*tsdf_model, (*detectors[cur_category]), discretize_infos[cur_category], params, &cur_sample);
      tsdf_detection::AdjustOneSampleScales(*tsdf_model, (*detectors[cur_category]), discretize_infos[cur_category], params, &cur_sample);
  }
  cout << "finished adjusting positions and scales" << endl;
  using namespace tsdf_detection;
  std::sort(sample_collections.samples.begin(), sample_collections.samples.end(), [](const Sample& lhs, const Sample& rhs) {
      return lhs.predict_score() > rhs.predict_score();
  });
  sample_collections.WriteOBBs(output_prefix + "/detect_res_all_obb_nmsed.txt");
  sample_collections.WriteOBBsToPLY(output_prefix + "/detect_res_all_obb_nmsed.ply");
  return 0;
}

