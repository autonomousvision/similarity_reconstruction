/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>

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
#include "tsdf_representation/tsdf_hash.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/utilities/pcl_utility.h"
#include "common/utilities/eigen_utility.h"
#include "common/utilities/common_utility.h"
#include "common/data_load/urban_reconstruction_data_load.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_operation/tsdf_io.h"
#include "detection/detector.h"
#include "detection/obb_intersection.h"

namespace bpo = boost::program_options;
namespace bfs = boost::filesystem;
using namespace std;
using namespace Eigen;

int
main (int argc, char** argv)
{
  using namespace std;
  //srand((unsigned)time(NULL));
  srand(1);
  bpo::options_description opts_desc("Allowed options");

  string input_scene_filename;
  string annotate_obb_file;
  string output_prefix;
  float min_non_empty_weight = 0;
  vector<int> sample_size_vec;
  vector<float> detect_deltas;
  int total_thread = 8;
  float svm_param_c = 1;
  float svm_param_w1 = 10;
  opts_desc.add_options()
          ("help,h", "produce help message")
          ("scene_file", bpo::value<string>(&input_scene_filename)->required (), "input TSDF model for sampling")
          ("annotations", bpo::value<string>(&annotate_obb_file)->required(), "detected obb file as positive samples")
          ("output_prefix", bpo::value<string>(&output_prefix)->required (), "output dir path")
          ("sample_size", bpo::value<vector<int>>(&sample_size_vec)->required()->multitoken(), "side length in voxel in x, y, z directions")
          ("total_thread", bpo::value<int>(&total_thread)->default_value(8), "thread number for running detection")
          ("detect_deltas", bpo::value<vector<float>>(&detect_deltas)->multitoken(), "deltas in detection")
          ("svm_param_c", bpo::value<float>(&svm_param_c)->default_value(100), "svm c, larger c data term larger weight")
          ("svm_param_w1", bpo::value<float>(&svm_param_w1)->default_value(10), "svm w1")
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
  CHECK_EQ(sample_size_vec.size(), 3);
  FLAGS_log_dir = output_prefix;
  google::InitGoogleLogging("...");

  // show input options
  Vector3i sample_size(sample_size_vec[0], sample_size_vec[1], sample_size_vec[2]);
  LOG(INFO) << "sample size: \n" << sample_size;
  LOG(INFO) << "non empty weight: "<< min_non_empty_weight;

  LOG(INFO) << "Reading annotated oriented boundingbox file. ";
  vector<vector<tsdf_utility::OrientedBoundingBox>> category_training_obbs;
  tsdf_utility::InputAnnotatedOBB(annotate_obb_file, &category_training_obbs);
  for (int i = 0; i < category_training_obbs.size(); ++i) {
      LOG(INFO) << "Category " << i << ": " << category_training_obbs[i].size() << " samples.";
  }

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
  char train_options[256];
  sprintf(train_options, "-s 0 -t 0 -c %f -w1 %.9f ", svm_param_c, svm_param_w1);
  tsdf_detection::DetectionParams params;
  params.save_prefix = output_prefix;
  // params.NMS_overlap_threshold = 0.01;
  params.NMS_overlap_threshold = 0.01;
  params.detection_total_thread = total_thread;
  params.do_NMS = false;
  params.hard_negative_mining_iterations = 3;
  params.min_nonempty_voxel_weight = 0;
  params.min_score_to_keep = 0;
  params.minimum_occupied_ratio = 0.01;
  params.obb_matching_thresh = 0.3;
  params.train_options = train_options;
  params.max_hard_negative_number = 1e6;
  params.positive_jitter_num = 30;

   for (int category = 0; category < category_training_obbs.size(); ++category) {
      LOG(INFO) << "training for category " << category ;
      params.save_prefix = output_prefix + "/category_" + utility::int2str(category, 0);
      bfs::create_directories(bfs::path(params.save_prefix));
      vector<tsdf_utility::OrientedBoundingBox> current_training_obbs = category_training_obbs[category];
      if (current_training_obbs.empty()) continue;

      // the template oriented bounding box for sampling negatives / doing detection
      //Eigen::Vector3f mean_sidelengths(0, 0, 0);
      //for (auto& obbi : current_training_obbs) {
      //    obbi = obbi.ExtendSidesByPercent(Eigen::Vector3f(0.1, 0.1, 0.2));
      //    mean_sidelengths += obbi.SideLengths();
      //}
      //mean_sidelengths /= (current_training_obbs.size());
      //Eigen::Vector3f obb_pos = current_training_obbs[0].BottomCenter();
      //tsdf_utility::OrientedBoundingBox template_bb(mean_sidelengths, obb_pos[0], obb_pos[1], obb_pos[2],
      //        current_training_obbs[0].AngleRangeTwoPI());
      tsdf_utility::OrientedBoundingBox template_bb = current_training_obbs[0];
      Eigen::Vector3f mean_sidelengths = template_bb.SideLengths();

      // scene discretization info
      // when doing detection (in negative mining), the detector is placed at discretized positions
      // the SceneDiscretizeInfo converts between discretized positions and positions in the world coordinate
      float delta_x, delta_y, delta_rotation;
      if (!detect_deltas.empty()) {
          CHECK_EQ(detect_deltas.size(), 3);
          delta_x = detect_deltas[0];
          delta_y = detect_deltas[1];
          delta_rotation = detect_deltas[2]/180.0 * M_PI;
      } else {
          tsdf_detection::StepSizeFromOBB(mean_sidelengths, delta_x, delta_y, delta_rotation);
      }
      tsdf_detection::SceneDiscretizeInfo discretize_info(Vector2f(min_pt[0], max_pt[0]),
                      Vector2f(min_pt[1], max_pt[1]),
                      Vector3f(delta_x, delta_y, delta_rotation));
      discretize_info.DisplayDiscretizeInfo();

      tsdf_detection::SampleCollection pos_samples;
      pos_samples.AddSamplesFromOBBs(current_training_obbs, sample_size, *tsdf_model, params.min_nonempty_voxel_weight, 1);  // add pos samples
      tsdf_detection::Detector detector;
      tsdf_detection::SampleTemplate sample_template(template_bb, sample_size);
      detector.sample_template(sample_template);
      tsdf_detection::TrainDetector(*tsdf_model, discretize_info, params, pos_samples, &detector);
      LOG(INFO) << "saving trained svm model";
      detector.SaveToFile(params.save_prefix + "/trained_svm_model.svm");
  }
  return 0;
}



