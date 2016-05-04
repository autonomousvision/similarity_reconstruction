/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include <iostream>
#include <string>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/pcl_macros.h>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <opencv2/opencv.hpp>

#include "convert_vrl_to_hash.h"
#include "tsdf_representation/tsdf_hash.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/utilities/pcl_utility.h"
#include "common/data_load/urban_reconstruction_data_load.h"
#include "tsdf_operation/tsdf_slice.h"
#include "tsdf_operation/tsdf_io.h"
#include <glog/logging.h>

using namespace std;
using namespace utility;

int
main (int argc, char** argv)
{
  namespace bpo = boost::program_options;
  namespace bfs = boost::filesystem;
  bpo::options_description opts_desc("Allowed options");

  string input_dir;
  string vri_suffix;
  int ramp_size;
  string output_prefix;
  float mesh_min_weight;
  int start_image;
  int end_image;

  std::string tsdf_fpath;
  float pos_trunc_dist = 2.0;
  float neg_trunc_dist = 2.0;

  float neg_dist_full_weight_delta;
  float neg_weight_thresh;
  float neg_weight_dist_thresh;

  float voxel_length;
  opts_desc.add_options()
                  ("help,h", "produce help message")
                  ("in-dir", bpo::value<std::string>(&input_dir)->required (), "input directory for vri")
                  ("rampsize", bpo::value<int>(&ramp_size)->required(), "ramp size: 5/8")
                  ("out-prefix", bpo::value<std::string>(&output_prefix)->required (), "output prefix")
                  ("vri-suffix", bpo::value<std::string>(&vri_suffix)->required(), "vri suffix")
                  ("start_image", bpo::value<int>(&start_image)->required(), "start frame number")
                  ("end_image", bpo::value<int>(&end_image)->required(), "end frame number")
                  ("mesh_min_weight", bpo::value<float>(&mesh_min_weight)->default_value(0), "minimum weight doing marching cubes")
                  ("save_tsdf_bin", "save tsdf binary file")
                  ("voxel_length", bpo::value<float>(&voxel_length)->default_value(0.2), "voxel length")
                  ("input-tsdf-filepath" , bpo::value<string>(&tsdf_fpath), "input tsdf model file")
                  ("pos_truncation_limit", bpo::value<float>(&pos_trunc_dist), "positive truncation limit of signed distance function")
                  ("neg_truncation_limit", bpo::value<float>(&neg_trunc_dist), "negative truncation limit of signed distance function")
                  ("neg_full_weight_delta", bpo::value<float>(), "negative full weight distance, default -voxel_length/5")
                  ("neg_weight_dist_thresh", bpo::value<float>(), "negative inflection distance, default 0.05")
                  ("neg_weight_thresh", bpo::value<float>(), "negative inflection weight, default -voxel_length*3 ")
          ;

  bpo::variables_map opts;
  bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
  bpo::notify(opts);
  if(opts.count("help")) {
    cout << "Usage: " << bfs::basename(argv[0]) << " --in [in_dir] --out [out_dir] [OPTS]" << endl;
    cout << endl;
    cout << opts_desc << endl;
    return (1);
  }
  ///////////////////////////////////////////////////////
  using namespace std;
  std::cout << "reading files. " << std::endl;
  vector<string> vri_filelist;
  ListFilesWithinFrameRange(input_dir, vri_suffix, start_image, end_image, &vri_filelist);
  std::cout << "read in " << vri_filelist.size() << " vri files" << std::endl;
  if (vri_filelist.empty()) return -1;

  {
      pos_trunc_dist = voxel_length * ramp_size;
      neg_trunc_dist = -voxel_length * ramp_size;
      if (opts.count("pos_truncation_limit")) pos_trunc_dist = opts["pos_truncation_limit"].as<float>();
      if (opts.count("neg_truncation_limit")) neg_trunc_dist = opts["neg_truncation_limit"].as<float>();
      CHECK_LT(neg_trunc_dist, 0);

      neg_dist_full_weight_delta = - voxel_length / 5.0;
      neg_weight_thresh = 0.05;
      neg_weight_dist_thresh =  - voxel_length * 3;
      if (opts.count("neg_full_weight_delta")) neg_dist_full_weight_delta = opts["neg_full_weight_delta"].as<float>();
      if (opts.count("neg_weight_dist_thresh")) neg_weight_dist_thresh = opts["neg_weight_dist_thresh"].as<float>();
      if (opts.count("neg_weight_thresh")) neg_weight_thresh = opts["neg_weight_thresh"].as<float>();
      CHECK_LT(neg_dist_full_weight_delta, 0);
      CHECK_LT(neg_weight_dist_thresh, neg_dist_full_weight_delta);
      CHECK_LT(neg_trunc_dist, neg_weight_dist_thresh);
      CHECK_LE(neg_weight_thresh, 1);
      CHECK_LE(0, neg_weight_thresh);
  }

  cpu_tsdf::TSDFHashing::Ptr scene_tsdf (new cpu_tsdf::TSDFHashing(Eigen::Vector3f(0, 0, 0),
                                                                   voxel_length, pos_trunc_dist, neg_trunc_dist));
  if (opts.count("input-tsdf-filepath") && !tsdf_fpath.empty() )
  {
      if (!bfs::exists(tsdf_fpath))
      {
          cerr << "open input tsdf file failed: \n" << tsdf_fpath << endl;
          exit(1);
      }
      fprintf(stderr, "opening tsdf-file..\n%s\n", tsdf_fpath.c_str());
      std::ifstream is(tsdf_fpath, std::ios_base::in);
      boost::archive::binary_iarchive ia(is);
      scene_tsdf.reset(new cpu_tsdf::TSDFHashing);
      ia >> (*scene_tsdf);
  }

  for (int i = 0; i < vri_filelist.size(); ++i)
  {
      string input_filename = vri_filelist[i];
      std::cout << "begin vri tsdf converting, " << i+1 << "th file." << std::endl;
      std::cout << input_filename << std::endl;
      cpu_tsdf::TSDFHashing::Ptr tsdf (new cpu_tsdf::TSDFHashing);
      if (!ReadFromVRIFile(input_filename, ramp_size, tsdf.get()))
      {
          std::cout << "Reading vri file failed." << endl;
          exit(1);
      }
      Eigen::Vector3i bsz;
      tsdf->getVoxelBoundingBoxSize(bsz);
      std::cout << "bounding box size for tsdf: " << bsz(0) << " " << bsz(1) << " " << bsz(2) << std::endl;
      cv::Vec3f min_pt, max_pt;
      tsdf->getBoundingBoxInWorldCoord(min_pt, max_pt);
      Eigen::Vector3f offset;
      offset = tsdf->offset();
  //    fprintf(stderr, "bbox min_pt: %f, %f, %f\n", min_pt[0], min_pt[1], min_pt[2]);
  //    fprintf(stderr, "bbox max_pt: %f, %f, %f\n", max_pt[0], max_pt[1], max_pt[2]);
  //    fprintf(stderr, "offset: %f %f %f\n", offset(0), offset(1), offset(2));

      cpu_tsdf::MergeTSDF((*tsdf), scene_tsdf.get());
  }
  cpu_tsdf::WriteTSDFModel(scene_tsdf, output_prefix, true, true, mesh_min_weight);
  return 0;
}



