/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/pcl_macros.h>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_slice.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/utilities/pcl_utility.h"
#include "common/data_load/urban_reconstruction_data_load.h"
#include "tsdf_hash_utilities/utility.h"
#include "consistency_check.h"
#include "detection/detector.h"
#include "tsdf_operation/tsdf_clean.h"

using namespace std;
using namespace utility;
namespace bpo = boost::program_options;
namespace bfs = boost::filesystem;

int
main (int argc, char** argv)
{

  bpo::options_description opts_desc("Allowed options");

  string tsdf_filename;
  string obb_filename;
  string out_filename;
  vector<string> data_root_dirs;
  string cam_info_prefix;
  string skymap_prefix;
  string depth_prefix;
  int start_image;
  int end_image;
  float mesh_min_weight;
  float sky_map_thresh;
  float max_cam_distance;
  int st_neighbor;
  int ed_neighbor;
  bool depthmap_check;
  bool skymap_check;
  float filter_noise;

  opts_desc.add_options()
                  ("help,h", "produce help message")
                  ("in_model", bpo::value<string>(&tsdf_filename)->required(), "input tsdf file")
                  ("detect_obb_file", bpo::value<string>(&obb_filename)->required(), "input obb file")
                  ("output_filename", bpo::value<std::string>(&out_filename)->required (), "output ply path")
                  ("data_roots", bpo::value<vector<string>>(&data_root_dirs)->required()->multitoken(), "data roots for 2 cameras")
                  ("cam_info_prefix", bpo::value<string>(&cam_info_prefix)->required(), "cam_info_prefix (not including data root path)")
                  ("skymap_prefix", bpo::value<string>(&skymap_prefix)->required(), "skymap_prefix (not including data root path)")
                  ("depth_prefix", bpo::value<string>(&depth_prefix)->required(), "depth_prefix (not including data root path)")
                  ("start_image", bpo::value<int>(&start_image)->required(), "start frame number (not including data root path)")
                  ("end_image", bpo::value<int>(&end_image)->required(), "end frame number (not including data root path)")
                  ("mesh_min_weight", bpo::value<float>(&mesh_min_weight)->default_value(0), "mesh_min_weight")
                  ("sky_map_thresh", bpo::value<float>(&sky_map_thresh)->default_value(3), "the threshold of sky map labelings to remove a mesh vertex / TSDF point")
                  ("max_cam_distance", bpo::value<float>(&max_cam_distance)->required(), "max_cam_distance")
                  ("clean_tsdf", "whether clean tsdf or mesh")
                  ("st_neighbor", bpo::value<int>(&st_neighbor)->default_value(-1), "starting neighbor grid point for cleaning TSDF")
                  ("ed_neighbor", bpo::value<int>(&ed_neighbor)->default_value(2), "ending neighbor grid point for cleaning TSDF")
                  ("depthmap_check", bpo::value<bool>(&depthmap_check)->default_value(true), "whether to check depth map")
                  ("skymap_check", bpo::value<bool>(&skymap_check)->default_value(true), "whether to check sky map")
                  ("filter_noise", bpo::value<float>(&filter_noise)->default_value(120), "filter noise, minimum triangle size")
                  ("logtostderr", bpo::bool_switch(&FLAGS_logtostderr)->default_value(false), "log to std error")
                  ("alsologtostderr", bpo::bool_switch(&FLAGS_alsologtostderr)->default_value(true), "also log to std error")
    ;

  bpo::variables_map opts;
  bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
  bpo::notify(opts);
  if(opts.count("help")) {
    cout << "Clean & Check the TSDF/mesh model according to skymap and depth maps" << endl;
    cout << endl;
    cout << opts_desc << endl;
    return EXIT_FAILURE;
  }
  FLAGS_log_dir = bfs::path(out_filename).parent_path().string();
  google::InitGoogleLogging("...");

  std::cout << "begin reading tsdf. " << std::endl;
  cpu_tsdf::TSDFHashing::Ptr tsdf(new cpu_tsdf::TSDFHashing);
  {
      cout << "Reading scene model file. \n" << tsdf_filename<< endl;
      ifstream is(tsdf_filename);
      boost::archive::binary_iarchive ia(is);
      ia >> *tsdf;
  }

  LOG(INFO) << "Reading detected box file\n " <<  obb_filename << endl;
  //std::vector<std::vector<tsdf_utility::OrientedBoundingBox>> obbvec;
  //std::vector<tsdf_utility::OrientedBoundingBox> detected_obbs;
  //tsdf_utility::InputAnnotatedOBB(obb_filename, &obbvec);
  //for (int i = 0; i < obbvec.size(); ++i) {
  //    detected_obbs.insert(detected_obbs.end(), obbvec[i].begin(), obbvec[i].end());
  //}
  //std::vector<int> sample_model_idx(detected_obbs.size(), 0);

  tsdf_detection::SampleCollection sample_collection;
  sample_collection.ReadOBBs(obb_filename);
  std::vector<tsdf_utility::OrientedBoundingBox> detected_obbs;
  std::vector<int> sample_model_idx;
  sample_collection.GetOBBCollection(&detected_obbs, &sample_model_idx);
  tsdf_utility::OutputOBBsAsPly(detected_obbs, out_filename + ".obb.ply");

  ///
    //std::vector<tsdf_utility::OrientedBoundingBox> obbs;
    //std::vector<int> sample_model_idx;
    //std::vector<cpu_tsdf::OrientedBoundingBox> old_obbs;
    //cpu_tsdf::ReadOrientedBoundingBoxes(obb_filename, &old_obbs, &sample_model_idx);
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
    //obbs = tsdf_utility::ComputeNewOBBsFromOlds(old_obbs);
    //std::vector<tsdf_utility::OrientedBoundingBox> detected_obbs;
    //detected_obbs = obbs;
  ///
  LOG(INFO) << "Read " << detected_obbs.size() << " obbs. ";

  vector<string> cam_filelist;
  vector<string> skymap_filelist;
  vector<string> depth_filelist;
  for (int i = 0; i < data_root_dirs.size(); ++i)
  {
      const string& data_root_dir = data_root_dirs[i];
      cout << "Reading cam infos\n" << data_root_dir + "/" + cam_info_prefix << endl;
      string cam_dir = (bfs::path(data_root_dir)/cam_info_prefix).string();
      ListFilesWithinFrameRange(cam_dir, ".txt", start_image, end_image, &cam_filelist);

       cout << "Reading sky maps\n" << data_root_dir + "/" + skymap_prefix << endl;
       string skymap_dir = (bfs::path(data_root_dir)/skymap_prefix).string();
       ListFilesWithinFrameRange(skymap_dir, ".png", start_image, end_image, &skymap_filelist);

       cout << "Reading depth maps\n" << data_root_dir + "/" + depth_prefix << endl;
       string depth_dir = (bfs::path(data_root_dir)/depth_prefix).string();
       ListFilesWithinFrameRange(depth_dir, ".png", start_image, end_image, &depth_filelist);
  }
  if (cam_filelist.empty() || skymap_filelist.size() != cam_filelist.size() || depth_filelist.size() != cam_filelist.size()) {
      LOG(FATAL) << "Reading camera/skymap/depth files failed. ";
  }

  std::vector<RectifiedCameraPair> cam_infos;
  ReadCameraInfos(cam_filelist, cam_infos);
  {
      float depth_image_scaling_factor = max_cam_distance/65535.0;
      cv::Mat temp_map = cv::imread(skymap_filelist[0], -1);
      InitializeCamInfos(temp_map.cols, temp_map.rows,
                         cv::Vec3d(utility::EigenVectorToCvVector3(tsdf->offset())), tsdf->voxel_length(),
                         depth_image_scaling_factor, max_cam_distance, cam_infos);
  }

  if (opts.count("clean_tsdf")) {
      CleanTSDFWithSkyMapAndDepthMap(tsdf, detected_obbs, cam_infos, skymap_filelist, depth_filelist, sky_map_thresh, out_filename,
                                     st_neighbor, ed_neighbor, skymap_check, depthmap_check);
      cpu_tsdf::CleanTSDF(tsdf, filter_noise, st_neighbor, ed_neighbor);
      cpu_tsdf::WriteTSDFModel(tsdf, out_filename + ".tsdf_consistency_cleaned.ply", true, true, mesh_min_weight);
  } else {
      pcl::PolygonMesh::Ptr pmesh = cpu_tsdf::TSDFToPolygonMesh(tsdf, mesh_min_weight, -1);
      CleanMeshWithSkyMapAndDepthMap(
                              *pmesh,
                              detected_obbs,
                              cam_infos,
                              skymap_filelist,
                              depth_filelist,
                              sky_map_thresh,
                              out_filename,
                              NULL,
                              skymap_check,
                              depthmap_check
                              );
      cpu_tsdf::CleanMesh(*pmesh, filter_noise);
      pcl::io::savePLYFileBinary(out_filename + ".tsdf_consistency_cleaned.ply", *pmesh);
  }
  cpu_tsdf::WriteForVisualization((bfs::path(out_filename).parent_path() / "visualization").string(), tsdf, mesh_min_weight, &detected_obbs);
  return 0;
}



