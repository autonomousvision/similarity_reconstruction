#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/pcl_macros.h>
//#include <pcl/segmentation/extract_clusters.h>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "tsdf_representation/tsdf_hash.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/fisheye_camera/RectifiedCameraPair.h"
#include "common/data_load/urban_reconstruction_data_load.h"
#include "common/utilities/pcl_utility.h"
#include "tsdf_operation/tsdf_smooth.h"
#include "tsdf_operation/diffusion_hole_filling.h"

//using std::vector;
//using std::string;
using namespace std;


//bool
//reprojectPoint (const pcl::PointXYZRGBA &pt, int &u, int &v)
//{
//  u = (pt.x * focal_length_x_ / pt.z) + principal_point_x_;
//  v = (pt.y * focal_length_y_ / pt.z) + principal_point_y_;
//  return (!pcl_isnan (pt.z) && pt.z > 0 && u >= 0 && u < width_ && v >= 0 && v < height_);
//}

//void
//remapCloud (pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
//            const Eigen::Affine3d &pose,
//            pcl::PointCloud<pcl::PointXYZRGBA> &cloud_remapped)
//{
//  // Reproject
//  cloud_remapped = pcl::PointCloud<pcl::PointXYZRGBA> (width_, height_);
//  // Initialize to nan
//#pragma omp parallel for
//  for (size_t i = 0; i < cloud_remapped.size (); i++)
//  {
//    cloud_remapped[i].z = std::numeric_limits<float>::quiet_NaN ();
//  }
//  cloud_remapped.is_dense = false;
//  for (size_t i = 0; i < cloud->size (); i++)
//  {
//    const pcl::PointXYZRGBA &pt = cloud->at (i);
//    int u, v;
//    if (!reprojectPoint (pt, u, v))
//      continue;
//    pcl::PointXYZRGBA &pt_remapped = cloud_remapped (u, v);
//    if (pcl_isnan (pt_remapped.z) || pt_remapped.z > pt.z)
//      pt_remapped = pt;
//  }
//}




//void IntegratePointCloud(std::vector<RectifiedCameraPair>& cam_infos,
//                         const std::vector<std::string>& image_filelist,
//                         const std::vector<std::string>& depth_filelist,
//                         double maxCamDistance,
//                         double depth_image_scaling_factor,
//                         std::vector<cv::Vec3d>& points3d, std::vector<cv::Vec3b>& pointscolor)
//{
//    for (int i = 0; i < cam_infos.size(); ++i)
//    {
//        cv::Mat depth_map = cv::imread(depth_filelist[i], -1);
//        cv::Mat image = cv::imread(image_filelist[i], 1);  // always color
//        cam_infos[i].SetVoxelScalingParameters(cv::Vec3d(0,0,0),
//                                               1.0,
//                                               depth_image_scaling_factor);
//        cam_infos[i].InitializeBackProjectionBuffers(depth_map.cols, depth_map.rows);
//        int imHeight = depth_map.rows;
//        int imWidth = depth_map.cols;
//        for(int y=0; y<imHeight; y++)
//        {
//            for(int x=0; x<imWidth; x++)
//            {
//                const unsigned short quant_depth = depth_map.at<unsigned short>(y,x);
//                const float pz = float(quant_depth)*cam_infos[i].DepthImageScaling();  // unit: in meter
//                if(pz > 0.0f && pz < maxCamDistance)
//                {
//                    cv::Vec3d tpoint = cam_infos[i].RectifiedRefImagePointToVoxel3DPoint(x, y, quant_depth);  // unit: in voxel
//                    //printf("%f %f %f\n",float(x), float(y), float(pz) );
//                    points3d.push_back(static_cast<cv::Vec3d>(tpoint));
//                    cv::Vec3b tcolor = image.at<cv::Vec3b>(y,x);
//                    pointscolor.push_back(tcolor);
//                }
//            }
//        }
//    }
//}
//
//void IntegratePointCloud(std::vector<RectifiedCameraPair>& cam_infos,
//                         const std::vector<std::string>& image_filelist,
//                         const std::vector<std::string>& depth_filelist,
//                         double maxCamDistance,
//                         double depth_image_scaling_factor,
//                         pcl::PointCloud<pcl::PointXYZRGBA>& pointcloud) {
//  vector<cv::Vec3d> points3d;
//  vector<cv::Vec3b> pointscolor;
//  IntegratePointCloud(cam_infos, image_filelist, depth_filelist,
//                      maxCamDistance, depth_image_scaling_factor, points3d, pointscolor);
//  for(int i = 0; i < points3d.size(); ++i) {
//    pcl::PointXYZRGBA pnew;
//    pnew.x = points3d[i][0];
//    pnew.y = points3d[i][1];
//    pnew.z = points3d[i][2];
//    uchar r,g,b;
//    b = pointscolor[i][0];
//    g = pointscolor[i][1];
//    r = pointscolor[i][2];
//    unsigned int rgb = (r << 16) | (g << 8) | b;
//    pnew.rgba = *(float *)(&rgb); // makes the point red
//    pointcloud.push_back(pnew);
//  }
//  pointcloud.is_dense = true;
//  pointcloud.height = 1;
//}
//

void MaskImageSidesAsZero(const int side_width, cv::Mat* image)
{
//    cv::imshow("test0", *image);
//    cv::waitKey();
//    cv::destroyWindow("test0");

    cv::Mat band_left = cv::Mat(*image, cv::Rect(0, 0, side_width, image->rows));
    cv::Mat band_right = cv::Mat(*image, cv::Rect(image->cols - side_width, 0, side_width, image->rows));

    band_left.setTo(0);
    band_right.setTo(0);

//    cv::imshow("test", *image);
//    cv::waitKey();
//    cv::destroyWindow("test");
}

int
main (int argc, char** argv)
{
  namespace bpo = boost::program_options;
  namespace bfs = boost::filesystem;
  bpo::options_description opts_desc("Allowed options");
  double world_offset_x = 0;
  double world_offset_y = 0;
  //std::string output_tsdf_filepath;
  std::string tsdf_fpath;
  float max_camera_distance = 35.0;
  float pos_trunc_dist = 2.0;
  float neg_trunc_dist = 2.0;
  int not_use_side_column_length = 100;
  int start_image = -1;
  int end_image = -1;
  int neighbor_add_limit = 1;
  double stepsize = 1;

  float neg_dist_full_weight_delta;
  float neg_weight_thresh;
  float neg_weight_dist_thresh;

  opts_desc.add_options()
    ("help,h", "produce help message")
    ("in-root", bpo::value<std::string> ()->required (), "Input root dir (should be rect/ directory under a camera)")
    ("depth-prefix", bpo::value<std::string>(), "Depth prefix (folder name under rect/ directory)")
    ("image-prefix", bpo::value<std::string>(), "Image prefix (folder name under rect/ directory)")
    ("param-prefix", bpo::value<std::string>(), "Param prefix (folder name under rect/ directory)")
    ("out", bpo::value<std::string> ()->required (), "Output path (the complete path of the output ply file)")
    ("flatten", "Remove duplicated mesh vertices during marching cubes")
    ("cleanup", "Remove isolated meshes (Remove faces which aren't within 2 marching cube widths from any others)")
    ("save-ascii", "Save ply file as ASCII rather than binary")
    ("max-camera-distance", bpo::value<float>(&max_camera_distance)->default_value(35.0), "Maximum allowed depth value (Must be the same as the max_depth parameter specified in convert_to_depthmap.m)")
    ("startimage", bpo::value<int>(&start_image), "starting image number (starts from 0)")
    ("endimage", bpo::value<int>(&end_image), "ending image number (including)")
    ("voxel_length", bpo::value<float>(), "voxel length")
    ("use_confidence", "use confidence of depth map or not (confidence map is generated during depth filtering)")
    ("min_weight", bpo::value<float>(), "minimum weight threshold for meshing")
    ("do_fuse", "do volumetric fusion")
    // ("smooth", "Smooth the TSDF but not modifying the TSDF points originally with values. (i.e. only \"dilation\")")
    ("diffusion-smooth", "do hole filling as described in the paper \"Filling holes in complex surfaces using volumetric diffusion\"")
    ("niteration" , bpo::value<int>(), "diffusion smooth iteration number")
    ("input-tsdf-filepath" , bpo::value<string>(&tsdf_fpath), "input tsdf model file")
    ("use_semantic", bpo::value<string>(), "use the 2D semantic label. Not used now.")
    ("pos_truncation_limit", bpo::value<float>(&pos_trunc_dist), "positive truncation limit of signed distance function")
    ("neg_truncation_limit", bpo::value<float>(&neg_trunc_dist), "negative truncation limit of signed distance function")
    ("cam_offset_x", bpo::value<double>(&world_offset_x)->default_value(0), "world offset x, minus the number from camera position to shift the cameras")
    ("cam_offset_y", bpo::value<double>(&world_offset_y)->default_value(0), "world offset y, minus the number from camera position to shift the cameras")
                  ("not_use_side_column_length", bpo::value<int>(&not_use_side_column_length)->default_value(100), "not using the area for two sides of fish-eye image")
                  ("neighbor_add_limit", bpo::value<int>(&neighbor_add_limit)->default_value(1), "how many neighborhood blocks will be added as affected blocks")
                  ("stepsize", bpo::value<double>(&stepsize)->default_value(1), "the stepsize on depth map when back projecting pixels")
                  ("neg_full_weight_delta", bpo::value<float>(), "negative full weight distance, default -voxel_length/5")
                  ("neg_weight_dist_thresh", bpo::value<float>(), "negative inflection distance, default 0.05")
                  ("neg_weight_thresh", bpo::value<float>(), "negative inflection weight, default -voxel_length*3 ")
          ;

  bpo::variables_map opts;
  bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
  bool badargs = false;
  try { bpo::notify(opts); }
  catch(...) { badargs = true; }
  if(opts.count("help") || badargs) {
    cout << endl;
    cout << opts_desc << endl;
    return (1);
  }
  ///////////////////////////////////////////////////////

  bool flatten = opts.count ("flatten");
  bool cleanup = opts.count ("cleanup");
  bool save_ascii = opts.count ("save-ascii");



  float min_weight = 0.3;
  if(opts.count("min_weight")) min_weight = opts["min_weight"].as<float>();
  pcl::console::TicToc tt;
  tt.tic ();
  // Initialize
  cv::Vec3d offset(0,0,0);
  Eigen::Vector3f eigen_offset;
  eigen_offset(0) = offset[0]; eigen_offset(1) = offset[1]; eigen_offset(2) = offset[2]; 
  float voxel_length = 0.1;
  if (opts.count("voxel_length")) voxel_length = opts["voxel_length"].as<float>();
//  float truncation_limit = voxel_length * 8;
//  if (opts.count("truncation_limit")) truncation_limit = opts["truncation_limit"].as<float>();
  pos_trunc_dist = voxel_length * 8;
  neg_trunc_dist = -pos_trunc_dist;
  if (opts.count("pos_truncation_limit")) pos_trunc_dist = opts["pos_truncation_limit"].as<float>();
  if (opts.count("neg_truncation_limit")) neg_trunc_dist = opts["neg_truncation_limit"].as<float>();

  //    const float neg_dist_full_weight_threshold = - voxel_length_ / 5.0;
  //    const float neg_weight_thresh1 = 0.05;
  //    const float neg_weight_dist_thresh =  - voxel_length_ * 3;
  neg_dist_full_weight_delta = - voxel_length / 5.0;
  neg_weight_thresh = 0.05;
  neg_weight_dist_thresh =  - voxel_length * 3;
//  ("neg_full_weight_delta", bpo::value<float>(), "negative full weight distance")
//  ("neg_weight_dist_thresh", bpo::value<float>(), "negative inflection distance")
//  ("neg_weight_thresh", bpo::value<float>(), "negative inflection weight")
  if (opts.count("neg_full_weight_delta")) neg_dist_full_weight_delta = opts["neg_full_weight_delta"].as<float>();
  if (opts.count("neg_weight_dist_thresh")) neg_weight_dist_thresh = opts["neg_weight_dist_thresh"].as<float>();
  if (opts.count("neg_weight_thresh")) neg_weight_thresh = opts["neg_weight_thresh"].as<float>();
  CHECK_LT(neg_dist_full_weight_delta, 0);
  CHECK_LT(neg_weight_dist_thresh, neg_dist_full_weight_delta);
  CHECK_LT(neg_trunc_dist, neg_weight_dist_thresh);
  CHECK_LE(neg_weight_thresh, 1);
  CHECK_LE(0, neg_weight_thresh);
  cout << "neg_full_weight_delta: " << neg_dist_full_weight_delta << endl;
  cout << "neg_weight_dist_thresh: " << neg_weight_dist_thresh << endl;
  cout << "neg_weight_thresh: " << neg_weight_thresh << endl;

  cpu_tsdf::TSDFHashing::Ptr tsdf (new cpu_tsdf::TSDFHashing(eigen_offset, voxel_length, pos_trunc_dist, neg_trunc_dist, neighbor_add_limit));
  std::string out_ply_file = opts["out"].as<std::string> ();
  // std::string tsdf_fpath = opts["input-tsdf-filepath"].as<string>();
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
      tsdf.reset(new cpu_tsdf::TSDFHashing);
      ia >> (*tsdf);
  }
  if (opts.count("do_fuse"))
  {
      // Read in all data
      string urban_data_root = opts["in-root"].as<string>();
      string param_prefix = "param2";
      string image_prefix = "img_00";
      string depth_prefix = "simplefusion-disp-ref-0_5-2frame-newdata-errconf-801-851";
      string semantic_prefix = "";
      if(opts.count("param-prefix")) param_prefix = opts["param-prefix"].as<string>();
      if(opts.count("image-prefix")) image_prefix = opts["image-prefix"].as<string>();
      if(opts.count("depth-prefix")) depth_prefix = opts["depth-prefix"].as<string>();
      if(opts.count("use_semantic")) semantic_prefix = opts["use_semantic"].as<string>();
      bool use_confidence = opts.count("use_confidence");
      bool use_semantic_label = opts.count("use_semantic");
      vector<RectifiedCameraPair> cam_infos;
      vector<string> image_filelist;
      vector<string> depth_filelist;
      vector<string> confidence_filelist;
      vector<string> semantic_filelist;
//      if(!LoadUrbanReconstructionData(urban_data_root, param_prefix, image_prefix, depth_prefix, semantic_prefix,
//                  &cam_infos, &image_filelist, &depth_filelist, &confidence_filelist, &semantic_filelist, use_confidence)) {
      if(!LoadUrbanReconstructionData(urban_data_root, param_prefix, image_prefix, depth_prefix, semantic_prefix,
                                      start_image, end_image,
                                      &cam_infos, &image_filelist, &depth_filelist, &confidence_filelist, &semantic_filelist, use_confidence)) {
          printf("Reading files failed ...\n");
          exit(1);
      }
      if(depth_filelist.empty()) {
          printf("No depth maps founded.\n");
          exit(1);
      }


      //if (opts.count("max-camera-distance")) max_camera_distance = opts["max-camera-distance"].as<float>();
      float depth_image_scaling_factor = max_camera_distance/65535.0;
      cv::Mat temp_depth = cv::imread(depth_filelist[0], -1);
      InitializeCamInfos(temp_depth.cols, temp_depth.rows, cv::Vec3d(utility::EigenVectorToCvVector3(tsdf->offset())), tsdf->voxel_length(),
              depth_image_scaling_factor, max_camera_distance, cam_infos);
      Eigen::Vector3d world_offset(world_offset_x, world_offset_y, 0);
      cv::Matx34d world_offset_mat;
      world_offset_mat << 0, 0, 0, world_offset[0],
              0, 0, 0, world_offset[1],
              0, 0, 0, 0;
      cout << world_offset_mat << endl;
      for (int i = 0; i < cam_infos.size(); ++i)
      {
          cv::Matx34d cur_P1, cur_P2;
          cam_infos[i].GetExtrinsicPair(&cur_P1, &cur_P2);
          cam_infos[i].SetExtrinsicPair(cur_P1 - world_offset_mat, cur_P2 - world_offset_mat);
          cam_infos[i].InitializeBackProjectionBuffers();
      }

      for(int i = 0; i < 2; ++i)
      {
          cout << "info for: " <<  i << "th frame:" << endl;
          cout << "depth: " << depth_filelist[i] << endl;
          cout << "image: " << image_filelist[i] << endl;
          if (use_confidence)
            cout << "confidence: " << confidence_filelist[i] << endl;
          cout << "cam_info: " << cam_infos[i] << endl;
      }

//      int startimage = 0;
//      int endimage = depth_filelist.size()-1;
//      if(opts.count("startimage")) { startimage = std::max(opts["startimage"].as<int>(), startimage); }
//      if(opts.count("endimage")) { endimage = std::min(opts["endimage"].as<int>(), endimage); }
//      std::vector<cv::Vec3d> debug_points;
//      std::vector<cv::Vec3b> debug_colors;

      // Set up visualization
      for (size_t i = 0; i < depth_filelist.size(); i++)
      {
          PCL_INFO ("On file %d / %d\n", i+1, depth_filelist.size ());
          PCL_INFO ("depth: %s\n", depth_filelist[i].c_str ());
          // add depth to organized point cloud
          cv::Mat depth_map = cv::imread(depth_filelist[i], -1);  // as is 16_u
          MaskImageSidesAsZero(not_use_side_column_length, &depth_map);
          cv::Mat image = cv::imread(image_filelist[i], 1);  // always color
          cv::Mat confidence_map;
          if (use_confidence)
          {
              PCL_INFO ("confidence: %s\n", confidence_filelist[i].c_str());
              confidence_map = cv::imread(confidence_filelist[i], -1);
              MaskImageSidesAsZero(not_use_side_column_length, &confidence_map);
          } else {
              PCL_INFO ("confidence is 1\n");
              confidence_map = cv::Mat::zeros(image.rows, image.cols, CV_16UC1) + 65535;
          }
          // semantic label mat
          cv::Mat semantic_label_mat;
          if (use_semantic_label && !semantic_filelist[i].empty())
            {
              PCL_INFO ("semantic_label: %s\n", semantic_filelist[i].c_str());
              semantic_label_mat = LoadSemanticLabelAsMat(semantic_filelist[i]);
            }

          //cam_infos[i].ProjectToWorldPointCloud(depth_map, image, &debug_points, &debug_colors);

          //tsdf->integrateCloud_Spherical_Queue(depth_map, confidence_map, image, semantic_label_mat, cam_infos[i]);
          tsdf->integrateCloud_Spherical_Queue_DoubleImCoord
                  (depth_map, confidence_map, image, semantic_label_mat, cam_infos[i], stepsize,
                   neg_dist_full_weight_delta,
                   neg_weight_thresh,
                   neg_weight_dist_thresh);

          bfs::path output_file_path(out_ply_file);
          std::string outputname((output_file_path.parent_path()/output_file_path.stem()).string());
          outputname += "_frame_" + boost::lexical_cast<string>(i) + "_" + bfs::path(depth_filelist[i]).stem().string() + ".ply";
//          Eigen::Vector3f min_pt(1252.7, 3784.2, 123.0);
//          Eigen::Vector3f max_pt(1253.9, 3785.4, 124.2);
//          Eigen::Vector3f min_pt(1256.0, 3785.8, 120.2);
//          Eigen::Vector3f max_pt(1256.4, 3786.2, 120.6);
          //CC_POINT_#111(1254.800049;3784.800049;122.599998)
//          Eigen::Vector3f min_pt(1254.4, 3784.4, 122.2);
//          Eigen::Vector3f max_pt(1255.2, 3785.2, 123.0);
//          Eigen::Vector3f min_pt(1254.7, 3783.1, 123.5);
//          Eigen::Vector3f max_pt(1255.3, 3783.7, 124.1);
//            Eigen::Vector3f min_pt(1249.5, 3784.9, 122.5);
//            Eigen::Vector3f max_pt(1250.1, 3785.5, 123.1);
//          Eigen::Vector3f min_pt(1052.3, 3736.4, 117.8);
//          Eigen::Vector3f max_pt(1052.7, 3736.8, 118.2);
//            tsdf->OutputTSDFGrid(outputname, &min_pt, &max_pt, &(cam_infos[i]), &(depth_map), &depth_image_scaling_factor);

//          if((i - startimage)%20 == 0 && i > 60)
//          {
//              bfs::path output_file_path(out_ply_file);
//              std::string pt_cloud_name((output_file_path.parent_path()/output_file_path.stem()).string());
//              pt_cloud_name += "_frame_" + boost::lexical_cast<string>(i) + "_" + bfs::path(depth_filelist[i]).stem().string() + "_ptcloud.ply";
//              utility::Write3DPointToFilePCL(pt_cloud_name, debug_points, &debug_colors);


//          }

      }
//      fprintf(stderr, "writing out TSDF file\n");
//      bfs::path output_file_path(out_ply_file);
//      std::string outputname((output_file_path.parent_path()/output_file_path.stem()).string());
//      outputname += "_origin_bin_tsdf_file.bin";
//      std::ofstream os(outputname, std::ios_base::out);
//      boost::archive::binary_oarchive oa(os);
//      oa << *tsdf;

      //tsdf->RemoveDuplicateSurfaceTSDF(min_weight);
      fprintf(stderr, "writing out TSDF file\n");
      bfs::path output_file_path(out_ply_file);
      std::string outputname((output_file_path.parent_path()/output_file_path.stem()).string());
      outputname += "_bin_tsdf_file.bin";
      std::ofstream os(outputname, std::ios_base::out);
      boost::archive::binary_oarchive oa(os);
      oa << *tsdf;
  }
  // do smoothing for TSDF.
  //bfs::path output_file_path(out_dir);
  //std::string outputname((output_file_path.parent_path()/output_file_path.stem()).string());
  //outputname += "_gridTSDF_beforesmooth.ply";
  //tsdf->OutputTSDFGrid(outputname);

//  if(opts.count("smooth"))
//  {
//      cpu_tsdf::TSDFHashing smoothed_tsdf;
//      SmoothTSDFFunctionOnlyExtension(tsdf.get(), 1, &smoothed_tsdf);
//      *tsdf = smoothed_tsdf;
//  }

  if(opts.count("diffusion-smooth"))
  {
      int niteration = 1;
      if (opts.count("niteration")) niteration = opts["niteration"].as<int>();
      cpu_tsdf::TSDFHashing smoothed_tsdf;
      cpu_tsdf::DiffusionHoleFilling(tsdf.get(), niteration, &smoothed_tsdf);
      *tsdf = smoothed_tsdf;
  }

  // Save
  //boost::filesystem::create_directory (out_dir);
  cpu_tsdf::MarchingCubesTSDFHashing mc;

  mc.setMinWeight(min_weight);
  mc.setInputTSDF (tsdf);
  pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh);
  fprintf(stderr, "perform reconstruction: \n");
  mc.reconstruct (*mesh);
  if (flatten)
    utility::flattenVertices (*mesh);
  if (cleanup)
    utility::cleanupMesh (*mesh);
  PCL_INFO ("Entire pipeline took %f ms\n", tt.toc ());
  if (save_ascii)
    pcl::io::savePLYFile (out_ply_file, *mesh);
  else
    pcl::io::savePLYFileBinary (out_ply_file, *mesh);
}



