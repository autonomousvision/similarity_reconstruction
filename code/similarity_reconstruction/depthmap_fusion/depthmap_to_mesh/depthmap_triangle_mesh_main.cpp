/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include <opencv2/opencv.hpp>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/segmentation/extract_clusters.h>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <string>
#include <vector>

#include "common/fisheye_camera/RectifiedCameraPair.h"
#include "common/data_load/urban_reconstruction_data_load.h"
#include "depthmap.h"
#include "common/utilities/pcl_utility.h"
#include "tsdf_hash_utilities/utility.h"

using std::vector;
using std::string;
using namespace std;
using namespace cv;
using namespace pcl;

//void InitializeCamInfos(int depth_width, int depth_height, const cv::Vec3d& offset, double voxel_length, double depth_image_scaling_factor, double cam_max_distance, std::vector<RectifiedCameraPair>& cam_infos) {
//  for (int i = 0; i < cam_infos.size(); ++i) {
//      cam_infos[i].SetVoxelScalingParameters(offset,
//                                             voxel_length,
//                                             depth_image_scaling_factor, cam_max_distance);
//      //cam_infos[i].InitializeBackProjectionBuffers(depth_width, depth_height);
//  }
//}

int
main (int argc, char** argv)
{
  namespace bpo = boost::program_options;
  namespace bfs = boost::filesystem;
  bpo::options_description opts_desc("Allowed options");

  string input_root;
  string depth_prefix;
  string param_prefix;
  string image_prefix;
  string output_dir;
  int start_image;
  int end_image;
  float max_cam_distance;
  float dd_factor;
  float voxel_length;
  int margin;

  float check_edge_ratio;
  float check_view_angle;

  opts_desc.add_options()
    ("help,h", "produce help message")
    ("in-root", bpo::value<std::string> (&input_root)->required ()->required(), "Input root dir")
    ("depth-prefix", bpo::value<std::string>(&depth_prefix)->required(), "Depth prefix")
    ("param-prefix", bpo::value<std::string>(&param_prefix)->required(), "Param prefix (folder name under rect/ directory)")
    ("image-prefix", bpo::value<std::string>(&image_prefix)->required(), "Image prefix (folder name under rect/ directory)")
    ("out", bpo::value<std::string> (&output_dir)->required (), "Output path")
    ("dd_factor", bpo::value<float> (&dd_factor)->required(), "depth difference factor")
    ("max-camera-distance", bpo::value<float>(&max_cam_distance)->required(), "Maximum allowed depth value")
    ("startimage", bpo::value<int>(&start_image)->required(), "starting image number")
    ("endimage", bpo::value<int>(&end_image)->required(), "ending image number")
    // ("use_confidence", "use confidence map")
    ("voxel_length", bpo::value<float>(&voxel_length)->default_value(0.2), "voxel length")
    ("flatten", "Remove duplicated mesh vertices during marching cubes")
    ("margin", bpo::value<int>(&margin)->default_value(0), "how much of the depth map should be cropped")
          ("edge_ratio", bpo::value<float>(&check_edge_ratio)->default_value(0), "edge ratio for filtering slanted triangles")
          ("view_angle", bpo::value<float>(&check_view_angle)->default_value(0), "view angle for filtering slanted triangles")
    ;

  bpo::variables_map opts;
  bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
  bool badargs = false;
  /*try*/ { bpo::notify(opts); }
//  catch(...) { badargs = true; }
  if(opts.count("help") || badargs) {
    cout << "Usage: " << bfs::basename(argv[0]) << " --in [in_dir] --out [out_dir] [OPTS]" << endl;
    cout << endl;
    cout << opts_desc << endl;
    return (1);
  }
  ///////////////////////////////////////////////////////
  bool flatten = opts.count ("flatten");

  // Read in all data
  cout << "Read in data " << endl;
  vector<RectifiedCameraPair> cam_infos;
  vector<string> depth_filelist;
  vector<string> cam_info_filelist;
  vector<string> image_filelist;
  bfs::path depth_path = bfs::path(input_root)/depth_prefix;
  // bfs::path param_path = bfs::path(input_root)/param_prefix;
  ListFilesWithinFrameRange(depth_path.string(), ".png", start_image, end_image, &depth_filelist);
  FindCorrespondingFiles(depth_filelist, (bfs::path(input_root)/param_prefix).string(),
                         ".txt", &cam_info_filelist);
  FindCorrespondingFiles(depth_filelist, (bfs::path(input_root)/image_prefix).string(),
                         ".png", &image_filelist);
  ReadCameraInfos(cam_info_filelist, cam_infos);
  cout << "read in depth files: " << depth_filelist.size() << endl;
  cout << "read in cam files: " << cam_infos.size() << endl;
  cout << "read in image files: " << image_filelist.size() << endl;
  if(depth_filelist.empty() || cam_infos.empty() || cam_infos.size() != depth_filelist.size()
          || image_filelist.empty() || image_filelist.size() != cam_infos.size())
  {
    printf("No depth maps or cam_infos founded, ");
    printf("or the number of depth maps and cam_infos don't agree.\n");
    exit(1);
  }
  float depth_image_scaling_factor = max_cam_distance/65535.0;
  cv::Mat temp_depth = cv::imread(depth_filelist[0], -1);
  cv::Vec3d offset(0, 0, 0);
  InitializeCamInfos(temp_depth.cols, temp_depth.rows,
                     offset, voxel_length, depth_image_scaling_factor, max_cam_distance, cam_infos);
  for(int i = 0; i < 1; ++i)
  {
      cout << i << "th frame:" << endl;
      cout << "depth: " << depth_filelist[i] << endl;
      cout << "cam file:" << cam_info_filelist[i] << endl;
      cout << "cam_info: " << cam_infos[i] << endl;
      cout << "image file: " << image_filelist[0] << endl;
  }
  cout << "dd_factor " << dd_factor << endl;

  cout << "convert depth maps to meshes " << endl;
  const bfs::path output_dir_path(output_dir);
  for (size_t i = 0; i < depth_filelist.size(); i++)
  {
    PCL_INFO ("On file %d / %d\n", i + 1, depth_filelist.size ());
    PCL_INFO ("depth: %s\n", depth_filelist[i].c_str ());
    // add depth to organized point cloud
    cv::Mat depth_map = cv::imread(depth_filelist[i], -1);  // as is
    cpu_tsdf::MaskImageSidesAsZero(margin, &depth_map);
    cv::Mat image = cv::imread(image_filelist[i], 1);  // always color
    pcl::PolygonMesh mesh;
//    cv::imshow("show", depth_map);
//    cv::waitKey();
    cpu_tsdf::DepthMapTriangulate(depth_map, image, cam_infos[i],
                                  dd_factor, check_edge_ratio, check_view_angle,
                                  &mesh);
    bfs::path cur_file_path(depth_filelist[i]);
    std::string saving_filename = ((output_dir_path/cur_file_path.stem()).replace_extension("depthmesh.ply")).string();
    if (flatten)
      utility::flattenVertices(mesh);
    pcl::io::savePLYFileBinary (saving_filename, mesh);
    cout << "saved " << i + 1 << "th depth" << endl;
  }
  return 0;
}



