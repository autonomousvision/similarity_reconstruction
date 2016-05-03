#include <iostream>
#include <string>
#include <vector>

#include <pcl/console/print.h>
#include <pcl/console/time.h>
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

//using std::vector;
//using std::string;
using namespace std;
using namespace utility;

int
main (int argc, char** argv)
{
  namespace bpo = boost::program_options;
  namespace bfs = boost::filesystem;
  bpo::options_description opts_desc("Allowed options");

  opts_desc.add_options()
    ("help,h", "produce help message")
    ("in", bpo::value<std::string>()->required (), "input vri file")
    ("rampsize", bpo::value<int>()->required(), "ramp size: 5/8")
    ("out", bpo::value<std::string>()->required (), "output ply path")
    ("verbose", "Verbose")
    ("save-ascii", "Save ply file as ASCII rather than binary")
    //("use_confidence", "use confidence map")
    ("mesh_min_weight", bpo::value<float>(), "minimum weight doing marching cubes")
    ("save_tsdf_bin", "save tsdf binary file")
    ;

  bpo::variables_map opts;
  bpo::store(bpo::parse_command_line(argc, argv, opts_desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), opts);
  bool badargs = false;
  try { bpo::notify(opts); }
  catch(...) { badargs = true; }
  if(opts.count("help") || badargs) {
    cout << "Usage: " << bfs::basename(argv[0]) << " --in [in_dir] --out [out_dir] [OPTS]" << endl;
    cout << endl;
    cout << opts_desc << endl;
    return (1);
  }
  ///////////////////////////////////////////////////////

  bool verbose = opts.count ("verbose");
  bool save_ascii = opts.count ("save-ascii");
  //bool use_confidence = opts.count("use_confidence");
  int ramp_size = opts["rampsize"].as<int>();
  string input_filename = opts["in"].as<string>();
  string output_plyfilename = opts["out"].as<string>();
  float mesh_min_weight = 0;
  if (opts.count("mesh_min_weight")) mesh_min_weight = opts["mesh_min_weight"].as<float>();

  std::cout << "begin vri tsdf converting. " << std::endl;
  pcl::console::TicToc tt;
  tt.tic ();
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
  fprintf(stderr, "bbox min_pt: %f, %f, %f\n", min_pt[0], min_pt[1], min_pt[2]);
  fprintf(stderr, "bbox max_pt: %f, %f, %f\n", max_pt[0], max_pt[1], max_pt[2]);
  fprintf(stderr, "offset: %f %f %f\n", offset(0), offset(1), offset(2));
  if (opts.count("save_tsdf_bin"))
  {
      string output_dir = bfs::path(output_plyfilename).remove_filename().string();
      string output_tsdffilename = (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string() + "_tsdf.bin")).string();
      std::cout << "save tsdf file path: " << output_tsdffilename << std::endl;
      std::ofstream os(output_tsdffilename, std::ios_base::out);
      boost::archive::binary_oarchive oa(os);
      oa << *tsdf;
  }

  std::cout << "begin marching cubes" << std::endl;
  cpu_tsdf::MarchingCubesTSDFHashing mc;
  mc.setMinWeight(mesh_min_weight);
  mc.setInputTSDF (tsdf);
  pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh);
  fprintf(stderr, "perform reconstruction: \n");
  mc.reconstruct (*mesh);
  PCL_INFO ("Entire pipeline took %f ms\n", tt.toc ());
  //flattenVertices(*mesh);
  if (save_ascii)
      pcl::io::savePLYFile (output_plyfilename, *mesh);
  else
      pcl::io::savePLYFileBinary (output_plyfilename, *mesh);

  return 0;
}



