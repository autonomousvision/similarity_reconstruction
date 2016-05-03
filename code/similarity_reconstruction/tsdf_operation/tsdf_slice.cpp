/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "tsdf_slice.h"
#include <numeric>
#include <queue>
#include <iostream>

// #include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <boost/lexical_cast.hpp>
// #include <pcl/io/vtk_lib_io.h>
// #include <pcl/pcl_macros.h>

#include "../tsdf_representation/tsdf_hash.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/utilities/common_utility.h"
//#include "../vrl_to_hash/ply2vri/ply2vri.h"
//#include "../vrl_to_hash/convert_vrl_to_hash.h"
//#include "../vrl_to_hash/vrl_grid_representation/OccGridRLE.h"

using utility::min_vec3;
using utility::max_vec3;
using utility::CvVectorToEigenVector3;
using utility::EigenVectorToCvVector3;

namespace
{
// some private functions
/**
 * @brief ComputeIntegral2DHist compute the integral image of the 2D histogram
 * @param hist_2d_xy
 * @param integral_2d_xy
 * @return
 */
bool ComputeIntegral2DHist(const std::vector<std::vector<float>> hist_2d_xy, std::vector<std::vector<float>>* integral_2d_xy);

/**
 * @brief MinimumAreaWithSemanticLabel Computing the minimum rectangle area
 * containing at least sum_threshold points with semantic labels.
 * @param integral_2d_xy
 * @param sum_threshold
 * @param t
 * @param l
 * @param b
 * @param r
 * @return
 */
bool MinimumAreaWithSemanticLabel(const std::vector<std::vector<float>>& integral_2d_xy, float sum_threshold, int* t, int* l, int* b, int*r);

bool ComputeIntegral2DHist(const std::vector<std::vector<float> > hist_2d_xy, std::vector<std::vector<float> > *integral_2d_xy)
{
  if (hist_2d_xy.empty()) return false;
  const int x_length = hist_2d_xy.size();
  const int y_length = hist_2d_xy[0].size();
  (*integral_2d_xy).resize(x_length + 1);
  for (int i = 0; i < x_length + 1; ++i)
    {
      (*integral_2d_xy)[i].resize(y_length + 1);
      (*integral_2d_xy)[i][0] = 0;
    }
  for (int i = 0; i < y_length + 1; ++i)
    {
      (*integral_2d_xy)[0][i] = 0;
    }
  for (int x = 1; x < x_length + 1; ++x)
    for (int y = 1; y < y_length + 1; ++y)
      {
        float cur_sum = hist_2d_xy[x-1][y-1] +
            (*integral_2d_xy)[x-1][y] +
            (*integral_2d_xy)[x][y-1] -
            (*integral_2d_xy)[x-1][y-1];
        (*integral_2d_xy)[x][y] = cur_sum;
      }
  return true;
}

bool MinimumAreaWithSemanticLabel(const std::vector<std::vector<float> > &integral_2d_xy,
                                            float sum_threshold, int *t, int *l, int *b, int *r)
{
  // get minimum rectangle with semantic label
  // breadth first search + pruning
  struct Configuration
  {
    float conf[4][2]; // t, l, b, r; (top, left, bottom, right) points
  };
  int x_length = integral_2d_xy.size() - 1;
  int y_length = integral_2d_xy[0].size() - 1;
  Configuration init;
  init.conf[0][0] = 0; init.conf[0][1] = y_length;  // not including
  init.conf[1][0] = 0; init.conf[1][1] = x_length;
  init.conf[2][0] = 0; init.conf[2][1] = y_length;
  init.conf[3][0] = 0; init.conf[3][1] = x_length;
  std::queue<Configuration> config_queue;
  config_queue.push(init);
  Configuration minimum_conf;
  minimum_conf.conf[0][0] = 0; minimum_conf.conf[0][1] = 0;  // not including
  minimum_conf.conf[1][0] = 0; minimum_conf.conf[1][1] = 0;
  minimum_conf.conf[2][0] = 0; minimum_conf.conf[2][1] = 0;
  minimum_conf.conf[3][0] = 0; minimum_conf.conf[3][1] = 0;
  float minimum_area = FLT_MAX;
  while(!config_queue.empty())
    {
      Configuration head = config_queue.front();
      config_queue.pop();
      float max_interval = head.conf[0][1] - head.conf[0][0];
      int max_interval_idx = 0;
      for (int i = 1; i < 4; ++i)
        {
          if (max_interval < head.conf[i][1] - head.conf[i][0])
            {
              max_interval = head.conf[i][1] - head.conf[i][0];
              max_interval_idx = i;
            }

        }
      // current configuration
      int i = head.conf[3][1]; //r
      int j = head.conf[2][1]; //b
      int p = head.conf[1][0]; //l
      int q = head.conf[0][0]; //t
      if (!(i > p && j > q)) continue;
      float max_total_semantic_label =
          integral_2d_xy[i][j] -
          integral_2d_xy[p][j] -
          integral_2d_xy[i][q] +
          integral_2d_xy[p][q];
      float cur_max_area = (i-p) * (j-q);
      if (max_interval < 1 && max_total_semantic_label >= sum_threshold &&
          minimum_area > cur_max_area)
        {
          // update current minimum area
          minimum_area = cur_max_area;
          minimum_conf.conf[0][0] = minimum_conf.conf[0][1] = q;
          minimum_conf.conf[1][0] = minimum_conf.conf[1][1] = p;
          minimum_conf.conf[2][0] = minimum_conf.conf[2][1] = j;
          minimum_conf.conf[3][0] = minimum_conf.conf[3][1] = i;
          continue;
        }
      else if (max_interval >= 1 && max_total_semantic_label >= sum_threshold)  // split
        {
          // split along the longest axis
          float mid_pt = std::floor((head.conf[max_interval_idx][0] +
              head.conf[max_interval_idx][1])/2.0);
          Configuration child1 = head;
          child1.conf[max_interval_idx][1] = mid_pt;
          Configuration child2 = head;
          child2.conf[max_interval_idx][0] = mid_pt + 1.0;
          config_queue.push(child1);
          config_queue.push(child2);
        }
    }
  *t = minimum_conf.conf[0][0];
  *l = minimum_conf.conf[1][0];
  *b = minimum_conf.conf[2][1];
  *r = minimum_conf.conf[3][1];
  return true;
}

#ifdef BOUNDINGBOX_1D_PROJECTION
bool ProjectTSDFTo1DHist(const TSDFHashing* tsdf_volume,
                         int semantic_label,
                         const Eigen::Matrix3f& orientation,
                         const Eigen::Vector3f& pmin_pt,
                         const Eigen::Vector3f& lengths,
                         const float voxel_length,
                         std::vector<std::vector<float>>* hist_1d_xyz,
                         int* total_cnt);
bool MinimumIntervalWithSemanticLabel(const std::vector<float>& hist_1d, float sum_threshold, int* st, int* ed); // ed: not including.

bool ProjectTSDFTo1DHist(const cpu_tsdf::TSDFHashing *tsdf_volume, int semantic_label,
                                   const Eigen::Matrix3f &orientation, const Eigen::Vector3f &pmin_pt, const Eigen::Vector3f &lengths,
                                   const float voxel_length, std::vector<std::vector<float> > *hist_1d_xyz, int *total_cnt)
{
  const Eigen::Matrix3f orientation_trans = orientation.transpose();
  hist_1d_xyz->resize(3);
  (*hist_1d_xyz)[0].resize(std::ceil(lengths[0]/voxel_length) + 1);
  (*hist_1d_xyz)[1].resize(std::ceil(lengths[1]/voxel_length) + 1);
  (*hist_1d_xyz)[2].resize(std::ceil(lengths[2]/voxel_length) + 1);
  int cnt = 0;
  for (TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
      cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
      float d, w;
      cv::Vec3b color;
      int vsemantic_label;
      VoxelData::VoxelState st;
      citr->RetriveData(&d, &w, &color, &st, &vsemantic_label);
      if (w > 0 && vsemantic_label == semantic_label)
        {
          cv::Vec3f world_coord;
          world_coord = tsdf_volume->Voxel2World(cur_voxel_coord);
          Eigen::Vector3f obb_coord = orientation_trans * (CvVectorToEigenVector3(world_coord) - pmin_pt);
          assert(obb_coord[0] >= -1e-5 && obb_coord[1] >= -1e-5 && obb_coord[2] >= -1e-5);
          (*hist_1d_xyz)[0][int(obb_coord[0]/voxel_length)] += 1;
          (*hist_1d_xyz)[1][int(obb_coord[1]/voxel_length)] += 1;
          (*hist_1d_xyz)[2][int(obb_coord[2]/voxel_length)] += 1;
          cnt++;
        }  // end if
    }  // end for
  *total_cnt = cnt;
  return true;
}

bool MinimumIntervalWithSemanticLabel(const std::vector<float> &hist_1d, float percent_threshold, int *st, int *ed)
{
  assert(!hist_1d.empty());
  const float hist_sum =std::accumulate(hist_1d.begin(),hist_1d.end(),0.0f);
  const float thresh_sum = hist_sum * percent_threshold;

  int st_idx = 0;
  int ed_idx = 1;  // not including
  float cur_sum = hist_1d[0];
  int min_interval = INT_MAX;
  int min_interval_st = -1;
  int min_interval_ed = -1;
  int min_sum = -1;

  while(st_idx < hist_1d.size())
    {
      while(ed_idx < hist_1d.size() && cur_sum < thresh_sum)
        {
          cur_sum += hist_1d[ed_idx];
          ed_idx++;
        }
      if (cur_sum >= thresh_sum && min_interval > ed_idx - st_idx)
        {
          min_interval = ed_idx - st_idx;
          min_interval_st = st_idx;
          min_interval_ed = ed_idx;
          min_sum = cur_sum;
        }
      cur_sum -= hist_1d[st_idx];
      st_idx++;
    }
  *st = min_interval_st;
  *ed = min_interval_ed;
  return min_sum == -1;
}
#endif
}  // end namespace

bool cpu_tsdf::ExtractSamplesFromAffineTransform(const TSDFHashing &scene_tsdf, const std::vector<Eigen::Affine3f> &affine_transforms, const TSDFGridInfo &options, Eigen::SparseMatrix<float, Eigen::ColMajor> *samples, Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
{
    const int sample_num = affine_transforms.size();
    const int feature_dim = options.boundingbox_size()[0] * options.boundingbox_size()[1] * options.boundingbox_size()[2];
    samples->resize(feature_dim, sample_num);
    samples->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
    weights->resize(feature_dim, sample_num);
    weights->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
    for (int i = 0; i < sample_num; ++i)
    {
        Eigen::SparseVector<float> sample(feature_dim);
        Eigen::SparseVector<float> weight(feature_dim);
        ExtractOneSampleFromAffineTransform(scene_tsdf, affine_transforms[i], options,
                                            &sample,
                                            &weight);
        Eigen::SparseVector<float>::InnerIterator it_s(sample);
        for (Eigen::SparseVector<float>::InnerIterator it(weight); it && it_s; ++it, ++it_s)
        {
            CHECK_EQ(it_s.index(), it.index());
            samples->insert(it.index(), i) = it_s.value();
            weights->insert(it.index(), i) = it.value();
        }
    }
    return true;
}

//bool cpu_tsdf::ExtractSamplesFromAffineTransform2(const TSDFHashing &scene_tsdf, const std::vector<Eigen::Affine3f> &affine_transforms, const TSDFGridInfo &options, Eigen::SparseMatrix<float, Eigen::ColMajor> *samples, Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
//{
//    const int sample_num = affine_transforms.size();
//    const int feature_dim = options.boundingbox_size()[0] * options.boundingbox_size()[1] * options.boundingbox_size()[2];
//    samples->resize(feature_dim, sample_num);
//    samples->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    weights->resize(feature_dim, sample_num);
//    weights->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    for (int i = 0; i < sample_num; ++i)
//    // for (int i = 0; i < 1; ++i)
//    {
//        Eigen::SparseVector<float> sample(feature_dim);
//        Eigen::SparseVector<float> weight(feature_dim);
//        ExtractOneSampleFromAffineTransform2(scene_tsdf, affine_transforms[i], options,
//                                            &sample,
//                                            &weight);
//      //  ExtractOneSampleFromAffineTransform(scene_tsdf, affine_transforms[i], options,
//      //                                      &sample,
//      //                                      &weight);
//        Eigen::SparseVector<float>::InnerIterator it_s(sample);
//        for (Eigen::SparseVector<float>::InnerIterator it(weight); it && it_s; ++it, ++it_s)
//        {
//            CHECK_EQ(it_s.index(), it.index());
//            samples->insert(it.index(), i) = it_s.value();
//            weights->insert(it.index(), i) = it.value();
//        }
//    }
//    return true;
//
//}

bool cpu_tsdf::ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const TSDFGridInfo &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
{
    using namespace std;
    // cout << "affine now: \n" << affine_transform.matrix() << endl;
    const Eigen::Vector3f& offset = options.offset();
    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size();
    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
    sample->resize(total_voxel_size);
    //sample->reserve(total_voxel_size * 0.6);
    weight->resize(total_voxel_size);
    //weight->reserve(total_voxel_size * 0.6);
    for (int x = 0; x < options.boundingbox_size()[0]; ++x)
        for (int y = 0; y < options.boundingbox_size()[1]; ++y)
            for (int z = 0; z < options.boundingbox_size()[2]; ++z)
            {
                Eigen::Vector3f current_world_point = offset;
                current_world_point[0] += options.voxel_lengths()[0] *  x;
                current_world_point[1] += options.voxel_lengths()[1] *  y;
                current_world_point[2] += options.voxel_lengths()[2] *  z;
                Eigen::Vector3f transformed_world_point = affine_transform * current_world_point;
                float cur_d = 0;
                float cur_w = 0;
                //if (tsdf_origin.RetriveDataFromWorldCoord_NearestNeighbor(cur_world_coord, &cur_d, &cur_w))
                if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) && /*cur_w > 0*/ cur_w > options.min_model_weight())
                {
                    int current_index = z +
                            (y + x * options.boundingbox_size()[1]) * options.boundingbox_size()[2];
                    sample->coeffRef(current_index) = cur_d;
                    weight->coeffRef(current_index) = cur_w;
                    //std::cerr << "world_pt: " << transformed_world_point << " cur_d: " << cur_d << " cur_w: " << cur_w << std::endl;
                }
            }
    return true;
}

// convert to mesh -> vri (large truncated distance) -> tsdf
//bool cpu_tsdf::ExtractOneSampleFromAffineTransform2(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const TSDFGridInfo &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    static int count = 0;
//    using namespace std;
//    cpu_tsdf::TSDFHashing::Ptr sliced_tsdf(new cpu_tsdf::TSDFHashing);
//    cpu_tsdf::OrientedBoundingBox obb;
//    cpu_tsdf::AffineToOrientedBB(affine_transform, &obb);
//    const double extension = 0.4;
//    obb.Extension(Eigen::Vector3f(extension, extension, extension));
//    cpu_tsdf::SliceTSDF(&scene_tsdf, PointInOrientedBox(obb), sliced_tsdf.get());
//    // write tsdf mesh
//    string tsdf_mesh_filename = string("/tmp/tmp") + boost::lexical_cast<string>(count) + ".ply";
//    std::cout << "begin marching cubes" << std::endl;
//    // pcl::PolygonMesh::Ptr mesh = cpu_tsdf::TSDFToPolygonMesh(sliced_tsdf, options.min_model_weight(), 0.00005);
//    pcl::PolygonMesh::Ptr mesh = cpu_tsdf::TSDFToPolygonMesh(sliced_tsdf, 0, 0.04);
//    std::cout << "save model at ply file path: " << tsdf_mesh_filename << std::endl;
//    pcl::io::savePLYFile (tsdf_mesh_filename, *mesh);
//    std::cout << "save finished" << std::endl;
//    // read in the mesh and convert to vri
//    string vri_filename = string("/tmp/tmp") + boost::lexical_cast<string>(count) + ".vri";
//    PLY2VRIOptions vri_options;
//    vri_options.rampSize = 10;
//    vri_options.exact_ramp_size = 6;
//    vri_options.resolution = scene_tsdf.voxel_length();
//    vri_options.hardEdges = true;
//    vri_options.useConf = false;
//    vri_options.stripEdges = 0;
//    vri_options.algorithm = 0;
//    vri_options.templateVRI = NULL;
//    vri_options.input_filename = tsdf_mesh_filename;
//    vri_options.output_filename = vri_filename;
//    ReadPLYAndConvertToVRI(vri_options);
//    // read in the vri and convert to tsdf (with larger truncation limit)
//    OccGridRLE rle_grid(1, 1, 1, CHUNK_SIZE);
//    if (!rle_grid.read(const_cast<char*>(vri_options.output_filename.c_str()))) return false;
//    cout << "reading VRI file finished" << endl;
//    cpu_tsdf::TSDFHashing::Ptr converted_sliced_tsdf(new cpu_tsdf::TSDFHashing);
//    float original_scene_max_trunc, original_scene_min_trunc;
//    scene_tsdf.getDepthTruncationLimits(original_scene_max_trunc, original_scene_min_trunc);
//    float new_truncation_limit =  vri_options.resolution * vri_options.rampSize;
//    converted_sliced_tsdf->Init(vri_options.resolution,
//                                Eigen::Vector3f(rle_grid.origin[0],rle_grid.origin[1],rle_grid.origin[2]),
//                                new_truncation_limit, -new_truncation_limit);
//    ConvertRLEGridToHashMap(rle_grid, vri_options.rampSize, converted_sliced_tsdf.get(), false);
//    cpu_tsdf::ScaleTSDFWeight(converted_sliced_tsdf.get(), converted_sliced_tsdf->getVoxelMaxWeight());
//    // convert the new sliced tsdf to sample and weight vector
//    ExtractOneSampleFromAffineTransform(*converted_sliced_tsdf, affine_transform, options, sample, weight);
//    count++;
//    return true;
//}

//bool ExtractOneSampleFromAffineTransformNearestNeighbor(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const TSDFGridInfo &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset();
//    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size();
//    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
//    sample->resize(total_voxel_size);
//    //sample->reserve(total_voxel_size * 0.6);
//    weight->resize(total_voxel_size);
//    //weight->reserve(total_voxel_size * 0.6);
//    for (int x = 0; x < options.boundingbox_size()[0]; ++x)
//        for (int y = 0; y < options.boundingbox_size()[1]; ++y)
//            for (int z = 0; z < options.boundingbox_size()[2]; ++z)
//            {
//                Eigen::Vector3f current_world_point = offset;
//                current_world_point[0] += options.voxel_lengths()[0] *  x;
//                current_world_point[1] += options.voxel_lengths()[1] *  y;
//                current_world_point[2] += options.voxel_lengths()[2] *  z;
//                Eigen::Vector3f transformed_world_point = affine_transform * current_world_point;
//                float cur_d, cur_w;
//                if (scene_tsdf.RetriveDataFromWorldCoord_NearestNeighbor(transformed_world_point, &cur_d, &cur_w) && cur_w > options.min_model_weight())
//                // if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) && /*cur_w > 0*/ cur_w > options.min_model_weight())
//                {
//                    int current_index = z +
//                            (y + x * options.boundingbox_size()[1]) * options.boundingbox_size()[2];
//                    sample->coeffRef(current_index) = cur_d;
//                    weight->coeffRef(current_index) = cur_w;
//                }
//            }
//    return true;
//}
//
//bool cpu_tsdf::ExtractSamplesFromAffineTransform(const TSDFHashing &scene_tsdf, const std::vector<Eigen::Affine3f> &affine_transforms, const PCAOptions &options, Eigen::SparseMatrix<float, Eigen::ColMajor> *samples, Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
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
//
//        ///////////////////////////////////////////////
//        //        Eigen::Matrix3f test_r;
//        //        Eigen::Vector3f test_scale;
//        //        Eigen::Vector3f test_trans;
//        //        utility::EigenAffine3fDecomposition(
//        //                    affine_transforms[i],
//        //                    &test_r,
//        //                    &test_scale,
//        //                    &test_trans);
//        //        TSDFHashing::Ptr cur_tsdf(new TSDFHashing);
//        //        ConvertDataVectorToTSDFWithWeight(
//        //        sample,
//        //        weight,
//        //        options,
//        //        cur_tsdf.get());
//        //        bfs::path output_path(options.save_path);
//        //        string save_path = (output_path.parent_path()/output_path.stem()).string() + "_check_affine_conversion_" + boost::lexical_cast<string>(i) + ".ply";
//        //        SaveTSDFModel(cur_tsdf, save_path, false, true, options.min_model_weight);
//        ///////////////////////////////////////////////
//    }
//    return true;
//}
//
//bool cpu_tsdf::ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const PCAOptions &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset;
//    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size;
//    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
//    sample->resize(total_voxel_size);
//    //sample->reserve(total_voxel_size * 0.6);
//    weight->resize(total_voxel_size);
//    //weight->reserve(total_voxel_size * 0.6);
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

//bool ExtractOneSampleFromAffineTransform2(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const PCAOptions &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset;
//    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size;
//    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
//    sample->resize(total_voxel_size);
//    weight->resize(total_voxel_size);
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
//        if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) &&
//                        cur_w > options.min_model_weight)
//                {
//                    int current_index = z +
//                            (y + x * options.boundingbox_size[1]) * options.boundingbox_size[2];
//                    sample->coeffRef(current_index) = cur_d;
//                    weight->coeffRef(current_index) = cur_w;
//                }
//            }
//    return true;
//}

bool cpu_tsdf::SliceTSDFWithBoundingbox(const cpu_tsdf::TSDFHashing *tsdf_volume,
                         const Eigen::Matrix3f &voxel_world_rotation,
                         const Eigen::Vector3f &offset,
                         const Eigen::Vector3f &world_side_lengths,
                         const float voxel_length,
                         cpu_tsdf::TSDFHashing *sliced_tsdf)
{
  using namespace std;
  const float original_voxel_length = tsdf_volume->voxel_length();
  float max_dist_pos, max_dist_neg;
  tsdf_volume->getDepthTruncationLimits(max_dist_pos, max_dist_neg);

  std::cout << "begin slicing TSDF. " << std::endl;
  Eigen::Vector3f box_vertices[8];
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        {
          box_vertices[x * 4 + y * 2 + z] = offset + x * world_side_lengths[0] * voxel_world_rotation.col(0) +
              y * world_side_lengths[1] * voxel_world_rotation.col(1) +
              z * world_side_lengths[2] * voxel_world_rotation.col(2);
        }
  Eigen::Vector3f aabb_min = box_vertices[0];
  Eigen::Vector3f aabb_max = box_vertices[0];
  for (int i = 1; i < 8; ++i)
    {
      aabb_min = min_vec3(aabb_min, box_vertices[i]);
      aabb_max = max_vec3(aabb_max, box_vertices[i]);
    }
  Eigen::Vector3i voxel_side_length = ((aabb_max - aabb_min)/voxel_length).cast<int>() +
      Eigen::Vector3i(1, 1, 1);
  sliced_tsdf->Init(voxel_length, aabb_min, max_dist_pos, max_dist_neg);
  cout << "side lengths: \n" << voxel_side_length << endl;
  cout << "aabbmin: \n" << aabb_min << endl;
  cout << "aabbmax: \n" << aabb_max << endl;
  cout << "voxel_length: \n" << voxel_length << endl;
  cout << "max, min trunc dist: " << max_dist_pos << "; " << max_dist_neg << endl;
  TSDFHashing::update_hashset_type update_hashset;
  // all bricks in the AABB are added..
  for (int ix = 0; ix < voxel_side_length[0]; ix += VoxelHashMap::kBrickSideLength)
    for (int iy = 0; iy < voxel_side_length[1]; iy += VoxelHashMap::kBrickSideLength)
      for (int iz = 0; iz < voxel_side_length[2]; iz += VoxelHashMap::kBrickSideLength)
        {
          update_hashset.insert(VoxelHashMap::BrickPosition(ix, iy, iz));
        }
  cout << "update_hashset size: " << update_hashset.size() << endl;

  // then add the voxels in the oriented bounding box to sliced_tsdf
  struct TSDFVoxelUpdater
  {
    TSDFVoxelUpdater(const TSDFHashing* tsdf_origin, const TSDFHashing* sliced_tsdf,
                     const Eigen::Matrix3f& obb_rotation_trans, const Eigen::Vector3f& obb_offset,
                     const Eigen::Vector3f& obb_sidelength)
      : tsdf_origin(tsdf_origin), sliced_tsdf(sliced_tsdf),
        obb_rotation_trans(obb_rotation_trans),
        obb_offset(obb_offset),
        obb_sidelength(obb_sidelength) {}
    bool operator () (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
    {
      Eigen::Vector3f world_coord = sliced_tsdf->Voxel2World(CvVectorToEigenVector3(cv::Vec3f(cur_voxel_coord)));
      Eigen::Vector3f obb_coord = obb_rotation_trans * (world_coord - obb_offset);
      if (! (obb_coord[0] >= 0 && obb_coord[0] <= obb_sidelength[0] &&
             obb_coord[1] >= 0 && obb_coord[1] <= obb_sidelength[1] &&
             obb_coord[2] >= 0 && obb_coord[2] <= obb_sidelength[2]))
        {
          return false;
        }
      float cur_d;
      float cur_w;
      cv::Vec3b cur_color;
      if (!tsdf_origin->RetriveDataFromWorldCoord(world_coord, &cur_d, &cur_w, &cur_color))
      {
          return false;
      }
      *d = cur_d;
      *w = cur_w;
      *color = cur_color;
      return true;
    }
    const TSDFHashing* tsdf_origin;
    const TSDFHashing* sliced_tsdf;
    const Eigen::Matrix3f obb_rotation_trans;
    const Eigen::Vector3f obb_offset;
    const Eigen::Vector3f obb_sidelength;
  };
  TSDFVoxelUpdater updater(tsdf_volume, sliced_tsdf, voxel_world_rotation.transpose(), offset, world_side_lengths);
  sliced_tsdf->UpdateBricksInQueue(update_hashset,
                                   updater);
  cout << "finished slicing TSDF. " << endl;
  return true;
}

bool cpu_tsdf::SliceTSDFWithBoundingbox_NearestNeighbor(const cpu_tsdf::TSDFHashing *tsdf_volume,
                         const Eigen::Matrix3f &voxel_world_rotation,
                         const Eigen::Vector3f &offset,
                         const Eigen::Vector3f &world_side_lengths,
                         const float voxel_length,
                         cpu_tsdf::TSDFHashing *sliced_tsdf)
{
  using namespace std;
  const float original_voxel_length = tsdf_volume->voxel_length();
  float max_dist_pos, max_dist_neg;
  tsdf_volume->getDepthTruncationLimits(max_dist_pos, max_dist_neg);

  std::cout << "begin slicing TSDF. " << std::endl;
  Eigen::Vector3f box_vertices[8];
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        {
          box_vertices[x * 4 + y * 2 + z] = offset + x * world_side_lengths[0] * voxel_world_rotation.col(0) +
              y * world_side_lengths[1] * voxel_world_rotation.col(1) +
              z * world_side_lengths[2] * voxel_world_rotation.col(2);
        }
  Eigen::Vector3f aabb_min = box_vertices[0];
  Eigen::Vector3f aabb_max = box_vertices[0];
  for (int i = 1; i < 8; ++i)
    {
      aabb_min = min_vec3(aabb_min, box_vertices[i]);
      aabb_max = max_vec3(aabb_max, box_vertices[i]);
    }
  Eigen::Vector3i voxel_side_length = ((aabb_max - aabb_min)/voxel_length).cast<int>() +
      Eigen::Vector3i(1, 1, 1);
  sliced_tsdf->Init(voxel_length, aabb_min, max_dist_pos, max_dist_neg);
  cout << "side lengths: \n" << voxel_side_length << endl;
  cout << "aabbmin: \n" << aabb_min << endl;
  cout << "aabbmax: \n" << aabb_max << endl;
  cout << "voxel_length: \n" << voxel_length << endl;
  cout << "max, min trunc dist: " << max_dist_pos << "; " << max_dist_neg << endl;
  TSDFHashing::update_hashset_type update_hashset;
  // all bricks in the AABB are added..
  for (int ix = 0; ix < voxel_side_length[0]; ix += VoxelHashMap::kBrickSideLength)
    for (int iy = 0; iy < voxel_side_length[1]; iy += VoxelHashMap::kBrickSideLength)
      for (int iz = 0; iz < voxel_side_length[2]; iz += VoxelHashMap::kBrickSideLength)
        {
          update_hashset.insert(VoxelHashMap::BrickPosition(ix, iy, iz));
        }
  cout << "update_hashset size: " << update_hashset.size() << endl;

  // then add the voxels in the oriented bounding box to sliced_tsdf
  struct TSDFVoxelUpdater
  {
    TSDFVoxelUpdater(const TSDFHashing* tsdf_origin, const TSDFHashing* sliced_tsdf,
                     const Eigen::Matrix3f& obb_rotation_trans, const Eigen::Vector3f& obb_offset,
                     const Eigen::Vector3f& obb_sidelength)
      : tsdf_origin(tsdf_origin), sliced_tsdf(sliced_tsdf),
        obb_rotation_trans(obb_rotation_trans),
        obb_offset(obb_offset),
        obb_sidelength(obb_sidelength) {}
    bool operator () (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
    {
      Eigen::Vector3f world_coord = sliced_tsdf->Voxel2World(CvVectorToEigenVector3(cv::Vec3f(cur_voxel_coord)));
      Eigen::Vector3f obb_coord = obb_rotation_trans * (world_coord - obb_offset);
      if (! (obb_coord[0] >= 0 && obb_coord[0] <= obb_sidelength[0] &&
             obb_coord[1] >= 0 && obb_coord[1] <= obb_sidelength[1] &&
             obb_coord[2] >= 0 && obb_coord[2] <= obb_sidelength[2]))
        {
          return false;
        }
      float cur_d;
      float cur_w;
      cv::Vec3b cur_color;
      if (!tsdf_origin->RetriveDataFromWorldCoord_NearestNeighbor(world_coord, &cur_d, &cur_w, &cur_color))
      {
          return false;
      }
      *d = cur_d;
      *w = cur_w;
      *color = cur_color;
      return true;
    }
    const TSDFHashing* tsdf_origin;
    const TSDFHashing* sliced_tsdf;
    const Eigen::Matrix3f obb_rotation_trans;
    const Eigen::Vector3f obb_offset;
    const Eigen::Vector3f obb_sidelength;
  };
  TSDFVoxelUpdater updater(tsdf_volume, sliced_tsdf, voxel_world_rotation.transpose(), offset, world_side_lengths);
  sliced_tsdf->UpdateBricksInQueue(update_hashset,
                                   updater);
  cout << "finished slicing TSDF. " << endl;
  return true;
}


bool cpu_tsdf::GetTSDFSemanticPartAxisAlignedBoundingbox(const cpu_tsdf::TSDFHashing *tsdf_volume,
                                              int semantic_label,
                                              int neighborhood,
                                              Eigen::Vector3f *pmin_pt, Eigen::Vector3f *lengths)
{
  fprintf(stderr, "begin getting bounding box..");
  cv::Vec3f min_pt(FLT_MAX, FLT_MAX, FLT_MAX);
  cv::Vec3f max_pt(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  const float voxel_length = tsdf_volume->voxel_length();
  cv::Vec3f delta(voxel_length * neighborhood, voxel_length * neighborhood, voxel_length * neighborhood);
  for (TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
      cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
      float d, w;
      cv::Vec3b color;
      int vsemantic_label;
      VoxelData::VoxelState st;
      citr->RetriveData(&d, &w, &color, &st, &vsemantic_label);
      if (w > 0 && vsemantic_label == semantic_label)
        {
          cv::Vec3f world_coord;
          world_coord = tsdf_volume->Voxel2World(cur_voxel_coord);
          min_pt = min_vec3(min_pt, world_coord - delta);
          max_pt = max_vec3(max_pt, world_coord + delta);
        }  // end if
    }  // end for
  *pmin_pt = CvVectorToEigenVector3(min_pt);
  *lengths = CvVectorToEigenVector3(max_pt - min_pt);
  fprintf(stderr, "finished\n");
  return min_pt != cv::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
}

bool cpu_tsdf::GetTSDFSemanticPartOrientedBoundingbox(const cpu_tsdf::TSDFHashing *tsdf_volume, int semantic_label, int neighborhood,
                                                      const Eigen::Matrix3f &orientation, Eigen::Vector3f *pmin_pt, Eigen::Vector3f *lengths)
{
  fprintf(stderr, "begin getting bounding box..");
  const Eigen::Matrix3f orientation_trans = orientation.transpose();
  const Eigen::Vector3f offset = tsdf_volume->getVoxelOriginInWorldCoord();
  Eigen::Vector3f min_pt(FLT_MAX, FLT_MAX, FLT_MAX);
  Eigen::Vector3f max_pt(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  const float voxel_length = tsdf_volume->voxel_length();
  Eigen::Vector3f delta(voxel_length * neighborhood, voxel_length * neighborhood, voxel_length * neighborhood);
  for (TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
      cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
      float d, w;
      cv::Vec3b color;
      int vsemantic_label;
      VoxelData::VoxelState st;
      citr->RetriveData(&d, &w, &color, &st, &vsemantic_label);
      if (w > 0 && vsemantic_label == semantic_label)
        {
          cv::Vec3f world_coord;
          world_coord = tsdf_volume->Voxel2World(cur_voxel_coord);
          Eigen::Vector3f obb_coord = orientation_trans * (CvVectorToEigenVector3(world_coord));
          // here min_pt, max_pt refer to oriented bounding box coordinate
          min_pt = min_vec3(min_pt, Eigen::Vector3f(obb_coord - delta));
          max_pt = max_vec3(max_pt, Eigen::Vector3f(obb_coord + delta));
        }  // end if
    }  // end for
  // min_pt in world coordinate
  *pmin_pt = orientation * min_pt;
  // lengths in world coordinate
  *lengths = max_pt - min_pt;
  std::cout << "min_pt: \n" << *pmin_pt << std::endl;
  std::cout << "lengths: \n" << *lengths << std::endl;
  fprintf(stderr, "finished\n");
  return min_pt != Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
}

bool cpu_tsdf::GetTSDFSemanticMajorPartOrientedBoundingbox2D(const cpu_tsdf::TSDFHashing *tsdf_volume, int semantic_label,
                                                             int neighborhood, float thresh_percent,
                                                             const Eigen::Matrix3f &orientation, Eigen::Vector3f *pmin_pt, Eigen::Vector3f *lengths)
{
  // 1. get the oriented boundingbox for this semantic label
  fprintf(stderr, "begin major part obb computing 2D\n");
  Eigen::Vector3f whole_min_pt;
  Eigen::Vector3f whole_lengths;
  if (!GetTSDFSemanticPartOrientedBoundingbox(tsdf_volume, semantic_label, 0, orientation, &whole_min_pt, &whole_lengths)) return false;

  // 2. project the points to 2D in the oriented boundingbox.
  const int ground_orient_idx[2] = {0, 2};
  Eigen::Vector3f ground_orientations[2] = {orientation.col(ground_orient_idx[0]), orientation.col(ground_orient_idx[1])};
  float ground_box_lengths[2] = {whole_lengths[ground_orient_idx[0]], whole_lengths[ground_orient_idx[1]]};
  std::vector<std::vector<float>> hist_2d_xy;
  const float voxel_length = tsdf_volume->voxel_length();
  int total_cnt = 0;
  ProjectTSDFTo2DHist(tsdf_volume, semantic_label, ground_orientations,
                      whole_min_pt, ground_box_lengths, voxel_length, &hist_2d_xy, &total_cnt);
  // get integral image
  std::vector<std::vector<float>> integral_2d_xy;
  ComputeIntegral2DHist(hist_2d_xy, &integral_2d_xy);
  // get minimum area containing the semantic label (semantic labeled points above a threshold)
  int t, l, b, r;
  MinimumAreaWithSemanticLabel(integral_2d_xy, total_cnt * thresh_percent, &t, &l, &b, &r);  //l, r: x direction
  // 3. extend the boundingbox by (2 * neighborhood * voxel_length)
  Eigen::Vector3f cur_st_world = whole_min_pt;
  Eigen::Vector3f cur_lengths = whole_lengths;
  cur_st_world = cur_st_world + (l - neighborhood) * voxel_length * orientation.col(ground_orient_idx[0]) +
      (t - neighborhood) * voxel_length * orientation.col(ground_orient_idx[1]);
  cur_lengths[ground_orient_idx[0]] = (r - l + 2 * neighborhood) * voxel_length;
  cur_lengths[ground_orient_idx[1]] = (b - t + 2 * neighborhood) * voxel_length;
  *pmin_pt = cur_st_world;
  *lengths = cur_lengths;
  std::cout << "major part min_pt: \n" << *pmin_pt << std::endl;
  std::cout << "major part lengths: \n" << *lengths << std::endl;
  fprintf(stderr, "finished 2D\n");
  return true;
}



// bool cpu_tsdf::ComputeOrientedBoundingboxVertices(const Eigen::Matrix3f &voxel_world_rotation, const Eigen::Vector3f &offset, const Eigen::Vector3f &world_side_lengths,
//                                                   Eigen::Vector3f* box_vertices)
// {
//   for (int x = 0; x < 2; ++x)
//     for (int y = 0; y < 2; ++y)
//       for (int z = 0; z < 2; ++z)
//         {
//           box_vertices[x * 4 + y * 2 + z] = offset + x * world_side_lengths[0] * voxel_world_rotation.col(0) +
//               y * world_side_lengths[1] * voxel_world_rotation.col(1) +
//               z * world_side_lengths[2] * voxel_world_rotation.col(2);
//         }
//   return true;
// }


// bool cpu_tsdf::SaveOrientedBoundingbox(const Eigen::Matrix3f &orientation, const Eigen::Vector3f &offset, const Eigen::Vector3f &lengths, const std::string &filename)
// {
//   pcl::PolygonMesh mesh;
//   Eigen::Vector3f box_vertices[8];
//   ComputeOrientedBoundingboxVertices(orientation, offset, lengths, box_vertices);
//   pcl::PointCloud<pcl::PointXYZRGB> vertices_new;
//   for (size_t i = 0; i < 8; i++)
//   {
//     pcl::PointXYZRGB pt_final;
//     pt_final.x = box_vertices[i][0];
//     pt_final.y = box_vertices[i][1];
//     pt_final.z = box_vertices[i][2];
//     pt_final.r = 255;
//     pt_final.g = 255;
//     pt_final.b = 255;
//     vertices_new.push_back (pt_final);
//   }
//   mesh.polygons.resize(12);
//   for (size_t i = 0; i < 12; i++)
//     {
//       mesh.polygons[i].vertices.resize(3);
//     }
//   mesh.polygons[0].vertices[0] = 0; mesh.polygons[0].vertices[1] = 1; mesh.polygons[0].vertices[2] = 3;
//   mesh.polygons[1].vertices[0] = 0; mesh.polygons[1].vertices[1] = 3; mesh.polygons[1].vertices[2] = 2;
//   mesh.polygons[2].vertices[0] = 6; mesh.polygons[2].vertices[1] = 7; mesh.polygons[2].vertices[2] = 5;
//   mesh.polygons[3].vertices[0] = 6; mesh.polygons[3].vertices[1] = 5; mesh.polygons[3].vertices[2] = 4;
//   mesh.polygons[4].vertices[0] = 0; mesh.polygons[4].vertices[1] = 4; mesh.polygons[4].vertices[2] = 5;
//   mesh.polygons[5].vertices[0] = 0; mesh.polygons[5].vertices[1] = 5; mesh.polygons[5].vertices[2] = 1;
//   mesh.polygons[6].vertices[0] = 1; mesh.polygons[6].vertices[1] = 5; mesh.polygons[6].vertices[2] = 7;
//   mesh.polygons[7].vertices[0] = 1; mesh.polygons[7].vertices[1] = 7; mesh.polygons[7].vertices[2] = 3;
//   mesh.polygons[8].vertices[0] = 3; mesh.polygons[8].vertices[1] = 7; mesh.polygons[8].vertices[2] = 6;
//   mesh.polygons[9].vertices[0] = 3; mesh.polygons[9].vertices[1] = 6; mesh.polygons[9].vertices[2] = 2;
//   mesh.polygons[10].vertices[0] = 2; mesh.polygons[10].vertices[1] = 6; mesh.polygons[10].vertices[2] = 4;
//   mesh.polygons[11].vertices[0] = 2; mesh.polygons[11].vertices[1] = 4; mesh.polygons[11].vertices[2] = 0;
//   pcl::toPCLPointCloud2 (vertices_new, mesh.cloud);
//   pcl::io::savePLYFile (filename, mesh);
//   return true;
// }

bool cpu_tsdf::ProjectTSDFTo2DHist(const cpu_tsdf::TSDFHashing* tsdf_volume, int semantic_label, const Eigen::Vector3f* ground_orientation,
                                   const Eigen::Vector3f& pmin_pt, const float *lengths,
                                   const float voxel_length, std::vector<std::vector<float>>* hist_2d_xy, int* total_cnt)
{
  hist_2d_xy->resize(std::ceil(lengths[0]/voxel_length) + 1);
  const int y_length = std::ceil(lengths[1]/voxel_length) + 1;
  for (int i = 0; i < hist_2d_xy->size(); ++i)
    {
      (*hist_2d_xy)[i].resize(y_length);
    }
  int cnt = 0;
  for (cpu_tsdf::TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
      cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
      float d, w;
      cv::Vec3b color;
      int vsemantic_label;
      cpu_tsdf::VoxelData::VoxelState st;
      citr->RetriveData(&d, &w, &color, &st, &vsemantic_label);
      if (w > 0 && vsemantic_label == semantic_label)
        {
          cv::Vec3f world_coord;
          world_coord = tsdf_volume->Voxel2World(cur_voxel_coord);
          Eigen::Vector3f cur_offseted_point = (CvVectorToEigenVector3(world_coord) - pmin_pt);
          float x = ground_orientation[0].dot(cur_offseted_point);
          float y = ground_orientation[1].dot(cur_offseted_point);
          assert(x >= -1e-5 && y >= -1e-5);
          if (x > lengths[0] || y > lengths[1]) continue;
          (*hist_2d_xy)[int(x/voxel_length)][int(y/voxel_length)] += 1;
          cnt++;
        }  // end if
    }  // end for
  *total_cnt = cnt;
  return true;
}

bool cpu_tsdf::ProjectTSDFTo2DHist(const cpu_tsdf::TSDFHashing *tsdf_volume, int semantic_label, const Eigen::Vector3f *ground_orientation,
                                   const Eigen::Vector3f &pmin_pt, const float *lengths, const float voxel_length,
                                   cv::Mat_<float> *im_hist_2d_xy, int *total_cnt)
{
  std::vector<std::vector<float>> hist_2d_xy;
  if (!ProjectTSDFTo2DHist(tsdf_volume, semantic_label, ground_orientation, pmin_pt, lengths,
                           voxel_length, &hist_2d_xy, total_cnt)) return false;

  int rows = hist_2d_xy.size();
  int cols = hist_2d_xy[0].size();
  im_hist_2d_xy->create(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      {
        im_hist_2d_xy->at<float>(i, j) = hist_2d_xy[i][j];
      }
  return true;
}

#ifdef BOUNDINGBOX_1D_PROJECTION
bool cpu_tsdf::GetTSDFSemanticMajorPartOrientedBoundingbox1D(const cpu_tsdf::TSDFHashing *tsdf_volume, int semantic_label, int neighborhood, float thresh_percent,
                                                           const Eigen::Matrix3f &orientation, Eigen::Vector3f *pmin_pt, Eigen::Vector3f *lengths)
{
  fprintf(stderr, "begin major part obb computing\n");
  Eigen::Vector3f whole_min_pt;
  Eigen::Vector3f whole_lengths;
  if (!GetTSDFSemanticPartOrientedBoundingbox(tsdf_volume, semantic_label, 0, orientation, &whole_min_pt, &whole_lengths)) return false;
  std::vector<std::vector<float>> hist_1d_xyz;
  const float voxel_length = tsdf_volume->voxel_length();
  int total_cnt = 0;
  ProjectTSDFTo1DHist(tsdf_volume, semantic_label, orientation,
                      whole_min_pt, whole_lengths, voxel_length, &hist_1d_xyz, &total_cnt);
  Eigen::Vector3f cur_st_world = whole_min_pt;
  Eigen::Vector3f cur_lengths = whole_lengths;
  for (int i = 0; i < 3; ++i)  // for x and y directions
    {
      if (i == 0 || i == 1) continue;
      int cur_st;
      int cur_ed;
      MinimumIntervalWithSemanticLabel(hist_1d_xyz[i], thresh_percent, &cur_st, &cur_ed);
      cur_st_world = cur_st_world + (cur_st - neighborhood) * voxel_length * orientation.col(i);
      cur_lengths[i] = (cur_ed - cur_st + 2 * neighborhood) * voxel_length;
    }
  *pmin_pt = cur_st_world;
  *lengths = cur_lengths;
  std::cout << "major part min_pt: " << *pmin_pt << std::endl;
  std::cout << "major part lengths: " << *lengths << std::endl;
  fprintf(stderr, "finished\n");
  return true;
}
#endif


bool cpu_tsdf::ComputeOrientedBoundingbox(const cpu_tsdf::TSDFHashing *tsdf_origin, const Eigen::Matrix3f &orientation, Eigen::Vector3f *offset, Eigen::Vector3f *sidelengths)
{
    Eigen::Vector3f oriented_min_pt(FLT_MAX, FLT_MAX, FLT_MAX);
    Eigen::Vector3f oriented_max_pt(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    const Eigen::Matrix3f orientation_t = orientation.transpose();
    for (TSDFHashing::const_iterator citr = tsdf_origin->begin(); citr !=  tsdf_origin->end(); ++citr)
    {
        float d, w;
        cv::Vec3b color;
        if (citr->RetriveData(&d, &w, &color))
        {
            Eigen::Vector3f proj_pt = orientation_t * utility::CvVectorToEigenVector3(tsdf_origin->Voxel2World(citr.VoxelCoord()));
            oriented_min_pt = utility::min_vec3(oriented_min_pt, proj_pt);
            oriented_max_pt = utility::max_vec3(oriented_max_pt, proj_pt);
        }
    }
    std::cout << "o_min_pt: \n" << oriented_min_pt << std::endl;
    std::cout << "o_max_pt: \n" << oriented_max_pt << std::endl;
    *offset = orientation * oriented_min_pt;
    *sidelengths = oriented_max_pt - oriented_min_pt;
    return true;
}


//bool cpu_tsdf::MergeTSDF(const TSDFHashing &src, cpu_tsdf::TSDFHashing *target,
//                         float neg_dist_full_weight_delta,
//                         float neg_weight_thresh,
//                         float neg_weight_dist_thresh)
//{
////    CHECK_EQ(src.offset()[0], target->offset()[0]);
////    CHECK_EQ(src.offset()[1], target->offset()[1]);
////    CHECK_EQ(src.offset()[2], target->offset()[2]);
//    for (TSDFHashing::const_iterator citr = src.begin(); citr != src.end(); ++citr)
//    {
//        float d, w;
//        cv::Vec3b color;
//        if (citr->RetriveData(&d, &w, &color))
//        {
//            cv::Vec3f world_point = src.Voxel2World(citr.VoxelCoord());
//            //float neg_w = 1.0;
//            float neg_w = target->ComputeTSDFWeight(d, neg_dist_full_weight_delta, neg_weight_thresh, neg_weight_dist_thresh);
//            target->AddObservationFromWorldCoord(world_point, d, neg_w * w, color);
//        }
//    }
//    return true;
//}

bool cpu_tsdf::MergeTSDFNearestNeighbor(const TSDFHashing &src, cpu_tsdf::TSDFHashing *target)
{
    for (TSDFHashing::const_iterator citr = src.begin(); citr != src.end(); ++citr)
    {
        float d, w;
        cv::Vec3b color;
        if (citr->RetriveData(&d, &w, &color))
        {
            //std::cout << "src, voxel: \n" << citr.VoxelCoord() << std::endl;
            cv::Vec3f world_point = src.Voxel2World(citr.VoxelCoord());
            cv::Vec3i cur_voxel = citr.VoxelCoord();
            cv::Vec3i transformed_voxel = cv::Vec3i(target->World2Voxel(world_point));
            CHECK_EQ(cur_voxel[0], transformed_voxel[0]);
            CHECK_EQ(cur_voxel[1], transformed_voxel[1]);
            CHECK_EQ(cur_voxel[2], transformed_voxel[2]);
            target->AddObservationFromWorldCoord(world_point, d, w, color);
        }
    }
    return true;
}


bool cpu_tsdf::MergeTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &srcs, cpu_tsdf::TSDFHashing *target)
{
    for (int i = 0; i < srcs.size(); ++i)
    {
        MergeTSDF(*(srcs[i]), target);
    }
    return true;
}


bool cpu_tsdf::CleanTSDFPart(cpu_tsdf::TSDFHashing *tsdf_volume, const cpu_tsdf::OrientedBoundingBox &obb)
{
    PointInOrientedBox pred(obb);
    using namespace std;
    const float original_voxel_length = tsdf_volume->voxel_length();
    float max_dist_pos, max_dist_neg;
    tsdf_volume->getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    Eigen::Vector3f voxel_world_offset = tsdf_volume->getVoxelOriginInWorldCoord();
    //sliced_tsdf->Init(original_voxel_length, voxel_world_offset, max_dist_pos, max_dist_neg);
    cout << "Begin filtering TSDF. " << endl;
    cout << "voxel length: \n" << original_voxel_length << endl;
    cout << "aabbmin: \n" << voxel_world_offset << endl;
    cout << "max, min trunc dist: " << max_dist_pos << "; " << max_dist_neg << endl;
    for (TSDFHashing::iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        int vsemantic_label;
        VoxelData::VoxelState st;
        citr->RetriveData(&d, &w, &color, &st, &vsemantic_label);
        if (w > 0 && pred(tsdf_volume, cur_voxel_coord, d, w, color, vsemantic_label))
        {
            //sliced_tsdf->AddObservation(cur_voxel_coord, d, w, color, vsemantic_label);
            //oper(cur_voxel_coord, d, w, color, vsemantic_label);
            citr->ClearVoxel();
        }  // end if
    }  // end for
    //sliced_tsdf->DisplayInfo();
    cout << "End filtering TSDF. " << endl;
    return true;
    //return !sliced_tsdf->Empty();
}


bool cpu_tsdf::CleanTSDFPart(cpu_tsdf::TSDFHashing *tsdf_volume, const Eigen::Affine3f &affine)
{
    cpu_tsdf::OrientedBoundingBox obb;
    cpu_tsdf::AffineToOrientedBB(affine, &obb);
    return CleanTSDFPart(tsdf_volume, obb);
}


bool cpu_tsdf::MergeTSDF(const cpu_tsdf::TSDFHashing &tsdf_origin, cpu_tsdf::TSDFHashing *target)
{
    using namespace std;
    cout << "begin merge TSDF. " << endl;
    TSDFHashing::update_hashset_type brick_update_hashset;
    for (TSDFHashing::const_iterator citr = tsdf_origin.begin(); citr != tsdf_origin.end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        citr->RetriveData(&d, &w, &color);
        if (w > 0)
        {
            cv::Vec3f world_coord = tsdf_origin.Voxel2World(cv::Vec3f(cur_voxel_coord));
            cv::Vec3i voxel_coord_target = cv::Vec3i(target->World2Voxel(world_coord));
            target->AddBrickUpdateList(voxel_coord_target, &brick_update_hashset);
        }  // end if
    }  // end for
    cout << "update list size: " << brick_update_hashset.size() << endl;

    struct TSDFTargetVoxelUpdater
    {
        TSDFTargetVoxelUpdater(const TSDFHashing& tsdf_origin, const TSDFHashing& vtsdf_target)
            : tsdf_origin(tsdf_origin), tsdf_target(vtsdf_target){}
        bool operator () (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
        {
            // for every voxel in its new location, retrive its value in the original tsdf
            cv::Vec3f world_coord = tsdf_target.Voxel2World(cv::Vec3f(cur_voxel_coord));
            float cur_d;
            float cur_w;
            cv::Vec3b cur_color;
            if (!tsdf_origin.RetriveDataFromWorldCoord(utility::CvVectorToEigenVector3(world_coord), &cur_d, &cur_w, &cur_color)) return false;
            *d = cur_d;
            *w = cur_w;
            *color = cur_color;
            return true;
        }
        const TSDFHashing& tsdf_origin;
        const TSDFHashing& tsdf_target;
    };
    TSDFTargetVoxelUpdater updater(tsdf_origin, *target);
    target->UpdateBricksInQueue(brick_update_hashset, updater);
    cout << "finished merging TSDF. " << endl;
    return true;
}


bool cpu_tsdf::ScaleTSDFPart(cpu_tsdf::TSDFHashing *tsdf_volume, const Eigen::Affine3f &affine, const float scale)
{
    cpu_tsdf::OrientedBoundingBox obb;
    cpu_tsdf::AffineToOrientedBB(affine, &obb);
    return ScaleTSDFPart(tsdf_volume, obb, scale);
}


bool cpu_tsdf::ScaleTSDFPart(cpu_tsdf::TSDFHashing *tsdf_volume, const cpu_tsdf::OrientedBoundingBox &obb, const float scale)
{
    PointInOrientedBox pred(obb);
    using namespace std;
    const float original_voxel_length = tsdf_volume->voxel_length();
    float max_dist_pos, max_dist_neg;
    tsdf_volume->getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    Eigen::Vector3f voxel_world_offset = tsdf_volume->getVoxelOriginInWorldCoord();
    //sliced_tsdf->Init(original_voxel_length, voxel_world_offset, max_dist_pos, max_dist_neg);
    cout << "Begin filtering TSDF. " << endl;
    cout << "voxel length: \n" << original_voxel_length << endl;
    cout << "aabbmin: \n" << voxel_world_offset << endl;
    cout << "max, min trunc dist: " << max_dist_pos << "; " << max_dist_neg << endl;
    for (TSDFHashing::iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        int vsemantic_label;
        VoxelData::VoxelState st;
        citr->RetriveData(&d, &w, &color, &st, &vsemantic_label);
        if (w > 0 && pred(tsdf_volume, cur_voxel_coord, d, w, color, vsemantic_label))
        {
            //sliced_tsdf->AddObservation(cur_voxel_coord, d, w, color, vsemantic_label);
            //oper(cur_voxel_coord, d, w, color, vsemantic_label);
            citr->SetWeight(w * scale);
            //citr->ClearVoxel();
        }  // end if
    }  // end for
    //sliced_tsdf->DisplayInfo();
    cout << "End filtering TSDF. " << endl;
    return true;
    //return !sliced_tsdf->Empty();
}


bool cpu_tsdf::ScaleTSDFParts(cpu_tsdf::TSDFHashing *tsdf_volume, const std::vector<cpu_tsdf::OrientedBoundingBox> &obb, const float scale)
{
    for (int i = 0; i < obb.size(); ++i)
    {
        ScaleTSDFPart(tsdf_volume, obb[i], scale);
    }
}


bool cpu_tsdf::ScaleTSDFParts(cpu_tsdf::TSDFHashing *tsdf_volume, const std::vector<Eigen::Affine3f> &trans, const float scale)
{
    for (int i = 0; i < trans.size(); ++i)
    {
        ScaleTSDFPart(tsdf_volume, trans[i], scale);
    }
}

bool cpu_tsdf::ExtractSampleFromOBB(
        const cpu_tsdf::TSDFHashing &scene_tsdf,
        const tsdf_utility::OrientedBoundingBox &obb,
        const Eigen::Vector3i &sample_size,
        const float min_nonempty_weight,
        Eigen::SparseVector<float> *sample,
        Eigen::SparseVector<float> *weight)
{
    using namespace std;
    const Eigen::Matrix3f sample_oriented_deltas = obb.SamplingOrientedDeltas(sample_size);
    const int feat_dim = sample_size.prod();
    sample->resize(feat_dim);
    weight->resize(feat_dim);
    Eigen::Vector3f offset = obb.Offset();
    for (int x = 0; x < sample_size[0]; ++x) {
        for (int y = 0; y < sample_size[1]; ++y) {
            for (int z = 0; z < sample_size[2]; ++z) {
                Eigen::Vector3f sampling_pt = offset + (sample_oriented_deltas * Eigen::Vector3f(x, y, z));
                float cur_d = 0;
                float cur_w = 0;
                if(scene_tsdf.RetriveDataFromWorldCoord(sampling_pt, &cur_d, &cur_w) && cur_w > min_nonempty_weight)
                {
                    int current_index = z + (y + x * sample_size[1]) * sample_size[2];
                    sample->coeffRef(current_index) = cur_d;
                    weight->coeffRef(current_index) = cur_w;
                }
            }
        }
    }
    return true;
}


bool cpu_tsdf::ExtractSamplesFromOBBs(const cpu_tsdf::TSDFHashing &scene_tsdf, const std::vector<tsdf_utility::OrientedBoundingBox> &obbs, const Eigen::Vector3i &sample_size, const float min_nonempty_weight, Eigen::SparseMatrix<float, Eigen::ColMajor> *samples, Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
{
    const int sample_num = obbs.size();
    const int feature_dim = sample_size.prod();
    samples->resize(feature_dim, sample_num);
    samples->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
    weights->resize(feature_dim, sample_num);
    weights->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
    for (int i = 0; i < sample_num; ++i)
    {
        Eigen::SparseVector<float> sample(feature_dim);
        Eigen::SparseVector<float> weight(feature_dim);
        ExtractSampleFromOBB(scene_tsdf, obbs[i], sample_size, min_nonempty_weight, &sample, &weight);
        samples->col(i) = sample;
        weights->col(i) = weight;
    }
    return true;
}
