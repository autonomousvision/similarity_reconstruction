/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "marching_cubes_tsdf_hash.h"
#include <Eigen/Eigen>
#include "../tsdf_representation/tsdf_hash.h"
#include "common/utilities/pcl_utility.h"

void
cpu_tsdf::MarchingCubesTSDFHashing::setInputTSDF (cpu_tsdf::TSDFHashing::ConstPtr tsdf_volume)
{
  tsdf_volume_ = tsdf_volume;
  // Set the grid resolution so it mimics the tsdf's
  Eigen::Vector3i res(1, 1, 1);
  setGridResolution (res.x(), res.y(), res.z());
  Eigen::Vector3f voxel_origin_in_world, voxel_unit_in_world;
  voxel_origin_in_world = tsdf_volume_->getVoxelOriginInWorldCoord();
  tsdf_volume_->getVoxelUnit3DPointInWorldCoord(voxel_unit_in_world);

  pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointXYZ min_p, max_p;
  min_p.x = voxel_origin_in_world(0); min_p.y = voxel_origin_in_world(1); min_p.z = voxel_origin_in_world(2); 
  max_p.x = voxel_unit_in_world(0); max_p.y = voxel_unit_in_world(1); max_p.z = voxel_unit_in_world(2); 
  dummy_cloud->points.push_back (min_p);
  dummy_cloud->points.push_back (max_p);
  setInputCloud (dummy_cloud);

  this->min_p_(0) = voxel_origin_in_world(0); this->min_p_(1) = voxel_origin_in_world(1); this->min_p_(2) = voxel_origin_in_world(2); 
  this->max_p_(0) = voxel_unit_in_world(0); this->max_p_(1) = voxel_unit_in_world(1); this->max_p_(2) = voxel_unit_in_world(2); 
 
  // No extending
  setPercentageExtendGrid (0);
  setIsoLevel (0);
}

void
cpu_tsdf::MarchingCubesTSDFHashing::voxelizeData ()
{
  // Do nothing, extending getGridValue instead to save memory
}

float
cpu_tsdf::MarchingCubesTSDFHashing::getGridValue (Eigen::Vector3i pos)
{
    float d, w;
    cv::Vec3b color;
    if(!tsdf_volume_->RetriveData(pos, &d, &w, &color) || w <= w_min_) return (std::numeric_limits<float>::quiet_NaN ());
    return d;
}

void
cpu_tsdf::MarchingCubesTSDFHashing::performReconstruction (pcl::PolygonMesh &output)
{
  // getBoundingBox ();
  voxelizeData ();
  // Run the actual marching cubes algorithm, store it into a point cloud,
  // and copy the point cloud + connectivity into output
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  //{
  //  reconstructVoxel (cloud);
  //  //pcl::transformPointCloud (cloud, cloud, tsdf_volume_->getGlobalTransform ());
  //  pcl::toPCLPointCloud2 (cloud, output.cloud);
  //}
  {
      reconstructVoxelWithColor (cloud);
      pcl::toPCLPointCloud2 (cloud, output.cloud);
  }

  output.polygons.resize (cloud.size () / 3);
  for (size_t i = 0; i < output.polygons.size (); ++i)
  {
    pcl::Vertices v;
    v.vertices.resize (3);
    for (int j = 0; j < 3; ++j)
    {
      v.vertices[j] = static_cast<int> (i) * 3 + j;
    }
    output.polygons[i] = v;
  }
}

bool
cpu_tsdf::MarchingCubesTSDFHashing::getValidNeighborList1D (std::vector<float> &leaf,
                                                     Eigen::Vector3i &index3d)
{
  leaf = std::vector<float> (8, 0.0f);

  leaf[0] = getGridValue (index3d);
  if (pcl_isnan (leaf[0]))
    return (false);
  leaf[1] = getGridValue (index3d + Eigen::Vector3i (1, 0, 0));
  if (pcl_isnan (leaf[1]))
    return (false);
  leaf[2] = getGridValue (index3d + Eigen::Vector3i (1, 0, 1));
  if (pcl_isnan (leaf[2]))
    return (false);
  leaf[3] = getGridValue (index3d + Eigen::Vector3i (0, 0, 1));
  if (pcl_isnan (leaf[3]))
    return (false);
  leaf[4] = getGridValue (index3d + Eigen::Vector3i (0, 1, 0));
  if (pcl_isnan (leaf[4]))
    return (false);
  leaf[5] = getGridValue (index3d + Eigen::Vector3i (1, 1, 0));
  if (pcl_isnan (leaf[5]))
    return (false);
  leaf[6] = getGridValue (index3d + Eigen::Vector3i (1, 1, 1));
  if (pcl_isnan (leaf[6]))
    return (false);
  leaf[7] = getGridValue (index3d + Eigen::Vector3i (0, 1, 1));
  if (pcl_isnan (leaf[7]))
    return (false);
  return (true);
}

void
cpu_tsdf::MarchingCubesTSDFHashing::reconstructVoxel (pcl::PointCloud<pcl::PointXYZ> &output)
{
  fprintf(stderr, "begin reconstruction..");
  for(TSDFHashing::const_iterator itr = tsdf_volume_->begin(); itr != tsdf_volume_->end(); ++itr)
  {
      cv::Vec3i cur_voxel_coord = itr.VoxelCoord();
      float d, w;
      cv::Vec3b color;
      itr->RetriveData(&d, &w, &color);
      if (w > w_min_)
      {
          Eigen::Vector3i idx;
          idx(0) = cur_voxel_coord[0];
          idx(1) = cur_voxel_coord[1];
          idx(2) = cur_voxel_coord[2];
          std::vector<float> leaf_node;
          if (!getValidNeighborList1D (leaf_node, idx))
              continue;
          createSurface (leaf_node, idx, output);
      }  // end if
  }  // end for
  fprintf(stderr, "finished\n");
}

bool
cpu_tsdf::MarchingCubesTSDFHashing::getValidNeighborList1DWithColor (
        const Eigen::Vector3i &index3d,
        std::vector<float> &leaf,
        std::vector<cv::Vec3b> &color_leaf)
{
  leaf.resize(8) ;
  color_leaf.resize(8);
  float w;
  if ( getGridValueWithColor(index3d, &leaf[0], &w, &color_leaf[0]) &&
       getGridValueWithColor(index3d + Eigen::Vector3i (1, 0, 0), &leaf[1], &w, &color_leaf[1]) &&
       getGridValueWithColor(index3d + Eigen::Vector3i (1, 0, 1), &leaf[2], &w, &color_leaf[2]) &&
       getGridValueWithColor(index3d + Eigen::Vector3i (0, 0, 1), &leaf[3], &w, &color_leaf[3]) &&
       getGridValueWithColor(index3d + Eigen::Vector3i (0, 1, 0), &leaf[4], &w, &color_leaf[4]) &&
       getGridValueWithColor(index3d + Eigen::Vector3i (1, 1, 0), &leaf[5], &w, &color_leaf[5]) &&
       getGridValueWithColor(index3d + Eigen::Vector3i (1, 1, 1), &leaf[6], &w, &color_leaf[6]) &&
       getGridValueWithColor(index3d + Eigen::Vector3i (0, 1, 1), &leaf[7], &w, &color_leaf[7]) 
     )
  {
      return true;
  }
  return false;
}

template <typename PointNT> void
cpu_tsdf::MarchingCubesTSDFHashing::createSurfaceWithColor (
        std::vector<float> &leaf_node,
        const std::vector<cv::Vec3b>& leaf_color,
        Eigen::Vector3i &index_3d,
        pcl::PointCloud<PointNT> &cloud)
{
  using pcl::edgeTable;
  using pcl::triTable;
  int cubeindex = 0;
  Eigen::Vector3f vertex_list[12];
  cv::Vec3b color_list[12];

  if (leaf_node[0] < iso_level_) cubeindex |= 1;
  if (leaf_node[1] < iso_level_) cubeindex |= 2;
  if (leaf_node[2] < iso_level_) cubeindex |= 4;
  if (leaf_node[3] < iso_level_) cubeindex |= 8;
  if (leaf_node[4] < iso_level_) cubeindex |= 16;
  if (leaf_node[5] < iso_level_) cubeindex |= 32;
  if (leaf_node[6] < iso_level_) cubeindex |= 64;
  if (leaf_node[7] < iso_level_) cubeindex |= 128;

  // Cube is entirely in/out of the surface
  if (edgeTable[cubeindex] == 0)
    return;

  //Eigen::Vector4f index_3df (index_3d[0], index_3d[1], index_3d[2], 0.0f);
  Eigen::Vector3f center;// TODO coeff wise product = min_p_ + Eigen::Vector4f (1.0f/res_x_, 1.0f/res_y_, 1.0f/res_z_) * index_3df * (max_p_ - min_p_);
  center[0] = min_p_[0] + (max_p_[0] - min_p_[0]) * float (index_3d[0]) / float (res_x_);
  center[1] = min_p_[1] + (max_p_[1] - min_p_[1]) * float (index_3d[1]) / float (res_y_);
  center[2] = min_p_[2] + (max_p_[2] - min_p_[2]) * float (index_3d[2]) / float (res_z_);

  std::vector<Eigen::Vector3f> p;
  p.resize (8);
  for (int i = 0; i < 8; ++i)
  {
    Eigen::Vector3f point = center;
    if(i & 0x4)
      point[1] = static_cast<float> (center[1] + (max_p_[1] - min_p_[1]) / float (res_y_));

    if(i & 0x2)
      point[2] = static_cast<float> (center[2] + (max_p_[2] - min_p_[2]) / float (res_z_));

    if((i & 0x1) ^ ((i >> 1) & 0x1))
      point[0] = static_cast<float> (center[0] + (max_p_[0] - min_p_[0]) / float (res_x_));

    p[i] = point;
  }


  // Find the vertices where the surface intersects the cube
  if (edgeTable[cubeindex] & 1)
  {
    interpolateEdge (p[0], p[1], leaf_node[0], leaf_node[1], vertex_list[0]);
    interpolateColor(leaf_color[0], leaf_color[1], leaf_node[0], leaf_node[1], color_list[0]);
  }
  if (edgeTable[cubeindex] & 2)
  {
    interpolateEdge (p[1], p[2], leaf_node[1], leaf_node[2], vertex_list[1]);
    interpolateColor(leaf_color[1], leaf_color[2], leaf_node[1], leaf_node[2], color_list[1]);
  }
  if (edgeTable[cubeindex] & 4)
  {
    interpolateEdge (p[2], p[3], leaf_node[2], leaf_node[3], vertex_list[2]);
    interpolateColor(leaf_color[2], leaf_color[3], leaf_node[2], leaf_node[3], color_list[2]);
  }
  if (edgeTable[cubeindex] & 8)
  {
    interpolateEdge (p[3], p[0], leaf_node[3], leaf_node[0], vertex_list[3]);
    interpolateColor(leaf_color[3], leaf_color[0], leaf_node[3], leaf_node[0], color_list[3]);
  }
  if (edgeTable[cubeindex] & 16)
  {
    interpolateEdge (p[4], p[5], leaf_node[4], leaf_node[5], vertex_list[4]);
    interpolateColor(leaf_color[4], leaf_color[5], leaf_node[4], leaf_node[5], color_list[4]);
  }
  if (edgeTable[cubeindex] & 32)
  {
    interpolateEdge (p[5], p[6], leaf_node[5], leaf_node[6], vertex_list[5]);
    interpolateColor(leaf_color[5], leaf_color[6], leaf_node[5], leaf_node[6], color_list[5]);
  }
  if (edgeTable[cubeindex] & 64)
  {
    interpolateEdge (p[6], p[7], leaf_node[6], leaf_node[7], vertex_list[6]);
    interpolateColor(leaf_color[6], leaf_color[7], leaf_node[6], leaf_node[7], color_list[6]);
  }
  if (edgeTable[cubeindex] & 128)
  {
    interpolateEdge (p[7], p[4], leaf_node[7], leaf_node[4], vertex_list[7]);
    interpolateColor(leaf_color[7], leaf_color[4], leaf_node[7], leaf_node[4], color_list[7]);
  }
  if (edgeTable[cubeindex] & 256)
  {
    interpolateEdge (p[0], p[4], leaf_node[0], leaf_node[4], vertex_list[8]);
    interpolateColor(leaf_color[0], leaf_color[4], leaf_node[0], leaf_node[4], color_list[8]);
  }
  if (edgeTable[cubeindex] & 512)
  {
    interpolateEdge (p[1], p[5], leaf_node[1], leaf_node[5], vertex_list[9]);
    interpolateColor(leaf_color[1], leaf_color[5], leaf_node[1], leaf_node[5], color_list[9]);
  }
  if (edgeTable[cubeindex] & 1024)
  {
    interpolateEdge (p[2], p[6], leaf_node[2], leaf_node[6], vertex_list[10]);
    interpolateColor(leaf_color[2], leaf_color[6], leaf_node[2], leaf_node[6], color_list[10]);
  }
  if (edgeTable[cubeindex] & 2048)
  {
    interpolateEdge (p[3], p[7], leaf_node[3], leaf_node[7], vertex_list[11]);
    interpolateColor(leaf_color[3], leaf_color[7], leaf_node[3], leaf_node[7], color_list[11]);
  }

  // Create the triangle
  for (int i = 0; triTable[cubeindex][i] != -1; i+=3)
  {
    PointNT p1,p2,p3;
    p1.x = vertex_list[triTable[cubeindex][i  ]][0];
    p1.y = vertex_list[triTable[cubeindex][i  ]][1];
    p1.z = vertex_list[triTable[cubeindex][i  ]][2];
    p1.r = color_list[triTable[cubeindex][i  ]][0];
    p1.g = color_list[triTable[cubeindex][i  ]][1];
    p1.b = color_list[triTable[cubeindex][i  ]][2];
    cloud.push_back (p1);
    p2.x = vertex_list[triTable[cubeindex][i+1]][0];
    p2.y = vertex_list[triTable[cubeindex][i+1]][1];
    p2.z = vertex_list[triTable[cubeindex][i+1]][2];
    p2.r = color_list[triTable[cubeindex][i+1]][0];
    p2.g = color_list[triTable[cubeindex][i+1]][1];
    p2.b = color_list[triTable[cubeindex][i+1]][2];
    cloud.push_back (p2);
    p3.x = vertex_list[triTable[cubeindex][i+2]][0];
    p3.y = vertex_list[triTable[cubeindex][i+2]][1];
    p3.z = vertex_list[triTable[cubeindex][i+2]][2];
    p3.r = color_list[triTable[cubeindex][i+2]][0];
    p3.g = color_list[triTable[cubeindex][i+2]][1];
    p3.b = color_list[triTable[cubeindex][i+2]][2];
    cloud.push_back (p3);
  }
}

void
cpu_tsdf::MarchingCubesTSDFHashing::reconstructVoxelWithColor (pcl::PointCloud<pcl::PointXYZRGB> &output)
{
  fprintf(stderr, "begin color reconstruction..");
  std::vector<float> leaf_node(8);
  std::vector<cv::Vec3b> leaf_color(8);
  for(TSDFHashing::const_iterator citr = tsdf_volume_->begin(); citr != tsdf_volume_->end(); ++citr)
  {
      cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
      float d, w;
      cv::Vec3b color;
      citr->RetriveData(&d, &w, &color);
      if (w > w_min_)
      {
          Eigen::Vector3i idx(cur_voxel_coord[0], cur_voxel_coord[1], cur_voxel_coord[2]);
          if (!getValidNeighborList1DWithColor (idx, leaf_node, leaf_color))
              continue;
          createSurfaceWithColor (leaf_node, leaf_color, idx, output);
      }  // end if
  }  // end for
  fprintf(stderr, "finished\n");
}


pcl::PolygonMesh::Ptr cpu_tsdf::TSDFToPolygonMesh(cpu_tsdf::TSDFHashing::ConstPtr tsdf_model, float mesh_min_weight, float flatten_dist)
{
    cpu_tsdf::MarchingCubesTSDFHashing mc;
    mc.setMinWeight(mesh_min_weight);
    mc.setInputTSDF (tsdf_model);
    pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh);
    mc.reconstruct (*mesh);
    float default_dist;
    if (flatten_dist <= 0)
    {
        default_dist = tsdf_model->voxel_length()/50.0;
    }
    else
    {
        default_dist = flatten_dist;
    }
    // default_dist = 0.00005;
    utility::flattenVertices(*mesh, default_dist);
    utility::flattenVertices(*mesh, default_dist);
    utility::flattenVertices(*mesh, default_dist);
    return mesh;
}
