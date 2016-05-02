/*
 * For hole filling, not used
*/
#pragma once
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/PolygonMesh.h>
#include <opencv2/opencv.hpp>

#include "common/fisheye_camera/RectifiedCameraPair.h"

namespace cv{
class Mat;
}
//namespace pcl{
//struct Vertices;
//template<typename T>
//class PointCloud;
//struct PointXYZ;
//struct PolygonMesh;
//}

//class RectifiedCameraPair;

namespace cpu_tsdf
{
class DepthMapTriangleMesh
{
  public:
      DepthMapTriangleMesh(const cv::Mat& depthmap, const RectifiedCameraPair& cam_info, double dd_factor);
      bool SearchHittingTriangle(const cv::Vec3d voxel_point, const cv::Mat& depthmap, const RectifiedCameraPair& cam_info, cv::Vec3d* tri_pts);

  private:
      enum TriangleIndextype {
          INVALID = -1,
          LEFT_CUT = 0,
          RIGHT_CUT = 1
      };  // LEFT_CUT: top-left to bottom right. RIGHT_CUT: the other way
      pcl::PointCloud<pcl::PointXYZ> point_cloud_;
      std::vector<pcl::Vertices> triangles_;
      //std::vector<pcl::Vertices> polygons_;
      std::vector<int> triangle_indices_;  // -1 invalid value
      std::vector<char> triangle_type_;
      // 0--1
      // |  |
      // 2--3
      static const int TriangleIndices[4][3];
};
}
