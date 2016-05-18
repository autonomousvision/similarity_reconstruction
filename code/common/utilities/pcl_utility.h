/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "common_utility.h"

namespace utility {
void
flattenVerticesNoRGB (pcl::PolygonMesh &mesh, float min_dist = 0.00005);

void ClearMeshWithVertKeepArray(pcl::PolygonMesh &mesh, const std::vector<bool>& kept_verts);

void
  flattenVertices(pcl::PolygonMesh &mesh, float min_dist = 0.00005);

  template<typename T>
  void ComputeBoundingbox(const pcl::PointCloud<T>& cloud, Eigen::Vector3f* min_pt, Eigen::Vector3f* max_pt)
  {
    if (cloud.empty()) return;
    (*min_pt).setConstant(FLT_MAX);
    (*max_pt).setConstant(-FLT_MAX);
    for (size_t i = 0; i < cloud.size (); i++)
      {
        const Eigen::Vector3f current_pt = cloud.at(i).getVector3fMap();
        *min_pt = min_vec3(*min_pt, current_pt);
        *max_pt = max_vec3(*max_pt, current_pt);
      }
  }

  // Converting mesh to point cloud with normals
  pcl::PointCloud<pcl::PointNormal>::Ptr
	  meshToFaceCloud(const pcl::PolygonMesh &mesh);

  void cleanupMesh(pcl::PolygonMesh &mesh, float face_dist=0.02, int min_neighbors=5);

  //void SaveTSDFMesh(cpu_tsdf::TSDFHashing::Ptr tsdf_model, float mesh_min_weight, const string& output_filename, const bool save_ascii);

  void Write3DPointToFilePCL(const std::string& fname, const std::vector<cv::Vec3d>& points3d, const std::vector<cv::Vec3b>* colors = NULL);

  template<typename T>
  cv::Vec3f PCLPoint2CvVec(const T& pclpt)
  {
      return cv::Vec3f(pclpt.x, pclpt.y, pclpt.z);
  }

}

