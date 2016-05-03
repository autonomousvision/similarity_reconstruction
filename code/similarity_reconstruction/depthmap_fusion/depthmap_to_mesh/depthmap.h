#pragma once
#include <vector>
#include <cstddef>

namespace cv{
class Mat;
}
namespace pcl{
struct Vertices;
template<typename T>
class PointCloud;
struct PointXYZ;
struct PointXYZRGB;
struct PolygonMesh;
}
class RectifiedCameraPair;

namespace cpu_tsdf
{
    void DepthMapToPointCloudInVoxelCoord(const cv::Mat& depthmap,
            const RectifiedCameraPair& cam_info,
            pcl::PointCloud<pcl::PointXYZ>* voxel_coord_cloud);
    void DepthMapToPointCloudInWorldCoord(
            const cv::Mat& depthmap,
            const cv::Mat &image,
            const RectifiedCameraPair& cam_info,
            pcl::PointCloud<pcl::PointXYZRGB> *voxel_coord_cloud);
    void DepthMapTriangulate(const cv::Mat& depthmap,
            const cv::Mat& image,
            const RectifiedCameraPair& cam_info, const float dd_factor,
            pcl::PointCloud<pcl::PointXYZRGB>* point_cloud_world_coord,
            std::vector<pcl::Vertices>* triangles,
            std::vector<std::vector<int>>* pixel_triangles_list
            );

    void DepthMapTriangulateOld(const cv::Mat& depthmap,
            const cv::Mat& image,
            const RectifiedCameraPair& cam_info,
                             double dd_factor,
                             float check_edge_ratio,
                             float check_view_angle,
            pcl::PointCloud<pcl::PointXYZRGB> *point_cloud_voxel_coord,
            std::vector<pcl::Vertices>* triangles,
            std::vector<int>* triangle_indices = NULL,
            std::vector<char>* triangle_type = NULL);
    void DepthMapTriangulate(const cv::Mat& depthmap,
            const cv::Mat &image,
            const RectifiedCameraPair& cam_info,
            double dd_factor, double check_edge_ratio, double check_view_angle,
            pcl::PolygonMesh* mesh);
}
