/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "depthmap.h"
#include <vector>
#include <climits>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <glog/logging.h>
#include "common/fisheye_camera/RectifiedCameraPair.h"
using namespace std;


inline cv::Vec3f PCLPoint2CvVec(const pcl::PointXYZRGB& point)
{
    return cv::Vec3f(point.x, point.y, point.z);
}

inline bool CheckSlantedTriangleEdgeRatio(const cv::Vec3f* vertices, const float ratio)
{
    float len1 = cv::norm(vertices[0], vertices[1]);
    float len2 = cv::norm(vertices[1], vertices[2]);
    float len3 = cv::norm(vertices[2], vertices[0]);

    float min_len = std::min(len1, std::min(len2, len3));
    float max_len = std::max(len1, std::max(len2, len3));
    CHECK_GT(min_len, 0);
    return (max_len/min_len) > ratio;
}

inline bool CheckSlantedTriangleEdgeRatio(const pcl::PointXYZRGB* pcl_vertices, const float ratio)
{
    cv::Vec3f vertices[3];
    vertices[0] = PCLPoint2CvVec(pcl_vertices[0]);
    vertices[1] = PCLPoint2CvVec(pcl_vertices[1]);
    vertices[2] = PCLPoint2CvVec(pcl_vertices[2]);
    return CheckSlantedTriangleEdgeRatio(vertices, ratio);
}

inline bool CheckSlantedTriangleNormal(const cv::Vec3f* vertices, const cv::Vec3f cam_center, const float cos_thresh)
{
    cv::Vec3f tri_normal = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
    cv::Vec3f view_ray = cam_center - vertices[0];
    cv::Vec3f tri_normal_n = cv::normalize(tri_normal);
    cv::Vec3f view_ray_n = cv::normalize(view_ray);
    return fabs(tri_normal_n.dot(view_ray_n)) < cos_thresh;
}

inline bool CheckSlantedTriangleNormal(const pcl::PointXYZRGB* pcl_vertices, const cv::Vec3f cam_center, const float cos_thresh)
{
    cv::Vec3f vertices[3];
    vertices[0] = PCLPoint2CvVec(pcl_vertices[0]);
    vertices[1] = PCLPoint2CvVec(pcl_vertices[1]);
    vertices[2] = PCLPoint2CvVec(pcl_vertices[2]);
    return CheckSlantedTriangleNormal(vertices, cam_center, cos_thresh);
}

inline bool IsDepthDiscontinous(double* spherical_err, double* depths, double dd_factor, int i1, int i2)
{
    /* Find index that corresponds to smaller depth. */
    int i_min = i1;
    int i_max = i2;
    if (depths[i2] < depths[i1])
        std::swap(i_min, i_max);

    /* Check if indices are a diagonal. */
    if (i1 + i2 == 3)
        dd_factor *= M_SQRT2;

    /* Check for depth discontinuity. */
    if (depths[i_max] - depths[i_min] > spherical_err[i_min] * dd_factor)
        return true;

    return false;
}

template<typename Pred>
void RemoveSlantedTriangles(
        const pcl::PointCloud<pcl::PointXYZRGB>& point_cloud_world_coord,
        const std::vector<std::vector<int>>& pixel_triangles_list,
        std::vector<pcl::Vertices>* triangles,
        Pred pred)
{
    vector<bool> reserve_triangles(triangles->size(), true);
    // after the function the invalid triangles are set to empty vertices
    // pixel_triangles_list is not changed, and may correspond to invalid triangles
    for (int i = 0; i < triangles->size(); ++i)
    {
        //cout << "i " << i << endl;
        pcl::Vertices& cur_triangle = (*triangles)[i];
        /* invalid triangle */
        if (cur_triangle.vertices.empty())
        {
            continue;
        }
        pcl::PointXYZRGB vertices[3];
        vertices[0] = point_cloud_world_coord.at(cur_triangle.vertices[0]);
        vertices[1] = point_cloud_world_coord.at(cur_triangle.vertices[1]);
        vertices[2] = point_cloud_world_coord.at(cur_triangle.vertices[2]);

        // if not slanted, continue
        if (!pred(vertices)) continue;

        // cur_triangle.vertices.clear();
        // if slanted, remove itself and neighboring triangles
        for (int j = 0; j < 3; ++j)
        {
            using namespace std;
            //cout << "j" << j << endl;
            //cout << cur_triangle.vertices[j] << endl;
            const std::vector<int> & triangle_list = pixel_triangles_list[cur_triangle.vertices[j]];
            for (int k = 0; k < triangle_list.size(); ++k)
            {
                int neighbor_tri_idx = triangle_list[k];
                // CHECK_EQ((*triangles)[neighbor_tri_idx].vertices.size(), 3);
                // (*triangles)[neighbor_tri_idx].vertices.clear();
                reserve_triangles[neighbor_tri_idx] = false;
            }
        }
    }
    for (int i = 0 ; i < triangles->size(); ++i)
    {
        if (!reserve_triangles[i])
        {
            (*triangles)[i].vertices.clear();
        }
    }
}

inline pcl::Vertices AddTriangle(const int* indices)
{
    pcl::Vertices vert;
    vert.vertices.resize(3);
    vert.vertices[0] = indices[0];
    vert.vertices[1] = indices[1];
    vert.vertices[2] = indices[2];
    return vert;
}

inline pcl::Vertices AddTriangle(int i1, int i2, int i3)
{
    pcl::Vertices vert;
    vert.vertices.resize(3);
    vert.vertices[0] = i1;
    vert.vertices[1] = i2;
    vert.vertices[2] = i3;
    return vert;
}

void cpu_tsdf::DepthMapToPointCloudInVoxelCoord(const cv::Mat& depthmap,
        const RectifiedCameraPair& cam_info,
        pcl::PointCloud<pcl::PointXYZ>* voxel_coord_cloud)
{
    assert(voxel_coord_cloud);
    voxel_coord_cloud->width = depthmap.cols;
    voxel_coord_cloud->height = depthmap.rows;
    voxel_coord_cloud->resize(depthmap.cols * depthmap.rows);
    voxel_coord_cloud->is_dense = false;
    const pcl::PointXYZ nan_point(NAN, NAN, NAN);
    for(int y = 0; y < depthmap.rows; ++y)
        for(int x = 0; x < depthmap.cols; ++x)
        {
            unsigned short quant_depth = depthmap.at<unsigned short>(y, x);
            pcl::PointXYZ& p = (*voxel_coord_cloud)(x, y);
            if (quant_depth > 0)
            {
                //cv::Vec3d world_point = cam_info.RectifiedRefImagePointToVoxel3DPoint(x, y, quant_depth);
                cv::Vec3d world_point = cam_info.RectifiedImagePointToVoxel3DPoint(x, y, quant_depth);
                p.x = world_point[0];
                p.y = world_point[1];
                p.z = world_point[2];
            }
            else
            {
                p = nan_point;   
            }
        }
}

void cpu_tsdf::DepthMapToPointCloudInWorldCoord(
        const cv::Mat& depthmap,
        const cv::Mat& image,
        const RectifiedCameraPair& cam_info,
        pcl::PointCloud<pcl::PointXYZRGB>* voxel_coord_cloud)
{
    assert(voxel_coord_cloud);
    voxel_coord_cloud->width = depthmap.cols;
    voxel_coord_cloud->height = depthmap.rows;
    voxel_coord_cloud->resize(depthmap.cols * depthmap.rows);
    voxel_coord_cloud->is_dense = false;
    pcl::PointXYZRGB nan_point;
    nan_point.x = nan_point.y = nan_point.z = NAN;
    nan_point.r = nan_point.g = nan_point.b = 0;
    for(int y = 0; y < depthmap.rows; ++y)
        for(int x = 0; x < depthmap.cols; ++x)
        {
            unsigned short quant_depth = depthmap.at<unsigned short>(y, x);
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            pcl::PointXYZRGB& p = (*voxel_coord_cloud)(x, y);
            if (quant_depth > 0)
            {
                //cv::Vec3d world_point = cam_info.RectifiedRefImagePointToVoxel3DPoint(x, y, quant_depth);
                cv::Vec3d voxel_point = cam_info.RectifiedImagePointToVoxel3DPoint(x, y, quant_depth);
                cv::Vec3d world_point = cam_info.Voxel3DPointToWorldCoord(voxel_point);
                p.x = world_point[0];
                p.y = world_point[1];
                p.z = world_point[2];
                p.r = color[2];
                p.g = color[1];
                p.b = color[0];
            }
            else
            {
                p = nan_point;
            }
        }
}

void cpu_tsdf::DepthMapTriangulate(const cv::Mat& depthmap,
        const cv::Mat& image,
        const RectifiedCameraPair& cam_info,
        const float dd_factor,
        pcl::PointCloud<pcl::PointXYZRGB>* point_cloud_world_coord,
        std::vector<pcl::Vertices>* triangles,
        std::vector<std::vector<int>>* pixel_triangles_list
        )
{
    triangles->clear();
    if (depthmap.cols == 0 || depthmap.rows == 0) return;

    // DepthMapToPointCloudInVoxelCoord(depthmap, cam_info, point_cloud_voxel_coord);
    DepthMapToPointCloudInWorldCoord(depthmap, image, cam_info, point_cloud_world_coord);

    const int w = depthmap.cols;
    const int h = depthmap.rows;

    pixel_triangles_list->resize(w * h);

    /* Iterate over 2x2-blocks in the depthmap and create triangles. */
    for (int y = 0; y < h - 1; ++y)
    {
        for (int x = 0; x < w - 1; ++x)
        {
            /* Cache the four depth values. */
            int point_index[4] = { y * w + x, y * w + x + 1, (y+1) * w + x, (y+1) * w + x + 1 };
            double depths[4] = { depthmap.at<unsigned short>(y, x) * cam_info.DepthImageScaling(),
                depthmap.at<unsigned short>(y, x+1) * cam_info.DepthImageScaling(),
                depthmap.at<unsigned short>(y+1, x) * cam_info.DepthImageScaling(),
                depthmap.at<unsigned short>(y+1, x+1) * cam_info.DepthImageScaling() };

            /* Create a mask representation of the available depth values. */
            int mask = 0;
            int pixels = 0;
            for (int j = 0; j < 4; ++j)
                if (depths[j] > 0.0)
                {
                    mask |= 1 << j;
                    pixels += 1;
                }

            /* At least three valid depth values are required. */
            if (pixels < 3)
                continue;

            /* Possible triangles, vertex indices relative to 2x2 block. */
            // 0--1
            // |  |
            // 2--3
            const int tris[4][3] = {
                { 0, 2, 1 }, { 0, 3, 1 }, { 0, 2, 3 }, { 1, 2, 3 }
            };

            /* Decide which triangles to issue. */
            int tri[2] = { 0, 0 };

            switch (mask)
            {
                case 7: tri[0] = 1; break;
                case 11: tri[0] = 2; break;
                case 13: tri[0] = 3; break;
                case 14: tri[0] = 4; break;
                case 15:
                         {
                             /* Choose the triangulation with smaller diagonal. */
                             float ddiff1 = std::abs(depths[0] - depths[3]);
                             float ddiff2 = std::abs(depths[1] - depths[2]);
                             if (ddiff1 < ddiff2)
                             { tri[0] = 2; tri[1] = 3; }
                             else
                             { tri[0] = 1; tri[1] = 4; }
                             break;
                         }
                default: continue;
            }

            /* Omit depth discontinuity detection if dd_factor is zero. */
            if (dd_factor > 0.0)
            {
                /* Cache pixel footprints. */
                double spherical_err[4] = {0};
                for (int j = 0; j < 4; ++j)
                {
                    if (depths[j] == 0.0)
                        continue;
                    spherical_err[j] = cam_info.DepthErrorWithDisparity(x + (j%2), depths[j], M_PI/(float(depthmap.cols)));
                }

                /* Check for depth discontinuities. */
                for (int j = 0; j < 2 && tri[j] != 0; ++j)
                {
                    const int* tv = tris[tri[j] - 1];
                    if (IsDepthDiscontinous(spherical_err, depths, dd_factor, tv[0], tv[1])) tri[j] = 0;
                    if (IsDepthDiscontinous(spherical_err, depths, dd_factor, tv[1], tv[2])) tri[j] = 0;
                    if (IsDepthDiscontinous(spherical_err, depths, dd_factor, tv[2], tv[0])) tri[j] = 0;
                }
            }

//            if (check_edge_ratio > 0)
//            {
//                for (int j = 0; j < 2; ++j)
//                {
//                    if (tri[j] == 0) continue;
//                    const int* tv = tris[tri[j] - 1];
//                    cv::Vec3f vertices[3];
//                    vertices[0] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[0]]));
//                    vertices[1] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[1]]));
//                    vertices[2] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[2]]));
//                    if (CheckSlantedTriangleEdgeRatio(vertices, check_edge_ratio)) tri[j] = 0;
//                }
//            }

//            if (check_view_angle > 0)
//            {
//                for (int j = 0; j < 2; ++j)
//                {
//                    if (tri[j] == 0) continue;
//                    const int* tv = tris[tri[j] - 1];
//                    cv::Vec3f vertices[3];
//                    vertices[0] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[0]]));
//                    vertices[1] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[1]]));
//                    vertices[2] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[2]]));
//                    cv::Vec3f ref_cam_center = cam_info.RefCameraCenterInWorldCoord();
//                    if (CheckSlantedTriangleNormal(vertices, ref_cam_center, check_view_angle)) tri[j] = 0;
//                }
//            }

            /* Build triangles. */
            for (int j = 0; j < 2; ++j)
            {
                if (tri[j] == 0) continue;
                const int *pointer_to_indices = tris[tri[j] - 1];
                Eigen::Vector3i cur_pixels(
                            point_index[pointer_to_indices[0]],
                            point_index[pointer_to_indices[1]],
                            point_index[pointer_to_indices[2]]
                        );
                //cout << "cur_pixels: \n "  << cur_pixels << endl;
                triangles->push_back(AddTriangle(cur_pixels[0],
                            cur_pixels[1],
                            cur_pixels[2]));
                // back reference
                //cout << "pushed value: " << triangles->size() - 1 << endl;
                (*pixel_triangles_list)[cur_pixels[0]].push_back(triangles->size() - 1);
                (*pixel_triangles_list)[cur_pixels[1]].push_back(triangles->size() - 1);
                (*pixel_triangles_list)[cur_pixels[2]].push_back(triangles->size() - 1);
            }
        }
    }
    return;
}

void cpu_tsdf::DepthMapTriangulateOld(
        const cv::Mat& depthmap,
        const cv::Mat& image,
        const RectifiedCameraPair& cam_info,
        double dd_factor,
        float check_edge_ratio,
        float check_view_angle,
        pcl::PointCloud<pcl::PointXYZRGB>* point_cloud_world_coord,
        std::vector<pcl::Vertices>* triangles,
        std::vector<int>* triangle_indices,
        std::vector<char>* triangle_type)
{
    if (depthmap.cols == 0 || depthmap.rows == 0) return;

    // DepthMapToPointCloudInVoxelCoord(depthmap, cam_info, point_cloud_voxel_coord);
    DepthMapToPointCloudInWorldCoord(depthmap, image, cam_info, point_cloud_world_coord);

    const int w = depthmap.cols;
    const int h = depthmap.rows;

    if (triangle_indices && triangle_type)
    {
        // invalid value: -1
        triangle_indices->assign((w - 1) * (h - 1), -1);
        // invalid value: 0
        triangle_type->assign((w - 1) * (h - 1), char(0));
    }

    /* Iterate over 2x2-blocks in the depthmap and create triangles. */
    for (int y = 0; y < h - 1; ++y)
    {
        for (int x = 0; x < w - 1; ++x)
        {
            /* Cache the four depth values. */
            int point_index[4] = { y * w + x, y * w + x + 1, (y+1) * w + x, (y+1) * w + x + 1 };
            double depths[4] = { depthmap.at<unsigned short>(y, x) * cam_info.DepthImageScaling(),
                depthmap.at<unsigned short>(y, x+1) * cam_info.DepthImageScaling(),
                depthmap.at<unsigned short>(y+1, x) * cam_info.DepthImageScaling(),
                depthmap.at<unsigned short>(y+1, x+1) * cam_info.DepthImageScaling() };

            /* Create a mask representation of the available depth values. */
            int mask = 0;
            int pixels = 0;
            for (int j = 0; j < 4; ++j)
                if (depths[j] > 0.0)
                {
                    mask |= 1 << j;
                    pixels += 1;
                }

            /* At least three valid depth values are required. */
            if (pixels < 3)
                continue;

            /* Possible triangles, vertex indices relative to 2x2 block. */
            // 0--1
            // |  |
            // 2--3
            const int tris[4][3] = {
                { 0, 2, 1 }, { 0, 3, 1 }, { 0, 2, 3 }, { 1, 2, 3 }
            };

            /* Decide which triangles to issue. */
            int tri[2] = { 0, 0 };

            switch (mask)
            {
                case 7: tri[0] = 1; break;
                case 11: tri[0] = 2; break;
                case 13: tri[0] = 3; break;
                case 14: tri[0] = 4; break;
                case 15:
                         {
                             /* Choose the triangulation with smaller diagonal. */
                             float ddiff1 = std::abs(depths[0] - depths[3]);
                             float ddiff2 = std::abs(depths[1] - depths[2]);
                             if (ddiff1 < ddiff2)
                             { tri[0] = 2; tri[1] = 3; }
                             else
                             { tri[0] = 1; tri[1] = 4; }
                             break;
                         }
                default: continue;
            }

            /* Omit depth discontinuity detection if dd_factor is zero. */
            if (dd_factor > 0.0)
            {
                /* Cache pixel footprints. */
                double spherical_err[4] = {0};
                for (int j = 0; j < 4; ++j)
                {
                    if (depths[j] == 0.0)
                        continue;
                    spherical_err[j] = cam_info.DepthErrorWithDisparity(x + (j%2), depths[j], M_PI/(float(depthmap.cols)));
                }

                /* Check for depth discontinuities. */
                for (int j = 0; j < 2 && tri[j] != 0; ++j)
                {
                    const int* tv = tris[tri[j] - 1];
                    if (IsDepthDiscontinous(spherical_err, depths, dd_factor, tv[0], tv[1])) tri[j] = 0;
                    if (IsDepthDiscontinous(spherical_err, depths, dd_factor, tv[1], tv[2])) tri[j] = 0;
                    if (IsDepthDiscontinous(spherical_err, depths, dd_factor, tv[2], tv[0])) tri[j] = 0;
                }
            }

            if (check_edge_ratio > 0)
            {
                for (int j = 0; j < 2; ++j)
                {
                    if (tri[j] == 0) continue;
                    const int* tv = tris[tri[j] - 1];
                    cv::Vec3f vertices[3];
                    vertices[0] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[0]]));
                    vertices[1] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[1]]));
                    vertices[2] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[2]]));
                    if (CheckSlantedTriangleEdgeRatio(vertices, check_edge_ratio)) tri[j] = 0;
                }
            }

            if (check_view_angle > 0)
            {
                for (int j = 0; j < 2; ++j)
                {
                    if (tri[j] == 0) continue;
                    const int* tv = tris[tri[j] - 1];
                    cv::Vec3f vertices[3];
                    vertices[0] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[0]]));
                    vertices[1] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[1]]));
                    vertices[2] = PCLPoint2CvVec(point_cloud_world_coord->at(point_index[tv[2]]));
                    cv::Vec3f ref_cam_center = cam_info.RefCameraCenterInWorldCoord();
                    if (CheckSlantedTriangleNormal(vertices, ref_cam_center, check_view_angle)) tri[j] = 0;
                }
            }

            if (triangle_indices && triangle_type)
            {
                char cur_type = char(tri[0] | (tri[1]<<4));
                int cur_index = triangles->size();
                (*triangle_indices)[y*(w-1)+x] = cur_index;
                (*triangle_type)[y*(w-1)+x] = cur_type;
            }

            /* Build triangles. */
            for (int j = 0; j < 2; ++j)
            {
                if (tri[j] == 0) continue;
                const int *pointer_to_indices = tris[tri[j] - 1];
                triangles->push_back(AddTriangle(point_index[pointer_to_indices[0]],
                            point_index[pointer_to_indices[1]],
                            point_index[pointer_to_indices[2]]));
            }
        }
    }
    return;
}

void cpu_tsdf::DepthMapTriangulate(
        const cv::Mat& depthmap,
        const cv::Mat& image,
        const RectifiedCameraPair& cam_info,
        double dd_factor,
        double check_edge_ratio,
        double check_view_angle,
        pcl::PolygonMesh* mesh)
{
    assert(mesh);
    mesh->polygons.clear();
    pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
//    DepthMapTriangulateOld(depthmap, image, cam_info,
//                        dd_factor, check_edge_ratio, check_view_angle,
//                        &point_cloud, &(mesh->polygons));

    std::vector<std::vector<int>> pixel_triangles_list;
    DepthMapTriangulate(depthmap, image, cam_info, dd_factor,
                        &point_cloud, &(mesh->polygons), &pixel_triangles_list);

    if (check_edge_ratio)
    {
        RemoveSlantedTriangles(point_cloud, pixel_triangles_list, &(mesh->polygons),
                               [check_edge_ratio](pcl::PointXYZRGB* vertices)
        {
            return CheckSlantedTriangleEdgeRatio(vertices, check_edge_ratio);
        });
    }

    if (check_view_angle)
    {
        cv::Vec3f ref_cam_center = cam_info.RefCameraCenterInWorldCoord();
        RemoveSlantedTriangles(point_cloud, pixel_triangles_list, &(mesh->polygons),
                               [check_view_angle, ref_cam_center](pcl::PointXYZRGB* vertices)
        {
            return CheckSlantedTriangleNormal(vertices, ref_cam_center, check_view_angle);
        });
    }

    // remove invalid triangles
    auto last = std::remove_if(mesh->polygons.begin(), mesh->polygons.end(), [](const pcl::Vertices& val)
    {
        return val.vertices.empty();
    });
    std::cout << "num of invalid tris: " << mesh->polygons.end() - last << std::endl;
    mesh->polygons.erase(last, mesh->polygons.end());
    pcl::toPCLPointCloud2 (point_cloud, mesh->cloud);
}
