#pragma once
#include <vector>
#include <string>
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "common/fisheye_camera/RectifiedCameraPair.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_operation/tsdf_slice.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "tsdf_operation/tsdf_clean.h"

void SelectVisibleCamerasCoarse(const Eigen::Vector3f& point,
                                const std::vector<RectifiedCameraPair>& cameras,
                                std::vector<bool>* selected);

void CheckMeshVerticesWithSkyMap(const pcl::PolygonMesh &mesh,
        const std::vector<RectifiedCameraPair>& cameras,
        const std::vector<cv::Mat>& sky_masks, const int sky_thresh,
        std::vector<bool>* kept_vertices
        );

void CheckMeshVerticesWithSkyMapCheckOBB(const pcl::PolygonMesh &mesh,
        const std::vector<RectifiedCameraPair>& cameras,
        const std::vector<std::string> &skymap_filelist,
        const std::vector<tsdf_utility::OrientedBoundingBox>& obbs,
        const int sky_thresh,
        std::vector<bool>* kept_vertices
        );

void CleanTSDFWithSkyMap(cpu_tsdf::TSDFHashing::Ptr tsdf,
        const std::vector<tsdf_utility::OrientedBoundingBox>& obbs,
        const std::vector<RectifiedCameraPair> cameras,
        const std::vector<std::string>& skymap_filelist,
        float mesh_min_weight,
        int sky_thresh, const std::string &save_path);

void CleanMeshWithSkyMap(
        pcl::PolygonMesh& mesh,
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
        const std::vector<RectifiedCameraPair> cameras,
        const std::vector<std::string> &skymap_filelist,
        int sky_thresh,
        const std::string& save_path);

void PrepareVisibleCamsForOBB(const tsdf_utility::OrientedBoundingBox& obb,
        const std::vector<RectifiedCameraPair>& cameras,
        const std::vector<std::string>& skymap_filelist,
        std::vector<RectifiedCameraPair>* visible_cams,
        std::vector<cv::Mat>* visible_maps, std::vector<std::string> *filenames = NULL);

void CleanMeshWithSkyMapAndDepthMap(pcl::PolygonMesh &mesh,
                         const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
                         const std::vector<RectifiedCameraPair> cameras,
                         const std::vector<std::string> &skymap_filelist,
                         const std::vector<std::string> & depth_filelist,
                         int sky_thresh, const std::string &save_path, std::vector<bool> *kept_verteces = NULL, bool skymap_check = true, bool depthmap_check = true);

void CleanTSDFWithSkyMapAndDepthMap(cpu_tsdf::TSDFHashing::Ptr tsdf,
                         const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
                         const std::vector<RectifiedCameraPair> cameras,
                         const std::vector<std::string> &skymap_filelist,
                         const std::vector<std::string> & depth_filelist,
                         int sky_thresh, const std::string &save_path, int st_neighbor, int ed_neighbor, bool skymap_check = true, bool depthmap_check = true);

void CheckMeshVerticesWithSkyMapAndDepMapCheckOBB(const pcl::PolygonMesh &mesh,
        const std::vector<RectifiedCameraPair> &cameras,
        const std::vector<std::string> &skymap_filelist,
        const std::vector<std::string> &depthmap_filelist,
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
        const int sky_thresh, const bool sky_map_check, const bool depth_map_check,
        std::vector<bool> *kept_vertices);
////////////////////////////////////////////////////////////////////////


