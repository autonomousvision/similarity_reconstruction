/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "consistency_check.h"
#include <vector>
#include <string>
#include <cmath>
#include <climits>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_operation/tsdf_clean.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "common/utilities/common_utility.h"
#include "common/utilities/eigen_utility.h"
#include "common/utilities/pcl_utility.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_clean.h"
using namespace std;

void SelectVisibleCamerasCoarse(
        const Eigen::Vector3f &point,
        const std::vector<RectifiedCameraPair> &cameras,
        std::vector<bool> *selected)
{
    selected->resize(cameras.size(), false);
    for (int i = 0; i < cameras.size(); ++i)
    {
        if (PointVisibleCoarse(point, cameras[i]))
        {
            (*selected)[i] = true;
        }
    }
}

void CheckMeshVerticesWithSkyMapAndDepMapCheckOBB(
        const pcl::PolygonMesh &mesh,
        const std::vector<RectifiedCameraPair> &cameras,
        const std::vector<std::string> &skymap_filelist,
        const std::vector<std::string> &depthmap_filelist,
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
        const int sky_thresh,
        const bool sky_map_check,
        const bool depth_map_check,
        std::vector<bool> *kept_vertices)
{
    cout << "begin check consistency, skymap and depth map" << endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
    kept_vertices->resize(vertices->size(), true);
    vector<vector<int>> obb_vert_idx(obbs.size());

    for (int i = 0; i < vertices->size(); ++i)
    {
        const pcl::PointXYZRGB& pt = vertices->at(i);
        Eigen::Vector3f cur_pt_pos(pt.x, pt.y, pt.z);
        for (int inbox = 0; inbox < obbs.size(); ++inbox)
        {
            if (cpu_tsdf::VerticeInOBB(cur_pt_pos, tsdf_utility::OldOBBFromNew(obbs[inbox])))
            {
                obb_vert_idx[inbox].push_back(i);
            }
        }
    }
    for (int inbox = 0; inbox < obbs.size(); ++inbox)
    {
        vector<cv::Mat> cache_depthmaps;
        vector<cv::Mat> cache_skymaps;
        vector<RectifiedCameraPair> cache_cams;
        PrepareVisibleCamsForOBB(
                obbs[inbox],
                cameras,
                skymap_filelist,
                &cache_cams,
                &cache_skymaps);
        std::vector<string> cur_depthmap_files;
        PrepareVisibleCamsForOBB(
                        obbs[inbox],
                        cameras,
                        depthmap_filelist,
                        &cache_cams,
                        &cache_depthmaps,
                        &cur_depthmap_files);
        for (int i = 0; i < obb_vert_idx[inbox].size(); ++i)  // iterate over vertices in this OBB
        {
            const pcl::PointXYZRGB& pt = vertices->at(obb_vert_idx[inbox][i]);
            Eigen::Vector3f cur_pt_pos(pt.x, pt.y, pt.z);
            int sky_cnt = 0;
            int consistent_cnt = 0;
            int occluded_cnt = 0;  // 3d mesh vertex is before an observed depth value
            int occluding_cnt = 0;
            int valid_projection = 0;
            for (int camj = 0; camj < cache_cams.size(); ++camj)
            {
                const RectifiedCameraPair& cur_cam = cache_cams[camj];
                cv::Vec3d vox_coord = cur_cam.WorldCoordToVoxel3DPoint(cv::Vec3d(utility::EigenVectorToCvVector3(cur_pt_pos)));
                int imx, imy;
                float length;
                bool in_image = false;
                if ( (in_image = cur_cam.Voxel3DPointToImageCoord(vox_coord, &imx, &imy, &length)) &&
                     cache_skymaps[camj].at<uchar>(imy, imx) > 0 &&
                     sky_map_check)
                {
                    sky_cnt ++;
                }
                if ( in_image && cache_depthmaps[camj].at<ushort>(imy, imx) > 0 && depth_map_check)
                {
                    valid_projection++;
                    ushort obs_depth = cache_depthmaps[camj].at<ushort>(imy, imx);
                    float fobs_depth = cur_cam.DepthImageScaling() * (float)obs_depth;
                    const float support_thresh = cur_cam.DepthErrorWithDisparity(imx, fobs_depth, M_PI/float(cache_depthmaps[camj].cols));
                    static const int supp_times = 4;

                    if (fabs(fobs_depth - length) < support_thresh * supp_times) {
                        consistent_cnt++;
                    }
                    else if (length + supp_times * support_thresh < fobs_depth) {
                        occluded_cnt++;
                    } else {
                        occluding_cnt++;
                    }
                }

            }  // end for camj
            if (sky_cnt > sky_thresh) {
                (*kept_vertices)[obb_vert_idx[inbox][i]] = false;
            }
            static const int non_noise_observation_thresh = 0;
            // the newly introduced part occluded by observed data
            if (occluded_cnt > non_noise_observation_thresh && float(occluded_cnt)/float(occluded_cnt + consistent_cnt) > 0.5 ) {
                (*kept_vertices)[obb_vert_idx[inbox][i]] = false;
            }
        }  // end vertices
    }  // end for inbox
}

void PrepareVisibleCamsForOBB(const tsdf_utility::OrientedBoundingBox &obb, const std::vector<RectifiedCameraPair> &cameras, const std::vector<std::string> &skymap_filelist, std::vector<RectifiedCameraPair> *visible_cams, std::vector<cv::Mat> *visible_maps,
                              std::vector<string> *filenames)
{
    //cout << "prepare visible cams" << endl;
    Eigen::Vector3f obb_center = obb.Center();
    std::vector<bool> selected;
    SelectVisibleCamerasCoarse(obb_center, cameras, &selected);
    std::vector<RectifiedCameraPair> cur_cams;
    std::vector<cv::Mat> cur_skymaps;
    for (int j = 0; j < selected.size(); ++j)
    {
        if (selected[j])
        {
            cur_cams.push_back(cameras[j]);
            cv::Mat curmat = cv::imread(skymap_filelist[j], CV_LOAD_IMAGE_UNCHANGED);
            cur_skymaps.push_back(curmat);  // shallow copy
            if (filenames) filenames->push_back(skymap_filelist[j]);
        }
    }
    *visible_cams = cur_cams;
    *visible_maps = cur_skymaps;
}

void CleanMeshWithSkyMapAndDepthMap(pcl::PolygonMesh &mesh,
                         const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
                         const std::vector<RectifiedCameraPair> cameras,
                         const std::vector<string> &skymap_filelist,
                         const std::vector<string> & depth_filelist,
                         int sky_thresh, const string &save_path,
                         std::vector<bool>* kept_verteces,
                         bool skymap_check,
                         bool depthmap_check)
{
    vector<bool> keep_vertices;
    CheckMeshVerticesWithSkyMapAndDepMapCheckOBB(
            mesh, cameras, skymap_filelist, depth_filelist, obbs, sky_thresh, skymap_check, depthmap_check, &keep_vertices);
    utility::ClearMeshWithVertKeepArray(mesh, keep_vertices);
    if (kept_verteces)
        *kept_verteces = keep_vertices;
}

void CleanTSDFWithSkyMapAndDepthMap(
        cpu_tsdf::TSDFHashing::Ptr tsdf,
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
        const std::vector<RectifiedCameraPair> cameras,
        const std::vector<string> &skymap_filelist,
        const std::vector<string> &depth_filelist,
        int sky_thresh,
        const string &save_path,
        int st_neighbor,
        int ed_neighbor,
        bool skymap_check,
        bool depthmap_check
        )
{
    pcl::PolygonMesh::Ptr mesh = cpu_tsdf::TSDFToPolygonMesh(tsdf, 0);
    vector<bool> keep_vertices;
    CheckMeshVerticesWithSkyMapAndDepMapCheckOBB(
            *mesh, cameras, skymap_filelist, depth_filelist, obbs, sky_thresh, skymap_check, depthmap_check, &keep_vertices);
    cpu_tsdf::CleanTSDFFromMeshVerts(tsdf, *mesh, keep_vertices, st_neighbor, ed_neighbor);
    return;

    // mesh = cpu_tsdf::TSDFToPolygonMesh(tsdf, 0);
    // // CleanMeshWithSkyMapAndDepthMap(
    // //                         *mesh,
    // //                         obbs,
    // //                         cameras,
    // //                         skymap_filelist,
    // //                         depth_filelist,
    // //                         sky_thresh,
    // //                         save_path,
    // //                         NULL,
    // //                         skymap_check,
    // //                         depthmap_check
    // //                         );
    // cpu_tsdf::TSDFHashing::Ptr template_tsdf(new cpu_tsdf::TSDFHashing);
    // float max_trunc, neg_trunc;
    // tsdf->getDepthTruncationLimits(max_trunc, neg_trunc);
    // template_tsdf->Init(tsdf->voxel_length(), tsdf->offset(), max_trunc, neg_trunc);
    // Mesh2TSDFRangeTemplate(*mesh, template_tsdf.get());
    // SetTSDFAccordingToTemplateWeight(*template_tsdf, tsdf.get());

    return;
}

void Mesh2TSDFRangeTemplate(const pcl::PolygonMesh &mesh, cpu_tsdf::TSDFHashing *tsdf)
{
    // tsdf->Init(tsdf_info.voxel_lengths()[0], tsdf_info.offset(), tsdf_info.max_dist_pos(), tsdf_info.max_dist_neg());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
    float max_trunc, neg_trunc;
    tsdf->getDepthTruncationLimits(max_trunc, neg_trunc);
    int tsdf_valid_range_dist = ceil(max<float>(fabs(max_trunc), fabs(neg_trunc)) / tsdf->voxel_length());
    Eigen::Vector2i neighbor(-tsdf_valid_range_dist, tsdf_valid_range_dist+1);
    for (int i = 0; i < vertices->size(); ++i) {
        const pcl::PointXYZRGB& pt = vertices->at(i);
        Eigen::Vector3f cur_pt_pos(pt.x, pt.y, pt.z);
        Eigen::Vector3i vox_coord = utility::round(tsdf->World2Voxel(cur_pt_pos)).cast<int>();
        for (int ix = neighbor[0]; ix < neighbor[1]; ++ix)
            for (int iy = neighbor[0]; iy < neighbor[1]; ++iy)
                for (int iz = neighbor[0]; iz < neighbor[1]; ++iz) {
                    Eigen::Vector3i cur_vox = vox_coord + Eigen::Vector3i(ix, iy, iz);
                    tsdf->AddObservation(utility::EigenVectorToCvVector3(cur_vox), 1.0f, 1.0f, cv::Vec3b(0, 0, 0));
                }
    }
}

void SetTSDFAccordingToTemplateWeight(const cpu_tsdf::TSDFHashing& template_tsdf, cpu_tsdf::TSDFHashing* tsdf) {
    using namespace cpu_tsdf;
    float max_trunc, neg_trunc;
    tsdf->getDepthTruncationLimits(max_trunc, neg_trunc);
    for (TSDFHashing::iterator citr = tsdf->begin(); citr != tsdf->end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        if (!template_tsdf.RetriveData(cur_voxel_coord, &d, &w, &color) || w <= std::numeric_limits<float>::epsilon() ) {
            citr->SetTSDFValue(0, 0, cv::Vec3b(0, 0, 0), max_trunc, neg_trunc);
        }
    }  // end for
}
