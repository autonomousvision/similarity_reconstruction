#include "consistency_check.h"
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/search/kdtree.h>
#include <boost/dynamic_bitset.hpp>

#include "utility/utility.h"
#include "common/utility/common_utility.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_clean.h"
#include "common/utility/pcl_utility.h"
#include "utility/oriented_boundingbox.h"
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


//void CheckMeshVerticesWithSkyMap(const pcl::PolygonMesh &mesh,
//                                 const std::vector<RectifiedCameraPair> &cameras,
//                                 const std::vector<cv::Mat> &sky_masks,
//                                 const int sky_thresh,
//                                 std::vector<bool> *kept_vertices)
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
//    kept_vertices->resize(vertices->size(), true);
//    cv::Mat testmat(sky_masks[0].rows, sky_masks[0].cols, CV_8UC1);
//    testmat.setTo(0);
//    for (int i = 0; i < vertices->size(); ++i)
//    {
//        const pcl::PointXYZRGB pt = vertices->at(i);
//        Eigen::Vector3f cur_pt_pos(pt.x, pt.y, pt.z);
//        int sky_cnt = 0;
//        for (int camj = 0; camj < cameras.size(); ++camj)
//        {
//            const RectifiedCameraPair& cur_cam = cameras[camj];
//            cv::Vec3d vox_coord = cur_cam.WorldCoordToVoxel3DPoint(cv::Vec3d(utility::EigenVectorToCvVector3(cur_pt_pos)));
//            int imx, imy;
//            if ( cur_cam.Voxel3DPointToImageCoord(vox_coord, &imx, &imy) && sky_masks[camj].at<uchar>(imy, imx) > 0 )
//            {
//                testmat.at<uchar>(imy, imx) = 255;
//                sky_cnt ++;
//            }
//        }  // end for
//        if (sky_cnt > sky_thresh)
//        {
//            (*kept_vertices)[i] = false;
//        }
//    }  // end for
//}


//void CleanTSDFWithSkyMap(
//        cpu_tsdf::TSDFHashing::Ptr tsdf,
//        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
//        const std::vector<RectifiedCameraPair> cameras,
//        const std::vector<std::string> &skymap_filelist,
//        float mesh_min_weight, int sky_thresh,
//        const std::string& save_path)
//{
//    bfs::path pref = bfs::path(save_path).replace_extension();
//    for (int i = 0; i < obbs.size(); ++i)
//    {
//        // 1. slice the target tsdf
//        cpu_tsdf::TSDFHashing::Ptr cur_tsdf_slice(new cpu_tsdf::TSDFHashing);
//        cpu_tsdf::SliceTSDF<cpu_tsdf::PointInOrientedBox>(tsdf.get(), cpu_tsdf::PointInOrientedBox(tsdf_utility::OldOBBFromNewOBB(obbs[i])), cur_tsdf_slice.get());
//        // cpu_tsdf::SliceTSDF<cpu_tsdf::PointInOrientedBox>(tsdf.get(), cpu_tsdf::PointInOrientedBox(obbs[i]), cur_tsdf_slice.get());
//        cpu_tsdf::WriteTSDFModel(cur_tsdf_slice, pref.string() + "_before_" + utility::int2str(i) + ".ply", false, true, mesh_min_weight);

//        // 2. prepare the visible cams and skymaps
//        cout << "prepare visible cams" << endl;
//        std::vector<RectifiedCameraPair> cur_cams;
//        std::vector<cv::Mat> cur_skymaps;
//        PrepareVisibleCamsForOBB(obbs[i], cameras, skymap_filelist, &cur_cams, &cur_skymaps);

//        // 3. prepare mesh
//        pcl::PolygonMesh::Ptr mesh = cpu_tsdf::TSDFToPolygonMesh(cur_tsdf_slice, mesh_min_weight);
//        //pcl::io::savePLYFileBinary("/home/dell/tet1.ply", *mesh);

//        // 4. compute reserved vertex
//        std::vector<bool> kept_vertices;
//        CheckMeshVerticesWithSkyMap(*mesh, cur_cams, cur_skymaps, sky_thresh, &kept_vertices);

////        std::vector<bool> kept_vertices2;
////        CheckMeshVerticesWithDepthMapCheckOBB();

//        // 5. clean tsdf according to vertices
//        cpu_tsdf::CleanTSDFFromMeshVerts(cur_tsdf_slice, *mesh, kept_vertices, 0, 1);
//        cpu_tsdf::WriteTSDFModel(cur_tsdf_slice, pref.string() + "_after_" + utility::int2str(i) + ".ply", false, true, mesh_min_weight);

//        // 6. substitue the cleaned part in
//        cpu_tsdf::CleanTSDFPart(tsdf.get(), tsdf_utility::OldOBBFromNewOBB(obbs[i]));
//        cpu_tsdf::MergeTSDFNearestNeighbor(*cur_tsdf_slice, tsdf.get());
//    }  // end for obbs
//}

//void CheckMeshVerticesWithSkyMapCheckOBB(
//        const pcl::PolygonMesh &mesh,
//        const std::vector<RectifiedCameraPair> &cameras,
//        const std::vector<std::string> &skymap_filelist,
//        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
//        const int sky_thresh,
//        std::vector<bool> *kept_vertices)
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
//    kept_vertices->resize(vertices->size(), true);
//    int prev_intersect_obb = -1;
//    vector<cv::Mat> cache_skymaps;
//    vector<RectifiedCameraPair> cache_cams;
//    for (int i = 0; i < vertices->size(); ++i)
//    {
//        const pcl::PointXYZRGB pt = vertices->at(i);
//        Eigen::Vector3f cur_pt_pos(pt.x, pt.y, pt.z);
//        int inbox = 0;
//        for (; inbox < obbs.size(); ++inbox)
//        {
//            if (cpu_tsdf::VerticeInOBB(cur_pt_pos, tsdf_utility::OldOBBFromNewOBB(obbs[inbox])))
//            {
//                break;
//            }
//        }
//        // not in any obb
//        if (inbox == obbs.size())
//        {
//            continue;
//        }
//        // update cached visible maps
//        else if (inbox != prev_intersect_obb)
//        {
//            // check visible maps
//            // cout << "vert " << i << " box " << inbox << endl;
//            PrepareVisibleCamsForOBB(
//                    obbs[inbox],
//                    cameras,
//                    skymap_filelist,
//                    &cache_cams,
//                    &cache_skymaps);
//        }
//        int sky_cnt = 0;
//        for (int camj = 0; camj < cache_cams.size(); ++camj)
//        {
//            const RectifiedCameraPair& cur_cam = cache_cams[camj];
//            cv::Vec3d vox_coord = cur_cam.WorldCoordToVoxel3DPoint(cv::Vec3d(utility::EigenVectorToCvVector3(cur_pt_pos)));
//            int imx, imy;
//            if ( cur_cam.Voxel3DPointToImageCoord(vox_coord, &imx, &imy) && cache_skymaps[camj].at<uchar>(imy, imx) > 0 )
//            {
//                //cout << "hit img" << endl;
//                sky_cnt ++;
//            }
//        }  // end for camj
//        if (sky_cnt >= sky_thresh)
//        {
//            (*kept_vertices)[i] = false;
//            //cout << i << " set false" << endl;
//        }
//        prev_intersect_obb = inbox;
//    }  // end for vert i
//}

//void CheckMeshVerticesWithDepthMapCheckOBB(
//        const pcl::PolygonMesh &mesh,
//        const std::vector<RectifiedCameraPair> &cameras,
//        const std::vector<std::string> &depthmap_filelist,
//        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
//        std::vector<bool> *kept_vertices)
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
//    kept_vertices->resize(vertices->size(), true);
//    int prev_intersect_obb = -1;
//    vector<cv::Mat> cache_depthmaps;
//    vector<RectifiedCameraPair> cache_cams;
//    for (int i = 0; i < vertices->size(); ++i)
//    {
//        const pcl::PointXYZRGB pt = vertices->at(i);
//        Eigen::Vector3f cur_pt_pos(pt.x, pt.y, pt.z);
//        int inbox = 0;
//        for (; inbox < obbs.size(); ++inbox)
//        {
//            if (cpu_tsdf::VerticeInOBB(cur_pt_pos, tsdf_utility::OldOBBFromNewOBB(obbs[inbox])))
//            {
//                break;
//            }
//        }
//        // not in any obb
//        if (inbox == obbs.size())
//        {
//            continue;
//        }
//        // update cached visible maps
//        else if (inbox != prev_intersect_obb)
//        {
//            // check visible maps
//            // cout << "vert " << i << " box " << inbox << endl;
//            PrepareVisibleCamsForOBB(
//                    obbs[inbox],
//                    cameras,
//                    depthmap_filelist,
//                    &cache_cams,
//                    &cache_depthmaps);
//        }
//        int consistent_cnt = 0;
//        int inconsistent_cnt = 0;  // 3d mesh vertex is before an observed depth value
//        for (int camj = 0; camj < cache_cams.size(); ++camj)
//        {
//            const RectifiedCameraPair& cur_cam = cache_cams[camj];
//            cv::Vec3d vox_coord = cur_cam.WorldCoordToVoxel3DPoint(cv::Vec3d(utility::EigenVectorToCvVector3(cur_pt_pos)));
//            int imx, imy;
//            float length;
//            if ( cur_cam.Voxel3DPointToImageCoord(vox_coord, &imx, &imy, &length) && cache_depthmaps[camj].at<ushort>(imy, imx) > 0 )
//            {
//                ushort obs_depth = cache_depthmaps[camj].at<ushort>(imy, imx);
//                float fobs_depth = cur_cam.DepthImageScaling() * (float)obs_depth;
//                const float support_thresh = cur_cam.DepthErrorWithDisparity(imx, fobs_depth, M_PI/float(cache_depthmaps[camj].cols));

//                if (fabs(fobs_depth - length) < support_thresh * 3)
//                {
//                    consistent_cnt++;
//                }
//                else if (length + 3 * support_thresh < fobs_depth)
//                {
//                    inconsistent_cnt++;
//                }

////                cout << "support_thresh: " << support_thresh << " ";
////                cout << "fobs_depth: " << fobs_depth << " ";
////                cout << "length: " << length << endl;
////                cout << "consistent_cnt: " << consistent_cnt << " ";
////                cout << "inconsistent_cnt: " << inconsistent_cnt << endl;
//            }
//        }  // end for camj
//        if (inconsistent_cnt > 1 && consistent_cnt < 2 /*&& consistent_cnt < 2*/)
//        {
//            (*kept_vertices)[i] = false;
//        }
//        prev_intersect_obb = inbox;
//    }  // end for vert i
//}

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
                // break;
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
            //float debug_min_dep_diff = 1e6;
            //float min_sup_times = 1e6;
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
                //if (inbox == 19 && fabs(cur_pt_pos[0] - 1070.2) < 0.6 && fabs(cur_pt_pos[1] - 3918.29) < 0.6 && fabs(cur_pt_pos[2] - 122.2) < 0.6 ) {
                //    using namespace std;
//              //      cout << camj << "th cam" << endl;
//              //      cout << "curptpos: " << cur_pt_pos << endl;
                //    //cout << cur_cam << endl;
                //    cout << "skycnt: "  << sky_cnt << endl;
                //    cout << "camj: "<< camj << endl;
                //    cout << "camj: "<< cur_depthmap_files [camj] << endl;
                //    cout << "ver idx: " << obb_vert_idx[inbox][i] << endl;
                //    cout << "valid " << valid_projection << endl;
                //    cout << "occluding " << occluding_cnt << endl;
                //    cout << "consis " << consistent_cnt << endl;
                //    cout << "occlud " << occluded_cnt << endl;
                //    //cout << "debug min dep diff: " << debug_min_dep_diff << endl;
                //    //cout << "debug_min supp times: " << min_sup_times << endl;
                //}  // end debug if
                }
                if ( in_image && cache_depthmaps[camj].at<ushort>(imy, imx) > 0 && depth_map_check)
                {
                    valid_projection++;
                    ushort obs_depth = cache_depthmaps[camj].at<ushort>(imy, imx);
                    float fobs_depth = cur_cam.DepthImageScaling() * (float)obs_depth;
                    const float support_thresh = cur_cam.DepthErrorWithDisparity(imx, fobs_depth, M_PI/float(cache_depthmaps[camj].cols));
                    static const int supp_times = 4;
                    //float debug_diff_dep = fabs(fobs_depth - length);
                    //if (debug_diff_dep < debug_min_dep_diff) {
                    //    debug_min_dep_diff = debug_diff_dep;
                    //}
                    //float debug_supp_times = debug_diff_dep / support_thresh;
                    //if (debug_supp_times < min_sup_times) {
                    //    min_sup_times = debug_supp_times;
                    //}
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
            // the newly introduced part cannot be seen
            //if (valid_projection > non_noise_observation_thresh && occluding_cnt == valid_projection) {
            //    (*kept_vertices)[obb_vert_idx[inbox][i]] = false;
            //    //if (inbox == 19 && fabs(cur_pt_pos[0] - 1077.6) < 0.8 && fabs(cur_pt_pos[1] - 3925.54) < 0.8 && fabs(cur_pt_pos[2] - 116.4) < 0.8 ) {
            //    //    using namespace std;
//          //    //      cout << camj << "th cam" << endl;
//          //    //      cout << "curptpos: " << cur_pt_pos << endl;
            //    //    //cout << cur_cam << endl;
            //    //    cout << "ver idx: " << obb_vert_idx[inbox][i] << endl;
            //    //    cout << "valid " << valid_projection << endl;
            //    //    cout << "occluding " << occluding_cnt << endl;
            //    //    cout << "consis " << consistent_cnt << endl;
            //    //    cout << "occlud " << occluded_cnt << endl;
            //    //    cout << "debug min dep diff: " << debug_min_dep_diff << endl;
            //    //    cout << "debug_min supp times: " << min_sup_times << endl;
            //    //}  // end debug if
            //} // end if
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
//            cout << j << "th cam" << endl;
//            cout << "filename: " << skymap_filelist[j] << endl;
        }
    }
    *visible_cams = cur_cams;
    *visible_maps = cur_skymaps;
}

//void CleanMeshWithSkyMap(pcl::PolygonMesh &mesh,
//                         const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
//                         const std::vector<RectifiedCameraPair> cameras,
//                         const std::vector<string> &skymap_filelist,
//                         int sky_thresh, const string &save_path)
//{
//    vector<bool> keep_vertices;
//    CheckMeshVerticesWithSkyMapCheckOBB(mesh, cameras, skymap_filelist, obbs, sky_thresh, &keep_vertices);
//    utility::ClearMeshWithVertKeepArray(mesh, keep_vertices);
//}

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
    CleanTSDFFromMeshVerts(tsdf, *mesh, keep_vertices, st_neighbor, ed_neighbor);
    return;
}
