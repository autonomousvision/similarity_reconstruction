/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include <vector>
#include <cmath> // std::abs
#include <cassert>
#include <memory>
#include <iostream>
#include <climits>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "common/fisheye_camera/RectifiedCameraPair.h"
//#define _ZCDEBUG

using std::vector;
using std::cout;
using std::endl;
using namespace cv;
namespace fs = boost::filesystem;

const double EPSILON = 5e-2;

void DebugPointCloud(const std::vector<RectifiedCameraPair>& cam_infos,
                     const std::vector<std::unique_ptr<cv::Mat>>& depths,
                     const std::vector<std::unique_ptr<cv::Mat>>& images,
                     double maxCamDistance,
                     std::vector<cv::Vec3d>& points3d, std::vector<cv::Vec3i>& pointscolor)
{
    for (int i = 0; i < depths.size(); ++i)
    {
        int imHeight = depths[i]->rows;
        int imWidth = depths[i]->cols;
        for(int y=0; y<imHeight; y++)
        {
            for(int x=0; x<imWidth; x++)
            {
                const unsigned short quant_depth = (depths[i])->at<unsigned short>(y,x);
                const float pz = float(quant_depth)*cam_infos[i].DepthImageScaling();  // unit: in meter
                if(pz > 0.0f && pz < maxCamDistance)
                {
                    cv::Vec3d tpoint = cam_infos[i].RectifiedImagePointToVoxel3DPoint(x, y, quant_depth);  // unit: in voxel
                    //printf("%f %f %f\n",float(x), float(y), float(pz) );
                    points3d.push_back(static_cast<cv::Vec3d>(tpoint));
                    cv::Vec3b tcolor = images[i]->at<cv::Vec3b>(y,x);
                    pointscolor.push_back(tcolor);
                }
            }
        }
    }
}

void WritePointCloudToFile(const std::string& filename, const std::vector<cv::Vec3d>& point3d, const std::vector<cv::Vec3i>& pointscolor)
{
    FILE* hf = fopen(filename.c_str(), "w");
    assert(hf);
    assert(point3d.size() == pointscolor.size());
    for(int i = 0; i < point3d.size(); ++i)
    {

        fprintf(hf, "%f %f %f %d %d %d\n", point3d[i][0], point3d[i][1], point3d[i][2],
                pointscolor[i][2], pointscolor[i][1], pointscolor[i][0]);
    }
    fclose(hf);
}

void VisibilityFusion(const vector<RectifiedCameraPair>& cam_infos,
                      const vector<std::unique_ptr<cv::Mat>>& depths,
                      cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance)
{
    assert(cam_infos.size() > 1);
    assert(cam_infos.size() == depths.size());
    const int ref_depth = 0;
    *fusioned_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    *confidence_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    // rectified k -> rectified ref
    vector<cv::Matx33d> rotation_k2ref(cam_infos.size());
    vector<cv::Vec3d> translation_k2ref(cam_infos.size());
    vector<cv::Matx33d> rotation_ref2k(cam_infos.size());
    vector<cv::Vec3d> translation_ref2k(cam_infos.size());
    // store all transform matrices
    const Matx33d& rectify_ref = cam_infos[ref_depth].ReferenceRectifyMat();
    Matx33d ref_rotation;
    Vec3d ref_translation;
    cam_infos[ref_depth].ReferenceCameraPose(&ref_rotation, &ref_translation);
    rotation_k2ref[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    rotation_ref2k[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    for (int i = 1; i < cam_infos.size(); ++i)
    {
        Matx33d k_rotation;
        Vec3d k_translation;
        cam_infos[i].ReferenceCameraPose(&k_rotation, &k_translation);
        Matx33d rectify_k = cam_infos[i].ReferenceRectifyMat();
        Matx33d rotation_k_ref = rectify_ref.t() * ref_rotation.t() * k_rotation * rectify_k;
        Vec3d translation_k_ref = rectify_ref.t() * ref_rotation.t() * (k_translation - ref_translation);
        rotation_k2ref[i] = rotation_k_ref;
        translation_k2ref[i] = translation_k_ref;
        rotation_ref2k[i] = rotation_k_ref.t();
        translation_ref2k[i] = -rotation_k_ref.t() * translation_k_ref;
    }
    // generate all depth candidates
    cout << "generate all depth candidates.. ";
    const RectifiedCameraPair& ref_cam_info = cam_infos[ref_depth];
    const double depth_image_scaling_factor = ref_cam_info.DepthImageScaling();
    vector<vector<vector<double> > > ref_depth_candidate_map(depths[ref_depth]->rows);
    vector<vector<bool> > ref_depth_indicator_map(depths[ref_depth]->rows);
    for (int r = 0; r < ref_depth_candidate_map.size(); ++r)
    {
        ref_depth_candidate_map[r].resize(depths[ref_depth]->cols);
        ref_depth_indicator_map[r].resize(depths[ref_depth]->cols, false);
    }
    for (int i = 0; i < cam_infos.size(); ++i)
    {
        // for each new depthmap reset the indicator map
        for (int r = 0; r < ref_depth_indicator_map.size(); ++r)
        {
            ref_depth_indicator_map[r].assign(depths[ref_depth]->cols, false);
        }
        // for all depth maps, including the reference pair, reproject to ref view
        const RectifiedCameraPair& cur_cam_info = cam_infos[i];
        for (int r = 0; r < depths[i]->rows; ++r)
        {
            for (int c = 0; c < depths[i]->cols; ++c)
            {
                unsigned short cur_int_depth = depths[i]->at<unsigned short>(r, c);
                double cur_true_depth = (double)cur_int_depth * depth_image_scaling_factor;
                if (cur_true_depth < 0.0 + 1e-9 || cur_true_depth > maxCamDistance)
                {
                    continue;
                }
                cv::Vec3d k_rectified_cam_3d_pt =
                    cur_cam_info.RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(c, r, cur_true_depth);
                cv::Vec3d ref_rectified_cam_3d_pt = rotation_k2ref[i] * k_rectified_cam_3d_pt + translation_k2ref[i];
                cv::Point2i ref_rectified_im_pt;
                ref_cam_info.RectifiedCoordPointToImageCoord(ref_rectified_cam_3d_pt[0],
                        ref_rectified_cam_3d_pt[1],
                        ref_rectified_cam_3d_pt[2],
                        &ref_rectified_im_pt.x,
                        &ref_rectified_im_pt.y);

                if(i == 0 && r > 0)
                {
                    if(r != ref_rectified_im_pt.y || c != ref_rectified_im_pt.x)
                    {
                        using std::cout;
                        using std::endl;
                        cout << "error happend!" << endl;
                        cout << r <<" " << c << endl;
                        cout << "rec_rectified_im_pt: " << ref_rectified_im_pt.y <<" " << ref_rectified_im_pt.x << endl;
                    }
                }

                double ref_true_depth = cv::norm(ref_rectified_cam_3d_pt);
                if ( 0 <= ref_rectified_im_pt.x && ref_rectified_im_pt.x < depths[ref_depth]->cols &&
                        0 <= ref_rectified_im_pt.y && ref_rectified_im_pt.y < depths[ref_depth]->rows &&
                        ref_true_depth <= maxCamDistance )
                {
                    if (ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] == false)
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(ref_true_depth);
                        ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] = true;
                    }
                    else
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back()
                            = std::min(ref_true_depth, ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back());
                    }
                }  //end if
            }  //end for c
        }  // end for r

    }  // end for i

    // for every candidates on each pixel of the ref depth map, select the best.
    cout << "selecting best candidate...\n";
    for (int r = 0; r < depths[ref_depth]->rows; ++r)
    {
        for (int c = 0; c < depths[ref_depth]->cols; ++c)
        {
            vector<double>& cur_list = ref_depth_candidate_map[r][c];
            std::sort(cur_list.begin(), cur_list.end());
            for (int i = 0; i < cur_list.size(); ++i)    // for every candidate
            {
                int occlude_cnt = i;
                int free_violation_cnt = 0;
                int support = 0;
                int all_depth_cnt = 0;
                for (int q = 1; q < depths.size(); ++q)
                {
                    cv::Vec3d rect_ref3d_pt =
                        ref_cam_info.RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(c, r, cur_list[i]);
                    cv::Vec3d q_rectified_3d_pt =
                        rotation_ref2k[q] * rect_ref3d_pt + translation_ref2k[q];
                    cv::Point2i q_rect_im_pt;
                    cam_infos[q].RectifiedCoordPointToImageCoord(q_rectified_3d_pt[0],
                            q_rectified_3d_pt[1],
                            q_rectified_3d_pt[2],
                            &q_rect_im_pt.x,
                            &q_rect_im_pt.y);
                    double ref2q_depth = cv::norm(q_rectified_3d_pt);
                    if ( 0 <= q_rect_im_pt.x && q_rect_im_pt.x < depths[q]->cols &&
                            0 <= q_rect_im_pt.y && q_rect_im_pt.y < depths[q]->rows )
                    {
                        double q_depth_val = double(depths[q]->at<unsigned short>(q_rect_im_pt.y, q_rect_im_pt.x)) * cam_infos[q].DepthImageScaling();
                        if (q_depth_val > ref2q_depth + EPSILON)
                        {
                            free_violation_cnt++;
                        }
                        if (std::abs(q_depth_val - ref2q_depth) <= EPSILON)
                        {
                            support++;
                        }
                        all_depth_cnt++;
                    }
                }
                if (occlude_cnt - free_violation_cnt > 0)
                {
                    fusioned_map->at<double>(r, c) = cur_list[i];
                    confidence_map->at<double>(r, c) = double(support)/all_depth_cnt;
                    break;
                }  // end if
            }  // end for i
        }  // end for c
    }  // end for r
    return;
}


void SimpleFusion(const vector<RectifiedCameraPair>& cam_infos,
                  const vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  const std::string& save_dir /*= std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug")*/)
{
    assert(cam_infos.size() > 1);
    assert(cam_infos.size() == depths.size());
    const int ref_depth = 0;
    *fusioned_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    *confidence_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    // rectified k -> rectified ref
    vector<cv::Matx33d> rotation_k2ref(cam_infos.size());
    vector<cv::Vec3d> translation_k2ref(cam_infos.size());
    vector<cv::Matx33d> rotation_ref2k(cam_infos.size());
    vector<cv::Vec3d> translation_ref2k(cam_infos.size());
    // store all transform matrices
    const Matx33d& rectify_ref = cam_infos[ref_depth].ReferenceRectifyMat();
    Matx33d ref_rotation;
    Vec3d ref_translation;
    cam_infos[ref_depth].ReferenceCameraPose(&ref_rotation, &ref_translation);
    rotation_k2ref[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    rotation_ref2k[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    for (int i = 1; i < cam_infos.size(); ++i)
    {
        Matx33d k_rotation;
        Vec3d k_translation;
        cam_infos[i].ReferenceCameraPose(&k_rotation, &k_translation);
        Matx33d rectify_k = cam_infos[i].ReferenceRectifyMat();
        Matx33d rotation_k_ref = rectify_ref.t() * ref_rotation.t() * k_rotation * rectify_k;
        Vec3d translation_k_ref = rectify_ref.t() * ref_rotation.t() * (k_translation - ref_translation);
        rotation_k2ref[i] = rotation_k_ref;
        translation_k2ref[i] = translation_k_ref;
        rotation_ref2k[i] = rotation_k_ref.t();
        translation_ref2k[i] = -rotation_k_ref.t() * translation_k_ref;
    }
    // generate all depth candidates
    cout << "generate all depth candidates.. ";
    const RectifiedCameraPair& ref_cam_info = cam_infos[ref_depth];
    const double depth_image_scaling_factor = ref_cam_info.DepthImageScaling();
    vector<vector<vector<double> > > ref_depth_candidate_map(depths[ref_depth]->rows);
    vector<vector<bool> > ref_depth_indicator_map(depths[ref_depth]->rows);
    for (int r = 0; r < ref_depth_candidate_map.size(); ++r)
    {
        ref_depth_candidate_map[r].resize(depths[ref_depth]->cols);
        ref_depth_indicator_map[r].resize(depths[ref_depth]->cols, false);
    }
    for (int i = 1; i < cam_infos.size(); ++i)
    {
#ifdef _ZCDEBUG
        cv::Mat cur_depth_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_16UC1);
        cv::Mat cur_rgb_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_8UC3);
        cv::Mat image = cv::imread(image_file_list[i]);
#endif // _ZCDEBUG
        // for each new depthmap reset the indicator map
        for (int r = 0; r < ref_depth_indicator_map.size(); ++r)
        {
            ref_depth_indicator_map[r].assign(depths[ref_depth]->cols, false);
        }
        // for all depth maps, including the reference pair, reproject to ref view
        const RectifiedCameraPair& cur_cam_info = cam_infos[i];
        for (int r = 0; r < depths[i]->rows; ++r)
        {
            for (int c = 0; c < depths[i]->cols; ++c)
            {
                unsigned short cur_int_depth = depths[i]->at<unsigned short>(r, c);
                double cur_true_depth = (double)cur_int_depth * depth_image_scaling_factor;
                if (cur_true_depth <= 0.0 + 1e-10 || cur_true_depth > maxCamDistance)
                {
                    continue;
                }
                cv::Vec3d k_rectified_cam_3d_pt =
                    cur_cam_info.RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(c, r, cur_true_depth);
                cv::Vec3d ref_rectified_cam_3d_pt = rotation_k2ref[i] * k_rectified_cam_3d_pt + translation_k2ref[i];
                cv::Point2i ref_rectified_im_pt;
                ref_cam_info.RectifiedCoordPointToImageCoord(ref_rectified_cam_3d_pt[0],
                        ref_rectified_cam_3d_pt[1],
                        ref_rectified_cam_3d_pt[2],
                        &ref_rectified_im_pt.x,
                        &ref_rectified_im_pt.y);
                double ref_true_depth = cv::norm(ref_rectified_cam_3d_pt);
                if ( 0 <= ref_rectified_im_pt.x && ref_rectified_im_pt.x < depths[ref_depth]->cols &&
                        0 <= ref_rectified_im_pt.y && ref_rectified_im_pt.y < depths[ref_depth]->rows &&
                        ref_true_depth <= maxCamDistance )
                {
                    if (ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] == false)
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(ref_true_depth);
                        ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] = true;
                    }
                    else
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back()
                            = std::min(ref_true_depth, ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back());
                    }
#ifdef _ZCDEBUG
                    cur_depth_map.at<ushort>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = ushort(ref_true_depth/30.0*65535);
                    cv::Vec3b cur_color = image.at<cv::Vec3b>(r, c);
                    cur_rgb_map.at<cv::Vec3b>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = cur_color;

#endif // _ZCDEBUG

                }  //end if
            }  //end for c
        }  // end for r
#ifdef _ZCDEBUG
        fs::path fbasename = fs::path(param_file_list[i]).stem();
        std::string save_path = save_dir + "/" + fbasename.string() + "-warped.png";
        imwrite(save_path, cur_depth_map);
        std::string save_path_orig = save_dir + "/" + fbasename.string() + "-orig.png";
        imwrite(save_path_orig, *(depths[i]));
        std::string save_path_rgb = save_dir + "/" + fbasename.string() + "-rgb.png";
        imwrite(save_path_rgb, cur_rgb_map);
#endif // _ZCDEBUG
    }// end for i
    // for every candidates on each pixel of the ref depth map, select the best.
    const double DIST_RANGE_EPSILON_K = 0.01/2.5/2.5;
    cout << "selecting best candidate...\n";
    for (int r = 0; r < depths[ref_depth]->rows; ++r)
    {
        for (int c = 0; c < depths[ref_depth]->cols; ++c)
        {
            const double cur_depth = double(depths[ref_depth]->at<ushort>(r, c)) * depth_image_scaling_factor;
            if (!(cur_depth > 0.0 && cur_depth <= maxCamDistance)) {
                continue;
            }
            vector<double>& cur_list = ref_depth_candidate_map[r][c];
            if (cur_list.empty()) continue;
////////////////////////////////////////////////////////////////////////
            const double DIST_RANGE_EPSILON = DIST_RANGE_EPSILON_K * cur_depth * cur_depth;
            double accept_dist_range_low = cur_depth - DIST_RANGE_EPSILON;
            double accept_dist_range_high = cur_depth + DIST_RANGE_EPSILON;
            int support_cnt = 0;
            for (int i = 0; i < cur_list.size(); ++i)
            {
                if (cur_list[i] > accept_dist_range_low &&
                cur_list[i] < accept_dist_range_high)
                {
                    support_cnt++;
                }

            }
            double support_percent = (double)(support_cnt+1) / (cur_list.size()+1);
            if ( support_percent > 0.9 && support_cnt >= 1) {
                fusioned_map->at<double>(r, c) = cur_depth;
                confidence_map->at<double>(r, c) = support_percent;
            }
        }  // end for c
    }  // end for r
    return;
}

void SimpleFusionDisparity(const vector<RectifiedCameraPair>& cam_infos,
                  const vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  double support_thresh,
                  const std::string& save_dir /*= std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug")*/)
{
    assert(cam_infos.size() > 1);
    assert(cam_infos.size() == depths.size());
    const int ref_depth = 0;
    *fusioned_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    *confidence_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    // rectified k -> rectified ref
    vector<cv::Matx33d> rotation_k2ref(cam_infos.size());
    vector<cv::Vec3d> translation_k2ref(cam_infos.size());
    vector<cv::Matx33d> rotation_ref2k(cam_infos.size());
    vector<cv::Vec3d> translation_ref2k(cam_infos.size());
    // store all transform matrices
    const Matx33d& rectify_ref = cam_infos[ref_depth].ReferenceRectifyMat();
    Matx33d ref_rotation;
    Vec3d ref_translation;
    cam_infos[ref_depth].ReferenceCameraPose(&ref_rotation, &ref_translation);
    rotation_k2ref[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    rotation_ref2k[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    for (int i = 1; i < cam_infos.size(); ++i)
    {
        Matx33d k_rotation;
        Vec3d k_translation;
        cam_infos[i].ReferenceCameraPose(&k_rotation, &k_translation);
        Matx33d rectify_k = cam_infos[i].ReferenceRectifyMat();
        Matx33d rotation_k_ref = rectify_ref.t() * ref_rotation.t() * k_rotation * rectify_k;
        Vec3d translation_k_ref = rectify_ref.t() * ref_rotation.t() * (k_translation - ref_translation);
        rotation_k2ref[i] = rotation_k_ref;
        translation_k2ref[i] = translation_k_ref;
        rotation_ref2k[i] = rotation_k_ref.t();
        translation_ref2k[i] = -rotation_k_ref.t() * translation_k_ref;
    }
    // generate all depth candidates
    cout << "generate all depth candidates.. ";
    const RectifiedCameraPair& ref_cam_info = cam_infos[ref_depth];
    const double depth_image_scaling_factor = ref_cam_info.DepthImageScaling();
    vector<vector<vector<double> > > ref_depth_candidate_map(depths[ref_depth]->rows);
    vector<vector<vector<ushort> > > ref_depth_index_map(depths[ref_depth]->rows);
    vector<vector<bool> > ref_depth_indicator_map(depths[ref_depth]->rows);
    for (int r = 0; r < ref_depth_candidate_map.size(); ++r)
    {
        ref_depth_candidate_map[r].resize(depths[ref_depth]->cols);
        ref_depth_index_map[r].resize(depths[ref_depth]->cols);
        ref_depth_indicator_map[r].resize(depths[ref_depth]->cols, false);
    }
    for (int i = 1; i < cam_infos.size(); ++i)
    {
#ifdef _ZCDEBUG
        cv::Mat cur_depth_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_16UC1);
        cv::Mat cur_rgb_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_8UC3);
        cv::Mat image = cv::imread(image_file_list[i]);
#endif // _ZCDEBUG
        // for each new depthmap reset the indicator map
        for (int r = 0; r < ref_depth_indicator_map.size(); ++r)
        {
            ref_depth_indicator_map[r].assign(depths[ref_depth]->cols, false);
        }
        // for all depth maps, including the reference pair, reproject to ref view
        const RectifiedCameraPair& cur_cam_info = cam_infos[i];
        for (int r = 0; r < depths[i]->rows; ++r)
        {
            for (int c = 0; c < depths[i]->cols; ++c)
            {
                unsigned short cur_int_depth = depths[i]->at<unsigned short>(r, c);
                double cur_true_depth = (double)cur_int_depth * depth_image_scaling_factor;
                if (cur_true_depth <= 0.0 + 1e-10 || cur_true_depth > maxCamDistance)
                {
                    continue;
                }
                cv::Vec3d k_rectified_cam_3d_pt =
                    cur_cam_info.RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(c, r, cur_true_depth);
                cv::Vec3d ref_rectified_cam_3d_pt = rotation_k2ref[i] * k_rectified_cam_3d_pt + translation_k2ref[i];
                cv::Point2i ref_rectified_im_pt;
                ref_cam_info.RectifiedCoordPointToImageCoord(ref_rectified_cam_3d_pt[0],
                        ref_rectified_cam_3d_pt[1],
                        ref_rectified_cam_3d_pt[2],
                        &ref_rectified_im_pt.x,
                        &ref_rectified_im_pt.y);
                double ref_true_depth = cv::norm(ref_rectified_cam_3d_pt);
                if ( 0 <= ref_rectified_im_pt.x && ref_rectified_im_pt.x < depths[ref_depth]->cols &&
                        0 <= ref_rectified_im_pt.y && ref_rectified_im_pt.y < depths[ref_depth]->rows &&
                        ref_true_depth <= maxCamDistance )
                {
                    if (ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] == false)
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(ref_true_depth);
                        ref_depth_index_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(static_cast<ushort>(i));
                        ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] = true;
                    }
                    else
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back()
                            = std::min(ref_true_depth, ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back());
                    }
                    assert(ref_depth_index_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].size() == ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].size());
#ifdef _ZCDEBUG
                    cur_depth_map.at<ushort>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = ushort(ref_true_depth/30.0*65535);
                    cv::Vec3b cur_color = image.at<cv::Vec3b>(r, c);
                    cur_rgb_map.at<cv::Vec3b>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = cur_color;

#endif // _ZCDEBUG

                }  //end if
            }  //end for c
        }  // end for r
#ifdef _ZCDEBUG
        fs::path fbasename = fs::path(param_file_list[i]).stem();
        std::string save_path = save_dir + "/" + fbasename.string() + "-warped.png";
        imwrite(save_path, cur_depth_map);
        std::string save_path_orig = save_dir + "/" + fbasename.string() + "-orig.png";
        imwrite(save_path_orig, *(depths[i]));
        std::string save_path_rgb = save_dir + "/" + fbasename.string() + "-rgb.png";
        imwrite(save_path_rgb, cur_rgb_map);
#endif // _ZCDEBUG
    }// end for i
    // for every candidates on each pixel of the ref depth map, select the best.
    const double DISP_ERROR_EPSILON = M_PI/ depths[ref_depth]->cols;  // one pixel matching error.
    const int col_start = 180;
    const int col_end = depths[ref_depth]->cols - col_start;
    cout << "selecting best candidate...\n";
    for (int r = 0; r < depths[ref_depth]->rows; ++r)
    {
        for (int c = col_start; c < col_end; ++c)
        {
            const double cur_depth = double(depths[ref_depth]->at<ushort>(r, c)) * depth_image_scaling_factor;
            if (!(cur_depth > 0.0 && cur_depth <= maxCamDistance)) {
                continue;
            }
            vector<double>& cur_list = ref_depth_candidate_map[r][c];
            if (cur_list.empty()) continue;
            double ref_angular_disp = cam_infos[ref_depth].DepthToAngularDisparity(c, cur_depth);
////////////////////////////////////////////////////////////////////////
            int support_cnt = 0;
            int occlusion_cnt = 0;
            int free_violation_cnt = 0;
            cv::Vec3d rect_ref3d_pt =
                        ref_cam_info.RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(c, r, cur_depth);
            double accept_disp_range_low = ref_angular_disp - DISP_ERROR_EPSILON;
            double accept_disp_range_high = ref_angular_disp + DISP_ERROR_EPSILON;
            for (int i = 0; i < cur_list.size(); ++i)
            {
                //bool ref_supported = false;
                //bool q_supported = false;
                double cur_angular_disp = cam_infos[ref_depth].DepthToAngularDisparity(c, cur_list[i]);
                if (cur_angular_disp > accept_disp_range_high)
                {
                    // too close
                    occlusion_cnt++;
                }
                else if (cur_angular_disp >= accept_disp_range_low &&
                cur_angular_disp <= accept_disp_range_high)
                {
                    support_cnt++;
                    //ref_supported = true;
                }
                /////////////////////////////////////////////////////////////////////
                int q = ref_depth_index_map[r][c][i];  // corresponding depth map
                cv::Vec3d q_rectified_3d_pt =
                rotation_ref2k[q] * rect_ref3d_pt + translation_ref2k[q];
                cv::Point2i q_rect_im_pt;
                cam_infos[q].RectifiedCoordPointToImageCoord(q_rectified_3d_pt[0],
                q_rectified_3d_pt[1],
                q_rectified_3d_pt[2],
                &q_rect_im_pt.x,
                &q_rect_im_pt.y);
                double ref2q_depth = cv::norm(q_rectified_3d_pt);
                if ( 0 <= q_rect_im_pt.x && q_rect_im_pt.x < depths[q]->cols &&
                0 <= q_rect_im_pt.y && q_rect_im_pt.y < depths[q]->rows )
                {
                    double q_depth = double(depths[q]->at<unsigned short>(q_rect_im_pt.y, q_rect_im_pt.x)) * cam_infos[q].DepthImageScaling();
                    double ref2q_angular_disp = cam_infos[q].DepthToAngularDisparity(q_rect_im_pt.x, ref2q_depth);
                    double q_angular_disp = cam_infos[q].DepthToAngularDisparity(q_rect_im_pt.x, q_depth);
                    if (q_angular_disp < ref2q_angular_disp - DISP_ERROR_EPSILON)
                    {
                        free_violation_cnt++;
                    }
                    else if (std::abs(q_angular_disp - ref2q_angular_disp) <= DISP_ERROR_EPSILON)
                    {
                        //q_supported = true;
                        support_cnt++;
                    }
                }
            }
            double confidence_val = ((double)(support_cnt) - (occlusion_cnt + free_violation_cnt) ) / ((depths.size()-1)*2.0);
            if ( confidence_val > support_thresh && support_cnt >= 4) {
                fusioned_map->at<double>(r, c) = cur_depth;
                assert(confidence_val >= 0);
                confidence_map->at<double>(r, c) = confidence_val;
            }
////////////////////////////////////////////////////////////////////////
        }  // end for c
    }  // end for r
    return;
}

void SimpleFusionDisparity_onlyrefsupport(const vector<RectifiedCameraPair>& cam_infos,
                  const vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  double support_thresh,
                  const std::string& save_dir /*= std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug")*/)
{
    assert(cam_infos.size() > 1);
    assert(cam_infos.size() == depths.size());
    const int ref_depth = 0;
    *fusioned_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    *confidence_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    // rectified k -> rectified ref
    vector<cv::Matx33d> rotation_k2ref(cam_infos.size());
    vector<cv::Vec3d> translation_k2ref(cam_infos.size());
    vector<cv::Matx33d> rotation_ref2k(cam_infos.size());
    vector<cv::Vec3d> translation_ref2k(cam_infos.size());
    // store all transform matrices
    const Matx33d& rectify_ref = cam_infos[ref_depth].ReferenceRectifyMat();
    Matx33d ref_rotation;
    Vec3d ref_translation;
    cam_infos[ref_depth].ReferenceCameraPose(&ref_rotation, &ref_translation);
    rotation_k2ref[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    rotation_ref2k[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    for (int i = 1; i < cam_infos.size(); ++i)
    {
        Matx33d k_rotation;
        Vec3d k_translation;
        cam_infos[i].ReferenceCameraPose(&k_rotation, &k_translation);
        Matx33d rectify_k = cam_infos[i].ReferenceRectifyMat();
        Matx33d rotation_k_ref = rectify_ref.t() * ref_rotation.t() * k_rotation * rectify_k;
        Vec3d translation_k_ref = rectify_ref.t() * ref_rotation.t() * (k_translation - ref_translation);
        rotation_k2ref[i] = rotation_k_ref;
        translation_k2ref[i] = translation_k_ref;
        rotation_ref2k[i] = rotation_k_ref.t();
        translation_ref2k[i] = -rotation_k_ref.t() * translation_k_ref;
    }
    // generate all depth candidates
    cout << "generate all depth candidates.. ";
    const RectifiedCameraPair& ref_cam_info = cam_infos[ref_depth];
    const double depth_image_scaling_factor = ref_cam_info.DepthImageScaling();
    vector<vector<vector<double> > > ref_depth_candidate_map(depths[ref_depth]->rows);
    vector<vector<vector<ushort> > > ref_depth_index_map(depths[ref_depth]->rows);
    vector<vector<bool> > ref_depth_indicator_map(depths[ref_depth]->rows);
    for (int r = 0; r < ref_depth_candidate_map.size(); ++r)
    {
        ref_depth_candidate_map[r].resize(depths[ref_depth]->cols);
        ref_depth_index_map[r].resize(depths[ref_depth]->cols);
        ref_depth_indicator_map[r].resize(depths[ref_depth]->cols, false);
    }
    for (int i = 1; i < cam_infos.size(); ++i)
    {
#ifdef _ZCDEBUG
        cv::Mat cur_depth_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_16UC1);
        cv::Mat cur_rgb_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_8UC3);
        cv::Mat image = cv::imread(image_file_list[i]);
#endif // _ZCDEBUG
        // for each new depthmap reset the indicator map
        for (int r = 0; r < ref_depth_indicator_map.size(); ++r)
        {
            ref_depth_indicator_map[r].assign(depths[ref_depth]->cols, false);
        }
        // for all depth maps, including the reference pair, reproject to ref view
        const RectifiedCameraPair& cur_cam_info = cam_infos[i];
        for (int r = 0; r < depths[i]->rows; ++r)
        {
            for (int c = 0; c < depths[i]->cols; ++c)
            {
                unsigned short cur_int_depth = depths[i]->at<unsigned short>(r, c);
                double cur_true_depth = (double)cur_int_depth * depth_image_scaling_factor;
                if (cur_true_depth <= 0.0 + 1e-10 || cur_true_depth > maxCamDistance)
                {
                    continue;
                }
                cv::Vec3d k_rectified_cam_3d_pt =
                    cur_cam_info.RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(c, r, cur_true_depth);
                cv::Vec3d ref_rectified_cam_3d_pt = rotation_k2ref[i] * k_rectified_cam_3d_pt + translation_k2ref[i];
                cv::Point2i ref_rectified_im_pt;
                ref_cam_info.RectifiedCoordPointToImageCoord(ref_rectified_cam_3d_pt[0],
                        ref_rectified_cam_3d_pt[1],
                        ref_rectified_cam_3d_pt[2],
                        &ref_rectified_im_pt.x,
                        &ref_rectified_im_pt.y);
                double ref_true_depth = cv::norm(ref_rectified_cam_3d_pt);
                if ( 0 <= ref_rectified_im_pt.x && ref_rectified_im_pt.x < depths[ref_depth]->cols &&
                        0 <= ref_rectified_im_pt.y && ref_rectified_im_pt.y < depths[ref_depth]->rows &&
                        ref_true_depth <= maxCamDistance )
                {
                    if (ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] == false)
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(ref_true_depth);
                        ref_depth_index_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(static_cast<ushort>(i));
                        ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] = true;
                    }
                    else
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back()
                            = std::min(ref_true_depth, ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back());
                    }
                    assert(ref_depth_index_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].size() == ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].size());
#ifdef _ZCDEBUG
                    cur_depth_map.at<ushort>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = ushort(ref_true_depth/30.0*65535);
                    cv::Vec3b cur_color = image.at<cv::Vec3b>(r, c);
                    cur_rgb_map.at<cv::Vec3b>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = cur_color;

#endif // _ZCDEBUG

                }  //end if
            }  //end for c
        }  // end for r
#ifdef _ZCDEBUG
        fs::path fbasename = fs::path(param_file_list[i]).stem();
        std::string save_path = save_dir + "/" + fbasename.string() + "-warped.png";
        imwrite(save_path, cur_depth_map);
        std::string save_path_orig = save_dir + "/" + fbasename.string() + "-orig.png";
        imwrite(save_path_orig, *(depths[i]));
        std::string save_path_rgb = save_dir + "/" + fbasename.string() + "-rgb.png";
        imwrite(save_path_rgb, cur_rgb_map);
#endif // _ZCDEBUG
    }// end for i
    // for every candidates on each pixel of the ref depth map, select the best.
    const double DISP_ERROR_EPSILON = M_PI/ depths[ref_depth]->cols*2;  // one pixel matching error.
    const int col_start = 180;
    const int col_end = depths[ref_depth]->cols - col_start;
    cout << "selecting best candidate...\n";
    for (int r = 0; r < depths[ref_depth]->rows; ++r)
    {
        for (int c = col_start; c < col_end; ++c)
        {
            const double cur_depth = double(depths[ref_depth]->at<ushort>(r, c)) * depth_image_scaling_factor;
            if (!(cur_depth > 0.0 && cur_depth <= maxCamDistance)) {
                continue;
            }
            vector<double>& cur_list = ref_depth_candidate_map[r][c];
            if (cur_list.empty()) continue;
            double ref_angular_disp = cam_infos[ref_depth].DepthToAngularDisparity(c, cur_depth);
////////////////////////////////////////////////////////////////////////
            int support_cnt = 0;
            double accept_disp_range_low = ref_angular_disp - DISP_ERROR_EPSILON;
            double accept_disp_range_high = ref_angular_disp + DISP_ERROR_EPSILON;
            for (int i = 0; i < cur_list.size(); ++i)
            {
                double cur_angular_disp = cam_infos[ref_depth].DepthToAngularDisparity(c, cur_list[i]);

                if (cur_angular_disp >= accept_disp_range_low &&
                cur_angular_disp <= accept_disp_range_high)
                {
                    support_cnt++;
                    //ref_supported = true;
                }
            }
            double confidence_val = (double)(support_cnt+1) / (depths.size());
            if ( confidence_val > support_thresh && support_cnt >= 2) {
                fusioned_map->at<double>(r, c) = cur_depth;
                assert(confidence_val >= 0);
                confidence_map->at<double>(r, c) = confidence_val;
            }
        }  // end for c
    }  // end for r
    return;
}

void Fusion_SphericalError(const vector<RectifiedCameraPair>& cam_infos,
                  const vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  double support_thresh,
                  const std::string& save_dir /*= std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug")*/)
{
    assert(cam_infos.size() > 1);
    assert(cam_infos.size() == depths.size());
    const int ref_depth = 0;
    *fusioned_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    *confidence_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_64FC1);
    // rectified k -> rectified ref
    vector<cv::Matx33d> rotation_k2ref(cam_infos.size());
    vector<cv::Vec3d> translation_k2ref(cam_infos.size());
    vector<cv::Matx33d> rotation_ref2k(cam_infos.size());
    vector<cv::Vec3d> translation_ref2k(cam_infos.size());
    // store all transform matrices
    const Matx33d& rectify_ref = cam_infos[ref_depth].ReferenceRectifyMat();
    Matx33d ref_rotation;
    Vec3d ref_translation;
    cam_infos[ref_depth].ReferenceCameraPose(&ref_rotation, &ref_translation);
    rotation_k2ref[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    rotation_ref2k[ref_depth] = cv::Matx33d::eye();
    translation_k2ref[ref_depth] = cv::Vec3d(0,0,0);
    for (int i = 1; i < cam_infos.size(); ++i)
    {
        Matx33d k_rotation;
        Vec3d k_translation;
        cam_infos[i].ReferenceCameraPose(&k_rotation, &k_translation);
        Matx33d rectify_k = cam_infos[i].ReferenceRectifyMat();
        Matx33d rotation_k_ref = rectify_ref.t() * ref_rotation.t() * k_rotation * rectify_k;
        Vec3d translation_k_ref = rectify_ref.t() * ref_rotation.t() * (k_translation - ref_translation);
        rotation_k2ref[i] = rotation_k_ref;
        translation_k2ref[i] = translation_k_ref;
        rotation_ref2k[i] = rotation_k_ref.t();
        translation_ref2k[i] = -rotation_k_ref.t() * translation_k_ref;
    }
    // generate all depth candidates
    cout << "generate all depth candidates.. ";
    const RectifiedCameraPair& ref_cam_info = cam_infos[ref_depth];
    const double depth_image_scaling_factor = ref_cam_info.DepthImageScaling();
    vector<vector<vector<double> > > ref_depth_candidate_map(depths[ref_depth]->rows);
    vector<vector<vector<ushort> > > ref_depth_index_map(depths[ref_depth]->rows);
    vector<vector<bool> > ref_depth_indicator_map(depths[ref_depth]->rows);
    for (int r = 0; r < ref_depth_candidate_map.size(); ++r)
    {
        ref_depth_candidate_map[r].resize(depths[ref_depth]->cols);
        ref_depth_index_map[r].resize(depths[ref_depth]->cols);
        ref_depth_indicator_map[r].resize(depths[ref_depth]->cols, false);
    }
    for (int i = 1; i < cam_infos.size(); ++i)
    {
#ifdef _ZCDEBUG
        cv::Mat cur_depth_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_16UC1);
        cv::Mat cur_rgb_map = cv::Mat::zeros(depths[ref_depth]->rows, depths[ref_depth]->cols, CV_8UC3);
        cv::Mat image = cv::imread(image_file_list[i]);
#endif // _ZCDEBUG
        // for each new depthmap reset the indicator map
        for (int r = 0; r < ref_depth_indicator_map.size(); ++r)
        {
            ref_depth_indicator_map[r].assign(depths[ref_depth]->cols, false);
        }
        // for all depth maps, including the reference pair, reproject to ref view
        const RectifiedCameraPair& cur_cam_info = cam_infos[i];
        for (int r = 0; r < depths[i]->rows; ++r)
        {
            for (int c = 0; c < depths[i]->cols; ++c)
            {
                unsigned short cur_int_depth = depths[i]->at<unsigned short>(r, c);
                double cur_true_depth = (double)cur_int_depth * depth_image_scaling_factor;
                if (cur_true_depth <= 0.0 + 1e-10 || cur_true_depth > maxCamDistance)
                {
                    continue;
                }
                cv::Vec3d k_rectified_cam_3d_pt =
                    cur_cam_info.RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(c, r, cur_true_depth);
                cv::Vec3d ref_rectified_cam_3d_pt = rotation_k2ref[i] * k_rectified_cam_3d_pt + translation_k2ref[i];
                cv::Point2i ref_rectified_im_pt;
                ref_cam_info.RectifiedCoordPointToImageCoord(ref_rectified_cam_3d_pt[0],
                        ref_rectified_cam_3d_pt[1],
                        ref_rectified_cam_3d_pt[2],
                        &ref_rectified_im_pt.x,
                        &ref_rectified_im_pt.y);
                double ref_true_depth = cv::norm(ref_rectified_cam_3d_pt);
                if ( 0 <= ref_rectified_im_pt.x && ref_rectified_im_pt.x < depths[ref_depth]->cols &&
                        0 <= ref_rectified_im_pt.y && ref_rectified_im_pt.y < depths[ref_depth]->rows &&
                        ref_true_depth <= maxCamDistance )
                {
                    if (ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] == false)
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(ref_true_depth);
                        ref_depth_index_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].push_back(static_cast<ushort>(i));
                        ref_depth_indicator_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x] = true;
                    }
                    else
                    {
                        ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back()
                            = std::min(ref_true_depth, ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].back());
                    }
                    assert(ref_depth_index_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].size() == ref_depth_candidate_map[ref_rectified_im_pt.y][ref_rectified_im_pt.x].size());
#ifdef _ZCDEBUG
                    cur_depth_map.at<ushort>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = ushort(ref_true_depth/30.0*65535);
                    cv::Vec3b cur_color = image.at<cv::Vec3b>(r, c);
                    cur_rgb_map.at<cv::Vec3b>(ref_rectified_im_pt.y, ref_rectified_im_pt.x) = cur_color;

#endif // _ZCDEBUG

                }  //end if
            }  //end for c
        }  // end for r
#ifdef _ZCDEBUG
        fs::path fbasename = fs::path(param_file_list[i]).stem();
        std::string save_path = save_dir + "/" + fbasename.string() + "-warped.png";
        imwrite(save_path, cur_depth_map);
        std::string save_path_orig = save_dir + "/" + fbasename.string() + "-orig.png";
        imwrite(save_path_orig, *(depths[i]));
        std::string save_path_rgb = save_dir + "/" + fbasename.string() + "-rgb.png";
        imwrite(save_path_rgb, cur_rgb_map);
#endif // _ZCDEBUG
    }// end for i
    // for every candidates on each pixel of the ref depth map, select the best.
    const double DISP_ERROR_EPSILON = M_PI/ depths[ref_depth]->cols;  // one pixel matching error.
    const int col_start = 5;
    const int col_end = depths[ref_depth]->cols - col_start;
    cout << "selecting best candidate...spherical\n";
    for (int r = 0; r < depths[ref_depth]->rows; ++r)
    {
        for (int c = col_start; c < col_end; ++c)
        {
            const double cur_depth = double(depths[ref_depth]->at<ushort>(r, c)) * depth_image_scaling_factor;
            if (!(cur_depth > 0.0 && cur_depth <= maxCamDistance)) {
                continue;
            }
            vector<double>& cur_list = ref_depth_candidate_map[r][c];
            if (cur_list.empty()) continue;
            double ref_angular_disp = cam_infos[ref_depth].DepthToAngularDisparity(c, cur_depth);
////////////////////////////////////////////////////////////////////////
            int support_cnt = 0;
            double accept_disp_range_low = ref_angular_disp - DISP_ERROR_EPSILON;
            double accept_disp_range_high = ref_angular_disp + DISP_ERROR_EPSILON;
            for (int i = 0; i < cur_list.size(); ++i)
            {
                double cur_angular_disp = cam_infos[ref_depth].DepthToAngularDisparity(c, cur_list[i]);
                if (cur_angular_disp >= accept_disp_range_low &&
                cur_angular_disp <= accept_disp_range_high)
                {
                    support_cnt++;
                }
            }
            double confidence_val = (double)(support_cnt+1) / (depths.size());  // [0,1], support count normalized
            double error_value = cam_infos[ref_depth].DepthErrorWithDisparity(c, cur_depth, M_PI/float(depths[0]->cols));
            double confidence_val_new = std::exp(-2*error_value);

            if ( confidence_val > support_thresh && support_cnt >= 2) {
                fusioned_map->at<double>(r, c) = cur_depth;
                assert(confidence_val_new >= 0);
                confidence_map->at<double>(r, c) = confidence_val_new /** confidence_val*/;
            }
////////////////////////////////////////////////////////////////////////
        }  // end for c
    }  // end for r
    return;
}
