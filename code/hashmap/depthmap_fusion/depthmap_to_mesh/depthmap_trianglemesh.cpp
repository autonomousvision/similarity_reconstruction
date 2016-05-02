#include "depthmap_trianglemesh.h"
#include "depthmap.h"

const int cpu_tsdf::DepthMapTriangleMesh::TriangleIndices[4][3] =
{
    { 0, 2, 1 }, { 0, 3, 1 }, { 0, 2, 3 }, { 1, 2, 3 }
};

cpu_tsdf::DepthMapTriangleMesh::DepthMapTriangleMesh(const cv::Mat& depthmap,
        const RectifiedCameraPair& cam_info, double dd_factor)
{
    DepthMapTriangulate(depthmap,
            cam_info,
            dd_factor,
            &point_cloud_,
            &triangles_,
            &triangle_indices_,
            &triangle_type_);
}

bool cpu_tsdf::DepthMapTriangleMesh::SearchHittingTriangle(const cv::Vec3d voxel_point, 
        const cv::Mat& depthmap, const RectifiedCameraPair& cam_info, cv::Vec3d* tri_pts)
{
    const int w = depthmap.cols;
    const int h = depthmap.rows;

    float length;
    float cur_imx, cur_imy;
    if(!cam_info.Voxel3DPointToImageCoord(voxel_point, &cur_imx, &cur_imy, &length))
    {
        return false;
    }
    int base_imx, base_imy;
    base_imx = (int)cur_imx;
    base_imy = (int)cur_imy;

    char cur_type = triangle_type_[base_imy * (w-1) + base_imx];
    int cur_first_tri_index = triangle_indices_[base_imy * (w-1) + base_imx];
    if (cur_type == 0 || cur_first_tri_index == -1) return false;

    int point_indebase_imx[4] = { base_imy * w + base_imx, base_imy * w + base_imx + 1, 
        (base_imy+1) * w + base_imx, (base_imy+1) * w + base_imx + 1 };
    double depths[4] = { depthmap.at<unsigned short>(base_imy, base_imx) * cam_info.DepthImageScaling(),
        depthmap.at<unsigned short>(base_imy, base_imx+1) * cam_info.DepthImageScaling(),
        depthmap.at<unsigned short>(base_imy+1, base_imx) * cam_info.DepthImageScaling(),
        depthmap.at<unsigned short>(base_imy+1, base_imx+1) * cam_info.DepthImageScaling() };
    float local_x, local_y;
    local_x = cur_imx - base_imx;
    local_y = cur_imy - base_imy;
    /* Possible triangles, vertex indices relative to 2x2 block. */
    // 0--1
    // |  |
    // 2--3
    const int tris[4][3] = {
        { 0, 2, 1 }, { 0, 3, 1 }, { 0, 2, 3 }, { 1, 2, 3 }
    };
    int cur_tri_index = -1;
    if (cur_type&2/*LEFT_CUT*/)
    {
        cur_tri_index = local_x > local_y ? 2 : 3;
    }
    else
    {
        cur_tri_index = (1 - local_x) > local_y ? 1 : 4;
    }


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
    float observed_length = depth.at<ushort>(cur_imy, cur_imx) * cam_info.DepthImageScaling();
    float w_inc = confidence.at<ushort>(cur_imy, cur_imx) / 65535.0;
    if(observed_length == 0.0 || w_inc == 0.0) continue;
    float d_inc = observed_length - length;
    cv::Vec3b cur_color = image.at<cv::Vec3b>(cur_imy, cur_imx);
    cv::Vec3b rgb_color(cur_color[2], cur_color[1], cur_color[0]);
    voxel_hash_map_.AddObservation(bdata, offset, d_inc, w_inc, rgb_color,
            max_dist_pos_,
            max_dist_neg_);
}

//class DepthMapTriangleMesh
//{
//  public:
//      DepthMapTriangleMesh(const cv::Mat& depthmap,
//              const RectifiedCameraPair& cam_info);
//      bool SearchHittingTriangle(const cv::Vec3d voxel_point, const RectifiedCameraPair& cam_info, cv::Vec3d* tri_pts);

//  private:
//      //enum TriangleIndextype {
//      //    INVALID = -1,
//      //    LEFT_CUT = 0,
//      //    RIGHT_CUT = 1
//      //}  // LEFT_CUT: top-left to bottom right. RIGHT_CUT: the other way
//      pcl::PointCloud<PointXYZ> point_cloud_;
//      std::vector<pcl::Vertices> triangles;
//      std::vector<int> triangle_indices_;  // -1 invalid value
//      std::vector<char> triangle_type_;  // 0 invalid value
//      // 0--1
//      // |  |
//      // 2--3
//      static const int TriangleIndices[4][3];
//};
