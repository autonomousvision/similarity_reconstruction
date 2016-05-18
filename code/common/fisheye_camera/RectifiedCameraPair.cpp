/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "RectifiedCameraPair.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

using std::sin;
using std::cos;
using std::acos;
using std::asin;
using std::string;
using std::cout;
using std::endl;
using cv::Vec;
using cv::Vec3d;
using cv::Point2d;
using cv::Point3d;
using cv::Mat;
using cv::Matx33d;
using cv::Matx34d;

template<typename _Tp, int m, int n>
std::istream& operator >>(std::istream &is, cv::Matx<_Tp, m, n> &mat) {
    string stamp;
    is >> stamp;
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            is >> mat(r, c);
        }
    }

    //cout << "Mat: " << stamp << Mat(mat) << endl;
    return is;
}

std::istream& operator >>(std::istream &is, RectifiedCameraPair &params) {
    string stamp;
    double theta1, theta2, theta_n;
    double phi1, phi2, phi_n;
    Matx33d R1, R2;
    Matx34d P1, P2;
    Matx34d Tr;
    is >> stamp >> theta1;
    is >> stamp >> theta2;
    is >> stamp >> theta_n;
    is >> stamp >> phi1;
    is >> stamp >> phi2;
    is >> stamp >> phi_n;

//    cout << "theta1 " << theta1 << endl;
//    cout << "theta2 " << theta2 << endl;
//    cout << "thetan " << theta_n << endl;
//    cout << "phi1 " << phi1 << endl;
//    cout << "phi2 " << phi2 << endl;
//    cout << "phi3 " << phi_n << endl;

    is >> R1;
    is >> R2;
    is >> P1;
    is >> P2;
    is >> Tr;


    params.SetCameraParameters(theta1, theta2, theta_n,
                 phi1, phi2, phi_n,
                 R1, R2,
                 P1, P2);
    //cout << params << endl;
    return is;
}

std::ostream& operator <<(std::ostream &os, const RectifiedCameraPair &params) {
    os << "theta1 " << params.theta1_ << endl;
    os << "theta2 " << params.theta2_<< endl;
    os << "thetan " << params.theta_n_ << endl;
    os << "phi1 " << params.phi1_ << endl;
    os << "phi2 " << params.phi2_<< endl;
    os << "phi3 " << params.phi_n_ << endl;

    os << "R1 " << cv::Mat(params.R1_) << endl;
    os << "R2 " << cv::Mat(params.R2_)<< endl;
    os << "P1 " << cv::Mat(params.P1_)<< endl;
    os << "P2 " << cv::Mat(params.P2_) << endl;
    os << "Tr " << cv::Mat(params.Tr_) << endl;

    os << "depth scaling " << params.depth_image_scaling_ << endl;
    os << "imwidth " << params.im_width_ << endl;
    os << "imheight " << params.im_height_ << endl;

    os << "t_ " << params.t_ << endl;

    os << "ext_R1_trans_rectify_scaled_ " << cv::Mat(params.ext_R1_trans_rectify_scaled_) << endl;
    os << "ext_t1_inv_offseted_scaled_ " << cv::Mat(params.ext_t1_inv_offseted_scaled_) << endl;
    os << "max_cam_distance_ " << params.max_cam_distance_ << endl;

    os << "voxel_world offset: " << cv::Mat(params.voxel_world_offset_) << endl;
    os << "voxel_length: " << (params.voxel_length_) << endl;
    return os;
}

RectifiedCameraPair::RectifiedCameraPair()
    :theta1_(0), theta2_(0), theta_n_(0),
    phi1_(0), phi2_(0), phi_n_(0),
    im_width_(-1), im_height_(-1),
    depth_image_scaling_(0),
    delta_theta_inv_(0), delta_phi_inv_(0),
    delta_theta_(0), delta_phi_(0), t_(0),
    max_cam_distance_(0),
    voxel_world_offset_(0,0,0),
    voxel_length_(0)
{};

RectifiedCameraPair::RectifiedCameraPair(double theta1, double theta2, double theta_n,
                                         double phi1, double phi2, double phi_n,
                                         const cv::Matx33d& R1, const cv::Matx33d& R2,
                                         const cv::Matx34d& P1, const cv::Matx34d& P2)
                                         : im_width_(-1), im_height_(-1),
                                         depth_image_scaling_(0),
                                         delta_theta_inv_(0), delta_phi_inv_(0),
                                         delta_theta_(0), delta_phi_(0), t_(0),
                                         max_cam_distance_(0),
                                         voxel_world_offset_(0,0,0),
                                         voxel_length_(0)
{
    SetCameraParameters(theta1, theta2, theta_n,
                 phi1, phi2, phi_n,
                 R1, R2,
                 P1, P2);
}

void RectifiedCameraPair::SetCameraParameters(double theta1, double theta2, double theta_n,
                            double phi1, double phi2, double phi_n,
                            const cv::Matx33d& R1, const cv::Matx33d& R2,
                            const cv::Matx34d& P1, const cv::Matx34d& P2)
{
    SetIntrinsic(theta1, theta2, theta_n,
                 phi1, phi2, phi_n,
                 R1, R2);
    SetExtrinsicPair(P1, P2);
    //SetRelativeMotion(Tr);
    InitializeBackProjectionBuffers();
}

void RectifiedCameraPair::SetIntrinsic(double theta1, double theta2, double theta_n,
                                       double phi1, double phi2, double phi_n,
                                       const cv::Matx33d& R1, const cv::Matx33d& R2) {
    using std::max;
    using std::min;
    theta1_ = max<double>(theta1, 0.0);  // theta: [0,pi]
    theta2_ = min<double>(theta2, M_PI);
    theta_n_ = theta_n;
    phi1_ = max<double>(phi1, -M_PI_2);  // phi: [-pi/2,pi/2]
    phi2_ = min<double>(phi2, M_PI_2);
    phi_n_ = phi_n;
    R1_ = R1;
    R2_ = R2;
    //R1_trans_ = R1_.t();


//    cout << "theta1 " << theta1_ << endl;
//    cout << "theta2 " << theta2_ << endl;
//    cout << "thetan " << theta_n_ << endl;
//    cout << "phi1 " << phi1_ << endl;
//    cout << "phi2 " << phi2_ << endl;
//    cout << "phi3 " << phi_n_ << endl;
//    cout << "R1 " << cv::Mat(R1_) << endl;
//    cout << "R2 " << cv::Mat(R2_) << endl;
}

void RectifiedCameraPair::SetExtrinsicPair(const cv::Matx34d& P1, const cv::Matx34d& P2) {
    P1_ = P1;
    P2_ = P2;

    cv::Matx33d rotation1 = P1_.get_minor<3, 3>(0, 0);
    cv::Matx31d translation1 = (P1_.get_minor<3, 1>(0, 3));

    cv::Matx33d rotation2 = P2_.get_minor<3, 3>(0, 0);
    cv::Matx31d translation2 = (P2_.get_minor<3, 1>(0, 3));

    cv::Matx33d Tr_rotate_part = rotation2.t() * rotation1;
    cv::Matx31d Tr_trans_part = rotation2.t() * (translation1 - translation2);
    Tr_ = cv::Matx34d(Tr_rotate_part(0,0), Tr_rotate_part(0,1), Tr_rotate_part(0,2), Tr_trans_part(0, 0),
                  Tr_rotate_part(1,0), Tr_rotate_part(1,1), Tr_rotate_part(1,2), Tr_trans_part(1, 0),
                  Tr_rotate_part(2,0), Tr_rotate_part(2,1), Tr_rotate_part(2,2), Tr_trans_part(2, 0));

    // R1, t1: maps world coord to camera coord.
    ext_R1_ = Matx33d(P1_(0,0), P1_(1,0), P1_(2,0),
                     P1_(0,1), P1_(1,1), P1_(2,1),
                     P1_(0,2), P1_(1,2), P1_(2,2));  // transposed
    ext_t1_ = -ext_R1_ * Vec3d(P1_(0, 3), P1_(1, 3), P1_(2, 3));

    //ext_R1_trans_ = ext_R1_.t();
    //ext_t1_inv_ = Vec3d(P1_(0, 3), P1_(1, 3), P1_(2, 3));

//    ext_R2_ = Matx33d(P2_(0,0), P2_(1,0), P2_(2,0),
//                     P2_(0,1), P2_(1,1), P2_(2,1),
//                     P2_(0,2), P2_(1,2), P2_(2,2));  // transposed
//    ext_t2_ = -ext_R2_ * Vec3d(P2_(0, 3), P2_(1, 3), P2_(2, 3));

//    using std::cout;
//    using std::endl;
//
//    cout << "ext_R1: " << Mat(ext_R1_) << endl;
//    cout << "ext_t1: " << Mat(ext_t1_) << endl;
//    cout << "ext_R2: " << Mat(ext_R2_) << endl;
//    cout << "ext_t2: " << Mat(ext_t2_) << endl;
}

//void RectifiedCameraPair::SetRelativeMotion(const cv::Matx34d& Tr) {
//    Tr_ = Tr;
//}

void RectifiedCameraPair::SetVoxelScalingParameters(const cv::Vec3d& volume_offset/*in world coordinate*/, double voxel_length, double depth_scaling, double max_cam_distance) {
    // for 2D to 3D (local rectified coord to global voxel coord)
    double voxel_scale = 1.0/voxel_length;
    ext_R1_trans_rectify_scaled_ = ext_R1_.t() * R1_ * (voxel_scale);
    Vec3d ext_t1_inv(P1_(0, 3), P1_(1, 3), P1_(2, 3));
    ext_t1_inv_offseted_scaled_ = (ext_t1_inv - volume_offset) * (voxel_scale);

    // for 3D to 2D (voxel coord to camera local rectified coord)
    // R1_^{-1} * (ext_R1_* ((\vec{x} \times {voxel_length}) + offset)+ ext_t1_ )
    // R1_^T * ext_R1 * {voxel_length} * \vec{x}
    // + R1_^T * ext_t1_ + R1_^T * ext_R1_ * offset (= R1_^T * (ext_t1_ + ext_R1_ * offset))
    Matx33d R1_t = R1_.t();
    rectify_trans_ext_R1_scaled_ = R1_t * ext_R1_ * voxel_length;  // for 3d -> 2d
    rectify_inv_ext_t1_plus_ext_R1_off_ = R1_t * (ext_t1_ + ext_R1_ * volume_offset);  // for 3d -> 2d

    depth_image_scaling_ = depth_scaling;
    max_cam_distance_ = max_cam_distance;

    voxel_world_offset_ = volume_offset;
    voxel_length_ = voxel_length;
}

void RectifiedCameraPair::InitializeBackProjectionBuffers() {
    im_width_ = theta_n_;
    im_height_ = phi_n_;

    sin_thetas_.resize(im_width_);
    cos_thetas_.resize(im_width_);
    sin_phis_.resize(im_height_);
    cos_phis_.resize(im_height_);

    delta_theta_ = (theta2_ - theta1_)/theta_n_;
    for (int x = 0; x < im_width_; ++x) {
        double theta = theta1_ + x * delta_theta_;
        sin_thetas_[x] = sin(theta);
        cos_thetas_[x] = cos(theta);
    }

    delta_phi_ = (phi2_ - phi1_)/phi_n_;
    for (int v = 0; v < im_height_; ++v) {
        double phi = phi1_ + v * delta_phi_;
        sin_phis_[v] = sin(phi);
        cos_phis_[v] = cos(phi);
     }

     delta_theta_inv_ = 1.0/delta_theta_;
     delta_phi_inv_ = 1.0/delta_phi_;

     Matx33d rotation;
     cv::Matx31d translation;
     rotation = Tr_.get_minor<3, 3>(0, 0);
     translation = (Tr_.get_minor<3, 1>(0, 3));
     cv::Matx31d translation_rect = -R1_.t() * rotation.t() * translation;
     t_ = translation_rect(2, 0);
}

void RectifiedCameraPair::ClearBackProjectionBuffers() {
    im_width_ = im_height_ = 0;
    sin_thetas_.clear();
    cos_thetas_.clear();
    sin_phis_.clear();
    cos_phis_.clear();
}

//inline cv::Vec3d RectifiedCameraPair::RectifiedRefImagePointToCamera3DPoint(int x, int y, double depth) {
//    // assert buffer has been initialized
//    assert(im_width_ > 0 && im_height_ > 0);
//    assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
//    assert(!sin_thetas_.empty());
//    Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
//                       sin_thetas_[x]*sin_phis_[y],
//                       cos_thetas_[x]);
//    return (R1_*sphere_point) * depth;
//}
//
//inline cv::Vec3d RectifiedCameraPair::RefCamera3DPointToWorld3DPoint(const cv::Vec3d& camera_point) {
//    return (ext_R1_trans_ * (camera_point) + ext_t1_inv_);
//}
//
//inline cv::Vec3d RectifiedCameraPair::RectifiedRefImagePointToWorld3DPoint(int x, int y, double depth) {
//    assert(im_width_ > 0 && im_height_ > 0);
//    assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
//    assert(!sin_thetas_.empty());
//    Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
//                       sin_thetas_[x]*sin_phis_[y],
//                       cos_thetas_[x]);
//    return (ext_R1_trans_rectify_ * sphere_point * depth + ext_t1_inv_);
//}
//
//inline cv::Vec3i RectifiedCameraPair::RecrifiedRefImagePointToVoxel3DPoint(int x, int y, unsigned short depth) {
//    assert(im_width_ > 0 && im_height_ > 0);
//    assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
//    assert(!sin_thetas_.empty());
//    assert(depth_image_scaling_ > 0);
//    Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
//                       sin_thetas_[x]*sin_phis_[y],
//                       cos_thetas_[x]);
//    return static_cast<Vec3i>(ext_R1_trans_rectify_scaled * sphere_point * ((double)depth * depth_image_scaling)
//                                + ext_t1_inv_offseted_scaled_);
//}
//
//inline cv::Vec3d RectifiedCameraPair::World3DPointToRefCamera3DPoint(const cv::Vec3d& world_point) {
//    return ext_R1_ * world_point + ext_t1_;
//}
//
//inline cv::Vec2d RectifiedCameraPair::RefCamera3DPointToRectifiedRefImagePoint(const cv::Vec3d& camera_point) {
//    // To sphere
//    double point_length = std::sqrt(camera_point.x*camera_point.x + camera_point.y*camera_point.y + camera_point.z*camera_point.z);
//    Vec3d sphere_point = camera_point * (1.0/point_length);
//
//    // rectified coord system
//    Vec3d x_r = R1_trans_ * (sphere_point);
//
//    double theta = acos(x_r[2]);
//    double phi = asin(x_r[1]/sin(theta));
//
//    return Vec2d((theta - theta1_) * delta_theta_inv_, (phi - phi1_) * delta_phi_inv_);
//}
//
////inline cv::Vec2i Voxel3DPointToRectifiedRefImagePoint(const cv::Vec3i& voxel_point) {
////
////}


bool PointVisibleCoarse(const Eigen::Vector3f &point, const RectifiedCameraPair &pair, const float dist_thresh, const float angle_thresh)
{
    const cv::Vec3f cvpoint = utility::EigenVectorToCvVector3(point);
    const cv::Matx34d& P = pair.ReferenceCameraPose();
    const cv::Vec3f cam_pos = pair.RefCameraCenterInWorldCoord();
    float dist = cv::norm(cvpoint, cam_pos);
    if (dist > dist_thresh) return false;
    Eigen::Vector3d rect_pt = utility::CvVectorToEigenVector3(
            pair.Voxel3DPointToLocalRectifiedCoord(pair.WorldCoordToVoxel3DPoint(utility::EigenVectorToCvVector3(point)))
                );
    //if (rect_pt[0] > 0 && rect_pt.normalized().dot(Eigen::Vector3d(1, 0, 0)) > 0.3) return true;
    if (rect_pt[0] > 0 && rect_pt.normalized().dot(Eigen::Vector3d(1, 0, 0)) > angle_thresh) return true;
    return false;
//    cv::Vec3f x_axis(P(0, 0), P(1, 0), P(2, 0));
//    const cv::Vec3f center_dir = cvpoint - cam_pos;
//    float cosv = ((center_dir-x_axis)/norm(center_dir-x_axis)).dot(x_axis);
//    cout << "cosv " << cosv << endl;
//    if (cosv < angle_thresh) return false;
    // return true;
}


void InitializeCamInfos(int depth_width, int depth_height, const cv::Vec3d &offset, double voxel_length, double depth_image_scaling_factor, double cam_max_distance, std::vector<RectifiedCameraPair> &cam_infos) {
    for (int i = 0; i < cam_infos.size(); ++i) {
        cam_infos[i].SetVoxelScalingParameters(offset,
                                               voxel_length,
                                               depth_image_scaling_factor, cam_max_distance);
        //cam_infos[i].InitializeBackProjectionBuffers(depth_width, depth_height);
    }
}
