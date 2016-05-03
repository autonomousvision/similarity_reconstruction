/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Eigen>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <math.h>
#include "common/utilities/eigen_utility.h"


#include <opencv2/opencv.hpp>

class RectifiedCameraPair
{
    friend std::ostream& operator <<(std::ostream &os, const RectifiedCameraPair &params);

    // camera pair parameters:
    // intrinsic: \theta_1,2,n \phi_1,2,n (same for both cameras), R_1,2 (rectification rotation),
    // extrinsic: P1, P2.
public:
    RectifiedCameraPair();
    RectifiedCameraPair(double theta1, double theta2, double theta_n,
                        double phi1, double phi2, double phi_n,
                        const cv::Matx33d& R1, const cv::Matx33d& R2,
                        const cv::Matx34d& P1, const cv::Matx34d& P2);
    void SetIntrinsic(double theta1, double theta2, double theta_n,
                      double phi1, double phi2, double phi_n,
                      const cv::Matx33d& R1, const cv::Matx33d& R2);
    void SetExtrinsicPair(const cv::Matx34d& P1, const cv::Matx34d& P2);
    void GetExtrinsicPair(cv::Matx34d* P1, cv::Matx34d* P2) { *P1 = P1_; *P2 = P2_; }
    void InitializeBackProjectionBuffers();
    //void SetRelativeMotion(const cv::Matx34d& Tr);
    void SetCameraParameters(double theta1, double theta2, double theta_n,
                             double phi1, double phi2, double phi_n,
                             const cv::Matx33d& R1, const cv::Matx33d& R2,
                             const cv::Matx34d& P1, const cv::Matx34d& P2);
    void SetVoxelScalingParameters(const cv::Vec3d& volume_offset, double voxel_scale_factor, double depth_image_scaling_factor, double max_cam_distance);


    void ClearBackProjectionBuffers();

    inline cv::Vec3d RectifiedImagePointToUnrectifiedCamera3DPoint(int x, int y, unsigned short depth) const {
        // assert buffer has been initialized
        assert(im_width_ > 0 && im_height_ > 0);
        assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
        assert(!sin_thetas_.empty());
        assert(!cos_phis_.empty());
        assert(depth_image_scaling_ > 0);
        cv::Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
                           sin_thetas_[x]*sin_phis_[y],
                           cos_thetas_[x]);
        return (R1_*sphere_point) * ((double)depth * depth_image_scaling_);
    }

    inline cv::Vec3d RectifiedImagePointToRectifiedCamera3DPoint(int x, int y, unsigned short depth) const {
        // assert buffer has been initialized
        assert(im_width_ > 0 && im_height_ > 0);
        assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
        assert(!sin_thetas_.empty());
        assert(!cos_phis_.empty());
        assert(depth_image_scaling_ > 0);
        cv::Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
                           sin_thetas_[x]*sin_phis_[y],
                           cos_thetas_[x]);
        return (sphere_point) * ((double)depth * depth_image_scaling_);
    }

    inline cv::Vec3d RectifiedImagePointToRectifiedCam3DPointNoDepthScaling(int x, int y, double depth) const {
        // assert buffer has been initialized
        assert(im_width_ > 0 && im_height_ > 0);
        assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
        assert(!sin_thetas_.empty());
        assert(!cos_phis_.empty());
        assert(depth_image_scaling_ > 0);
        cv::Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
                           sin_thetas_[x]*sin_phis_[y],
                           cos_thetas_[x]);
        return (sphere_point) * ((double)depth);
    }

    inline cv::Vec3d RectifiedImagePointToVoxel3DPoint(int x, int y, unsigned short depth) const {
        assert(im_width_ > 0 && im_height_ > 0);
        assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
        assert(!sin_thetas_.empty());
        assert(!cos_phis_.empty());
        assert(depth_image_scaling_ > 0);
        cv::Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
                           sin_thetas_[x]*sin_phis_[y],
                           cos_thetas_[x]);
       return static_cast<cv::Vec3d>(ext_R1_trans_rectify_scaled_ * sphere_point * ((double)depth * depth_image_scaling_)
                                  + ext_t1_inv_offseted_scaled_);
    }

    inline cv::Vec3d RectifiedImagePointToVoxel3DPointDouble(double x, double y, unsigned short depth) const {
        assert(im_width_ > 0 && im_height_ > 0);
        assert(0 <= x && x < im_width_ && 0 <= y && y < im_height_);
        assert(!sin_thetas_.empty());
        assert(!cos_phis_.empty());
        assert(depth_image_scaling_ > 0);

//        std::cout << "double im coord"<<std::endl;
//        std::cout << "im x " << x << std::endl;
//        std::cout << "im y " << y << std::endl;

        double theta = theta1_ + delta_theta_ * x;
        double phi = phi1_ + delta_phi_ * y;
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);

        cv::Vec3d sphere_point(
                    sin_theta * cos_phi,
                    sin_theta * sin_phi,
                    cos_theta);
//        cv::Vec3d sphere_point(sin_thetas_[x]*cos_phis_[y],
//                           sin_thetas_[x]*sin_phis_[y],
//                           cos_thetas_[x]);
       return static_cast<cv::Vec3d>(ext_R1_trans_rectify_scaled_ * sphere_point * ((double)depth * depth_image_scaling_)
                                  + ext_t1_inv_offseted_scaled_);
    }

    inline cv::Vec3d LocalCameraOffsetToWorldOrigin() const { return cv::Vec3d(P1_(0, 3), P1_(1, 3), P1_(2, 3)); }

    inline cv::Vec3d Voxel3DPointToLocalRectifiedCoord(const cv::Vec3d& voxel_point) const {
        return rectify_trans_ext_R1_scaled_ * voxel_point + rectify_inv_ext_t1_plus_ext_R1_off_;
    }

    inline cv::Vec3d LocalRectified3DPointToVoxelCoord(const cv::Vec3d& local_rectified_pt) const {
        return ext_R1_trans_rectify_scaled_ * local_rectified_pt + ext_t1_inv_offseted_scaled_;
    }

    // the direction stored as column vector. (i.e. principal axis of the volume box)
    inline const cv::Matx33d& Voxel3DDirectionInRectifiedCoord() const { return rectify_trans_ext_R1_scaled_; }

    inline const cv::Matx34d& ReferenceCameraPose() const { return P1_; }

    inline void ReferenceCameraPose(cv::Matx33d* rotation, cv::Vec3d* translation) const {
       *rotation = cv::Matx33d(P1_(0,0), P1_(0,1), P1_(0,2),
                               P1_(1,0), P1_(1,1), P1_(1,2),
                               P1_(2,0), P1_(2,1), P1_(2,2));
       *translation = cv::Vec3d(P1_(0, 3), P1_(1, 3), P1_(2, 3));
    }

    // interface conform to update8AddLoopSSESingleInteger
    template<typename T>
    inline void RectifiedCoordPointToImageCoord(T px, T py, T pz, int* imx, int* imy) const {
        assert(delta_theta_inv_ > 0);
        using namespace std;

        T length = std::sqrt(px*px + py*py + pz*pz);
        px = px/length;
        py = py/length;
        pz = pz/length;

        T theta;
        T phi;
        theta = acos(pz);
        phi = asin(py/sin(theta));

        //*imx = (int)((theta - theta1_) * delta_theta_inv_);
        //*imy = (int)((phi - phi1_) * delta_phi_inv_);

        *imx = (int)round((theta - theta1_) * delta_theta_inv_);
        *imy = (int)round((phi - phi1_) * delta_phi_inv_);

    }

    template<typename T>
    inline void RectifiedCoordPointToImageCoord(T px, T py, T pz, float* imx, float* imy) const {
        assert(delta_theta_inv_ > 0);
        using namespace std;

        T length = std::sqrt(px*px + py*py + pz*pz);
        px = px/length;
        py = py/length;
        pz = pz/length;

        T theta;
        T phi;
        theta = acos(pz);
        phi = asin(py/sin(theta));

        *imx = ((theta - theta1_) * delta_theta_inv_);
        *imy = ((phi - phi1_) * delta_phi_inv_);
    }

    inline void RectifiedSpherePointToImageCoordArray(const float *px, const float *py, const float *pz, int* imx, int* imy) const {
        assert(delta_theta_inv_ > 0);
        using namespace std;
        static const int LENGTH = 4;
        float theta[LENGTH];
        float phi[LENGTH];
        theta[0] = acos(pz[0]);theta[1] = acos(pz[1]);theta[2] = acos(pz[2]);theta[3] = acos(pz[3]);
        phi[0] = asin(py[0]/sin(theta[0])); phi[1] = asin(py[1]/sin(theta[1]));
        phi[2] = asin(py[2]/sin(theta[2])); phi[3] = asin(py[3]/sin(theta[3]));

        imx[0] = (int)((theta[0] - theta1_) * delta_theta_inv_);
        imx[1] = (int)((theta[1] - theta1_) * delta_theta_inv_);
        imx[2] = (int)((theta[2] - theta1_) * delta_theta_inv_);
        imx[3] = (int)((theta[3] - theta1_) * delta_theta_inv_);

        imy[0] = (int)((phi[0] - phi1_) * delta_phi_inv_);
        imy[1] = (int)((phi[1] - phi1_) * delta_phi_inv_);
        imy[2] = (int)((phi[2] - phi1_) * delta_phi_inv_);
        imy[3] = (int)((phi[3] - phi1_) * delta_phi_inv_);
    }

    inline bool Voxel3DPointToImageCoord(const cv::Vec3d& voxel_point, float* imx, float* imy, float* length=NULL) const {
        cv::Vec3d rectified_local_pt = Voxel3DPointToLocalRectifiedCoord(voxel_point);
        if (rectified_local_pt[0] < 0 || cv::norm(rectified_local_pt) > max_cam_distance_) return false;
        RectifiedCoordPointToImageCoord<double>(rectified_local_pt[0], rectified_local_pt[1], rectified_local_pt[2], imx, imy);
        if (!(0<=*imx && *imx<im_width_ && 0<=*imy && *imy<im_height_)) return false;
        if (length) *length = cv::norm(rectified_local_pt);
        return true;
    }

    inline bool Voxel3DPointToImageCoord(const cv::Vec3d& voxel_point, int* imx, int* imy, float* length=NULL) const {
        cv::Vec3d rectified_local_pt = Voxel3DPointToLocalRectifiedCoord(voxel_point);
        //std::cout << "rectified_localpt " << rectified_local_pt << std::endl;
        if (rectified_local_pt[0] < 0 || cv::norm(rectified_local_pt) > max_cam_distance_) return false;
        RectifiedCoordPointToImageCoord<double>(rectified_local_pt[0], rectified_local_pt[1], rectified_local_pt[2], imx, imy);
        if (!(0<=*imx && *imx<im_width_ && 0<=*imy && *imy<im_height_)) return false;
        if (length) *length = cv::norm(rectified_local_pt);
        return true;
    }

    inline cv::Vec3d Voxel3DPointToWorldCoord(const cv::Vec3d& voxel_point) const
    {
        assert(voxel_length_>0);
        return voxel_point*voxel_length_ + voxel_world_offset_;
    }

    inline cv::Vec3d WorldCoordToVoxel3DPoint(const cv::Vec3d& world_point) const
    {
        assert(voxel_length_>0);
        return (world_point - voxel_world_offset_)*(1.0/voxel_length_);
    }

    const cv::Matx33d& ReferenceRectifyMat() const { return R1_; }
    const double DepthImageScaling() const { return depth_image_scaling_; }

    inline double DepthToAngularDisparity(int ix, double depth) const {
        double theta = theta1_ + ix * delta_theta_;
        const double d = depth;
        double angular_disp;
        if (t_ < 0)
        {
            angular_disp = M_PI_2 + std::atan( -(d + std::abs(t_)*std::cos(theta))/(std::abs(t_)*std::sin(theta)) );
        }
        else
        {
            angular_disp = M_PI_2 + std::atan( -(d + std::abs(t_)*std::cos(M_PI-theta))/(std::abs(t_)*std::sin(M_PI-theta)) );
        }
        return angular_disp;
    }

    inline double DepthErrorWithDisparity(int ix, double depth, double err_gamma) const
    {
        double theta = theta1_ + ix * delta_theta_;
        theta = t_<0? theta:M_PI-theta;
        double angular_disp = DepthToAngularDisparity(ix, depth);
        double sin_theta = std::sin(theta);
        double sin_gamma_squared = std::sin(angular_disp);
        sin_gamma_squared *= sin_gamma_squared;
        return (sin_theta/sin_gamma_squared)*std::abs(t_)*err_gamma;
    }

    bool ProjectToWorldPointCloud(const cv::Mat& depthmap, const cv::Mat& image, std::vector<cv::Vec3d>* points, std::vector<cv::Vec3b>* colors)
    {
        for (int y = 0; y < depthmap.rows; ++y)
        {
            for (int x = 0; x < depthmap.cols; ++x)
            {
                unsigned short depth_u16 = depthmap.at<unsigned short>(y, x);
                if (depth_u16 > 0)
                {
                    cv::Vec3d voxel_pt = this->RectifiedImagePointToVoxel3DPoint(x, y, depth_u16);
                    cv::Vec3d world_pt = this->Voxel3DPointToWorldCoord(voxel_pt);
                    points->push_back(world_pt);

                    cv::Vec3b cur_color = image.at<cv::Vec3b>(y, x);
                    std::swap(cur_color[0], cur_color[2]);
                    colors->push_back(cur_color);
                }
            }
        }
        return true;
    }

    inline cv::Vec3f RefCameraCenterInWorldCoord() const
    {
        return cv::Vec3f(P1_(0, 3), P1_(1, 3), P1_(2, 3));
    }

private:
    double theta1_;
    double theta2_;
    double theta_n_;
    double phi1_;
    double phi2_;
    double phi_n_;
    cv::Matx33d R1_, R2_; // 3x3
    cv::Matx34d P1_, P2_; // 3x4
    cv::Matx34d Tr_; //3x4

    cv::Matx33d ext_R1_;
    cv::Vec3d ext_t1_;

    // buffer for rectified image -> 3D
    int im_width_;
    int im_height_;
    std::vector<double> sin_thetas_;
    std::vector<double> cos_thetas_;
    std::vector<double> sin_phis_;
    std::vector<double> cos_phis_;

    // converting to voxel coord
    cv::Matx33d ext_R1_trans_rectify_scaled_; // for 2d -> 3d (rectified sphere -> unrectified sphere -> world 3D point -> voxel 3D point)
    cv::Vec3d ext_t1_inv_offseted_scaled_; // for 2d -> 3d (rectified sphere -> unrectified sphere -> world 3D point -> voxel 3D point)

    // voxel coord to camera coord
    // R1_^{-1} * (ext_R1_* ((\vec{x} \times {voxel_length}) + offset)+ ext_t1_ )
    // R1_^T * ext_R1 * {voxel_length} * \vec{x}
    // + R1_^T * ext_t1_ + R1_^T * ext_R1_ * offset (= R1_^T * (ext_t1_ + ext_R1_ * offset))
    cv::Matx33d rectify_trans_ext_R1_scaled_;  // for 3d -> 2d
    cv::Vec3d rectify_inv_ext_t1_plus_ext_R1_off_;  // for 3d -> 2d
    double depth_image_scaling_;

    double delta_theta_inv_;
    double delta_phi_inv_;
    double delta_theta_;
    double delta_phi_;

    double t_;  // for disparity -> depth conversion
    double max_cam_distance_;

    cv::Vec3d voxel_world_offset_; // voxel_pt = (world_pt - offset_)/voxel_length
    double voxel_length_;
};

bool PointVisibleCoarse(const Eigen::Vector3f& point, const RectifiedCameraPair& pair, const float dist_thresh = 30, const float angle_thresh = 0.05);

void InitializeCamInfos(int depth_width, int depth_height, const cv::Vec3d& offset, double voxel_length, double depth_image_scaling_factor, double cam_max_distance, std::vector<RectifiedCameraPair>& cam_infos);

std::istream& operator >>(std::istream &is, RectifiedCameraPair &params);
std::ostream& operator <<(std::ostream &os, const RectifiedCameraPair &params);


