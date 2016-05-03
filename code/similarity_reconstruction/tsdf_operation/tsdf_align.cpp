/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "tsdf_align.h"

#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <cmath>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "tsdf_operation/tsdf_clean.h"
#include "common/utilities/common_utility.h"
#include "common/utilities/eigen_utility.h"
#include "../tsdf_representation/tsdf_hash.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_pca.h"
#include "tsdf_operation/tsdf_slice.h"
#include "tsdf_operation/tsdf_utility.h"

using namespace utility;
using namespace std;
using namespace cv;
namespace bfs = boost::filesystem;

////////////////////////////////////////////////////////////
// 1D Rotation, 2D Translation, 3D Scaling, Observation term
class TSDFAlignCostFunction2DRObsWeight : public ceres::SizedCostFunction<2, 1, 2, 3>
{
public:
    TSDFAlignCostFunction2DRObsWeight(
                    const cpu_tsdf::TSDFHashing* vtsdf_model,
                    const cv::Vec3d& vtemplate_point_world_coord,
                    const double vtemplate_point_val,
                    const double vtrans_z,
                    const cv::Vec3i& vdebug_voxel_coord)
            : SizedCostFunction(),
              tsdf_model(vtsdf_model),
              template_point_world_coord(vtemplate_point_world_coord),
              template_point_val(vtemplate_point_val),
              trans_z(vtrans_z),
              debug_voxel_coord(vdebug_voxel_coord)
    {}
    virtual ~TSDFAlignCostFunction2DRObsWeight() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const
    {
        // For one point in the template, get its corresponding TSDF value in the model
        // Compute the difference (we want to align template to model)
        // rotation 1, translation 2, scale 3.
        cv::Vec3d rotation_axis_angle(0, 0,  parameters[0][0]);
        cv::Vec3d translation(parameters[1][0], parameters[1][1], trans_z);
        cv::Vec3d scale3d(parameters[2][0], parameters[2][1], parameters[2][2]);
        cv::Vec3d scaled_pt(
                scale3d[0] * template_point_world_coord[0],
                scale3d[1] * template_point_world_coord[1],
                scale3d[2] * template_point_world_coord[2]);
        cv::Vec3d rotated_pt;
        cv::Vec3d translated_pt;
        ceres::AngleAxisRotatePoint(rotation_axis_angle.val, scaled_pt.val,
                                    rotated_pt.val);
        translated_pt = rotated_pt + translation;
        float model_val = -1;
        float model_weight = 0;
        const double sigma = 0.5;
        //const double weight_obs = exp(-0.5 * ((template_point_val * template_point_val) / (sigma * sigma))) ;
        const double weight_obs = 1.0;
        if (tsdf_model->RetriveDataFromWorldCoord(cv::Vec3f(translated_pt), &model_val, &model_weight) &&
                model_weight > min_model_weight)
        {
            //cout << "model_weight: " << model_weight << endl;
            residuals[0] = sqrt(model_weight) * (model_val - template_point_val);
            residuals[1] = weight_obs * sqrt_lambda_obs * sqrt(max_model_weight - model_weight);
        }
        else
        {
            residuals[0] = 0.0;
            residuals[1] = weight_obs * sqrt_lambda_obs * sqrt(max_model_weight);
        }
        // compute the jacobians
        if (jacobians)
        {
            cv::Vec3f grad;
            cv::Vec3f wgrad;
            if (model_weight < min_model_weight ||
                    !tsdf_model->RetriveGradientFromWorldCoord(cv::Vec3f(translated_pt), &grad, &wgrad))
            {
                if (jacobians[0])
                {
                    jacobians[0][0] = 0.0;
                    jacobians[0][1] = 0.0;
                }
                if (jacobians[1])
                {
                    jacobians[1][0] = jacobians[1][1]  = 0.0;
                    jacobians[1][2] = jacobians[1][3]  = 0.0;
                }
                if (jacobians[2])
                {
                    jacobians[2][0] = jacobians[2][1] = jacobians[2][2] = 0.0;
                    jacobians[2][3] = jacobians[2][4] = jacobians[2][5] = 0.0;
                }
                return true;
            }
            double w_square_root = sqrt(model_weight);
            double w_square_root_r = 1.0/sqrt(model_weight);
            Matx13d diff_grad_wrt_pointx_t = (((0.5 * w_square_root_r * ( -template_point_val + model_val )) * wgrad) + w_square_root * grad).t() ;
            double max_minus_w_square_root_r = 1.0/sqrt(max_model_weight - model_weight);
            Matx13d obs_grad_wrt_pointx_t =  - weight_obs * sqrt_lambda_obs * 0.5 * max_minus_w_square_root_r * wgrad.t() ;

            Matx33d rotation_mat;
            cv::Rodrigues(rotation_axis_angle, rotation_mat);
            if (jacobians[0])  // rotation, 1*3 mat
            {
                double theta = parameters[0][0];
                double sin_theta = sin(theta);
                double cos_theta = cos(theta);
                Matx31d cur_derivative_wrt_theta =
                        Matx31d(-sin_theta * scale3d[0] * template_point_world_coord[0] - cos_theta * scale3d[1] * template_point_world_coord[1],
                        cos_theta * scale3d[0] * template_point_world_coord[0] - sin_theta * scale3d[1] * template_point_world_coord[1],
                        0);
                Matx<double, 1, 1> diff_grad_rotation = diff_grad_wrt_pointx_t * cur_derivative_wrt_theta;
                jacobians[0][0] = diff_grad_rotation(0, 0);
                Matx<double, 1, 1> obs_grad_rotation = obs_grad_wrt_pointx_t * cur_derivative_wrt_theta;
                jacobians[0][1] = obs_grad_rotation(0, 0);

//                //Matx<double, 1, 1> diff_grad_rotation = diff_grad_wrt_pointx_t * cur_derivative_wrt_theta;
//                jacobians[0][0] = 0;
//                //Matx<double, 1, 1> obs_grad_rotation = obs_grad_wrt_pointx_t * cur_derivative_wrt_theta;
//                jacobians[0][1] = 0;
            }
            if (jacobians[1])
            {
                cv::Matx13d diff_grad_trans = diff_grad_wrt_pointx_t;
                jacobians[1][0] = diff_grad_trans(0);
                jacobians[1][1] = diff_grad_trans(1);
                cv::Matx13d obs_grad_trans = obs_grad_wrt_pointx_t;
                jacobians[1][2] = obs_grad_trans(0);
                jacobians[1][3] = obs_grad_trans(1);

//                //cv::Matx13d diff_grad_trans = diff_grad_wrt_pointx_t;
//                jacobians[1][0] =0;
//                jacobians[1][1] =0;
//                //cv::Matx13d obs_grad_trans = obs_grad_wrt_pointx_t;
//                jacobians[1][2] =0;
//                jacobians[1][3] =0;
            }
            if (jacobians[2])
            {
                cv::Matx33d derivative_wrt_scales(
                        rotation_mat(0, 0) * template_point_world_coord[0],
                        rotation_mat(0, 1) * template_point_world_coord[1],
                        rotation_mat(0, 2) * template_point_world_coord[2],
                        rotation_mat(1, 0) * template_point_world_coord[0],
                        rotation_mat(1, 1) * template_point_world_coord[1],
                        rotation_mat(1, 2) * template_point_world_coord[2],
                        rotation_mat(2, 0) * template_point_world_coord[0],
                        rotation_mat(2, 1) * template_point_world_coord[1],
                        rotation_mat(2, 2) * template_point_world_coord[2]
                            );
                cv::Matx13d diff_grad_scale = diff_grad_wrt_pointx_t * derivative_wrt_scales;
                jacobians[2][0] = diff_grad_scale(0, 0);
                jacobians[2][1] = diff_grad_scale(0, 1);
                jacobians[2][2] = diff_grad_scale(0, 2);
                cv::Matx13d obs_grad_scale = obs_grad_wrt_pointx_t * derivative_wrt_scales;
                jacobians[2][3] = obs_grad_scale(0, 0);
                jacobians[2][4] = obs_grad_scale(0, 1);
                jacobians[2][5] = obs_grad_scale(0, 2);

//                cv::Matx13d diff_grad_scale = diff_grad_wrt_pointx_t * derivative_wrt_scales;
//                jacobians[2][0] = 0;
//                jacobians[2][1] = 0;
//                jacobians[2][2] = diff_grad_scale(0, 2);
//                cv::Matx13d obs_grad_scale = obs_grad_wrt_pointx_t * derivative_wrt_scales;
//                jacobians[2][3] = 0;
//                jacobians[2][4] = 0;
//                jacobians[2][5] = obs_grad_scale(0, 2);
            }
        }
        return true;
    }
    static double min_model_weight;
    static double max_model_weight;
    static double sqrt_lambda_obs;
private:
    const cpu_tsdf::TSDFHashing* tsdf_model;
    const cv::Vec3d template_point_world_coord;
    const double template_point_val;
    const double trans_z;
    const cv::Vec3i debug_voxel_coord;
};
double TSDFAlignCostFunction2DRObsWeight::min_model_weight = 0.0;
double TSDFAlignCostFunction2DRObsWeight::max_model_weight = 20.0;
double TSDFAlignCostFunction2DRObsWeight::sqrt_lambda_obs = 1.0;

// loss term: observation term & TSDF difference
// parameters: 1D rotation, 2D translation, 3D scales
class TSDFAlignRobustCostFunction2DRObsWeight : public ceres::SizedCostFunction<2, 1, 2, 3>
{
public:
    TSDFAlignRobustCostFunction2DRObsWeight(
                    const cpu_tsdf::TSDFHashing* vtsdf_model,
                    const cv::Vec3d& vtemplate_point_world_coord,
                    const double vtemplate_point_val,
                    const double vtrans_z,
                    const cv::Vec3i& vdebug_voxel_coord,
                    const ceres::LossFunction* loss)
            : SizedCostFunction(),
              tsdf_model(vtsdf_model),
              template_point_world_coord(vtemplate_point_world_coord),
              template_point_val(vtemplate_point_val),
              trans_z(vtrans_z),
              debug_voxel_coord(vdebug_voxel_coord),
              loss_(loss)
    {}
    virtual ~TSDFAlignRobustCostFunction2DRObsWeight() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const
    {
        // For one point in the template, get its corresponding TSDF value in the model
        // Compute the difference (we want to align template to model)
        // rotation 1, translation 2, scale 3.
        cv::Vec3d rotation_axis_angle(0, 0,  parameters[0][0]);
        cv::Vec3d translation(parameters[1][0], parameters[1][1], trans_z);
        cv::Vec3d scale3d(parameters[2][0], parameters[2][1], parameters[2][2]);
        cv::Vec3d scaled_pt(
                scale3d[0] * template_point_world_coord[0],
                scale3d[1] * template_point_world_coord[1],
                scale3d[2] * template_point_world_coord[2]);
        cv::Vec3d rotated_pt;
        cv::Vec3d translated_pt;
        ceres::AngleAxisRotatePoint(rotation_axis_angle.val, scaled_pt.val,
                                    rotated_pt.val);
        translated_pt = rotated_pt + translation;
        const double weight_obs = 1.0;
        float model_val = -1;
        float model_weight = 0;
        double error = 0;
        double robust_squared_errors[3] = {0};
        //cout << "translatedpt: "<< translated_pt << endl;

        if (tsdf_model->RetriveDataFromWorldCoord(cv::Vec3f(translated_pt), &model_val, &model_weight) &&
                model_weight > min_model_weight)
        {
            error = model_val - template_point_val;  // the derivative is w.r.t model_val
            (loss_)->Evaluate(error * error, robust_squared_errors);
            residuals[0] = sqrt(model_weight * robust_squared_errors[0]);
            residuals[1] = weight_obs * sqrt_lambda_obs * sqrt(max_model_weight - model_weight);
            //cout << "translatedpt: "<< translated_pt << endl;
            //cout << "modelval: "<< model_val << endl;
            //cout << "modelweight: "<< model_weight << endl;
            //cout << "r1: " << residuals[0] << endl;
            //cout << "r2: " << residuals[1] << endl;
        }
        else
        {
            error = 0.0;
            model_weight = 0.0;
            model_val = -1;
            residuals[0] = 0.0;
            residuals[1] = weight_obs * sqrt_lambda_obs * sqrt(max_model_weight);
        }
        // compute the jacobians
        if (jacobians)
        {
            cv::Vec3f grad;
            cv::Vec3f wgrad;
            if (model_weight < min_model_weight ||
                    !tsdf_model->RetriveGradientFromWorldCoord(cv::Vec3f(translated_pt), &grad, &wgrad))
            {
                if (jacobians[0])
                {
                    jacobians[0][0] = 0.0;
                    jacobians[0][1] = 0.0;
                }
                if (jacobians[1])
                {
                    jacobians[1][0] = jacobians[1][1]  = 0.0;
                    jacobians[1][2] = jacobians[1][3]  = 0.0;
                }
                if (jacobians[2])
                {
                    jacobians[2][0] = jacobians[2][1] = jacobians[2][2] = 0.0;
                    jacobians[2][3] = jacobians[2][4] = jacobians[2][5] = 0.0;
                }
                return true;
            }
            double diff_loss_reverse_square = 1.0 / (residuals[0] + 1e-5);
            Matx13d diff_grad_wrt_pointx_t = 0.5 * diff_loss_reverse_square * (robust_squared_errors[0] * wgrad + model_weight * robust_squared_errors[1] * 2 * error * grad).t();
            double max_minus_w_square_root_r = 1.0/sqrt(max_model_weight - model_weight);
            Matx13d obs_grad_wrt_pointx_t =  - (weight_obs) * sqrt_lambda_obs * 0.5 * max_minus_w_square_root_r * wgrad.t() ;
            Matx33d rotation_mat;
            cv::Rodrigues(rotation_axis_angle, rotation_mat);
            if (jacobians[0])  // rotation, 1*3 mat
            {
                double theta = parameters[0][0];
                double sin_theta = sin(theta);
                double cos_theta = cos(theta);
                Matx31d cur_derivative_wrt_theta =
                        Matx31d(-sin_theta * scale3d[0] * template_point_world_coord[0] - cos_theta * scale3d[1] * template_point_world_coord[1],
                        cos_theta * scale3d[0] * template_point_world_coord[0] - sin_theta * scale3d[1] * template_point_world_coord[1],
                        0);
                Matx<double, 1, 1> diff_grad_rotation = diff_grad_wrt_pointx_t * cur_derivative_wrt_theta;
                jacobians[0][0] = diff_grad_rotation(0, 0);
                Matx<double, 1, 1> obs_grad_rotation = obs_grad_wrt_pointx_t * cur_derivative_wrt_theta;
                jacobians[0][1] = obs_grad_rotation(0, 0);
            }
            if (jacobians[1])
            {
                cv::Matx13d diff_grad_trans = diff_grad_wrt_pointx_t;
                jacobians[1][0] = diff_grad_trans(0);
                jacobians[1][1] = diff_grad_trans(1);
                cv::Matx13d obs_grad_trans = obs_grad_wrt_pointx_t;
                jacobians[1][2] = obs_grad_trans(0);
                jacobians[1][3] = obs_grad_trans(1);
            }
            if (jacobians[2])
            {
                cv::Matx33d derivative_wrt_scales(
                        rotation_mat(0, 0) * template_point_world_coord[0],
                        rotation_mat(0, 1) * template_point_world_coord[1],
                        rotation_mat(0, 2) * template_point_world_coord[2],
                        rotation_mat(1, 0) * template_point_world_coord[0],
                        rotation_mat(1, 1) * template_point_world_coord[1],
                        rotation_mat(1, 2) * template_point_world_coord[2],
                        rotation_mat(2, 0) * template_point_world_coord[0],
                        rotation_mat(2, 1) * template_point_world_coord[1],
                        rotation_mat(2, 2) * template_point_world_coord[2]
                            );
                cv::Matx13d diff_grad_scale = diff_grad_wrt_pointx_t * derivative_wrt_scales;
                jacobians[2][0] = diff_grad_scale(0, 0);
                jacobians[2][1] = diff_grad_scale(0, 1);
                jacobians[2][2] = diff_grad_scale(0, 2);
                cv::Matx13d obs_grad_scale = obs_grad_wrt_pointx_t * derivative_wrt_scales;
                jacobians[2][3] = obs_grad_scale(0, 0);
                jacobians[2][4] = obs_grad_scale(0, 1);
                jacobians[2][5] = obs_grad_scale(0, 2);
            }
        }
        return true;
    }
    static double min_model_weight;
    static double max_model_weight;
    static double sqrt_lambda_obs;
private:
    const cpu_tsdf::TSDFHashing* tsdf_model;
    const cv::Vec3d template_point_world_coord;
    const double template_point_val;
    const double trans_z;
    const cv::Vec3i debug_voxel_coord;
    const ceres::LossFunction* loss_;
};
double TSDFAlignRobustCostFunction2DRObsWeight::min_model_weight = 0.0;
double TSDFAlignRobustCostFunction2DRObsWeight::max_model_weight = 20.0;
double TSDFAlignRobustCostFunction2DRObsWeight::sqrt_lambda_obs = 1.0;

class TSDFAlignAverageScalesFunctor {
public:
    TSDFAlignAverageScalesFunctor() {}
    template <typename T>
    bool operator()(const T* const scale3d , const T* const average_scale, T* e) const {
//        e[0] = sqrt_lambda_average_scale* (scale3d[0] - average_scale[0]);
//        e[1] = sqrt_lambda_average_scale* (scale3d[1] - average_scale[1]);
//        e[2] = sqrt_lambda_average_scale* (scale3d[2] - average_scale[2]);
        e[0] =  (scale3d[0] - average_scale[0]);
        e[1] =  (scale3d[1] - average_scale[1]);
        e[2] =  (scale3d[2] - average_scale[2]);
        return true;
    }
};

class TSDFAlignAverageRotationFunctor {
public:
    TSDFAlignAverageRotationFunctor() {}
    template <typename T>
    bool operator()(const T* const scale , const T* const average_scale, T* e) const {
        e[0] = sqrt_lambda_average_rotation * (scale - average_scale);
        return true;
    }
    static double sqrt_lambda_average_rotation;
};
double TSDFAlignAverageRotationFunctor::sqrt_lambda_average_rotation = 1.0;

class TSDFAlignRotationRegularizationFunctor {
public:
    TSDFAlignRotationRegularizationFunctor(double origin_rot){ origin_rot_ = origin_rot; }
    template <typename T>
    bool operator()(const T* const rot , T* e) const {
        e[0] = *rot - (T)origin_rot_;
        return true;
    }
private:
    double origin_rot_;
};

class TSDFAlignScaleRegularizationFunctor {
public:
    TSDFAlignScaleRegularizationFunctor(const double* origin_scale)
    {
        origin_scale_[0] = origin_scale[0];
        origin_scale_[1] = origin_scale[1];
        origin_scale_[2] = origin_scale[2];
    }
    template <typename T>
    bool operator()(const T* const scale , T* e) const {
        e[0] = scale[0] - (T)origin_scale_[0];
        e[1] = scale[1] - (T)origin_scale_[1];
        e[2] = scale[2] - (T)origin_scale_[2];
        return true;
    }
private:
    double origin_scale_[3];
};

class TSDFAlignTransRegularizationFunctor {
public:
    TSDFAlignTransRegularizationFunctor(double* origin_trans)
    {
        origin_trans_[0] = origin_trans[0];
        origin_trans_[1] = origin_trans[1];
    }
    template <typename T>
    bool operator()(const T* const trans , T* e) const {
        e[0] = trans[0] - (T)origin_trans_[0];
        e[1] = trans[1] - (T)origin_trans_[1];
        return true;
    }
private:
    double origin_trans_[3];
};

class TSDFAlignTargetZScaleRegularizationFunctor {
public:
    TSDFAlignTargetZScaleRegularizationFunctor(const double target_zscale)
    {
        target_zscale_ = target_zscale;
        cout << target_zscale_ << endl;
    }
    template <typename T>
    bool operator()(const T* const scale , T* e) const {
        e[0] = scale[2] - (T)target_zscale_;
        return true;
    }
private:
    double target_zscale_;
};

//class OutputObbCallback : public ceres::IterationCallback {
// public:
//    OutputObbCallback(
//            const std::vector<double>* vproblem_sample_rotations,
//            const std::vector<cv::Vec3d>* vproblem_sample_scales3d,
//            const std::vector<cv::Vec3d>* vproblem_sample_trans,
//            const std::vector<cv::Vec3d>* vproblem_model_scales,
//            const std::string vpath,
//                    // only for debugging
//                    cpu_tsdf::TSDFHashing::ConstPtr vscene_tsdf,
//                    const cpu_tsdf::PCAOptions&  voptions,
//                    const std::vector<int> &vmodel_assign_idx,
//                    const std::vector<double> &outlier_gammasv
//                    //const std::vector<Eigen::Vector3f> &vmodel_average_scales
//            )
//        :problem_sample_rotations(vproblem_sample_rotations),
//         problem_sample_scales3d(vproblem_sample_scales3d),
//         problem_sample_trans(vproblem_sample_trans),
//         problem_model_scales(vproblem_model_scales),
//         save_path(vpath),
//         scene_tsdf(vscene_tsdf),
//         options(voptions),
//         model_assign_idx(vmodel_assign_idx),
//         outlier_gammas(outlier_gammasv)
//         //model_average_scales(vmodel_average_scales)
//    { }

//  ~OutputObbCallback() {}

//  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
//  {
//      std::vector<Eigen::Affine3f> affine_transforms;
//      for (int i = 0; i < problem_sample_rotations->size(); ++i)
//      {
//          cv::Vec3f rot_axis_angle(0, 0, (*problem_sample_rotations)[i]);
//          cv::Matx33f rot_mat;
//          cv::Rodrigues(rot_axis_angle, rot_mat);
//          cv::Vec3f scales = cv::Vec3d((*problem_sample_scales3d)[i]);
//          cv::Vec3f trans = cv::Vec3d((*problem_sample_trans)[i]);
//          cout << "scales: \n" << scales << endl;
//          cout << "trans: \n" << trans << endl;
//          Eigen::Matrix4f affine_mat = Eigen::Matrix4f::Zero();
//          affine_mat.block<3, 3>(0, 0) = utility::CvMatxToEigenMat(rot_mat) *(utility::CvVectorToEigenVector3(scales)).asDiagonal();
//          affine_mat.block<3, 1>(0, 3) = utility::CvVectorToEigenVector3(trans);
//          affine_mat(3, 3) = 1;
//          Eigen::Affine3f affine(affine_mat);
//          cpu_tsdf::OrientedBoundingBox obb;
//          //cpu_tsdf::AffineToOrientedBB(affine, &obb);
//          //////////////////////////////////
//          Eigen::Matrix3f rot;
//          Eigen::Vector3f scales3d;
//          Eigen::Vector3f trans3d;
//          utility::EigenAffine3fDecomposition(affine, &rot, &scales3d, &trans3d);
//          obb.bb_offset = trans3d - (
//                      (rot * scales3d.asDiagonal() * Eigen::Vector3f(1, 1, 0))
//                      / 2.0);
//          obb.bb_orientation = rot;
//          obb.bb_sidelengths = scales3d;
//          Eigen::Affine3f cur_affine;
//          cpu_tsdf::OBBToAffine(obb, &cur_affine);
//          affine_transforms.push_back(cur_affine);
//          /////////////////////////////////
//          cpu_tsdf::WriteOrientedBoundingboxPly(
//                      obb.bb_orientation,
//                      obb.bb_offset,
//                      obb.bb_sidelengths,
//                      save_path + "_iter_" + boost::lexical_cast<string>(summary.iteration) + "_sample_" + boost::lexical_cast<string>(i) + ".ply");
//      }
//      //////////////////////////////////////////////
//      // added for saving intermediate result
//      std::vector<Eigen::SparseVector<float> > model_means;
//      std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > model_bases;
//      std::vector<Eigen::VectorXf> projected_coeffs;
//      /*const */Eigen::SparseMatrix<float, Eigen::ColMajor> reconstructed_sample_weights;
//      Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
//      Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
//      cpu_tsdf::TSDFGridInfo grid_info(*scene_tsdf, options.boundingbox_size, options.min_model_weight);
//      ExtractSamplesFromAffineTransform(
//                              *scene_tsdf,
//                              affine_transforms,
//                              grid_info,
//                              &samples,
//                              &weights
//                              );
//      bfs::path write_dir_block2(bfs::path(save_path).parent_path()/(string("pca_part_debug") + "_iter_" + boost::lexical_cast<string>(summary.iteration)));
//      bfs::create_directories(write_dir_block2);
//      options.save_path = (write_dir_block2/bfs::path(save_path).stem()).string() + "_ModelCoeffPCA";
//      Eigen::SparseMatrix<float, Eigen::ColMajor> valid_obs_weight_mat;
//      float pos_trunc, neg_trunc;
//      scene_tsdf->getDepthTruncationLimits(pos_trunc, neg_trunc);
//      int noise_clean_counter_thresh = 3;
//      int compo_thresh = -1;
//      int PCA_component_num = 0;
//      int PCA_max_iter = 25;
//      //weights is modified
//      CleanNoiseInSamples(
//                              samples, model_assign_idx, outlier_gammas, &weights, &valid_obs_weight_mat,
//                              noise_clean_counter_thresh, pos_trunc, neg_trunc);
//      // clean isolated parts
//      CleanTSDFSampleMatrix(*scene_tsdf, samples, options.boundingbox_size, compo_thresh, &weights, -1, 2);
//      std::vector<Eigen::SparseVector<float>> model_mean_weights;
//      cpu_tsdf::OptimizeModelAndCoeff(samples, weights, model_assign_idx,
//                            outlier_gammas,
//                            PCA_component_num, PCA_max_iter,
//                            &model_means, &model_mean_weights, &model_bases, &projected_coeffs,
//                            options);
//      // recompute reconstructed model's weights
//      std::vector<Eigen::SparseVector<float>> recon_samples;
//      const std::vector<int>& sample_model_assign = model_assign_idx;
//      cpu_tsdf::PCAReconstructionResult(model_means, model_bases, projected_coeffs, sample_model_assign, &recon_samples);
//      Eigen::SparseMatrix<float, Eigen::ColMajor> recon_sample_mat = cpu_tsdf::SparseVectorsToEigenMat(recon_samples);
//      reconstructed_sample_weights = valid_obs_weight_mat;
//     // clean noise in reconstructed samples
//      {
//          CleanTSDFSampleMatrix(*scene_tsdf, recon_sample_mat,
//                                options.boundingbox_size, compo_thresh,
//                                &reconstructed_sample_weights, -1, 2);
//      }
//      // save
//      {
//          bfs::path parent_dir = write_dir_block2;
//          bfs::path iccv_save_dir = parent_dir/"iccv_result";
//          WriteResultsForICCV(scene_tsdf, affine_transforms,
//                              model_means, model_bases, projected_coeffs,
//                              reconstructed_sample_weights, sample_model_assign, options, iccv_save_dir.string(), "res.ply");

//          // options.save_path = cur_save_path;
//          //TSDFGridInfo tsdf_info(scene_tsdf, options.boundingbox_size, 0);
//          //cpu_tsdf::WriteTSDFsFromMatWithWeight(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//          //                                      options.save_path + "_ModelCoeffPCA_reconstructed_withweight.ply");
//          //cpu_tsdf::WriteTSDFsFromMatsWithWeight_Matlab(recon_sample_mat, *reconstructed_sample_weights, tsdf_info,
//          //                                              options.save_path + "_ModelCoeffPCA_reconstructed_withweight.mat");
//          //cpu_tsdf::WriteTSDFsFromMatNoWeight(recon_sample_mat, tsdf_info,
//          //                                    options.save_path + "_ModelCoeffPCA_reconstructed_noweight.ply");

//      }
//      //////////////////////////////


//    return ceres::SOLVER_CONTINUE;
//    // try interleaving optmization and pca
//    // return ceres::SOLVER_ABORT;
//  }
//
// private:
//  const std::vector<double>* problem_sample_rotations;
//  const std::vector<cv::Vec3d>* problem_sample_scales3d;  // used in scale consistency
//  const std::vector<cv::Vec3d>* problem_sample_trans;
//  const std::vector<cv::Vec3d>* problem_model_scales;
//  const std::string save_path;
//  // only for debug saving
//  cpu_tsdf::TSDFHashing::ConstPtr scene_tsdf;
//  cpu_tsdf::PCAOptions  options;
//  const std::vector<int> &model_assign_idx;
//  const std::vector<double> &outlier_gammas;
//  // const std::vector<Eigen::Vector3f> &model_average_scales;


//};

////////////////////////////////////////////////////////////

namespace cpu_tsdf
{

//void WriteResultsForICCV(
//        TSDFHashing::ConstPtr original_scene,
//        const std::vector<Eigen::Affine3f> &affine_transforms,
//        const std::vector<Eigen::SparseVector<float> > &model_means,
//        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
//        const std::vector<Eigen::VectorXf> &projected_coeffs,
//        const Eigen::SparseMatrix<float, Eigen::ColMajor>& reconstructed_sample_weights,
//        const std::vector<int> &model_assign_idx,
//        const PCAOptions& pca_options,
//        const string &output_dir, const string &file_prefix)
//{
//    PCAOptions my_option = pca_options;
//    static int call_count = 0;
//    char call_count_str[10] = {'\0'};
//    sprintf(call_count_str, "%d", call_count);

//    bfs::path write_dir(output_dir);
//    bfs::create_directories(write_dir);
//    bfs::path write_prefix = write_dir/file_prefix;

//    // save original scene mesh file
//    const float mesh_min_weight = my_option.min_model_weight;
//    cpu_tsdf::WriteTSDFModel(original_scene,
//                             (write_prefix.replace_extension("original_scene.ply")).string(),
//                             false, true, mesh_min_weight);
//    // save all the bounding boxes and cropped tsdfs
//    my_option.save_path = (write_prefix.replace_extension("obbs.ply")).string();
//    WriteAffineTransformsAndTSDFs(*original_scene, affine_transforms, my_option);
//    // save reconstructed samples
//    cpu_tsdf::TSDFGridInfo grid_info(*original_scene, my_option.boundingbox_size, mesh_min_weight);
//    std::vector<cpu_tsdf::TSDFHashing::Ptr> reconstructed_samples_original_pos;
//    cpu_tsdf::ReconstructTSDFsFromPCAOriginPos(
//            model_means,
//            model_bases,
//            projected_coeffs,
//            reconstructed_sample_weights,
//            model_assign_idx,
//            affine_transforms,
//            grid_info,
//            original_scene->voxel_length(),
//            original_scene->offset(),
//            &reconstructed_samples_original_pos);
//    cpu_tsdf::WriteTSDFModels(reconstructed_samples_original_pos,
//                              write_prefix.replace_extension("_frecon_tsdf.ply").string(),
//                              false, true, mesh_min_weight);
//    // save merged scene model
//    cpu_tsdf::TSDFHashing::Ptr copied_scene(new cpu_tsdf::TSDFHashing);
//    *copied_scene = *original_scene;
//    cpu_tsdf::MergeTSDFs(reconstructed_samples_original_pos, copied_scene.get());
//    cpu_tsdf::CleanTSDF(copied_scene, 100);
//    cpu_tsdf::WriteTSDFModel(copied_scene,
//                             (write_prefix.replace_extension("tsdf_consistency_cleaned_tsdf.ply")).string(),
//                             true, true, mesh_min_weight);
//    std::vector<cpu_tsdf::OrientedBoundingBox> obbs;
//    cpu_tsdf::AffinesToOrientedBBs(affine_transforms, &obbs);
//    cpu_tsdf::WriteOrientedBoundingBoxes(
//                            (write_prefix.string().substr(0, std::string::npos - 4) + ("_out_obbs.txt")),
//                            obbs, model_assign_idx);
//}

//double AlignTSDF(const TSDFHashing& tsdf_model,
//                 const TSDFHashing& tsdf_template, const float min_model_weight, Eigen::Affine3f* transform)
//{
//    cv::Matx33f rotation = cv::Matx33f::eye();
//    cv::Vec3f trans(0, 0, 0);
//    float scale = 1.0;
//    double final_cost;
//    final_cost =  AlignTSDF(tsdf_model, tsdf_template, min_model_weight, rotation, trans, scale);
//    transform->setIdentity();
//    transform->prescale(scale);
//    transform->prerotate(CvMatxToEigenMat<float, 3, 3>(rotation));
//    transform->pretranslate(CvVectorToEigenVector3(trans));
//    return final_cost;
//}

// align template to model
//double AlignTSDF(const TSDFHashing& tsdf_model,
//                 const TSDFHashing& tsdf_template, const float min_model_weight, cv::Matx33f& rotation, cv::Vec3f& trans, float& scale)
//{
//    using namespace ceres;
//    cv::Matx33d double_rotation(rotation);
//    cv::Vec3d problem_axis_angle_rotation;
//    cv::Rodrigues(double_rotation, problem_axis_angle_rotation);
//    cv::Vec3d problem_trans = trans;
//    double problem_scale = scale;
//    // problem
//    std::cout << "building problem." << std::endl;
//    ceres::Problem problem;
//    TSDFAlignRobustCostFunction2DRObsWeight::min_model_weight = min_model_weight;
//    for (TSDFHashing::const_iterator citr = tsdf_template.begin(); citr != tsdf_template.end(); ++citr)
//    {
//        cv::Vec3i voxel_coord = citr.VoxelCoord();
//        float d, w;
//        cv::Vec3b color;
//        citr->RetriveData(&d, &w, &color);
//        if (w > 0)
//        {
//            cv::Vec3f world_coord = tsdf_template.Voxel2World((cv::Vec3f(voxel_coord)));
//            ceres::CostFunction* cost_function = new TSDFAlignCostFunction(&tsdf_model, world_coord, d);
//            problem.AddResidualBlock(cost_function, NULL, problem_axis_angle_rotation.val, problem_trans.val, &problem_scale);
//        }
//    }
//    // solving
//    std::cout << "begin solving." << std::endl;
//    ceres::Solver::Options options;
//    options.linear_solver_type = ceres::DENSE_QR;
//    //options.check_gradients = true;
//    options.minimizer_progress_to_stdout = true;
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << "\n";
//    // get the solution
//    std::cout << "solving finished." << std::endl;
//    cv::Rodrigues(problem_axis_angle_rotation, double_rotation);
//    rotation = cv::Matx33f(double_rotation);
//    trans = cv::Vec3f(problem_trans);
//    scale = float(problem_scale);
//    cout << "rotation mat: " << (cv::Mat)rotation << endl;
//    cout << "trans mat: " << (cv::Mat)trans << endl;
//    cout << "scale : " << scale << endl;
//    return summary.final_cost;
//}

//bool TransformTSDF(const TSDFHashing& tsdf_origin,
//                   const Eigen::Affine3f& transform,
//                   TSDFHashing* tsdf_translated,
//                   const float* pvoxel_length,
//                   const Eigen::Vector3f* pscene_offset)
//{
//    Eigen::Matrix3f rotation;
//    Eigen::Vector3f scaling;
//    Eigen::Vector3f trans;
//    utility::EigenAffine3fDecomposition(transform, &rotation, &scaling, &trans);
//    //transform.computeRotationScaling(&rotation, &scaling);
//    //float scale = scaling(0, 0);
//    //Eigen::Vector3f trans = transform.translation();

//    cout << "Eigen Rotation Matrix:\n " << rotation << endl;
//    cout << "Eigen Scaling Matrix:\n " << scaling << endl;
//    cout << "Eigen Translation:\n " << trans << endl;
//    return TransformTSDF(tsdf_origin, EigenMatToCvMatx<float, 3, 3>(rotation),
//                         EigenVectorToCvVector3(trans), EigenVectorToCvVector3(scaling), tsdf_translated, pvoxel_length, pscene_offset);

//}

//bool TransformTSDF(const TSDFHashing& tsdf_origin,
//                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const cv::Vec3f& scale, TSDFHashing* tsdf_translated,
//                   const float* pvoxel_length, const Eigen::Vector3f *pscene_offset)
//{
//    cout << "begin transform TSDF. " << endl;
//    cout << "rotation: " << (cv::Mat)rotation << endl;
//    cout << "trans: " << (cv::Mat)trans << endl;
//    cout << "scale: " << scale << endl;

//    cv::Matx33f scale_mat = cv::Matx33f::eye();
//    scale_mat(0, 0) = scale[0];
//    scale_mat(1, 1) = scale[1];
//    scale_mat(2, 2) = scale[2];
//    cv::Matx33f scaled_rot = rotation * scale_mat;
//    float voxel_length = pvoxel_length ? *pvoxel_length:tsdf_origin.voxel_length();
//    Eigen::Vector3f offset = pscene_offset ? *pscene_offset : tsdf_origin.offset();
//    // cv::Vec3f cvoffset = EigenVectorToCvVector3(offset);
//    // tsdf_translated->offset(CvVectorToEigenVector3(scaled_rot * cvoffset + trans));
//    float max_dist_pos, max_dist_neg;
//    tsdf_origin.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
//    // tsdf_translated->Init(voxel_length, CvVectorToEigenVector3(scaled_rot * cvoffset + trans), max_dist_pos, max_dist_neg);
//    tsdf_translated->Init(voxel_length, offset, max_dist_pos, max_dist_neg);

//    TSDFHashing::update_hashset_type brick_update_hashset;
//    for (TSDFHashing::const_iterator citr = tsdf_origin.begin(); citr != tsdf_origin.end(); ++citr)
//    {
//        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
//        float d, w;
//        cv::Vec3b color;
//        citr->RetriveData(&d, &w, &color);
//        if (w > 0)
//        {
//            cv::Vec3f world_coord_origin = tsdf_origin.Voxel2World(cv::Vec3f(cur_voxel_coord));
//            cv::Vec3f world_coord_origin_transformed =
//                    scaled_rot * (world_coord_origin) + trans;
//            cv::Vec3i voxel_coord_transformed = cv::Vec3i(tsdf_translated->World2Voxel(world_coord_origin_transformed));
//            tsdf_translated->AddBrickUpdateList(voxel_coord_transformed, &brick_update_hashset);
//        }  // end if
//    }  // end for
//    cout << "update list size: " << brick_update_hashset.size() << endl;

//    struct TSDFTranslatedVoxelUpdater
//    {
//        TSDFTranslatedVoxelUpdater(const TSDFHashing& tsdf_origin, const TSDFHashing& tsdf_template,
//                                   const cv::Matx33f& inverse_scaled_rot, const cv::Vec3f& inverse_trans)
//            : tsdf_origin(tsdf_origin), tsdf_template(tsdf_template), inverse_scaled_rot(inverse_scaled_rot),
//              inverse_trans(inverse_trans) {}
//        bool operator () (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
//        {
//            // for every voxel in its new location, retrive its value in the original tsdf
//            cv::Vec3f world_coord_transformed = tsdf_template.Voxel2World(cv::Vec3f(cur_voxel_coord));
//            cv::Vec3f world_coord_untransformed = inverse_scaled_rot * world_coord_transformed + inverse_trans;
//            float cur_d;
//            float cur_w;
//            cv::Vec3b cur_color;
//            if (!tsdf_origin.RetriveDataFromWorldCoord(world_coord_untransformed, &cur_d, &cur_w, &cur_color)) return false;
//            *d = cur_d;
//            *w = cur_w;
//            *color = cur_color;
//            return true;
//        }
//        const TSDFHashing& tsdf_origin;
//        const TSDFHashing& tsdf_template;
//        const cv::Matx33f& inverse_scaled_rot;
//        const cv::Vec3f inverse_trans;
//    };
//    // cv::Matx33f inverse_scaled_rot = rotation.t() * (1.0 / scale);
//    cv::Matx33f inverse_scale_mat = cv::Matx33f::eye();
//    inverse_scale_mat(0, 0) = 1.0/scale[0];
//    inverse_scale_mat(1, 1) = 1.0/scale[1];
//    inverse_scale_mat(2, 2) = 1.0/scale[2];
//    cv::Matx33f inverse_scaled_rot = inverse_scale_mat * rotation.t();
//    cv::Vec3f inverse_trans = inverse_scaled_rot * (-trans);
//    TSDFTranslatedVoxelUpdater updater(tsdf_origin, *tsdf_translated, inverse_scaled_rot, inverse_trans);
//    tsdf_translated->UpdateBricksInQueue(brick_update_hashset,
//                                         updater);
//    cout << "finished transform TSDF. " << endl;
//    return true;
//}

//bool TransformTSDF(const TSDFHashing& tsdf_origin,
//                   const cv::Matx33f& rotation, const cv::Vec3f& trans, const float scale, TSDFHashing* tsdf_translated)
//{
//    cout << "begin transform TSDF. " << endl;
//    cout << "rotation: " << (cv::Mat)rotation << endl;
//    cout << "trans: " << (cv::Mat)trans << endl;
//    cout << "scale: " << scale << endl;

//    cv::Matx33f scaled_rot = rotation * scale;
//    float voxel_length = tsdf_origin.voxel_length();
//    Eigen::Vector3f offset = tsdf_origin.offset();
//    cv::Vec3f cvoffset = EigenVectorToCvVector3(offset);
//    float max_dist_pos, max_dist_neg;
//    tsdf_origin.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
//    tsdf_translated->Init(voxel_length, CvVectorToEigenVector3(scaled_rot * cvoffset + trans), max_dist_pos, max_dist_neg);
//    TSDFHashing::update_hashset_type brick_update_hashset;
//    for (TSDFHashing::const_iterator citr = tsdf_origin.begin(); citr != tsdf_origin.end(); ++citr)
//    {
//        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
//        float d, w;
//        cv::Vec3b color;
//        citr->RetriveData(&d, &w, &color);
//        if (w > 0)
//        {
//            cv::Vec3f world_coord_origin = tsdf_origin.Voxel2World(cv::Vec3f(cur_voxel_coord));
//            cv::Vec3f world_coord_origin_transformed =
//                    scaled_rot * (world_coord_origin) + trans;
//            cv::Vec3i voxel_coord_transformed = cv::Vec3i(tsdf_translated->World2Voxel(world_coord_origin_transformed));
//            tsdf_translated->AddBrickUpdateList(voxel_coord_transformed, &brick_update_hashset);
//        }  // end if
//    }  // end for
//    cout << "update list size: " << brick_update_hashset.size() << endl;
    
//    struct TSDFTranslatedVoxelUpdater
//    {
//        TSDFTranslatedVoxelUpdater(const TSDFHashing& tsdf_origin, const TSDFHashing& tsdf_template,
//                                   const cv::Matx33f& inverse_scaled_rot, const cv::Vec3f& inverse_trans)
//            : tsdf_origin(tsdf_origin), tsdf_template(tsdf_template), inverse_scaled_rot(inverse_scaled_rot),
//              inverse_trans(inverse_trans) {}
//        bool operator () (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color) const
//        {
//            // for every voxel in its new location, retrive its value in the original tsdf
//            cv::Vec3f world_coord_transformed = tsdf_template.Voxel2World(cv::Vec3f(cur_voxel_coord));
//            cv::Vec3f world_coord_untransformed = inverse_scaled_rot * world_coord_transformed + inverse_trans;
//            float cur_d;
//            float cur_w;
//            cv::Vec3b cur_color;
//            if (!tsdf_origin.RetriveDataFromWorldCoord(world_coord_untransformed, &cur_d, &cur_w, &cur_color)) return false;
//            *d = cur_d;
//            *w = cur_w;
//            *color = cur_color;
//            return true;
//        }
//        const TSDFHashing& tsdf_origin;
//        const TSDFHashing& tsdf_template;
//        const cv::Matx33f& inverse_scaled_rot;
//        const cv::Vec3f inverse_trans;
//    };
//    cv::Matx33f inverse_scaled_rot = rotation.t() * (1.0/ scale);
//    cv::Vec3f inverse_trans = inverse_scaled_rot * (-trans);
//    TSDFTranslatedVoxelUpdater updater(tsdf_origin, *tsdf_translated, inverse_scaled_rot, inverse_trans);
//    tsdf_translated->UpdateBricksInQueue(brick_update_hashset,
//                                         updater);
//    cout << "finished transform TSDF. " << endl;
//    return true;
//}

bool InitialAlign(TSDFHashing& tsdf_model, TSDFHashing& tsdf_template, cv::Matx33f& rotation,
                  cv::Vec3f& trans, float& scale)
{
    tsdf_model.CentralizeTSDF();
    tsdf_template.CentralizeTSDF();
    trans = cv::Vec3f(0, 0, 0);
    rotation = cv::Matx33f::eye();
    scale = 1.0;
	return true;
}

//bool OptimizeTransformAndScalesImpl_Debug(
//        const TSDFHashing& scene_tsdf,
//        const std::vector<const TSDFHashing*> model_reconstructed_samples,
//        const std::vector<int>& sample_model_assignment,
//        const float min_model_weight,
//        std::vector<Eigen::Affine3f>* transforms, /*#samples, input and output*/
//        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
//        )
//{
//    using namespace ceres;
//    using namespace std;
//    const int sample_num = model_reconstructed_samples.size();
//    const int model_num = model_scales->size();
//    // ceres::Problem problem;
//    TSDFAlignCostFunction3DScale::min_model_weight = min_model_weight;
//    // saves all the parameters and pass them to problem
//    std::vector<cv::Vec3d> problem_sample_rotations(sample_num);
//    std::vector<cv::Vec3d> problem_sample_scales3d(sample_num);  // used in scale consistency
//    std::vector<cv::Vec3d> problem_sample_trans(sample_num);
//    std::vector<cv::Vec3d> problem_model_scales(model_num);
//    for (int i = 0; i < model_num; ++i) problem_model_scales[i] = utility::EigenVectorToCvVector3((*model_scales)[i]);
//
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        ceres::Problem problem;
//        const TSDFHashing& current_model = *(model_reconstructed_samples[sample_i]);
//
//        Eigen::Matrix3f rotation_mat;
//        Eigen::Vector3f scale3d;
//        Eigen::Vector3f trans;
//        utility::EigenAffine3fDecomposition((*transforms)[sample_i], &rotation_mat, &scale3d, &trans);
//
//        cv::Matx33d cv_rotation = utility::EigenMatToCvMatx(rotation_mat.cast<double>().eval());
//        cv::Vec3d problem_axis_angle_rotation;
//        cv::Rodrigues(cv_rotation, problem_axis_angle_rotation);
//        problem_sample_rotations[sample_i] = problem_axis_angle_rotation;
//        problem_sample_scales3d[sample_i] = utility::EigenVectorToCvVector3(scale3d.cast<double>().eval());
//        problem_sample_trans[sample_i] = utility::EigenVectorToCvVector3(trans.cast<double>().eval());
//
//        // problem
//        std::cout << "building problem for sample " << sample_i << std::endl;
//        cout << "initial rotation: \n" << cv_rotation << endl;
//        cout << "initial scale: \n" << problem_sample_scales3d[sample_i] << endl;
//        cout << "initial trans: \n" << problem_sample_trans[sample_i] << endl;
//        for (TSDFHashing::const_iterator citr = current_model.begin(); citr != current_model.end(); ++citr)
//        {
//            cv::Vec3i voxel_coord = citr.VoxelCoord();
//            float d, w;
//            cv::Vec3b color;
//            citr->RetriveData(&d, &w, &color);
//            if (w > 0)
//            {
//                cv::Vec3f world_coord = current_model.Voxel2World((cv::Vec3f(voxel_coord)));
//                ceres::CostFunction* cost_function = new TSDFAlignCostFunction3DScale(&scene_tsdf, world_coord, d);
////                CostFunction* cost_function =
////                  new NumericDiffCostFunction<TSDFAlignCostFunction3DScaleNoGrad, ceres::CENTRAL, 1, 3, 3, 3>(
////                      new TSDFAlignCostFunction3DScaleNoGrad(&scene_tsdf, world_coord, d));
//                problem.AddResidualBlock(cost_function, NULL,
//                                         problem_sample_rotations[sample_i].val,
//                                         problem_sample_trans[sample_i].val,
//                                         problem_sample_scales3d[sample_i].val);
//            }
//        }  // end for citr
//        std::cout << "begin solving." << std::endl;
//        ceres::Solver::Options options;
//        options.max_num_iterations = 100;
//        options.linear_solver_type = ceres::DENSE_QR;
//        // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//        // options.check_gradients = true;
//        options.minimizer_progress_to_stdout = true;
//        ceres::Solver::Summary summary;
//        ceres::Solve(options, &problem, &summary);
//        std::cout << summary.FullReport() << "\n";
//
//        cv::Matx33d rotation_matcv;
//        //cv::Rodrigues( cv::Vec3f(problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
//        cv::Rodrigues( (problem_sample_rotations[sample_i]), rotation_matcv);  // mind type...
//        Eigen::Matrix3f rotation_mat_eigen = utility::CvMatxToEigenMat(cv::Matx33f(rotation_matcv));
//        Eigen::Vector3f scale3d_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_scales3d[sample_i]));
//        Eigen::Vector3f trans_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_trans[sample_i]));
//        Eigen::Matrix4f affine_trans = Eigen::Matrix4f::Zero();
//        affine_trans.block<3, 3>(0, 0) = rotation_mat_eigen * scale3d_eigen.asDiagonal();
//        affine_trans.block<3, 1>(0, 3) = trans_eigen;
//        affine_trans(3, 3) = 1;
//        (*transforms)[sample_i].matrix() = affine_trans;
//        cout << "optimized rotation: \n" << rotation_matcv << endl;
//        cout << "optimized scale: \n" << problem_sample_scales3d[sample_i]  << endl;
//        cout << "optimized trans: \n" << problem_sample_trans[sample_i]  << endl;
//
//    }  // end for sample_i
//
//    // add scale consistency blocks
////    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
////    {
////        cv::Vec3d& current_scale3d = problem_sample_scales3d[sample_i];
////        cv::Vec3d& current_model_scale = problem_model_scales[sample_model_assignment[sample_i]];
////        //ceres::CostFunction* cost_function = new TSDFAlignCostFunction3DScale(&scene_tsdf, world_coord, d);
////        ceres::CostFunction* cost_function
////                = new AutoDiffCostFunction<TSDFAlignAverageScalesFunctor, 3, 3, 3>(
////                    new TSDFAlignAverageScalesFunctor());
////        problem.AddResidualBlock(cost_function, NULL, current_scale3d.val, current_model_scale.val);
////    }
//
//    // solving
////    std::cout << "begin solving." << std::endl;
////    ceres::Solver::Options options;
////    options.linear_solver_type = ceres::DENSE_QR;
////    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
////    // options.check_gradients = true;
////    options.minimizer_progress_to_stdout = true;
////    ceres::Solver::Summary summary;
////    ceres::Solve(options, &problem, &summary);
////    std::cout << summary.FullReport() << "\n";
//
//    // get the solution
////    std::cout << "solving finished." << std::endl;
////    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
////    {
////        cv::Matx33d rotation_mat;
////        //cv::Rodrigues( cv::Vec3f(problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
////        cv::Rodrigues( (problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
////        Eigen::Matrix3f rotation_mat_eigen = utility::CvMatxToEigenMat(cv::Matx33f(rotation_mat));
////        Eigen::Vector3f scale3d_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_scales3d[sample_i]));
////        Eigen::Vector3f trans_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_trans[sample_i]));
////        Eigen::Matrix4f affine_trans = Eigen::Matrix4f::Zero();
////        affine_trans.block<3, 3>(0, 0) = rotation_mat_eigen * scale3d_eigen.asDiagonal();
////        affine_trans.block<3, 1>(0, 3) = trans_eigen;
////        affine_trans(3, 3) = 1;
////        (*transforms)[sample_i].matrix() = affine_trans;
////        cout << "optimized rotation: \n" << rotation_mat << endl;
////        cout << "optimized scale: \n" << problem_sample_scales3d[sample_i]  << endl;
////        cout << "optimized trans: \n" << problem_sample_trans[sample_i]  << endl;
////    }
//    for (int model_i = 0; model_i < model_num; ++model_i)
//    {
//        (*model_scales)[model_i] = utility::CvVectorToEigenVector3(cv::Vec3f(problem_model_scales[model_i]));
//    }
//    return true;
//}

//bool OptimizeTransformAndScalesImplRobustLoss(
//        const TSDFHashing& scene_tsdf,
//        const std::vector<const TSDFHashing*> model_reconstructed_samples,
//        const std::vector<int>& sample_model_assignment,
//        const PCAOptions& pca_options,
//        const std::string& save_path,
//        std::vector<Eigen::Affine3f>* transforms, /*#samples, input and output*/
//        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
//        )
//{
//    const float min_model_weight = pca_options.min_model_weight;
//    const float lambda_scale_diff= pca_options.lambda_scale_diff;
//    const float lambda_obs= pca_options.lambda_observation;
//    const float lambda_reg_rot= pca_options.lambda_reg_rot;
//    const float lambda_reg_scale= pca_options.lambda_reg_scale;
//    const float lambda_reg_trans= pca_options.lambda_reg_trans;
//    const float lambda_reg_z_scale = pca_options.lambda_reg_zscale;  // currently only zero is used.
//
//    using namespace ceres;
//    using namespace std;
//    const int sample_num = model_reconstructed_samples.size();
//    const int model_num = model_scales->size();
//    TSDFAlignRobustCostFunction2DRObsWeight::min_model_weight = min_model_weight;
//    TSDFAlignRobustCostFunction2DRObsWeight::max_model_weight = TSDFHashing::getVoxelMaxWeight() + 1.0;
//    TSDFAlignRobustCostFunction2DRObsWeight::sqrt_lambda_obs = sqrt(lambda_obs);
//
//    cout << "sqrt_lambda_obs: " << TSDFAlignRobustCostFunction2DRObsWeight::sqrt_lambda_obs << endl;
//    ceres::Problem problem;
//    // saves all the parameters and pass them to problem
//    std::vector<double> problem_sample_rotations(sample_num);
//    std::vector<cv::Vec3d> problem_sample_scales3d(sample_num);  // used in scale consistency
//    std::vector<cv::Vec3d> problem_sample_trans(sample_num);
//    std::vector<cv::Vec3d> problem_model_scales(model_num);
//    for (int i = 0; i < model_num; ++i) problem_model_scales[i] = utility::EigenVectorToCvVector3((*model_scales)[i]);
//
//    int cnt = 0;
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        const TSDFHashing& current_model = *(model_reconstructed_samples[sample_i]);
//
//        Eigen::Matrix3f rotation_mat;
//        Eigen::Vector3f scale3d;
//        Eigen::Vector3f trans;
//        utility::EigenAffine3fDecomposition((*transforms)[sample_i], &rotation_mat, &scale3d, &trans);
//
//        cv::Matx33d cv_rotation = utility::EigenMatToCvMatx(rotation_mat.cast<double>().eval());
//        cv::Vec3d problem_axis_angle_rotation;
//        cv::Rodrigues(cv_rotation, problem_axis_angle_rotation);
//        if (fabs(problem_axis_angle_rotation[0]) >= 1e-4)
//        {
//            cout << "fabs(problem_axis_angle_rotation[0]) >= 1e-4" << endl;
//            cout << problem_axis_angle_rotation << endl;
//            cout << cv_rotation << endl;
//        }
//        CHECK(fabs(problem_axis_angle_rotation[0]) < 1e-4);
//        CHECK(fabs(problem_axis_angle_rotation[1]) < 1e-4);
//        // problem_sample_rotations[sample_i] = norm(problem_axis_angle_rotation);
//        problem_sample_rotations[sample_i] = (problem_axis_angle_rotation[2]);
//        problem_sample_scales3d[sample_i] = utility::EigenVectorToCvVector3(scale3d.cast<double>().eval());
//        problem_sample_trans[sample_i] = utility::EigenVectorToCvVector3(trans.cast<double>().eval());
//        problem_sample_trans[sample_i][2] -= (0.5 * scale3d[2]);
//
//        // problem
//        std::cout << "building problem for sample " << sample_i << std::endl;
//        cout << "initial rotation: \n" << cv_rotation << endl;
//        cout << "initial scale: \n" << problem_sample_scales3d[sample_i] << endl;
//        cout << "initial trans: \n" << problem_sample_trans[sample_i] << endl;
//        for (TSDFHashing::const_iterator citr = current_model.begin(); citr != current_model.end(); ++citr)
//        {
//            cv::Vec3i voxel_coord = citr.VoxelCoord();
//            float d, w;
//            cv::Vec3b color;
//            citr->RetriveData(&d, &w, &color);
//            if (w > 0)
//            {
//                cv::Vec3f world_coord = current_model.Voxel2World((cv::Vec3f(voxel_coord)));
//                ceres::CostFunction* cost_function =
//                        new TSDFAlignRobustCostFunction3DScaleWeight2DR(&scene_tsdf, world_coord + cv::Vec3f(0, 0, 0.5)
//                                                              /*move the unit cube centered at origin 0.5 unit upward, so that the bottom center coincides with the origin*/,
//                                                              d, problem_sample_trans[sample_i][2]);
//        // only the observation term
//        // ceres::CostFunction* cost_function_observation =
//        //         new TSDFAlignRobustCostFunctionObs(&scene_tsdf, world_coord + cv::Vec3f(0, 0, 0.5)
//        //                                      /*move the unit cube centered at origin 0.5 unit upward, so that the bottom center coincides with the origin*/,
//        //                                      d, problem_sample_trans[sample_i][2], voxel_coord);
//
////                ceres::CostFunction* cost_function =
////                        new TSDFAlignRobustCostFunction2DRObsWeight(&scene_tsdf, world_coord + cv::Vec3f(0, 0, 0.5), d,
////                                                              problem_sample_trans[sample_i][2], voxel_coord);
//
//                problem.AddResidualBlock(cost_function,
//                                         new ceres::HuberLoss(1.0),  // no weighting coeff
//                                         &(problem_sample_rotations[sample_i]),
//                                         problem_sample_trans[sample_i].val,
//                                         problem_sample_scales3d[sample_i].val);
//                problem.AddResidualBlock(cost_function_observation,
//                                         new ScaledLoss(NULL/*no robust loss here*/, lambda_obs, TAKE_OWNERSHIP),
//                                         &(problem_sample_rotations[sample_i]),
//                                         problem_sample_trans[sample_i].val,
//                                         problem_sample_scales3d[sample_i].val);
//                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].val, 0, 0.01);
//                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].val, 1, 0.01);
//                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].val, 2, 0.01);
//                cnt++;
//            }
//        }  // end for citr
////        if (sample_i == 0) // fix the scale of the first sample
////        {
////            problem.SetParameterBlockConstant(problem_sample_scales3d[sample_i].val);
////        }
//    }  // end for sample_i
//
//    // add scale consistency blocks
//    cout << "lambda_scale_diff: " << lambda_scale_diff << endl;
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        cv::Vec3d& current_scale3d = problem_sample_scales3d[sample_i];
//        cv::Vec3d& current_model_scale = problem_model_scales[sample_model_assignment[sample_i]];
//        ceres::CostFunction* cost_function
//                = new AutoDiffCostFunction<TSDFAlignAverageScalesFunctor, 3, 3, 3>(
//                    new TSDFAlignAverageScalesFunctor());
//        problem.AddResidualBlock(
//                    cost_function,
//                    new ScaledLoss(new ceres::HuberLoss(1.0), lambda_scale_diff, TAKE_OWNERSHIP),
//                    current_scale3d.val, current_model_scale.val);
//    }
//
//    // add rotation regularization blocks
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        ceres::CostFunction* cost_function
//                = new AutoDiffCostFunction<TSDFAlignRotationRegularizationFunctor, 1, 1>(
//                    new TSDFAlignRotationRegularizationFunctor(problem_sample_rotations[sample_i]));
//        problem.AddResidualBlock(
//                    cost_function,
//                    new ScaledLoss(new ceres::HuberLoss(1.0), lambda_reg_rot, TAKE_OWNERSHIP),
//                    &(problem_sample_rotations[sample_i]));
//    }
//
//    // add scale regularization blocks
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        ceres::CostFunction* cost_function
//                = new AutoDiffCostFunction<TSDFAlignScaleRegularizationFunctor, 3, 3>(
//                    new TSDFAlignScaleRegularizationFunctor(problem_sample_scales3d[sample_i].val));
//        problem.AddResidualBlock(
//                    cost_function,
//                    new ScaledLoss(new ceres::HuberLoss(1.0), lambda_reg_scale, TAKE_OWNERSHIP),
//                    (problem_sample_scales3d[sample_i].val));
//    }
//
//    // add translation regularization blocks
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        ceres::CostFunction* cost_function
//                = new AutoDiffCostFunction<TSDFAlignTransRegularizationFunctor, 2, 2>(
//                    new TSDFAlignTransRegularizationFunctor(problem_sample_trans[sample_i].val));
//        problem.AddResidualBlock(
//                    cost_function,
//                    new ScaledLoss(new ceres::HuberLoss(1.0), lambda_reg_trans, TAKE_OWNERSHIP),
//                    (problem_sample_trans[sample_i].val));
//    }
//
//    // target z scale regularization
//    // const std::vector<float>& tscales = pca_options.target_zscales;
////    std::vector<double> tscales(sample_num);
////    std::vector<double> cur_scales(sample_num);
////    for (int i = 0; i < sample_num; ++i)
////    {
////        cur_scales[i] = (problem_sample_scales3d[i].val[2]);
////    }
////    Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
////    Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
////    ExtractSamplesFromAffineTransform(scene_tsdf, *transforms, pca_options, &samples, &weights);
////    cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, pca_options.boundingbox_size, pca_options.min_model_weight);
////    ComputeTargetZScaleOneCluster(samples, weights, cur_scales,
////                                  grid_info, &tscales, 0.005);
////    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
////    {
////        ceres::CostFunction* cost_function
////                = new AutoDiffCostFunction<TSDFAlignTargetZScaleRegularizationFunctor, 1, 3>(
////                    new TSDFAlignTargetZScaleRegularizationFunctor(tscales[sample_i]));
////        problem.AddResidualBlock(cost_function,
////                                 new ScaledLoss(NULL, lambda_reg_z_scale, TAKE_OWNERSHIP),
////                                 (problem_sample_scales3d[sample_i].val));
////        cout << lambda_reg_z_scale << endl;
////    }
//
//    // solving
//    std::cout << "begin solving." << std::endl;
//    ceres::Solver::Options options;
//    //only for debugging
//    /////////////////////////////////////////
////    options.update_state_every_iteration = true;
////    std::unique_ptr<ceres::IterationCallback> debug_callback(new OutputObbCallback
////                                                             (
////                                                            &problem_sample_rotations,
////                                                            &problem_sample_scales3d,
////                                                            &problem_sample_trans,
////                                                            &problem_model_scales,
////                                                            save_path + "_in_opt")
////                                                             );
////    options.callbacks.push_back(debug_callback.get());
//    /////////////////////////////////////////
//    options.max_num_iterations = 100;
//    // options.linear_solver_type = ceres::DENSE_QR;
//    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//    // options.check_gradients = true;
//    // options.gradient_check_relative_precision = 1;
//    options.minimizer_progress_to_stdout = true;
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << "\n";
//
//    //debug
////#define TEST1
//#ifdef TEST1
//    double d_scalez = 0.1;
//    double init_scale = 0.1;
//    printf ("initial scalez: %f\n", problem_sample_scales3d[0][2]);
//    std::vector<std::pair<float, float>> plotdata;
//    cpu_tsdf::PCAOptions tmp_options = pca_options;
//    string init_path = tmp_options.save_path;
//    float old_scale = problem_sample_scales3d[0][2];
//    const int start_scale_cnt = 0;
//    for (int tt = start_scale_cnt; tt < 300; tt+=1)
//    {
//        double cur_scalez = d_scalez * tt + init_scale;
//        double cost = -1;
//        problem_sample_scales3d[0][2] = cur_scalez;
//        problem.Evaluate(Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
//        //printf("%f %f\n", cur_scalez, cost);
//        plotdata.push_back(std::make_pair(cur_scalez, cost));
//
//
//        //tmp_options.offset = Eigen::Vector3f(-0.5, -0.5, 0);
//        tmp_options.save_path = init_path + "_house_debug_" + boost::lexical_cast<string>(tt) + "_curscale_" + boost::lexical_cast<string>(cur_scalez);
//        for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//        {
//            cv::Matx33d rotation_mat;
//            //cv::Rodrigues( cv::Vec3f(problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
//            cv::Rodrigues( cv::Vec3d(0, 0, problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
//            Eigen::Matrix3f rotation_mat_eigen = utility::CvMatxToEigenMat(cv::Matx33f(rotation_mat));
//            Eigen::Vector3f scale3d_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_scales3d[sample_i]));
//            Eigen::Vector3f trans_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_trans[sample_i]))
//                    + rotation_mat_eigen * scale3d_eigen.asDiagonal() * Eigen::Vector3f(0, 0, 0.5);
//            Eigen::Matrix4f affine_trans = Eigen::Matrix4f::Zero();
//            affine_trans.block<3, 3>(0, 0) = rotation_mat_eigen * scale3d_eigen.asDiagonal();
//            affine_trans.block<3, 1>(0, 3) = trans_eigen;
//            affine_trans(3, 3) = 1;
//            //(*transforms)[sample_i].matrix() = affine_trans;
////            cout << "sample: " << sample_i << endl;
////            cout << "optimized rotation: \n" << rotation_mat << endl;
////            cout << "optimized scale: \n" << scale3d_eigen << endl;
////            cout << "optimized trans: \n" << trans_eigen << endl;
//
//            Eigen::SparseVector<float> sample;
//            Eigen::SparseVector<float> weight;
//            // cpu_tsdf::ExtractOneSampleFromAffineTransform(scene_tsdf,  Eigen::Affine3f(affine_trans), tmp_options, &sample, &weight);
//            std::vector<Eigen::Affine3f> transvec(1);
//            transvec[0] = Eigen::Affine3f(affine_trans);
//            cpu_tsdf::WriteAffineTransformsAndTSDFs(scene_tsdf, transvec, tmp_options, true);
//        }
//    }
//    for (int ti = 0; ti < plotdata.size(); ++ti)
//    {
//        printf("%d %f %f\n", start_scale_cnt+ti, plotdata[ti].first, plotdata[ti].second);
//    }
//    problem_sample_scales3d[0][2] = old_scale;
//#endif
//    //exit(1);
//
//    // get the solution
//    std::cout << "solving finished." << std::endl;
//    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
//    {
//        cv::Matx33d rotation_mat;
//        //cv::Rodrigues( cv::Vec3f(problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
//        cv::Rodrigues( cv::Vec3d(0, 0, problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
//        CHECK_LE(rotation_mat(1, 0) * rotation_mat(0, 1), 0);
//        Eigen::Matrix3f rotation_mat_eigen = utility::CvMatxToEigenMat(cv::Matx33f(rotation_mat));
//        Eigen::Vector3f scale3d_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_scales3d[sample_i]));
//        Eigen::Vector3f trans_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_trans[sample_i])) +
//                rotation_mat_eigen * scale3d_eigen.asDiagonal() * Eigen::Vector3f(0, 0, 0.5);
//        Eigen::Matrix4f affine_trans = Eigen::Matrix4f::Zero();
//        affine_trans.block<3, 3>(0, 0) = rotation_mat_eigen * scale3d_eigen.asDiagonal();
//        CHECK_GE(scale3d_eigen[0], 0);
//        CHECK_GE(scale3d_eigen[1], 0);
//        CHECK_GE(scale3d_eigen[2], 0);
//        affine_trans.block<3, 1>(0, 3) = trans_eigen;
//        affine_trans(3, 3) = 1;
//        (*transforms)[sample_i].matrix() = affine_trans;
//        cout << "sample: " << sample_i << endl;
//        cout << "optimized rotation: \n" << rotation_mat << endl;
//        cout << "optimized scale: \n" << scale3d_eigen << endl;
//        cout << "optimized trans: \n" << trans_eigen << endl;
//    }
//    for (int model_i = 0; model_i < model_num; ++model_i)
//    {
//        (*model_scales)[model_i] = utility::CvVectorToEigenVector3(cv::Vec3f(problem_model_scales[model_i]));
//    }
//    return true;
//}


bool OptimizeTransformAndScalesImpl(
        const TSDFHashing& scene_tsdf,
        const std::vector<const TSDFHashing*> model_reconstructed_samples,
        const std::vector<int>& sample_model_assignment,
        const std::vector<double>& gammas, // outliers
        PCAOptions& pca_options,
        const std::string& save_path,
        std::vector<Eigen::Affine3f>* transforms, /*#samples, input and output*/
        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
        )
{
    const float min_model_weight = pca_options.min_model_weight;
    const float lambda_scale_diff= pca_options.lambda_scale_diff;
    const float lambda_obs= pca_options.lambda_observation;

    const float lambda_reg_rot= pca_options.lambda_reg_rot;
    const float lambda_reg_scale= pca_options.lambda_reg_scale;
    const float lambda_reg_trans= pca_options.lambda_reg_trans;
    const float lambda_reg_z_scale = pca_options.lambda_reg_zscale;

    using namespace ceres;
    using namespace std;
    const int sample_num = model_reconstructed_samples.size();
    const int model_num = model_scales->size();
    TSDFAlignCostFunction2DRObsWeight::min_model_weight = min_model_weight;
    TSDFAlignCostFunction2DRObsWeight::max_model_weight = TSDFHashing::getVoxelMaxWeight() + 1.0;
    TSDFAlignCostFunction2DRObsWeight::sqrt_lambda_obs = sqrt(lambda_obs);
    // TSDFAlignCostFunction3DScaleWeight2DR::min_model_weight = min_model_weight;

    cout << "sqrt_lambda_obs: " << TSDFAlignCostFunction2DRObsWeight::sqrt_lambda_obs << endl;
    ceres::Problem problem;
    // saves all the parameters and pass them to problem
    std::vector<double> problem_sample_rotations(sample_num);
    std::vector<cv::Vec3d> problem_sample_scales3d(sample_num);  // used in scale consistency
    std::vector<cv::Vec3d> problem_sample_trans(sample_num);
    std::vector<cv::Vec3d> problem_model_scales(model_num);
    for (int i = 0; i < model_num; ++i) problem_model_scales[i] = utility::EigenVectorToCvVector3((*model_scales)[i]);

    static const double inlier_thresh = 1e-5;
    int cnt = 0;
    //int solver_var_count = 0;
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        const TSDFHashing& current_model = *(model_reconstructed_samples[sample_i]);

        Eigen::Matrix3f rotation_mat;
        Eigen::Vector3f scale3d;
        Eigen::Vector3f trans;
        utility::EigenAffine3fDecomposition((*transforms)[sample_i], &rotation_mat, &scale3d, &trans);

        cv::Matx33d cv_rotation = utility::EigenMatToCvMatx(rotation_mat.cast<double>().eval());
        cv::Vec3d problem_axis_angle_rotation;
        cv::Rodrigues(cv_rotation, problem_axis_angle_rotation);
        if (fabs(problem_axis_angle_rotation[0]) >= 1e-4)
        {
            cout << "fabs(problem_axis_angle_rotation[0]) >= 1e-4" << endl;
            cout << problem_axis_angle_rotation << endl;
            cout << cv_rotation << endl;
        }
        CHECK(fabs(problem_axis_angle_rotation[0]) < 1e-4);
        CHECK(fabs(problem_axis_angle_rotation[1]) < 1e-4);
        problem_sample_rotations[sample_i] = (problem_axis_angle_rotation[2]);
        problem_sample_scales3d[sample_i] = utility::EigenVectorToCvVector3(scale3d.cast<double>().eval());
        problem_sample_trans[sample_i] = utility::EigenVectorToCvVector3(trans.cast<double>().eval());
        problem_sample_trans[sample_i][2] -= (0.5 * scale3d[2]);

        if (gammas[sample_i] > inlier_thresh)
        {
            continue;
        }

        // problem
        std::cout << "building problem for sample " << sample_i << std::endl;
        std::cout << "building problem for inlier sample " << sample_i << std::endl;
        cout << "initial rotation: \n" << cv_rotation << endl;
        cout << "initial scale: \n" << problem_sample_scales3d[sample_i] << endl;
        cout << "initial trans: \n" << problem_sample_trans[sample_i] << endl;
        for (TSDFHashing::const_iterator citr = current_model.begin(); citr != current_model.end(); ++citr)
        {
            cv::Vec3i voxel_coord = citr.VoxelCoord();
            float d, w;
            cv::Vec3b color;
            citr->RetriveData(&d, &w, &color);
            if (w > 0)
            {
                cv::Vec3f world_coord = current_model.Voxel2World((cv::Vec3f(voxel_coord)));
//                ceres::CostFunction* cost_function =
//                        new TSDFAlignCostFunction3DScaleWeight2DR(&scene_tsdf, world_coord, d, problem_sample_trans[sample_i][2]);

                ceres::CostFunction* cost_function =
                        new TSDFAlignCostFunction2DRObsWeight(&scene_tsdf, world_coord + cv::Vec3f(0, 0, 0.5), d,
                                                              problem_sample_trans[sample_i][2], voxel_coord);

                problem.AddResidualBlock(cost_function, NULL,
                                         &(problem_sample_rotations[sample_i]),
                                         problem_sample_trans[sample_i].val,
                                         problem_sample_scales3d[sample_i].val);
                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].val, 0, 0.01);
                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].val, 1, 0.01);
                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].val, 2, 0.01);
                cnt++;
            }
        }  // end for citr
//        if (sample_i == 0) // fix the scale of the first sample
//        {
//            problem.SetParameterBlockConstant(problem_sample_scales3d[sample_i].val);
//        }
    }  // end for sample_i

    // add scale consistency blocks
    cout << "lambda_scale_diff: " << lambda_scale_diff << endl;
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        cv::Vec3d& current_scale3d = problem_sample_scales3d[sample_i];
        cv::Vec3d& current_model_scale = problem_model_scales[sample_model_assignment[sample_i]];
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignAverageScalesFunctor, 3, 3, 3>(
                    new TSDFAlignAverageScalesFunctor());
        problem.AddResidualBlock(
                    cost_function,
                    new ScaledLoss(NULL, lambda_scale_diff, TAKE_OWNERSHIP),
                    current_scale3d.val, current_model_scale.val);
    }

    // add rotation regularization blocks
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignRotationRegularizationFunctor, 1, 1>(
                    new TSDFAlignRotationRegularizationFunctor(problem_sample_rotations[sample_i]));
        problem.AddResidualBlock(
                    cost_function,
                    new ScaledLoss(NULL, lambda_reg_rot, TAKE_OWNERSHIP),
                    &(problem_sample_rotations[sample_i]));
    }

    // add scale regularization blocks
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignScaleRegularizationFunctor, 3, 3>(
                    new TSDFAlignScaleRegularizationFunctor(problem_sample_scales3d[sample_i].val));
        problem.AddResidualBlock(cost_function,
                                 new ScaledLoss(NULL, lambda_reg_scale, TAKE_OWNERSHIP),
                                 (problem_sample_scales3d[sample_i].val));
    }

    // add translation regularization blocks
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignTransRegularizationFunctor, 2, 2>(
                    new TSDFAlignTransRegularizationFunctor(problem_sample_trans[sample_i].val));
        problem.AddResidualBlock(
                    cost_function,
                    new ScaledLoss(NULL, lambda_reg_trans, TAKE_OWNERSHIP),
                    (problem_sample_trans[sample_i].val));
    }

    // target z scale regularization
    // const std::vector<float>& tscales = pca_options.target_zscales;
    //std::vector<double> tscales(sample_num);
    //std::vector<double> cur_scales(sample_num);
    //for (int i = 0; i < sample_num; ++i)
    //{
    //    if (gammas[i] > inlier_thresh) continue;
    //    cur_scales[i] = (problem_sample_scales3d[i].val[2]);
    //}
    //Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
    //Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
    //cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, pca_options.boundingbox_size, pca_options.min_model_weight);
    //ExtractSamplesFromAffineTransform(scene_tsdf, *transforms, grid_info, &samples, &weights);
    //ComputeTargetZScaleOneCluster(samples, weights, cur_scales,
    //                              grid_info, &tscales, 0.00001);
    //for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    //{
    //    if (gammas[sample_i] > inlier_thresh) continue;
    //    ceres::CostFunction* cost_function
    //            = new AutoDiffCostFunction<TSDFAlignTargetZScaleRegularizationFunctor, 1, 3>(
    //                new TSDFAlignTargetZScaleRegularizationFunctor(tscales[sample_i]));
    //    problem.AddResidualBlock(cost_function,
    //                             new ScaledLoss(NULL, lambda_reg_z_scale, TAKE_OWNERSHIP),
    //                             (problem_sample_scales3d[sample_i].val));
    //    cout << lambda_reg_z_scale << endl;
    //}

    // solving
    std::cout << "begin solving." << std::endl;
    ceres::Solver::Options options;
    //only for debugging
    /////////////////////////////////////////
    bfs::path write_dir = bfs::path(save_path).parent_path()/"iterate_alignment_opt_res";
    bfs::create_directories(write_dir);
    string cur_save_path = (write_dir/bfs::path(save_path).stem()).string();
    options.update_state_every_iteration = true;
    //std::unique_ptr<ceres::IterationCallback> debug_callback(new OutputObbCallback
    //                                                         (
    //                                                        &problem_sample_rotations,
    //                                                        &problem_sample_scales3d,
    //                                                        &problem_sample_trans,
    //                                                        &problem_model_scales,
    //                                                        cur_save_path + "_in_opt")
    //                                                         );
    cpu_tsdf::TSDFHashing::Ptr scene_ptr(new cpu_tsdf::TSDFHashing);
    *scene_ptr = scene_tsdf;
    //std::unique_ptr<ceres::IterationCallback> debug_callback(new OutputObbCallback
    //                                                         (
    //                                                        &problem_sample_rotations,
    //                                                        &problem_sample_scales3d,
    //                                                        &problem_sample_trans,
    //                                                        &problem_model_scales,
    //                                                        cur_save_path + "_in_opt",
    //                                                        scene_ptr,
    //                                                        pca_options,
    //                                                        sample_model_assignment, gammas)
    //                                                         );
    // options.callbacks.push_back(debug_callback.get());
    /////////////////////////////////////////
    options.max_num_iterations = 100;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.check_gradients = true;
    // options.gradient_check_relative_precision = 1;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    //debug
//#define TEST1
#ifdef TEST1
    double d_scalez = 0.1;
    double init_scale = 0.1;
    printf ("initial scalez: %f\n", problem_sample_scales3d[0][2]);
    std::vector<std::pair<float, float>> plotdata;
    cpu_tsdf::PCAOptions tmp_options = pca_options;
    string init_path = tmp_options.save_path;
    float old_scale = problem_sample_scales3d[0][2];
    const int start_scale_cnt = 0;
    for (int tt = start_scale_cnt; tt < 300; tt+=1)
    {
        double cur_scalez = d_scalez * tt + init_scale;
        double cost = -1;
        problem_sample_scales3d[0][2] = cur_scalez;
        problem.Evaluate(Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
        //printf("%f %f\n", cur_scalez, cost);
        plotdata.push_back(std::make_pair(cur_scalez, cost));


        //tmp_options.offset = Eigen::Vector3f(-0.5, -0.5, 0);
        tmp_options.save_path = init_path + "_house_debug_" + boost::lexical_cast<string>(tt) + "_curscale_" + boost::lexical_cast<string>(cur_scalez);
        for (int sample_i = 0; sample_i < sample_num; ++sample_i)
        {
            cv::Matx33d rotation_mat;
            //cv::Rodrigues( cv::Vec3f(problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
            cv::Rodrigues( cv::Vec3d(0, 0, problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
            Eigen::Matrix3f rotation_mat_eigen = utility::CvMatxToEigenMat(cv::Matx33f(rotation_mat));
            Eigen::Vector3f scale3d_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_scales3d[sample_i]));
            Eigen::Vector3f trans_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_trans[sample_i]))
                    + rotation_mat_eigen * scale3d_eigen.asDiagonal() * Eigen::Vector3f(0, 0, 0.5);
            Eigen::Matrix4f affine_trans = Eigen::Matrix4f::Zero();
            affine_trans.block<3, 3>(0, 0) = rotation_mat_eigen * scale3d_eigen.asDiagonal();
            affine_trans.block<3, 1>(0, 3) = trans_eigen;
            affine_trans(3, 3) = 1;
            //(*transforms)[sample_i].matrix() = affine_trans;
//            cout << "sample: " << sample_i << endl;
//            cout << "optimized rotation: \n" << rotation_mat << endl;
//            cout << "optimized scale: \n" << scale3d_eigen << endl;
//            cout << "optimized trans: \n" << trans_eigen << endl;

            Eigen::SparseVector<float> sample;
            Eigen::SparseVector<float> weight;
            // cpu_tsdf::ExtractOneSampleFromAffineTransform(scene_tsdf,  Eigen::Affine3f(affine_trans), tmp_options, &sample, &weight);
            std::vector<Eigen::Affine3f> transvec(1);
            transvec[0] = Eigen::Affine3f(affine_trans);
            cpu_tsdf::WriteAffineTransformsAndTSDFs(scene_tsdf, transvec, tmp_options, true);
        }
    }
    for (int ti = 0; ti < plotdata.size(); ++ti)
    {
        printf("%d %f %f\n", start_scale_cnt+ti, plotdata[ti].first, plotdata[ti].second);
    }
    problem_sample_scales3d[0][2] = old_scale;
#endif
    //exit(1);

    // get the solution
    std::cout << "solving finished." << std::endl;
    // int solver_result_cnt = 0;
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        // if (gammas[sample_i] > inlier_thresh) continue;
        cv::Matx33d rotation_mat;
        cv::Rodrigues( cv::Vec3d(0, 0, problem_sample_rotations[sample_i]), rotation_mat);  // mind type...
        CHECK_LE(rotation_mat(1, 0) * rotation_mat(0, 1), 0);
        Eigen::Matrix3f rotation_mat_eigen = utility::CvMatxToEigenMat(cv::Matx33f(rotation_mat));
        Eigen::Vector3f scale3d_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_scales3d[sample_i]));
        Eigen::Vector3f trans_eigen = utility::CvVectorToEigenVector3(cv::Vec3f(problem_sample_trans[sample_i])) +
                rotation_mat_eigen * scale3d_eigen.asDiagonal() * Eigen::Vector3f(0, 0, 0.5);
        Eigen::Matrix4f affine_trans = Eigen::Matrix4f::Zero();
        affine_trans.block<3, 3>(0, 0) = rotation_mat_eigen * scale3d_eigen.asDiagonal();
        CHECK_GE(scale3d_eigen[0], 0);
        CHECK_GE(scale3d_eigen[1], 0);
        CHECK_GE(scale3d_eigen[2], 0);
        affine_trans.block<3, 1>(0, 3) = trans_eigen;
        affine_trans(3, 3) = 1;
        (*transforms)[sample_i].matrix() = affine_trans;
        cout << "sample: " << sample_i << endl;
        cout << "optimized rotation: \n" << rotation_mat << endl;
        cout << "optimized scale: \n" << scale3d_eigen << endl;
        cout << "optimized trans: \n" << trans_eigen << endl;
    }
    for (int model_i = 0; model_i < model_num; ++model_i)
    {
        (*model_scales)[model_i] = utility::CvVectorToEigenVector3(cv::Vec3f(problem_model_scales[model_i]));
    }
    pca_options.lambda_observation *= 0.3; // decays during iteration
    return true;
}

bool OptimizeTransformAndScalesImplRobustLoss1(
        const TSDFHashing& scene_tsdf,
        const std::vector<const TSDFHashing*> model_reconstructed_samples,
        const std::vector<int>& sample_model_assignment,
        const std::vector<double>& gammas, // outliers
        tsdf_utility::OptimizationParams& params,
        std::vector<tsdf_utility::OrientedBoundingBox>* obbs, /*#samples, input and output*/
        std::vector<Eigen::Vector3f>* model_scales /*#model/clusters, input and output*/
        )
{
    const float min_model_weight = params.min_meshing_weight;
    const float lambda_average_scale = params.lambda_average_scale;
    const float lambda_obs = params.lambda_observation;
    const float lambda_regularization = params.lambda_regularization;

    using namespace ceres;
    using namespace Eigen;
    using namespace std;
    const int sample_num = model_reconstructed_samples.size();
    const int model_num = model_scales->size();
    TSDFAlignRobustCostFunction2DRObsWeight::min_model_weight = min_model_weight;
    TSDFAlignRobustCostFunction2DRObsWeight::max_model_weight = TSDFHashing::getVoxelMaxWeight() + 1.0;
    TSDFAlignRobustCostFunction2DRObsWeight::sqrt_lambda_obs = sqrt(lambda_obs);

    cout << "sqrt_lambda_obs: " << TSDFAlignRobustCostFunction2DRObsWeight::sqrt_lambda_obs << endl;
    ceres::Problem problem;
    // saves all the parameters and pass them to problem
    std::vector<double> problem_sample_rotations(sample_num);
    std::vector<Vector3d> problem_sample_scales3d(sample_num);  // used in scale consistency
    std::vector<Vector3d> problem_sample_trans(sample_num);
    std::vector<Vector3d> problem_model_scales(model_num);
    for (int i = 0; i < model_num; ++i) problem_model_scales[i] = (*model_scales)[i].cast<double>();

    //std::unique_ptr<ceres::LossFunction> robust_loss_function(new HuberLoss(robust_loss_outlier_threshold));
    std::unique_ptr<ceres::LossFunction> robust_loss_function(new ceres::TrivialLoss());
    static const double inlier_thresh = 1e-5;
    int cnt = 0;
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        const TSDFHashing& current_model = *(model_reconstructed_samples[sample_i]);

        problem_sample_rotations[sample_i] = (*obbs)[sample_i].AngleRangePosNegPI();
        problem_sample_scales3d[sample_i] = (*obbs)[sample_i].SideLengths().cast<double>();
        problem_sample_trans[sample_i] = (*obbs)[sample_i].BottomCenter().cast<double>();

        if (gammas[sample_i] > inlier_thresh) continue;

        // problem
        std::cout << "building problem for sample " << sample_i << std::endl;
        std::cout << "building problem for inlier sample " << sample_i << std::endl;
        cout << "initial rotation: \n" << problem_sample_rotations[sample_i] << endl;
        cout << "initial scale: \n" << problem_sample_scales3d[sample_i] << endl;
        cout << "initial trans: \n" << problem_sample_trans[sample_i] << endl;
        for (TSDFHashing::const_iterator citr = current_model.begin(); citr != current_model.end(); ++citr)
        {
            cv::Vec3i voxel_coord = citr.VoxelCoord();
            float d, w;
            cv::Vec3b color;
            citr->RetriveData(&d, &w, &color);
            if (w > 0)
            {
                cv::Vec3f world_coord = current_model.Voxel2World((cv::Vec3f(voxel_coord)));
                //cout << "d " << d << endl;
                //cout << "w " << w << endl;
                //cout << "color " << color << endl;
                //cout << "voxel " << voxel_coord << endl;
                //cout << "worldcoord" << world_coord<< endl;
                ceres::CostFunction* cost_function =
                        new TSDFAlignRobustCostFunction2DRObsWeight(&scene_tsdf, world_coord, d,
                                                              problem_sample_trans[sample_i][2], voxel_coord, robust_loss_function.get());
                problem.AddResidualBlock(cost_function, NULL,
                                         &(problem_sample_rotations[sample_i]),
                                         problem_sample_trans[sample_i].data(),
                                         problem_sample_scales3d[sample_i].data());
                // scales GT than 0.01
                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].data(), 0, 0.01);
                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].data(), 1, 0.01);
                problem.SetParameterLowerBound(problem_sample_scales3d[sample_i].data(), 2, 0.01);
                cnt++;
            }
        }  // end for citr
    }  // end for sample_i

    // add scale consistency blocks
    cout << "lambda_average_scale: " << lambda_average_scale  << endl;
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        Vector3d& current_scale3d = problem_sample_scales3d[sample_i];
        Vector3d& current_model_scale = problem_model_scales[sample_model_assignment[sample_i]];
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignAverageScalesFunctor, 3, 3, 3>(
                    new TSDFAlignAverageScalesFunctor());
        problem.AddResidualBlock(
                    cost_function,
                    new ScaledLoss(NULL, lambda_average_scale , TAKE_OWNERSHIP),
                    current_scale3d.data(), current_model_scale.data());
    }

    // add rotation regularization blocks
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignRotationRegularizationFunctor, 1, 1>(
                    new TSDFAlignRotationRegularizationFunctor(problem_sample_rotations[sample_i]));
        problem.AddResidualBlock(
                    cost_function,
                    new ScaledLoss(NULL, lambda_regularization, TAKE_OWNERSHIP),
                    &(problem_sample_rotations[sample_i]));
    }

    // add scale regularization blocks
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignScaleRegularizationFunctor, 3, 3>(
                    new TSDFAlignScaleRegularizationFunctor(problem_sample_scales3d[sample_i].data()));
        problem.AddResidualBlock(cost_function,
                                 new ScaledLoss(NULL, lambda_regularization, TAKE_OWNERSHIP),
                                 (problem_sample_scales3d[sample_i].data()));
    }

    // add translation regularization blocks
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        if (gammas[sample_i] > inlier_thresh) continue;
        ceres::CostFunction* cost_function
                = new AutoDiffCostFunction<TSDFAlignTransRegularizationFunctor, 2, 2>(
                    new TSDFAlignTransRegularizationFunctor(problem_sample_trans[sample_i].data()));
        problem.AddResidualBlock(
                    cost_function,
                    new ScaledLoss(NULL, lambda_regularization, TAKE_OWNERSHIP),
                    (problem_sample_trans[sample_i].data()));
    }

    // solving
    std::cout << "begin solving." << std::endl;
    ceres::Solver::Options options;
    //only for debugging
    /////////////////////////////////////////
    //bfs::path write_dir = bfs::path(params.save_path).parent_path()/"iterate_alignment_opt_res";
    //bfs::create_directories(write_dir);
    //string cur_save_path = (write_dir/bfs::path(params.save_path).stem()).string();
    //options.update_state_every_iteration = true;
    //cpu_tsdf::TSDFHashing::Ptr scene_ptr(new cpu_tsdf::TSDFHashing);
    //*scene_ptr = scene_tsdf;
    //std::unique_ptr<ceres::IterationCallback> debug_callback(new OutputObbCallback
    //                                                         (
    //                                                        &problem_sample_rotations,
    //                                                        &problem_sample_scales3d,
    //                                                        &problem_sample_trans,
    //                                                        &problem_model_scales,
    //                                                        cur_save_path + "_in_opt",
    //                                                        scene_ptr,
    //                                                        pca_options,
    //                                                        sample_model_assignment, gammas)
    //                                                         );
    // options.callbacks.push_back(debug_callback.get());
    /////////////////////////////////////////
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // get the solution
    std::cout << "solving finished." << std::endl;
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        (*obbs)[sample_i] = tsdf_utility::OrientedBoundingBox(problem_sample_scales3d[sample_i].cast<float>(),
                                                           problem_sample_trans[sample_i].cast<float>(),
                                                           problem_sample_rotations[sample_i]);
    }
    for (int model_i = 0; model_i < model_num; ++model_i)
    {
        (*model_scales)[model_i] = problem_model_scales[model_i].cast<float>();
    }
    params.lambda_observation *= 0.3; // decays during iteration
    return true;
}

void ComputeTargetZScaleOneCluster(
        std::vector<const TSDFHashing *> tsdfs,
        const std::vector<double>& current_zscale,
        std::vector<double> *target_scale,
        const Eigen::Vector3i &boundingbox_size, const float percentile)
{
    cout << "computing z scale " << endl;
    std::vector<float> highest_percentiles(tsdfs.size());
    for (int i = 0; i < tsdfs.size(); ++i)
    {
        cpu_tsdf::TSDFHashing::Ptr cur_tsdf(new cpu_tsdf::TSDFHashing);
        *cur_tsdf = *(tsdfs[i]);
        std::vector<float> heights;
        cout << "begin clean z scale " << i << endl;
        CleanTSDF(cur_tsdf, -1);
        if (i == 2)
        {
            cpu_tsdf::WriteTSDFModel(cur_tsdf, "/home/dell/test1.ply", false, true, 0);
        }

        for (cpu_tsdf::TSDFHashing::iterator itr = cur_tsdf->begin(); itr != cur_tsdf->end(); ++itr)
        {
            float d, w;
            cv::Vec3b color;
            if (itr->RetriveData(&d, &w, &color) && w > 0 && d != 0)
            {
                Eigen::Vector3i cur_vcoord = utility::CvVectorToEigenVector3(itr.VoxelCoord());
                double z_height_normed = (double(cur_vcoord[2]))/double(boundingbox_size[2]);
                heights.push_back(z_height_normed);
            }
        }
        cout << "hight size " << heights.size() << endl;
        // int nth = round(percentile * heights.size());
        int nth = std::min<int>(20, heights.size() - 1);
        std::nth_element(heights.begin(), heights.begin() + nth, heights.end(), std::greater<float>());
        highest_percentiles[i] = heights[nth];
        cout << i << "th highest perventile " << highest_percentiles[i] << endl;
    }

    // align the scales to the first sample
    target_scale->resize(tsdfs.size());
    for (int i = 0; i < tsdfs.size(); ++i)
    {
        (*target_scale)[i] = (highest_percentiles[i]/highest_percentiles[0]) * current_zscale[i];
        cout << i << "th original scale: " << current_zscale[i] << endl;
        cout << i << "th scale target: " << (*target_scale)[i] << endl;
        cout << i << "ith scale target scaling factor: " <<  (highest_percentiles[i]/highest_percentiles[0])  << endl;
    }
    return;
}

void ComputeTargetZScaleOneCluster(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const std::vector<double> &cur_zscale,
        const TSDFGridInfo &grid_info,
        std::vector<double> *target_scale, const float percentile)
{
    std::vector<cpu_tsdf::TSDFHashing::Ptr> tsdfs;
    ConvertDataMatrixToTSDFs(samples, weights, grid_info, &tsdfs);

    std::vector<const cpu_tsdf::TSDFHashing*> ptsdfs(tsdfs.size());
    for (int i = 0; i < tsdfs.size(); ++i)
    {
        ptsdfs[i] = tsdfs[i].get();
    }
    return ComputeTargetZScaleOneCluster(ptsdfs, cur_zscale, target_scale, grid_info.boundingbox_size(), percentile);
}

}  // namespace cpu_tsdf


