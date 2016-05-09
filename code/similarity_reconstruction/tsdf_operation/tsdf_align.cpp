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
};
double TSDFAlignCostFunction2DRObsWeight::min_model_weight = 0.0;
double TSDFAlignCostFunction2DRObsWeight::max_model_weight = 5.0;
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
        if (tsdf_model->RetriveDataFromWorldCoord(cv::Vec3f(translated_pt), &model_val, &model_weight) &&
                model_weight > min_model_weight)
        {
            error = model_val - template_point_val;  // the derivative is w.r.t model_val
            (loss_)->Evaluate(error * error, robust_squared_errors);
            residuals[0] = sqrt(model_weight * robust_squared_errors[0]);
            residuals[1] = weight_obs * sqrt_lambda_obs * sqrt(max_model_weight - model_weight);
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
double TSDFAlignRobustCostFunction2DRObsWeight::max_model_weight = 5.0;
double TSDFAlignRobustCostFunction2DRObsWeight::sqrt_lambda_obs = 1.0;

class TSDFAlignAverageScalesFunctor {
public:
    TSDFAlignAverageScalesFunctor() {}
    template <typename T>
    bool operator()(const T* const scale3d , const T* const average_scale, T* e) const {
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

namespace cpu_tsdf
{

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

    ceres::Problem problem;
    // saves all the parameters and pass them to problem
    std::vector<double> problem_sample_rotations(sample_num);
    std::vector<Vector3d> problem_sample_scales3d(sample_num);  // used in scale consistency
    std::vector<Vector3d> problem_sample_trans(sample_num);
    std::vector<Vector3d> problem_model_scales(model_num);
    for (int i = 0; i < model_num; ++i) problem_model_scales[i] = (*model_scales)[i].cast<double>();

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
        for (TSDFHashing::const_iterator citr = current_model.begin(); citr != current_model.end(); ++citr)
        {
            cv::Vec3i voxel_coord = citr.VoxelCoord();
            float d, w;
            cv::Vec3b color;
            citr->RetriveData(&d, &w, &color);
            if (w > 0)
            {
                cv::Vec3f world_coord = current_model.Voxel2World((cv::Vec3f(voxel_coord)));
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

}  // namespace cpu_tsdf


