/*
 * IO related functions
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <matio.h>

#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"

namespace cpu_tsdf
{

/******************************
  read/write oriented obbs
******************************/
// obb to ply
bool WriteOrientedBoundingboxPly(const cpu_tsdf::OrientedBoundingBox& obb, const std::string &filename);

bool WriteOrientedBoundingboxPly(const Eigen::Matrix3f &orientation, const Eigen::Vector3f &offset, const Eigen::Vector3f &lengths, const std::string &filename);

bool WriteOrientedBoundingboxesPly(const std::vector<cpu_tsdf::OrientedBoundingBox> &obb, const std::string &filename, const std::vector<double>& outlier_gammas = std::vector<double>());

bool WriteAffineTransformPly(const Eigen::Affine3f& affine, const std::string& filename);

bool WriteAffineTransformsPly(const std::vector<Eigen::Affine3f>& obbs, const std::string& filename);

// obb to/from text
bool WriteOrientedBoundingBoxes(
        const std::string &filename,
        const std::vector<cpu_tsdf::OrientedBoundingBox> obbs,
        const std::vector<int> sample_model_assign,
        const std::vector<bool> is_train_sample = std::vector<bool>());

bool ReadOrientedBoundingBoxes(
        const std::string &filename,
        std::vector<cpu_tsdf::OrientedBoundingBox> *obbs,
        std::vector<int> *sample_model_assign,
        std::vector<bool> *is_train_sample = NULL);

/**************************
 * Read/Write TSDF models
 * *************************/
bool WriteTSDFModel(TSDFHashing::ConstPtr tsdf_model, const std::string& output_filename,
                   bool save_tsdf_bin, bool save_mesh, float mesh_min_weight);

bool WriteTSDFModels(const std::vector<TSDFHashing::Ptr>& tsdf_models, const std::string& output_filename,
                     bool save_tsdf_bin, bool save_mesh, float mesh_min_weight,
                     const std::vector<double>& outlier_gammas = std::vector<double>());

/* deprecated */
void WriteTSDFMesh(TSDFHashing::ConstPtr tsdf_model, float mesh_min_weight, const std::string &output_filename, const bool save_ascii);

/**************************
 * Read/Write TSDF models in obb/affine transforms
 * *************************/
bool WriteOBBsAndTSDFs(
        const cpu_tsdf::TSDFHashing& scene_tsdf,
        const std::vector<tsdf_utility::OrientedBoundingBox>& obbs,
        const cpu_tsdf::TSDFGridInfo &tsdf_info,
        const std::string& save_path,
        bool save_text_data = false);

bool WriteAffineTransformsAndTSDFs(
        const cpu_tsdf::TSDFHashing& scene_tsdf,
        const std::vector<Eigen::Affine3f>& affine_transforms,
        const cpu_tsdf::TSDFGridInfo &tsdf_info,
        const std::string& save_path,
        bool save_text_data = false);

//bool WriteObbsAndTSDFs(
//        const cpu_tsdf::TSDFHashing& scene_tsdf,
//        const std::vector<cpu_tsdf::OrientedBoundingBox>& obbs,
//        const Eigen::Vector3i& sample_obb_voxel_size,
//        const std::string& save_path,
//        const float vmin_model_weight,
//        bool save_text_data = false);

inline bool WriteObbsAndTSDFs(
        const cpu_tsdf::TSDFHashing& scene_tsdf,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& obbs,
        const cpu_tsdf::TSDFGridInfo &tsdf_info,
        const std::string& save_path,
        bool save_text_data = false)
{
    std::vector<Eigen::Affine3f> affines;
    OBBsToAffines(obbs, &affines);
    return WriteAffineTransformsAndTSDFs(scene_tsdf,
                                         affines,
                                         tsdf_info,
                                         save_path,
                                         save_text_data);
}

/* deprecated */
//bool WriteAffineTransformsAndTSDFs(const cpu_tsdf::TSDFHashing& scene_tsdf,
//                           const std::vector<Eigen::Affine3f>& affine_transforms, const PCAOptions &options, bool save_text_data = false);

/**************************
 * Read/Write TSDF models in feature vectors
 * *************************/
void WriteTSDFsFromMatNoWeight(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat,
        const Eigen::Vector3i& boundingbox_size,
        const float voxel_length,
        const Eigen::Vector3f& offset,
        const float max_dist_pos,
        const float max_dist_neg,
        const std::string& save_filepath);

void WriteTSDFsFromMatWithWeight(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
        const Eigen::Vector3i& boundingbox_size,
        const float voxel_length,
        const Eigen::Vector3f& offset,
        const float max_dist_pos,
        const float max_dist_neg,
        const std::string& save_filepath);

inline void WriteTSDFsFromMatNoWeight(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat,
        const TSDFGridInfo& tsdf_info,
        const std::string& save_filepath)
{
    WriteTSDFsFromMatNoWeight(data_mat, tsdf_info.boundingbox_size(), tsdf_info.voxel_lengths()[0],
                      tsdf_info.offset(), tsdf_info.max_dist_pos(), tsdf_info.max_dist_neg(), save_filepath);
}

inline void WriteTSDFsFromVectorsNoWeight(
        const std::vector<Eigen::SparseVector<float>>& data_vecs,
        const TSDFGridInfo& tsdf_info,
        const std::string& save_filepath)
{
    WriteTSDFsFromMatNoWeight(SparseVectorsToEigenMat(data_vecs), tsdf_info.boundingbox_size(), tsdf_info.voxel_lengths()[0],
                      tsdf_info.offset(), tsdf_info.max_dist_pos(), tsdf_info.max_dist_neg(), save_filepath);
}

inline void WriteTSDFsFromMatWithWeight(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
        const TSDFGridInfo& tsdf_info,
        const std::string& save_filepath)
{
    WriteTSDFsFromMatWithWeight(data_mat, weight_mat, tsdf_info.boundingbox_size(), tsdf_info.voxel_lengths()[0],
                      tsdf_info.offset(), tsdf_info.max_dist_pos(), tsdf_info.max_dist_neg(), save_filepath);
}

inline void WriteTSDFsFromVectorsWithWeight(
        const std::vector<Eigen::SparseVector<float>>& data_vecs,
        const std::vector<Eigen::SparseVector<float>>& weight_vecs,
        const TSDFGridInfo& tsdf_info,
        const std::string& save_filepath)
{
    WriteTSDFsFromMatWithWeight(
                SparseVectorsToEigenMat(data_vecs),
                SparseVectorsToEigenMat(weight_vecs),
                tsdf_info.boundingbox_size(),
                tsdf_info.voxel_lengths()[0],
                tsdf_info.offset(), tsdf_info.max_dist_pos(),
                tsdf_info.max_dist_neg(),
                save_filepath);
}


void WriteTSDFsFromMatWithWeight_Matlab(const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
        const TSDFGridInfo& tsdf_info,
        const std::string& save_filepath, const std::string &var_suffix = std::string());

bool WriteTSDFFromVectorWithWeight_Matlab(const Eigen::SparseVector<float>& data_mat,
        const Eigen::SparseVector<float>& weight_mat,
        const TSDFGridInfo& tsdf_info,
        const std::string& save_filepath, const std::string &var_suffix = std::string());

bool Write3DArrayMatlab(const std::vector<float>& data,
                        const Eigen::Vector3i& bbsize, const std::string& varname, const std::string& save_filepath);

bool Write3DArrayMatlab(const Eigen::VectorXf& data,
                        const Eigen::Vector3i& bbsize, const std::string& varname, const std::string& save_filepath);

bool Read3DArrayMatlab(const std::string& filepath, const std::string& varname,
                       Eigen::VectorXf* data);

bool Read3DArraysMatlab(const std::string &filepath, const std::string &filter, const Eigen::Vector3i& bbsize, std::vector<Eigen::VectorXf> *datas);

bool ReadMatrixMatlab(const std::string &filepath, const std::string & varname, Eigen::MatrixXf *matrix);

bool WriteMatrixMatlab(const std::string &filepath, const std::string &variable_name, const Eigen::MatrixXf &matrix);














}
