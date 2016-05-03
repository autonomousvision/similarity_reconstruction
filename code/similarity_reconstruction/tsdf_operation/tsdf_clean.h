/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <cstdint>
#include <unordered_map>

#include <pcl/pcl_macros.h>
#include <pcl/PolygonMesh.h>

#include "tsdf_representation/tsdf_hash.h"

namespace cpu_tsdf {
class TSDFHashing;
}

namespace cpu_tsdf {
bool ComputeTSDFCleanVoxels(const cpu_tsdf::TSDFHashing::Ptr tsdf,
        int compo_thresh,
        std::vector<Eigen::Vector3i> *voxel_to_remove
        , int neighborst, int neighbored);
void CleanTSDFVoxels(const std::vector<Eigen::Vector3i>& voxel_to_remove, cpu_tsdf::TSDFHashing::Ptr tsdf);
void CleanTSDF(cpu_tsdf::TSDFHashing::Ptr tsdf,
        int compo_thresh
        , int neighborst = 0, int neighbored = 1);
void CleanTSDFs(
        std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdfs,
        int compo_thresh,
        int neighborst = 0,
        int neighbored = 1
        );
void CleanTSDFSampleVector(const cpu_tsdf::TSDFHashing& scene_tsdf,
        const Eigen::SparseVector<float>& sample,
        const Eigen::Vector3i boundingbox_size,
        const int compo_thresh,
        Eigen::SparseVector<float>* weight
        , int neighborst, int neighbored);
void CleanTSDFSampleMatrix(const cpu_tsdf::TSDFHashing& scene_tsdf,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
        const Eigen::Vector3i boundingbox_size,
        const int compo_thresh,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* weights
        , int neighborst, int neighbored);
void ComputeRemovedVoxelsFromRemovedTris(
        const std::vector<bool>& reserved_tri,
        const pcl::PolygonMesh& mesh,
        const cpu_tsdf::TSDFHashing* tsdf,
        std::vector<Eigen::Vector3i>* removed_voxels,
        int neighborst = -1,
        int neighbored = 2);
void ComputeRemovedVoxelsFromRemovedVerts(
        const std::vector<bool>& reserved_verts,
        const pcl::PolygonMesh& mesh,
        const cpu_tsdf::TSDFHashing* tsdf,
        std::vector<Eigen::Vector3i>* removed_voxels,
        int neighborst = -1,
        int neighbored = 2);

void CleanTSDFFromMeshVerts(
        cpu_tsdf::TSDFHashing::Ptr tsdf,
        const pcl::PolygonMesh& mesh,
        const std::vector<bool>& reserved_verts,
        int neighborst = -1,
        int neighbored = 2);

bool CleanMesh(
        pcl::PolygonMesh& mesh,
        float compo_thresh
        );


bool CleanNoiseInSamples(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const std::vector<int>& model_assign_idx,
        const std::vector<double>& outlier_gammas,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* pweights, Eigen::SparseMatrix<float, Eigen::ColMajor> *valid_obs_weight_mat,
        float counter_thresh, float pos_trunc, float neg_trunc);

bool CleanNoiseInSamplesOneCluster(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* weights, Eigen::SparseVector<float> *valid_obs_positions,
        float counter_thresh, float pos_trunc, float neg_trunc);

}
