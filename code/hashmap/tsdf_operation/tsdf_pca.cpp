#include "tsdf_pca.h"
#include <cstdio>
#include <Eigen/Eigen>
#include <sys/types.h>
#include <sys/wait.h>

// #include "tsdf_operation/tsdf_align.h"
#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_representation/voxel_hashmap.h"
#include "common/utility/common_utility.h"
#include "common/utility/pcl_utility.h"
#include "common/utility/eigen_utility.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "utility/utility.h"
#include "tsdf_operation/tsdf_io.h"
#include "tsdf_operation/tsdf_transform.h"
// #include "tsdf_operation/tsdf_slice.h"
// #include "tsdf_operation/tsdf_clean.h"
// #include "tsdf_operation/tsdf_slice.h"
#include "common/utility/matlab_utility.h"
#include "tsdf_utility.h"
#include "utility/oriented_boundingbox.h"

using Eigen::ColMajor;
using Eigen::RowMajor;
using std::cout;
using std::endl;
using std::cerr;
using utility::min_vec3;
using utility::max_vec3;
using utility::flattenVertices;
using utility::CvVectorToEigenVector3;
using utility::EigenVectorToCvVector3;
namespace bfs = boost::filesystem;
using namespace std;

namespace {
using namespace cpu_tsdf;
// Declarations for (PCA related) functions only used in this file
void OrthogonalizeVector(const Eigen::SparseMatrix<float, Eigen::ColMajor>& base_mat, const int current_components, Eigen::VectorXf* base_vec, float *norm);

bool InitializeDataMatrixFromTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
                                   Eigen::SparseMatrix<float, Eigen::ColMajor> *centrlized_data_mat,
                                   Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
                                   Eigen::SparseVector<float> *mean_mat,
                                   cpu_tsdf::TSDFHashing* tsdf_template,
                                   Eigen::Vector3i* bounding_box);

void InitialEstimate(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat,
                     const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat,
                     const int component_num,
                     Eigen::SparseMatrix<float, Eigen::RowMajor> *base_mat_row_major, Eigen::MatrixXf *coeff_mat
                     );

void ComputeCoeffInEStepOneVec(const Eigen::SparseMatrix<float, Eigen::ColMajor>& centralized_data_mat,
                               const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
                               const Eigen::VectorXf &base_vec,
                               Eigen::VectorXf* coeff_vec);

void ComputeBasisInMStepOneVec(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
                               const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
                               const Eigen::VectorXf &coeff_vec,
                               Eigen::VectorXf *base_vec);

void ComputeCoeffInEStep(const Eigen::SparseMatrix<float, Eigen::ColMajor>& centralized_data_mat,
                         const Eigen::SparseMatrix<float, Eigen::ColMajor>& weight_mat,
                         const Eigen::SparseMatrix<float, Eigen::ColMajor>& base_mat,
                         Eigen::MatrixXf* coeff_mat);

void ComputeBasisInMStep(const Eigen::SparseMatrix<float, Eigen::RowMajor>& centralized_data_mat_row_major,
                         const Eigen::SparseMatrix<float, Eigen::RowMajor>& weight_mat_row_major,
                         const Eigen::MatrixXf& coeff_mat,
                         Eigen::SparseMatrix<float, Eigen::RowMajor>* base_mat_row_major
                         );



/**
 * @brief cpu_tsdf::OrthogonalizeVector
 * @param base_mat
 * @param current_components use [0, current_components) columns of base_mat for orthogonalization
 * @param base_vec Input vector to be orthogonalized, also Output orthogonalized vector
 * @param norm Output, the vector norm
 */
void OrthogonalizeVector(const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
                                   const int current_components,
                                   Eigen::VectorXf *base_vec, float* norm)
{
    for (int i = 0; i < current_components; ++i)
    {
        Eigen::VectorXf current_column = base_mat.col(i);
        *base_vec -= (base_vec->dot(current_column) * current_column);
    }
    *norm = base_vec->norm();
    base_vec->normalize();
    return;
}

/**
 * @brief cpu_tsdf::InitializeDataMatrixFromTSDFs
 * Compute multiple matrix for PCA from TSDFs
 * @param tsdf_models
 * @param centrlized_data_mat
 * @param weight_mat
 * @param mean_mat D * 1 mean mat
 * @param tsdf_template Mean TSDF
 * @param bounding_box Size of the bounding box
 * @return
 */
bool InitializeDataMatrixFromTSDFs(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
                                             Eigen::SparseMatrix<float, Eigen::ColMajor> *centrlized_data_mat,
                                             Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
                                             Eigen::SparseVector<float> *mean_mat,
                                             cpu_tsdf::TSDFHashing* tsdf_template,
                                             Eigen::Vector3i* bounding_box)  // also compute the mean
{
    std::cout << "Begin TSDFsToMatrix and averaging TSDFs. " << std::endl;
    if (tsdf_models.empty()) return false;

    // compute the bounding box of the models
    cv::Vec3f boundingbox_min, boundingbox_max;
    tsdf_models[0]->getBoundingBoxInWorldCoord(boundingbox_min, boundingbox_max);
    float voxel_length = FLT_MAX;
    float max_dist_pos = FLT_MIN;
    float max_dist_neg = FLT_MAX;
    for (int i = 1; i < tsdf_models.size(); ++i)
    {
        cv::Vec3f cur_min, cur_max;
        tsdf_models[i]->getBoundingBoxInWorldCoord(cur_min, cur_max);
        boundingbox_min = min_vec3(boundingbox_min, cur_min);
        boundingbox_max = max_vec3(boundingbox_max, cur_max);  // get intersection of all the bounding boxes
        voxel_length = std::min(voxel_length, tsdf_models[i]->voxel_length());
        float cur_max_dist_pos, cur_max_dist_neg;
        tsdf_models[i]->getDepthTruncationLimits(cur_max_dist_pos, cur_max_dist_neg);
        max_dist_pos = std::max(max_dist_pos, cur_max_dist_pos);
        max_dist_neg = std::min(max_dist_neg, cur_max_dist_neg);
    }
    float mean_sample_voxel_length = voxel_length /** 2*/;
    // tsdf_template->CopyHashParametersFrom(tsdf_models[0]);
    tsdf_template->Init(mean_sample_voxel_length, CvVectorToEigenVector3(boundingbox_min), max_dist_pos, max_dist_neg);
    Eigen::Vector3i voxel_bounding_box_size = CvVectorToEigenVector3((boundingbox_max - boundingbox_min)/mean_sample_voxel_length).cast<int>();
    const int data_dim = voxel_bounding_box_size[0] * voxel_bounding_box_size[1] * voxel_bounding_box_size[2];
    const int sample_num = tsdf_models.size();
    centrlized_data_mat->resize(data_dim, sample_num);
    weight_mat->resize(data_dim, sample_num);
    mean_mat->resize(data_dim);
    mean_mat->setZero();

    TSDFHashing::update_hashset_type brick_update_set;
    const int neighbor_limit = 1 * VoxelHashMap::kBrickSideLength;
    for (int i = 0; i < tsdf_models.size(); ++i)
    {
        const TSDFHashing& cur_tsdf = *(tsdf_models[i]);
        for (TSDFHashing::const_iterator citr = cur_tsdf.begin(); citr != cur_tsdf.end(); ++citr)
        {
            cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
            float d, w;
            cv::Vec3b color;
            citr->RetriveData(&d, &w, &color);
            if (w > 0)
            {
                cv::Vec3f world_coord = cur_tsdf.Voxel2World(cv::Vec3f(cur_voxel_coord));
                cv::Vec3i voxel_coord_template = cv::Vec3i(tsdf_template->World2Voxel(world_coord));
                tsdf_template->AddBrickUpdateList(voxel_coord_template, &brick_update_set);
            }  // end if
        }  // end for
    }  // end for

    // compute mean/data/weight matrices
    struct TemplateVoxelUpdater // functor to update every voxel of tsdf_template
    {
        TemplateVoxelUpdater(const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models,
                             const TSDFHashing* tsdf_template, const Eigen::Vector3i& bb_size, Eigen::SparseVector<float>* vmean_mat)
            : tsdf_models(tsdf_models), tsdf_template(tsdf_template), voxel_bounding_box_size(bb_size),
              valid_sample(tsdf_models.size(), false), cur_dists(tsdf_models.size()), cur_weights(tsdf_models.size()),
              mean_mat(vmean_mat) {}
        bool operator() (const cv::Vec3i& cur_voxel_coord, float* d, float* w, cv::Vec3b* color)
        {
            std::fill(valid_sample.begin(), valid_sample.end(), false);
            std::fill(cur_dists.begin(), cur_dists.end(), 0);
            std::fill(cur_weights.begin(), cur_weights.end(), 0);

            cv::Vec3f cur_world_coord = tsdf_template->Voxel2World(cv::Vec3f(cur_voxel_coord));
            const int data_dim_idx = cur_voxel_coord[2] +
                    (cur_voxel_coord[1] + cur_voxel_coord[0] * voxel_bounding_box_size[1]) * voxel_bounding_box_size[2];
            float final_d = 0;
            float final_w = 0;
            for (int i = 0; i < tsdf_models.size(); ++i)
            {
                float cur_d;
                float cur_w;
                if(!tsdf_models[i]->RetriveDataFromWorldCoord(cur_world_coord, &cur_d, &cur_w))
                {
                    continue;
                }
                valid_sample[i] = true;
                cur_dists[i] = cur_d;
                cur_weights[i] = cur_w;

                final_d += (cur_d * cur_w);
                final_w += cur_w;
            }
            if (final_w < 1e-5) return false;
            final_d /= (float)(final_w);
            *d = final_d;
            *w = final_w;
            *color = cv::Vec3b(255, 255, 255);
            for (int sample_dim_idx = 0; sample_dim_idx < valid_sample.size(); ++sample_dim_idx)
            {
                if (valid_sample[sample_dim_idx])
                {
                    data_list.push_back(Eigen::Triplet<float>(data_dim_idx, sample_dim_idx, cur_dists[sample_dim_idx] - final_d));
                    weight_list.push_back(Eigen::Triplet<float>(data_dim_idx, sample_dim_idx, cur_weights[sample_dim_idx]));
                }
            }
            mean_mat->coeffRef(data_dim_idx) = final_d;
            return true;
        }
        const std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdf_models;
        const TSDFHashing* tsdf_template;
        const Eigen::Vector3i voxel_bounding_box_size;
        std::vector<Eigen::Triplet<float>> data_list;
        std::vector<Eigen::Triplet<float>> weight_list;
        std::vector<uchar> valid_sample;
        std::vector<float> cur_dists;
        std::vector<float> cur_weights;
        Eigen::SparseVector<float>* mean_mat;
    };
    TemplateVoxelUpdater new_updater(tsdf_models, tsdf_template, voxel_bounding_box_size, mean_mat);
    tsdf_template->UpdateBricksInQueue(brick_update_set, new_updater);
    centrlized_data_mat->setFromTriplets(new_updater.data_list.begin(),
                                         new_updater.data_list.end());
    weight_mat->setFromTriplets(new_updater.weight_list.begin(),
                                new_updater.weight_list.end());
    *bounding_box = voxel_bounding_box_size;
    std::cout << "End averaging TSDFs. Initial estimate finished" << std::endl;
    return true;
}

/**
 * @brief cpu_tsdf::InitialEstimate
 * Do an initial estimate of the basis. May use random initialization or do an inital PCA
 * @param centralized_data_mat_row_major
 * @param weight_mat_row_major
 * @param component_num
 * @param base_mat_row_major Output, estimated base_mat
 * @param coeff_mat Output estimated coefficient mat
 */
void InitialEstimate(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
                               const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
                               const int component_num,
                               Eigen::SparseMatrix<float, Eigen::RowMajor> *base_mat_row_major, Eigen::MatrixXf *coeff_mat)
{

    const int sample_num = centralized_data_mat_row_major.cols();
    const int data_dim = centralized_data_mat_row_major.rows();
    base_mat_row_major->resize(data_dim, component_num);
    base_mat_row_major->setZero();
    base_mat_row_major->reserve(data_dim * sample_num * 0.7);
    std::cerr << "making initial estimate: " << std::endl;
    std::cerr << "date_dim: " << data_dim << std::endl;
    std::cerr << "sample_num: " << sample_num << std::endl;
    std::cerr << "data_mat_nnz: " << centralized_data_mat_row_major.nonZeros() << std::endl;

    //1. random:
    //  for (int i = 0; i < data_dim; ++i)
    //    {
    //      bool flag = false;
    //      for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(centralized_data_mat_row_major,i); it; ++it)
    //        {
    //          flag = true;
    //          break;
    //        }
    //      if (flag)
    //        {
    //          for (int j = 0; j < component_num; ++j)
    //          {
    //            base_mat_row_major->insert(i, j) = float(rand())/RAND_MAX - 0.5;
    //          }
    //        }
    //      //std::cerr << i << std::endl;
    //    }
    //2. pca
    Eigen::MatrixXf weighted_D = weight_mat_row_major.cwiseProduct(centralized_data_mat_row_major);
    Eigen::MatrixXf WD_trans_WD = weighted_D.transpose() * weighted_D;
    Eigen::MatrixXf W_trans_W = (weight_mat_row_major.transpose() * weight_mat_row_major).eval();
    WD_trans_WD = WD_trans_WD.cwiseQuotient(W_trans_W) * (data_dim/sample_num);
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(WD_trans_WD ,Eigen::ComputeFullV);
    double tolerance =
            std::numeric_limits<float>::epsilon() * std::max(WD_trans_WD.cols(), WD_trans_WD.rows()) *svd.singularValues().array().abs()(0);
    *coeff_mat =
            (((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array(), 0)).array().sqrt().matrix().asDiagonal()
             * svd.matrixV().transpose()).middleRows(0, component_num);
    ComputeBasisInMStep(centralized_data_mat_row_major, weight_mat_row_major, *coeff_mat, base_mat_row_major);
    std::cerr << "finished making initial estimate " << std::endl;
    return;
}


/**
 * @brief cpu_tsdf::ComputeCoeffInEStep
 * E step, compute coefficient matrix
 * @param centralized_data_mat
 * @param weight_mat
 * @param base_mat
 * @param coeff_mat
 */
void ComputeCoeffInEStep(const Eigen::SparseMatrix<float, Eigen::ColMajor> &centralized_data_mat,
                                   const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat,
                                   const Eigen::SparseMatrix<float, Eigen::ColMajor> &base_mat,
                                   Eigen::MatrixXf *coeff_mat)
{
    const int sample_num = centralized_data_mat.cols();
    const int data_dim = centralized_data_mat.rows();
    const int component_num = base_mat.cols();
    Eigen::SparseMatrix<float, Eigen::ColMajor> W_j_U(data_dim, component_num);
    for (int j = 0; j < sample_num; ++j)
    {
        W_j_U = (Eigen::VectorXf(weight_mat.col(j)).asDiagonal()) * base_mat;
        Eigen::MatrixXf temp1 = Eigen::MatrixXf((base_mat.transpose() * W_j_U).eval());
        coeff_mat->col(j) =
                Eigen::VectorXf(
                    (Eigen::MatrixXf(utility::PseudoInverse(temp1)) *
                     (W_j_U.transpose() * centralized_data_mat.col(j).eval()))
                    );
    }
}

/**
 * @brief cpu_tsdf::ComputeBasisInMStep
 * M step, compute the bases matrix (slow)
 * @param centralized_data_mat_row_major
 * @param weight_mat_row_major
 * @param coeff_mat
 * @param base_mat_row_major
 */
void ComputeBasisInMStep(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
                                   const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
                                   const Eigen::MatrixXf &coeff_mat, Eigen::SparseMatrix<float, Eigen::RowMajor> *base_mat_row_major)
{
    const int sample_num = centralized_data_mat_row_major.cols();
    const int data_dim = centralized_data_mat_row_major.rows();
    const int component_num = coeff_mat.rows();
    Eigen::VectorXf data_row_vec =
            Eigen::VectorXf::Zero(sample_num, 1);
    base_mat_row_major->setZero();
    for (int i = 0; i < data_dim; ++i)
    {
        if (i % 100000 == 0)
            std::cerr << "cur_i: " << i << std::endl;
        bool flag = false;
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itr(centralized_data_mat_row_major, i);
             itr; ++itr)
        {
            flag = true;
            data_row_vec.coeffRef(itr.col()) = itr.value();
        }
        if (flag)
        {
            Eigen::MatrixXf A_Wi = coeff_mat * Eigen::MatrixXf(weight_mat_row_major.row(i)).asDiagonal();
            Eigen::VectorXf row_i_dense =
                    Eigen::MatrixXf( A_Wi * coeff_mat.transpose() ).
                    jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).
                    solve(A_Wi * data_row_vec);
            for (int k = 0; k < component_num; ++k)
            {
                if (abs(row_i_dense[k]) > std::numeric_limits<double>::epsilon())
                {
                    base_mat_row_major->insert(i, k) = row_i_dense[k];
                }
            }
            data_row_vec.setZero();
        }  // end if
    }  // end for i
}

// compute only one row of coeffcient matrix
void ComputeCoeffInEStepOneVec(const Eigen::SparseMatrix<float, Eigen::ColMajor> &centralized_data_mat,
                                         const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat,
                                         const Eigen::VectorXf &base_vec,
                                         Eigen::VectorXf *coeff_vec)
{
    const int sample_num = centralized_data_mat.cols();
    const int data_dim = centralized_data_mat.rows();
    Eigen::VectorXf tmpA(data_dim);
    for (int j = 0; j < sample_num; ++j)
    {
        tmpA = weight_mat.col(j).cwiseProduct(base_vec);
        (*coeff_vec)(j) = centralized_data_mat.col(j).dot(tmpA) / (tmpA.dot(base_vec));
        if (std::isnan((*coeff_vec)(j))) (*coeff_vec)(j) = 0;
    }
}

// compute only one column of base matrix
void ComputeBasisInMStepOneVec(const Eigen::SparseMatrix<float, Eigen::RowMajor> &centralized_data_mat_row_major,
                                         const Eigen::SparseMatrix<float, Eigen::RowMajor> &weight_mat_row_major,
                                         const Eigen::VectorXf &coeff_vec, Eigen::VectorXf *base_vec)
{
    const int sample_num = centralized_data_mat_row_major.cols();
    const int data_dim = centralized_data_mat_row_major.rows();
    Eigen::VectorXf data_row_vec =
            Eigen::VectorXf::Zero(sample_num, 1);
    base_vec->setZero();
    for (int i = 0; i < data_dim; ++i)
    {
        //      if (i % 10000 == 0)
        //        std::cerr << "cur_i: " << i << std::endl;

        bool flag = false;
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itr(centralized_data_mat_row_major, i);
             itr; ++itr)
        {
            flag = true;
            data_row_vec.coeffRef(itr.col()) = itr.value();
        }
        if (flag)
        {
            Eigen::VectorXf tmpB = weight_mat_row_major.row(i).transpose().cwiseProduct(coeff_vec);
            (*base_vec)(i) = data_row_vec.dot(tmpB) / (tmpB.dot(coeff_vec));
            if (std::isnan((*base_vec)(i))) (*base_vec)(i) = 0;
            data_row_vec.setZero();
        }  // end if
    }  // end for i
}

// utility function to save intermediate results
static void WPCASaveRelatedMatrices(const Eigen::SparseMatrix<float, ColMajor>* centralized_data_mat,
                                    const Eigen::SparseVector<float>* mean_mat,
                                    const Eigen::SparseMatrix<float, ColMajor>* weight_mat,
                                    const Eigen::SparseMatrix<float, ColMajor>* base_mat,
                                    const Eigen::MatrixXf *coeff_mat,
                                    const string& save_filepath
                                    )
{
    cout << "save related matrices" << endl;
    string output_dir = bfs::path(save_filepath).remove_filename().string();
    std::string output_plyfilename = save_filepath;
    string output_datamat =
            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
                                      + "_output_data_mat.txt")).string();
    string output_meanmat =
            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
                                      + "_output_mean_mat.txt")).string();
    string output_weightmat =
            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
                                      + "_output_weight_mat.txt")).string();
    string output_initbasemat =
            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
                                      + "_output_base_mat.txt")).string();
    string output_initcoeffmat =
            (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
                                      + "_output_coeff_mat.txt")).string();
    if (centralized_data_mat)
        utility::WriteEigenMatrix(*centralized_data_mat, output_datamat);
    if (mean_mat)
        utility::WriteEigenMatrix(Eigen::MatrixXf(*mean_mat), output_meanmat);
    if (weight_mat)
        utility::WriteEigenMatrix(*weight_mat, output_weightmat);
    if (base_mat)
        utility::WriteEigenMatrix(*base_mat, output_initbasemat);
    if (coeff_mat)
        utility::WriteEigenMatrix(*coeff_mat, output_initcoeffmat);
    cout << "save related matrices finished" << endl;
}

void WPCASaveTSDFsFromMats(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& projected_data_mat,
        const Eigen::Vector3i& boundingbox_size,
        const float voxel_length,
        const Eigen::Vector3f& offset,
        const float max_dist_pos,
        const float max_dist_neg,
        const string& save_filepath)
{
    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
    cpu_tsdf::ConvertDataMatrixToTSDFsNoWeight(voxel_length,
                                               offset,
                                               max_dist_pos,
                                               max_dist_neg,
                                               boundingbox_size,
                                               projected_data_mat,
                                               &projected_tsdf_models
                                               );
    string output_dir = bfs::path(save_filepath).remove_filename().string();
    string output_modelply =
            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
                                      + "_recovered_models.ply")).string();
    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_modelply, false, true, 0);
}

void WPCASaveTSDFsFromMats(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& projected_data_mat,
        const cpu_tsdf::TSDFHashing& mean_tsdf,
        const Eigen::Vector3i& boundingbox_size,
        const string& save_filepath)
{
    const float voxel_length = mean_tsdf.voxel_length();
    const Eigen::Vector3f offset = mean_tsdf.offset();
    float max_dist_pos, max_dist_neg;
    mean_tsdf.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
    cpu_tsdf::ConvertDataMatrixToTSDFsNoWeight(voxel_length,
                                               offset,
                                               max_dist_pos,
                                               max_dist_neg,
                                               boundingbox_size,
                                               projected_data_mat,
                                               &projected_tsdf_models
                                               );
    string output_dir = bfs::path(save_filepath).remove_filename().string();
    string output_modelply =
            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
                                      + "_recovered_models.ply")).string();
    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_modelply, false, true, 0);
}

void WPCASaveUncentralizedTSDFs(
        const Eigen::SparseMatrix<float, ColMajor>& centralized_data_mat,
        const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
        const Eigen::SparseVector<float>& mean_mat,
        const Eigen::Vector3i& boundingbox_size,
        const float voxel_length,
        const Eigen::Vector3f& offset,
        const float max_dist_pos,
        const float max_dist_neg,
        const string& save_filepath)
{
    cout << "save uncentralized tsdf" << endl;
    const int data_dim = centralized_data_mat.rows();
    const int sample_num = centralized_data_mat.cols();
    // original tsdf models
    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
                     (mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
                     centralized_data_mat;
    WPCASaveTSDFsFromMats(projected_data_mat, boundingbox_size,
                          voxel_length, offset, max_dist_pos, max_dist_neg, save_filepath);
}

void WPCASaveUncentralizedTSDFs(const Eigen::SparseMatrix<float, ColMajor>& centralized_data_mat,
                                       const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
                                       const Eigen::SparseVector<float>& mean_mat,
                                       const cpu_tsdf::TSDFHashing& mean_tsdf,
                                       const Eigen::Vector3i& boundingbox_size,
                                       const string& save_filepath)
{
    cout << "save uncentralized tsdf" << endl;
    const int data_dim = centralized_data_mat.rows();
    const int sample_num = centralized_data_mat.cols();
    // original tsdf models
    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
                     (mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
                     centralized_data_mat;
    WPCASaveTSDFsFromMats(projected_data_mat, mean_tsdf, boundingbox_size, save_filepath);
}

// utility function to save intermediate results
void WPCASaveRecoveredTSDFs(const Eigen::SparseVector<float>& mean_part_mat,
                                   // these two cannot be const because bug(?) in Eigen
                                   // If set to const, the production of these two will be computed in some "conservative" way
                                   // and leads to infinite recursion?
                                   Eigen::SparseMatrix<float, ColMajor>& base_mat,
                                   Eigen::MatrixXf& coeff_mat,
                                   const int component_num,
                                   const Eigen::Vector3i& boundingbox_size,
                                   const float voxel_length,
                                   const Eigen::Vector3f& offset,
                                   const float max_dist_pos,
                                   const float max_dist_neg,
                                   const string& save_filepath
                                   )
{
    cout << "save pca recovered tsdf" << endl;
    const int data_dim = base_mat.rows();
    const int sample_num = coeff_mat.cols();
    // tsdf models
    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
                  (mean_part_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
                     (base_mat).leftCols(component_num) * (coeff_mat).topRows(component_num).sparseView().eval();
    WPCASaveTSDFsFromMats(projected_data_mat, boundingbox_size,
                          voxel_length, offset, max_dist_pos, max_dist_neg, save_filepath);
}

void WPCASaveRecoveredTSDFs(const Eigen::SparseVector<float>& mean_part_mat,
                                   const cpu_tsdf::TSDFHashing& mean_tsdf,
                                   // these two cannot be const because bug(?) in Eigen
                                   // If set to const, the production of these two will be computed in some "conservative" way
                                   // and leads to infinite recursion?
                                   Eigen::SparseMatrix<float, ColMajor>& base_mat,
                                   Eigen::MatrixXf& coeff_mat,
                                   const int component_num,
                                   const Eigen::Vector3i& boundingbox_size,
                                   const string& save_filepath
                                   )
{
    cout << "save pca recovered tsdf" << endl;
    const int data_dim = base_mat.rows();
    const int sample_num = coeff_mat.cols();
    // tsdf models
    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
                  (mean_part_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
                     (base_mat).leftCols(component_num) * (coeff_mat).topRows(component_num).sparseView().eval();
    WPCASaveTSDFsFromMats(projected_data_mat, mean_tsdf, boundingbox_size, save_filepath);
}
}

namespace cpu_tsdf {

void ReconstructTSDFsFromPCAOriginPos(
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& recon_sample_weight,
        const std::vector<int> &sample_model_assign,
        const std::vector<tsdf_utility::OrientedBoundingBox>& obbs,
        const cpu_tsdf::TSDFGridInfo& grid_info,
        const float voxel_length,
        const Eigen::Vector3f& scene_tsdf_offset,
        std::vector<cpu_tsdf::TSDFHashing::Ptr>* reconstructed_samples_original_pos) {
    std::vector<Eigen::SparseVector<float>> recon_samples;
    PCAReconstructionResult(model_means, model_bases, projected_coeffs, sample_model_assign, &recon_samples);
    std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
    ConvertDataVectorsToTSDFsWithWeight(recon_samples,
                                        cpu_tsdf::EigenMatToSparseVectors( recon_sample_weight), grid_info, &recon_tsdfs);
    vector<Eigen::Affine3f> affine_transforms;
    for (const auto& obbi : obbs) {
        affine_transforms.push_back(obbi.AffineTransform());
    }
    TransformTSDFs(recon_tsdfs, affine_transforms, reconstructed_samples_original_pos, &voxel_length, &scene_tsdf_offset);
}

void ReconstructTSDFsFromPCAOriginPos(
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& recon_sample_weight,
        const std::vector<int> &sample_model_assign,
        const std::vector<Eigen::Affine3f>& affine_transforms,
        const cpu_tsdf::TSDFGridInfo& grid_info,
        const float voxel_length,
        const Eigen::Vector3f& scene_tsdf_offset,
        std::vector<cpu_tsdf::TSDFHashing::Ptr>* reconstructed_samples_original_pos)
{
    std::vector<Eigen::SparseVector<float>> recon_samples;
    PCAReconstructionResult(model_means, model_bases, projected_coeffs, sample_model_assign, &recon_samples);
    std::vector<cpu_tsdf::TSDFHashing::Ptr> recon_tsdfs;
    ConvertDataVectorsToTSDFsWithWeight(recon_samples,
                                        cpu_tsdf::EigenMatToSparseVectors( recon_sample_weight), grid_info, &recon_tsdfs);
    //cpu_tsdf::WriteTSDFModels(recon_tsdfs, "/home/dell/p01.ply", false, true, 0);
    //    cpu_tsdf::WriteTSDFsFromMatWithWeight(
    //                cpu_tsdf::SparseVectorsToEigenMat(recon_samples),
    //                recon_sample_weight, grid_info, "/home/dell/p1.ply");
    //ConvertDataVectorsToTSDFsNoWeight(recon_samples, grid_info, &recon_tsdfs);
    TransformTSDFs(recon_tsdfs, affine_transforms, reconstructed_samples_original_pos, &voxel_length, &scene_tsdf_offset);
    //cpu_tsdf::WriteTSDFModels(*reconstructed_samples_original_pos, "/home/dell/p2.ply", false, true, 0);
}

bool PCAReconstructionResult(
        const std::vector<Eigen::SparseVector<float> > &model_means,
        const std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > &model_bases,
        const std::vector<Eigen::VectorXf> &projected_coeffs,
        const std::vector<int> &model_assign_idx,
        std::vector<Eigen::SparseVector<float> > *reconstructed_samples)
{
    const int sample_num = model_assign_idx.size();
    reconstructed_samples->resize(sample_num);
    for (int sample_i = 0; sample_i < sample_num; ++sample_i)
    {
        int current_model = model_assign_idx[sample_i];
        (*reconstructed_samples)[sample_i] = (model_means[current_model]);
        if (model_bases[current_model].rows() * model_bases[current_model].cols() > 0)
        {
            // assume projected_coeffs here is not empty
            (*reconstructed_samples)[sample_i] += (model_bases[current_model]) * projected_coeffs[sample_i].sparseView().eval();
        }
    }
    return true;
}

bool OptimizeModelAndCoeffOld(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const std::vector<int> &model_assign_idx,
        const std::vector<double> &outlier_gammas,
        const int component_num, const int max_iter,
        std::vector<Eigen::SparseVector<float> > *model_means,
        std::vector<Eigen::SparseVector<float> > *model_mean_weight,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        PCAOptions &options)
{
    using namespace std;
    const int sample_number = samples.cols();
    const int feature_dim = samples.rows();
    const int model_number = *(std::max_element(model_assign_idx.begin(), model_assign_idx.end())) + 1;
    vector<vector<int>> cluster_sample_idx;
    GetClusterSampleIdx(model_assign_idx,  outlier_gammas, model_number, &cluster_sample_idx);
    model_means->resize(model_number);
    model_bases->resize(model_number);
    projected_coeffs->resize(sample_number);
    model_mean_weight->resize(model_number);
    for (int i = 0; i < model_number; ++i)
    {
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_samples(feature_dim, cluster_sample_idx[i].size());
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_weights(feature_dim, cluster_sample_idx[i].size());
        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
        {
            cur_samples.col(j) = samples.col(cluster_sample_idx[i][j]);
            cur_weights.col(j) = weights.col(cluster_sample_idx[i][j]);
        }
        Eigen::MatrixXf current_projected_coeffs;
        const std::string prev_save_path = options.save_path;
        bfs::path bfs_prefix(options.save_path);
        options.save_path = (bfs_prefix.parent_path()/bfs_prefix.stem()).string() + "_cluster_" + boost::lexical_cast<string>(i) + ".ply";
        WeightedPCADeflationOrthogonal(cur_samples, cur_weights, component_num, max_iter,
                                       &((*model_means)[i]),
                                       &((*model_mean_weight)[i]),
                                       &((*model_bases)[i]),
                                       &current_projected_coeffs,
                                       options);
        options.save_path = prev_save_path;
        if (component_num == 0) continue;
        assert(current_projected_coeffs.cols() == cluster_sample_idx[i].size());
        for (int j = 0; j < current_projected_coeffs.cols(); ++j)
        {
            (*projected_coeffs)[cluster_sample_idx[i][j]] = current_projected_coeffs.col(j);
        }
    }  // end for i
    return true;
}

bool OptimizeModelAndCoeff(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const std::vector<int> &model_assign_idx,
        const std::vector<double> &outlier_gammas,
        const int component_num, const int max_iter,
        std::vector<Eigen::SparseVector<float> > *model_means,
        std::vector<Eigen::SparseVector<float> > *model_mean_weight,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        PCAOptions &options)
{
    using namespace std;
    const int sample_number = samples.cols();
    const int feature_dim = samples.rows();
    const int model_number = *(std::max_element(model_assign_idx.begin(), model_assign_idx.end())) + 1;
    vector<vector<int>> cluster_sample_idx;
    GetClusterSampleIdx(model_assign_idx,  outlier_gammas, model_number, &cluster_sample_idx);
    model_means->resize(model_number);
    model_bases->resize(model_number);
    projected_coeffs->resize(sample_number);
    model_mean_weight->resize(model_number);
    for (int i = 0; i < model_number; ++i)
    {
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_samples(feature_dim, cluster_sample_idx[i].size());
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_weights(feature_dim, cluster_sample_idx[i].size());
        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
        {
            cur_samples.col(j) = samples.col(cluster_sample_idx[i][j]);
            cur_weights.col(j) = weights.col(cluster_sample_idx[i][j]);
        }
        Eigen::MatrixXf current_projected_coeffs;
        const std::string prev_save_path = options.save_path;
        bfs::path bfs_prefix(options.save_path);
        options.save_path = (bfs_prefix.parent_path()/bfs_prefix.stem()).string() + "_cluster_" + boost::lexical_cast<string>(i) + ".ply";
        WeightedPCADeflationOrthogonal(cur_samples, cur_weights, component_num, max_iter,
                                       &((*model_means)[i]),
                                       &((*model_mean_weight)[i]),
                                       &((*model_bases)[i]),
                                       &current_projected_coeffs,
                                       options);

    //    const int percentage_to_trim = 10;
    //    GrassmanAverageWeightedZC_CallProgram(cur_samples, cur_weights, component_num, percentage_to_trim,
    //                                   &((*model_means)[i]),
    //                                   &((*model_mean_weight)[i]),
    //                                   &((*model_bases)[i]),
    //                                   &current_projected_coeffs,
    //                                   options);

    //    const Eigen::Vector3i data_dim_size = options.boundingbox_size;
    //RPCAMJB_WeightedZC_CallProgram(cur_samples, cur_weights, component_num, data_dim_size,
    //                                      &((*model_means)[i]),
    //                                      &((*model_mean_weight)[i]),
    //                                      &((*model_bases)[i]),
    //                                      &current_projected_coeffs,
    //                                      options);
        options.save_path = prev_save_path;
        // if (component_num == 0) continue;
        assert(current_projected_coeffs.cols() == cluster_sample_idx[i].size());
        for (int j = 0; j < current_projected_coeffs.cols(); ++j)
        {
            (*projected_coeffs)[cluster_sample_idx[i][j]] = current_projected_coeffs.col(j);
        }
    }  // end for i
    return true;
}

bool OptimizeModelAndCoeff(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights,
        const std::vector<int> &model_assign_idx,
        const std::vector<double> &outlier_gammas,
        const int component_num, const int max_iter,
        std::vector<Eigen::SparseVector<float> > *model_means,
        std::vector<Eigen::SparseVector<float> > *model_mean_weight,
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > *model_bases,
        std::vector<Eigen::VectorXf> *projected_coeffs,
        tsdf_utility::OptimizationParams& params)
{
    using namespace std;
    const int sample_number = samples.cols();
    const int feature_dim = samples.rows();
    const int model_number = *(std::max_element(model_assign_idx.begin(), model_assign_idx.end())) + 1;
    vector<vector<int>> cluster_sample_idx;
    GetClusterSampleIdx(model_assign_idx,  outlier_gammas, model_number, &cluster_sample_idx);
    model_means->resize(model_number);
    model_bases->resize(model_number);
    projected_coeffs->resize(sample_number);
    model_mean_weight->resize(model_number);
    for (int i = 0; i < model_number; ++i)
    {
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_samples(feature_dim, cluster_sample_idx[i].size());
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_weights(feature_dim, cluster_sample_idx[i].size());
        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
        {
            cur_samples.col(j) = samples.col(cluster_sample_idx[i][j]);
            cur_weights.col(j) = weights.col(cluster_sample_idx[i][j]);
        }
        Eigen::MatrixXf current_projected_coeffs;
        const std::string prev_save_path = params.save_path;
        bfs::path bfs_prefix(params.save_path);
        params.save_path = (bfs_prefix.parent_path()/bfs_prefix.stem()).string() + "_cluster_" + boost::lexical_cast<string>(i) + ".ply";
        WeightedPCADeflationOrthogonal(cur_samples, cur_weights, component_num, max_iter,
                                       &((*model_means)[i]),
                                       &((*model_mean_weight)[i]),
                                       &((*model_bases)[i]),
                                       &current_projected_coeffs,
                                       params);
        params.save_path = prev_save_path;
        // if (component_num == 0) continue;
        assert(current_projected_coeffs.cols() == cluster_sample_idx[i].size());
        for (int j = 0; j < current_projected_coeffs.cols(); ++j)
        {
            (*projected_coeffs)[cluster_sample_idx[i][j]] = current_projected_coeffs.col(j);
        }
    }  // end for i
    return true;
}

bool GrassmanAverageWeightedZC_CallProgram(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
        const int component_num, const int percentage_to_trim,
        Eigen::SparseVector<float> *mean_mat,
        Eigen::SparseVector<float>* mean_weight,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
        Eigen::MatrixXf *coeff_mat,
        const PCAOptions& options) {
    const char* programPath = "/home/dell/codebase/mpi_project/urban_reconstruction/code/build/grassman/bin/grassman_average_main";
    Eigen::MatrixXf data_mat = samples;
    Eigen::MatrixXf weight_mat = weights;
    const std::string filepath = "/tmp/tmpgrassman_input.mat";
    const std::string rpcares_filepath = "/tmp/tmpgrassman_rpcaout.mat";
    bfs::remove(filepath);
    bfs::remove(rpcares_filepath);
    matlabutility::WriteMatrixMatlab(filepath, "data_mat", data_mat);
    matlabutility::WriteMatrixMatlab(filepath, "weight_mat", weight_mat);
    char cpercent[20] = {0};
    sprintf(cpercent, "%d", percentage_to_trim);
    char compnum[20] = {0};
    sprintf(compnum, "%d", component_num + 1);
    std::vector<std::string> args;
    args.push_back(programPath);
    args.push_back(std::string("--input_mat_file"));
    args.push_back(filepath);
    args.push_back("--output_mat_file");
    args.push_back(rpcares_filepath);
    args.push_back("--data_var_name");
    args.push_back("data_mat");
    args.push_back("weight_var_name");
    args.push_back("weight_mat");
    args.push_back("--num_component");
    args.push_back(compnum);
    args.push_back("--percentage_to_trim");
    args.push_back(cpercent);
    args.push_back("--alsologtostderr");
    std::unique_ptr<char* []> arg_ptrs(new char*[args.size() + 1]);
    for (int i = 0; i < args.size(); ++i) {
        arg_ptrs[i] = const_cast<char*>(args[i].c_str());
        // arg_ptrs[i] = (args[i].c_str());
    }
    arg_ptrs[args.size()] = NULL;
    ///////////////////////////////////////////////////
    // run program
    cerr << "begin call program" << endl;
    pid_t pid = fork(); /* Create a child process */
    switch (pid) {
    case -1: /* Error */
        std::cerr << "Uh-Oh! fork() failed.\n";
        exit(EXIT_FAILURE);
    case 0: /* Child process */ {
        // execl(programPath, args.c_str()); /* Execute the program */
        std::cerr << programPath << std::endl;
        int ii = 0;
        while (arg_ptrs[ii] != NULL) {
            std::cerr << arg_ptrs[ii] << std::endl;
            ii++;
        }
        // std::cerr << args[1] << std::endl;
        execv(programPath, arg_ptrs.get()); /* Execute the program */
        std::cerr << "Uh-Oh! execl() failed! " << std::endl;
        ; /* execl doesn't return unless there's an error */
        exit(EXIT_FAILURE);
    }
    default: /* Parent process */
        std::cout << "Process created with pid " << pid << "\n";
        int status;
        while (!WIFEXITED(status)) {
            waitpid(pid, &status, 0); /* Wait for the process to complete */
        }
        std::cout << "Process exited with " << WEXITSTATUS(status) << "\n";
        if (WEXITSTATUS(status) != 0) {
            cerr << "grassman lib failed" << endl;
            exit(EXIT_FAILURE);
        }
    }
    ///////////////////////////////////////////////////
    matlabutility::ReadPCAOutputResult(rpcares_filepath, mean_mat, mean_weight, base_mat, coeff_mat);
    return true;
}

bool RPCAMJB_WeightedZC_CallProgram(
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
        const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
        const int component_num, const Eigen::Vector3i data_dim_size,
        Eigen::SparseVector<float> *mean_mat,
        Eigen::SparseVector<float>* mean_weight,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
        Eigen::MatrixXf *coeff_mat,
        const PCAOptions& options) {
    const char* programPath = "/home/dell/codebase/mpi_project/urban_reconstruction/code/build/rpca_mjb/bin/rpca_mjb_main";
    Eigen::MatrixXf data_mat = samples;
    Eigen::MatrixXf weight_mat = weights;
    const std::string filepath = "/tmp/tmprpcamjb_input.mat";
    const std::string rpcares_filepath = "/tmp/tmprpcamjb_rpcaout.mat";
    bfs::remove(filepath);
    bfs::remove(rpcares_filepath);
    matlabutility::WriteMatrixMatlab(filepath, "data_mat", data_mat);
    matlabutility::WriteMatrixMatlab(filepath, "weight_mat", weight_mat);
    char compnum[20] = {0};
    sprintf(compnum, "%d", component_num);
    char dim_size_char[3][20] = {0};
    sprintf(dim_size_char[0], "%d", data_dim_size[0]);
    sprintf(dim_size_char[1], "%d", data_dim_size[1]);
    sprintf(dim_size_char[2], "%d", data_dim_size[2]);
    std::vector<std::string> args;
    args.push_back(programPath);
    args.push_back(std::string("--input_mat_file"));
    args.push_back(filepath);
    args.push_back("--output_mat_file");
    args.push_back(rpcares_filepath);
    args.push_back("--data_var_name");
    args.push_back("data_mat");
    args.push_back("weight_var_name");
    args.push_back("weight_mat");
    args.push_back("--num_component");
    args.push_back(compnum);
    args.push_back("--data_dim_size");
    args.push_back(dim_size_char[0]);
    args.push_back(dim_size_char[1]);
    args.push_back(dim_size_char[2]);
    args.push_back("--alsologtostderr");
    std::unique_ptr<char* []> arg_ptrs(new char*[args.size() + 1]);
    for (int i = 0; i < args.size(); ++i) {
        arg_ptrs[i] = const_cast<char*>(args[i].c_str());
        // arg_ptrs[i] = (args[i].c_str());
    }
    arg_ptrs[args.size()] = NULL;
    ///////////////////////////////////////////////////
    // run program
    cerr << "begin call program" << endl;
    pid_t pid = fork(); /* Create a child process */
    switch (pid) {
    case -1: /* Error */
        std::cerr << "Uh-Oh! fork() failed.\n";
        exit(EXIT_FAILURE);
    case 0: /* Child process */ {
        // execl(programPath, args.c_str()); /* Execute the program */
        std::cerr << programPath << std::endl;
        int ii = 0;
        while (arg_ptrs[ii] != NULL) {
            std::cerr << arg_ptrs[ii] << std::endl;
            ii++;
        }
        // std::cerr << args[1] << std::endl;
        execv(programPath, arg_ptrs.get()); /* Execute the program */
        std::cerr << "Uh-Oh! execl() failed! " << std::endl;
        ; /* execl doesn't return unless there's an error */
        exit(EXIT_FAILURE);
    }
    default: /* Parent process */
        std::cout << "Process created with pid " << pid << "\n";
        int status;
        while (!WIFEXITED(status)) {
            waitpid(pid, &status, 0); /* Wait for the process to complete */
        }
        std::cout << "Process exited with " << WEXITSTATUS(status) << "\n";
        if (WEXITSTATUS(status) != 0) {
            cerr << "grassman lib failed" << endl;
            exit(EXIT_FAILURE);
        }
    }
    ///////////////////////////////////////////////////
    matlabutility::ReadPCAOutputResult(rpcares_filepath, mean_mat, mean_weight, base_mat, coeff_mat);
    return true;
}

bool WeightedPCADeflationOrthogonal(
                const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples,
                const Eigen::SparseMatrix<float, Eigen::ColMajor>& weights,
                const int component_num, const int max_iter,
        Eigen::SparseVector<float> *mean_mat,
        Eigen::SparseVector<float>* mean_weight,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
        Eigen::MatrixXf *coeff_mat,
        const PCAOptions& options)
{
    const int sample_num = samples.cols();
    const int data_dim = samples.rows();
    if (sample_num == 0 || data_dim == 0) return false;
    // #samples: N, #dim: D, #principal_components: K
    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
    cout << "begin weighted pca" << endl;
    cerr << "1. compute mean mat" << endl;
    // ((Samples.*Weight)) * ones(N, 1) ./ Weight * ones(N, 1)
    Eigen::VectorXf onesN = Eigen::MatrixXf::Ones(sample_num, 1);
    // *mean_mat = (((samples.cwiseProduct(weights)) * onesN).cwiseQuotient( (weights * onesN) )).sparseView();
    Eigen::SparseVector<float> sum_vec = ((samples.cwiseProduct(weights)) * onesN).sparseView();
    Eigen::SparseVector<float> sum_weights = (weights * onesN).sparseView();
    mean_mat->resize(data_dim);
    mean_mat->reserve(sum_weights.nonZeros());
    for (Eigen::SparseVector<float>::InnerIterator it(sum_weights); it; ++it)
    {
        const float cur_weight = it.value();
        const int cur_idx = it.index();
        if (cur_weight > 0)
        {
            mean_mat->insert(cur_idx) = sum_vec.coeff(cur_idx) / cur_weight;
        }
    }
    *mean_weight = sum_weights / sample_num;
    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat = samples - (*mean_mat) * onesN.transpose();
    if (!options.save_path.empty())
    {
        const string* save_filepath = &options.save_path;
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation_ortho"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_original" + "pp.ply")).string();
        // 1. save the original mats for matlab debug
        // WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, &weights, NULL, NULL, output_filename);
        // 2. save the tsdfs
//        WPCASaveUncentralizedTSDFs(centralized_data_mat, weights, *mean_mat,
//                                   options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
//                                   output_filename);
        WPCASaveMeanTSDF(*mean_mat, weights,
                         options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
                         output_filename);
    }
    if (component_num == 0) return true;
    base_mat->resize(data_dim, sample_num);
    coeff_mat->resize(component_num, sample_num);
    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(weights);
    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
    // initialize base_mat U
    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
    cout << "data_dim: " << data_dim << endl;
    cout << "sample_num: " << sample_num << endl;
    cout << "compo_num: " << component_num << endl;
    cerr << "2. Initial Estimate" << endl;
    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
    *base_mat = base_mat_row_major;
    if (!options.save_path.empty())
    {
        const string* save_filepath = &options.save_path;
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation_ortho"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_init_estimated" + ".ply")).string();
        // 1. save the initial mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, &weights,
                                base_mat, coeff_mat, output_filename);
        // 2. save the tsdfs
        WPCASaveRecoveredTSDFs(*mean_mat, *base_mat, *coeff_mat, component_num,
                               options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
                               output_filename);
    }
    // do EM
    cerr << "3. Do EM" << endl;
    const float thresh = 1e-4;
    Eigen::VectorXf prev_base(data_dim);
    Eigen::VectorXf current_base(data_dim);
    Eigen::VectorXf current_coeff(sample_num);
    const int iteration_number = std::max(sample_num, max_iter);
    for (int k = 0; k < component_num; ++k)
    {
        current_base = base_mat->col(k);
        current_coeff = coeff_mat->row(k);
        prev_base = current_base;
        cout << "Computing " << k << "th component." << endl;
        for (int i = 0; i < iteration_number; ++i)
        {
            cout << "Iteration: \nEstep " << i <<  endl;
            //E step
            ComputeCoeffInEStepOneVec(centralized_data_mat, weights, current_base, &current_coeff);
            //M step
            cout << "Mstep " << i <<  endl;
            ComputeBasisInMStepOneVec(centralized_data_mat_row_major, weight_mat_row_major, current_coeff, &current_base);
            //orthogonalize:
            float component_norm = 0;
            OrthogonalizeVector(*base_mat, k, &current_base, &component_norm);
            current_coeff *= component_norm;
            ////////////////////////////////////////////////
            /// save
            if (!options.save_path.empty() && (i%10) == 0)
            {
                const string* save_filepath = &options.save_path;
                string output_dir = bfs::path(*save_filepath).remove_filename().string();
                string output_filename =
                        (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                                  + "_tsdf_wpca_deflation_ortho"
                                                  + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                                  + "_comp_" + boost::lexical_cast<string>(k)
                                                  + "_itr_" + boost::lexical_cast<string>(i)
                                                  + ".ply")).string();
                Eigen::SparseMatrix<float, ColMajor> temp_base = current_base.sparseView();
                Eigen::MatrixXf temp_coeff = current_coeff;
                WPCASaveRelatedMatrices(NULL, NULL, NULL,
                                        &temp_base, &temp_coeff, output_filename);
                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
                        =
                        (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval()
                        + temp_base * temp_coeff.transpose().sparseView().eval();
                WPCASaveTSDFsFromMats(temp_projected_data_mat,
                                      options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
                                      output_filename);
//                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
//                        = (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
//                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval();
//                WPCASaveRecoveredTSDFs(temp_projected_data_mat, *mean_tsdf,
//                                       temp_base, temp_coeff, 1, *boundingbox_size, output_filename);
            }
            ////////////////////////////////////////////////
            //test convergence
            float t = (prev_base.normalized() - current_base.normalized()).cwiseAbs().sum();
            cerr << "difference.. " << t << endl;
            if (t < thresh)
            {
                cerr << "converge reached.. " << endl;
                break;
            }
            prev_base = current_base;
        }  // end for i (iteration)
        for (int p = 0; p < data_dim; ++p)
        {
            if (abs(current_base[p]) > std::numeric_limits<double>::epsilon())
            {
                //base_mat->insert(p, k) = current_base[p];
                base_mat->coeffRef(p, k) = current_base[p];
            }
        }
        (*coeff_mat).row(k) = current_coeff;
        centralized_data_mat -= weights.cwiseProduct(current_base.sparseView() * current_coeff.transpose());
        centralized_data_mat_row_major = centralized_data_mat;
        if (!options.save_path.empty())
        {
            const string* save_filepath = &options.save_path;
            string output_dir = bfs::path(*save_filepath).remove_filename().string();
            string output_filename =
                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                              + "_tsdf_wpca_deflation"
                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                              + "_comp_" + boost::lexical_cast<string>(k) + ".ply")).string();
            WPCASaveRelatedMatrices(&centralized_data_mat, NULL, NULL,
                                    base_mat, coeff_mat, output_filename);
            WPCASaveRecoveredTSDFs(*mean_mat, *base_mat, *coeff_mat, k + 1,
                                   options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
                                   output_filename);
        }
    }  // end for k
    return true;
}

bool TSDFWeightedPCADeflationOrthogonal_Old(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
                                                  const int component_num, const int max_iter,
                                                  cpu_tsdf::TSDFHashing *mean_tsdf,
                                                  Eigen::SparseVector<float> *mean_mat,
                                                  Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
                                                  Eigen::MatrixXf *coeff_mat, Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
                                                  Eigen::Vector3i *boundingbox_size, const string *save_filepath)
{
    // #samples: N, #dim: D, #principal_components: K
    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
    cout << "begin weighted pca" << endl;
    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
    cout << "1. convert TSDFs to sparse matrix" << endl;
    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, weight_mat, mean_mat, mean_tsdf, boundingbox_size)) return false;
    const int data_dim = centralized_data_mat.rows();
    const int sample_num = centralized_data_mat.cols();
    if (save_filepath)
    {
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation_ortho"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_original" + ".ply")).string();
        // 1. save the original mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat, NULL, NULL, output_filename);
        // 2. save the tsdfs
        WPCASaveUncentralizedTSDFs(centralized_data_mat, *weight_mat, *mean_mat, *mean_tsdf,
                                   *boundingbox_size, output_filename);
        WPCASaveMeanTSDF(*mean_mat, *mean_tsdf, *weight_mat, *boundingbox_size, output_filename);
    }
    base_mat->resize(data_dim, sample_num);
    coeff_mat->resize(component_num, sample_num);
    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(*weight_mat);
    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
    // initialize base_mat U
    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
    cout << "data_dim: " << data_dim << endl;
    cout << "sample_num: " << sample_num << endl;
    cout << "compo_num: " << component_num << endl;
    cerr << "2. Initial Estimate" << endl;
    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
    *base_mat = base_mat_row_major;
    if (save_filepath)
    {
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation_ortho"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_init_estimated" + ".ply")).string();
        // 1. save the initial mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
                                base_mat, coeff_mat, output_filename);
        // 2. save the tsdfs
        WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
                               *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
    }
    // do EM
    cerr << "3. Do EM" << endl;
    const float thresh = 1e-4;
    Eigen::VectorXf prev_base(data_dim);
    Eigen::VectorXf current_base(data_dim);
    Eigen::VectorXf current_coeff(sample_num);
    const int iteration_number = std::max(sample_num, 25);
    for (int k = 0; k < component_num; ++k)
    {
        current_base = base_mat->col(k);
        current_coeff = coeff_mat->row(k);
        prev_base = current_base;
        cout << "Computing " << k << "th component." << endl;
        for (int i = 0; i < iteration_number; ++i)
        {
            cout << "Iteration: \nEstep " << i <<  endl;
            //E step
            ComputeCoeffInEStepOneVec(centralized_data_mat, *weight_mat, current_base, &current_coeff);
            //M step
            cout << "Mstep " << i <<  endl;
            ComputeBasisInMStepOneVec(centralized_data_mat_row_major, weight_mat_row_major, current_coeff, &current_base);
            //orthogonalize:
            float component_norm = 0;
            OrthogonalizeVector(*base_mat, k, &current_base, &component_norm);
            current_coeff *= component_norm;
            ////////////////////////////////////////////////
            /// save
            if (save_filepath)
            {
                string output_dir = bfs::path(*save_filepath).remove_filename().string();
                string output_filename =
                        (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                                  + "_tsdf_wpca_deflation_ortho"
                                                  + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                                  + "_comp_" + boost::lexical_cast<string>(k)
                                                  + "_itr_" + boost::lexical_cast<string>(i)
                                                  + ".ply")).string();
                Eigen::SparseMatrix<float, ColMajor> temp_base = current_base.sparseView();
                Eigen::MatrixXf temp_coeff = current_coeff;
                WPCASaveRelatedMatrices(NULL, NULL, NULL,
                                        &temp_base, &temp_coeff, output_filename);
                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
                        =
                        (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval()
                        + temp_base * temp_coeff.transpose().sparseView().eval();
                WPCASaveTSDFsFromMats(temp_projected_data_mat, *mean_tsdf, *boundingbox_size, output_filename);
//                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
//                        = (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
//                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval();
//                WPCASaveRecoveredTSDFs(temp_projected_data_mat, *mean_tsdf,
//                                       temp_base, temp_coeff, 1, *boundingbox_size, output_filename);
            }
            ////////////////////////////////////////////////
            //test convergence
            float t = (prev_base.normalized() - current_base.normalized()).cwiseAbs().sum();
            cerr << "difference.. " << t << endl;
            if (t < thresh)
            {
                cerr << "converge reached.. " << endl;
                break;
            }
            prev_base = current_base;
        }
        for (int p = 0; p < data_dim; ++p)
        {
            if (abs(current_base[p]) > std::numeric_limits<double>::epsilon())
            {
                //base_mat->insert(p, k) = current_base[p];
                base_mat->coeffRef(p, k) = current_base[p];
            }
        }
        (*coeff_mat).row(k) = current_coeff;
        centralized_data_mat -= weight_mat->cwiseProduct(current_base.sparseView() * current_coeff.transpose());
        centralized_data_mat_row_major = centralized_data_mat;
        if (save_filepath)
        {
            string output_dir = bfs::path(*save_filepath).remove_filename().string();
            string output_filename =
                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                              + "_tsdf_wpca_deflation"
                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                              + "_comp_" + boost::lexical_cast<string>(k) + ".ply")).string();
            WPCASaveRelatedMatrices(&centralized_data_mat, NULL, NULL,
                                    base_mat, coeff_mat, output_filename);
            WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
                                   *base_mat, *coeff_mat, k + 1, *boundingbox_size, output_filename);
        }
    }
    return true;
}
void WPCASaveMeanTSDF(
        const Eigen::SparseVector<float>& mean_mat,
        const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
        const Eigen::Vector3i& boundingbox_size,
        const float voxel_length,
        const Eigen::Vector3f& offset,
        const float max_dist_pos,
        const float max_dist_neg,
        const string& save_filepath)
{
    cout << "save mean tsdf" << endl;
    const int sample_num = weight_mat.cols();
    //mean tsdf (in mean_mat)
    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
    Eigen::SparseMatrix<float, Eigen::ColMajor> tweight = (weight_mat * Eigen::VectorXf::Ones(sample_num, 1) / sample_num).sparseView();
    cpu_tsdf::ConvertDataMatrixToTSDFs(voxel_length,
                             offset,
                             max_dist_pos,
                             max_dist_neg,
                             boundingbox_size,
                             mean_mat,
                             tweight,
                             &projected_tsdf_models
                             );
    string output_dir = bfs::path(save_filepath).remove_filename().string();
    string output_meanply =
            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
                                      + "mean.ply")).string();
    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_meanply, false, true, 0);
}

void WPCASaveMeanTSDF(const Eigen::SparseVector<float>& mean_mat,
                             const cpu_tsdf::TSDFHashing& mean_tsdf,
                             const Eigen::SparseMatrix<float, ColMajor>& weight_mat,
                             const Eigen::Vector3i& boundingbox_size,
                             const string& save_filepath)
{
    cout << "save mean tsdf" << endl;
    //mean tsdf (in mean_mat)
    const float voxel_length = mean_tsdf.voxel_length();
    const Eigen::Vector3f offset = mean_tsdf.offset();
    float max_dist_pos, max_dist_neg;
    mean_tsdf.getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
    Eigen::SparseMatrix<float, Eigen::ColMajor> tweight = (weight_mat * Eigen::VectorXf::Ones(weight_mat.cols(), 1)).sparseView();
    cpu_tsdf::ConvertDataMatrixToTSDFs(voxel_length,
                             offset,
                             max_dist_pos,
                             max_dist_neg,
                             boundingbox_size,
                             mean_mat,
                             tweight,
                             &projected_tsdf_models
                             );
    string output_dir = bfs::path(save_filepath).remove_filename().string();
    string output_meanply =
            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
                                      + "mean.ply")).string();
    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_meanply, false, true, 0);

    //assert(*mean_tsdf == *(projected_tsdf_models[0]));
    //mean tsdf - tsdf version
//    {
//        std::string output_plyfilename = *save_filepath;
//        cout << "meshing pca projected models and template - meantsdf." << endl;
//        cpu_tsdf::TSDFHashing::Ptr ptr_temp_mean_tsdf(new cpu_tsdf::TSDFHashing);
//        *ptr_temp_mean_tsdf = *mean_tsdf;
//        ptr_temp_mean_tsdf->DisplayInfo();
//        {
//            std::cout << "begin marching cubes for model "   << std::endl;
//            cpu_tsdf::MarchingCubesTSDFHashing mc;
//            mc.setMinWeight(0);
//            mc.setInputTSDF (ptr_temp_mean_tsdf);
//            pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh);
//            fprintf(stderr, "perform reconstruction: \n");
//            mc.reconstruct (*mesh);
//            //PCL_INFO ("Entire pipeline took %f ms\n", tt.toc ());
//            flattenVertices(*mesh);
//            //string output_dir = bfs::path(input_model_filenames[i]).remove_filename().string();
//            string output_dir = bfs::path(output_plyfilename).remove_filename().string();
//            string output_modelply =
//                    (bfs::path(output_dir) / (bfs::path(output_plyfilename).stem().string()
//                                              + "_tsdf_pca_projected_modelply_"
//                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
//                                              + "_origin_mean_tsdfv" + ".ply")).string();
//            std::cout << "save tsdf file path: " << output_modelply << std::endl;
//            pcl::io::savePLYFileBinary (output_modelply, *mesh);
//        }
//    }
}

bool TSDFWeightedPCA_NewWrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
                               const int component_num, const int max_iter,
                               cpu_tsdf::TSDFHashing *mean_tsdf, std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models,
                               const std::string *save_filepath)
{
    Eigen::SparseMatrix<float, Eigen::ColMajor> weight_mat;
    Eigen::SparseMatrix<float, Eigen::ColMajor> base_mat;
    Eigen::MatrixXf coeff_mat;
    Eigen::SparseVector<float> mean_mat;
    Eigen::Vector3i voxel_bounding_box_size;

    // 1st: test whether the PCA part are identical
    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
    cout << "00. convert TSDFs to sparse matrix" << endl;
    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, &weight_mat, &mean_mat, mean_tsdf, &voxel_bounding_box_size)) return false;

    Eigen::SparseVector<float> new_mean_mat;
    PCAOptions options;
    options.boundingbox_size = voxel_bounding_box_size;
    options.lambda_scale_diff = 1.0;
    tsdf_models[0]->getDepthTruncationLimits(options.max_dist_pos, options.max_dist_neg);
    options.min_model_weight = 0;
    options.offset = Eigen::Vector3f(0, 0, 0);
    options.voxel_length = tsdf_models[0]->voxel_length();
    if (save_filepath)
        options.save_path = *save_filepath;
   // cpu_tsdf::WeightedPCADeflationOrthogonal(centralized_data_mat, weiht_mat, component_num, max_iter, &new_mean_mat, &base_mat, &coeff_mat, options);

    //  TSDFWeightedPCA(tsdf_models, component_num, max_iter,
    //                  mean_tsdf, &mean_mat,
    //                  &base_mat, &coeff_mat, &weight_mat,
    //                  &voxel_bounding_box_size, save_filepath);
    //  TSDFWeightedPCADeflation(tsdf_models, component_num, max_iter,
    //                           mean_tsdf, &mean_mat,
    //                           &base_mat, &coeff_mat, &weight_mat,
    //                           &voxel_bounding_box_size, save_filepath);
    TSDFWeightedPCADeflationOrthogonal_Old(tsdf_models, component_num, max_iter,
                                       mean_tsdf, &mean_mat,
                                       &base_mat, &coeff_mat, &weight_mat,
                                       &voxel_bounding_box_size, save_filepath);

    if (save_filepath)
    {
        string output_file = (*save_filepath).substr(0, save_filepath->length() - 4) + "_boundingbox.txt";
        ofstream os(output_file);
        os << voxel_bounding_box_size << endl;
    }
    const int data_dim = base_mat.rows();
    const int sample_num = tsdf_models.size();
    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
            mean_mat * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
            base_mat * coeff_mat.sparseView().eval();
    const float voxel_length = mean_tsdf->voxel_length();
    const Eigen::Vector3f offset = mean_tsdf->offset();
    float max_dist_pos, max_dist_neg;
    mean_tsdf->getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    return ConvertDataMatrixToTSDFsNoWeight(voxel_length,
                                            offset,
                                            max_dist_pos,
                                            max_dist_neg,
                                            voxel_bounding_box_size,
                                            projected_data_mat,
                                            projected_tsdf_models
                                            );
    //  return ConvertDataMatrixToTSDFs(voxel_length,
    //                                  offset,
    //                                  max_dist_pos,
    //                                  max_dist_neg,
    //                                  voxel_bounding_box_size,
    //                                  projected_data_mat,
    //                                  weight_mat,
    //                                  projected_tsdf_models
    //                                  );
}


/**
 * @brief cpu_tsdf::TSDFWeightedPCA
 * Wrapper function for doing PCA
 * @param tsdf_models
 * @param component_num
 * @param max_iter
 * @param mean_tsdf
 * @param projected_tsdf_models
 * @param save_filepath
 * @return
 */
bool TSDFWeightedPCA_Wrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models, const int component_num, const int max_iter,
                               cpu_tsdf::TSDFHashing *mean_tsdf, std::vector<cpu_tsdf::TSDFHashing::Ptr> *projected_tsdf_models,
                               const std::string *save_filepath)
{
    Eigen::SparseMatrix<float, Eigen::ColMajor> weight_mat;
    Eigen::SparseMatrix<float, Eigen::ColMajor> base_mat;
    Eigen::MatrixXf coeff_mat;
    Eigen::SparseVector<float> mean_mat;
    Eigen::Vector3i voxel_bounding_box_size;
    //  TSDFWeightedPCA(tsdf_models, component_num, max_iter,
    //                  mean_tsdf, &mean_mat,
    //                  &base_mat, &coeff_mat, &weight_mat,
    //                  &voxel_bounding_box_size, save_filepath);
    //  TSDFWeightedPCADeflation(tsdf_models, component_num, max_iter,
    //                           mean_tsdf, &mean_mat,
    //                           &base_mat, &coeff_mat, &weight_mat,
    //                           &voxel_bounding_box_size, save_filepath);
    TSDFWeightedPCADeflationOrthogonal_Old(tsdf_models, component_num, max_iter,
                                       mean_tsdf, &mean_mat,
                                       &base_mat, &coeff_mat, &weight_mat,
                                       &voxel_bounding_box_size, save_filepath);

    if (save_filepath)
    {
        string output_file = (*save_filepath).substr(0, save_filepath->length() - 4) + "_boundingbox.txt";
        ofstream os(output_file);
        os << voxel_bounding_box_size << endl;
    }
    const int data_dim = base_mat.rows();
    const int sample_num = tsdf_models.size();
    Eigen::SparseMatrix<float, Eigen::ColMajor> projected_data_mat =
            mean_mat * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval() +
            base_mat * coeff_mat.sparseView().eval();
    const float voxel_length = mean_tsdf->voxel_length();
    const Eigen::Vector3f offset = mean_tsdf->offset();
    float max_dist_pos, max_dist_neg;
    mean_tsdf->getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    return ConvertDataMatrixToTSDFsNoWeight(voxel_length,
                                            offset,
                                            max_dist_pos,
                                            max_dist_neg,
                                            voxel_bounding_box_size,
                                            projected_data_mat,
                                            projected_tsdf_models
                                            );
    //  return ConvertDataMatrixToTSDFs(voxel_length,
    //                                  offset,
    //                                  max_dist_pos,
    //                                  max_dist_neg,
    //                                  voxel_bounding_box_size,
    //                                  projected_data_mat,
    //                                  weight_mat,
    //                                  projected_tsdf_models
    //                                  );
}

bool TSDFWeightedPCA_Wrapper(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models, const int component_num,
                               const int max_iter,
                               cpu_tsdf::TSDFHashing* mean_tsdf, Eigen::SparseVector<float> *mean_mat,
                               Eigen::SparseMatrix<float, ColMajor> *base_mat,
                               Eigen::MatrixXf *coeff_mat,
                               Eigen::SparseMatrix<float, ColMajor> *weight_mat,
                               Eigen::Vector3i *boundingbox_size,
                               const std::string* save_filepath)
{
    // #samples: N, #dim: D, #principal_components: K
    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
    cout << "begin weighted pca" << endl;
    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
    cout << "1. convert TSDFs to sparse matrix" << endl;
    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, weight_mat, mean_mat, mean_tsdf, boundingbox_size)) return false;
    const int data_dim = centralized_data_mat.rows();
    const int sample_num = centralized_data_mat.cols();
    if (save_filepath)
    {
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_original" + ".ply")).string();
        // 1. save the original mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat, NULL, NULL, output_filename);
        // 2. save the tsdfs
        WPCASaveUncentralizedTSDFs(centralized_data_mat, *weight_mat, *mean_mat, *mean_tsdf,
                                   *boundingbox_size, output_filename);
        WPCASaveMeanTSDF(*mean_mat, *mean_tsdf, *weight_mat, *boundingbox_size, output_filename);
    }
    base_mat->resize(data_dim, sample_num);
    coeff_mat->resize(component_num, sample_num);
    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(*weight_mat);
    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
    // initialize base_mat U
    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
    cout << "data_dim: " << data_dim << endl;
    cout << "sample_num: " << sample_num << endl;
    cout << "compo_num: " << component_num << endl;
    cerr << "2. Initial Estimate" << endl;
    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
    *base_mat = base_mat_row_major;
    /// save
    if (save_filepath)
    {
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_init_estimated" + ".ply")).string();
        // 1. save the initial mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
                                base_mat, coeff_mat, output_filename);
        // 2. save the tsdfs
        WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
                               *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
    }
    // do EM
    cerr << "3. Do EM" << endl;
    for (int i = 0; i < max_iter; ++i)
    {
        cout << "Iteration: \nEstep " << i <<  endl;
        //E step
        ComputeCoeffInEStep(centralized_data_mat, *weight_mat, *base_mat, coeff_mat);
        //M step
        cout << "Mstep " << i <<  endl;
        ComputeBasisInMStep(centralized_data_mat_row_major, weight_mat_row_major, *coeff_mat, &base_mat_row_major);
        *base_mat = base_mat_row_major;  // update colmajor base mat;
        /// save
        if (save_filepath)
        {
            string output_dir = bfs::path(*save_filepath).remove_filename().string();
            string output_filename =
                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                              + "_tsdf_wpca"
                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                              + "_itr_" + boost::lexical_cast<string>(i) + ".ply")).string();
            WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
                                    base_mat, coeff_mat, output_filename);
            WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
                                   *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
        }
    }
    return true;
}

bool TSDFWeightedPCADeflation(const std::vector<cpu_tsdf::TSDFHashing::Ptr> &tsdf_models,
                                        const int component_num,
                                        const int max_iter,
                                        cpu_tsdf::TSDFHashing *mean_tsdf,
                                        Eigen::SparseVector<float> *mean_mat,
                                        Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat,
                                        Eigen::MatrixXf *coeff_mat, Eigen::SparseMatrix<float, Eigen::ColMajor> *weight_mat,
                                        Eigen::Vector3i *boundingbox_size, const string *save_filepath)
{
    // #samples: N, #dim: D, #principal_components: K
    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
    cout << "begin weighted pca" << endl;
    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat;
    cout << "1. convert TSDFs to sparse matrix" << endl;
    if (!InitializeDataMatrixFromTSDFs(tsdf_models, &centralized_data_mat, weight_mat, mean_mat, mean_tsdf, boundingbox_size)) return false;
    const int data_dim = centralized_data_mat.rows();
    const int sample_num = centralized_data_mat.cols();
    if (save_filepath)
    {
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_original" + ".ply")).string();
        // 1. save the original mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat, NULL, NULL, output_filename);
        // 2. save the tsdfs
        WPCASaveUncentralizedTSDFs(centralized_data_mat, *weight_mat, *mean_mat, *mean_tsdf,
                                   *boundingbox_size, output_filename);
        WPCASaveMeanTSDF(*mean_mat, *mean_tsdf, *weight_mat, *boundingbox_size, output_filename);
    }
    base_mat->resize(data_dim, sample_num);
    coeff_mat->resize(component_num, sample_num);
    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(*weight_mat);
    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
    // initialize base_mat U
    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
    cout << "data_dim: " << data_dim << endl;
    cout << "sample_num: " << sample_num << endl;
    cout << "compo_num: " << component_num << endl;
    cerr << "2. Initial Estimate" << endl;
    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
    *base_mat = base_mat_row_major;
    if (save_filepath)
    {
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_init_estimated" + ".ply")).string();
        // 1. save the initial mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, weight_mat,
                                base_mat, coeff_mat, output_filename);
        // 2. save the tsdfs
        WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
                               *base_mat, *coeff_mat, component_num, *boundingbox_size, output_filename);
    }
    // do EM
    cerr << "3. Do EM" << endl;
    const float thresh = 1e-4;
    Eigen::VectorXf prev_base(data_dim);
    Eigen::VectorXf current_base(data_dim);
    Eigen::VectorXf current_coeff(sample_num);
    const int iteration_number = std::max(sample_num, 25);
    for (int k = 0; k < component_num; ++k)
    {
        current_base = base_mat->col(k);
        current_coeff = coeff_mat->row(k);
        prev_base = current_base;
        cout << "Computing " << k << "th component." << endl;
        for (int i = 0; i < iteration_number; ++i)
        {
            cout << "Iteration: \nEstep " << i <<  endl;
            //E step
            ComputeCoeffInEStepOneVec(centralized_data_mat, *weight_mat, current_base, &current_coeff);
            //M step
            cout << "Mstep " << i <<  endl;
            ComputeBasisInMStepOneVec(centralized_data_mat_row_major, weight_mat_row_major, current_coeff, &current_base);
            ////////////////////////////////////////////////
            /// save
            if (save_filepath)
            {
                string output_dir = bfs::path(*save_filepath).remove_filename().string();
                string output_filename =
                        (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                                  + "_tsdf_wpca_deflation"
                                                  + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                                  + "_comp_" + boost::lexical_cast<string>(k)
                                                  + "_itr_" + boost::lexical_cast<string>(i)
                                                  + ".ply")).string();
                Eigen::SparseMatrix<float, ColMajor> temp_base = current_base.sparseView();
                Eigen::MatrixXf temp_coeff = current_coeff;
                WPCASaveRelatedMatrices(NULL, NULL, NULL,
                                        &temp_base, &temp_coeff, output_filename);
                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
                        =
                        (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval()
                        + temp_base * temp_coeff.transpose().sparseView().eval();
                WPCASaveTSDFsFromMats(temp_projected_data_mat, *mean_tsdf, *boundingbox_size, output_filename);
            }
            ////////////////////////////////////////////////
            //test convergence
            float t = (prev_base.normalized() - current_base.normalized()).cwiseAbs().sum();
            cerr << "difference.. " << t << endl;
            if (t < thresh)
            {
                cerr << "converge reached.. " << endl;
                break;
            }
            prev_base = current_base;
        }
        for (int p = 0; p < data_dim; ++p)
        {
            if (abs(current_base[p]) > std::numeric_limits<double>::epsilon())
            {
                base_mat->coeffRef(p, k) = current_base[p];
            }
        }
        (*coeff_mat).row(k) = current_coeff;
        centralized_data_mat -= weight_mat->cwiseProduct(current_base.sparseView() * current_coeff.transpose());
        centralized_data_mat_row_major = centralized_data_mat;
        if (save_filepath)
        {
            string output_dir = bfs::path(*save_filepath).remove_filename().string();
            string output_filename =
                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                              + "_tsdf_wpca_deflation"
                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                              + "_comp_" + boost::lexical_cast<string>(k) + ".ply")).string();
            WPCASaveRelatedMatrices(&centralized_data_mat, NULL, NULL,
                                    base_mat, coeff_mat, output_filename);
            WPCASaveRecoveredTSDFs(*mean_mat, *mean_tsdf,
                                   *base_mat, *coeff_mat, k + 1, *boundingbox_size, output_filename);
        }
    }
    return true;
}

bool WeightedPCADeflationOrthogonal(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples, const Eigen::SparseMatrix<float, Eigen::ColMajor> &weights, const int component_num, const int max_iter, Eigen::SparseVector<float> *mean_mat, Eigen::SparseVector<float> *mean_weight, Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf *coeff_mat, const tsdf_utility::OptimizationParams &options)
{
    const int sample_num = samples.cols();
    const int data_dim = samples.rows();
    if (sample_num == 0 || data_dim == 0) return false;
    // #samples: N, #dim: D, #principal_components: K
    // BaseMat: D * K; coeff_mat: K * N; mean_mat: D * 1;
    cout << "begin weighted pca" << endl;
    cerr << "1. compute mean mat" << endl;
    // ((Samples.*Weight)) * ones(N, 1) ./ Weight * ones(N, 1)
    Eigen::VectorXf onesN = Eigen::MatrixXf::Ones(sample_num, 1);
    // *mean_mat = (((samples.cwiseProduct(weights)) * onesN).cwiseQuotient( (weights * onesN) )).sparseView();
    Eigen::SparseVector<float> sum_vec = ((samples.cwiseProduct(weights)) * onesN).sparseView();
    Eigen::SparseVector<float> sum_weights = (weights * onesN).sparseView();
    mean_mat->resize(data_dim);
    mean_mat->reserve(sum_weights.nonZeros());
    for (Eigen::SparseVector<float>::InnerIterator it(sum_weights); it; ++it)
    {
        const float cur_weight = it.value();
        const int cur_idx = it.index();
        if (cur_weight > 0)
        {
            mean_mat->insert(cur_idx) = sum_vec.coeff(cur_idx) / cur_weight;
        }
    }
    *mean_weight = sum_weights / sample_num;
    Eigen::SparseMatrix<float, ColMajor> centralized_data_mat = samples - (*mean_mat) * onesN.transpose();
    if (!options.save_path.empty())
    {
        const string* save_filepath = &options.save_path;
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation_ortho"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_original" + "pp.ply")).string();
        // 1. save the original mats for matlab debug
        // WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, &weights, NULL, NULL, output_filename);
        // 2. save the tsdfs
        //        WPCASaveUncentralizedTSDFs(centralized_data_mat, weights, *mean_mat,
        //                                   options.boundingbox_size, options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg,
        //                                   output_filename);
        WPCASaveMeanTSDF(*mean_mat, weights,
                         options.sample_size, options.VoxelLength(), options.Offset(), options.max_dist_pos, options.max_dist_neg,
                         output_filename);
    }
    if (component_num == 0) return true;
    base_mat->resize(data_dim, sample_num);
    coeff_mat->resize(component_num, sample_num);
    Eigen::SparseMatrix<float, RowMajor> weight_mat_row_major(weights);
    Eigen::SparseMatrix<float, RowMajor> centralized_data_mat_row_major(centralized_data_mat);
    // initialize base_mat U
    Eigen::SparseMatrix<float, RowMajor> base_mat_row_major;
    cout << "data_dim: " << data_dim << endl;
    cout << "sample_num: " << sample_num << endl;
    cout << "compo_num: " << component_num << endl;
    cerr << "2. Initial Estimate" << endl;
    InitialEstimate(centralized_data_mat_row_major, weight_mat_row_major, component_num, &base_mat_row_major, coeff_mat);
    *base_mat = base_mat_row_major;
    if (!options.save_path.empty())
    {
        const string* save_filepath = &options.save_path;
        string output_dir = bfs::path(*save_filepath).remove_filename().string();
        string output_filename =
                (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                          + "_tsdf_wpca_deflation_ortho"
                                          + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                          + "_init_estimated" + ".ply")).string();
        // 1. save the initial mats for matlab debug
        WPCASaveRelatedMatrices(&centralized_data_mat, mean_mat, &weights,
                                base_mat, coeff_mat, output_filename);
        // 2. save the tsdfs
        WPCASaveRecoveredTSDFs(*mean_mat, *base_mat, *coeff_mat, component_num,
                               options.sample_size, options.VoxelLength(), options.Offset(), options.max_dist_pos, options.max_dist_neg,
                               output_filename);
    }
    // do EM
    cerr << "3. Do EM" << endl;
    const float thresh = 1e-4;
    Eigen::VectorXf prev_base(data_dim);
    Eigen::VectorXf current_base(data_dim);
    Eigen::VectorXf current_coeff(sample_num);
    const int iteration_number = std::max(sample_num, max_iter);
    for (int k = 0; k < component_num; ++k)
    {
        current_base = base_mat->col(k);
        current_coeff = coeff_mat->row(k);
        prev_base = current_base;
        cout << "Computing " << k << "th component." << endl;
        for (int i = 0; i < iteration_number; ++i)
        {
            cout << "Iteration: \nEstep " << i <<  endl;
            //E step
            ComputeCoeffInEStepOneVec(centralized_data_mat, weights, current_base, &current_coeff);
            //M step
            cout << "Mstep " << i <<  endl;
            ComputeBasisInMStepOneVec(centralized_data_mat_row_major, weight_mat_row_major, current_coeff, &current_base);
            //orthogonalize:
            float component_norm = 0;
            OrthogonalizeVector(*base_mat, k, &current_base, &component_norm);
            current_coeff *= component_norm;
            ////////////////////////////////////////////////
            /// save
            if (!options.save_path.empty() && (i%10) == 0)
            {
                const string* save_filepath = &options.save_path;
                string output_dir = bfs::path(*save_filepath).remove_filename().string();
                string output_filename =
                        (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                                  + "_tsdf_wpca_deflation_ortho"
                                                  + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                                  + "_comp_" + boost::lexical_cast<string>(k)
                                                  + "_itr_" + boost::lexical_cast<string>(i)
                                                  + ".ply")).string();
                Eigen::SparseMatrix<float, ColMajor> temp_base = current_base.sparseView();
                Eigen::MatrixXf temp_coeff = current_coeff;
                WPCASaveRelatedMatrices(NULL, NULL, NULL,
                                        &temp_base, &temp_coeff, output_filename);
                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
                        =
                        (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval()
                        + temp_base * temp_coeff.transpose().sparseView().eval();
        WPCASaveTSDFsFromMats(temp_projected_data_mat,
                              options.sample_size, options.VoxelLength(), options.Offset(), options.max_dist_pos, options.max_dist_neg,
                              output_filename);
                //                Eigen::SparseMatrix<float, Eigen::ColMajor> temp_projected_data_mat
                //                        = (*mean_mat) * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Constant(1, sample_num, 1)).sparseView().eval()
                //                        + (*base_mat).leftCols(k) * (*coeff_mat).topRows(k).sparseView().eval();
                //                WPCASaveRecoveredTSDFs(temp_projected_data_mat, *mean_tsdf,
                //                                       temp_base, temp_coeff, 1, *boundingbox_size, output_filename);
            }
            ////////////////////////////////////////////////
            //test convergence
            float t = (prev_base.normalized() - current_base.normalized()).cwiseAbs().sum();
            cerr << "difference.. " << t << endl;
            if (t < thresh)
            {
                cerr << "converge reached.. " << endl;
                break;
            }
            prev_base = current_base;
        }  // end for i (iteration)
        for (int p = 0; p < data_dim; ++p)
        {
            if (abs(current_base[p]) > std::numeric_limits<double>::epsilon())
            {
                //base_mat->insert(p, k) = current_base[p];
                base_mat->coeffRef(p, k) = current_base[p];
            }
        }
        (*coeff_mat).row(k) = current_coeff;
        centralized_data_mat -= weights.cwiseProduct(current_base.sparseView() * current_coeff.transpose());
        centralized_data_mat_row_major = centralized_data_mat;
        if (!options.save_path.empty())
        {
            const string* save_filepath = &options.save_path;
            string output_dir = bfs::path(*save_filepath).remove_filename().string();
            string output_filename =
                    (bfs::path(output_dir) / (bfs::path(*save_filepath).stem().string()
                                              + "_tsdf_wpca_deflation"
                                              + "_pcanum_" + boost::lexical_cast<string>(component_num)
                                              + "_comp_" + boost::lexical_cast<string>(k) + ".ply")).string();
            WPCASaveRelatedMatrices(&centralized_data_mat, NULL, NULL,
                                    base_mat, coeff_mat, output_filename);
            WPCASaveRecoveredTSDFs(*mean_mat, *base_mat, *coeff_mat, k + 1,
                              options.sample_size, options.VoxelLength(), options.Offset(), options.max_dist_pos, options.max_dist_neg,
                                   output_filename);
        }
    }  // end for k
    return true;
}



//void GetClusterSampleIdx(const std::vector<int> &model_assign_idx, const int model_number, std::vector<std::vector<int>>* cluster_sample_idx)
//{
//    const int sample_number = model_assign_idx.size();
//    (*cluster_sample_idx).resize(model_number);
//    for (int i = 0; i < sample_number; ++i)
//    {
//        (*cluster_sample_idx)[model_assign_idx[i]].push_back(i);
//    }
//    return;
//}



}
