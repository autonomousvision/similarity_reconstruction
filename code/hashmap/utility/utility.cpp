#include "utility.h"

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
//#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Eigen>

#include "common/utility/pcl_utility.h"
#include "common/utility/eigen_utility.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"

namespace bfs = boost::filesystem;
using namespace std;



namespace cpu_tsdf {
bool ComputeOrientedBoundingboxVertices(const Eigen::Matrix3f &voxel_world_rotation, const Eigen::Vector3f &offset, const Eigen::Vector3f &world_side_lengths,
	Eigen::Vector3f* box_vertices)
{
	for (int x = 0; x < 2; ++x)
		for (int y = 0; y < 2; ++y)
			for (int z = 0; z < 2; ++z)
			{
		box_vertices[x * 4 + y * 2 + z] = offset + x * world_side_lengths[0] * voxel_world_rotation.col(0) +
			y * world_side_lengths[1] * voxel_world_rotation.col(1) +
			z * world_side_lengths[2] * voxel_world_rotation.col(2);
			}
	return true;
}

bool AffineToOrientedBB(const Eigen::Affine3f &trans, OrientedBoundingBox *obb)
{
    Eigen::Matrix3f rot;
    Eigen::Vector3f scales3d;
    Eigen::Vector3f trans3d;
    utility::EigenAffine3fDecomposition(trans, &rot, &scales3d, &trans3d);
    obb->bb_offset = trans3d - (
                (rot * scales3d.asDiagonal() * Eigen::Vector3f::Ones())
                / 2.0);
    obb->bb_orientation = rot;
    obb->bb_sidelengths = scales3d;
    //obb->voxel_lengths = 0.2;
    return true;
}


bool AffinesToOrientedBBs(const std::vector<Eigen::Affine3f> &trans, std::vector<OrientedBoundingBox> *obbs)
{
    for (int i = 0; i < trans.size(); ++i)
    {
        OrientedBoundingBox cur_obb;
        AffineToOrientedBB(trans[i], &cur_obb);
        obbs->push_back(cur_obb);
    }
    return true;
}

void OBBsToAffines(const std::vector<OrientedBoundingBox> &obbs, std::vector<Eigen::Affine3f> *transforms)
{
    transforms->resize(obbs.size());
    for (int i = 0; i < obbs.size(); ++i)
    {
        OBBToAffine(obbs[i], &((*transforms)[i]));
    }
}

void OBBToAffine(const OrientedBoundingBox &obb, Eigen::Affine3f *transform)
{
    Eigen::Vector3f scale3d = obb.bb_sidelengths;
    Eigen::Matrix3f rotation = obb.bb_orientation;
    Eigen::Vector3f offset = obb.BoxCenter();
    Eigen::Matrix4f transform_mat = Eigen::Matrix4f::Zero();
    transform_mat.block<3, 3>(0, 0) = rotation * scale3d.asDiagonal();
    transform_mat.block<3, 1>(0, 3) = offset;
    transform_mat.coeffRef(3, 3) = 1;
    transform->matrix() = transform_mat;
}

Eigen::SparseMatrix<float, Eigen::ColMajor> SparseVectorsToEigenMat(const std::vector<Eigen::SparseVector<float> > &samples)
{
    if (samples.empty()) return Eigen::SparseMatrix<float, Eigen::ColMajor>();
    Eigen::SparseMatrix<float, Eigen::ColMajor> res(samples[0].size(), samples.size());
    for (int i = 0; i < samples.size(); ++i)
    {
        res.col(i) = samples[i];
    }
    return res;
}

std::vector<Eigen::SparseVector<float> > EigenMatToSparseVectors(const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples)
{
    std::vector<Eigen::SparseVector<float> > res(samples.cols());
    for (int i = 0; i < samples.cols(); ++i)
    {
        res[i] = samples.col(i);
    }
    return res;
}

//bool ExtractSamplesFromAffineTransform(const TSDFHashing &scene_tsdf, const std::vector<Eigen::Affine3f> &affine_transforms, const PCAOptions &options, Eigen::SparseMatrix<float, Eigen::ColMajor> *samples, Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
//{
//    const int sample_num = affine_transforms.size();
//    const int feature_dim = options.boundingbox_size[0] * options.boundingbox_size[1] * options.boundingbox_size[2];
//    samples->resize(feature_dim, sample_num);
//    samples->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    weights->resize(feature_dim, sample_num);
//    weights->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    for (int i = 0; i < sample_num; ++i)
//    {
//        Eigen::SparseVector<float> sample(feature_dim);
//        Eigen::SparseVector<float> weight(feature_dim);
//        ExtractOneSampleFromAffineTransform(scene_tsdf, affine_transforms[i], options,
//                                            &sample,
//                                            &weight);
//        Eigen::SparseVector<float>::InnerIterator it_s(sample);
//        for (Eigen::SparseVector<float>::InnerIterator it(weight); it && it_s; ++it, ++it_s)
//        {
//            CHECK(it_s.index() == it.index());
//            samples->insert(it.index(), i) = it_s.value();
//            weights->insert(it.index(), i) = it.value();
//        }

//        ///////////////////////////////////////////////
//        //        Eigen::Matrix3f test_r;
//        //        Eigen::Vector3f test_scale;
//        //        Eigen::Vector3f test_trans;
//        //        utility::EigenAffine3fDecomposition(
//        //                    affine_transforms[i],
//        //                    &test_r,
//        //                    &test_scale,
//        //                    &test_trans);
//        //        TSDFHashing::Ptr cur_tsdf(new TSDFHashing);
//        //        ConvertDataVectorToTSDFWithWeight(
//        //        sample,
//        //        weight,
//        //        options,
//        //        cur_tsdf.get());
//        //        bfs::path output_path(options.save_path);
//        //        string save_path = (output_path.parent_path()/output_path.stem()).string() + "_check_affine_conversion_" + boost::lexical_cast<string>(i) + ".ply";
//        //        SaveTSDFModel(cur_tsdf, save_path, false, true, options.min_model_weight);
//        ///////////////////////////////////////////////
//    }
//    return true;
//}

//bool ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const PCAOptions &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset;
//    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size;
//    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
//    sample->resize(total_voxel_size);
//    //sample->reserve(total_voxel_size * 0.6);
//    weight->resize(total_voxel_size);
//    //weight->reserve(total_voxel_size * 0.6);
//    for (int x = 0; x < options.boundingbox_size[0]; ++x)
//        for (int y = 0; y < options.boundingbox_size[1]; ++y)
//            for (int z = 0; z < options.boundingbox_size[2]; ++z)
//            {
//                Eigen::Vector3f current_world_point = offset;
//                current_world_point[0] += options.voxel_length *  x;
//                current_world_point[1] += options.voxel_length *  y;
//                current_world_point[2] += options.voxel_length *  z;
//                Eigen::Vector3f transformed_world_point = affine_transform * current_world_point;
//                float cur_d, cur_w;
//                if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) && /*cur_w > 0*/ cur_w > options.min_model_weight)
//                {
//                    int current_index = z +
//                            (y + x * options.boundingbox_size[1]) * options.boundingbox_size[2];
//                    sample->coeffRef(current_index) = cur_d;
//                    weight->coeffRef(current_index) = cur_w;
//                }
//            }
//    return true;
//}

//bool ExtractOneSampleFromAffineTransform2(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const PCAOptions &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset;
//    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size;
//    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
//    sample->resize(total_voxel_size);
//    weight->resize(total_voxel_size);
//    for (int x = 0; x < options.boundingbox_size[0]; ++x)
//        for (int y = 0; y < options.boundingbox_size[1]; ++y)
//            for (int z = 0; z < options.boundingbox_size[2]; ++z)
//            {
//                Eigen::Vector3f current_world_point = offset;
//                current_world_point[0] += options.voxel_length *  x;
//                current_world_point[1] += options.voxel_length *  y;
//                current_world_point[2] += options.voxel_length *  z;
//                Eigen::Vector3f transformed_world_point = affine_transform * current_world_point;
//                float cur_d, cur_w;
//        if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) &&
//                        cur_w > options.min_model_weight)
//                {
//                    int current_index = z +
//                            (y + x * options.boundingbox_size[1]) * options.boundingbox_size[2];
//                    sample->coeffRef(current_index) = cur_d;
//                    weight->coeffRef(current_index) = cur_w;
//                }
//            }
//    return true;
//}

bool ConvertDataVectorToTSDFWithWeight(
        const Eigen::SparseVector<float> &tsdf_data_vec,
        const Eigen::SparseVector<float> &tsdf_weight_vec,
        const PCAOptions &options, TSDFHashing *tsdf)
{
    tsdf->Init(options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg);
    const int size_yz = options.boundingbox_size[1] * options.boundingbox_size[2];
    const Eigen::Vector3i voxel_bounding_box_size = options.boundingbox_size;
    // const float ratio = options.ratio_original_voxel_length_to_unit_cube_vlength;
    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_weight_vec); it; ++it)
    {
        int data_dim_idx = it.index();
        const float weight = it.value();
        if ( weight == 0 || weight < options.min_model_weight ) continue;
        const float dist = tsdf_data_vec.coeff(data_dim_idx);
        cv::Vec3i pos;
        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
        pos[0] = data_dim_idx / size_yz;
        tsdf->AddObservation(pos, dist, weight, cv::Vec3b(127, 127, 127));
    }
    //assert(tsdf_weight_vec.nonZeros() == tsdf->vo)
    tsdf->DisplayInfo();
    return true;
}

bool ConvertDataVectorToTSDFWithWeightAndWorldPos(
        const Eigen::SparseVector<float> &tsdf_data_vec,
        const Eigen::SparseVector<float> &tsdf_weight_vec,
        const PCAOptions &options,
        TSDFHashing *tsdf,
        std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f>>* idx_worldpos)
{
    //tsdf->CopyHashParametersFrom(options.tsdf_for_parameters);
    tsdf->Init(options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg);
    const int size_yz = options.boundingbox_size[1] * options.boundingbox_size[2];
    const Eigen::Vector3i voxel_bounding_box_size = options.boundingbox_size;
    // const float ratio = options.ratio_original_voxel_length_to_unit_cube_vlength;
    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_weight_vec); it; ++it)
    {
        int data_dim_idx = it.index();
        const float weight = it.value();
        if ( weight < options.min_model_weight ) continue;
        const float dist = tsdf_data_vec.coeff(data_dim_idx);
        cv::Vec3i pos;
        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
        pos[0] = data_dim_idx / size_yz;
        tsdf->AddObservation(pos, dist, weight, cv::Vec3b(255, 255, 255));

        cv::Vec3f worldpos = tsdf->Voxel2World(cv::Vec3f(pos));
        idx_worldpos->insert(std::make_pair(data_dim_idx,
                                            std::make_pair(
                                                utility::CvVectorToEigenVector3(pos),
                                                utility::CvVectorToEigenVector3(worldpos))));
    }
    //assert(tsdf_weight_vec.nonZeros() == tsdf->vo)
    tsdf->DisplayInfo();
    return true;
}

Eigen::Vector3i TSDFGridInfo::boundingbox_size() const
{
    return boundingbox_size_;
}

void TSDFGridInfo::boundingbox_size(const Eigen::Vector3i &value)
{
    boundingbox_size_ = value;
}

Eigen::Vector3f TSDFGridInfo::offset() const
{
    return offset_;
}

void TSDFGridInfo::offset(const Eigen::Vector3f &value)
{
    offset_ = value;
}

Eigen::Vector3f TSDFGridInfo::voxel_lengths() const
{
    return voxel_lengths_;
}

void TSDFGridInfo::voxel_lengths(Eigen::Vector3f value)
{
    voxel_lengths_ = value;
}

float TSDFGridInfo::max_dist_pos() const
{
    return max_dist_pos_;
}

void TSDFGridInfo::max_dist_pos(float value)
{
    max_dist_pos_ = value;
}

float TSDFGridInfo::max_dist_neg() const
{
    return max_dist_neg_;
}

void TSDFGridInfo::max_dist_neg(float value)
{
    max_dist_neg_ = value;
}

float TSDFGridInfo::min_model_weight() const
{
    return min_model_weight_;
}

void TSDFGridInfo::min_model_weight(float value)
{
    min_model_weight_ = value;
}

void TSDFGridInfo::InitFromVoxelBBSize(const TSDFHashing &tsdf_model, const float vmin_mesh_weight)
{
    //offset_ = Eigen::Vector3f(-0.5, -0.5, -0.5);
    offset_ = Eigen::Vector3f(-0.5, -0.5, 0);
    voxel_lengths_ = Eigen::Vector3f::Ones().cwiseQuotient(boundingbox_size_.cast<float>() - Eigen::Vector3f::Ones());
    min_model_weight_ = vmin_mesh_weight;
    tsdf_model.getDepthTruncationLimits(max_dist_pos_, max_dist_neg_);
}

TSDFGridInfo::TSDFGridInfo(const TSDFHashing &tsdf_model,
        const Eigen::Vector3i obb_boundingbox_voxel_size,
        const float vmin_mesh_weight)
{
    boundingbox_size_ = obb_boundingbox_voxel_size;
    InitFromVoxelBBSize(tsdf_model, vmin_mesh_weight);
}

//bool ExtractSamplesFromAffineTransform(const TSDFHashing &scene_tsdf, const std::vector<Eigen::Affine3f> &affine_transforms, const TSDFGridInfo &options, Eigen::SparseMatrix<float, Eigen::ColMajor> *samples, Eigen::SparseMatrix<float, Eigen::ColMajor> *weights)
//{
//    const int sample_num = affine_transforms.size();
//    const int feature_dim = options.boundingbox_size()[0] * options.boundingbox_size()[1] * options.boundingbox_size()[2];
//    samples->resize(feature_dim, sample_num);
//    samples->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    weights->resize(feature_dim, sample_num);
//    weights->reserve(Eigen::VectorXi::Constant(feature_dim, feature_dim * 0.6));
//    for (int i = 0; i < sample_num; ++i)
//    {
//        Eigen::SparseVector<float> sample(feature_dim);
//        Eigen::SparseVector<float> weight(feature_dim);
//        ExtractOneSampleFromAffineTransform(scene_tsdf, affine_transforms[i], options,
//                                            &sample,
//                                            &weight);
//        Eigen::SparseVector<float>::InnerIterator it_s(sample);
//        for (Eigen::SparseVector<float>::InnerIterator it(weight); it && it_s; ++it, ++it_s)
//        {
//            CHECK_EQ(it_s.index(), it.index());
//            samples->insert(it.index(), i) = it_s.value();
//            weights->insert(it.index(), i) = it.value();
//        }
//    }
//    return true;

//}

//bool ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const TSDFGridInfo &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset();
//    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size();
//    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
//    sample->resize(total_voxel_size);
//    //sample->reserve(total_voxel_size * 0.6);
//    weight->resize(total_voxel_size);
//    //weight->reserve(total_voxel_size * 0.6);
//    for (int x = 0; x < options.boundingbox_size()[0]; ++x)
//        for (int y = 0; y < options.boundingbox_size()[1]; ++y)
//            for (int z = 0; z < options.boundingbox_size()[2]; ++z)
//            {
//                Eigen::Vector3f current_world_point = offset;
//                current_world_point[0] += options.voxel_lengths()[0] *  x;
//                current_world_point[1] += options.voxel_lengths()[1] *  y;
//                current_world_point[2] += options.voxel_lengths()[2] *  z;
//                Eigen::Vector3f transformed_world_point = affine_transform * current_world_point;
//                float cur_d, cur_w;
//                //if (tsdf_origin.RetriveDataFromWorldCoord_NearestNeighbor(cur_world_coord, &cur_d, &cur_w))
//                if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) && /*cur_w > 0*/ cur_w > options.min_model_weight())
//                {
//                    int current_index = z +
//                            (y + x * options.boundingbox_size()[1]) * options.boundingbox_size()[2];
//                    sample->coeffRef(current_index) = cur_d;
//                    weight->coeffRef(current_index) = cur_w;
//                }
//            }
//    return true;
//}

//bool ExtractOneSampleFromAffineTransformNearestNeighbor(const TSDFHashing &scene_tsdf, const Eigen::Affine3f &affine_transform, const TSDFGridInfo &options, Eigen::SparseVector<float> *sample, Eigen::SparseVector<float> *weight)
//{
//    const Eigen::Vector3f& offset = options.offset();
//    const Eigen::Vector3i& voxel_bb_size = options.boundingbox_size();
//    const int total_voxel_size = voxel_bb_size[0] * voxel_bb_size[1] * voxel_bb_size[2];
//    sample->resize(total_voxel_size);
//    //sample->reserve(total_voxel_size * 0.6);
//    weight->resize(total_voxel_size);
//    //weight->reserve(total_voxel_size * 0.6);
//    for (int x = 0; x < options.boundingbox_size()[0]; ++x)
//        for (int y = 0; y < options.boundingbox_size()[1]; ++y)
//            for (int z = 0; z < options.boundingbox_size()[2]; ++z)
//            {
//                Eigen::Vector3f current_world_point = offset;
//                current_world_point[0] += options.voxel_lengths()[0] *  x;
//                current_world_point[1] += options.voxel_lengths()[1] *  y;
//                current_world_point[2] += options.voxel_lengths()[2] *  z;
//                Eigen::Vector3f transformed_world_point = affine_transform * current_world_point;
//                float cur_d, cur_w;
//                if (scene_tsdf.RetriveDataFromWorldCoord_NearestNeighbor(transformed_world_point, &cur_d, &cur_w) && cur_w > options.min_model_weight())
//                // if(scene_tsdf.RetriveDataFromWorldCoord(transformed_world_point, &cur_d, &cur_w) && /*cur_w > 0*/ cur_w > options.min_model_weight())
//                {
//                    int current_index = z +
//                            (y + x * options.boundingbox_size()[1]) * options.boundingbox_size()[2];
//                    sample->coeffRef(current_index) = cur_d;
//                    weight->coeffRef(current_index) = cur_w;
//                }
//            }
//    return true;
//}


bool ConvertDataVectorToTSDFWithWeight(
        const Eigen::SparseVector<float> &tsdf_data_vec,
        const Eigen::SparseVector<float> &tsdf_weight_vec,
        const TSDFGridInfo &tsdf_info,
        TSDFHashing *tsdf)
{
    //tsdf->CopyHashParametersFrom(options.tsdf_for_parameters);
    tsdf->Init(tsdf_info.voxel_lengths()[0], tsdf_info.offset(), tsdf_info.max_dist_pos(), tsdf_info.max_dist_neg());
    const int size_yz = tsdf_info.boundingbox_size()[1] * tsdf_info.boundingbox_size()[2];
    const Eigen::Vector3i voxel_bounding_box_size = tsdf_info.boundingbox_size();
    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_weight_vec); it; ++it)
    {
        int data_dim_idx = it.index();
        const float weight = it.value();
        if ( weight < tsdf_info.min_model_weight() ) continue;
        const float dist = tsdf_data_vec.coeff(data_dim_idx);
        cv::Vec3i pos;
        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
        pos[0] = data_dim_idx / size_yz;
        tsdf->AddObservation(pos, dist, weight, cv::Vec3b(127, 127, 127));
    }
    tsdf->DisplayInfo();
    return true;
}


bool ConvertDataVectorToTSDFWithWeightAndWorldPos(
        const Eigen::SparseVector<float> &tsdf_data_vec,
        const Eigen::SparseVector<float> &tsdf_weight_vec,
        const TSDFGridInfo &tsdf_info,
        TSDFHashing *tsdf,
        std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f> > *idx_worldpos)
{
    //tsdf->CopyHashParametersFrom(options.tsdf_for_parameters);
    tsdf->Init(tsdf_info.voxel_lengths()[0], tsdf_info.offset(), tsdf_info.max_dist_pos(), tsdf_info.max_dist_neg());
    const int size_yz = tsdf_info.boundingbox_size()[1] * tsdf_info.boundingbox_size()[2];
    const Eigen::Vector3i voxel_bounding_box_size = tsdf_info.boundingbox_size();
    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_weight_vec); it; ++it)
    {
        int data_dim_idx = it.index();
        const float weight = it.value();
        if ( weight < tsdf_info.min_model_weight() ) continue;
        const float dist = tsdf_data_vec.coeff(data_dim_idx);
        cv::Vec3i pos;
        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
        pos[0] = data_dim_idx / size_yz;
        tsdf->AddObservation(pos, dist, weight, cv::Vec3b(255, 255, 255));

        cv::Vec3f worldpos = tsdf->Voxel2World(cv::Vec3f(pos));
        idx_worldpos->insert(std::make_pair(data_dim_idx,
                                            std::make_pair(
                                                utility::CvVectorToEigenVector3(pos),
                                                utility::CvVectorToEigenVector3(worldpos))));
    }
    tsdf->DisplayInfo();
    return true;
}

bool ConvertDataMatrixToTSDFs(const float voxel_length, const Eigen::Vector3f &offset, const float max_dist_pos, const float max_dist_neg, const Eigen::Vector3i &voxel_bounding_box_size, const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat, const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat, std::vector<TSDFHashing::Ptr> *projected_tsdf_models)
{
    std::cout << "Begin convert Data Mat to TSDFs. " << std::endl;
    const int data_dim = data_mat.rows();
    const int sample_num = data_mat.cols();
    (*projected_tsdf_models).clear();
    (*projected_tsdf_models).resize(sample_num);
    for (int i = 0; i < sample_num; ++i)
    {
        (*projected_tsdf_models)[i].reset(new cpu_tsdf::TSDFHashing());
        cpu_tsdf::TSDFHashing* cur_tsdf = (*projected_tsdf_models)[i].get();
        cur_tsdf->Init(voxel_length, offset, max_dist_pos, max_dist_neg);
        const int size_yz = voxel_bounding_box_size[1] * voxel_bounding_box_size[2];
        for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator it_w(weight_mat, i); it_w; ++it_w)
        {
            int data_dim_idx = it_w.row();
            assert(it_w.col() == i);
            if (it_w.value() < 1e-4) continue;
            float data = data_mat.coeff(data_dim_idx, i);
            cv::Vec3i pos;
            pos[2] = data_dim_idx % voxel_bounding_box_size[2];
            pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
            pos[0] = data_dim_idx / size_yz;
            cur_tsdf->AddObservation(pos, data, it_w.value(), cv::Vec3b(255, 255, 255));
        }
        cur_tsdf->DisplayInfo();
    }
    std::cout << "End convert Data Mat to TSDFs. " << std::endl;
    return true;
}

bool ConvertDataMatrixToTSDFsNoWeight(const float voxel_length, const Eigen::Vector3f &offset, const float max_dist_pos, const float max_dist_neg, const Eigen::Vector3i &voxel_bounding_box_size, const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat, std::vector<TSDFHashing::Ptr> *projected_tsdf_models)
{
    std::cout << "Begin convert Data Mat to TSDFs. " << std::endl;
    const int data_dim = data_mat.rows();
    const int sample_num = data_mat.cols();
    (*projected_tsdf_models).clear();
    (*projected_tsdf_models).resize(sample_num);
    for (int i = 0; i < sample_num; ++i)
    {
        (*projected_tsdf_models)[i].reset(new cpu_tsdf::TSDFHashing());
        cpu_tsdf::TSDFHashing* cur_tsdf = (*projected_tsdf_models)[i].get();
        cur_tsdf->Init(voxel_length, offset, max_dist_pos, max_dist_neg);
        const int size_yz = voxel_bounding_box_size[1] * voxel_bounding_box_size[2];
        for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator itr(data_mat, i); itr; ++itr)
        {
            int data_dim_idx = itr.row();
            assert(itr.col() == i);
            float data = itr.value();
            cv::Vec3i pos;
            pos[2] = data_dim_idx % voxel_bounding_box_size[2];
            pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
            pos[0] = data_dim_idx / size_yz;
            cur_tsdf->AddObservation(pos, data, 1, cv::Vec3b(255, 255, 255));
        }
        cur_tsdf->DisplayInfo();
        //char ch;
        //std::cin >> ch;
    }
    std::cout << "End convert Data Mat to TSDFs. " << std::endl;
    return true;
}

bool ConvertDataVectorToTSDFNoWeight(const Eigen::SparseVector<float> &tsdf_data_vec, const PCAOptions &options, TSDFHashing *tsdf)
{
    //tsdf->CopyHashParametersFrom(options.tsdf_for_parameters);
    tsdf->Init(options.voxel_length, options.offset, options.max_dist_pos, options.max_dist_neg);
    const int size_yz = options.boundingbox_size[1] * options.boundingbox_size[2];
    const Eigen::Vector3i voxel_bounding_box_size = options.boundingbox_size;
    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_data_vec); it; ++it)
    {
        int data_dim_idx = it.index();
        float dist = it.value();
        cv::Vec3i pos;
        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
        pos[0] = data_dim_idx / size_yz;
        tsdf->AddObservation(pos, dist, tsdf->getVoxelMaxWeight(), cv::Vec3b(255, 255, 255));
    }
    return true;
}

bool ConvertDataVectorToTSDFNoWeight(const Eigen::SparseVector<float> &tsdf_data_vec, const cpu_tsdf::TSDFGridInfo &options, TSDFHashing *tsdf)
{
    //tsdf->CopyHashParametersFrom(options.tsdf_for_parameters);
    tsdf->Init(options.voxel_lengths()[0], options.offset(), options.max_dist_pos(), options.max_dist_neg());
    const int size_yz = options.boundingbox_size()[1] * options.boundingbox_size()[2];
    const Eigen::Vector3i voxel_bounding_box_size = options.boundingbox_size();
    for (Eigen::SparseVector<float>::InnerIterator it(tsdf_data_vec); it; ++it)
    {
        int data_dim_idx = it.index();
        float dist = it.value();
        cv::Vec3i pos;
        pos[2] = data_dim_idx % voxel_bounding_box_size[2];
        pos[1] = (data_dim_idx / voxel_bounding_box_size[2]) % voxel_bounding_box_size[1];
        pos[0] = data_dim_idx / size_yz;
        tsdf->AddObservation(pos, dist, tsdf->getVoxelMaxWeight(), cv::Vec3b(255, 255, 255));
    }
    return true;
}

bool ConvertDataVectorsToTSDFsNoWeight(const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec, const PCAOptions &options, std::vector<TSDFHashing::Ptr> *tsdfs)
{
    tsdfs->resize(tsdf_data_vec.size());
    for(int i = 0; i < tsdf_data_vec.size(); ++i)
    {
        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
        ConvertDataVectorToTSDFNoWeight(
                    tsdf_data_vec[i],
                    options,
                    ((*tsdfs)[i].get()));
    }
    return true;
}

bool ConvertDataVectorsToTSDFsNoWeight(const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec, const TSDFGridInfo &options, std::vector<TSDFHashing::Ptr> *tsdfs)
{
    tsdfs->resize(tsdf_data_vec.size());
    for(int i = 0; i < tsdf_data_vec.size(); ++i)
    {
        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
        ConvertDataVectorToTSDFNoWeight(
                    tsdf_data_vec[i],
                    options,
                    ((*tsdfs)[i].get()));
    }
    return true;
}

bool ConvertDataVectorsToTSDFsWithWeight(const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec, const std::vector<Eigen::SparseVector<float> > &tsdf_weight_vec, const PCAOptions &options, std::vector<TSDFHashing::Ptr> *tsdfs)
{
    tsdfs->resize(tsdf_data_vec.size());
    for(int i = 0; i < tsdf_data_vec.size(); ++i)
    {
        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
        ConvertDataVectorToTSDFWithWeight(
                    tsdf_data_vec[i],
                    tsdf_weight_vec[i],
                    options,
                    ((*tsdfs)[i].get()));
    }
    return true;
}

bool ConvertDataVectorsToTSDFsWithWeight(const std::vector<Eigen::SparseVector<float> > &tsdf_data_vec, const std::vector<Eigen::SparseVector<float> > &tsdf_weight_vec, const TSDFGridInfo &options, std::vector<TSDFHashing::Ptr> *tsdfs)
{
    tsdfs->resize(tsdf_data_vec.size());
    for(int i = 0; i < tsdf_data_vec.size(); ++i)
    {
        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
        ConvertDataVectorToTSDFWithWeight(
                    tsdf_data_vec[i],
                    tsdf_weight_vec[i],
                    options,
                    ((*tsdfs)[i].get()));
    }
    return true;
}

bool ConvertDataVectorsToTSDFsWithWeight(const std::vector<Eigen::SparseVector<float> > &tsdf_data, Eigen::SparseMatrix<float, Eigen::ColMajor> &tsdf_weights, const PCAOptions &options, std::vector<TSDFHashing::Ptr> *tsdfs)
{
    tsdfs->resize(tsdf_data.size());
    for(int i = 0; i < tsdf_data.size(); ++i)
    {
        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
        ConvertDataVectorToTSDFWithWeight(
                    tsdf_data[i],
                    tsdf_weights.col(i),
                    options,
                    ((*tsdfs)[i].get()));
    }
    return true;
}

bool ConvertDataVectorsToTSDFsWithWeight(const std::vector<Eigen::SparseVector<float> > &tsdf_data, Eigen::SparseMatrix<float, Eigen::ColMajor> &tsdf_weights, const TSDFGridInfo &options, std::vector<TSDFHashing::Ptr> *tsdfs)
{
    tsdfs->resize(tsdf_data.size());
    for(int i = 0; i < tsdf_data.size(); ++i)
    {
        (*tsdfs)[i].reset(new cpu_tsdf::TSDFHashing);
        ConvertDataVectorToTSDFWithWeight(
                    tsdf_data[i],
                    tsdf_weights.col(i),
                    options,
                    ((*tsdfs)[i].get()));
    }
    return true;
}


void MaskImageSidesAsZero(const int side_width, cv::Mat *image)
{
    //    cv::imshow("test0", *image);
    //    cv::waitKey();
    //    cv::destroyWindow("test0");

    cv::Mat band_left = cv::Mat(*image, cv::Rect(0, 0, side_width, image->rows));
    cv::Mat band_right = cv::Mat(*image, cv::Rect(image->cols - side_width, 0, side_width, image->rows));

    band_left.setTo(0);
    band_right.setTo(0);

    //    cv::imshow("test", *image);
    //    cv::waitKey();
    //    cv::destroyWindow("test");
}

void OrientedBoundingBox::Extension(const Eigen::Vector3f &extension_each_side)
{
    bb_offset = bb_offset - bb_orientation * extension_each_side;
    bb_sidelengths += extension_each_side * 2.0;
}

void OrientedBoundingBox::Display() const
{
    using namespace std;
    cout << "bb_offset: \n" << bb_offset << endl;
    cout << "bb_orientation: \n" << bb_orientation << endl;
    cout << "bb_sidelengths: \n" << bb_sidelengths << endl;
    cout << "voxel_lengths: \n" << voxel_lengths << endl;
}

OrientedBoundingBox OrientedBoundingBox::Rescale(const Eigen::Vector3f &scales) const
{
    OrientedBoundingBox new_bb = *this;
    Eigen::Vector3f bb_center = BoxCenter();
    new_bb.bb_sidelengths = (new_bb.bb_sidelengths).cwiseProduct(scales);
    new_bb.bb_offset =  bb_center - new_bb.bb_orientation * new_bb.bb_sidelengths / 2.0;
    return new_bb;
}

OrientedBoundingBox OrientedBoundingBox::RescaleNoBottom(const Eigen::Vector3f &scales) const
{
    OrientedBoundingBox new_bb = *this;
    Eigen::Vector3f bb_bottom_center = BoxBottomCenter();
    new_bb.bb_sidelengths = (new_bb.bb_sidelengths).cwiseProduct(scales);
    new_bb.bb_offset =  bb_bottom_center - new_bb.bb_orientation * Eigen::Vector3f(new_bb.bb_sidelengths[0],
            new_bb.bb_sidelengths[1], 0) / 2.0;
    return new_bb;
}

void GetClusterSampleIdx(
        const std::vector<int> &sample_cluster_idx,
        const std::vector<double> &outlier_gammas,
        const int model_number,
        std::vector<std::vector<int> > *cluster_sample_idx)
{
    const int sample_number = sample_cluster_idx.size();
    (*cluster_sample_idx).resize(model_number);
    for (int i = 0; i < sample_number; ++i)
    {
        (*cluster_sample_idx)[sample_cluster_idx[i]].push_back(i);
    }
    if (!outlier_gammas.empty())
    {
        for (int i = 0; i < model_number; ++i)
        {
            (*cluster_sample_idx)[i].erase(remove_if((*cluster_sample_idx)[i].begin(), (*cluster_sample_idx)[i].end(), [&outlier_gammas](const int& sample_idx){
                return outlier_gammas[sample_idx] > 1e-5;
            }), (*cluster_sample_idx)[i].end());
        }
    }
    return;
}

//OrientedBoundingBox OrientedBoundingBox::JitterSample(float transx, float transy, float angle, const Eigen::Vector3f &scales)
//{
//    Eigen::Vector3f toffset = bb_offset + Eigen::Vector3f(transx, transy, 0);
//    Eigen::Matrix3f torient = bb_orientation * Eigen::AngleAxisf(angle, Eigen::Vector3)
//}


}  // end namespace cpu_tsdf
