/*
 * Slicing part of the TSDF
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
/**
  * 1. SliceTSDF*(): Getting part of the TSDF satisfying some condition
  * 2. GetTSDFSemanticPart*(): Computing the bounding box of the TSDF containing specific semantic label
  *    GetTSDFSemanticMajorPart*(): Computing the minimum bounding box containing at least (threshold) TSDF points with a semantic label.
  *    e.g. compute the minimum bounding box containing 97% of TSDF grid points with the label "house 1" can filter out some noise.
  */

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include <string>

#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"

namespace cpu_tsdf
{
class TSDFHashing;
}

namespace cpu_tsdf
{

bool ExtractSamplesFromOBBs(
        const TSDFHashing &scene_tsdf,
        const std::vector<tsdf_utility::OrientedBoundingBox> &obbs,
        const Eigen::Vector3i& sample_size, const float min_nonempty_weight,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights);
bool ExtractSampleFromOBB(const TSDFHashing &scene_tsdf,
        const tsdf_utility::OrientedBoundingBox& obb,
        const Eigen::Vector3i& sample_size, const float min_nonempty_weight,
        Eigen::SparseVector<float> *sample,
        Eigen::SparseVector<float> *weight);


bool ExtractSamplesFromAffineTransform(
        const TSDFHashing &scene_tsdf,
        const std::vector<Eigen::Affine3f> &affine_transforms,
        const TSDFGridInfo &options,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights);

bool ExtractOneSampleFromAffineTransform(const TSDFHashing &scene_tsdf,
                                         const Eigen::Affine3f &affine_transform,
                                         const TSDFGridInfo &options,
                                         Eigen::SparseVector<float> *sample,
                                         Eigen::SparseVector<float> *weight);

bool MergeTSDFNearestNeighbor(const TSDFHashing &src, cpu_tsdf::TSDFHashing *target);

bool MergeTSDFs(const std::vector<TSDFHashing::Ptr>& srcs, cpu_tsdf::TSDFHashing *target);

bool MergeTSDF(
        const TSDFHashing& tsdf_origin,
        cpu_tsdf::TSDFHashing *target);

/**
 * Getting part of TSDF satisfying some condition.
 * @param tsdf_volume: the original TSDF
 * @param pred: a functor taking data from a voxel and returns whether this voxel satisfies the condition
 * @param sliced_tsdf: the output TSDF with only the voxels satisfying the condition
 */
template<typename Pred>
bool SliceTSDF(const TSDFHashing* tsdf_volume, const Pred& pred, TSDFHashing* sliced_tsdf)
{
    using namespace std;
    const float original_voxel_length = tsdf_volume->voxel_length();
    float max_dist_pos, max_dist_neg;
    tsdf_volume->getDepthTruncationLimits(max_dist_pos, max_dist_neg);
    Eigen::Vector3f voxel_world_offset = tsdf_volume->getVoxelOriginInWorldCoord();
    sliced_tsdf->Init(original_voxel_length, voxel_world_offset, max_dist_pos, max_dist_neg);
    cout << "Begin filtering TSDF. " << endl;
    cout << "voxel length: \n" << original_voxel_length << endl;
    cout << "aabbmin: \n" << voxel_world_offset << endl;
    cout << "max, min trunc dist: " << max_dist_pos << "; " << max_dist_neg << endl;
    for (TSDFHashing::const_iterator citr = tsdf_volume->begin(); citr != tsdf_volume->end(); ++citr)
    {
        cv::Vec3i cur_voxel_coord = citr.VoxelCoord();
        float d, w;
        cv::Vec3b color;
        int vsemantic_label;
        VoxelData::VoxelState st;
        citr->RetriveData(&d, &w, &color, &st, &vsemantic_label);
        if (w > 0 && pred(tsdf_volume, cur_voxel_coord, d, w, color, vsemantic_label))
        {
            sliced_tsdf->AddObservation(cur_voxel_coord, d, w, color, vsemantic_label);
        }  // end if
    }  // end for
    //sliced_tsdf->DisplayInfo();
    cout << "End filtering TSDF. " << endl;
    return !sliced_tsdf->Empty();
}

/**
 * Used with SliceTSDF to get voxels inside an oriented bounding box.
*/
struct PointInOrientedBox
{
    // box_rotation: each column is the direction of a side of the box.
    // box_offset: the offset of the box to the world coordinate
    // box_lengths: three side lengths of the box
    PointInOrientedBox(const cpu_tsdf::OrientedBoundingBox& obb)
        : box_rotation_trans_(obb.bb_orientation.transpose()),
          box_offset_(obb.bb_offset),
          box_lengths_(obb.bb_sidelengths) {}
    PointInOrientedBox(const Eigen::Matrix3f& box_rotation, const Eigen::Vector3f& box_offset,
                       const Eigen::Vector3f& box_lengths)
        : box_rotation_trans_(box_rotation.transpose()),
          box_offset_(box_offset),
          box_lengths_(box_lengths) {}
    inline bool operator () (const TSDFHashing* tsdf_volume, const cv::Vec3i& voxel_coord, float d, float w, const cv::Vec3b& color, int semantic_label) const
    {
        Eigen::Vector3f world_coord =
                tsdf_volume->Voxel2World(utility::CvVectorToEigenVector3(cv::Vec3f(voxel_coord)));
        Eigen::Vector3f obb_coord = box_rotation_trans_ * (world_coord - box_offset_);
        if (! (obb_coord[0] >= 0 && obb_coord[0] <= box_lengths_[0] &&
               obb_coord[1] >= 0 && obb_coord[1] <= box_lengths_[1] &&
               obb_coord[2] >= 0 && obb_coord[2] <= box_lengths_[2]))
        {
            return false;
        }
        return true;
    }
private:
    const Eigen::Matrix3f box_rotation_trans_;
    const Eigen::Vector3f box_offset_;
    const Eigen::Vector3f box_lengths_;
};

/**
 * @brief Used with Sliced TSDF to get voxels with specific semantic_label
 */
struct PointWithSemanticLabel
{
    PointWithSemanticLabel(int vsemantic_label) : semantic_label_(vsemantic_label) {}
    inline bool operator () (const TSDFHashing* tsdf_volume, const cv::Vec3i& voxel_coord, float d, float w, const cv::Vec3b& color, int semantic_label) const
    {
        return semantic_label_ == semantic_label;
    }

private:
    const int semantic_label_;
};

/**
 * @brief Used with Sliced TSDF to get all voxels with any semantic_label
 */
struct AllPointWithSemanticLabel
{
    AllPointWithSemanticLabel() {}
    inline bool operator () (const TSDFHashing* tsdf_volume, const cv::Vec3i& voxel_coord, float d, float w, const cv::Vec3b& color, int semantic_label) const
    {
        return semantic_label > -1;
    }
};

bool CleanTSDFPart(TSDFHashing* tsdf_volume, const cpu_tsdf::OrientedBoundingBox& obb);
bool CleanTSDFPart(TSDFHashing* tsdf_volume, const Eigen::Affine3f& obb);

bool ScaleTSDFPart(TSDFHashing* tsdf_volume, const cpu_tsdf::OrientedBoundingBox& obb, const float scale);
bool ScaleTSDFPart(TSDFHashing* tsdf_volume, const Eigen::Affine3f& affine, const float scale);

bool ScaleTSDFParts(TSDFHashing* tsdf_volume, const std::vector<cpu_tsdf::OrientedBoundingBox>& obb, const float scale);
bool ScaleTSDFParts(TSDFHashing* tsdf_volume, const std::vector<Eigen::Affine3f>& obb, const float scale);



/**
 * @brief Getting the TSDF part within an oriented bounding box.
 * Should be the same as SliceTSDF<PointInOrientedBox>
 */
bool SliceTSDFWithBoundingbox(const cpu_tsdf::TSDFHashing* tsdf_volume,
                              const Eigen::Matrix3f& voxel_world_rotation,
                              const Eigen::Vector3f& offset,
                              const Eigen::Vector3f& world_side_lengths,
                              const float voxel_length,
                              TSDFHashing* sliced_tsdf);

bool SliceTSDFWithBoundingbox_NearestNeighbor(const cpu_tsdf::TSDFHashing *tsdf_volume,
                         const Eigen::Matrix3f &voxel_world_rotation,
                         const Eigen::Vector3f &offset,
                         const Eigen::Vector3f &world_side_lengths,
                         const float voxel_length,
                         cpu_tsdf::TSDFHashing *sliced_tsdf);


/**
 * @brief GetTSDFSemanticPartAxisAlignedBoundingbox Get the axis aligned boundingbox for a semantic label (e.g. a house)
 * @param tsdf_volume Input TSDF
 * @param semantic_label
 * @param neighborhood how much the boundingbox should be extended (neighborhood * voxel_length)
 * @param pmin_pt output: the min point of the bounding box
 * @param lengths output: the lengths of the bounding box
 * @return if no such semantic label exist in the TSDF, return false
 */
bool GetTSDFSemanticPartAxisAlignedBoundingbox(const TSDFHashing* tsdf_volume, int semantic_label, int neighborhood, Eigen::Vector3f *pmin_pt, Eigen::Vector3f *lengths);

/**
 * @brief GetTSDFSemanticPartOrientedBoundingbox Get an oriented boundingbox for a semantic label
 * @param tsdf_volume
 * @param semantic_label
 * @param neighborhood
 * @param orientation Input, 3*3 matrix, each column is the axis-direction of the boundingbox
 * @param pmin_pt Output, the minimum point of the bounding box in world coordinate
 * @param lengths Output, the lengths of the bounding box
 * @return
 */
bool GetTSDFSemanticPartOrientedBoundingbox(const TSDFHashing* tsdf_volume, int semantic_label, int neighborhood,
                                            const Eigen::Matrix3f &orientation, Eigen::Vector3f *pmin_pt, Eigen::Vector3f *lengths);
/**
 * @brief GetTSDFSemanticMajorPartOrientedBoundingbox2D
 * Get an oriented boundingbox for a semantic label.
 * The boundingbox has the height containing all the semantic label,
 * and has the minimum ground rectangle area containing (thresh_percent) of the semantic label.
 * @param tsdf_volume
 * @param semantic_label
 * @param neighborhood
 * @param thresh_percent the percentage of semantic labeled points the boundingbox should contain
 * @param orientation
 * @param pmin_pt
 * @param lengths
 * @return
 */
bool GetTSDFSemanticMajorPartOrientedBoundingbox2D(const TSDFHashing* tsdf_volume, int semantic_label, int neighborhood, float thresh_percent,
                                                   const Eigen::Matrix3f &orientation, Eigen::Vector3f *pmin_pt, Eigen::Vector3f *lengths);

/**
 * @brief ProjectTSDFTo2DHist Compute a 2D histogram
 * @param tsdf_volume
 * @param semantic_label
 * @param ground_orientation
 * @param pmin_pt
 * @param lengths
 * @param voxel_length
 * @param hist_2d_xy
 * @param total_cnt
 * @return
 */
bool ProjectTSDFTo2DHist(const cpu_tsdf::TSDFHashing* tsdf_volume, int semantic_label, const Eigen::Vector3f *ground_orientation,
                         const Eigen::Vector3f& pmin_pt, const float *lengths,
                         const float voxel_length, std::vector<std::vector<float>>* hist_2d_xy, int* total_cnt);

bool ProjectTSDFTo2DHist(const cpu_tsdf::TSDFHashing* tsdf_volume, int semantic_label, const Eigen::Vector3f *ground_orientation,
                         const Eigen::Vector3f& pmin_pt, const float *lengths,
                         const float voxel_length, cv::Mat_<float>* im_hist_2d_xy, int* total_cnt);

bool ComputeOrientedBoundingbox(const cpu_tsdf::TSDFHashing* tsdf_origin, const Eigen::Matrix3f& orientation, Eigen::Vector3f* offset, Eigen::Vector3f* sidelengths);

}
