#pragma once
#include <vector>
#include <string>
#include <map>

#include <boost/filesystem.hpp>
#include <Eigen/Eigen>

#include "tsdf_representation/tsdf_hash.h"
#include "common/utility/eigen_utility.h"
#include "utility/svm_wrapper.h"
#include "utility/utility.h"

namespace cpu_tsdf {

//struct OrientedBoundingBox
//{
//    Eigen::Matrix3f bb_orientation;
//    Eigen::Vector3f bb_sidelengths;
//    Eigen::Vector3f bb_offset;
//    Eigen::Vector3f voxel_lengths;
//    inline Eigen::Vector3f BoxCenter() const
//    {
//        return bb_offset + (bb_orientation.col(0) * bb_sidelengths[0] + bb_orientation.col(1) * bb_sidelengths[1] + bb_orientation.col(2) * bb_sidelengths[2])/2.0;
//    }
//    inline void Clear()
//    {
//        bb_sidelengths.setZero();
//        voxel_lengths.setZero();
//        bb_offset.setZero();
//        bb_orientation.setZero();
//    }

////    inline Eigen::Vector3f BoxCenter2Offset(const Eigen::Vector3f& center)
////    {
////        return center - (bb_orientation.col(0) * bb_sidelengths[0] + bb_orientation.col(1) * bb_sidelengths[1] + bb_orientation.col(2) * bb_sidelengths[2])/2.0;
////    }
//};

class TSDFFeature
{
private:
    std::vector<bool> occupied_;
    std::vector<float> tsdf_vals;
    std::vector<float> tsdf_weights;
    OrientedBoundingBox obb;
    int box_index_;
public:
    TSDFFeature() {}
    TSDFFeature(const OrientedBoundingBox& vobb, const TSDFHashing& tsdf_origin) { ComputeFeature(vobb, tsdf_origin); }
    inline const OrientedBoundingBox& GetOrientedBoundingBox() const { return obb; }
    float occupied_ratio(float mesh_min_weight) const
    {
        int valid_pts_num = std::count_if(tsdf_weights.begin(), tsdf_weights.end(), [mesh_min_weight](const float val){return val > mesh_min_weight;});
        return float(valid_pts_num)/float(occupied_.size());
    }
    void Clear()
    {
        occupied_.clear();
        tsdf_vals.clear();
        tsdf_weights.clear();
        obb.Clear();
        box_index_ = -1;
    }
    inline int box_index() { return box_index_; }
    inline void box_index(int vidx) { box_index_ = vidx; }
    Eigen::Vector3f VoxelLengths() { return obb.voxel_lengths; }
    void GetFeatureVector(const cpu_tsdf::TSDFFeature* template_feature, std::vector<float>* out_feature, float min_weight, bool with_empty_bit = true) const
    {
        out_feature->resize(occupied_.size() * (with_empty_bit? 2:1));
        int tsdf_val_index = -1;
        int cur_out_idx = 0;
        for(int i = 0; i < occupied_.size(); ++i)
        {
            if (this->occupied_[i]) tsdf_val_index++;
            if (this->occupied_[i] && this->tsdf_weights[tsdf_val_index] >= min_weight)
            {
                (*out_feature)[cur_out_idx++] = tsdf_vals[tsdf_val_index];
                if (with_empty_bit) (*out_feature)[cur_out_idx++] = 0;
            }
            else
            {
                (*out_feature)[cur_out_idx++] = 0;
                if (with_empty_bit) (*out_feature)[cur_out_idx++] = 1;
            }
        }
        return;
    }
    void GetFeatureVector_nonDense(const cpu_tsdf::TSDFFeature* template_feature, std::vector<float>* out_feature, float min_weight, bool with_empty_bit = true) const
    {
        const std::vector<bool>* template_occupation = template_feature ? &(template_feature->occupied_ ): &(this->occupied_);
        const std::vector<float>* template_weight = template_feature ? &(template_feature->tsdf_weights ): &(this->tsdf_weights);
        assert(this->occupied_.size() == template_occupation->size());

        out_feature->clear();
        int tsdf_val_index = -1;
        int template_val_index = -1;
        for(int i = 0; i < template_occupation->size(); ++i)
        {
            if (this->occupied_[i]) tsdf_val_index++;
            if ((*template_occupation)[i]) template_val_index++;

            if  (!(*template_occupation)[i] || (*template_weight)[template_val_index] < min_weight) continue;
            if (this->occupied_[i] && this->tsdf_weights[tsdf_val_index] >= min_weight)
            {
                out_feature->push_back(tsdf_vals[tsdf_val_index]);
                if (with_empty_bit) out_feature->push_back(0);
            }
            else
            {
                out_feature->push_back(0);
                if (with_empty_bit) out_feature->push_back(1);
            }
        }
        return;
    }
    inline Eigen::Vector3i VoxelSideLengths() const
    {
//        return Eigen::Vector3i(utility::ceil(
//                                   Eigen::Vector3f(obb.bb_sidelengths.cwiseQuotient(obb.voxel_lengths))
//                                   ).cast<int>());
        return Eigen::Vector3i(utility::round(
                                   Eigen::Vector3f(obb.bb_sidelengths.cwiseQuotient(obb.voxel_lengths))
                                   ).cast<int>()) + Eigen::Vector3i::Ones();
    }
    void ComputeFeature(const OrientedBoundingBox& vobb, const cpu_tsdf::TSDFHashing& tsdf_origin);
    void OutputToQVis(const std::string& output_path)
    {
        Eigen::Vector3i voxel_side_lengths = VoxelSideLengths();
        boost::filesystem::path bfs_path(output_path);
        std::string raw_data_path = bfs_path.stem().string() + ".raw32f";
        FILE* hf = fopen(output_path.c_str(), "w");
        assert(hf);
        fprintf(hf, "ObjectFileName: %s\n", raw_data_path.c_str());
        fprintf(hf, "TaggedFileName: ---\n");
        fprintf(hf, "Resolution: %d %d %d\n", voxel_side_lengths[0], voxel_side_lengths[1], voxel_side_lengths[2]);
        fprintf(hf, "SliceThickness: %d %d %d\n", 1, 1, 1);
        fprintf(hf, "Format: FLOAT\n");
        fprintf(hf, "NbrTags: 0\n");
        fprintf(hf, "ObjectType: TEXTURE_VOLUME_OBJECT\n");
        fprintf(hf, "ObjectModel: RGBA\n");
        fprintf(hf, "GridType: EQUIDISTANT\n");
        fclose(hf);

        float* out_buffer = new float[occupied_.size()];
        std::string raw_data_full_path = (bfs_path.parent_path()/raw_data_path).string();
        hf = fopen(raw_data_full_path.c_str(), "wb");
        assert(hf);
        int tsdf_val_index = -1;
        for (int i = 0; i < occupied_.size(); ++i)
        {
            if (occupied_[i]) tsdf_val_index++;
            float val = occupied_[i] ? tsdf_vals[tsdf_val_index] : 0;
            out_buffer[i] = val;
        }
        fwrite(out_buffer, sizeof(float), occupied_.size(), hf);
        fclose(hf);
    }
    void OutputToText(const std::string& output_path, float min_weight)
    {
        FILE* hf = fopen(output_path.c_str(), "w");
        Eigen::Vector3i voxel_side_lengths = VoxelSideLengths();
        int i = -1;
        int tsdf_val_index = -1;
        for (int ix = 0; ix < voxel_side_lengths[0]; ++ix)
            for(int iy = 0; iy < voxel_side_lengths[1]; ++iy)
                for (int iz = 0; iz < voxel_side_lengths[2]; ++iz)
                {
                    i++;
                    if (occupied_[i])
                    {
                        tsdf_val_index++;
                        if (tsdf_weights[tsdf_val_index] < min_weight)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        continue;
                    }
                    Eigen::Vector3f cur_world_pos = this->obb.bb_offset + obb.bb_orientation.col(0) * (ix * obb.voxel_lengths[0]) +
                            obb.bb_orientation.col(1) * (iy * obb.voxel_lengths[1]) +
                            obb.bb_orientation.col(2) * (iz * obb.voxel_lengths[2]);
                    float value = occupied_[i]? tsdf_vals[tsdf_val_index]: -1.0;
                    uchar r,g,b;
                    utility::mapJet(value, -1, 0.8, r, g, b);
                    fprintf(hf, "%f %f %f %u %u %u\n", cur_world_pos[0], cur_world_pos[1], cur_world_pos[2], r, g, b);
                }
        fclose(hf);
        assert(i + 1 == voxel_side_lengths[0] * voxel_side_lengths[1] * voxel_side_lengths[2]);
    }
    void OutputToText2(const std::string& output_path, float min_weight) {
        std::vector<float> out_feat;
        GetFeatureVector(NULL, &out_feat, min_weight);
        utility::OutputVector(output_path, out_feat);
    }
};


struct ClassificationResult
{
    ClassificationResult():index(-1){}
    ClassificationResult(int vi, int vl, float vs) : index(vi), label(vl), score(vs) {}
    int index;  // denoting the position of the obb
    int label;
    float score;
};


//void ComputeFeature(const Eigen::Matrix3f &bb_orientation, const Eigen::Vector3f &bb_sidelengths, const Eigen::Vector3f &bb_offset,
//                    const Eigen::Vector3f& voxel_lengths, const cpu_tsdf::TSDFHashing& tsdf_origin, cpu_tsdf::TSDFFeature *feature);
//void ComputeFeature(const Eigen::Matrix3f &bb_orientation, const Eigen::Vector3f &bb_sidelengths, const Eigen::Vector3f &bb_offset,
//                    const Eigen::Vector3i& bb_size, const cpu_tsdf::TSDFHashing& tsdf_origin, cpu_tsdf::TSDFFeature *feature);
void GenerateBoundingbox(const cpu_tsdf::OrientedBoundingBox& template_bb,
                                   const Eigen::Vector3f& center_pos_world,
                                   const float theta,
                                   cpu_tsdf::OrientedBoundingBox *bounding_box);

void RandomGenerateBoundingbox(const cpu_tsdf::TSDFHashing &tsdf_origin,
                                           const cpu_tsdf::OrientedBoundingBox& template_bb,
                                           const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_bbs,
                                           cpu_tsdf::OrientedBoundingBox *bounding_boxes, const float similarity_thresh);
//void RandomGenerateBoundingboxes(const cpu_tsdf::TSDFHashing &tsdf_origin, const cpu_tsdf::OrientedBoundingBox& template_bb, const int bb_num,
//                                 const std::vector<OrientedBoundingBox>& avoided_bbs,
//                                 std::vector<OrientedBoundingBox>* bounding_boxes);
//void GenerateAllSlidingBoundingboxes(const cpu_tsdf::TSDFHashing& tsdf_origin, const std::vector<OrientedBoundingBox>* bbs);
bool GenerateOneSample(const cpu_tsdf::TSDFHashing &tsdf_origin,
                               const cpu_tsdf::OrientedBoundingBox& bounding_box,
                               const float template_occupied_ratio, const float mesh_min_weight,
                               cpu_tsdf::TSDFFeature *out_feature);
void GenerateSamples(const cpu_tsdf::TSDFHashing &tsdf_origin,
                               const std::vector<cpu_tsdf::OrientedBoundingBox> &bounding_boxes,
                               std::vector<cpu_tsdf::TSDFFeature> *features, std::vector<cpu_tsdf::TSDFHashing::Ptr> *feature_tsdfs);
void RandomGenerateSamples(const cpu_tsdf::TSDFHashing &tsdf_origin,
                                     const cpu_tsdf::TSDFFeature& template_feature, const int bb_num,
                                     const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_bbs,
                                     std::vector<cpu_tsdf::TSDFFeature> *features,
                                     const std::string* save_path, float mesh_min_weight, float similarity_thresh);
//void RandomGenerateSamples(const cpu_tsdf::TSDFHashing &tsdf_origin,
//                                     const cpu_tsdf::OrientedBoundingBox& template_bb, const int bb_num, const TSDFFeature &template_feature,
//                                     const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_bbs, std::vector<OrientedBoundingBox> *bounding_boxes, std::vector<TSDFFeature> *features, const std::string *save_path, float mesh_min_weight);
//void GenerateAllSlidingBBSamples(const cpu_tsdf::TSDFHashing& tsdf_origin, std::vector<cpu_tsdf::TSDFFeature>* features, std::vector<cpu_tsdf::TSDFHashing::Ptr>* feature_tsdfs);

//void JitteringForOneOBB(const cpu_tsdf::OrientedBoundingBox& template_bb, int bb_num, std::vector<OrientedBoundingBox> *jittered_bbs);
void GenerateSamplesJittering(const cpu_tsdf::TSDFHashing& tsdf_origin,
                              const cpu_tsdf::TSDFFeature& template_feature, int bb_num,
                              std::vector<cpu_tsdf::TSDFFeature> *features,
                              const std::string* save_path, float mesh_min_weight);

/*The first bb is taken as the template_bb*/
void GenerateSamplesJitteringForMultipleBB(const cpu_tsdf::TSDFHashing& tsdf_origin, const std::vector<cpu_tsdf::OrientedBoundingBox>& sample_bbs,
                                           int bb_num_each,
                                           std::vector<cpu_tsdf::TSDFFeature>* features,
                                           const std::string* save_path, float mesh_min_weight);
//////////////////////////////////////////
void SlidingBoxDetection(const cpu_tsdf::TSDFHashing& tsdf_model, const cpu_tsdf::OrientedBoundingBox& template_bb, SVMWrapper& svm,
                         const float delta_x, const float delta_y, const float delta_rotation,
                         float* x_st_ed, float* y_st_ed, std::vector<ClassificationResult>* res, float mesh_min_weight, const std::string *save_path);
void SlidingBoxDetectionReturningHardNegative(const cpu_tsdf::TSDFHashing &tsdf_model,
                                                        const cpu_tsdf::OrientedBoundingBox &template_bb,
                                                        SVMWrapper &svm,
                                                        const float delta_x, const float delta_y, const float delta_rotation,
                                                        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
                                                        const std::vector<std::vector<float>>& pos_feats,
                                                        const std::vector<cpu_tsdf::OrientedBoundingBox>& cached_bbs,
                                                        std::vector<OrientedBoundingBox> *hard_neg_bbs,
                                                        std::vector<std::vector<float>>* hard_negatives,
                                                        float similar_tresh,
                                                        float mesh_min_weight,  const std::string* save_path);

void SlidingBoxDetection_OneRotationAngle2(const cpu_tsdf::TSDFHashing &tsdf_model,
                                                    const cpu_tsdf::OrientedBoundingBox& template_bb,
                                                    const float template_occpy_ratio,
                                                    const SVMWrapper &svm,
                                                    const float delta_x, const float delta_y, const float delta_rotation, const float rotate_angle,
                                                    const Eigen::Vector3f& min_pt, const Eigen::Vector3f& max_pt,
                                                    std::vector<cpu_tsdf::ClassificationResult> * res, const float mesh_min_weight, const std::string *save_path);


void SlidingBoxDetection_OneRotationAngle(const cpu_tsdf::TSDFHashing& tsdf_model,
                                          const cpu_tsdf::OrientedBoundingBox& template_bb, const SVMWrapper& svm,
                                          const float delta_x, const float delta_y, const float delta_rotation, const float rotate_angle,
                                          const Eigen::Vector3f &min_pt, const Eigen::Vector3f &max_pt,
                                          std::vector<ClassificationResult> *res, const float mesh_min_weight, const std::string *save_path);

void SlidingBoxDetection_Parrellel(const TSDFHashing &tsdf_model, const cpu_tsdf::OrientedBoundingBox& template_bb, const SVMWrapper &svm, const float template_occpy_ratio,
                                   const int total_thread,
                                   const float delta_x, const float delta_y, const float delta_rotation,
                                   float* x_st_ed, float* y_st_ed, std::vector<ClassificationResult>* res, float mesh_min_weight, const std::string *save_path);

std::vector<float> NormalizeVector(const std::vector<float> vec);

bool TestIsVectorSimilar(const std::vector<std::vector<float>>& normed_vecs, const std::vector<float>& test_vec, const float thresh);
int TrainObjectDetector(const cpu_tsdf::TSDFHashing &tsdf_model,
                        const cpu_tsdf::OrientedBoundingBox &template_obb,
                        const std::vector<cpu_tsdf::OrientedBoundingBox>& positive_sample_obbs, const std::vector<bool> &positive_sample_for_training, int bb_jitter_num_each, int random_sample_num,
                        SVMWrapper *svm,
                        const std::string* save_path, float similarity_thresh,
                        float mesh_min_weight, const Eigen::Vector3f &detection_deltas, const int total_thread, const Eigen::Vector3f &scene_min_pt, const Eigen::Vector3f &scene_max_pt, const std::string &train_options, const std::string &predict_options,
                        std::vector<OrientedBoundingBox> &pos_bbs, std::vector<OrientedBoundingBox> &neg_bbs);

int AddTSDFFeatureToFeatVec(const std::vector<cpu_tsdf::TSDFFeature> tsdf_feats, const TSDFFeature &template_feat,
                            std::vector<std::vector<float>>* feat_vec, std::vector<cpu_tsdf::OrientedBoundingBox>* obbs_vec, const float mesh_min_weight);

int GetHardNegativeSamples(
        const std::vector<cpu_tsdf::ClassificationResult>& res,
        const cpu_tsdf::TSDFHashing& tsdf_model,
        const cpu_tsdf::TSDFFeature& template_feature,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const float similarity_thresh,
        const float* x_st_ed, const float* y_st_ed, const Eigen::Vector3f& deltas,
        std::vector<cpu_tsdf::TSDFFeature>* hard_neg_feats
        );
/*tsdf_feats has to be: the first gt_pos_num samples are positive and the others are negative*/
int IdentifyHardNegativeSamples(
        const std::vector<cpu_tsdf::OrientedBoundingBox> &all_obbs,
        const int gt_pos_num,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const std::vector<std::vector<float> > &predict_scores,
        const float similarity_thresh,
        std::vector<bool> *ishard_vec);

int GetIndexedSamples(std::vector<std::vector<float>>* tsdf_feats, std::vector<int>* labels, std::vector<std::vector<float>>* predict_scores, std::vector<OrientedBoundingBox> *bbs,
                      const std::vector<bool>& index_vec, const int reserved_number);

inline Eigen::Vector3i ComputeXYRVLen(const float* x_st_ed, const float* y_st_ed, const float delta_x, const float delta_y, const float delta_r)
{
    float x_world_length = x_st_ed[1] - x_st_ed[0];
    float y_world_length = y_st_ed[1] - y_st_ed[0];
    Eigen::Vector3i lengths;
    lengths[0] = floor(x_world_length/delta_x) + 1;
    lengths[1] = floor(y_world_length/delta_y) + 1;
    lengths[2] = floor((2 * M_PI - 1e-5)/delta_r) + 1;

    return lengths;
}

inline void BoxIndexToBoxPos(const float* x_st_ed, const float* y_st_ed,
                      const float delta_x, const float delta_y, const float delta_r,
                      const int box_index, float* pos_x, float* pos_y, float* angle)
{
    Eigen::Vector3i lengths = ComputeXYRVLen(x_st_ed, y_st_ed, delta_x, delta_y, delta_r);

//    lengths[0] = round(x_world_length/delta_x) ;
//    lengths[1] = round(y_world_length/delta_y) ;
//    lengths[2] = round((2 * M_PI )/delta_r) ;

    Eigen::Vector3i index;
    index[2] = box_index % lengths[2];
    int temp = (box_index / lengths[2]);
    index[1] =  temp % lengths[1];
    index[0] =  temp / lengths[1];

    *pos_x = x_st_ed[0] + index[0] * delta_x;
    *pos_y = y_st_ed[0] + index[1] * delta_y;
    *angle = 0 + index[2] * delta_r;
}

inline bool BoxPosToBoxIndex(const float* x_st_ed, const float* y_st_ed,
                      const float delta_x, const float delta_y, const float delta_r,
                      const float pos_x, const float pos_y, float angle,
                      int* box_index)
{
    if (angle >= 2 * M_PI)
    {
        angle -= 2 * M_PI;
    }
    if (angle < -1e-2)
    {
        angle += 2 * M_PI;
    }

    Eigen::Vector3i lengths = ComputeXYRVLen(x_st_ed, y_st_ed, delta_x, delta_y, delta_r);

//    lengths[0] = round(x_world_length/delta_x) ;
//    lengths[1] = round(y_world_length/delta_y) ;
//    lengths[2] = round((2 * M_PI )/delta_r) ;


    Eigen::Vector3i index;
    index[0] = round((pos_x - x_st_ed[0])/delta_x);
    index[1] = round((pos_y - y_st_ed[0])/delta_y);
    index[2] = round((angle - 0) / delta_r);

    if (index[0] < 0 || index[0] >= lengths[0] ||
            index[1] < 0 || index[1] >= lengths[1] ||
            index[2] < 0 || index[2] >= lengths[2] )
    {
        return false;
    }

    *box_index = (index[0] * lengths[1] + index[1]) * lengths[2] + index[2];
    return true;
}

//double OBBOverlapArea(const cpu_tsdf::OrientedBoundingBox& obb1, const cpu_tsdf::OrientedBoundingBox& obb2);

int GetHardNegativeSamplesParallel(
        const std::vector<cpu_tsdf::ClassificationResult>& res,
        const cpu_tsdf::TSDFHashing& tsdf_model,
        const cpu_tsdf::TSDFFeature& template_feature,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const float similarity_thresh,
        const float* x_st_ed, const float* y_st_ed, const Eigen::Vector3f& deltas,
        std::vector<cpu_tsdf::TSDFFeature>* hard_neg_feats,
        const int thread_num
        );
int GetHardNegativeSamplesParallel_thread(
        const std::vector<cpu_tsdf::ClassificationResult>& res,
        const cpu_tsdf::TSDFHashing& tsdf_model,
        const cpu_tsdf::TSDFFeature& template_feature,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const float similarity_thresh,
        const float* x_st_ed, const float* y_st_ed, const Eigen::Vector3f& deltas,
        std::vector<cpu_tsdf::TSDFFeature>* hard_neg_feats,
        int start,
        int end
        );

bool TestOBBSimilarity(
        const float similarity_thresh,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const cpu_tsdf::OrientedBoundingBox& cur_bb, int& cur_intersect_pos_sample, float& cur_intersect_area);


}

double OBBOverlapArea(const cpu_tsdf::OrientedBoundingBox& obb1, const cpu_tsdf::OrientedBoundingBox& obb2);
bool OBBsLargeOverlapArea(const cpu_tsdf::OrientedBoundingBox& obb1, const std::vector<cpu_tsdf::OrientedBoundingBox>& obbs, int *intersected_box = NULL, float *intersect_area = NULL, const float thresh = 0.2);
