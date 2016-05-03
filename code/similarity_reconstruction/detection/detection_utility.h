#pragma once
#include <vector>
#include <string>
#include <cstdint>

#include <Eigen/Eigen>
#include <glog/logging.h>

#include "common/utilities/common_utility.h"
#include "common/utilities/eigen_utility.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "detect_sample.h"

namespace cpu_tsdf {
class TSDFHashing;
}

namespace tsdf_detection {
struct XYAngleLess
{
    bool operator ()(const Eigen::Vector3f& lhs_xyangle, const Eigen::Vector3f& rhs_xyangle) const
    {
        using utility::EqualFloat;
        if (!EqualFloat(lhs_xyangle[0], rhs_xyangle[0]))
        {
            return lhs_xyangle[0] < rhs_xyangle[0];
        }
        else if (!EqualFloat(lhs_xyangle[1], rhs_xyangle[1]))
        {
            return lhs_xyangle[1] < rhs_xyangle[1];
        }
        else if (!EqualFloat(lhs_xyangle[2], rhs_xyangle[2]))
        {
            return lhs_xyangle[2] < rhs_xyangle[2];
        }
        return false;  // equal
    }
};

struct XYAngleEqual
{
    bool operator ()(const Eigen::Vector3f& lhs_xyangle, const Eigen::Vector3f& rhs_xyangle) const
    {
        using utility::EqualFloat;
        return (EqualFloat(lhs_xyangle[0], rhs_xyangle[0]) &&
                EqualFloat(lhs_xyangle[1], rhs_xyangle[1]) &&
                EqualFloat(lhs_xyangle[2], rhs_xyangle[2]));
    }
};

inline void NormalizeAngle(float* angle)
{
    while (*angle >= M_PI) *angle -= (2*M_PI);
    while (*angle < -M_PI) *angle += (2*M_PI);
    CHECK_GE(*angle, -M_PI);
    CHECK_LT(*angle, M_PI);
}

// angle: [-pi, pi)
inline void OBBToXYAngle(const cpu_tsdf::OrientedBoundingBox& obb, Eigen::Vector3f* xyangle)
{
    Eigen::Vector3f center = obb.BoxCenter();
    const Eigen::AngleAxisf aa(obb.bb_orientation);
    float angle = aa.angle();
    Eigen::Vector3f axis = aa.axis();
    //CHECK_LT(fabs(fabs(axis[2]) - 1), 1e-3);
    angle *= axis[2];
    NormalizeAngle(&angle);
    // angle: [-pi, pi)
    (*xyangle)[0] = center[0];
    (*xyangle)[1] = center[1];
    (*xyangle)[2] = angle;
}

class SceneDiscretizeInfo {
public:
    static const uint64_t InvalidObbIndex = (uint64_t)-1;
    SceneDiscretizeInfo(const Eigen::Vector2f& rangex, const Eigen::Vector2f& rangey, const Eigen::Vector3f& deltas);
    // convert from/to OBB positions to indexes, for NMS
    uint64_t DiscretOBBPos2Index(uint32_t discret_x, uint32_t discret_y, uint32_t discret_angle) const;
    void Index2DiscretOBBPos(uint64_t index, uint32_t* discret_pos) const;
    uint64_t OBBPos2Index(const Eigen::Vector3f& pos) const;
    Eigen::Vector3f Index2OBBPos(uint64_t index) const;
    Eigen::Vector3f DiscretOBBPos2OBBPos(uint32_t discret_x, uint32_t discret_y, uint32_t discret_angle) const;
    Eigen::Vector3f DiscretOBBPos2OBBPos(const Eigen::Vector3i& discret_pos) const;
    void OBBPos2DiscretOBBPos(const Eigen::Vector3f& pos, uint32_t* discret_pos) const;
    inline const Eigen::Vector3i& discret_ranges() const { return discret_ranges_; }
    inline const Eigen::Vector2f& rangex() const { return rangex_; }
    inline const Eigen::Vector2f& rangey() const { return rangey_; }
    inline const Eigen::Vector3f& deltas() const { return deltas_; }
    void DisplayDiscretizeInfo() const;
private:
    void ComputeDiscretRanges();
    Eigen::Vector2f rangex_;
    Eigen::Vector2f rangey_;
    Eigen::Vector3f deltas_;  // x, y, angle in radius
    Eigen::Vector3i discret_ranges_;  // x, y, angle
    uint64_t max_index_;
};

struct DetectionParams {
    DetectionParams()
        :save_prefix(), NMS_overlap_threshold(0.01), min_nonempty_voxel_weight(0),
         min_score_to_keep(-.5), detection_total_thread(8), do_NMS(true), do_final_NMS(true),
         unique_hard_negative(true), max_hard_negative_number(1e4), hard_negative_mining_iterations(3),
         positive_jitter_num(20), train_options("-s 0 -t 0 -c 1 -w1 10 "), obb_matching_thresh(0.15),
         minimum_occupied_ratio(0.01), jitter_x_sidelength_percent(1.0/30.0), jitter_y_sidelength_percent(1.0/30.0), jitter_angle(1.0/48.0) {}
    std::string save_prefix;
    std::string train_options;
    float NMS_overlap_threshold;
    float min_nonempty_voxel_weight;
    float min_score_to_keep;
    float obb_matching_thresh;
    float minimum_occupied_ratio;
    int detection_total_thread;
    int hard_negative_mining_iterations;
    int positive_jitter_num;
    uint64_t max_hard_negative_number;
    bool do_NMS;
    bool do_final_NMS;
    bool unique_hard_negative;
    float jitter_x_sidelength_percent;
    float jitter_y_sidelength_percent;
    float jitter_angle;
};

class SampleTemplate {
public:
    SampleTemplate() {}
    SampleTemplate(const tsdf_utility::OrientedBoundingBox& tobb, const Eigen::Vector3i& sample_sz)
        :obb_(tobb), sample_size_(sample_sz) {}
    const tsdf_utility::OrientedBoundingBox& OBB() const  { return obb_; }
    const Eigen::Vector3i& sample_size() const;
    inline float BottomZ() const { return obb_.BottomCenter()[2]; }
    inline float Angle() const { return obb_.AngleRangeTwoPI(); }
    inline Eigen::Vector3f SideLengths() const { return obb_.SideLengths(); }
    bool WriteToFile(const std::string& filename) const;
    bool ReadFromFile(const std::string& filename);
private:
    tsdf_utility::OrientedBoundingBox obb_;
    Eigen::Vector3i sample_size_;
};

tsdf_utility::OrientedBoundingBox OBBFromDiscretPos(const SampleTemplate& template_obb, const SceneDiscretizeInfo &discret_info, const Eigen::Vector3i& discret_pos);

tsdf_utility::OrientedBoundingBox OBBFromPos(const SampleTemplate& template_obb, const Eigen::Vector3f& obb_pos);

Sample SampleFromDiscretPos(const cpu_tsdf::TSDFHashing& scene_tsdf,
                          const SceneDiscretizeInfo& discret_info,
                          const SampleTemplate& template_obb,
                          const DetectionParams& params,
                          const Eigen::Vector3i& discret_pos,
                          const char gt_label, const float init_angle = 0);

Sample SampleFromPos(const cpu_tsdf::TSDFHashing& scene_tsdf,
                          const SampleTemplate& template_obb,
                          const DetectionParams& params,
                          const Eigen::Vector3f& pos,
                          uint64_t obb_index,
                          const char gt_label);

// bool FindIntersectionOBB(const tsdf_utility::OrientedBoundingBox& obb, const std::vector<tsdf_utility::OrientedBoundingBox>& to_match, int* matched_idx);
bool FindMatchingOBB(const tsdf_utility::OrientedBoundingBox& obb, const std::vector<tsdf_utility::OrientedBoundingBox>& to_match, int* matched_idx, float *similarity);

// Find correct detections given a detection obb list and positive samples list
// If multiple detected obbs are matched to the same positive sample, they are all considered correct
// used in hard negative mining
void FindCorrectOBBs(const DetectionParams &params, const std::vector<tsdf_utility::OrientedBoundingBox>& detections, const std::vector<tsdf_utility::OrientedBoundingBox>& pos_samples, std::vector<bool>* correct);

void RemoveCorrectDetections(const DetectionParams& params, const tsdf_detection::SampleCollection& pos_samples, tsdf_detection::SampleCollection* detections);

Sample RandomGenerateSample(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &discret_info, const DetectionParams &params, const SampleTemplate& template_obb);

Sample RandomGenerateSampleDiscret(const cpu_tsdf::TSDFHashing &scene_tsdf,
                            const SceneDiscretizeInfo &discret_info,
                            const DetectionParams &params,
                            const SampleTemplate &template_obb);

void RandomSampleNegatives(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &scene_info, const DetectionParams &params,
                           const SampleTemplate& obb_template, const SampleCollection& pos_samples, const int neg_num, SampleCollection *negs);

void JitterOneSample(const cpu_tsdf::TSDFHashing &scene_tsdf, const DetectionParams &params,
                             const SampleTemplate& obb_template, const Sample& pos_sample, int jitter_num, SampleCollection* jittered_samples);

void JitterSamples(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &discret_info, const DetectionParams &params,
                           const SampleTemplate& obb_template, const SampleCollection& samples, const int jitter_num, SampleCollection* jittered_samples);

void StepSizeFromOBB(const Eigen::Vector3f& side_lengths, float& delta_x, float& delta_y, float& delta_rotation);

void OutputAnnotatedOBB(const std::vector<std::vector<tsdf_utility::OrientedBoundingBox>>& obbs, const std::string& filename);
void InputAnnotatedOBB(const std::string& filename, std::vector<std::vector<tsdf_utility::OrientedBoundingBox>>* obbs);
}
