/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "detect_sample.h"
#include "detection_utility.h"
#include "tsdf_hash_utilities/svm_wrapper.h"

namespace cpu_tsdf {
class OrientedBoundingBox;
class TSDFHashing;
}

namespace tsdf_detection {

class Detector {
public:
    Detector():svm_(), sample_template_() {}
    Detector(const std::string& svm_modelpath);
    void Predict(const DetectionParams& params, Sample *sample, float *score, char *label) const;
    void Predict(const DetectionParams &params, SampleCollection* samples) const;
    void Train(const DetectionParams& params, const SampleCollection& pos, const SampleCollection& neg);
    const Eigen::Vector3i& sample_size() const { return sample_template_.sample_size(); }
    bool SaveToFile(const std::string &fname) const;
    bool ReadFromFile(const std::string &fname);
    inline const SampleTemplate& sample_template() const { return sample_template_; }
    void sample_template(const SampleTemplate& obb) { sample_template_ = obb; }
private:
    SVMWrapper svm_;  // cannot be copied when not empty
    SampleTemplate sample_template_;
};

void ExtendOBBs(std::vector<cpu_tsdf::OrientedBoundingBox>* obbs, const Eigen::Vector3f& meters);

void ExtendOBBsByPercentOfMinSide(std::vector<cpu_tsdf::OrientedBoundingBox> *obbs, float percent);

void ExtendOBBByPercentOfMinSide(cpu_tsdf::OrientedBoundingBox& obb, float percent);

void ExtendOBBNoBottom(cpu_tsdf::OrientedBoundingBox& obb, const Eigen::Vector3f& meters);

void ExtendOBB(cpu_tsdf::OrientedBoundingBox& obb, const Eigen::Vector3f& meters);

void ExtendOBBsNoBottom(std::vector<cpu_tsdf::OrientedBoundingBox> *obbs, const Eigen::Vector3f& meters);





void NMS_OneAngle(const Eigen::Vector3f& obb_sidelengths, const DetectionParams &params, tsdf_detection::SampleCollection* samples);

void NMS(const DetectionParams& params, SampleCollection *samples) ;

void Detect(const cpu_tsdf::TSDFHashing& scene_tsdf, const Detector& model,
        const SceneDiscretizeInfo &scene_info,
        const DetectionParams& params, const float init_angle,
        SampleCollection* samples
        );

void Detect_OneAngle(const cpu_tsdf::TSDFHashing& scene_tsdf, const Detector& model,
        const SceneDiscretizeInfo &scene_info,
        const DetectionParams& params,
        const int discret_angle, const float init_angle,
        SampleCollection* samples);

// Add new hard negatives to SampleCollection negs
// if no hard negatives, return false
bool MineNegatives(const cpu_tsdf::TSDFHashing& scene_tsdf, const Detector& model, const SceneDiscretizeInfo &scene_info, const DetectionParams& params, const SampleCollection& pos_samples, SampleCollection* negs);

void MineTrainIteration(const cpu_tsdf::TSDFHashing& scene_tsdf, const SceneDiscretizeInfo &scene_info, const DetectionParams& params, const SampleCollection& pos_samples, Detector* model, SampleCollection* negs);

void TrainDetector(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &scene_info, DetectionParams &params, const SampleCollection &pos_samples, Detector *model);
void RandomSampleNegatives(const cpu_tsdf::TSDFHashing& scene_tsdf, const SceneDiscretizeInfo &scene_info, const DetectionParams& params, const SampleCollection &pos_samples, const int neg_num, SampleCollection* negs);

void AdjustSamplesPos(const cpu_tsdf::TSDFHashing &scene_tsdf,
                      const Detector& model,
                      const SceneDiscretizeInfo &discret_info,
                      const DetectionParams &params,
                      SampleCollection* samples);

void AdjustOneSamplePos(const cpu_tsdf::TSDFHashing &scene_tsdf,
                      const Detector& model,
                      const SceneDiscretizeInfo &discret_info,
                      const DetectionParams &params,
                      Sample* sample);

void AdjustSamplesScales(const cpu_tsdf::TSDFHashing &scene_tsdf,
                         const Detector& model,
                         const SceneDiscretizeInfo &discret_info,
                         const DetectionParams &params,
                         SampleCollection* samples,
                         const Eigen::Vector3f& scale_lowerbd = Eigen::Vector3f(0.8, 0.8, 0.8),
                         const Eigen::Vector3f& scale_upperbd = Eigen::Vector3f(1.2, 1.2, 1.2));

void AdjustOneSampleScales(const cpu_tsdf::TSDFHashing &scene_tsdf,
                           const Detector& model,
                           const SceneDiscretizeInfo &discret_info,
                           const DetectionParams &params,
                           Sample* sample,
                           const Eigen::Vector3f& scale_lowerbd = Eigen::Vector3f(0.8, 0.8, 0.8),
                           const Eigen::Vector3f& scale_upperbd = Eigen::Vector3f(1.2, 1.2, 1.2));
}  // namespace tsdf_detection
