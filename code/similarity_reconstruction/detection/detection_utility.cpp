/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "detection_utility.h"
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include "tsdf_representation/tsdf_hash.h"
#include "tsdf_hash_utilities/utility.h"
#include "common/utilities/common_utility.h"
#include "common/utilities/eigen_utility.h"
#include "obb_intersection.h"
#include "detect_sample.h"

//bool tsdf_detection::RandomGenerateOneBoundingbox(
//        const Eigen::Vector3f& min_pt,
//        const Eigen::Vector3f& max_pt,
//        const float bottom_z,
//        const Eigen::Vector3f& side_lengths,
//        const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_obbs,
//        cpu_tsdf::OrientedBoundingBox *bounding_box)
//{
//    using namespace std;
//    Eigen::Vector3f scene_sidelengths = max_pt - min_pt;
//    int ctry = 0;
//    for (; ctry < 1000; ++ctry)
//    {
//        // x1, y1: center
//        float x1 = ( (float(rand())/RAND_MAX) * scene_sidelengths[0]) + min_pt[0];
//        float y1 = ( (float(rand())/RAND_MAX) * scene_sidelengths[1]) + min_pt[1];
//        static const float angle_resolution = 1.0;
//        float theta = int( (float(rand())/RAND_MAX) * (360.0/angle_resolution) ) * angle_resolution / 180.0 * M_PI;
//        NormalizeAngle(&theta);

//        cout << "maxpt: " << max_pt << endl;
//        cout << "minpt: " << min_pt << endl;
//        cout << "x1, y1: " << x1 << " " << y1 << endl;
//        cout << "angle: " << theta << endl;

//        Eigen::Vector3f xyangle(x1, y1, theta);
//        cpu_tsdf::OrientedBoundingBox cur_bb;
//        XYAngleToOBB(xyangle, bottom_z + side_lengths[2], side_lengths, &cur_bb);
//        if(tsdf_test::TestOBBsIntersection(cur_bb, avoided_obbs)) continue;

//        *bounding_box = (cur_bb);
//        break;
//    }
//    if (ctry == 1000)
//    {
//        LOG(WARNING) << "tried 1000 times to generate random boxes but failed";
//        return false;
//    }
//    return true;
//}




namespace tsdf_detection {
SceneDiscretizeInfo::SceneDiscretizeInfo(const Eigen::Vector2f &rangex, const Eigen::Vector2f &rangey, const Eigen::Vector3f &deltas)
    :rangex_(rangex), rangey_(rangey), deltas_(deltas) {
    ComputeDiscretRanges();
}

uint64_t SceneDiscretizeInfo::DiscretOBBPos2Index(uint32_t discret_x, uint32_t discret_y, uint32_t discret_angle) const {
    return ((uint64_t)discret_x * discret_ranges_[1] + discret_y) * discret_ranges_[2] + discret_angle;
}

void SceneDiscretizeInfo::Index2DiscretOBBPos(uint64_t index, uint32_t *discret_pos) const {
    CHECK_LT(index, max_index_);
    discret_pos[2] = index % discret_ranges_[2];
    uint32_t temp = index / discret_ranges_[2];
    discret_pos[1] = temp % discret_ranges_[1];
    discret_pos[0] = temp / discret_ranges_[1];
}

uint64_t SceneDiscretizeInfo::OBBPos2Index(const Eigen::Vector3f &pos) const {
    uint32_t discret_pos[3];
    OBBPos2DiscretOBBPos(pos, discret_pos);
    return DiscretOBBPos2Index(discret_pos[0], discret_pos[1], discret_pos[2]);
}

Eigen::Vector3f SceneDiscretizeInfo::Index2OBBPos(uint64_t index) const {
    uint32_t discret_pos[3];
    Index2DiscretOBBPos(index, discret_pos);
    // return Eigen::Vector3f(rangex_[0] + discret_pos[0] * deltas_[0], rangey_[0] + discret_pos[1] * deltas_[1], 0 + discret_pos[2] * deltas_[2]);
    return DiscretOBBPos2OBBPos(discret_pos[0], discret_pos[1], discret_pos[2]);
}

Eigen::Vector3f SceneDiscretizeInfo::DiscretOBBPos2OBBPos(uint32_t discret_x, uint32_t discret_y, uint32_t discret_angle) const
{
    return Eigen::Vector3f(rangex_[0] + discret_x * deltas_[0], rangey_[0] + discret_y * deltas_[1], 0 + discret_angle * deltas_[2]);
}

Eigen::Vector3f SceneDiscretizeInfo::DiscretOBBPos2OBBPos(const Eigen::Vector3i &discret_pos) const {
   return DiscretOBBPos2OBBPos(discret_pos[0], discret_pos[1], discret_pos[2]);
}

void SceneDiscretizeInfo::OBBPos2DiscretOBBPos(const Eigen::Vector3f &pos, uint32_t *discret_pos) const
{
    float angle = cpu_tsdf::WrapAngleRange2PI(pos[2]);
    discret_pos[0] = (uint32_t) round((pos[0] - rangex_[0])/deltas_[0]);
    discret_pos[1] = (uint32_t) round((pos[1] - rangey_[0])/deltas_[1]);
    discret_pos[2] = (uint32_t) round((angle - 0) / deltas_[2]);
    CHECK_LT(discret_pos[0], discret_ranges_[0]);
    CHECK_LT(discret_pos[1], discret_ranges_[1]);
    CHECK_LT(discret_pos[2], discret_ranges_[2]);
}

void SceneDiscretizeInfo::DisplayDiscretizeInfo() const
{
    std::cout << "Discretize Info: " << std::endl;
    std::cout << "discret range (x, y, angle): \n" << discret_ranges_ << std::endl;
}

void SceneDiscretizeInfo::ComputeDiscretRanges() {
    float x_world_length = rangex_[1] - rangex_[0];
    float y_world_length = rangey_[1] - rangey_[0];
    discret_ranges_[0] = ceil(x_world_length/deltas_[0]);
    discret_ranges_[1] = ceil(y_world_length/deltas_[1]);
    discret_ranges_[2] = ceil((2 * M_PI - 1e-5)/deltas_[2]);
    max_index_ = (uint64_t)discret_ranges_[0] * discret_ranges_[1] * discret_ranges_[2];
}

bool FindMatchingOBB(const tsdf_utility::OrientedBoundingBox &obb, const std::vector<tsdf_utility::OrientedBoundingBox> &to_match, int *matched_idx, float *similarity)
{
   *matched_idx = -1;
    *similarity = 0;
    for (int i = 0; i < to_match.size(); ++i) {
        float cur_similarity = 0;
        if (tsdf_test::OBBSimilarity(obb, to_match[i], &cur_similarity) && cur_similarity > *similarity) {
           *matched_idx = i;
            *similarity = cur_similarity;
        }
    }
    return *matched_idx >= 0;
}

void FindCorrectOBBs(const DetectionParams& params, const std::vector<tsdf_utility::OrientedBoundingBox> &detections, const std::vector<tsdf_utility::OrientedBoundingBox> &pos_samples, std::vector<bool> *correct)
{
    correct->assign(detections.size(), false);
    for (int i = 0; i < detections.size(); ++i) {
        int match_idx = -1;
        float similarity = 0;
        if (FindMatchingOBB(detections[i], pos_samples, &match_idx, &similarity) && similarity > params.obb_matching_thresh) {
            (*correct)[i] = true;
        }
    }
}

//void RemoveCorrectOBBs(const std::vector<tsdf_utility::OrientedBoundingBox> &pos_samples, std::vector<tsdf_utility::OrientedBoundingBox> *detections)
//{
//    std::vector<bool> to_remove;
//    FindCorrectOBBs((*detections), pos_samples, &to_remove);
//    cpu_tsdf::EraseElementsAccordingToFlags(to_remove, detections);
//}

void RemoveCorrectDetections(const DetectionParams& params, const SampleCollection &pos_samples, SampleCollection *detections)
{
    std::vector<tsdf_utility::OrientedBoundingBox> pos_obbs;
    pos_samples.GetOBBCollection(&pos_obbs);
    std::vector<tsdf_utility::OrientedBoundingBox> detected_obbs;
    detections->GetOBBCollection(&detected_obbs);
    std::vector<bool> to_remove;
    FindCorrectOBBs(params, detected_obbs, pos_obbs, &to_remove);
    cpu_tsdf::EraseElementsAccordingToFlags(to_remove, &(detections->samples));
}

const Eigen::Vector3i &SampleTemplate::sample_size() const
{
    return sample_size_;
}

bool SampleTemplate::WriteToFile(const std::string &filename) const
{
    std::ofstream ofs(filename);
    ofs << this->obb_ << std::endl;
    utility::operator << (ofs, sample_size_) << std::endl;
    return bool(ofs);
}

bool SampleTemplate::ReadFromFile(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs) {
        LOG(WARNING) << "Reading file: " << filename << "failed. ";
    }
    ifs >> obb_;
    Eigen::MatrixXf temp;
    utility::operator >> (ifs, temp);
    sample_size_ = temp.cast<int>().block(0, 0, 3, 1);
    return bool(ifs);
}

tsdf_utility::OrientedBoundingBox OBBFromDiscretPos(const SampleTemplate &template_obb, const SceneDiscretizeInfo& discret_info, const Eigen::Vector3i &discret_pos) {
    Eigen::Vector3f obb_pos = discret_info.DiscretOBBPos2OBBPos(discret_pos);
    return OBBFromPos(template_obb, obb_pos);
}

tsdf_utility::OrientedBoundingBox OBBFromPos(const SampleTemplate &template_obb, const Eigen::Vector3f &obb_pos) {
    return tsdf_utility::OrientedBoundingBox(template_obb.SideLengths(), Eigen::Vector3f(obb_pos[0], obb_pos[1], template_obb.BottomZ()), obb_pos[2]);
}

Sample SampleFromDiscretPos(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &discret_info,
                            const SampleTemplate &template_obb,
                            const DetectionParams& params,
                            const Eigen::Vector3i &discret_pos,
                            const char gt_label,
                            const float init_angle /*= 0*/) {
    Eigen::Vector3f pos = discret_info.DiscretOBBPos2OBBPos(discret_pos);
    return SampleFromPos(scene_tsdf, template_obb, params, Eigen::Vector3f(pos[0], pos[1], pos[2]+init_angle), discret_info.DiscretOBBPos2Index(discret_pos[0], discret_pos[1], discret_pos[2]), gt_label);
}

Sample SampleFromPos(const cpu_tsdf::TSDFHashing &scene_tsdf, const SampleTemplate &template_obb, const DetectionParams &params, const Eigen::Vector3f &pos, uint64_t obb_index, const char gt_label) {
    tsdf_utility::OrientedBoundingBox cur_obb = OBBFromPos(template_obb, pos);
    return Sample(cur_obb, template_obb.sample_size(), scene_tsdf, obb_index, params.min_nonempty_voxel_weight, gt_label);
}

Sample RandomGenerateSampleDiscret(const cpu_tsdf::TSDFHashing &scene_tsdf,
                            const SceneDiscretizeInfo &discret_info,
                            const DetectionParams &params,
                            const SampleTemplate &template_obb) {
    Eigen::Vector3i discret_ranges = discret_info.discret_ranges();
    Eigen::Vector3i discret_pos;
    discret_pos[0] = (float(rand())/RAND_MAX) * discret_ranges[0];
    discret_pos[1] = (float(rand())/RAND_MAX) * discret_ranges[1];
    discret_pos[2] = (float(rand())/RAND_MAX) * discret_ranges[2];
    discret_pos[0] = std::min(discret_pos[0], discret_ranges[0] - 1);
    discret_pos[1] = std::min(discret_pos[1], discret_ranges[1] - 1);
    discret_pos[2] = std::min(discret_pos[2], discret_ranges[2] - 1);
    return SampleFromDiscretPos(scene_tsdf, discret_info, template_obb, params, discret_pos, -1);
}

Sample RandomGenerateSample(const cpu_tsdf::TSDFHashing &scene_tsdf,
                            const SceneDiscretizeInfo &discret_info,
                            const DetectionParams &params,
                            const SampleTemplate &template_obb) {
    Eigen::Vector3f obb_pos;
    const Eigen::Vector2f& rangex = discret_info.rangex();
    const Eigen::Vector2f& rangey = discret_info.rangey();
    obb_pos[0] = (float(rand())/RAND_MAX) * (rangex[1] - rangex[0]) + rangex[0];
    obb_pos[1] = (float(rand())/RAND_MAX) * (rangey[1] - rangey[0]) + rangey[0];
    obb_pos[2] = (float(rand())/RAND_MAX) * (M_PI * 2) + 0;
    return SampleFromPos(scene_tsdf, template_obb, params, obb_pos, (uint64_t)-1, -1);
}

void RandomSampleNegatives(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &scene_info, const DetectionParams &params, const SampleTemplate &obb_template, const SampleCollection &pos_samples, const int neg_num, SampleCollection *negs) {
    std::vector<tsdf_utility::OrientedBoundingBox> pos_obbs;
    pos_samples.GetOBBCollection(&pos_obbs);
    Eigen::Vector2f rangex = scene_info.rangex();
    Eigen::Vector2f rangey = scene_info.rangey();
    Eigen::Vector3f sidelengths = obb_template.SideLengths();
    int gened_bb = 0;
    while (gened_bb < neg_num) {
        Sample cur_sample = RandomGenerateSample(scene_tsdf, scene_info, params, obb_template);
        //Eigen::Vector3f cur_pos = cur_sample.OBB().BottomCenter();
        //// heuristic
        //if (rangex[1] - cur_pos[0] < 0.5 * sidelengths[0] ||
        //    rangey[1] - cur_pos[1] < 0.5 * sidelengths[1] ||
        //    cur_pos[0] - rangex[0] < 0.5 * sidelengths[0] ||
        //    cur_pos[1] - rangey[0] < 0.5 * sidelengths[1])
        //    continue;
        if (tsdf_test::TestOBBsIntersection3D(cur_sample.OBB(), pos_obbs)) continue;
        // int match_idx = -1;
        // float similarity = 0;
        // if (FindMatchingOBB(cur_sample.OBB(), pos_obbs, &match_idx, &similarity) && similarity > params.obb_matching_thresh) continue;  // too close to positive sample OBB
        if (cur_sample.OccupiedRatio() < params.minimum_occupied_ratio) continue;
        negs->samples.push_back(cur_sample);
        gened_bb++;
    }
}

void JitterOneSample(const cpu_tsdf::TSDFHashing &scene_tsdf, const DetectionParams &params, const SampleTemplate &obb_template, const Sample &sample, int jitter_num, SampleCollection *jittered_samples) {
    tsdf_utility::OrientedBoundingBox sample_obb = sample.OBB();
    Eigen::Vector3f bottom_center = sample_obb.BottomCenter();
    Eigen::Vector3f side_lengths = sample_obb.SideLengths();
    float theta = sample_obb.AngleRangeTwoPI();
    const float dev_x = side_lengths[0] * params.jitter_x_sidelength_percent;
    const float dev_y = side_lengths[1] * params.jitter_y_sidelength_percent;
    const float dev_theta = params.jitter_angle;
    // const float dev_sides = 0.05;
    // std::normal_distribution<float> theta_distribution(theta, dev_theta);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> theta_distribution(theta - dev_theta, theta + dev_theta);
    std::normal_distribution<float> x_distribution(bottom_center[0], dev_x);
    std::normal_distribution<float> y_distribution(bottom_center[1], dev_y);
    //std::normal_distribution<float> sidex_distri(side_lengths[0], side_lengths[0] * dev_sides);
    //std::normal_distribution<float> sidey_distri(side_lengths[1], side_lengths[1] * dev_sides);
    //std::normal_distribution<float> sidez_distri(side_lengths[2], side_lengths[2] * dev_sides);
    for (int i = 0; i < jitter_num; ++i) {
        Eigen::Vector3f jittered_bottom_center = Eigen::Vector3f(x_distribution(generator),  y_distribution(generator), bottom_center[2]);
        float jittered_angle = theta_distribution(generator);
        // Eigen::Vector3f jittered_sidelengths(sidex_distri(generator), sidey_distri(generator), sidez_distri(generator));
        Eigen::Vector3f jittered_pos(jittered_bottom_center[0], jittered_bottom_center[1], jittered_angle);
       // tsdf_detection::SampleTemplate cur_template(tsdf_utility::OrientedBoundingBox(jittered_sidelengths, jittered_bottom_center, jittered_angle),
       //                                             obb_template.sample_size());
       // jittered_samples->samples.push_back(SampleFromPos(scene_tsdf, cur_template, params, jittered_pos, (uint64_t)-1, sample.gt_label()));
        jittered_samples->samples.push_back(SampleFromPos(scene_tsdf, obb_template, params, jittered_pos, (uint64_t)-1, sample.gt_label()));
    }
}

void JitterSamples(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &discret_info, const DetectionParams &params, const SampleTemplate &obb_template, const SampleCollection &samples, const int jitter_num, SampleCollection *jittered_samples) {
    jittered_samples->samples.insert(jittered_samples->samples.end(), samples.samples.begin(), samples.samples.end());
    for (int i = 0; i < samples.samples.size(); ++i) {
        LOG(INFO) << "generating jitter samples for " << i << "th sample.";
        SampleTemplate cur_template_obb(samples.samples[i].OBB(), obb_template.sample_size());
        JitterOneSample(scene_tsdf, params, cur_template_obb, samples.samples[i], jitter_num, jittered_samples);
    }
}

void StepSizeFromOBB(const Eigen::Vector3f &side_lengths, float &delta_x, float &delta_y, float &delta_rotation) {
    float min_sidelength = side_lengths.minCoeff();
    if (min_sidelength > 8) {// larger objects, use larger step size
        delta_x = 1; delta_y = 1; delta_rotation = 2 / 180.0 * M_PI;
    } else {
        delta_x = 0.5; delta_y = 0.5; delta_rotation = 1 / 180.0 * M_PI;
    }
}



//bool FindIntersectionOBB(const tsdf_utility::OrientedBoundingBox &obb, const std::vector<tsdf_utility::OrientedBoundingBox> &to_match, int *matched_idx)
//{
//    *matched_idx = -1;
//    *intersect_area = 0;
//    for (int i = 0; i < to_match.size(); ++i) {
//        if (tsdf_test::TestOBBIntersection(obb, to_match[i])) {
//           *matched_idx = i;
//        }
//    }
//    return *matched_idx >= 0;
//}


}
