/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "detector.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include "boost/threadpool.hpp"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "tsdf_hash_utilities/svm_wrapper.h"
#include "tsdf_hash_utilities/utility.h"
#include "detection_utility.h"
#include "detect_sample.h"
#include "tsdf_representation/tsdf_hash.h"
#include "obb_intersection.h"

using namespace std;
using tsdf_test::TestOBBsIntersection2D;

namespace tsdf_detection {

Detector::Detector(const string &svm_modelpath) {
    if(!ReadFromFile(svm_modelpath)) {
        throw std::runtime_error(string("Reading detector model file: ") + svm_modelpath + "failed.");
    }
}

void Detector::Predict(const DetectionParams& params, Sample *sample, float *score, char *label) const
{
    Eigen::SparseVector<float> feature;
    char gt_label;
    sample->GetSparseFeature(&feature, &gt_label, params.min_nonempty_voxel_weight);
    svm_.SVMPredict_Primal(feature, score, label);
    sample->predict_score(*score);
    sample->predict_label(*label);
}

void Detector::Predict(const DetectionParams &params, SampleCollection* samples) const
{
    for (int i = 0; i < samples->samples.size(); ++i) {
        float score;
        char label;
        this->Predict(params, &(samples->samples[i]), &score, &label);
    }
}

void Detector::Train(const DetectionParams& params, const SampleCollection &pos, const SampleCollection &neg)
{
    SampleCollection all_samples;
    all_samples.samples.assign(pos.samples.begin(), pos.samples.end());
    all_samples.samples.insert(all_samples.samples.end(), neg.samples.begin(), neg.samples.end());
    Eigen::SparseMatrix<float> features;
    vector<char> labels;
    all_samples.GetSparseFeatures(params.min_nonempty_voxel_weight, &features, &labels);
    all_samples.samples.clear();
    svm_.SVMTrain(features, labels, params.train_options, params.save_prefix + "svm");
}

bool Detector::SaveToFile(const string &fname) const
{
    return svm_.SaveSVMModel(fname) && sample_template_.WriteToFile(fname + ".obb");
}

bool Detector::ReadFromFile(const string &fname)
{
    if (!(svm_.LoadSVMModel(fname) && sample_template_.ReadFromFile(fname + ".obb"))) return false;
    svm_.ConvertToPrimalForm(sample_template_.sample_size().prod() * 2);
    return true;
}

void NMS_OneAngle(const Eigen::Vector3f &obb_sidelengths, const DetectionParams& params, SampleCollection *samples)
{
    // do NMS for OBBs of the same 2D rotation
    // assuming all OBBs have the same side lengths
    using namespace Eigen;
    if (samples->samples.empty()) return;
    sort(samples->samples.begin(), samples->samples.end(), [](const Sample& lhs, const Sample& rhs) {
        return lhs.predict_score() > rhs.predict_score();
    });
    // the vector2f: x, y of the center of the OBB; side lengths of the samples should be the same
    // computing overlap area for axis-aligned obb, side lengths l1xy and l2xy respectively:
    // d_c: center difference in x, y
    // max( (l1_x + l2_x)/2 - max(fabs(d_{cx}), fabs(l2_x - l1_x)/2), 0 )  // for overlapping segment in x
    // when l1s and l2s are the same, the overlapping area:
    // max(l_x - |d_{cx}|, 0) * max(l_y - |d_{cy}|, 0)
    // the OBBs are rotated to be axis-aligned
    vector<Vector2f> projected_rects(samples->samples.size());
    const Vector3f& side_lengths = obb_sidelengths;
    float bb_area = side_lengths[0] * side_lengths[1];
    for (int i = 0; i < samples->samples.size(); ++i) {
        const Sample& cur_sample = samples->samples[i];
        const tsdf_utility::OrientedBoundingBox& obb = cur_sample.OBB();
        Vector3f center = obb.BottomCenter();
        Matrix3f orient = obb.Orientations();
        Vector3f proj_center = orient.transpose() * center;
        projected_rects[i][0] = proj_center[0];
        projected_rects[i][1] = proj_center[1];
    }
    // NMS
    vector<bool> flag_removed(samples->samples.size(), false);
    const float NMS_overlap_thresh = params.NMS_overlap_threshold;
    for (int i = 0; i < samples->samples.size(); ++i) {
        if (flag_removed[i] == true) continue;
        const Sample& cur_sample = samples->samples[i];
        for (int j = i + 1; j < samples->samples.size(); ++j) {
            if (flag_removed[j] == true) continue;
            const Sample& sup_sample = samples->samples[j];
            Vector3f center_diff = cur_sample.OBB().BottomCenter() - sup_sample.OBB().BottomCenter();
            float overlap = std::max<float>(side_lengths[0] - fabs(center_diff[0]), 0.0f) * std::max<float>(side_lengths[1] - fabs(center_diff[1]), 0.0f);
            float overlap_percent = overlap / (bb_area * 2 - overlap);
            if (overlap_percent > NMS_overlap_thresh) {
                flag_removed[j] = true;
            }
        }
    }  // end for i
    cpu_tsdf::EraseElementsAccordingToFlags(flag_removed, &samples->samples);
}

void NMS(const DetectionParams& params, SampleCollection *samples) {
    using namespace Eigen;
    if (samples->samples.empty()) return;
    sort(samples->samples.begin(), samples->samples.end(), [](const Sample& lhs, const Sample& rhs) {
        return lhs.predict_score() > rhs.predict_score();
    });
    vector<bool> flag_removed(samples->samples.size(), false);
    const float NMS_overlap_thresh = params.NMS_overlap_threshold;
    for (int i = 0; i < samples->samples.size(); ++i) {
        if (flag_removed[i] == true) continue;
        const Sample& cur_sample = samples->samples[i];
        const tsdf_utility::OrientedBoundingBox& cur_obb = cur_sample.OBB();
        for (int j = i + 1; j < samples->samples.size(); ++j) {
            if (flag_removed[j] == true) continue;
            const Sample& sup_sample = samples->samples[j];
            const tsdf_utility::OrientedBoundingBox& sup_obb = sup_sample.OBB();
            if (!TestOBBsIntersection2D(cur_obb, sup_obb)) continue;
            // float overlap_percent = tsdf_test::OBBIOU(cur_obb, sup_obb);
            float overlap_volume = tsdf_test::OBBIntersectionVolume(cur_obb, sup_obb);
            float overlap_percent = overlap_volume / sup_obb.Volume();
            if (overlap_percent > NMS_overlap_thresh) {
                // cout << "doing nms for: " << i << "\t " << "removed : " << j << " overlap percent: " << overlap_percent <<endl;
                flag_removed[j] = true;
            }
        }
    }  // end for i
    cpu_tsdf::EraseElementsAccordingToFlags(flag_removed, &samples->samples);
}

void Detect_OneAngle(const cpu_tsdf::TSDFHashing& scene_tsdf,
        const Detector& model,
        const SceneDiscretizeInfo &discret_info,
        const DetectionParams& params,
        const int discret_angle,
        const float init_angle,
        SampleCollection* samples)
{
    using namespace Eigen;
    cout << "begin detector: angle: " << discret_angle << endl;
    const Vector3i& discret_ranges = discret_info.discret_ranges();
    for (int ix = 0; ix < discret_ranges[0]; ++ix)
        for (int iy = 0; iy < discret_ranges[1]; ++iy) {
            Sample cur_sample = SampleFromDiscretPos(scene_tsdf, discret_info, model.sample_template(), params, Eigen::Vector3i(ix, iy, discret_angle), -1, init_angle);
            if (cur_sample.OccupiedRatio() < params.minimum_occupied_ratio) continue;
            float score;
            char label;
            model.Predict(params, &cur_sample, &score, &label);
            if (score > params.min_score_to_keep) {
                samples->samples.push_back(cur_sample);
            }
        }
    //cout << "adjusting positions" << endl;
    //AdjustSamplesPos(scene_tsdf, model, discret_info, params, samples);
    //cout << "adjusting scales" << endl;
    //AdjustSamplesScales(scene_tsdf, model, discret_info, params, samples);
    //cout << "finished adjusting scales" << endl;
    if (params.do_NMS) {
        cout << "begin NMS: " << discret_angle << endl;
        NMS_OneAngle(model.sample_template().SideLengths(), params, samples);
    }
}

void Detect(const cpu_tsdf::TSDFHashing& scene_tsdf, const Detector& model,
        const SceneDiscretizeInfo &discret_info,
        const DetectionParams& params,
        const float init_angle,
        SampleCollection* samples
        ) {
    using namespace boost::threadpool;
    using namespace Eigen;
    const Vector3i& discret_ranges = discret_info.discret_ranges();
    vector<SampleCollection> sample_collections(discret_ranges[2]);
    {
        // Create a thread pool.
        pool tp(params.detection_total_thread);
        // Add tasks
         for (int iangle = 0; iangle < discret_ranges[2]; ++iangle) {
            tp.schedule(std::bind(&Detect_OneAngle, std::cref(scene_tsdf), std::cref(model),
                                  std::cref(discret_info), std::cref(params), iangle, init_angle, &(sample_collections[iangle])));
        }
    }
    // combine the results into one vector
    cout << "combining results" << endl;
    for (int iangle = 0; iangle < discret_ranges[2]; ++iangle) {
        cout << "parallel angle number: " << iangle << " detected samples: " << sample_collections[iangle].samples.size() << " threshold: " << params.min_score_to_keep << endl;
        samples->samples.insert(samples->samples.end(), sample_collections[iangle].samples.begin(), sample_collections[iangle].samples.end());
        sample_collections[iangle].samples.clear();
    }
    if (params.do_NMS) {
        cout << "begin final NMS" << endl;
        NMS(params, samples);
        cout << "end final NMS" << endl;
    }
}

bool MineNegatives(const cpu_tsdf::TSDFHashing &scene_tsdf, const Detector &model, const SceneDiscretizeInfo &discret_info, const DetectionParams &params, const SampleCollection &pos_samples, SampleCollection *negs)
{
    SampleCollection samples;
    Detect(scene_tsdf, model, discret_info, params, 0.0, &samples);
    cout << samples.samples.size() << endl;
    {
        static int cnt = 0;
        string dirpath = params.save_prefix + "/hardneg" + boost::lexical_cast<std::string>(cnt);
        boost::filesystem::create_directories(boost::filesystem::path(dirpath));
        SampleCollection pos_samples;
        copy_if(samples.samples.begin(), samples.samples.end(), std::back_inserter(pos_samples.samples), [](const Sample& s) {
            return s.predict_score() > 0.5;
        });
        pos_samples.WriteOBBsToPLY(dirpath + "/test_detectres.ply");
        cnt++;
    }
    cout << "2 " << samples.samples.size() << endl;
    RemoveCorrectDetections(params, pos_samples, &samples);
    cout << "3 " << samples.samples.size() << endl;
    // test whether there are new hns
    std::vector<bool> to_remove(samples.samples.size(), false);
    sort(negs->samples.begin(), negs->samples.end(), [](const Sample& lhs, const Sample& rhs) {
        return lhs.obb_index() < rhs.obb_index();
    });
    for (int i = 0; i < samples.samples.size(); ++i) {
        const Sample& samplei = samples.samples[i];
        if (std::binary_search(negs->samples.begin(), negs->samples.end(), samplei, [](const Sample& lhs, const Sample& rhs) {
                                return lhs.obb_index() < rhs.obb_index(); })) {
            to_remove[i] = true;
        }
    }
    cpu_tsdf::EraseElementsAccordingToFlags(to_remove, &(samples.samples));
    cout << samples.samples.size() << endl;
    ////// jitter hard samples
    //SampleCollection jitter_samples;
    //JitterSamples(scene_tsdf, discret_info, params, model.sample_template(), samples, params.positive_jitter_num * 3, &jitter_samples);
    //samples = jitter_samples;
    //cout << jitter_samples.samples.size() << endl;

    cout << "neg: " << negs->samples.size() << endl;
    negs->samples.insert(negs->samples.end(), samples.samples.begin(), samples.samples.end());
    cout << "neg: " << negs->samples.size() << endl;
    // sort by score in descending order
    sort(negs->samples.begin(), negs->samples.end(), [](const Sample& lhs, const Sample& rhs) {
        return lhs.predict_score() > rhs.predict_score();
    });
    // remove non-hard negatives
//    negs->samples.erase(remove_if(negs->samples.begin(), negs->samples.end(), [&params](const Sample& val){
//        return val.predict_score() < params.min_score_to_keep;
//    }), negs->samples.end());
    cout << "neg: " << negs->samples.size() << endl;
    cout << "current hn samples:  " << negs->samples.size() << endl;
    // truncate if too many hns
    if (negs->samples.size() > params.max_hard_negative_number) {
        negs->samples.erase(negs->samples.begin() + params.max_hard_negative_number , negs->samples.end());
    }
    cout << "truncated hn samples:  " << negs->samples.size() << endl;
    return samples.samples.size() > 0;
}

void MineTrainIterations(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &discret_info, const DetectionParams &params, const SampleCollection &pos_samples, Detector *model, SampleCollection *negs)
{
    for (int i = 0; i < params.hard_negative_mining_iterations; ++i) {
        bool new_hard_negative = MineNegatives(scene_tsdf, *model, discret_info, params, pos_samples, negs);
        if (!new_hard_negative) break;
        model->Train(params, pos_samples, *negs);
        model->SaveToFile(params.save_prefix + "/svm_hard_mine" + utility::int2str(i) + ".svm");
    }
}

void TrainDetector(const cpu_tsdf::TSDFHashing &scene_tsdf, const SceneDiscretizeInfo &discret_info, DetectionParams &params, const SampleCollection& pos_samples, Detector *model)
{
    LOG(INFO) << "sample generating begins.";
    SampleCollection jittered_pos;
    params.jitter_x_sidelength_percent = 1.0/30.0;
    params.jitter_y_sidelength_percent = 1.0/30.0;
    params.jitter_angle = M_PI/48;
    JitterSamples(scene_tsdf, discret_info, params, model->sample_template(), pos_samples, params.positive_jitter_num, &jittered_pos);
    string jitter_obb_path = params.save_prefix + "/pos_jitter";
    bfs::create_directories(jitter_obb_path);
    jittered_pos.WriteOBBsToPLY(jitter_obb_path + "/jitter.ply");
    LOG(INFO) << "pos sample jitter done.";
    const int sample_neg_num = 2000;
    SampleCollection negs;
    RandomSampleNegatives(scene_tsdf, discret_info, params, model->sample_template(), pos_samples, sample_neg_num, &negs);
    string neg_obb_path = params.save_prefix + "/sampled_neg";
    bfs::create_directories(neg_obb_path);
    negs.WriteOBBsToPLY(neg_obb_path + "/neg.ply");
    LOG(INFO) << "sample generating done.";
    model->Train(params, jittered_pos, negs);
    model->SaveToFile(params.save_prefix + "/svm_initial.svm");
    model->Predict(params, &negs);
    LOG(INFO) << "initial train done.";
    DetectionParams mine_params = params;
    mine_params.min_score_to_keep = -1;
    mine_params.jitter_x_sidelength_percent = 1.0/10.0;
    mine_params.jitter_y_sidelength_percent = 1.0/10.0;
    mine_params.jitter_angle = M_PI/16;
    MineTrainIterations(scene_tsdf, discret_info, mine_params, jittered_pos, model, &negs);
    LOG(INFO) << "train mine done.";
}

void AdjustSamplesPos(const cpu_tsdf::TSDFHashing &scene_tsdf,
                      const Detector& model,
                      const SceneDiscretizeInfo &discret_info,
                      const DetectionParams &params,
                      SampleCollection* samples) {
    using namespace boost::threadpool;
    //{
    //	pool tp(params.detection_total_thread);
    //	// Add tasks
    //	for (int i = 0; i < samples->samples.size(); ++i) {
    //		tp.schedule(std::bind(&AdjustOneSamplePos, std::cref(scene_tsdf), std::cref(model),
    //		                      std::cref(discret_info), std::cref(params), std::cref(template_obb), &(samples->samples[i])));
    //	}
    //}
    for (int i = 0; i < samples->samples.size(); ++i) {
        AdjustOneSamplePos(scene_tsdf, model, discret_info, params, &(samples->samples[i]));
    }
}

void AdjustOneSamplePos(const cpu_tsdf::TSDFHashing &scene_tsdf,
                      const Detector& model,
                      const SceneDiscretizeInfo &discret_info,
                      const DetectionParams &params,
                      Sample* sample) {
    int category = sample->category_label();
    const int new_step = 4;
    const Eigen::Vector3f& deltas = discret_info.deltas();
    Eigen::Vector3f origin_obb_pos = sample->OBB().OBBPos();
    Eigen::Vector3f new_deltas = deltas/(float)new_step; // for fine tuning the position of the OBB
    SampleTemplate cur_bb_template(sample->OBB(), model.sample_size());  // do not use the template OBB size
    float max_score = std::isnan(sample->predict_score()) ? std::numeric_limits<float>::lowest() : sample->predict_score();
    Eigen::Vector3f best_pos = origin_obb_pos;
    for (int ix = -new_step; ix <= new_step; ++ix )
        for (int iy = -new_step; iy <= new_step; ++iy )
            for (int iz = -new_step; iz <= new_step; ++iz ) {
                Eigen::Vector3f cur_pos = origin_obb_pos + Eigen::Vector3f(ix * new_deltas[0], iy * new_deltas[1], iz * new_deltas[2]);
                Sample cur_sample = SampleFromPos(scene_tsdf, cur_bb_template, params, cur_pos, (uint64_t)-1, sample->gt_label());
                if (cur_sample.OccupiedRatio() < params.minimum_occupied_ratio) continue;
                float cur_score;
                char cur_label;
                model.Predict(params, &cur_sample, &cur_score, &cur_label);
                if (cur_score > max_score) {
                    max_score = cur_score;
                    best_pos = cur_pos;
                }
            }
    (*sample) = SampleFromPos(scene_tsdf, cur_bb_template, params, best_pos, (uint64_t)sample->obb_index(), sample->gt_label());
    // (*sample) = SampleFromPos(scene_tsdf, cur_bb_template, params, best_pos, (uint64_t)-1, sample->gt_label());
    sample->predict_score(max_score);
    sample->category_label(category);
}


void AdjustSamplesScales(const cpu_tsdf::TSDFHashing &scene_tsdf,
                         const Detector& model,
                         const SceneDiscretizeInfo &discret_info,
                         const DetectionParams &params,
                         SampleCollection* samples,
                         const Eigen::Vector3f& scale_lowerbd,
                         const Eigen::Vector3f& scale_upperbd) {
    using namespace boost::threadpool;
    //{
    //	pool tp(params.detection_total_thread);
    //	// Add tasks
    //	for (int i = 0; i < samples->samples.size(); ++i) {
    //		tp.schedule(std::bind(&AdjustOneSamplePos, std::cref(scene_tsdf), std::cref(model),
    //		                      std::cref(discret_info), std::cref(params), std::cref(template_obb), &(samples->samples[i])));
    //	}
    //}
    for (int i = 0; i < samples->samples.size(); ++i) {
        AdjustOneSampleScales(scene_tsdf, model, discret_info, params, &(samples->samples[i]), scale_lowerbd, scale_upperbd);
    }
}

void AdjustOneSampleScales(const cpu_tsdf::TSDFHashing &scene_tsdf,
                           const Detector& model,
                           const SceneDiscretizeInfo &discret_info,
                           const DetectionParams &params,
                           Sample* sample,
                           const Eigen::Vector3f& scale_lowerbd,
                           const Eigen::Vector3f& scale_upperbd) {
    int category = sample->category_label();
    const Eigen::Vector3f scale_deltas(0.05, 0.05, 0.05);
    const Eigen::Vector3f fsteps = (scale_upperbd - scale_lowerbd).cwiseQuotient(scale_deltas);
    const Eigen::Vector3f steps = utility::round(fsteps);
    float max_score = std::isnan(sample->predict_score()) ? std::numeric_limits<float>::lowest() : sample->predict_score();
    Eigen::Vector3f best_scales(1, 1, 1);
    tsdf_utility::OrientedBoundingBox origin_obb = sample->OBB();
    const double angle = origin_obb.AngleRangeTwoPI();
    const Eigen::Vector3f bottom_center = origin_obb.BottomCenter();
    const Eigen::Vector3f side_lengths = origin_obb.SideLengths();
    const Eigen::Vector3f origin_pos = origin_obb.OBBPos();
    for (int ix = 0; ix <= steps[0]; ++ix)
        for (int iy = 0; iy <= steps[1]; ++iy)
            for (int iz = 0; iz <= steps[2]; ++iz) {
                Eigen::Vector3f cur_scale = scale_lowerbd + Eigen::Vector3f(ix * scale_deltas[0], iy * scale_deltas[1], iz * scale_deltas[2]);
                tsdf_utility::OrientedBoundingBox cur_obb(side_lengths.cwiseProduct(cur_scale), bottom_center, angle);
                SampleTemplate cur_bb_template(cur_obb, model.sample_size());
                Sample cur_sample = SampleFromPos(scene_tsdf, cur_bb_template, params, origin_pos, (uint64_t)-1, sample->gt_label());
                if (cur_sample.OccupiedRatio() < params.minimum_occupied_ratio) continue;
                float cur_score;
                char cur_label;
                model.Predict(params, &cur_sample, &cur_score, &cur_label);
                if (cur_score > max_score) {
                    max_score = cur_score;
                    best_scales = cur_scale;
                }
            }
    tsdf_utility::OrientedBoundingBox best_obb(side_lengths.cwiseProduct(best_scales), bottom_center, angle);
    SampleTemplate best_bb_template(best_obb, model.sample_size());
    (*sample) = SampleFromPos(scene_tsdf, best_bb_template, params, origin_pos, (uint64_t)sample->obb_index(), sample->gt_label());
    sample->predict_score(max_score);
    sample->category_label(category);
}


}  // namespace tsdf_detection
