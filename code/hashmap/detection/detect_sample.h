#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <iostream>

#include <Eigen/Eigen>

#include "utility/utility.h"
#include "utility/oriented_boundingbox.h"

namespace cpu_tsdf {
class TSDFHashing;
}

namespace tsdf_detection {

struct ClassificationResult
{
    static const char INVALID_VAL = 127;
    static const float INVALID_FLOAT;
    ClassificationResult() : gt_label(INVALID_VAL), predict_label(INVALID_VAL), predict_score(INVALID_FLOAT) {}
    float predict_score;
    char gt_label;
    char predict_label;
};

class Sample
{
    friend class SampleCollection;
public:
    Sample() {}
    explicit Sample(const tsdf_utility::OrientedBoundingBox &obb);
    Sample(const tsdf_utility::OrientedBoundingBox &obb, float score);
    Sample(const tsdf_utility::OrientedBoundingBox &obb, int category_label);
    Sample(const tsdf_utility::OrientedBoundingBox& obb, const Eigen::Vector3i &sample_size,
           const cpu_tsdf::TSDFHashing& tsdf_scene, const uint64_t obb_index, const float min_nonempty_weight, const int gt_label = -1);
    Sample& operator = ( const Sample& rhs);
    void ExtractFeature(const cpu_tsdf::TSDFHashing& tsdf_origin, const Eigen::Vector3i &sample_size, const float mesh_min_weight);
    void GetSparseFeature(Eigen::SparseVector<float> *out_feature, char* gt_label, float min_nonempty_weight) const;
    void GetDenseFeature(std::vector<float> *out_feature, char *gt_label, float min_nonempty_weight) const;
    void ClearFeature();
    inline float OccupiedRatio() const { return tsdf_weights_.size() > 0 ? float(tsdf_weights_.nonZeros())/float(tsdf_weights_.size()) : 0; }
    inline int FeatureDim() const { return tsdf_vals_.size() * 2; }
    void WriteFeatureToText(const std::string& output_path, float min_nonempty_weight) const;
    void WriteOBBToPLY(const std::string& output_path) const;
    std::ostream& OutputOBB(std::ostream& os) const;
    std::istream& InputOBB(std::istream& is);

    inline const tsdf_utility::OrientedBoundingBox& OBB() const { return obb_; }
    inline tsdf_utility::OrientedBoundingBox& OBB() { return obb_; }
    inline const uint64_t obb_index() const  { return obb_index_; }
    inline char gt_label() const { return classify_res_.gt_label; }
    inline char predict_label() const { return classify_res_.predict_label; }
    inline float predict_score() const { return classify_res_.predict_score; }
    inline void predict_label(char val) { classify_res_.predict_label = val; }
    inline void predict_score(float val) { classify_res_.predict_score = val; }
    inline int category_label() const { return category_label_; }
    inline void category_label(int val) { category_label_ = val; }
private:
    tsdf_utility::OrientedBoundingBox obb_;
    Eigen::SparseVector<float> tsdf_vals_;
    Eigen::SparseVector<float> tsdf_weights_;
    uint64_t obb_index_;
    ClassificationResult classify_res_;
    int category_label_;
};

struct SampleCollection {
    void ReadOBBs(const std::string& filepath);
    void WriteOBBs(const std::string& filepath) const;
    void WriteOBBsToPLY(const std::string& filepath) const;
    std::istream& InputOBBs(std::istream& is);
    std::ostream& OutputOBBs(std::ostream& os) const;
    void GetOBBCollection(std::vector<tsdf_utility::OrientedBoundingBox>* obbs, std::vector<int> *obj_categories = NULL, std::vector<float> *obj_scores = NULL) const;
    void GetSparseFeatures(float min_nonempty_weight, Eigen::SparseMatrix<float>* features, std::vector<char>* labels);
    void AddSamplesFromOBBs(const std::vector<tsdf_utility::OrientedBoundingBox>& obbs, const Eigen::Vector3i& sample_size,
                            const cpu_tsdf::TSDFHashing& tsdf_scene, const float min_nonempty_weight, const int gt_label);
    void AddSamplesFromOBBs(const std::vector<tsdf_utility::OrientedBoundingBox>& obbs, const std::vector<int>& sample_model_idx);
    std::vector<Sample> samples;
};

}  // namespace tsdf_detection
