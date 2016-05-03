#include "detect_sample.h"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>

#include "common/utilities/common_utility.h"
#include "tsdf_hash_utilities/oriented_boundingbox.h"
#include "tsdf_operation/tsdf_slice.h"

namespace bfs = boost::filesystem;

namespace tsdf_detection {
const float ClassificationResult::INVALID_FLOAT = -10000;

Sample::Sample(const tsdf_utility::OrientedBoundingBox &obb)
    : obb_(obb), obb_index_((uint64_t)-1), category_label_(-1)
{ }

Sample::Sample(const tsdf_utility::OrientedBoundingBox &obb, float score)
    : obb_(obb), obb_index_((uint64_t)-1), category_label_(-1)
{
    predict_score(score);
}

Sample::Sample(const tsdf_utility::OrientedBoundingBox &obb, int category_label)
    : obb_(obb), obb_index_((uint64_t)-1), category_label_(category_label)
{ }

Sample::Sample(const tsdf_utility::OrientedBoundingBox& obb,
               const Eigen::Vector3i &sample_size,
               const cpu_tsdf::TSDFHashing& tsdf_scene,
               const uint64_t obb_index,
               const float min_nonempty_weight,
               const int gt_label /*= -1*/)
    : obb_(obb), obb_index_(obb_index), category_label_(-1) {
    ExtractFeature(tsdf_scene, sample_size, min_nonempty_weight);
    classify_res_.gt_label = gt_label;
}

Sample &Sample::operator = (const Sample &rhs)
{
    if (this == &rhs) return *this;
    obb_ = rhs.obb_;
    tsdf_vals_ = rhs.tsdf_vals_;
    tsdf_weights_ = rhs.tsdf_weights_;
    obb_index_ = rhs.obb_index_;
    classify_res_ = rhs.classify_res_;
    category_label_ = rhs.category_label_;
}

void Sample::ClearFeature()
{
    tsdf_vals_.setZero();
    tsdf_weights_.setZero();
}

void Sample::ExtractFeature(const cpu_tsdf::TSDFHashing &tsdf_origin, const Eigen::Vector3i &sample_size, const float mesh_min_weight)
{
    tsdf_vals_.setZero();
    tsdf_weights_.setZero();
    cpu_tsdf::TSDFGridInfo grid_info(tsdf_origin, sample_size, mesh_min_weight);
    cpu_tsdf::ExtractOneSampleFromAffineTransform(tsdf_origin, obb_.AffineTransform(), grid_info, &tsdf_vals_, &tsdf_weights_);
}

void Sample::GetDenseFeature(std::vector<float> *out_feature, char *gt_label, float min_nonempty_weight) const
{
    Eigen::SparseVector<float> svec;
    GetSparseFeature(&svec, gt_label, min_nonempty_weight);
    (*out_feature).assign(svec.size(), 0);
    for (Eigen::SparseVector<float>::InnerIterator it(svec); it; ++it) {
        (*out_feature)[it.index()] = it.value();
    }
}

void Sample::GetSparseFeature(Eigen::SparseVector<float> *out_feature, char *gt_label, float min_nonempty_weight) const
{
    const int tsdf_feat_dim = tsdf_vals_.size();
    CHECK_GT(tsdf_feat_dim, 0);
    out_feature->resize(tsdf_vals_.size() * 2);
    out_feature->reserve(tsdf_vals_.nonZeros() * 2);
    for (Eigen::SparseVector<float>::InnerIterator itr(tsdf_weights_); itr; ++itr) {
        if (itr.value() > min_nonempty_weight) {
            out_feature->coeffRef(2 * itr.index()) = tsdf_vals_.coeff(itr.index());
            out_feature->coeffRef(2 * itr.index() + 1) = 1.0f;
        }
    }
    *gt_label = classify_res_.gt_label;
}

void Sample::WriteFeatureToText(const std::string &output_path, float min_nonempty_weight) const
{
    using namespace std;
    vector<float> feat;
    char gt_label;
    GetDenseFeature(&feat, &gt_label, min_nonempty_weight);
    string filename = utility::AppendPathSuffix(output_path, "boxidx_" + utility::int2str(obb_index_) + "_score_" + utility::double2str(predict_score())+ ".txt");
    utility::OutputVector(filename, feat);
}

void Sample::WriteOBBToPLY(const std::string &output_path) const
{
    OBB().WriteToPly(output_path);
}

std::ostream &Sample::OutputOBB(std::ostream &os) const
{
    using namespace std;
    using namespace tsdf_utility;
    os << category_label_ << endl;
    os << obb_index_ << endl;
    os << predict_score() << endl;
    os << OBB() << endl;
    return os;
}

std::istream &Sample::InputOBB(std::istream &is)
{
    using namespace tsdf_utility;
    is >> category_label_;
    is >> obb_index_;
    float score;
    is >> score;
    predict_score(score);
    is >> obb_;
    return is;
}

void SampleCollection::ReadOBBs(const std::string &filepath)
{
    std::ifstream ifs(filepath);
    InputOBBs(ifs);
}

void SampleCollection::WriteOBBs(const std::string &filepath) const
{
    std::ofstream ofs(filepath);
    OutputOBBs(ofs);
}

void SampleCollection::WriteOBBsToPLY(const std::string &filepath) const
{
    for (int i = 0; i < samples.size(); ++i) {
        std::string fname = filepath + "." + boost::lexical_cast<std::string>(i) +
                "." + boost::lexical_cast<std::string>(samples[i].obb_index()) + "." + boost::lexical_cast<std::string>(samples[i].predict_score()) + ".ply";
        samples[i].OBB().WriteToPly(fname);
    }
}

std::ostream &SampleCollection::OutputOBBs(std::ostream &os) const
{
    using namespace std;
    os << samples.size() << endl;
    for (int i = 0; i < samples.size(); ++i) {
        os << i << endl;
        samples[i].OutputOBB(os);
    }
    return os;
}

std::istream &SampleCollection::InputOBBs(std::istream &is)
{
    using namespace std;
    int total_num = -1;
    is >> total_num;
    for (int i = 0; i < total_num; ++i) {
        int temp;
        if (is >> temp) {
        samples.push_back(Sample());
        samples.back().InputOBB(is);
        }
    }
    return is;
}

void SampleCollection::GetOBBCollection(std::vector<tsdf_utility::OrientedBoundingBox> *obbs, std::vector<int>* obj_categories, std::vector<float>* obj_scores) const
{
    obbs->reserve(samples.size());
    for (int i = 0; i < samples.size(); ++i) {
        (*obbs).push_back(samples[i].OBB());
    }
    if (obj_categories) {
        obj_categories->reserve(samples.size());
        for (int i = 0; i < samples.size(); ++i) {
            (*obj_categories).push_back(samples[i].category_label());
        }
    }
    if (obj_scores) {
        obj_scores->reserve(samples.size());
        for (int i = 0; i < samples.size(); ++i) {
            (*obj_scores).push_back(samples[i].predict_score());
        }
    }
}

void SampleCollection::GetSparseFeatures(float min_nonempty_weight, Eigen::SparseMatrix<float> *features, std::vector<char> *labels)
{
    if (samples.empty()) return;
    features->resize(samples[0].FeatureDim(), samples.size() + 1);  // add an empty feature for training
    labels->resize(samples.size() + 1);
    std::vector<Eigen::Triplet<float>> entries(4096);
    for (int i = 0; i < samples.size(); ++i) {
        Eigen::SparseVector<float>::InnerIterator tsdf_itr(samples[i].tsdf_vals_);
        for (Eigen::SparseVector<float>::InnerIterator itr(samples[i].tsdf_weights_); itr; ++itr, ++tsdf_itr) {
            CHECK_EQ(itr.index(), tsdf_itr.index());
            if (itr.value() < min_nonempty_weight) continue;
            entries.push_back(Eigen::Triplet<float>(itr.index() * 2, i, tsdf_itr.value()));
            entries.push_back(Eigen::Triplet<float>(itr.index() * 2 + 1, i, 1.0));
        }
        (*labels)[i] = samples[i].gt_label();
    }
    for (int feati = 0; feati < features->rows() / 2; ++feati) {
        entries.push_back(Eigen::Triplet<float>(feati * 2 + 1, samples.size(), 1.0));
        (*labels)[samples.size()] =  -1;
    }
    features->setFromTriplets(entries.begin(), entries.end());
    return;
}

void SampleCollection::AddSamplesFromOBBs(const std::vector<tsdf_utility::OrientedBoundingBox> &obbs, const Eigen::Vector3i &sample_size, const cpu_tsdf::TSDFHashing &tsdf_scene, const float min_nonempty_weight, const int gt_label)
{
    for (const auto& obbi : obbs) {
        samples.push_back(Sample(obbi, sample_size, tsdf_scene, (uint64_t)-1, min_nonempty_weight, gt_label));
    }
}

void SampleCollection::AddSamplesFromOBBs(const std::vector<tsdf_utility::OrientedBoundingBox> &obbs, const std::vector<int> &sample_model_idx)
{
    for (int i = 0; i < obbs.size(); ++i) {
        samples.push_back(Sample(obbs[i], sample_model_idx[i]));
    }
}

void ReadSampleCollections(const std::string &filepath, std::vector<SampleCollection> *collections)
{
    using namespace std;
    ifstream ifs(filepath);
    int category_num;
    ifs >> category_num;
    collections->resize(category_num);
    for (int i = 0; i < category_num; ++i) {
        (*collections)[i].InputOBBs(ifs);
    }
}

void WriteSampleCollections(const std::vector<SampleCollection> &collections, const std::string &filepath)
{
    using namespace std;
    ofstream ofs(filepath);
    ofs << collections.size() << endl;
    for (int i = 0; i < collections.size(); ++i) {
        (collections)[i].OutputOBBs(ofs);
    }
}

void OBBsFromSampleCollections(const std::vector<SampleCollection> &collections, std::vector<tsdf_utility::OrientedBoundingBox> *obbs, std::vector<int> *sample_model_idx)
{
    for (int i = 0; i < collections.size(); ++i) {
        std::vector<tsdf_utility::OrientedBoundingBox> cur_obbs;
        collections[i].GetOBBCollection(&cur_obbs);
        obbs->insert(obbs->end(), cur_obbs.begin(), cur_obbs.end());
        sample_model_idx->insert(sample_model_idx->end(), cur_obbs.size(), i);
    }
}

}  // namespace tsdf_detection
