/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include "glog/logging.h"
#include "third_party/libsvm/libsvm-3.20/svm.h"

class SVMWrapper
{
public:
    SVMWrapper()
    {
        param.weight_label = NULL;
        param.weight = NULL;
        model = NULL;
        x_space = NULL;
        prob.x = NULL;
        prob.y = NULL;
        Clear();
    }
    ~SVMWrapper() { Clear(); }
    void Clear();
    bool LoadSVMModel(const std::string& model_path);
    bool SaveSVMModel(const std::string& save_path) const;
    int SVMTrain(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, const std::string& options, const std::string *save_path);
    int SVMTrain(const Eigen::SparseMatrix<float>& features, const std::vector<char> &labels, const std::string& options, const std::string& save_path);
    int SVMPredict(const std::vector<std::vector<float>> &features, const std::vector<int> &input_labels, const std::string & options,
                   std::vector<int> *labels, std::vector<std::vector<float> > *scores, float* accuracy, const std::string *save_path) const;
    int SVMPredict_Primal(const std::vector<std::vector<float>> &features, const std::vector<int> &input_labels, const std::string & options,
                   std::vector<int> *labels, std::vector<std::vector<float> > *scores, float* accuracy, const std::string *save_path) const;
    int SVMPredict_Primal(const std::vector<Eigen::SparseVector<float>> &features, const std::vector<int> &input_labels, const std::string & options,
                   std::vector<int> *labels, std::vector<std::vector<float> > *scores, float* accuracy, const std::string *save_path) const;
    int SVMPredict_Primal(const Eigen::SparseVector<float> &feature, float* score, char* label) const;
    void ConvertToPrimalForm(int feat_dim);
    double cur_obj() { CHECK_NOTNULL(model); return model->obj; }
private:
    int FeatureDimension();
    int SVMPredictImpl(const std::vector<std::vector<float>> &input_features, const std::vector<int> &input_labels, const int predict_probability, std::vector<int> *output_labels, std::vector<std::vector<float> > *scores, float* output_accuracy, const std::string *save_path) const;
    void do_cross_validation();
    int VectorFeaturesToSVMProblem(const std::vector<std::vector<float>>& features, const std::vector<int> &labels);
    int SparseFeaturesToSVMProblem(const Eigen::SparseMatrix<float> &samples, const std::vector<char> &labels);
    int OptionStringToSVMParam(const std::string& options);
    SVMWrapper(const SVMWrapper&);
    // training
    struct svm_parameter param;		// set by parse_command_line
    struct svm_problem prob;		// set by read_problem
    struct svm_model *model;
    struct svm_node *x_space;
    int cross_validation;
    int nr_fold;
    // testing
    std::vector<double> weights;
    double bias;
public:
    std::string model_path_;  // for debugging
    // std::vector<struct svm_node> x;
    // int predict_probability/*=0*/;
};
