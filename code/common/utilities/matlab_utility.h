#pragma once
#include <string>

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <glog/logging.h>
#include <matio.h>
#include "mclmcrrt.h"
#include "mclcppclass.h"

class mwArray;

namespace matlabutility {
void EigenSparseMatrixs2MatlabArray(const Eigen::SparseMatrix<float, Eigen::ColMajor>& mat, mwArray* mwarray);

void MatlabArray2EigenMatrix(const mwArray& mwarray, Eigen::MatrixXf* mat);
void MatlabArray2EigenSparseMatrix(const mwArray& mwarray, Eigen::SparseMatrix<float, Eigen::ColMajor>* mat);
void MatlabArray2EigenVector(const mwArray& mwarray, Eigen::VectorXf* vec);
void MatlabArray2EigenSparseVector(const mwArray& mwarray, Eigen::SparseVector<float>* vec);

bool ReadMatrixMatlab(const std::string &filepath, const std::string & varname, Eigen::MatrixXf *matrix);
bool WriteMatrixMatlab(const std::string &filepath, const std::string &variable_name, const Eigen::MatrixXf &matrix);

bool WritePCAOutputResult(const std::string & filepath,
                          Eigen::SparseVector<float> mean_mat,
                          Eigen::SparseVector<float> mean_weight,
                          Eigen::SparseMatrix<float, Eigen::ColMajor> base_mat,
                          Eigen::MatrixXf coeff_mat );

bool ReadPCAOutputResult(const std::string & filepath,
                          Eigen::SparseVector<float>* mean_mat,
                          Eigen::SparseVector<float>* mean_weight,
                          Eigen::SparseMatrix<float, Eigen::ColMajor>* base_mat,
                          Eigen::MatrixXf* coeff_mat );
}
