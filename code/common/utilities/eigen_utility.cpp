/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "eigen_utility.h"
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

namespace utility {

bool WriteEigenMatrix(const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat, const std::string &filename)
{
  return WriteEigenMatrix(Eigen::MatrixXf(data_mat), filename);
}

bool WriteEigenMatrix(const Eigen::MatrixXf &data_mat, const std::string &filename)
{
  std::ofstream fs(filename);
  const Eigen::MatrixXf& dense_mat = data_mat;
  fs << dense_mat.rows() << " " << dense_mat.cols() << std::endl;
  for (int i = 0; i < dense_mat.rows(); ++i)
    {
      for (int j = 0; j < dense_mat.cols(); ++j)
        {
          fs << dense_mat.coeff(i, j) << " ";
        }
      fs << '\n';
    }
  return true;
}

bool LoadMatrix(const std::string &filename, Eigen::SparseMatrix<float, Eigen::ColMajor> *mat)
{
  using namespace std;
  ifstream is(filename);
  if (!is) return false;
  int rows, cols;
  is >> rows >> cols;
  is.get();
  if (!is) return false;
  mat->resize(rows, cols);
  mat->setZero();
  for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
        {
          float cur_value = 0;
          is >> cur_value;
          if (cur_value != 0)
            {
              mat->insert(i, j) = cur_value;
            }
        }
      is.get();
      if (!is) return false;
    }
  return true;
}

bool LoadMatrix(const std::string &filename, Eigen::MatrixXf *mat)
{
  using namespace std;
  ifstream is(filename);
  if (!is) return false;
  int rows, cols;
  is >> rows >> cols;
  is.get();
  if (!is) return false;
  mat->resize(rows, cols);
  mat->setZero();
  for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
        {
          float cur_value = 0;
          is >> cur_value;
          (*mat)(i, j) = cur_value;
        }
      is.get();
      if (!is) return false;
    }
  return true;
}

void EigenAffine3fDecomposition(const Eigen::Affine3f transform, Eigen::Matrix3f *rotation, Eigen::Vector3f *scale3d, Eigen::Vector3f *trans)
{
    Eigen::Matrix4f transform_mat = transform.matrix();
    Eigen::Matrix3f A = transform_mat.block<3, 3>(0, 0);         // transform_mat(i+1 : i+rows, j+1 : j+cols)
    *scale3d = Eigen::Vector3f(A.col(0).norm(), A.col(1).norm(), A.col(2).norm());
    rotation->col(0) = A.col(0)/(*scale3d)[0];
    rotation->col(1) = A.col(1)/(*scale3d)[1];
    rotation->col(2) = A.col(2)/(*scale3d)[2];
    *trans = transform_mat.block<3, 1>(0, 3);
//    std::cout << transform_mat << std::endl;
//    std::cout << A << std::endl;
//    std::cout << *rotation << std::endl;
//    std::cout << *scale3d << std::endl;
//    std::cout << *trans << std::endl;
    CHECK_LE(((*rotation) * (*rotation).transpose() - Eigen::Matrix3f::Identity()).norm(), 1e-4);
}

void RSTToEigenAffine(const Eigen::Matrix3f &rotation, const Eigen::Vector3f &scale3d, const Eigen::Vector3f &trans, Eigen::Affine3f *transform)
{
    Eigen::Matrix4f transform_mat = Eigen::Matrix4f::Identity();
    transform_mat.block<3, 3>(0, 0) = rotation * scale3d.asDiagonal();
    transform_mat.block<3, 1>(0, 3) = trans;
    transform->matrix() = transform_mat;
}

Eigen::SparseMatrix<float, Eigen::ColMajor> EigenVectorsToEigenSparseMat(const std::vector<Eigen::VectorXf > &samples)
{
    if (samples.empty()) return Eigen::SparseMatrix<float, Eigen::ColMajor>();
    Eigen::SparseMatrix<float, Eigen::ColMajor> res(samples[0].size(), samples.size());
    for (int i = 0; i < samples.size(); ++i)
    {
        res.col(i) = samples[i].sparseView();
    }
    return res;
}

std::vector<Eigen::VectorXf> EigenSparseMatToEigenVectors(const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples)
{
    std::vector<Eigen::VectorXf> res(samples.cols());
    for (int i = 0; i < samples.cols(); ++i)
    {
        res[i] = samples.col(i);
    }
    return res;
}

 std::vector<Eigen::VectorXf> EigenMatToEigenVectors(const Eigen::MatrixXf& samples)
 {
    std::vector<Eigen::VectorXf> res(samples.cols());
    for (int i = 0; i < samples.cols(); ++i)
    {
        res[i] = samples.col(i);
    }
    return res;
 }

 std::vector<Eigen::SparseVector<float>> EigenMatToEigenSparseVectors(const Eigen::MatrixXf& samples)
 {
    std::vector<Eigen::SparseVector<float>> res(samples.cols());
    for (int i = 0; i < samples.cols(); ++i)
    {
        res[i] = samples.col(i).sparseView();
    }
    return res;
 }

 std::ostream &operator <<(std::ostream &fs, const Eigen::MatrixXf &data_mat)
 {
     fs.precision(std::numeric_limits<float>::max_digits10);
     const Eigen::MatrixXf& dense_mat = data_mat;
     fs << dense_mat.rows() << " " << dense_mat.cols() << std::endl;
     for (int i = 0; i < dense_mat.rows(); ++i)
       {
         for (int j = 0; j < dense_mat.cols(); ++j)
           {
             fs << std::fixed << dense_mat.coeff(i, j) << " ";
           }
         fs << '\n';
       }
     return fs;
 }

 std::istream &operator >>(std::istream &is, Eigen::MatrixXf &mat)
 {
     using namespace std;
     int rows, cols;
     is >> rows >> cols;
     is.get();
     mat.resize(rows, cols);
     mat.setZero();
     for (int i = 0; i < rows; ++i)
       {
         for (int j = 0; j < cols; ++j)
           {
             float cur_value = 0;
             is >> cur_value;
             (mat)(i, j) = cur_value;
           }
         is.get();
       }
     return is;
 }

}
