/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace utility {
template<typename _Matrix_Type_>
  _Matrix_Type_ PseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
  {
    Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
    return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
  }

  template <typename T>
  inline Eigen::Matrix<T, 3, 1> CvVectorToEigenVector3(const cv::Vec<T, 3>& cv_vec) {
    return Eigen::Matrix<T, 3, 1>(cv_vec[0], cv_vec[1], cv_vec[2]);
  }
  template <typename T>
  inline cv::Vec<T, 3> EigenVectorToCvVector3(const Eigen::Matrix<T, 3, 1>& eigen_vec) {
    return cv::Vec<T, 3>(eigen_vec(0), eigen_vec(1), eigen_vec(2));
  }

  template <typename T, int m, int n>
  inline Eigen::Matrix<T, m, n> CvMatxToEigenMat(const cv::Matx<T, m, n>& cv_mat)
  {
    Eigen::Matrix<T, m, n> res;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        {
          res(i, j) = cv_mat(i, j);
        }
    return res;
  }

  template <typename T, int m, int n>
  inline cv::Matx<T, m, n> EigenMatToCvMatx(const Eigen::Matrix<T, m, n>& eigen_mat)
  {
    cv::Matx<T, m, n> res;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        {
          res(i, j) = eigen_mat(i, j);
        }
    return res;
  }

  template <typename T, int m, int n>
  inline Eigen::Matrix<T, m, n> ceil(const Eigen::Matrix<T, m, n>& eigen_mat)
  {
      Eigen::Matrix<T, m, n> res;
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          {
            res(i, j) = std::ceil(eigen_mat(i, j));
          }
      return res;
  }

  template <typename T, int m, int n>
  inline Eigen::Matrix<T, m, n> floor(const Eigen::Matrix<T, m, n>& eigen_mat)
  {
      Eigen::Matrix<T, m, n> res;
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          {
            res(i, j) = std::floor(eigen_mat(i, j));
          }
      return res;
  }

  template <typename T, int m, int n>
  inline Eigen::Matrix<T, m, n> round(const Eigen::Matrix<T, m, n>& eigen_mat)
  {
      Eigen::Matrix<T, m, n> res;
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          {
            res(i, j) = std::round(eigen_mat(i, j));
          }
      return res;
  }

  inline cv::Vec3i round(const cv::Vec3f& vec)
  {
      return cv::Vec3i(std::round(vec[0]), std::round(vec[1]), std::round(vec[2]));
  }

void EigenAffine3fDecomposition(
          const Eigen::Affine3f transform,
          Eigen::Matrix3f* rotation,
          Eigen::Vector3f* scale3d,
          Eigen::Vector3f* trans);

void RSTToEigenAffine(
        const Eigen::Matrix3f& rotation,
        const Eigen::Vector3f& scale3d,
        const Eigen::Vector3f& trans,
        Eigen::Affine3f* transform
        );

  bool WriteEigenMatrix(const Eigen::SparseMatrix<float, Eigen::ColMajor>& data_mat, const std::string& filename);
  bool WriteEigenMatrix(const Eigen::MatrixXf& data_mat, const std::string& filename);
  std::ostream& operator << (std::ostream& ofs, const Eigen::MatrixXf& data_mat);
  template<typename _Scalar, int _Rows, int _Cols>
  inline std::ostream& operator << (std::ostream& fs, const Eigen::Matrix<_Scalar, _Rows, _Cols>& dense_mat) {
      fs.precision(std::numeric_limits<_Scalar>::max_digits10);
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

  template<typename _Scalar, int _Rows, int _Cols>
  inline std::istream& operator >> (std::istream& is, Eigen::Matrix<_Scalar, _Rows, _Cols>& mat) {
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
              is >> (mat)(i, j);
            }
          is.get();
        }
      return is;
  }

  bool LoadMatrix(const std::string& filename, Eigen::SparseMatrix<float, Eigen::ColMajor>* mat);
  bool LoadMatrix(const std::string& filename, Eigen::MatrixXf* mat);
  std::istream& operator >> (std::istream& ifs, Eigen::MatrixXf& mat);

  Eigen::SparseMatrix<float, Eigen::ColMajor> EigenVectorsToEigenSparseMat(const std::vector<Eigen::VectorXf> &samples);
  std::vector<Eigen::VectorXf> EigenSparseMatToEigenVectors(const Eigen::SparseMatrix<float, Eigen::ColMajor>& samples);
  std::vector<Eigen::VectorXf> EigenMatToEigenVectors(const Eigen::MatrixXf& samples);
  std::vector<Eigen::SparseVector<float>> EigenMatToEigenSparseVectors(const Eigen::MatrixXf& samples);

}

