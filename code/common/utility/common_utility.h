#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace bfs = boost::filesystem;

namespace utility {
template<typename T>
  inline T min_vec3(const T& lhs, const T& rhs)
  {
    T res;
    res[0] = std::min(lhs[0], rhs[0]);
    res[1] = std::min(lhs[1], rhs[1]);
    res[2] = std::min(lhs[2], rhs[2]);
    return res;
  }
  template<typename T>
  inline T max_vec3(const T& lhs, const T& rhs)
  {
    T res;
    res[0] = std::max(lhs[0], rhs[0]);
    res[1] = std::max(lhs[1], rhs[1]);
    res[2] = std::max(lhs[2], rhs[2]);
    return res;
  }

template<typename T>
  inline void GetFalseColors(const cv::Mat_<T>& original_image, cv::Mat& image, float max_val = -1)
  {
    assert(original_image.channels() == 1);

    // color map
    float map[8][4] = {{0,0,0,114},{0,0,1,185},{1,0,0,114},{1,0,1,174},
                       {0,1,0,114},{0,1,1,185},{1,1,0,114},{1,1,1,0}
                      };
    float sum = 0;
    for (int32_t i=0; i<8; i++)
      sum += map[i][3];

    float weights[8]; // relative weights
    float cumsum[8];  // cumulative weights
    cumsum[0] = 0;
    for (int32_t i=0; i<7; i++)
      {
        weights[i]  = sum/map[i][3];
        cumsum[i+1] = cumsum[i] + map[i][3]/sum;
      }

    // create color png image
    image.create(original_image.rows, original_image.cols, CV_8UC3);
    int height = original_image.rows;
    int width = original_image.cols;

    if (max_val == -1)
      {
        max_val = std::numeric_limits<T>::max();
      }
    else if (max_val == -2)
      {
        double minf, maxf;
        cv::minMaxLoc(original_image, &minf, &maxf);
        max_val = maxf;
      }

    // for all pixels do
    for (int32_t v=0; v<height; v++)
      {
        for (int32_t u=0; u<width; u++)
          {

            // get normalized value
            float val = std::min<double>(std::max<double>(double(original_image(v, u))/max_val, 0.0f),1.0f);

            // find bin
            int32_t i;
            for (i=0; i<7; i++)
              if (val<cumsum[i+1])
                break;

            // compute red/green/blue values
            float   w = 1.0-(val-cumsum[i])*weights[i];
            uint8_t r = (uint8_t)((w*map[i][0]+(1.0-w)*map[i+1][0]) * 255.0);
            uint8_t g = (uint8_t)((w*map[i][1]+(1.0-w)*map[i+1][1]) * 255.0);
            uint8_t b = (uint8_t)((w*map[i][2]+(1.0-w)*map[i+1][2]) * 255.0);

            // set pixel
            //image.set_pixel(u,v,png::rgb_pixel(r,g,b));
            image.at<cv::Vec3b>(v, u) = cv::Vec3b(b,g,r);
          }
      }

    // write to file
    //cv::imwrite(filename, image);
  }

  template<typename T>
  inline void WriteFalseColors(const cv::Mat_<T>& original_image, const std::string& filename, float max_val = -1)
  {
    cv::Mat colored_image;
    GetFalseColors(original_image, colored_image, max_val);
    cv::imwrite(filename, colored_image);
  }

//  inline bool OutputVector(const std::string& filename, const std::vector<float>& vec)
//  {
//      FILE* hf = fopen(filename.c_str(), "w");
//      for (int i = 0; i < vec.size(); i+=2)
//      {
//          fprintf(hf, "%f %f\n", vec[i], vec[i+1]);
//      }
//      fprintf(hf, "\n");
//      fclose(hf);
//      return true;
//  }

  inline bool OutputVector(const std::string& filename, const std::vector<float>& vec)
  {
      FILE* hf = fopen(filename.c_str(), "w");
      for (int i = 0; i < vec.size(); i++)
      {
          fprintf(hf, "%f\n", vec[i]);
      }
      fprintf(hf, "\n");
      fclose(hf);
      return true;
  }

  template<typename T>
  inline bool OutputVectorTemplate(const std::string& filename, const std::vector<T>& vec)
  {
      using namespace std;
      ofstream os(filename);
      for (int i = 0; i < vec.size(); i++)
      {
          os << vec[i] << endl;
      }
      os << endl;
      return true;
  }

  inline bool ReadinVector(const std::string& filename, std::vector<float>& vec)
  {
      FILE* hf = fopen(filename.c_str(), "r");
      int in_num = -1;
      float cur_num;
      while((in_num = fscanf(hf, "%f\n", &cur_num)) == 1)
      {
          vec.push_back(cur_num);
      }
      fclose(hf);
      return true;
  }

  inline void mapJet(double v, double vmin, double vmax, uchar& r, uchar& g, uchar& b)
  {
      r = 255;
      g = 255;
      b = 255;

      if (v < vmin) {
          v = vmin;
      }

      if (v > vmax) {
          v = vmax;
      }

      double dr, dg, db;

      if (v < 0.1242) {
          db = 0.504 + ((1. - 0.504) / 0.1242)*v;
          dg = dr = 0.;
      }
      else if (v < 0.3747) {
          db = 1.;
          dr = 0.;
          dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
      }
      else if (v < 0.6253) {
          db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
          dg = 1.;
          dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
      }
      else if (v < 0.8758) {
          db = 0.;
          dr = 1.;
          dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
      }
      else {
          db = 0.;
          dg = 0.;
          dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
      }

      r = (uchar)(255 * dr);
      g = (uchar)(255 * dg);
      b = (uchar)(255 * db);
  }

inline void Write3DPointToFile(const std::string& fname, const std::vector<cv::Vec3d>& points3d, const std::vector<cv::Vec3b>* colors = NULL)
{
      FILE* hf = fopen(fname.c_str(), "w");
      assert(hf);
      for(int i=0; i < points3d.size(); ++i) {
          fprintf(hf, "%f %f %f", points3d[i][0], points3d[i][1], points3d[i][2]);
          if (colors)
          {
              fprintf(hf, " %d %d %d\n", (*colors)[i][0], (*colors)[i][1], (*colors)[i][2]);
          }
          else
          {
              fprintf(hf, "\n");
          }
      }
      fclose(hf);

//      pcl::PointCloud<pcl::PointXYZRGB> pcl;
//      for (int tt = 0; tt < points3d.size(); ++tt)
//      {
//          pcl::PointXYZRGB curpt;
//          curpt.x = points3d[tt][0];
//          curpt.y = points3d[tt][1];
//          curpt.z = points3d[tt][2];
//          curpt.r = (*colors)[tt][0];
//          curpt.g = (*colors)[tt][1];
//          curpt.b = (*colors)[tt][2];
//          pcl.push_back(curpt);
//      }
//      pcl::io::savePLYFile (fname, pcl);
  }

inline std::string int2str(int n, int padding = 10)
{
    std::ostringstream ss;
    ss << std::setw(padding) << std::setfill('0') << n;
    return ss.str();
}

inline std::string double2str(double n, int padding = 12, int precision = 6)
{
    std::ostringstream ss;
    ss << std::setw(padding) << std::setprecision(precision) << std::setfill('0') << n;
    return ss.str();
}

inline std::string AppendPathSuffix(const std::string& filepath, const std::string& suffix)
{
    return boost::filesystem::path(filepath).replace_extension(suffix).string();
}

inline bool EqualFloat (float A, float B)
{
   static const float EPSILON = 1e-4;
   float diff = A - B;
   return (diff < EPSILON) && (-diff < EPSILON);
}

}
