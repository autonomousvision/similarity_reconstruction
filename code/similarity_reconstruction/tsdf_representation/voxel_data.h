/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <Eigen/Eigen>
/*
 * Class representing data within a voxel:
 * includes:
 * 1. TSDF value, weight, color,
 *
 * 2. VoxelState: Whether the voxel is FILLED, EMPTY or MODIFIABLE.
 * This is used when doing diffusion hole filling
 * (http://graphics.stanford.edu/papers/holefill-3dpvt02/).
 * The TSDF values in FILLED voxels are not modified so that the reconstructed parts don't get smoothed.
 * The diffusion process modifies the TSDF values of the EMPTY voxels to fill the holes, and at the same time changes their states to MODIFIABLE.
 * When doing multiple rounds of diffusion, the values of EMPTY or MODIFIABLE voxels are modified.
 *
 * 3. semantic_label (int): The semantic label associated with a grid point.
 * This is used when we want to extract objects from the reconstruction
 * e.g. A house can be labeled in the image annotation tool with number 1 (i.e. its semantic label)
 * and the labeld pixels back-project to grid points with semantic label 1.
 *
 * member functions: comments in the declaration.
*/

#include <cmath>
#include <iostream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <opencv2/opencv.hpp>

namespace cpu_tsdf {
  class VoxelData
  {
  private:
    // for boost serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & d_;
      ar & w_;
      ar & color_[0];
      ar & color_[1];
      ar & color_[2];
      ar & state_;
      if (version > 0)
        {
          ar & semantic_label_;
        }
    }

  public:
    // max allowed weight. If weight larger than the threshold it's clamped to max_weight.
    static const float max_weight;// = 20
    // see header part
    enum VoxelState { FILLED = 0, EMPTY = 1, MODIFIABLE = 2};
    VoxelData() : d_(-1), w_(0), color_(), state_(EMPTY), semantic_label_(-1)
    {
//        mean_d_pos_ = 0;
//        mean_d_neg_ = 0;
//        pos_total_w_ = 0;
//        neg_total_w_ = 0;
//        pos_color = cv::Vec3f(0,0,0);
//        neg_color = cv::Vec3f(0,0,0);
    }

    inline void SetWeight(const float weight)
    {
        w_ = std::max(0.0f, std::min(weight, max_weight));
    }

    // set TSDF value, updating state_ to FILLED/EMPTY automatically
    inline bool SetTSDFValue(const float dist,
                             const float weight,
                             const cv::Vec3b& vcolor,
                             const float max_dist_pos,
                             const float max_dist_neg)
    {
      if(weight < std::numeric_limits<float>::epsilon())
        {
          w_ = 0;
          d_  = 0;
          state_ = EMPTY;
          return true;
        }
      if(dist > max_dist_pos || dist < max_dist_neg) return false;
      w_ = std::min(weight, max_weight);
      d_ = dist;
      color_ = vcolor;
      assert(d_ >= max_dist_neg);
      assert(d_ <= max_dist_pos);
      state_ = FILLED;
      return true;
    }
    // set TSDF value including state
    inline bool SetTSDFValue(const float dist,
                             const float weight,
                             const cv::Vec3b& vcolor,
                             const VoxelState& st,
                             const float max_dist_pos,
                             const float max_dist_neg)
    {
        if(weight < std::numeric_limits<float>::epsilon())
        {
            w_ = 0;
            d_ = 0;
            state_ = EMPTY;
            return false;
        }
      if(dist > max_dist_pos || dist < max_dist_neg) return false;
      w_ = std::min(weight, max_weight);
      d_ = dist;
      color_ = vcolor;
      assert(d_ >= max_dist_neg);
      assert(d_ <= max_dist_pos);
      state_ = st;
      return true;
    }
//    // update TSDF value with averaging
    inline bool UpdateTSDFValue(const float dinc,
                                const float weightinc,
                                const cv::Vec3b& colorinc,
                                const int semantic_labelv,
                                const float max_dist_pos,
                                const float max_dist_neg)
    {
        //std::cout << "dinc: " << dinc << std::endl;
        //std::cout << "max_dist_pos: " << max_dist_pos<< std::endl;
      if(weightinc == 0.0) return false;
      if(dinc > max_dist_pos || dinc < max_dist_neg) return false;
      float w_new = w_ + weightinc;
      d_ = (d_ * w_ + dinc * weightinc)/(w_new);
      color_[0] = static_cast<uchar>((color_[0] * w_ + colorinc[0] * weightinc)/w_new);
      color_[1] = static_cast<uchar>((color_[1] * w_ + colorinc[1] * weightinc)/w_new);
      color_[2] = static_cast<uchar>((color_[2] * w_ + colorinc[2] * weightinc)/w_new);
      w_ = std::min(w_new, max_weight);
//      if (w_ > 1)
//      {
//          std::cout << "w_: " << w_ << std::endl;
//      }
      assert(d_ >= max_dist_neg);
      assert(d_ <= max_dist_pos);

//      if (dinc >= 0)
//      {
//          mean_d_pos_ = ((mean_d_pos_) * pos_total_w_ + dinc * weightinc) / (pos_total_w_ + weightinc);
//          pos_color  = (pos_color * pos_total_w_ + weightinc * cv::Vec3f(colorinc)) / (pos_total_w_ + weightinc);
//          pos_total_w_ = std::min(pos_total_w_ + weightinc, max_weight);
//      }
//      else
//      {
//          mean_d_neg_ = ((mean_d_neg_) * neg_total_w_ + dinc * weightinc) / (neg_total_w_ + weightinc);
//          neg_color  = (neg_color * neg_total_w_ + weightinc * cv::Vec3f(colorinc)) / (neg_total_w_ + weightinc);
//          neg_total_w_ = std::min(neg_total_w_ + weightinc, max_weight);
//      }

      state_ = w_ > 0 ? FILLED : EMPTY;
      if (semantic_labelv > 0)
        {
          semantic_label_ = semantic_labelv;
        }
      return true;
    }

//    inline bool UpdateTSDFValue(const float dinc,
//                                const float weightinc,
//                                const cv::Vec3b& colorinc,
//                                const int semantic_labelv,
//                                const float max_dist_pos,
//                                const float max_dist_neg)
//    {
//         if (dinc > max_dist_pos || dinc < max_dist_neg) return false;
//         if (weightinc <= 0.0 ) return false;

//         info_for_median_.push_back(SortInfo(dinc, weightinc, colorinc));
//         return true;
//    }

//    inline void ComputeMedian()
//    {
//        if (info_for_median_.empty()) return;
//        std::nth_element(info_for_median_.begin(),
//                                    info_for_median_.begin() + info_for_median_.size()/2,
//                                    info_for_median_.end(), [](const SortInfo& lhs, const SortInfo& rhs){
//            return lhs.curdist < rhs.curdist;
//        });
//        auto itr = info_for_median_.begin() + info_for_median_.size()/2;
////        if (info_for_median_.size() % 2 == 0)
////        {
////            d_ = ((itr - 1)->curdist + itr->curdist) / 2.0;
////            w_ = ((itr - 1)->curweight + itr->curweight) / 2.0;
////            color_ = cv::Vec3b((cv::Vec3f((itr - 1)->curcolor) + cv::Vec3f(itr->curcolor)) / 2.0);
////        }
////        else
//        {
//            d_ = (itr->curdist) ;
//            w_ = ( itr->curweight) ;
//            color_ = (itr->curcolor) ;
//        }
//    }

    inline bool RemoveDuplicateSurfaceTSDF(float min_mesh_weight)
    {
//        if (pos_total_w_ > min_mesh_weight && fabs(mean_d_pos_) < fabs(mean_d_neg_) + 1.0)
//        {
//            d_ = mean_d_pos_;
//            pos_total_w_ = pos_total_w_;
//            color_ = cv::Vec3b(pos_color);
//        }
        return true;
    }

    inline bool SetAllTSDFWeightToOne()
    {
        w_ = w_ > 0.01 ? 1.0 : w_;
        return true;
    }

    inline bool ClearVoxel()
    {
//        mean_d_pos_ = 0;
//        mean_d_neg_ = 0;
//        pos_total_w_ = 0;
//        neg_total_w_ = 0;
//        pos_color = cv::Vec3f(0,0,0);
//        neg_color = cv::Vec3f(0,0,0);
        w_ = 0;
        color_ = cv::Vec3b( 0,0,0);
        d_ = 0;
        state_ = EMPTY;
    }

    // getter function
    inline bool RetriveData(float* pd, float* pw, cv::Vec3b* pcolor, VoxelState* pstate = NULL, int* vsemantic_label = NULL) const
    {
      *pd = d_;
      // *pw = w_;
      *pw = std::min(w_, max_weight);
      *pcolor = color_;
      if (pstate) *pstate = (VoxelState)state_;
      if (vsemantic_label) *vsemantic_label = semantic_label_;
      return true;
    }

    inline bool RetriveDataDebug(float* pd, float* pw, cv::Vec3b* pcolor,
                                 Eigen::Vector4f* mean_w_s,
                                 VoxelState* pstate = NULL, int* vsemantic_label = NULL) const
    {
      *pd = d_;
      // *pw = w_;
      *pw = std::min(w_, max_weight);
      *pcolor = color_;
        *mean_w_s = Eigen::Vector4f(0, 0, 0, 0);
//      (*mean_w_s)[0] = mean_d_pos_;
//      (*mean_w_s)[1] = pos_total_w_;
//      (*mean_w_s)[2] = mean_d_neg_;
//      (*mean_w_s)[3] = neg_total_w_;
      if (pstate) *pstate = (VoxelState)state_;
      if (vsemantic_label) *vsemantic_label = semantic_label_;
      return true;
    }
  private:
    float d_;
    float w_;
    cv::Vec3b color_;
    int semantic_label_;
    unsigned char state_;

//    std::vector<float> dists_;
//    std::vector<float> weights_;
//    std::vector<float> colors_;
//    struct SortInfo
//    {
//        SortInfo(float dist, float weight, const cv::Vec3b& color)
//            :curdist(dist), curweight(weight), curcolor(color)
//        {}
//        float curdist;
//        float curweight;
//        cv::Vec3b curcolor;
//    };
//    std::vector<SortInfo> info_for_median_;

//    float mean_d_pos_;
//    float mean_d_neg_;
//    float pos_total_w_;
//    float neg_total_w_;
//    cv::Vec3f pos_color;
//    cv::Vec3f neg_color;
  };

}  // end namespace cpu_tsdf

BOOST_CLASS_VERSION(cpu_tsdf::VoxelData, 1)
