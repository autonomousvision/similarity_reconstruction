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
    { }

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
    // update TSDF value with averaging
    inline bool UpdateTSDFValue(const float dinc,
                                const float weightinc,
                                const cv::Vec3b& colorinc,
                                const int semantic_labelv,
                                const float max_dist_pos,
                                const float max_dist_neg)
    {
      if(weightinc == 0.0) return false;
      if(dinc > max_dist_pos || dinc < max_dist_neg) return false;
      float w_new = w_ + weightinc;
      d_ = (d_ * w_ + dinc * weightinc)/(w_new);
      color_[0] = static_cast<uchar>((color_[0] * w_ + colorinc[0] * weightinc)/w_new);
      color_[1] = static_cast<uchar>((color_[1] * w_ + colorinc[1] * weightinc)/w_new);
      color_[2] = static_cast<uchar>((color_[2] * w_ + colorinc[2] * weightinc)/w_new);
      w_ = std::min(w_new, max_weight);
      assert(d_ >= max_dist_neg);
      assert(d_ <= max_dist_pos);

      state_ = w_ > 0 ? FILLED : EMPTY;
      if (semantic_labelv > 0)
        {
          semantic_label_ = semantic_labelv;
        }
      return true;
    }

    inline bool SetAllTSDFWeightToOne()
    {
        w_ = w_ > 0.01 ? 1.0 : w_;
        return true;
    }

    inline bool ClearVoxel()
    {
        w_ = 0;
        color_ = cv::Vec3b( 0,0,0);
        d_ = 0;
        state_ = EMPTY;
    }

    // getter function
    inline bool RetriveData(float* pd, float* pw, cv::Vec3b* pcolor, VoxelState* pstate = NULL, int* vsemantic_label = NULL) const
    {
      *pd = d_;
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
      *pw = std::min(w_, max_weight);
      *pcolor = color_;
        *mean_w_s = Eigen::Vector4f(0, 0, 0, 0);
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
  };

}  // end namespace cpu_tsdf

BOOST_CLASS_VERSION(cpu_tsdf::VoxelData, 1)
