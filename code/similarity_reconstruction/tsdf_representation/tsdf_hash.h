/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
/*
 * TSDFHashing: representing the TSDF with hashmap
 * also stores information for world coordinate / voxel coordinate conversion
 *
 * functions:
 * 1. integrateCloud_Spherical_Queue (calls EnqueueModifiedBricks and UpdateBrick)
 * Integrating a new depth map into the existing TSDF representation
 * First the affected bricks are enqueued, and then all voxels in the affected bricks are updated
 *
 * 2. AddBrickUpdateList()
 *    UpdateBricksInQueue() (calls UpdateBrick())
 * By using these two functions we can implement a frequent pattern for updating the TSDFs
 * i.e. first the affected bricks for some operation (with its neighbors) are enqueued,
 * then the bricks in the queue are updated.
 * UpdateBricksInQueue takes a functor TSDFVoxelUpdater which is responsible for computing the
 * TSDF values/weights etc. for a voxel. The computed values are fused with the original value
 * of the voxel by weighted average.
 *
 * 3. Inserting TSDF value at some point
 * The SetTSDFValue() functions sets the TSDF value of a voxel. If it doesn't exist, a new brick is allocated.
 * The AddObservation() functions updates the TSDF value of a voxel by weighted average.
 * If the brick containing the voxel doesn't exist, it's allocated.
 *
 * 4. Getting TSDF value at some point
 * The RetriveData()/RetriveDataFromWorldCoord() functions gets TSDF value at some point.
 * If the voxel doesn't exist, it returns false and fills the returned TSDF data with invalid values.
 * RetriveGradient() functions gets TSDF gradient values at some point
 *
 * 5. iterating through all the voxels
 * typedef VoxelHashMap::iterator iterator;
 * typedef VoxelHashMap::const_iterator const_iterator;
 * usage: for (TSDFHashing::iterator itr = tsdf_model.begin(); itr != tsdf_model.end(); ++itr) { (*itr) //std::pair<const BrickPosition, BrickData> }
 *
 * 6. debugging
 * DisplayInfo(): display some information about the hashmap
 * OutputTSDFGrid(): output the TSDF grid to text file. Each point colored according to its TSDF value.
 *
 * 7. other getter/setter functions
 *
 * 8. converting between the voxel coordinate and the world coordinate
*/

#include <string>
#include <iostream>
#include <unordered_set>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "common/utilities/common_utility.h"
#include "voxel_hashmap.h"
#include "voxel_data.h"


namespace cv {
class Mat;
}

class RectifiedCameraPair;

namespace cpu_tsdf
{
  class TSDFHashing
  {
  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & voxel_hash_map_;
      ar & voxel_length_;
      ar & offset_(0);
      ar & offset_(1);
      ar & offset_(2);
      ar & max_dist_pos_;
      ar & max_dist_neg_;
      ar & neighbor_adding_limit_;
    }
  public:
    typedef boost::shared_ptr<TSDFHashing> Ptr;
    typedef boost::shared_ptr<const TSDFHashing> ConstPtr;
    typedef std::unordered_set<VoxelHashMap::BrickPosition, VoxelHashMap::BrickPositionHasher,
    VoxelHashMap::BrickPositionEqual> update_hashset_type;
#ifdef _DEBUG
    // (for debugging) check if two hash representations are the same
    bool operator == (const TSDFHashing& rhs) const
    {
      const float epsilon = 1e-5;
      if (Eigen::Vector3f(offset_ - rhs.offset_).cwiseAbs().sum() > epsilon) return false;
      if (fabs(voxel_length_ - rhs.voxel_length_) > epsilon) return false;
      cv::Vec3i voxel_min_pt, voxel_max_pt;
      voxel_hash_map_.getBoundingBoxInVoxelCoord(voxel_min_pt, voxel_max_pt);
      cv::Vec3i voxel_min_pt2, voxel_max_pt2;
      rhs.voxel_hash_map_.getBoundingBoxInVoxelCoord(voxel_min_pt, voxel_max_pt);
      if (voxel_min_pt != voxel_min_pt2) return false;
      if (voxel_max_pt != voxel_max_pt2) return false;
      for (int x = voxel_min_pt[0]; x <= voxel_max_pt[0]; ++x)
        for (int y = voxel_min_pt[1]; y <= voxel_max_pt[1]; ++y)
          for (int z = voxel_min_pt[2]; z <= voxel_max_pt[2]; ++z)
            {
              float d1, d2;
              float w1, w2;
              cv::Vec3b c1, c2;
              voxel_hash_map_.RetriveData(cv::Vec3i(x, y, z), &d1, &w1, &c1);
              rhs.voxel_hash_map_.RetriveData(cv::Vec3i(x, y, z), &d2, &w2, &c2);
              if (fabs(d1 - d2) < epsilon || fabs(w1 - w2) < epsilon)
                {
                  return false;
                }
            }
      return true;
    }
#endif

    TSDFHashing()
    : voxel_length_(0.2),
      offset_(),
      max_dist_pos_(voxel_length_*8), max_dist_neg_(-voxel_length_*8),
      neighbor_adding_limit_(1) {}
    TSDFHashing(const Eigen::Vector3f offset, float voxel_length)
      :offset_(offset), voxel_length_(voxel_length),
        max_dist_pos_(voxel_length_*8), max_dist_neg_(-voxel_length_*8), neighbor_adding_limit_(1) {}
    TSDFHashing(const Eigen::Vector3f offset, float voxel_length, float vmax_dist)
        :offset_(offset), voxel_length_(voxel_length),
          max_dist_pos_(vmax_dist), max_dist_neg_(-vmax_dist), neighbor_adding_limit_(1) {}
    TSDFHashing(const Eigen::Vector3f offset, float voxel_length,
                float pos_max_dist, float neg_max_dist)
        :offset_(offset), voxel_length_(voxel_length),
          max_dist_pos_(pos_max_dist), max_dist_neg_(neg_max_dist), neighbor_adding_limit_(1)
    { this->Init(voxel_length_, offset_, max_dist_pos_, max_dist_neg_); }
    TSDFHashing(const Eigen::Vector3f offset, float voxel_length, float pos_max_dist, float neg_max_dist, int vneighbor_adding_limit)
        :offset_(offset), voxel_length_(voxel_length),
          max_dist_pos_(pos_max_dist), max_dist_neg_(neg_max_dist), neighbor_adding_limit_(vneighbor_adding_limit) {}

    void CopyHashParametersFrom(const cpu_tsdf::TSDFHashing& tsdf);
    void Init(float voxel_length, const Eigen::Vector3f& offset, float max_dist_pos, float max_dist_neg);
    ~TSDFHashing(){};

    inline void SetAllTSDFWeightToOne()
    {
       voxel_hash_map_.SetAllTSDFWeightToOne();
    }

    static inline float getVoxelMaxWeight() { return VoxelHashMap::getVoxelMaxWeight(); }

    // 1. Integrating a new depth map (with confidence & RGB image and optionally semantic_labels) into
    // the 3D TSDF representation
    bool integrateCloud_Spherical_Queue (const cv::Mat& depth,
                                         const cv::Mat& confidence,
                                         const cv::Mat& image, const cv::Mat &semantic_label_mat,
                                         const RectifiedCameraPair& cam_info, float neg_dist_full_weight_delta, float neg_weight_thresh, float neg_weight_dist_thresh);
    bool integrateCloud_Spherical_Queue_DoubleImCoord (const cv::Mat& depth,
                                                      const cv::Mat& confidence,
                                                      const cv::Mat& image,
                                                      const cv::Mat& semantic_label_mat,
                                                      const RectifiedCameraPair& cam_info,
                                                      const double stepsize,
                                                       float neg_dist_full_weight_delta,
                                                       float neg_weight_thresh,
                                                       float neg_weight_dist_thresh);
    float ComputeTSDFWeight(float diff_observed_dist_cur_dist,
                                         float neg_dist_full_weight_delta,
                                         float neg_weight_thresh,
                                         float neg_weight_dist_thresh);

    // 2. By using these two functions we can implement a frequent pattern for updating the TSDFs
    // i.e. first the affected bricks for some operation (with its neighbors) are enqueued,
    // then the bricks in the queue are updated.
    // UpdateBricksInQueue takes a functor TSDFVoxelUpdater which is responsible for computing the
    // TSDF values/weights etc. for a voxel. The computed values are fused with the original value
    // of the voxel by weighted average.
    bool AddBrickUpdateList(const cv::Vec3i& voxel_point,
                            update_hashset_type* brick_update_hashset) const;
    template<typename TSDFVoxelUpdater>
    inline bool UpdateBricksInQueue(const update_hashset_type& update_hashset, TSDFVoxelUpdater& voxel_updater)
    {
      // fprintf(stderr, "update modified bricks\n");
        for(update_hashset_type::const_iterator itr = update_hashset.begin(); itr != update_hashset.end(); ++itr)
        {
            UpdateBrick(*itr, voxel_updater);
        }
      // voxel_hash_map_.DisplayHashMapInfo();
      // fprintf(stderr, "finished updating\n");
      return true;
    }
    // Update one brick
    template<typename TSDFVoxelUpdater>
    inline bool UpdateBrick(const VoxelHashMap::BrickPosition& bpos, TSDFVoxelUpdater& voxel_updater)
    {
      const int semantic_labelv = -1;
      VoxelHashMap::BrickData temp_bdata;
      VoxelHashMap::BrickData* bdata = NULL;
      bool newdata = false;
      bool notempty = false;
      if (!voxel_hash_map_.FindBrickData(bpos, &bdata))
        {
          bdata = &temp_bdata;  // use temp data
          newdata = true;
        }
      const int sidelength = VoxelHashMap::kBrickSideLength;
      cv::Vec3i base_voxel_pos(bpos[0], bpos[1], bpos[2]);
      for (int ix = 0; ix < sidelength; ++ix)
        for (int iy = 0; iy < sidelength; ++iy)
          for (int iz = 0; iz < sidelength; ++iz)
            {
              float length;
              int cur_imx, cur_imy;
              cv::Vec3i offset(ix, iy, iz);
              cv::Vec3i cur_voxel_pos = base_voxel_pos + offset;
              float d_inc, w_inc;
              cv::Vec3b rgb_color;
              if (voxel_updater(cur_voxel_pos, &d_inc, &w_inc, &rgb_color))
                {
                  notempty = true;
                  voxel_hash_map_.AddObservation(*bdata, offset, d_inc, w_inc, rgb_color, semantic_labelv,
                                                 max_dist_pos_, max_dist_neg_);
                }
            }
      if (newdata && notempty)
        {
          // if the brick is newly added, and it's not empty
          // then insert it into the hashmap
          bool res = voxel_hash_map_.InsertNewBrickData(bpos, *bdata);
          assert(true == res);
        }
      return true;
    }

    // 3. Series of functions for inserting values into the hashmap
    inline bool SetTSDFValue(const cv::Vec3i& voxel_coord,
                             const float dist,
                             const float weight,
                             const cv::Vec3b& vcolor)
    {
      return voxel_hash_map_.SetTSDFValue(voxel_coord, dist, weight, vcolor, max_dist_pos_, max_dist_neg_);
    }

    inline bool SetTSDFValue(const cv::Vec3i& voxel_coord,
                             const float dist,
                             const float weight,
                             const cv::Vec3b& vcolor,
                             const VoxelData::VoxelState st)
    {
      return voxel_hash_map_.SetTSDFValue(voxel_coord, dist, weight, vcolor, st, max_dist_pos_, max_dist_neg_);
    }

    inline bool SetTSDFValueFromWorldCoord(const cv::Vec3f& world_coord,
                                           const float dist,
                                           const float weight,
                                           const cv::Vec3b& vcolor)
    {
      cv::Vec3f voxel_coord = (world_coord - utility::EigenVectorToCvVector3(offset_))/voxel_length_;
      cv::Vec3i int_voxel_coord(voxel_coord);
      return voxel_hash_map_.SetTSDFValue(int_voxel_coord, dist, weight, vcolor, max_dist_pos_, max_dist_neg_);
    }

    inline bool AddObservation(const cv::Vec3i& int_voxel_coord,
                               const float dist,
                               const float weight,
                               const cv::Vec3b& vcolor, const int semantic_labelv = -1)
    {
      return voxel_hash_map_.AddObservation(int_voxel_coord, dist, weight, vcolor, semantic_labelv, max_dist_pos_, max_dist_neg_);
    }

    inline bool AddObservationFromWorldCoord(const cv::Vec3f& world_coord,
                                             const float dist,
                                             const float weight,
                                             const cv::Vec3b& vcolor)
    {
      const int semantic_labelv = -1;
      cv::Vec3f voxel_coord = (world_coord - utility::EigenVectorToCvVector3(offset_))/voxel_length_;

      //std::cout << "target voxel: \n" << voxel_coord << std::endl;
      cv::Vec3i int_voxel_coord(utility::round(voxel_coord));
      CHECK_LT(cv::norm((voxel_coord - cv::Vec3f(int_voxel_coord))), 1e-3);
      return voxel_hash_map_.AddObservation(int_voxel_coord, dist, weight, vcolor, semantic_labelv, max_dist_pos_, max_dist_neg_);
    }
    // 4. Methods for getting values at specific point from the hash map
    bool RetriveData(const Eigen::Vector3i& voxel_coord, float* d, float* w, cv::Vec3b* color) const;
    bool RetriveData(const Eigen::Vector3i& voxel_coord, float* d, float* w, cv::Vec3b* color, VoxelData::VoxelState* st) const;
    bool RetriveDataFromWorldCoord(const Eigen::Vector3f& world_coord, float* d, float* w = NULL, cv::Vec3b* pcolor =NULL) const;
    bool RetriveDataFromWorldCoord_NearestNeighbor(const Eigen::Vector3f& world_coord, float* d, float* w = NULL, cv::Vec3b* pcolor =NULL) const;
    bool RetriveGradientFromWorldCoord(const Eigen::Vector3f& world_coord, Eigen::Vector3f* grad, Eigen::Vector3f *wgrad = NULL) const;
    bool RetriveAbsGradientFromWorldCoord(const Eigen::Vector3f& world_coord, Eigen::Vector3f* grad) const;

    // RetriveData methods where the input voxel coordinate is cv::Vec3i
    inline bool RetriveData(const cv::Vec3i& voxel_coord, float* d, float* w, cv::Vec3b* color) const
    {
      return RetriveData(Eigen::Vector3i(voxel_coord[0], voxel_coord[1], voxel_coord[2]), d, w, color);
    }
    inline bool RetriveData(const cv::Vec3i& voxel_coord, float* d, float* w, cv::Vec3b* color, VoxelData::VoxelState* st) const
    {
      return RetriveData(Eigen::Vector3i(voxel_coord[0], voxel_coord[1], voxel_coord[2]), d, w, color, st);
    }
    inline bool RetriveDataFromWorldCoord(const cv::Vec3f& world_coord, float* d, float* w = NULL, cv::Vec3b* color = NULL) const
    {
      return RetriveDataFromWorldCoord(Eigen::Vector3f(world_coord[0], world_coord[1], world_coord[2]), d, w, color);
    }
    inline bool RetriveGradientFromWorldCoord(const cv::Vec3f& world_coord, cv::Vec3f* grad, cv::Vec3f* wgrad = NULL) const
    {
      Eigen::Vector3f egrad;
      Eigen::Vector3f ewgrad;
      bool res = RetriveGradientFromWorldCoord(Eigen::Vector3f(world_coord[0], world_coord[1], world_coord[2]), &egrad, &ewgrad);
      (*grad)[0] = egrad(0); (*grad)[1] = egrad(1); (*grad)[2] = egrad(2);
      if (wgrad) *wgrad = utility::EigenVectorToCvVector3(ewgrad);
      return res;
    }
    inline bool RetriveAbsGradientFromWorldCoord(const cv::Vec3f& world_coord, cv::Vec3f* grad) const
    {
      Eigen::Vector3f egrad;
      bool res = RetriveAbsGradientFromWorldCoord(Eigen::Vector3f(world_coord[0], world_coord[1], world_coord[2]), &egrad);
      (*grad)[0] = egrad(0); (*grad)[1] = egrad(1); (*grad)[2] = egrad(2);
      return res;
    }

    // Get the TSDF values and weights for the neighbors of a voxel
    // used in marching cubes
    void GetNeighborPointData(const cv::Vec3i& voxel_coord, int neighborhood,
                              std::vector<float>& d, std::vector<float>& w, std::vector<cv::Vec3b>& colors) const;
    void GetNeighborPointData(const cv::Vec3i& voxel_coord, int neighborhood,
                              std::vector<float>& d, std::vector<float>& w, std::vector<cv::Vec3b>& colors,
                              std::vector<VoxelData::VoxelState>& states) const;

    // 5. for iterating through all the voxels
    typedef VoxelHashMap::iterator iterator;
    typedef VoxelHashMap::const_iterator const_iterator;
    iterator begin() { return voxel_hash_map_.begin(); }
    const_iterator begin() const { return const_iterator(voxel_hash_map_.begin()); }
    iterator end() { return voxel_hash_map_.end(); }
    const_iterator end() const {return const_iterator(voxel_hash_map_.end()); }

    // 6. Display methods
    void DisplayInfo() { voxel_hash_map_.DisplayHashMapInfo(); }
    // Output the TSDF grid to a text file
    // Each grid point is colored by its value using a colormap
    void OutputTSDFGrid(const std::string& filepath,
            const Eigen::Vector3f *bb_min_pt_world ,
            const Eigen::Vector3f *bb_max_pt_world ,
            const RectifiedCameraPair *caminfo , const cv::Mat *depthmap , const float *depth_scale ) const;
    void OutputTSDFGrid(
            const std::string& filepath, const Eigen::Vector3f* bb_min_pt_world, const Eigen::Vector3f* bb_max_pt_world) const;



    // 7. Getter/setter functions
    inline float voxel_length() const { return voxel_length_; }
    inline void voxel_length(float v) { voxel_length_ = v; }

    // voxel to world offset (in world metric)
    inline Eigen::Vector3f offset() const { return offset_; }
    inline void offset(const Eigen::Vector3f& v) { offset_ = v; }

    inline int brick_neighbor_adding_limit() { return neighbor_adding_limit_; }

    inline void setDepthTruncationLimits (float max_dist_pos, float max_dist_neg)
    {
      max_dist_pos_ = max_dist_pos; max_dist_neg_ = max_dist_neg;
      neighbor_adding_limit_ = ceil(std::max(fabs(max_dist_pos_/voxel_length_/(float)VoxelHashMap::kBrickSideLength),
                                             fabs(max_dist_neg_/voxel_length_/(float)VoxelHashMap::kBrickSideLength)));
    }
    inline void getDepthTruncationLimits (float &max_dist_pos, float &max_dist_neg) const
    { max_dist_pos = max_dist_pos_; max_dist_neg = max_dist_neg_; }

    inline Eigen::Vector3f getVoxelOriginInWorldCoord() const {
      return offset_;
    }
    inline void getVoxelUnit3DPointInWorldCoord(Eigen::Vector3f& unit_point) const {
      Eigen::Vector3f origin = getVoxelOriginInWorldCoord();
      unit_point = origin + Eigen::Vector3f(voxel_length_, voxel_length_, voxel_length_);
    }
    inline void getBoundingBoxInWorldCoord(cv::Vec3f& min_pt, cv::Vec3f& max_pt) const
    {
      cv::Vec3i voxel_min_pt, voxel_max_pt;
      voxel_hash_map_.getBoundingBoxInVoxelCoord(voxel_min_pt, voxel_max_pt);
      min_pt = cv::Vec3f(voxel_min_pt) * voxel_length_ + utility::EigenVectorToCvVector3(offset_);
      max_pt = cv::Vec3f(voxel_max_pt) * voxel_length_ + utility::EigenVectorToCvVector3(offset_);
    }
    inline void getBoundingBoxInWorldCoord(Eigen::Vector3f& min_pt, Eigen::Vector3f& max_pt) const
    {
        cv::Vec3f wmin_pt, wmax_pt;
        getBoundingBoxInWorldCoord(wmin_pt, wmax_pt);
        min_pt = utility::CvVectorToEigenVector3(wmin_pt);
        max_pt = utility::CvVectorToEigenVector3(wmax_pt);
    }
    inline void getBoundingBoxInVoxelCoord(cv::Vec3i& min_pt, cv::Vec3i& max_pt) const
    {
        voxel_hash_map_.getBoundingBoxInVoxelCoord(min_pt, max_pt);
    }
    inline void getVoxelBoundingBoxSize(Eigen::Vector3i& bbox) const {
      cv::Vec3i voxel_min_pt, voxel_max_pt;
      voxel_hash_map_.getBoundingBoxInVoxelCoord(voxel_min_pt, voxel_max_pt);
      cv::Vec3i cvresolution = voxel_max_pt - voxel_min_pt;
      bbox(0) = cvresolution[0];
      bbox(1) = cvresolution[1];
      bbox(2) = cvresolution[2];
    }
    inline void RecomputeBoundingBoxInVoxelCoord()
    {
        voxel_hash_map_.RecomputeBoundingBoxInVoxelCoord();
    }
    inline void getWorldBoundingBoxSize(Eigen::Vector3f& bbox) const {
      Eigen::Vector3i voxel_bbox;
      getVoxelBoundingBoxSize(voxel_bbox);
      bbox = voxel_bbox.cast<float>() * voxel_length_;
    }
    inline void CentralizeTSDF() {
      cv::Vec3f min_pt_model, max_pt_model;
      getBoundingBoxInWorldCoord(min_pt_model, max_pt_model);
      cv::Vec3f center_model = (min_pt_model + max_pt_model)/2;
      offset_ -= (utility::CvVectorToEigenVector3(center_model) + Eigen::Vector3f::Random() );
    }
    void Clear() { voxel_hash_map_.Clear(); }

    // 8. voxel/world coordinate conversion
    inline Eigen::Vector3f World2Voxel(const Eigen::Vector3f& world_coord) const
    {
      return (world_coord - offset_)*(1.0/voxel_length_);
    }
    inline Eigen::Vector3f Voxel2World(const Eigen::Vector3f& voxel_coord) const
    {
      return (voxel_coord * voxel_length_  + offset_);
    }
    inline cv::Vec3f World2Voxel(const cv::Vec3f& world_coord) const
    {
      return (world_coord - utility::EigenVectorToCvVector3(offset_))*(1.0/voxel_length_);
    }
    inline cv::Vec3f Voxel2World(const cv::Vec3f& voxel_coord) const
    {
      return (voxel_coord * voxel_length_  + utility::EigenVectorToCvVector3(offset_));
    }
    inline bool Empty() { return voxel_hash_map_.Empty(); }

  public:
    // when doing bilinear interpolation, this threshold determines whether to return points that falls at the edge of the TSDF
    static float bilinear_interpolation_weight_thresh;
  private:
    // only used in integrateCloud_Spherical_Queue function.
    // first enqueue affected bricks and then update the TSDF values within them
    void EnqueueModifiedBricks(int imx, int imy, const RectifiedCameraPair& cam_info,
                               unsigned short quant_depth,
                               update_hashset_type& update_hashset);
    void EnqueueModifiedBricksDoubleImCoord(double imx, double imy, const RectifiedCameraPair& cam_info,
                                            unsigned short quant_depth,
                                            update_hashset_type& update_hashset);
    bool UpdateBrick(const VoxelHashMap::BrickPosition& bpos, const RectifiedCameraPair& cam_info,
                     const cv::Mat& depth, const cv::Mat& confidence, const cv::Mat& image, const cv::Mat &semantic_label,
                     float neg_dist_full_weight_delta, float neg_weight_thresh, float neg_weight_dist_thresh);

    VoxelHashMap voxel_hash_map_;
    // length of a voxel (in world coordinate)
    float voxel_length_;
    // offset of the voxel coordinate to world coordinate (measured in world coordinate)
    Eigen::Vector3f offset_;  // world_origin + offset = voxel
    // maximum trucating distance
    float max_dist_pos_;
    float max_dist_neg_;
    // how many neighboring bricks of an affected brick should be considered for updating.
    // usually set to 1. i.e. when a brick needs
    int neighbor_adding_limit_;
  };

  bool ComputeMedians(TSDFHashing* tsdf);

  bool ScaleTSDFWeight(
          TSDFHashing* tsdf, const float scaling_factor
          );

  bool SaveTSDFPPM(const TSDFHashing* tsdf, const cv::Vec3i min_pt, const cv::Vec3i max_pt,
          const std::string& outfname
          );
}  // namespace cpu_tsdf
//BOOST_CLASS_VERSION(cpu_tsdf::TSDFHashing, 1)

