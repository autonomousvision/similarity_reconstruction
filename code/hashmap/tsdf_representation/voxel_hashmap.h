#pragma once
#include <unordered_map>
#include <climits>
#include <utility>
#include <iostream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
//#include <boost/serialization/unordered_map.hpp>
#include <opencv2/opencv.hpp>

#include "voxel_data.h"
#include "utility/serialize_unordered_map.h"
#include "common/utility/common_utility.h"
#include "common/utility/eigen_utility.h"

namespace cpu_tsdf {
  class VoxelHashMap
  {
  private:
    // serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & max_pt_[0];
      ar & max_pt_[1];
      ar & max_pt_[2];
      ar & min_pt_[0];
      ar & min_pt_[1];
      ar & min_pt_[2];
      ar & voxel_hash_map_;
      DisplayHashMapInfo();
    }
  public:
    // getting the brick position where a voxel lies (a brick contains 8*8*8 voxels)
    static const int kShiftPos = 3;
    static const int kBrickSideLength = 8; //(2^3)
    static const unsigned int kBrickPosMask = 0xfffffff8;

    // the brick position (when valid) can always be divided by 8 (kBrickSideLength)
    union BrickPosition
    {
    private:
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
      {
        ar & pos[0];
        ar & pos[1];
        ar & pos[2];
      }
    public:
      struct { int x; int y; int z; };
      int pos[3];
      inline int operator [] (int idx) const { return pos[idx]; }
      //inline int& operator [] (int idx) { return pos[idx]; }
      inline cv::Vec3i ToCvVec3i() const { return cv::Vec3i(x, y, z); }
      explicit BrickPosition(const cv::Vec3i& vpos) : x(vpos[0] & kBrickPosMask), y(vpos[1] & kBrickPosMask), z(vpos[2] & kBrickPosMask) {}
      BrickPosition(int vx, int vy, int vz) : x(vx & kBrickPosMask), y(vy & kBrickPosMask), z(vz & kBrickPosMask) {}
      BrickPosition():x(INT_MIN), y(INT_MIN), z(INT_MIN) {}
    };
    // brick data
    struct BrickData
    {
    private:
      template<typename _Tr, typename _Ptr, typename _Ref, typename _VoxelHashMapItr>
      friend class VoxelIterator;
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
      {
        ar & voxeldatas_;
      }
    public:

      inline bool SetTSDFValue(const cv::Vec3i& index,
                               const float dist,
                               const float weight,
                               const cv::Vec3b& vcolor,
                               const float max_dist_pos,
                               const float max_dist_neg)
      {
        assert(index[0] < kBrickSideLength);
        assert(index[1] < kBrickSideLength);
        assert(index[2] < kBrickSideLength);
        assert(index[0] >= 0);
        assert(index[1] >= 0);
        assert(index[2] >= 0);
        return voxeldatas_[index[0]][index[1]][index[2]].SetTSDFValue(dist, weight, vcolor, max_dist_pos, max_dist_neg);
      }
      inline bool SetTSDFValue(const cv::Vec3i& index,
                               const float dist,
                               const float weight,
                               const cv::Vec3b& vcolor,
                               const VoxelData::VoxelState& st,
                               const float max_dist_pos,
                               const float max_dist_neg)
      {
        return voxeldatas_[index[0]][index[1]][index[2]].SetTSDFValue(dist, weight, vcolor, st, max_dist_pos, max_dist_neg);
      }
      inline bool UpdateTSDFValue(const cv::Vec3i& index,
                                  const float dinc,
                                  const float weightinc,
                                  const cv::Vec3b& color,
                                  const int vsemantic_label,
                                  const float max_dist_pos,
                                  const float max_dist_neg)
      {
        assert(index[0] < kBrickSideLength);
        assert(index[1] < kBrickSideLength);
        assert(index[2] < kBrickSideLength);
        assert(index[0] >= 0);
        assert(index[1] >= 0);
        assert(index[2] >= 0);
        return voxeldatas_[index[0]][index[1]][index[2]].UpdateTSDFValue(dinc, weightinc, color, vsemantic_label, max_dist_pos, max_dist_neg);
      }
      inline bool RetriveData(const cv::Vec3i& index, float* pd, float* pw, cv::Vec3b* pcolor,
                              VoxelData::VoxelState* st = NULL, int* vsemantic_label = NULL) const

      {
        assert(index[0] < kBrickSideLength);
        assert(index[1] < kBrickSideLength);
        assert(index[2] < kBrickSideLength);
        assert(index[0] >= 0);
        assert(index[1] >= 0);
        assert(index[2] >= 0);
        return voxeldatas_[index[0]][index[1]][index[2]].RetriveData(pd, pw, pcolor, st, vsemantic_label);
      }
      VoxelData voxeldatas_[kBrickSideLength][kBrickSideLength][kBrickSideLength];
    };

    // hash function for brick position
    struct BrickPositionHasher
    {
      static const int p1 = 73856093;
      static const int p2 = 19349669;
      static const int p3 = 83492791;
      size_t operator () (const BrickPosition& voxel_pos) const
      {
        return ((size_t)(voxel_pos.x>>kShiftPos)*p1)^
            ((size_t)(voxel_pos.y>>kShiftPos)*p2)^
            ((size_t)(voxel_pos.z>>kShiftPos)*p3);
      }
    };
    // equality definition used in hashmap
    struct BrickPositionEqual
    {
      bool operator () (const BrickPosition& lhs, const BrickPosition& rhs) const
      {
        return     ((lhs.x>>kShiftPos) == (rhs.x>>kShiftPos) &&
                    (lhs.y>>kShiftPos) == (rhs.y>>kShiftPos) &&
                    (lhs.z>>kShiftPos) == (rhs.z>>kShiftPos));
      }
    };

    typedef std::unordered_map<BrickPosition, BrickData, BrickPositionHasher, BrickPositionEqual> voxel_hashing_type;
    typedef voxel_hashing_type::iterator voxel_hashing_type_iterator;
    typedef voxel_hashing_type::const_iterator voxel_hashing_type_const_iterator;

    // an iterator to go through all voxels efficiently while keeping the "blocks" invisible to users
    // usage: for (VoxelHashMap::iterator itr = var.begin(); itr != var.end(); ++itr) { *itr/
    template<typename _Tr, typename _Ptr, typename _Ref, typename _VoxelHashMapItr>
    class VoxelIterator
    {
    public:
      typedef VoxelIterator<VoxelData, VoxelData*, VoxelData&, voxel_hashing_type_iterator> iterator;
      typedef VoxelIterator<VoxelData, const VoxelData*, const VoxelData&, voxel_hashing_type_const_iterator> const_iterator;
      typedef VoxelIterator<_Tr, _Ptr, _Ref, _VoxelHashMapItr> _Self;
      // here the value_type should be std::pair<BrickPosition, BrickData>
      typedef _Tr value_type;
      typedef _Ptr pointer;
      typedef _Ref reference;
      VoxelIterator(const _VoxelHashMapItr& itr,
                    unsigned short vx,
                    unsigned short vy,
                    unsigned short vz) : itr_(itr), ix_(vx), iy_(vy), iz_(vz)
      {
        assert(ix_ < kBrickSideLength);
        assert(iy_ < kBrickSideLength);
        assert(iz_ < kBrickSideLength);
      }
      VoxelIterator(const iterator& rhs) : itr_(rhs.itr_), ix_(rhs.ix_), iy_(rhs.iy_), iz_(rhs.iz_) {}
      cv::Vec3i VoxelCoord() { return cv::Vec3i(itr_->first.x + ix_, itr_->first.y + iy_, itr_->first.z + iz_); }
      _Self& operator++()  // prefix increment
      {
        increment();
        return *this;
      }
      _Self operator++(int)
      {
        _Self __tmp = *this;
        this->increment();
        return __tmp;
      }
      bool operator!=(const _Self& rhs) const
      {
        return this->itr_ != rhs.itr_ || this->ix_ != rhs.ix_ || this->iy_ != rhs.iy_ || this->iz_ != rhs.iz_;
      }
      inline reference operator*() const
      {
        return (itr_->second).voxeldatas_[ix_][iy_][iz_];
      }
      inline pointer operator->() const
      {
        return &(operator*());
      }
    private:
      void increment()
      {
        if(iz_ < kBrickSideLength - 1)
          {
            iz_++;
          }
        else if (iy_ < kBrickSideLength - 1)
          {
            iz_ = 0;
            iy_++;
          }
        else if (ix_ < kBrickSideLength - 1)
          {
            iy_ = iz_ = 0;
            ix_++;
          }
        else
          {
            ix_ = iy_ = iz_ = 0;
            itr_++;
          }
      }
      _VoxelHashMapItr itr_;
      unsigned short ix_, iy_, iz_;

//      inline void RetriveData(cv::Vec3i* pos, float* d, float* w, cv::Vec3b* color) const
//      {
//        *pos = cv::Vec3i(itr_->first.x + ix_, itr_->first.y + iy_, itr_->first.z + iz_);
//        (itr_->second).RetriveData(cv::Vec3i(ix_, iy_, iz_), d, w, color);
//      }
//      inline void RetriveData(cv::Vec3i* pos, float* d, float* w, cv::Vec3b* color, VoxelData::VoxelState* st) const
//      {
//        *pos = cv::Vec3i(itr_->first.x + ix_, itr_->first.y + iy_, itr_->first.z + iz_);
//        (itr_->second).RetriveData(cv::Vec3i(ix_, iy_, iz_), d, w, color, st);
//      }

//      inline bool SetTSDFValue(float d, float w, const cv::Vec3b& color,
//                               float max_dist_pos, float max_dist_neg)
//      {
//        return (itr_->second).SetTSDFValue(cv::Vec3i(ix_, iy_, iz_), d, w, color, max_dist_pos, max_dist_neg);
//      }
//      inline bool SetTSDFValue(float d, float w, const cv::Vec3b& color,
//                               const VoxelData::VoxelState& st, float max_dist_pos, float max_dist_neg)
//      {
//        return (itr_->second).SetTSDFValue(cv::Vec3i(ix_, iy_, iz_), d, w, color, st, max_dist_pos, max_dist_neg);
//      }
    };
    typedef VoxelIterator<VoxelData, VoxelData*, VoxelData&, voxel_hashing_type_iterator>  iterator;
    typedef VoxelIterator<VoxelData, const VoxelData*, const VoxelData&, voxel_hashing_type_const_iterator> const_iterator;
    iterator begin() { return iterator(voxel_hash_map_.begin(), 0, 0, 0); }
    const_iterator begin() const { return const_iterator(voxel_hash_map_.begin(), 0, 0, 0); }
    iterator end() { return iterator(voxel_hash_map_.end(), 0, 0, 0); }
    const_iterator end() const { return const_iterator(voxel_hash_map_.end(), 0, 0, 0); }


//    class ConstVoxelIterator
//    {
//    public:
//      ConstVoxelIterator(VoxelHashMap::voxel_hashing_type_const_iterator citr,
//                         unsigned short vx,
//                         unsigned short vy,
//                         unsigned short vz) : citr_(citr), ix_(vx), iy_(vy), iz_(vz)
//      {
//        assert(ix_ < kBrickSideLength);
//        assert(iy_ < kBrickSideLength);
//        assert(iz_ < kBrickSideLength);
//      }
//      ConstVoxelIterator& operator++()
//      {
//        if(iz_ < kBrickSideLength - 1)
//          {
//            iz_++;
//          }
//        else if (iy_ < kBrickSideLength - 1)
//          {
//            iz_ = 0;
//            iy_++;
//          }
//        else if (ix_ < kBrickSideLength - 1)
//          {
//            iy_ = iz_ = 0;
//            ix_++;
//          }
//        else
//          {
//            ix_ = iy_ = iz_ = 0;
//            citr_++;
//          }
//        //printf("(%d %d %d)\n", ix_, iy_, iz_);
//        return *this;
//      }  // pre
//      bool operator !=(const ConstVoxelIterator& rhs)
//      {
//        return this->citr_ != rhs.citr_ || this->ix_ != rhs.ix_ || this->iy_ != rhs.iy_ || this->iz_ != rhs.iz_;
//      }
//      inline void RetriveData(cv::Vec3i* pos, float* d, float* w, cv::Vec3b* color) const
//      {
//        *pos = cv::Vec3i(citr_->first.x + ix_, citr_->first.y + iy_, citr_->first.z + iz_);
//        (citr_->second).RetriveData(cv::Vec3i(ix_, iy_, iz_), d, w, color);
//      }
//      inline void RetriveData(cv::Vec3i* pos, float* d, float* w, cv::Vec3b* color, VoxelData::VoxelState* st, int* semantic_label) const
//      {
//        *pos = cv::Vec3i(citr_->first.x + ix_, citr_->first.y + iy_, citr_->first.z + iz_);
//        (citr_->second).RetriveData(cv::Vec3i(ix_, iy_, iz_), d, w, color, st, semantic_label);
//      }

//    private:
//      VoxelHashMap::voxel_hashing_type_const_iterator citr_;
//      unsigned short ix_, iy_, iz_;
//    };
//    ConstVoxelIterator begin() const { return ConstVoxelIterator(voxel_hash_map_.begin(), 0, 0, 0); }
//    ConstVoxelIterator end() const { return ConstVoxelIterator(voxel_hash_map_.end(), 0, 0, 0); }

    VoxelHashMap(): min_pt_(INT_MAX, INT_MAX, INT_MAX), max_pt_(INT_MIN, INT_MIN, INT_MIN) {}



    static inline float getVoxelMaxWeight() { return VoxelData::max_weight; }
    inline bool SetTSDFValue(const cv::Vec3i& voxel_coord,
                             const float dist,
                             const float weight,
                             const cv::Vec3b& vcolor,
                             const float max_dist_pos,
                             const float max_dist_neg)
    {
      static const unsigned int kNotAndMask32 = ~kBrickPosMask;
      BrickPosition cur_brick_voxel(voxel_coord);
      BrickData& bdata = voxel_hash_map_[cur_brick_voxel];
      UpdateBoundingBoxInVoxelCoord(cur_brick_voxel);
      return bdata.SetTSDFValue(cv::Vec3i(
                                voxel_coord[0]&kNotAndMask32,
                                voxel_coord[1]&kNotAndMask32,
                                voxel_coord[2]&kNotAndMask32),
          dist, weight, vcolor, max_dist_pos, max_dist_neg);
    }

    inline bool SetTSDFValue(const cv::Vec3i& voxel_coord,
                             const float dist,
                             const float weight,
                             const cv::Vec3b& vcolor,
                             const VoxelData::VoxelState st,
                             const float max_dist_pos,
                             const float max_dist_neg)
    {
      static const unsigned int kNotAndMask32 = ~kBrickPosMask;
      BrickPosition cur_brick_voxel(voxel_coord);
      BrickData& bdata = voxel_hash_map_[cur_brick_voxel];
      UpdateBoundingBoxInVoxelCoord(cur_brick_voxel);
      return bdata.SetTSDFValue(cv::Vec3i(
                                voxel_coord[0]&kNotAndMask32,
                                voxel_coord[1]&kNotAndMask32,
                                voxel_coord[2]&kNotAndMask32),
          dist, weight, vcolor, st, max_dist_pos, max_dist_neg);
    }

    inline bool AddObservation(const cv::Vec3i& voxel_coord, const float dinc, const float weightinc, const cv::Vec3b& color,
                               const int vsemantic_label,
                               const float max_dist_pos, const float max_dist_neg)
    {
      static const unsigned int kNotAndMask32 = ~kBrickPosMask;
      BrickPosition cur_brick_voxel(voxel_coord);
      BrickData& bdata = voxel_hash_map_[cur_brick_voxel];
      UpdateBoundingBoxInVoxelCoord(cur_brick_voxel);
      return bdata.UpdateTSDFValue(cv::Vec3i(
                                     voxel_coord[0]&kNotAndMask32,
                                   voxel_coord[1]&kNotAndMask32,
          voxel_coord[2]&kNotAndMask32),
          dinc, weightinc, color, vsemantic_label, max_dist_pos, max_dist_neg);
    }

    inline bool AddObservation(BrickData& bdata, const cv::Vec3i& brick_offset,
                               const float dinc, const float weightinc, const cv::Vec3b& color,
                               const int vsemantic_label,
                               const float max_dist_pos, const float max_dist_neg)
    {
      return bdata.UpdateTSDFValue(brick_offset, dinc, weightinc, color, vsemantic_label, max_dist_pos, max_dist_neg);
    }

    // insert a new brick filled with data.
    inline bool InsertNewBrickData(const BrickPosition& bpos,
                                   const BrickData& bdata)
    {
      voxel_hashing_type::iterator itr = voxel_hash_map_.find(bpos);
      const BrickPosition& cur_brick_voxel = bpos;
      if (itr == voxel_hash_map_.end())
        {
          UpdateBoundingBoxInVoxelCoord(cur_brick_voxel);
          voxel_hash_map_.insert(std::make_pair(bpos, bdata));
          return true;
        }
      return false;
    }

    // find the brick at given position, create an empty brick if it doesn't exist
    inline BrickData& RetriveBrickDataWithAllocation(const BrickPosition& bpos)
    {
      const BrickPosition& cur_brick_voxel = bpos;
      UpdateBoundingBoxInVoxelCoord(cur_brick_voxel);
      return voxel_hash_map_[bpos];
    }

    // find the brick at given position but not create an empty brick if it doesn't exist
    inline bool FindBrickData(const BrickPosition& bpos, BrickData** data)
    {
      voxel_hashing_type_iterator itr = voxel_hash_map_.find(bpos);
      if(itr != voxel_hash_map_.end())
        {
          *data = &(itr->second);
        }
      return (itr != voxel_hash_map_.end());
    }

//    inline BrickData& RetriveBrickData(const cv::Vec3i& pos)
//    {
//      const BrickPosition cur_brick_voxel(pos);
//      min_pt_ = min_vec3(cur_brick_voxel.ToCvVec3i(), min_pt_);
//      max_pt_ = max_vec3(cur_brick_voxel.ToCvVec3i() + cv::Vec3i(kBrickSideLength, kBrickSideLength, kBrickSideLength), max_pt_);
//      return voxel_hash_map_[cur_brick_voxel];
//    }

    inline bool RetriveData(const cv::Vec3i& voxel_coord, float* d, float* w, cv::Vec3b* color, VoxelData::VoxelState* st = NULL) const
    {
      static const unsigned int kNotAndMask32 = ~kBrickPosMask;
      voxel_hashing_type_const_iterator itr = voxel_hash_map_.find(BrickPosition(voxel_coord));
      if(itr == voxel_hash_map_.end())
        {
          *d = -1;
          *w = 0;
          *color = cv::Vec3b(0, 0, 0);
          if (st) *st = VoxelData::EMPTY;
          return false;
        }
      return (itr->second).RetriveData(cv::Vec3i(
                                       voxel_coord[0]&kNotAndMask32,
                                       voxel_coord[1]&kNotAndMask32,
                                       voxel_coord[2]&kNotAndMask32),
                                       d, w, color, st);
    }

    inline bool RetriveData(const BrickData& bdata, const cv::Vec3i& brick_offset, float* pd, float* pw, cv::Vec3b* pcolor,
                            VoxelData::VoxelState* st = NULL)
    {
      return bdata.RetriveData(brick_offset, pd, pw, pcolor, st);
    }

//    inline bool Find(const cv::Vec3i& voxel_coord, const BrickPosition** pos, const BrickData** data) const
//    {
//      voxel_hashing_type_const_iterator itr = voxel_hash_map_.find(BrickPosition(voxel_coord));
//      if(itr != voxel_hash_map_.end())
//        {
//          *pos = &(itr->first);
//          *data = &(itr->second);
//        }
//      return (itr != voxel_hash_map_.end());
//    }

//    inline bool Find(const cv::Vec3i& voxel_coord, const BrickPosition** pos, BrickData** data)
//    {
//      voxel_hashing_type_iterator itr = voxel_hash_map_.find(BrickPosition(voxel_coord));
//      if(itr != voxel_hash_map_.end())
//        {
//          *pos = &(itr->first);
//          *data = &(itr->second);
//        }
//      return (itr != voxel_hash_map_.end());
//    }

    void DisplayHashMapInfo() const;

    inline void UpdateBoundingBoxInVoxelCoord(const BrickPosition& cur_brick_voxel)
    {
      min_pt_ = utility::min_vec3(cur_brick_voxel.ToCvVec3i(), min_pt_);
      max_pt_ = utility::max_vec3(cur_brick_voxel.ToCvVec3i() + cv::Vec3i(kBrickSideLength, kBrickSideLength, kBrickSideLength), max_pt_);
    }

    inline void getBoundingBoxInVoxelCoord(cv::Vec3i& min_pt, cv::Vec3i& max_pt) const
    {
        // the update process above seems flawed some how.
        // recompute it here
        // adding this will affect sample aligning (in TSDFHashing::CentralizeTSDF)
        // const_cast<VoxelHashMap*>(this)->RecomputeBoundingBoxInVoxelCoord();  // not thread-safe
        min_pt = min_pt_;
        max_pt = max_pt_;
    }

    inline void RecomputeBoundingBoxInVoxelCoord()
    {
        min_pt_ = cv::Vec3i(INT_MAX, INT_MAX, INT_MAX);
        max_pt_ = cv::Vec3i(INT_MIN, INT_MIN, INT_MIN);
        for(const_iterator citr = const_cast<const VoxelHashMap*>(this)->begin(); citr != const_cast<const VoxelHashMap*>(this)->end(); ++citr)
        {
            float d, w;
            cv::Vec3b color;
            citr->RetriveData(&d, &w, &color);
            if (w > 0)
            {
                cv::Vec3i cur_pos = citr.VoxelCoord();
                min_pt_ = utility::min_vec3(min_pt_, cur_pos);
                max_pt_ = utility::max_vec3(max_pt_, cur_pos);
            }
        }
    }

    inline void RemoveDuplicateSurfaceTSDF(float min_mesh_weight)
    {
        for(iterator citr = (this)->begin(); citr != (this)->end(); ++citr)
        {
            citr->RemoveDuplicateSurfaceTSDF(min_mesh_weight);
        }
    }

    inline void SetAllTSDFWeightToOne()
    {
        for(iterator citr = (this)->begin(); citr != (this)->end(); ++citr)
        {
            citr->SetAllTSDFWeightToOne();
        }
    }

    //inline const voxel_hashing_type& getVoxelHashMap() const { return voxel_hash_map_; }

    inline void Clear()
    {
      voxel_hash_map_.clear();
      min_pt_ = cv::Vec3i(INT_MAX, INT_MAX, INT_MAX);
      max_pt_ = cv::Vec3i(INT_MIN, INT_MIN, INT_MIN);
    }

    inline bool Empty() { return voxel_hash_map_.empty(); }

  private:
    voxel_hashing_type voxel_hash_map_;
    cv::Vec3i min_pt_;
    cv::Vec3i max_pt_;  // bounding box
  };
}  // namespace cpu_tsdf

//BOOST_IS_BITWISE_SERIALIZABLE(cpu_tsdf::VoxelHashMap::BrickPosition)
BOOST_CLASS_IMPLEMENTATION(cpu_tsdf::VoxelHashMap::BrickPosition,boost::serialization::object_serializable)
