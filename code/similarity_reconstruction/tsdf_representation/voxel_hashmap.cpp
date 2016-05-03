#include "voxel_hashmap.h"
#include <iostream>

using std::endl;

namespace cpu_tsdf {
void VoxelHashMap::DisplayHashMapInfo() const
{
    fprintf(stderr, "hash element size: %lu, bucket number: %lu\n", voxel_hash_map_.size(),
            voxel_hash_map_.bucket_count());
    int total_bucket =  voxel_hash_map_.bucket_count();
    int total_used_bucket = 0;
    int total_collision = 0;
    int max_bucket_size = 0;
    for(int i = 0; i < total_bucket; ++i)
    {
        int cur_size = voxel_hash_map_.bucket_size(i);
        if(cur_size>0)
        {
            total_used_bucket++;
            total_collision += (cur_size-1);
            max_bucket_size = std::max(max_bucket_size, cur_size);
        }
    }
    fprintf(stderr, "hash bucket count: %d, total_used_bucket: %d\n total_collision: %d, max_bucket_size: %d\nAverage collision: %f\n",
            total_bucket, total_used_bucket, total_collision, max_bucket_size, (float)total_collision/total_used_bucket);
}

//void VoxelHashMap::RecomputeBoundingBoxInVoxelCoord()
//{
//    min_pt_ = cv::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
//    max_pt_ = cv::Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
//    for(const_iterator citr = const_cast<const VoxelHashMap*>(this)->begin(); citr != const_cast<const VoxelHashMap*>(this)->end(); ++citr)
//    {
//        float d, w;
//        cv::Vec3b color;
//        citr->RetriveData(&d, &w, &color);
//        if (w > 0)
//        {
//            cv::Vec3i cur_pos = citr.VoxelCoord();
//            min_pt_ = utility::min_vec3(min_pt_, cur_pos);
//            max_pt_ = utility::max_vec3(max_pt_, cur_pos);
//        }
//    }
//}

//std::ostream& operator << (std::ostream& os, const VoxelHashMap::BrickPosition& bpos)
//{
//    os << bpos[0] << " " <<  bpos[1] << " " <<  bpos[2] << " " <<  endl;
//    return os;
//}
//
//std::ostream& operator << (std::ostream& os, const VoxelHashMap::BrickData& bdata)
//{
//    for (int i = 0; i < VoxelHashMap::kBrickSideLength; ++i)
//        for (int j = 0; j < VoxelHashMap::kBrickSideLength; ++j)
//            for (int k = 0; k < VoxelHashMap::kBrickSideLength; ++k)
//    {
//        os << bdata.voxeldatas[i][j][k] << endl;
//    }
//    return os;
//}
//
//std::ostream& operator << (std::ostream& os, const VoxelHashMap::voxel_hashing_type& map)
//{
//    size_t size = map.size(); 
//    os << size << endl;
//    for (VoxelHashMap::voxel_hashing_type_const_iterator citr = map.begin(); citr != map.end(); ++citr)
//    {
//        os << citr->first << endl;
//        os << citr->second << endl;
//    }
//    return os;
//}
//
//std::ostream& operator << (std::ostream& os, const VoxelHashMap& map)
//{
//    os << map.max_pt_[0] << " " <<  map.max_pt_[1] << " " <<  map.max_pt_[2] << " " <<  endl;
//    assert(os.good());
//    os << map.min_pt_[0] << " " <<  map.min_pt_[1] << " " <<  map.min_pt_[2] << " " <<  endl;
//    os << map.voxel_hash_map_ << endl;
//    return os;
//}
//
//std::istream& operator >> (std::istream& is, VoxelHashMap::BrickPosition& bpos)
//{
//    is >> bpos[0] >> bpos[1] >> bpos[2];
//    return is;
//}
//
//std::istream& operator >> (std::istream& is, VoxelHashMap::BrickData& bdata)
//{
//    for (int i = 0; i < VoxelHashMap::kBrickSideLength; ++i)
//        for (int j = 0; j < VoxelHashMap::kBrickSideLength; ++j)
//            for (int k = 0; k < VoxelHashMap::kBrickSideLength; ++k)
//    {
//        is >> bdata.voxeldatas[i][j][k];
//    }
//    return is;
//}
//
//std::istream& operator >> (std::istream& is, VoxelHashMap::voxel_hashing_type& map)
//{
//    size_t size = 0;
//    is >> size;
//
//    for (size_t i = 0; i != size; ++i) {
//        VoxelHashMap::voxel_hashing_type::key_type key;
//        VoxelHashMap::voxel_hashing_type::mapped_type value;
//        is >> key >> value;
//        map[key] = value;
//    }
//    return is;
//}
//
//std::istream& operator >> (std::istream& is, VoxelHashMap& map)
//{
//    assert(is.good());
//    is >> map.max_pt_[0] >> map.max_pt_[1] >> map.max_pt_[2];
//    assert(is.good());
//    is >> map.min_pt_[0] >> map.min_pt_[1] >> map.min_pt_[2];
//    is >> map.voxel_hash_map_;
//    map.DisplayHashMapInfo();
//    return is;
//}
}  // namespace cpu_tsdf
