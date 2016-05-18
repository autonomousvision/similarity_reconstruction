/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
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

}  // namespace cpu_tsdf
