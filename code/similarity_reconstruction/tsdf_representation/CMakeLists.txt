add_library (voxel_hashmap STATIC
              voxel_hashmap.cpp
              voxel_hashmap.h
              voxel_data.cpp
              voxel_data.h
              ${HASHMAP_UTILITY_DIR}/serialize_unordered_map.h)
add_library (tsdf_hash STATIC
              tsdf_hash.cpp
              tsdf_hash.h)
target_link_libraries (voxel_hashmap ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries (tsdf_hash voxel_hashmap ${OpenCV_LIBS} ${Boost_LIBRARIES})



