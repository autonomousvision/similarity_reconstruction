add_library(tsdf_joint_optimization STATIC
        tsdf_joint_optimization.cpp
        tsdf_joint_optimization.h)
target_link_libraries(tsdf_joint_optimization tsdf_align tsdf_pca tsdf_clean tsdf_io hashmap_utility common_utility tsdf_hash voxel_hashmap ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(test_tsdf_optimization test_tsdf_optimization.cpp)
target_link_libraries(test_tsdf_optimization tsdf_joint_optimization tsdf_align tsdf_utility oriented_boundingbox tsdf_pca detector tsdf_io tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS test_tsdf_optimization DESTINATION bin)
