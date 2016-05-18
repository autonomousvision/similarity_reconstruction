add_library (consistency_check STATIC
              consistency_check.cpp
	      consistency_check.h
            )

target_link_libraries(consistency_check data_load tsdf_io marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(test_consistency_check test_consistency_check.cpp)
#target_link_libraries(test_consistency_check consistency_check detector data_load tsdf_io convert_vrl_to_hash marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(test_consistency_check consistency_check detector data_load tsdf_io marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS test_consistency_check DESTINATION bin)

