add_subdirectory(depthmap_to_mesh libdepthmaptomesh${LIB_SUFFIX})
add_subdirectory(vrl_to_hash libvrltohash${LIB_SUFFIX})

add_executable (depth_fusion_approx_main depth_fusion_approx_main.cpp)
target_link_libraries(depth_fusion_approx_main tsdf_io diffusion_hole_filling tsdf_smooth hashmap_utility common_utility ${PCL_IO_LIBRARIES} marching_cubes tsdf_hash voxel_hashmap data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS depth_fusion_approx_main DESTINATION bin)
