add_library (vrl_representation_lib STATIC
              vrl_grid_representation/ChunkAllocator.cc  
              vrl_grid_representation/ChunkAllocator.h  
              vrl_grid_representation/ChunkLink.cc  
              vrl_grid_representation/ChunkLink.h   
              vrl_grid_representation/defines.h  
              vrl_grid_representation/OccGrid.cc  
              vrl_grid_representation/OccGrid.h  
              vrl_grid_representation/OccGridRLE.cc  
              vrl_grid_representation/OccGridRLE.h  
              vrl_grid_representation/vrip.h
            )
add_library (convert_vrl_to_hash STATIC
              convert_vrl_to_hash.cpp
              convert_vrl_to_hash.h
            )
add_library (ply2vri_lib STATIC
              ply2vri/trimesh.cpp
              ply2vri/vec.h
              ply2vri/ply2vri.cpp
              ply2vri/ply2vri.h
              ply2vri/ply.h
              ply2vri/plyfile.c
              ply2vri/mat.h
              ply2vri/trimesh.h
           )

target_link_libraries (vrl_representation_lib)
target_link_libraries (convert_vrl_to_hash vrl_representation_lib tsdf_hash ${OpenCV_LIBS})
target_link_libraries (ply2vri_lib vrl_representation_lib)

add_executable (test_merge_vri test_merge_vri.cpp)
target_link_libraries(test_merge_vri data_load tsdf_io convert_vrl_to_hash marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS test_merge_vri DESTINATION bin)

add_executable (test_vri_hash_tsdf_conversion test_vri_hash_tsdf_conversion.cpp)
target_link_libraries(test_vri_hash_tsdf_conversion tsdf_io convert_vrl_to_hash marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS test_vri_hash_tsdf_conversion DESTINATION bin)
