# tsdf libraries

add_library(tsdf_utility STATIC
        tsdf_utility.cpp
        tsdf_utility.h)

add_library (tsdf_transform STATIC
              tsdf_transform.cpp
              tsdf_transform.h)

add_library (tsdf_pca STATIC
              tsdf_pca.cpp
              tsdf_pca.h)

add_library (tsdf_clean STATIC
              tsdf_clean.cpp
              tsdf_clean.h)

add_library (tsdf_io STATIC
              tsdf_io.cpp
              tsdf_io.h)

add_library (tsdf_smooth STATIC
              tsdf_smooth.cpp
              tsdf_smooth.h
            )

add_library (diffusion_hole_filling STATIC
              diffusion_hole_filling.cpp
              diffusion_hole_filling.h
            )

add_library (tsdf_align STATIC
              tsdf_align.cpp
              tsdf_align.h
            )
add_library (tsdf_slice STATIC
              tsdf_slice.cpp
              tsdf_slice.h
            )

target_link_libraries(tsdf_clean marching_cubes tsdf_hash hashmap_utility ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(tsdf_io oriented_boundingbox marching_cubes tsdf_hash tsdf_slice tsdf_align hashmap_utility ${MATIO_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries (tsdf_smooth tsdf_hash ${OpenCV_LIBS})
target_link_libraries (diffusion_hole_filling tsdf_smooth tsdf_hash ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries (tsdf_pca tsdf_transform tsdf_io ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries (tsdf_align tsdf_pca tsdf_hash voxel_hashmap tsdf_clean tsdf_transform ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIBRARIES})
target_link_libraries (tsdf_slice hashmap_utility tsdf_hash voxel_hashmap convert_vrl_to_hash vrl_representation_lib ${OpenCV_LIBS})
