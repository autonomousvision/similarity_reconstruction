add_library (marching_cubes STATIC
              marching_cubes_tsdf_hash.cpp
              marching_cubes_tsdf_hash.h)
target_link_libraries (marching_cubes tsdf_hash ${PCL_SURFACE_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS})
