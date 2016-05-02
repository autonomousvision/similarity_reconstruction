add_library (depthmap_triangle STATIC
              depthmap.cpp
	      depthmap.h
            )

target_link_libraries(depthmap_triangle hashmap_utility data_load common_utility ${PCL_IO_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS})

add_executable(depthmap_triangle_mesh_main depthmap_triangle_mesh_main.cpp)
target_link_libraries(depthmap_triangle_mesh_main depthmap_triangle ${PCL_IO_LIBRARIES}  ${Boost_LIBRARIES})
install(TARGETS depthmap_triangle_mesh_main DESTINATION bin)

