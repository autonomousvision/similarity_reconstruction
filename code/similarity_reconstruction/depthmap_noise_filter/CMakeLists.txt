set(visibility_fusion_SOURCES depthmap_filter.cpp depthmap_filter.h)
add_library(depthmap_filter ${visibility_fusion_SOURCES})

add_executable(depthmap_filter_main depthmap_filter_main.cpp)
target_link_libraries(depthmap_filter_main depthmap_filter common_utility data_load rect_params ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS depthmap_filter_main DESTINATION bin)
