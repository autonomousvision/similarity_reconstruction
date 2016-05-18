set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${COMMON_UTILITY_SOURCE_DIR})
add_library (common_utility STATIC
              eigen_utility.cpp
              eigen_utility.h
              pcl_utility.cpp
              pcl_utility.h
              common_utility.h
              timer.h
              #matlab_utility.h
              #matlab_utility.cpp
            )
target_link_libraries(common_utility ${Matlab_RT_LIBS} ${MATIO_LIBRARIES} ${GLOG_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} ${PCL_SEARCH_LIBRARIES} ${PCL_KDTREE_LIBRARIES} ${PCL_SURFACE_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )

