set(DATA_LOAD_SOURCES urban_reconstruction_data_load.cpp urban_reconstruction_data_load.h)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(data_load STATIC ${DATA_LOAD_SOURCES})
target_link_libraries(data_load rect_params boost_filesystem)

add_library(label_data_load STATIC label_data_load.cpp label_data_load.h)
target_link_libraries(label_data_load ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES})


