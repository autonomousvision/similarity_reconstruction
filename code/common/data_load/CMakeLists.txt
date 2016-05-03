set(DATA_LOAD_SOURCES urban_reconstruction_data_load.cpp urban_reconstruction_data_load.h)

add_library(data_load STATIC ${DATA_LOAD_SOURCES})
target_link_libraries(data_load rect_params boost_filesystem)



