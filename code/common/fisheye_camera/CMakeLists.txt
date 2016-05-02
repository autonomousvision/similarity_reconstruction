set(rect_params_SOURCES RectifiedCameraPair.cpp RectifiedCameraPair.h)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_library(rect_params STATIC ${rect_params_SOURCES})
target_link_libraries(rect_params common_utility)

