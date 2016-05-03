add_library (obb_intersection STATIC
             obb_intersection.cpp
             obb_intersection.h
	     2dobb.cpp
	     2dobb.h)
target_link_libraries(obb_intersection oriented_boundingbox hashmap_utility ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_library (detect_sample STATIC
             detect_sample.cpp
             detect_sample.h)
target_link_libraries(detect_sample oriented_boundingbox detection_utility marching_cubes tsdf_hash tsdf_slice tsdf_align hashmap_utility ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_library (detection_utility STATIC
             detection_utility.cpp
             detection_utility.h)
target_link_libraries(detection_utility detect_sample obb_intersection marching_cubes tsdf_hash tsdf_slice tsdf_align hashmap_utility ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_library (detector STATIC
             detector.cpp
             detector.h)
target_link_libraries(detector detection_utility svm_wrapper obb_intersection oriented_boundingbox detect_sample detection_utility marching_cubes tsdf_hash tsdf_slice tsdf_align hashmap_utility ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(train_detectors_main train_detectors_main.cpp)
target_link_libraries(train_detectors_main detector detect_sample detection_utility tsdf_io tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS train_detectors_main DESTINATION bin)

add_executable(detect_main detect_main.cpp)
target_link_libraries(detect_main detector detect_sample detection_utility tsdf_io tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS detect_main DESTINATION bin)




