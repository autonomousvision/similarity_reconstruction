
add_library (svm_wrapper STATIC
              svm_wrapper.cpp
              svm_wrapper.h
              ../../third_party/libsvm/libsvm-3.20/svm.cpp
              ../../third_party/libsvm/libsvm-3.20/svm.h)

add_library (hashmap_utility STATIC
              utility.cpp
              utility.h)

add_library(oriented_boundingbox STATIC
	oriented_boundingbox.cpp
	oriented_boundingbox.h)

target_link_libraries(svm_wrapper)

target_link_libraries (hashmap_utility tsdf_hash ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

target_link_libraries (oriented_boundingbox ${OpenCV_LIBS} ${Boost_LIBRARIES})
