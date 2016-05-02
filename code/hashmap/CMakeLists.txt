cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
if(DEFINED CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
Release RelWithDebInfo MinSizeRel.")
else()
   set(CMAKE_BUILD_TYPE RelWithDebinfo CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
endif()

project(cpu_tsdf_hash)

# dealing with RPATH (http://www.cmake.org/Wiki/CMake_RPATH_handling) when installed.
# not important when runnning the program from the build tree
set(LIB_SUFFIX CACHE STRING "suffix for the library directory need for x86-64 systems that use lib64 ")
# Use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# The RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")

# third party libraries
find_package(PCL 1.7 REQUIRED COMPONENTS common io surface kdtree search segmentation)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#message(WARNING "${PCL_LIBRARIES}")
#message(WARNING "${PCL_COMMON_LIBRARIES}")
#message(WARNING "${PCL_SEGMENTATION_LIBRARIES}")
#message(WARNING "pcl_io_libraries: ${PCL_IO_LIBRARIES}")

# set boost include paths here manaully!
#set(BOOST_INCLUDEDIR "/usr/local/include")
#set(BOOST_LIBRARYDIR "/usr/local/lib")
#find_package(Boost 1.54 EXACT REQUIRED COMPONENTS *)
#find_package(Boost 1.54 EXACT REQUIRED COMPONENTS serialization program_options filesystem system)
unset(Boost_LIBRARIES)
#set(Boost_USE_STATIC_LIBS   ON)
#set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.5 COMPONENTS serialization program_options filesystem graph REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
message(WARNING "boost library list: ${Boost_LIBRARIES}")

FIND_PACKAGE(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIRS})

FIND_PACKAGE(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)
find_package(OpenCV 2.4 REQUIRED)
#message(WARNING "${OpenCV_LIBS}")

set(threadpool_DIR "/home/dell/codebase/lib/threadpool/threadpool-0_2_5-src/threadpool" CACHE PATH "The directory for threadpool library")
include_directories(${threadpool_DIR})

#set(libgrassmanzc_DIR "grassman_average/cpp_lib_zc" CACHE PATH "The directory for grassman average library")
#include_directories(${libgrassmanzc_DIR})

find_package(HDF5)
  if(HDF5_FOUND)
    message(STATUS "HDF5-dir: ${HDF5_INCLUDE_DIR}")
    #set (external_deps ${external_deps} "HDF5")
  endif()
 find_package(MATIO REQUIRED)
 if(MATIO_FOUND)
   message(STATUS "Matio-dir: ${MATIO_INCLUDE_DIRS}")
   INCLUDE_DIRECTORIES(${MATIO_INCLUDE_DIRS})
 endif()

set(Matlab_ROOT_DIR "/usr/local/MATLAB/R2015b")
find_package(MATLAB 8.5 EXACT REQUIRED COMPONENT MX_LIBRARY ENG_LIBRARY MEX_COMPILER)
INCLUDE_DIRECTORIES(${Matlab_INCLUDE_DIRS})
message(STATUS "Matlab_INCLUDE_DIRS: ${Matlab_INCLUDE_DIRS}")
###message(WARNING "${Matlab_LIBRARIES}")
###message(WARNING "${Matlab_MX_LIBRARY}")
###message(STATUS Matlab_FOUND)
set(Matlab_RT_LIBS  mwmclmcrrt)
LINK_DIRECTORIES("/usr/local/MATLAB/R2015b/runtime/glnxa64/")
##LINK_DIRECTORIES("/home/dell/codebase/mpi_project/urban_reconstruction/code/urban_reconstruction/hashmap/grassman_average/cpp_lib_zc")
###set(Matlab_BIN_LIBS /usr/local/MATLAB/R2015b/bin/glnxa64/libmwcpp11compat.so /usr/local/MATLAB/R2015b/bin/glnxa64/libmwlaunchermain.so)
###LINK_DIRECTORIES("/usr/local/MATLAB/R2015b/bin/glnxax64/")
### import related libraries
###add_library(libmwcpp11compat UNKNOWN IMPORTED)
###set_target_properties(libmwcpp11compat PROPERTIES IMPORTED_LOCATION /usr/local/MATLAB/R2015b/bin/glnxa64/libmwcpp11compat.so)
###add_library(libmwlaunchermain UNKNOWN IMPORTED)
###set_target_properties( libmwlaunchermain PROPERTIES IMPORTED_LOCATION /usr/local/MATLAB/R2015b/bin/glnxa64//libmwlaunchermain.so)

#find_package(MATIO REQUIRED)
#INCLUDE_DIRECTORIES(${MATIO_INCLUDE_DIRS})

# common libraries for data loading, fisheye cameras, utility
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(ROOT_SOURCE_DIR "../")
include_directories(${ROOT_SOURCE_DIR})

set(THIRD_PARTY_DIR ${ROOT_SOURCE_DIR}/third_party)
include_directories(${THIRD_PARTY_DIR})

set(COMMON_SOURCE_DIR "../common/")
include_directories(${COMMON_SOURCE_DIR})

set(HASHMAP_UTILITY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utility)
add_subdirectory(${HASHMAP_UTILITY_DIR} libhashmaputility${LIB_SUFFIX})

set(RECT_PARAM_SOURCE_DIR ${COMMON_SOURCE_DIR}fisheye_camera)
add_subdirectory(${RECT_PARAM_SOURCE_DIR} librectparam${LIB_SUFFIX})

set(DATA_LOAD_SOURCE_DIR ${COMMON_SOURCE_DIR}data_load)
add_subdirectory(${DATA_LOAD_SOURCE_DIR} libdataload${LIB_SUFFIX})

set(COMMON_UTILITY_SOURCE_DIR ${COMMON_SOURCE_DIR}utility)
add_subdirectory(${COMMON_UTILITY_SOURCE_DIR} libcommonutility${LIB_SUFFIX})

set(DEPTHMAP_FUSION_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/depthmap-fusion)
add_subdirectory(${DEPTHMAP_FUSION_SOURCE_DIR} libdepthmapfusion${LIB_SUFFIX})

set(depthmap_triangularize_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/depthmap_triangularize)
add_subdirectory(${depthmap_triangularize_SOURCE_DIR} libdepthmaptriangularize${LIB_SUFFIX})

set(detection_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/detection)
add_subdirectory(${detection_SOURCE_DIR} libdetection${LIB_SUFFIX})

set(consistency_check_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/consistency_check)
add_subdirectory(${consistency_check_SOURCE_DIR} libconsistency_check${LIB_SUFFIX})

#set(libgrassmanzc_SOURCE_DIR "grassman_average/cpp_lib_zc" CACHE PATH "The directory for grassman average library")
#include_directories(${libgrassmanzc_SOURCE_DIR})
#add_subdirectory(${libgrassmanzc_SOURCE_DIR} libgrassmanzc${LIB_SUFFIX})


# tsdf libraries
add_library(tsdf_joint_optimization STATIC
        tsdf_operation/tsdf_joint_optimization.cpp
        tsdf_operation/tsdf_joint_optimization.h)

add_library(tsdf_utility STATIC
        tsdf_operation/tsdf_utility.cpp
        tsdf_operation/tsdf_utility.h)

add_library (tsdf_transform STATIC
              tsdf_operation/tsdf_transform.cpp
              tsdf_operation/tsdf_transform.h)

add_library (tsdf_pca STATIC
              tsdf_operation/tsdf_pca.cpp
              tsdf_operation/tsdf_pca.h)

add_library (tsdf_clean STATIC
              tsdf_operation/tsdf_clean.cpp
              tsdf_operation/tsdf_clean.h)

add_library (tsdf_io STATIC
              tsdf_operation/tsdf_io.cpp
              tsdf_operation/tsdf_io.h)

#add_library (svm_wrapper STATIC
#              utility/svm_wrapper.cpp
#              utility/svm_wrapper.h
#             ../third_party/libsvm/libsvm-3.20/svm.cpp
#              ../third_party/libsvm/libsvm-3.20/svm.h)

add_library (tsdf_feature_generate STATIC
              tsdf_operation/tsdf_feature_generate.cpp
              tsdf_operation/tsdf_feature_generate.h)
#add_library (hashmap_utility STATIC
#              utility/utility.cpp
#              utility/utility.h
#              macros.h)
add_library (voxel_hashmap STATIC
              tsdf_representation/voxel_hashmap.cpp
              tsdf_representation/voxel_data.cpp
              tsdf_representation/voxel_data.h
              tsdf_representation/voxel_hashmap.h
              utility/serialize_unordered_map.h)
add_library (tsdf_hash STATIC
              tsdf_representation/tsdf_hash.cpp
              tsdf_representation/tsdf_hash.h)
add_library (marching_cubes STATIC
              marching_cubes/marching_cubes_tsdf_hash.cpp
              marching_cubes/marching_cubes_tsdf_hash.h)
add_library (tsdf_smooth STATIC
              tsdf_operation/tsdf_smooth.cpp
              tsdf_operation/tsdf_smooth.h
            )
add_library (diffusion_hole_filling STATIC
              tsdf_operation/diffusion_hole_filling.cpp
              tsdf_operation/diffusion_hole_filling.h
            )
add_library (vrl_representation_lib STATIC
              vrl_to_hash/vrl_grid_representation/ChunkAllocator.cc  
              vrl_to_hash/vrl_grid_representation/ChunkAllocator.h  
              vrl_to_hash/vrl_grid_representation/ChunkLink.cc  
              vrl_to_hash/vrl_grid_representation/ChunkLink.h   
              vrl_to_hash/vrl_grid_representation/defines.h  
              vrl_to_hash/vrl_grid_representation/OccGrid.cc  
              vrl_to_hash/vrl_grid_representation/OccGrid.h  
              vrl_to_hash/vrl_grid_representation/OccGridRLE.cc  
              vrl_to_hash/vrl_grid_representation/OccGridRLE.h  
              vrl_to_hash/vrl_grid_representation/vrip.h
            )
add_library (convert_vrl_to_hash STATIC
              vrl_to_hash/convert_vrl_to_hash.cpp
              vrl_to_hash/convert_vrl_to_hash.h
            )
add_library (ply2vri_lib STATIC
              vrl_to_hash/ply2vri/trimesh.cpp
              vrl_to_hash/ply2vri/vec.h
              vrl_to_hash/ply2vri/ply2vri.cpp
              vrl_to_hash/ply2vri/ply2vri.h
              vrl_to_hash/ply2vri/ply.h
              vrl_to_hash/ply2vri/plyfile.c
              vrl_to_hash/ply2vri/mat.h
              vrl_to_hash/ply2vri/trimesh.h
           )
add_library (tsdf_align STATIC
              tsdf_operation/tsdf_align.cpp
              tsdf_operation/tsdf_align.h
            )
add_library (tsdf_slice STATIC
              tsdf_operation/tsdf_slice.cpp
              tsdf_operation/tsdf_slice.h
            )
add_library (tsdf_joint_align STATIC
              tsdf_operation/tsdf_joint_align.cpp
              tsdf_operation/tsdf_joint_align.h
            )

target_link_libraries(tsdf_joint_optimization tsdf_align tsdf_pca tsdf_clean tsdf_io hashmap_utility common_utility tsdf_hash voxel_hashmap ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(tsdf_clean marching_cubes tsdf_hash hashmap_utility ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(svm_wrapper)
target_link_libraries(tsdf_io oriented_boundingbox marching_cubes tsdf_hash tsdf_slice tsdf_joint_align tsdf_align hashmap_utility ${MATIO_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(tsdf_feature_generate detector svm_wrapper tsdf_hash tsdf_slice hashmap_utility ${OpenCV_LIBS} ${Boost_LIBRARIES})
#target_link_libraries (hashmap_utility tsdf_hash ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries (voxel_hashmap ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries (tsdf_hash voxel_hashmap ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries (marching_cubes tsdf_hash ${PCL_SURFACE_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries (tsdf_smooth tsdf_hash ${OpenCV_LIBS})
target_link_libraries (diffusion_hole_filling tsdf_smooth tsdf_hash ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries (vrl_representation_lib)
target_link_libraries (convert_vrl_to_hash vrl_representation_lib tsdf_hash ${OpenCV_LIBS})
target_link_libraries (ply2vri_lib vrl_representation_lib)
target_link_libraries (tsdf_pca tsdf_transform tsdf_io ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries (tsdf_align tsdf_pca tsdf_hash voxel_hashmap tsdf_clean tsdf_transform ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIBRARIES})
target_link_libraries (tsdf_slice hashmap_utility tsdf_hash voxel_hashmap convert_vrl_to_hash vrl_representation_lib ply2vri_lib ${OpenCV_LIBS})
target_link_libraries (tsdf_joint_align tsdf_pca tsdf_clean tsdf_io hashmap_utility common_utility tsdf_hash voxel_hashmap ${OpenCV_LIBS} ${Boost_LIBRARIES})


# executables
#add_executable(test_pca tsdf_operation/test/test_pca.cpp)
#target_link_libraries(test_pca  tsdf_joint_align tsdf_clean detector tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_pca DESTINATION bin)

#add_executable(test_robust_loss tsdf_operation/test/test_robust_loss.cpp)
#target_link_libraries(test_robust_loss  tsdf_joint_align detector tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_robust_loss DESTINATION bin)

#add_executable(test_merge_tsdf_multi_category tsdf_operation/test/test_merge_tsdf_multi_category.cpp)
#target_link_libraries(test_merge_tsdf_multi_category  detector tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_merge_tsdf_multi_category DESTINATION bin)

#add_executable(test_adjust_detection2 tsdf_operation/test/test_adjust_detection2.cpp)
#target_link_libraries(test_adjust_detection2  detector tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_adjust_detection2 DESTINATION bin)
#
#add_executable(test_adjust_detection tsdf_operation/test/test_adjust_detection.cpp)
#target_link_libraries(test_adjust_detection  detector tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_adjust_detection DESTINATION bin)

add_executable (crop_training_tsdf tsdf_operation/test/crop_training_tsdf.cpp)
target_link_libraries(crop_training_tsdf data_load tsdf_io convert_vrl_to_hash marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS crop_training_tsdf DESTINATION bin)

add_executable (merge_tsdfs_obbs tsdf_operation/test/merge_tsdfs_obbs.cpp)
target_link_libraries(merge_tsdfs_obbs data_load tsdf_io convert_vrl_to_hash marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS merge_tsdfs_obbs DESTINATION bin)

add_executable (test_merge_vri vrl_to_hash/test_merge_vri.cpp)
target_link_libraries(test_merge_vri data_load tsdf_io convert_vrl_to_hash marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS test_merge_vri DESTINATION bin)

#add_executable(test_sliding_window_object_detector2 tsdf_operation/test/test_sliding_window_object_detector2.cpp)
#target_link_libraries(test_sliding_window_object_detector2  detector tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_sliding_window_object_detector2 DESTINATION bin)

#add_executable(test_train_object_detector2 tsdf_operation/test/test_train_object_detector2.cpp)
#target_link_libraries(test_train_object_detector2 detector tsdf_feature_generate tsdf_io tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_train_object_detector2 DESTINATION bin)
#
#add_executable(test_read_from_xml tsdf_operation/test/test_read_from_xml.cpp)
#target_link_libraries(test_read_from_xml  tsdf_io tsdf_joint_align tsdf_align tsdf_slice tsdf_hash hashmap_utility common_utility data_load label_data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_read_from_xml DESTINATION bin)

#add_executable(test_opt_transform tsdf_operation/test/test_opt_transform.cpp)
#target_link_libraries(test_opt_transform  tsdf_io tsdf_joint_align tsdf_align tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_opt_transform DESTINATION bin)

#add_executable(test_joint_cluster_model tsdf_operation/test/test_joint_cluster_model.cpp)
##target_link_libraries(test_joint_cluster_model libgrassmanzc_cpp ${Matlab_RT_LIBS}  libmwcpp11compat libmwlaunchermain tsdf_joint_align tsdf_pca detector tsdf_io tsdf_joint_align tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
##target_link_libraries(test_joint_cluster_model libgrassmanzc_cpp ${Matlab_RT_LIBS} ${Matlab_BIN_LIBS}  tsdf_joint_align tsdf_pca detector tsdf_io tsdf_joint_align tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
##target_link_libraries(test_joint_cluster_model libgrassmanzc_cpp grassmanzc ${Matlab_RT_LIBS}  tsdf_joint_align tsdf_pca detector tsdf_io tsdf_joint_align tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#target_link_libraries(test_joint_cluster_model oriented_boundingbox tsdf_joint_align tsdf_pca detector tsdf_io tsdf_joint_align tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_joint_cluster_model DESTINATION bin)

add_executable(test_tsdf_optimization tsdf_operation/test/test_tsdf_optimization.cpp)
target_link_libraries(test_tsdf_optimization tsdf_joint_optimization tsdf_align tsdf_utility oriented_boundingbox tsdf_joint_align tsdf_pca detector tsdf_io tsdf_joint_align tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS test_tsdf_optimization DESTINATION bin)

#add_executable(compute_precision_recall_curve tsdf_operation/test/compute_precision_recall_curve.cpp)
#target_link_libraries(compute_precision_recall_curve  detector tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS compute_precision_recall_curve DESTINATION bin)

#add_executable(test_sliding_window_object_detector tsdf_operation/test/test_sliding_window_object_detector.cpp)
#target_link_libraries(test_sliding_window_object_detector  tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_sliding_window_object_detector DESTINATION bin)
#
#add_executable(test_train_object_detector tsdf_operation/test/test_train_object_detector.cpp)
#target_link_libraries(test_train_object_detector  tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_train_object_detector DESTINATION bin)

#add_executable(test_convert_obb_color tsdf_operation/test/test_convert_obb_color.cpp)
#target_link_libraries(test_convert_obb_color tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_convert_obb_color DESTINATION bin)

#add_executable(test_tsdf_feature_generate tsdf_operation/test/test_tsdf_feature_generate.cpp)
#target_link_libraries(test_tsdf_feature_generate  tsdf_io tsdf_feature_generate tsdf_hash hashmap_utility common_utility data_load marching_cubes ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_tsdf_feature_generate DESTINATION bin)

add_executable(visibility-fusion-main depthmap-fusion/visibility-fusion-main.cpp)
target_link_libraries(visibility-fusion-main visibility-fusion common_utility data_load rect_params ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS visibility-fusion-main DESTINATION bin)

#add_executable (test_tsdf_pca_coefficient tsdf_operation/test/test_tsdf_pca_coefficient.cpp)
#target_link_libraries(test_tsdf_pca_coefficient tsdf_io common_utility tsdf_joint_align hashmap_utility tsdf_align marching_cubes ${PCL_IO_LIBRARIES} data_load ${OpenCV_LIBS} ${Boost_LIBRARIES})
#install(TARGETS test_tsdf_pca_coefficient DESTINATION bin)

#add_executable (test_joint_align_pca tsdf_operation/test/test_joint_align_pca.cpp)
#target_link_libraries(test_joint_align_pca tsdf_io common_utility tsdf_joint_align hashmap_utility tsdf_align marching_cubes ${PCL_IO_LIBRARIES} data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
#install(TARGETS test_joint_align_pca DESTINATION bin)

#add_executable (test_tsdf_semantic_slice tsdf_operation/test/test_tsdf_semantic_slice.cpp)
#target_link_libraries(test_tsdf_semantic_slice tsdf_io tsdf_slice hashmap_utility common_utility marching_cubes tsdf_hash ${PCL_IO_LIBRARIES} data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
#install(TARGETS test_tsdf_semantic_slice DESTINATION bin)

#add_executable (test_tsdf_slice tsdf_operation/test/test_tsdf_slice.cpp)
#target_link_libraries(test_tsdf_slice tsdf_slice tsdf_io hashmap_utility common_utility marching_cubes tsdf_hash ${PCL_IO_LIBRARIES} data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
#install(TARGETS test_tsdf_slice DESTINATION bin)

#add_executable (test_joint_align tsdf_operation/test/test_joint_align.cpp)
#target_link_libraries(test_joint_align tsdf_io common_utility tsdf_joint_align hashmap_utility tsdf_align marching_cubes ${PCL_IO_LIBRARIES} data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
#install(TARGETS test_joint_align DESTINATION bin)

#add_executable (test_tsdf_align_automatic tsdf_operation/test/test_tsdf_align_automatic.cpp)
#target_link_libraries(test_tsdf_align_automatic tsdf_io tsdf_align marching_cubes hashmap_utility common_utility ${PCL_IO_LIBRARIES} data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
#install(TARGETS test_tsdf_align_automatic  DESTINATION bin)

#add_executable (test_tsdf_align tsdf_operation/test/test_tsdf_align.cpp)
#target_link_libraries(test_tsdf_align tsdf_align tsdf_io common_utility marching_cubes hashmap_utility ${PCL_IO_LIBRARIES} data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
#install(TARGETS test_tsdf_align  DESTINATION bin)

add_executable (test_vri_hash_tsdf_conversion vrl_to_hash/test_vri_hash_tsdf_conversion.cpp)
target_link_libraries(test_vri_hash_tsdf_conversion tsdf_io convert_vrl_to_hash marching_cubes tsdf_hash voxel_hashmap hashmap_utility common_utility ${PCL_IO_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS test_vri_hash_tsdf_conversion DESTINATION bin)

add_executable (fisheye_depths_fusion fisheye_depths_fusion.cpp)
target_link_libraries(fisheye_depths_fusion tsdf_io diffusion_hole_filling tsdf_smooth hashmap_utility common_utility ${PCL_IO_LIBRARIES} marching_cubes tsdf_hash voxel_hashmap data_load rect_params ${Boost_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS fisheye_depths_fusion DESTINATION bin)

add_executable(pca_tsdf_visualization tsdf_operation/test/pca_tsdf_visualization.cpp)
target_link_libraries(pca_tsdf_visualization tsdf_joint_align tsdf_pca detector tsdf_io tsdf_joint_align tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${Glog_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS pca_tsdf_visualization DESTINATION bin)

add_executable(tsdf_visualization tsdf_operation/test/tsdf_visualization.cpp)
target_link_libraries(tsdf_visualization tsdf_joint_align tsdf_pca detector tsdf_io tsdf_joint_align tsdf_align tsdf_pca tsdf_slice tsdf_hash hashmap_utility common_utility data_load marching_cubes ${Glog_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
install(TARGETS tsdf_visualization DESTINATION bin)
