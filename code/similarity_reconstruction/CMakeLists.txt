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

project(similarity_reconstruction)

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
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

set(ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")
set(THIRD_PARTY_DIR "${ROOT_SOURCE_DIR}third_party/")
include_directories(${ROOT_SOURCE_DIR})
include_directories(${THIRD_PARTY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# find third party library dependencies
find_package(PCL 1.7 REQUIRED COMPONENTS common io surface kdtree search segmentation visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Boost 1.5 COMPONENTS serialization program_options filesystem graph REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)
find_package(OpenCV 2.4 REQUIRED)

FIND_PACKAGE(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIRS})

FIND_PACKAGE(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(threadpool_DIR  "${THIRD_PARTY_DIR}/threadpool-0_2_5-src/threadpool" CACHE PATH "The directory for threadpool library")
include_directories(${threadpool_DIR})

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
set(Matlab_RT_LIBS  mwmclmcrrt)
LINK_DIRECTORIES("/usr/local/MATLAB/R2015b/runtime/glnxa64/")

###################
# add sub-directories
# common libraries for data loading, fisheye cameras, utility
set(COMMON_SOURCE_DIR "${ROOT_SOURCE_DIR}/common/")
include_directories(${COMMON_SOURCE_DIR})

set(RECT_PARAM_SOURCE_DIR ${COMMON_SOURCE_DIR}fisheye_camera)
add_subdirectory(${RECT_PARAM_SOURCE_DIR} librectparam${LIB_SUFFIX})

set(DATA_LOAD_SOURCE_DIR ${COMMON_SOURCE_DIR}data_load)
add_subdirectory(${DATA_LOAD_SOURCE_DIR} libdataload${LIB_SUFFIX})

set(COMMON_UTILITY_SOURCE_DIR ${COMMON_SOURCE_DIR}/utilities)
add_subdirectory(${COMMON_UTILITY_SOURCE_DIR} libcommonutilities${LIB_SUFFIX})

set(HASHMAP_UTILITY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tsdf_hash_utilities)
add_subdirectory(${HASHMAP_UTILITY_DIR} libtsdfhashutility${LIB_SUFFIX})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tsdf_representation libtsdfrepresentation${LIB_SUFFIX})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes libmarchingcubes${LIB_SUFFIX})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tsdf_operation libtsdfoperation${LIB_SUFFIX})

set(CONSISTENCY_CHECK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/consistency_check)
add_subdirectory(${CONSISTENCY_CHECK_DIR} libconsistencycheck${LIB_SUFFIX})

set(DEPTHMAP_FUSION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/depthmap_fusion)
add_subdirectory(${DEPTHMAP_FUSION_DIR} libdepthmapfusion${LIB_SUFFIX})

set(DEPTHMAP_NOISE_FILTER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/depthmap_noise_filter)
add_subdirectory(${DEPTHMAP_NOISE_FILTER_DIR} libdepthmapnoisefilter${LIB_SUFFIX})

set(DETECTION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/detection)
add_subdirectory(${DETECTION_DIR} libdetection${LIB_SUFFIX})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/joint_reconstruction libjointreconstruction${LIB_SUFFIX})


