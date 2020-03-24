cmake_minimum_required(VERSION 2.8.3)
project(mobilenet_ssd_tensorrt)


## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  ml_model_base
)
set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_50,code=compute_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
  -gencode arch=compute_62,code=sm_62
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


file(GLOB sources src/*.cpp src/tensorrt/*.cu src/tensorrt/*.cpp src/tensorrt/util/*.cpp src/tensorrt/util/cuda/*.cu)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ml_model_base
  DEPENDS  system_lib OpenCV  CUDA
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/third_party/tensorrt/include
)
link_directories(${CATKIN_DEVEL_PREFIX}/third_party/tensorrt/lib)
#link_directories("/usr/lib/x86_64-linux-gnu/")

## Declare a C++ library
cuda_add_library(${PROJECT_NAME} SHARED ${sources})
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  nvcaffe_parser
  nvinfer
  nvinfer_plugin
  nvparsers
)

#############
## Testing ##
#############
add_executable(test_once_${PROJECT_NAME} test/mobilenet_ssd_test_once.cpp)
target_link_libraries(test_once_${PROJECT_NAME}
  ${PROJECT_NAME}
)

add_executable(test_batch_${PROJECT_NAME} test/mobilenet_ssd_test_batch.cpp)
target_link_libraries(test_batch_${PROJECT_NAME}
  ${PROJECT_NAME}
)

#############
## Install ##
#############
#install(TARGETS ${PROJECT_NAME}
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
##   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

### Mark cpp header files for installation
#install(DIRECTORY include/${PROJECT_NAME}/
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#  FILES_MATCHING PATTERN "*.h"
#  PATTERN ".svn" EXCLUDE
#)

###########
### Test ##
###########

### mobilenet test
#add_executable(${PROJECT_NAME}_test
#  test/mobilenet_ssd_test.cpp
#)

#target_link_libraries(${PROJECT_NAME}_test
#  ${PROJECT_NAME}
#)

##############
### Install ##
##############
## Mark executables and/or libraries for installation

## Mark cpp header files for installation
#install(DIRECTORY include/${PROJECT_NAME}/
  #DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  #FILES_MATCHING PATTERN "*.h"
  #PATTERN ".svn" EXCLUDE
#)

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
