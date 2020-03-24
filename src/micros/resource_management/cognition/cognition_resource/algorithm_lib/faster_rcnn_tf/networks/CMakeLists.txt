cmake_minimum_required(VERSION 2.8.3)
project(networks)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2 -std=c++11)
#add_compile_options(-O2 -std=c++11 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

## Find catkin macros and libraries if
## COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz) is used,
## also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp rospy
  rpn_msr
  roi_pooling_layer
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

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
  CATKIN_DEPENDS rpn_msr roi_pooling_layer
#  DEPENDS system_lib
)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations

## Sources
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

# 添加include目录、Eigen头文件和tensorflow头文件
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/include
)
link_directories(${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/lib)

############
## output ##
############
add_library(${PROJECT_NAME}
  src/network.cpp
  src/vggnet_train.cpp
  src/vggnet_test.cpp
)
target_link_libraries(${PROJECT_NAME}
  tensorflow
  ${catkin_LIBRARIES}
)

#############
## Testing ##
#############
# test_vggnet_train
add_executable(test_vggnet_train test/test_vggnet_train.cpp)
target_link_libraries(test_vggnet_train
  ${PROJECT_NAME}
)

# test_vggnet_test
add_executable(test_vggnet_test test/test_vggnet_test.cpp)
target_link_libraries(test_vggnet_test
  ${PROJECT_NAME}
)
