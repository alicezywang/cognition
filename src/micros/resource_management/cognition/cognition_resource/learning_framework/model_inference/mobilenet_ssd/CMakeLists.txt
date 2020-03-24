cmake_minimum_required(VERSION 2.8.3)
project(mobilenet_ssd)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  ml_model_base
  ml_database
  pluginlib
  roscpp
  rospy
)

set(OpenCV_DIR /usr/local/include)
find_package(OpenCV 3.4.7 EXACT)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES mobilenet_ssd
  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  "include"
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/include
)
link_directories(${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/lib)

## Declare a C++ library
##mobilenet
add_library(${PROJECT_NAME}
  src/mobilenet_ssd_model.cpp
  src/mobilenet_ssd_dnn_model.cpp
  src/pb_to_tf_network.cpp
  src/pb_to_dnn_network.cpp
)

target_link_libraries(${PROJECT_NAME}
  tensorflow
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)

##########
## Test ##
##########

## mobilenet test
add_executable(${PROJECT_NAME}_test
  test/mobilenet_ssd_test.cpp
)

target_link_libraries(${PROJECT_NAME}_test
  ${PROJECT_NAME}
)
