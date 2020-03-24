cmake_minimum_required(VERSION 2.8.3)
project(model_recommend)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  "include"
#  "~/SPTAG/AnnService"
#  "~/SPTAG/Wrappers"
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/recommend.cpp
)

target_link_libraries(${PROJECT_NAME}
  tinyxml2
  #/usr/local/lib/_SPTAG.so
  ${catkin_LIBRARIES}
)

add_executable(recommend_test
  test/recommend_test.cpp
)

target_link_libraries(recommend_test
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
