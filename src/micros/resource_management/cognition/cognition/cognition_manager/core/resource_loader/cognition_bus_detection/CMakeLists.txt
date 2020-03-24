cmake_minimum_required(VERSION 2.8.3)
project(cognition_bus_detection)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  cognition_bus
)
## System dependencies are found with CMake's conventions
# find_package(OpenCV REQUIRED)
# find_package(Boost REQUIRED)
# find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)   # Qt Environment

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  #CATKIN_DEPENDS
  #DEPENDS system_lib
)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations

## Sources
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

# 添加tensorflow头文件和库文件
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

############
## output ##
############
## Declare a C++ library
add_library(${PROJECT_NAME} ${DIR_SRCS})
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

