cmake_minimum_required(VERSION 2.8.3)
project(tensorrt_cognition_plugin)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2 -std=c++11)

# 添加catkin工程组件
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  pluginlib
  orient_softbus_msgs
  #lib
  actor_core
  general_plugin
  cognition_dev
  cognition_bus
  model_recommend
)

## System dependencies are found with CMake's conventions
#find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS cognition_bus cognition_dev model_recommend
#  DEPENDS system_lib
)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
# Sources
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
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

## Declare a C++ executable

#############
## Testing ##
#############








