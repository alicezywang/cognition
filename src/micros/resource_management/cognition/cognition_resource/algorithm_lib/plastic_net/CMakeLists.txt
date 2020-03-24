cmake_minimum_required(VERSION 2.8.3)
project(plastic_net)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  ml_model_base
  pluginlib
  roscpp
  rospy
)

find_package(OpenCV REQUIRED)

#add_message_files(
#   FILES
#   evidenceFromDataBase.msg
#)
#generate_messages(
# DEPENDENCIES
# geometry_msgs
#)
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
  LIBRARIES plastic_net
  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

## Declare a C++ library
#add_library(${PROJECT_NAME}
#  src/plastictrain.cpp
#)

include_directories(
  "include"
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/include
)
link_directories(${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/lib)

############
## output ##
############
add_library(${PROJECT_NAME}
  src/plastic_net_model.cpp
  src/plastic_network.cpp
  src/imgdata.cpp
)
target_link_libraries(${PROJECT_NAME}
  tensorflow
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

#############
## Testing ##
#############
add_executable(${PROJECT_NAME}_train
  test/train.cpp
)
target_link_libraries(${PROJECT_NAME}_train
  ${PROJECT_NAME}
)
add_executable(${PROJECT_NAME}_test
  test/batch_evaluate_test.cpp
)
target_link_libraries(${PROJECT_NAME}_test
  ${PROJECT_NAME}
)
add_executable(imgdata_test
  test/imgdata_test.cpp
)
target_link_libraries(imgdata_test
  ${PROJECT_NAME}
)

#############
## Install ##
#############
