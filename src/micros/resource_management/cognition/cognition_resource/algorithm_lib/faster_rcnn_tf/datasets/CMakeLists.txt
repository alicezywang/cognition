cmake_minimum_required(VERSION 2.8.3)
project(datasets)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2 -std=c++11 -pthread)
#add_compile_options(-O2 -std=c++11 -pthread -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  utils
)

find_package(Boost REQUIRED COMPONENTS
  #  rege
  filesystem
  system
  serialization
)
# 添加cmake模块路径
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(SQLite3 REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include #必须,否则动态库找不到头文件
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS utils
#  DEPENDS system_lib
)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/include
)
link_directories(${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/lib)

# Sources
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

############
## output ##
############
add_library(${PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${SQLITE3_LIBRARIES}
)

#############
## Testing ##
#############
add_executable(data_test test/test.cpp)
target_link_libraries(data_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)
