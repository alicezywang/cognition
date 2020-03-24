cmake_minimum_required(VERSION 2.8.3)
project(plugin_test_softbus)

## Add support for C++11, supported in ROS Kinetic and newer
 add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  general_bus
  roscpp
  std_msgs
  actor_core
)
## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)   # Qt Environment

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES test_general_bus
#  CATKIN_DEPENDS general_bus general_plugin roscpp std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
#  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
## Specify libraries to link a library or executable target against
QT5_WRAP_CPP(QT_MOC_TEST_H src/ros_thread.h)
add_executable(${PROJECT_NAME}_node  ${QT_MOC_TEST_H} src/general_bus_node.cpp src/ros_thread.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  Qt5::Core Qt5::Gui Qt5::Widgets
)





