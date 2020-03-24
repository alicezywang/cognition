cmake_minimum_required(VERSION 2.8.3)
project(resource_collection)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
#    class_loader
#    rosconsole
    roslib
#    cmake_modules
)
#find_package(Boost REQUIRED COMPONENTS filesystem system)
#find_package(TinyXML2 REQUIRED)

#catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
#  CATKIN_DEPENDS class_loader rosconsole roslib
#  DEPENDS Boost TinyXML2
)

#include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${TinyXML2_INCLUDE_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(test_collection test/test_collection.cpp)
target_link_libraries(test_collection
    ${catkin_LIBRARIES}
#    ${Boost_LIBRARIES} ${TinyXML2_LIBRARIES}
)

install(TARGETS test_collection
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(DIRECTORY include/cognition/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

message(WARNING "~~~~~~~~~BIN = ${CATKIN_PACKAGE_BIN_DESTINATION}")
message(WARNING "~~~~~~~~~INCLUDE = ${CATKIN_PACKAGE_INCLUDE_DESTINATION}")
message(WARNING "~~~~~~~~~LIB = ${CATKIN_PACKAGE_LIB_DESTINATION}")
message(WARNING "~~~~~~~~~SHARE = ${CATKIN_PACKAGE_SHARE_DESTINATION}")
message(WARNING "~~~~~~~~~BIN = ${CATKIN_DEVEL_PREFIX}")
message(WARNING "~~~~~~~~~BIN = ${CMAKE_INSTALL_PREFIX}")


