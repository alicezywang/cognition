cmake_minimum_required(VERSION 2.8.3)
project(faster_rcnn)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2 -std=c++11)
#add_compile_options(-O2 -std=c++11 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

# 添加catkin工程组件
find_package(catkin REQUIRED COMPONENTS
  roscpp rospy
  roslib
  #lib
  utils
  networks
  roi_data_layer
  ml_model_base
)

## System dependencies are found with CMake's conventions
#find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS utils networks roi_data_layer
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
  ${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/include
)
link_directories(${CATKIN_DEVEL_PREFIX}/third_party/tensorflow/lib)


############
## output ##
############
## Declare a C++ library
add_library(${PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${PROJECT_NAME}
  tensorflow
  ${catkin_LIBRARIES}
)

#############
## Testing ##
#############
## Declare a C++ executable
add_executable(test_config test/config_test.cpp)
target_link_libraries(test_config ${catkin_LIBRARIES})

add_executable(test_bbox_transform test/bbox_transform_test.cpp)
target_link_libraries(test_bbox_transform ${catkin_LIBRARIES})

add_executable(test_test test/test_test.cpp)
target_link_libraries(test_test ${PROJECT_NAME})

add_executable(test_train test/test_train.cpp)
target_link_libraries(test_train ${PROJECT_NAME})

add_executable(demo test/demo.cpp)
target_link_libraries(demo ${PROJECT_NAME})

##muti_thread_test.cpp
#add_executable(muti_thread_test test/muti_thread_test.cpp)
#target_link_libraries(muti_thread_test ${PROJECT_NAME})

#############
##  Make   ##
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_fast_rcnn.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
