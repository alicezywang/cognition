cmake_minimum_required(VERSION 2.8.3)
project(rpn_msr)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2 -std=c++11)
#add_compile_options(-O2 -std=c++11 -DEIGEN_DONT_PARALLELIZE=1 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

#set(CUDA_NVCC_FLAGS "-O2 -std=c++11 --expt-relaxed-constexpr -DGOOGLE_CUDA=1 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp rospy
  #lib
  utils
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rpn_msr
  CATKIN_DEPENDS utils
  #DEPENDS system_lib
)

###########
## Build ##
###########
## Sources
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

## Specify additional locations of header files
## Your package locations should be listed before other locations
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
if(CUDA_NVCC_FLAGS)
  # Declare a GPU C++ library
  find_package(CUDA REQUIRED)
  add_compile_options(-DGOOGLE_CUDA=1)
  include_directories(${CUDA_INCLUDE_DIRS}) #添加CUDA库
  CUDA_ADD_LIBRARY(${PROJECT_NAME}
    ${DIR_SRCS}
    src/proposal_op_gpu.cu
  )
  target_link_libraries(${PROJECT_NAME}
    tensorflow
    ${CUDA_LIBRARIES}
    ${catkin_LIBRARIES}
  )
else()
  # Declare a CPU C++ library
  add_library(${PROJECT_NAME}
    ${DIR_SRCS}
  )
  target_link_libraries(${PROJECT_NAME}
    tensorflow
    ${catkin_LIBRARIES}
  )
endif()

#############
## Testing ##
#############
## Declare a C++ executable
add_executable(bbox_test
  test/bbox_test.cpp
  src/bbox.cpp
)
target_link_libraries(bbox_test
  ${catkin_LIBRARIES}
)

add_executable(generate_anchors_test
  test/generate_anchors_test.cpp
  src/generate_anchors.cpp
)
target_link_libraries(generate_anchors_test
  ${catkin_LIBRARIES}
)

add_executable(time_test
  src/time.cpp
  src/time_op.cpp
  test/time_test.cpp
)
target_link_libraries(time_test
  tensorflow
  ${catkin_LIBRARIES}
)

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_rpn_msr.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
