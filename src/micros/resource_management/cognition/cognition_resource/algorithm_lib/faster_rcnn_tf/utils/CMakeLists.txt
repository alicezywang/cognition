cmake_minimum_required(VERSION 2.8.3)
project(utils)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2 -std=c++11)
#add_compile_options(-O2 -std=c++11 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

#set(CUDA_NVCC_FLAGS "-O2 -std=c++11 -DGOOGLE_CUDA=1 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
)

find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  #CATKIN_DEPENDS roscpp rospy
  #DEPENDS system_lib
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

## Sources
AUX_SOURCE_DIRECTORY(src/utils DIR_SRCS)
AUX_SOURCE_DIRECTORY(src/nms DIR_SRCS)
AUX_SOURCE_DIRECTORY(src/fast_rcnn DIR_SRCS)

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
      src/nms/nms_kernel.cu
    )
    target_link_libraries(${PROJECT_NAME}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
    )
else()
    # Declare a CPU C++ library
    add_library(${PROJECT_NAME}
      ${DIR_SRCS}
    )
    target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
    )
endif()

#############
## Testing ##
#############
# Declare a C++ executable
## utils
add_executable(boxs_grid test/utils/boxes_grid_test.cpp)
target_link_libraries(boxs_grid
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

add_executable(timer test/utils/timer_test.cpp)
target_link_libraries(timer
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

add_executable(im test/utils/plot_test.cpp)
target_link_libraries(im
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

## nms
add_executable(cpu_nms test/nms/cpu_nms_test.cpp)
target_link_libraries(cpu_nms
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

## fast_rcnn
add_executable(config_test test/fast_rcnn/config_test.cpp)
target_link_libraries(config_test
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

add_executable(bbox_transform_test test/fast_rcnn/bbox_transform_test.cpp)
target_link_libraries(bbox_transform_test
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}
)

add_executable(useEigen test/fast_rcnn/useEigen.cpp)
