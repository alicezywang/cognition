cmake_minimum_required(VERSION 2.8.3)
project(roi_pooling_layer)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-O2 -std=c++11)
#add_compile_options(-O2 -std=c++11 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)

#set(CUDA_NVCC_FLAGS "-O2 -std=c++11 -DGOOGLE_CUDA=1 -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

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
    ## Declare a GPU C++ library
    find_package(CUDA REQUIRED)
    add_compile_options(-DGOOGLE_CUDA=1)
    include_directories(${CUDA_INCLUDE_DIRS}) #添加CUDA库
    CUDA_ADD_LIBRARY(${PROJECT_NAME}
      src/roi_pool.cpp
      src/roi_pool_op.cpp
      src/roi_pool_grad.cpp
      src/roi_pool_op_gpu.cu
    )
    target_link_libraries(${PROJECT_NAME}
      ${CUDA_LIBRARIES}
      tensorflow
    )
else()
    ## Declare a CPU C++ library
    add_library(${PROJECT_NAME}
      src/roi_pool.cpp
      src/roi_pool_op.cpp
      src/roi_pool_grad.cpp
    )
    target_link_libraries(${PROJECT_NAME}
      tensorflow
    )
endif()

#############
## Testing ##
#############
## Declare a C++ executable
add_executable(test_${PROJECT_NAME}
  test/roi_pool_test.cpp
)

target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/roi_pooling_layer_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
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
# catkin_add_gtest(test_${PROJECT_NAME} test/main.cpp)
# if(TARGET test_${PROJECT_NAME})
#   target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
