cmake_minimum_required(VERSION 2.8.3)
project(crowd_evaluate)

# Find catkin
find_package(catkin REQUIRED)

catkin_package(
    LIBRARIES ${PROJECT_NAME}
    INCLUDE_DIRS include
)

## Sources
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
set(serial_SRCS
     ${DIR_SRCS}
)

## Add serial library
add_library(${PROJECT_NAME} ${serial_SRCS})

target_link_libraries(${PROJECT_NAME} ${catkin_lIBRARIES})


## Uncomment for example
# add_executable(serial_example examples/serial_example.cc)
# add_dependencies(serial_example ${PROJECT_NAME})
# target_link_libraries(serial_example ${PROJECT_NAME})

## Include headers
include_directories(include)
