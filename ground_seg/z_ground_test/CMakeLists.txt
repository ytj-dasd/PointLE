cmake_minimum_required(VERSION 3.0.2)
project(z_ground_test)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
             nav_msgs
             pcl_ros
             roscpp
             rospy
             sensor_msgs
             std_msgs
             line_ground_seg
             mm_ground_seg
             gp_insac
             common_zs)

if(LINE_GROUND_SEG_FOUND)
    message(find line_ground_seg)
endif()


find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ground_test
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories( include  ${catkin_INCLUDE_DIRS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node test_extract.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME}_node_test_single test_single.cpp)
target_link_libraries(${PROJECT_NAME}_node_test_single
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME}_node_test test.cpp)
target_link_libraries(${PROJECT_NAME}_node_test
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME}_node_extract mm_ground_seg.cpp)
target_link_libraries(${PROJECT_NAME}_node_extract
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}
)



