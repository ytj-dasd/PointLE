cmake_minimum_required(VERSION 3.0.2)
project(line_ground_seg)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  common_zs
)
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

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
 LIBRARIES line_ground_seg
#  CATKIN_DEPENDS nav_msgs pcl_ros roscpp rospy sensors_msgs std_msgs
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
)

###########Find cpp###########
file(GLOB_RECURSE SRC src/*.cpp )

###########Build###########
add_library(line_ground_seg ${SRC})
target_link_libraries(line_ground_seg ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(${PROJECT_NAME}_node main.cpp )
target_link_libraries(${PROJECT_NAME}_node  line_ground_seg ${catkin_LIBRARIES} ${PCL_LIBRARIES})

###########Test###########
add_subdirectory(test)
