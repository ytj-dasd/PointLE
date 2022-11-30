cmake_minimum_required(VERSION 3.0.2)
project(mm_ground_seg)

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

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES mm_ground_seg
)


include_directories( include ${catkin_INCLUDE_DIRS})

file(GLOB_RECURSE SRC src/*.cpp)

add_library(mm_ground_seg  ${SRC})
target_link_libraries(mm_ground_seg ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS})

add_executable(${PROJECT_NAME}_node main.cpp)
target_link_libraries(${PROJECT_NAME}_node mm_ground_seg ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS})


add_subdirectory(test)
