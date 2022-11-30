cmake_minimum_required(VERSION 2.8.3)
project(gp_insac)

SET(CMAKE_BUILD_TYPE "Release")  
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")  

## Compile as C++11
# SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++14)
# supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  nav_msgs
  common_zs
)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES gp_insac
#  CATKIN_DEPENDS pcl_ros roscpp rospy sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

# find .h files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# find .cpp files 
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ground_extraction/ GROUND_SRCS)

# message(${GROUND_SRCS})

###### build ########
add_library(gp_insac ${GROUND_SRCS})
target_link_libraries(gp_insac ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(${PROJECT_NAME}_node main.cpp)
target_link_libraries(${PROJECT_NAME}_node  gp_insac ${catkin_LIBRARIES} ${PCL_LIBRARIES})

###### test ########
add_subdirectory(test)
