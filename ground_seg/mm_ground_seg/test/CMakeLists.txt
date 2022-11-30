
add_executable(${PROJECT_NAME}_node_test test.cpp)
target_link_libraries(${PROJECT_NAME}_node_test mm_ground_seg
  ${catkin_LIBRARIES} ${PCL_LIBRARIES}
)
