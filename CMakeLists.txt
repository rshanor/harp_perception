cmake_minimum_required(VERSION 2.8.3)
project(harp_perception)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)  

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  interactive_markers
  joy
  pcl_conversions
  pcl_ros
  pluginlib
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  harp_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS system_lib
)

include_directories(include 
  ${catkin_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

add_library(perception_tools src/perception_tools.cpp)
target_link_libraries(perception_tools ${roscpp_LIBRARIES})

add_library(depth_smoother src/depth_smoother.cpp)
target_link_libraries(depth_smoother ${roscpp_LIBRARIES} ${catkin_LIBRARIES} )

add_library(geometry_filter src/geometry_filter.cpp)
target_link_libraries(geometry_filter ${roscpp_LIBRARIES} ${catkin_LIBRARIES} )

add_executable(geometry_filter_server src/geometry_filter_server.cpp)
target_link_libraries(geometry_filter_server ${catkin_LIBRARIES} perception_tools depth_smoother geometry_filter)

#target_link_libraries(run_move_arm_server ${catkin_LIBRARIES})
###add_dependencies(run_move_arm_server ${harp_arm_EXPORTED_TARGETS} )
