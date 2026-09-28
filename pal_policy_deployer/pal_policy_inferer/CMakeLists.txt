cmake_minimum_required(VERSION 3.0.2)
project(pal_policy_inferer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  message_filters
  visualization_msgs
  interactive_markers
  pal_statistics
  kdl_parser
  tf2_ros
  sensor_msgs
  urdf
)

find_package(ZLIB REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    roscpp std_msgs sensor_msgs message_filters visualization_msgs interactive_markers
    kdl_parser urdf tf2_ros
)
find_package(Eigen3 REQUIRED NO_MODULE)

########################
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} /opt/onnxruntime/include)
include_directories(
  include
)
link_directories(/opt/onnxruntime/lib)
########################

add_executable(${PROJECT_NAME}_mjlab_node src/kang_mjlab_node.cpp)
target_include_directories(${PROJECT_NAME}_mjlab_node PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_mjlab_node
  ${ZLIB_LIBRARIES} 
  ${catkin_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  onnxruntime
  yaml-cpp
)

add_executable(${PROJECT_NAME}_ilab_node_v21 src/kang_ilab_node_v21.cpp)
target_link_libraries(${PROJECT_NAME}_ilab_node_v21
  ${catkin_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  onnxruntime
)

install(TARGETS ${PROJECT_NAME}_mjlab_node ${PROJECT_NAME}_ilab_node_v21
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
