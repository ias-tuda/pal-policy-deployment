cmake_minimum_required(VERSION 3.0.2)
project(pal_policy_inferer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rosbag
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_filters
  visualization_msgs
  interactive_markers
  tf
)

catkin_package(
  CATKIN_DEPENDS
    roscpp rosbag std_msgs sensor_msgs geometry_msgs nav_msgs message_filters visualization_msgs interactive_markers
)

find_package(Eigen3 REQUIRED NO_MODULE)

########################
# ONNX Runtime
########################

# Pfad zu ONNX Runtime (aus Dockerfile)
set(ONNXRUNTIME_DIR /opt/onnxruntime)

include_directories(
  SYSTEM
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ONNXRUNTIME_DIR}/include
  include
)

link_directories(${ONNXRUNTIME_DIR}/lib)

########################
# Executables
########################

add_executable(${PROJECT_NAME}_ee_reaching_node src/ee_reaching_node.cpp)

add_executable(${PROJECT_NAME}_ee_hold_default_node src/ee_hold_default_node.cpp)
target_link_libraries(${PROJECT_NAME}_ee_hold_default_node
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)
target_link_libraries(${PROJECT_NAME}_ee_reaching_node
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
  ${TF_LIBRARIES}
)

add_executable(${PROJECT_NAME}_mjlab_node src/kang_mjlab_node.cpp)
target_link_libraries(${PROJECT_NAME}_mjlab_node
  ${catkin_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

add_executable(${PROJECT_NAME}_mjlab_bag_replay_node src/kang_mjlab_bag_replay_node.cpp)
target_link_libraries(${PROJECT_NAME}_mjlab_bag_replay_node
  ${catkin_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

add_executable(${PROJECT_NAME}_mjlab_noise_node src/kang_mjlab_noise_node.cpp)
target_link_libraries(${PROJECT_NAME}_mjlab_noise_node
  ${catkin_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

add_executable(${PROJECT_NAME}_ilab_node_v21 src/kang_ilab_node_v21.cpp)
target_link_libraries(${PROJECT_NAME}_ilab_node_v21
  ${catkin_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

########################
# Install
########################

install(TARGETS
  ${PROJECT_NAME}_ee_reaching_node
  ${PROJECT_NAME}_ee_hold_default_node
  ${PROJECT_NAME}_mjlab_node
  ${PROJECT_NAME}_mjlab_bag_replay_node
  ${PROJECT_NAME}_mjlab_noise_node
  ${PROJECT_NAME}_ilab_node_v21
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
