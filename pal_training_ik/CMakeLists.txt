cmake_minimum_required(VERSION 3.0.2)
project(pal_training_ik)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  kdl_parser
  roscpp
  sensor_msgs
  std_msgs
  urdf
)
find_package(Eigen3 REQUIRED NO_MODULE)

catkin_package(CATKIN_DEPENDS kdl_parser roscpp sensor_msgs std_msgs urdf)

include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_executable(training_ik_node src/training_ik_node.cpp)
target_link_libraries(training_ik_node ${catkin_LIBRARIES})

install(TARGETS training_ik_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
