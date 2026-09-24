cmake_minimum_required(VERSION 3.0.2)
project(pal_policy_preprocessor)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_filters
  visualization_msgs
  interactive_markers
)

find_package(Eigen3 REQUIRED NO_MODULE)

catkin_package(
  CATKIN_DEPENDS
    roscpp rospy std_msgs sensor_msgs geometry_msgs nav_msgs message_filters visualization_msgs interactive_markers
)

########################
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} /opt/onnxruntime/include)
include_directories(
  include
)
link_directories(/opt/onnxruntime/lib)

########################
add_executable(interactive_marker src/interactive_marker_node.cpp)
target_link_libraries(interactive_marker ${catkin_LIBRARIES} ${Boost_LIBRARIES})

add_executable(joy_twist_accumulator src/joy_twist_accumulator.cpp)
target_link_libraries(joy_twist_accumulator ${catkin_LIBRARIES})
add_dependencies(joy_twist_accumulator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

##############
## Python scripts
catkin_install_python(PROGRAMS
  scripts/sysid_data_collector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/inekf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

##############
## Mark executables for installation
install(TARGETS interactive_marker joy_twist_accumulator
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
