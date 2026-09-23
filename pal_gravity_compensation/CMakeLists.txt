cmake_minimum_required(VERSION 3.0.2)
project(pal_gravity_compensation)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  kdl_parser
  roscpp
  sensor_msgs
  urdf
)
find_package(orocos_kdl REQUIRED)

catkin_package(CATKIN_DEPENDS kdl_parser roscpp sensor_msgs urdf
               DEPENDS orocos_kdl)

include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${orocos_kdl_INCLUDE_DIRS})

add_executable(gravity_compensation_node src/gravity_compensation_node.cpp)
target_link_libraries(gravity_compensation_node ${catkin_LIBRARIES} ${orocos_kdl_LIBRARIES})

install(TARGETS gravity_compensation_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
