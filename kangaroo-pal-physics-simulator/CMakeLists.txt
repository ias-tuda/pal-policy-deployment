cmake_minimum_required(VERSION 3.0.2)
project(kangaroo_pal_physics_simulator)

find_package(catkin REQUIRED
             roscpp
             sensor_msgs
             pal_utils
             pal_ariles
             kangaroo_leg_specifics)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kangaroo_pal_physics_simulator
#  CATKIN_DEPENDS kangaroo_controller_configuration kangaroo_description pal_physics_simulator
#  DEPENDS system_lib
)

include_directories(
  SYSTEM
  ${catkin_INCLUDE_DIRS}
)

add_executable(extended_joint_states_publisher_node src/extended_joint_states_publisher_node.cpp)
target_link_libraries(extended_joint_states_publisher_node ${catkin_LIBRARIES})

#############
## Install ##
#############

install(DIRECTORY launch config
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
