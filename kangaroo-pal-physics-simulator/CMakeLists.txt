cmake_minimum_required(VERSION 3.0.2)
project(kangaroo_pal_physics_simulator)

find_package(catkin REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kangaroo_pal_physics_simulator
#  CATKIN_DEPENDS kangaroo_controller_configuration kangaroo_description pal_physics_simulator
#  DEPENDS system_lib
)

include_directories(
# include
#  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(DIRECTORY launch config
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
