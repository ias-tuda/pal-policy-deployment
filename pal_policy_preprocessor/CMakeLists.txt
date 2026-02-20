cmake_minimum_required(VERSION 3.8)
project(pal_policy_preprocessor)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_executable(${PROJECT_NAME}_kangaroo_locomotion
  src/preprocessor.cpp
  src/preprocessor_kangaroo_locomotion.cpp
)

ament_auto_add_executable(${PROJECT_NAME}_tiago_reaching
  src/preprocessor.cpp
  src/preprocessor_tiago_reaching.cpp
)

ament_auto_add_executable(interactive_marker_node
  src/interactive_marker_node.cpp
)

ament_auto_package(
  INSTALL_TO_SHARE
  config
)

