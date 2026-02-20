cmake_minimum_required(VERSION 3.8)
project(pal_policy_deployer)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install(PROGRAMS
  scripts/pid_gain_override_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(
  INSTALL_TO_SHARE
  models
  launch
  config
)

