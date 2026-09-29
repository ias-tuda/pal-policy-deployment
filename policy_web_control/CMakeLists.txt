cmake_minimum_required(VERSION 3.0.2)
project(policy_web_control)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS scripts/policy_web_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
