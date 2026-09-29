cmake_minimum_required(VERSION 3.0.2)
project(kangaroo_camera_calibration)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/board_detector.py
  scripts/pose_teacher.py
  scripts/collect.py
  scripts/solve.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config launch rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
