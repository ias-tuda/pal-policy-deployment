cmake_minimum_required(VERSION 3.0.2)
project(voice_bridge)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

# Keep the scripts' Python 3 shebang on ROS Melodic (catkin uses Python 2 here), as pal_policy_deployer does.
install(PROGRAMS scripts/voice_bridge_node.py scripts/mock_supervisor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
