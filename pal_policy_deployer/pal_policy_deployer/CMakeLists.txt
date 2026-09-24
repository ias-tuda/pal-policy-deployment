cmake_minimum_required(VERSION 3.0.2)
project(pal_policy_deployer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  message_generation
  message_filters
  visualization_msgs
  interactive_markers
)

add_message_files(FILES
  StampedInt8Array.msg
  StampedFloat64Array.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS
    roscpp std_msgs message_runtime message_filters visualization_msgs interactive_markers
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/velocity_command.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Keep this script's Python 3 shebang on ROS Melodic (catkin uses Python 2 here).
install(PROGRAMS scripts/move_to_start_config.py scripts/spawn_grasp_controller.py
  scripts/check_start_distribution.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

foreach(dir config launch models)
    install(DIRECTORY ${dir}/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
