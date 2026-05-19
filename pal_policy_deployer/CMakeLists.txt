cmake_minimum_required(VERSION 3.0.2)
project(pal_policy_deployer)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  roscpp
  roslib
  sensor_msgs
  std_msgs
  tf
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
    geometry_msgs rospy roscpp sensor_msgs std_msgs tf message_runtime message_filters visualization_msgs interactive_markers
)

include_directories(${catkin_INCLUDE_DIRS})

foreach(dir config launch models)
    install(DIRECTORY ${dir}/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

catkin_install_python(PROGRAMS
  scripts/plot_obs.py
  scripts/preview_confirm_reaching_pose.py
  scripts/publish_reaching_pose_commands.py
  scripts/publish_reaching_sampling_boxes.py
  scripts/simple_left_reach_debug.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
