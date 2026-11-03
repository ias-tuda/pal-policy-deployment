cmake_minimum_required(VERSION 3.0.2)
project(tracking_ros)

# The tracker's service interface only; the tracker itself runs outside this workspace.
find_package(catkin REQUIRED COMPONENTS message_generation)

add_service_files(FILES SwitchTask.srv Reregister.srv)
generate_messages()

catkin_package(CATKIN_DEPENDS message_runtime)
