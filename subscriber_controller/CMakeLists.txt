cmake_minimum_required(VERSION 2.8.3)
project(subscriber_controller)

SET(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

set(CATKIN_COMPONENTS
	pal_base_ros_controller 
	pluginlib 
	controller_interface
	ddynamic_reconfigure
	pal_statistics
	sensor_msgs
	realtime_tools
        nav_msgs
        tf2
        tf2_ros
        rbdl)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_COMPONENTS})

catkin_package(
      INCLUDE_DIRS include
      LIBRARIES ${PROJECT_NAME}
      CATKIN_DEPENDS ${CATKIN_COMPONENTS}
    #  DEPENDS system_lib
    )

include_directories(
    include
    )

include_directories(
    SYSTEM
    ${catkin_INCLUDE_DIRS}
    )

add_library(${PROJECT_NAME} 
    src/subscriber_controller.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(trajectory_publisher src/trajectory_publisher.cpp)
target_link_libraries(trajectory_publisher ${PROJECT_NAME} ${catkin_LIBRARIES})


install(TARGETS ${PROJECT_NAME} trajectory_publisher
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

install(DIRECTORY include/
    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
    )
install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

install(DIRECTORY config/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)

install(FILES controller_plugins.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(PROGRAMS
    scripts/hold_position.py
    scripts/sim_policy.py
    scripts/publish_current_as_desired.py
    scripts/publish_current_arms_as_desired.py
    scripts/publish_arm_target_as_desired.py
    scripts/publish_current_legs_as_desired.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
