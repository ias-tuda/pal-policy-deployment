/*
 * Copyright 2020 PAL Robotics SL. All Rights Reserved
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited,
 * unless it was supplied under the terms of a license agreement or
 * nondisclosure agreement with PAL Robotics SL. In this case it may not be
 * copied or disclosed except in accordance with the terms of that agreement.
 */
#ifndef SUBSCRIBER_CONTROLLER_H
#define SUBSCRIBER_CONTROLLER_H

#include <nav_msgs/Odometry.h>
#include <pal_base_ros_controller/base_robot_with_estimator_controller.h>
#include <pluginlib/class_list_macros.h>
#include <realtime_tools/realtime_publisher.h>
#include <sensor_msgs/JointState.h>
#include <pluginlib/class_list_macros.hpp>
#include <unordered_map>
#include <control_toolbox/pid.h>

namespace subscriber_controller
{
class SubscriberController : public pal_base_ros_controller::BaseRobotWithEsimatorController
{
public:
  // Load controller (NON RT)
  bool loadEtras(ros::NodeHandle &control_nh) override;

  // Control loop (RT)
  void updateExtras(const ros::Time &time, const ros::Duration &period) override;

  // First time control loop (RT)
  void startingExtras(const ros::Time &time) override;

  // Last control loop after stopping controller (RT)
  void stoppingExtra(const ros::Time &time) override;

private:
  // Subscriber CB
  void controlCb(const sensor_msgs::JointStateConstPtr &desired_state);

  // Subscriber
  ros::Subscriber state_sub_;
  // Mutex
  std::timed_mutex mutex_;

  sensor_msgs::JointState desired_state_;
  std::vector<double> initial_torque_;
  std::vector<double> initial_position_;
  std::vector<double> desired_torque_;

  // Impedance gains
  std::vector<control_toolbox::Pid> gains_;

  // Real time publishers
  boost::shared_ptr<realtime_tools::RealtimePublisher<sensor_msgs::JointState>> joint_states_pub_;
  boost::shared_ptr<realtime_tools::RealtimePublisher<nav_msgs::Odometry>> base_state_pub_;

  // ROS msgs
  sensor_msgs::JointState actual_js_state_;
  nav_msgs::Odometry actual_base_state_;

  // Registration variables
  pal_statistics::RegistrationsRAII registered_variables_;

  // Time profiler
  pal_robot_tools::TimeProfilerPtr tp_;

  int ms_mutex_;

  Eigen::Vector3d base_linear_vel_;
  Eigen::Vector3d base_angular_vel_;
  Eigen::Quaterniond base_orientation_;
  Eigen::Vector3d base_position_;
};
}

PLUGINLIB_EXPORT_CLASS(subscriber_controller::SubscriberController,
                       controller_interface::ControllerBase);

#endif  // SUBSCRIBER_CONTROLLER_H
