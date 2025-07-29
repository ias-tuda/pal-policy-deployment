/*
 * Copyright 2020 PAL Robotics SL. All Rights Reserved
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited,
 * unless it was supplied under the terms of a license agreement or
 * nondisclosure agreement with PAL Robotics SL. In this case it may not be
 * copied or disclosed except in accordance with the terms of that agreement.
 */

#include <subscriber_controller/subscriber_controller.h>

namespace subscriber_controller
{
bool SubscriberController::loadEtras(ros::NodeHandle &control_nh)
{
  // Create real time publishers to publish the actial state
  ros::TransportHints hints;
  hints.tcpNoDelay(true);
  joint_states_pub_.reset(new realtime_tools::RealtimePublisher<sensor_msgs::JointState>(
      control_nh, "actual_js_state", 1));
  base_state_pub_.reset(new realtime_tools::RealtimePublisher<nav_msgs::Odometry>(
      control_nh, "actual_base_state", 1));

  // Create a simple subscriber (there are no RT subscribers)
  state_sub_ =
      control_nh.subscribe("desired_state", 1, &SubscriberController::controlCb, this, hints);

  actual_js_state_.name = getControlledJointNames();

  // Read the initial torques from the sensors
  ros::Time start_time = ros::Time::now();
  size_t samples = 0;
  std::vector<double> a_initial_torque;
  std::vector<double> a_initial_position;
  a_initial_torque.resize(actual_js_state_.name.size());
  a_initial_position.resize(actual_js_state_.name.size());

  while ((ros::Time::now() - start_time) < ros::Duration(1.0))
  {
    for (size_t i = 0; i < actual_js_state_.name.size(); i++)
    {
      a_initial_torque[i] += getJointMeasuredTorque(i);
      a_initial_position[i] += getActualJointPosition(i);
    }
    samples++;
    ros::Duration(0.01).sleep();
  }

  for (size_t i = 0; i < actual_js_state_.name.size(); i++)
  {
    initial_torque_.push_back(a_initial_torque[i] / double(samples));
    initial_position_.push_back(a_initial_position[i] / double(samples));
    desired_torque_.push_back(initial_torque_[i]);
  }

  // Create TimeProfiler and register a new timer called subcriber. One
  // TimeProfiler could have several timers.
  tp_.reset(new pal_robot_tools::TimeProfiler);
  tp_->registerTimer("subscriber");

  // Register variables in the introspection data topic
  REGISTER_VARIABLE("/introspection_data", "subscriber_time",
                    tp_->getLastCycleTimePtr("subscriber"), &registered_variables_);
  REGISTER_VARIABLE("/introspection_data", "subscriber_periodicity",
                    tp_->getPeriodicityPtr("subscriber"), &registered_variables_);

  // Mutex time
  ms_mutex_ = getControllerDt().toSec() * 1000;

  std::string gains_ns = "gains";
  XmlRpc::XmlRpcValue gains_param;

  if (!control_nh.getParam(gains_ns, gains_param)) {
      ROS_ERROR("Failed to get 'gains' parameter.");
      return false;
  }

  if (gains_param.getType() != XmlRpc::XmlRpcValue::TypeStruct) {
      ROS_ERROR("'gains' parameter is not a struct.");
      return false;
  }

  PAL_ASSERT_PERSIST_EQUAL(gains_param.size(), actual_js_state_.name.size());
  gains_.resize(actual_js_state_.name.size());

  for (auto it = gains_param.begin(); it != gains_param.end(); ++it) {
      std::string joint_name = it->first;
      std::string joint_ns = gains_ns + "/" + joint_name;
      control_toolbox::Pid pid;
      if (pid.init(ros::NodeHandle(control_nh, joint_ns))) {
          auto it2 = std::find(actual_js_state_.name.begin(), actual_js_state_.name.end(), joint_name);
          if(it2 == actual_js_state_.name.end())
          {
            ROS_ERROR_STREAM("Couldn't find " << joint_name << " in controlled joint names");
            return false;
          }
          size_t i = std::distance(actual_js_state_.name.begin(), it2);
          gains_[i] = pid;
          ROS_INFO_STREAM("Loaded PID for joint: " << joint_name);
      } else {
          ROS_ERROR_STREAM("Failed to load PID for joint: " << joint_name);
          return false;
      }
  }

  return true;
}

void SubscriberController::updateExtras(const ros::Time &time, const ros::Duration &period)
{
  // Read actual values
  actual_js_state_.position = getActualJointPositions();
  actual_js_state_.velocity = getActualJointVelocities();
  actual_js_state_.effort = desired_torque_;
  base_linear_vel_ = getEstimatedFloatingBaseLinearVelocity();
  base_angular_vel_ = getEstimatedFloatingBaseAngularVelocity();
  base_orientation_ = getEstimatedFloatingBaseOrientation();
  base_position_ = getEstimatedFloatingBasePosition();

  // Convert them to ROS msgs
  // Convert eigen vector to geometry_msgs Vector3
  actual_base_state_.twist.twist.linear.x = base_linear_vel_(0);
  actual_base_state_.twist.twist.linear.y = base_linear_vel_(1);
  actual_base_state_.twist.twist.linear.z = base_linear_vel_(2);
  actual_base_state_.twist.twist.angular.x = base_angular_vel_(0);
  actual_base_state_.twist.twist.angular.y = base_angular_vel_(1);
  actual_base_state_.twist.twist.angular.z = base_angular_vel_(2);
  // Convert eigen vector to the geometry_msgs Point
  actual_base_state_.pose.pose.position.x = base_position_(0);
  actual_base_state_.pose.pose.position.y = base_position_(1);
  actual_base_state_.pose.pose.position.z = base_position_(2);
  // Convert eigen quaternion to the geometry_msgs Quaternion
  actual_base_state_.pose.pose.orientation.x = base_orientation_.x();
  actual_base_state_.pose.pose.orientation.y = base_orientation_.y();
  actual_base_state_.pose.pose.orientation.z = base_orientation_.z();
  actual_base_state_.pose.pose.orientation.w = base_orientation_.w();

  // Publish values using the RT publisher
  if (joint_states_pub_->trylock())
  {
    actual_js_state_.header.stamp = time;
    joint_states_pub_->msg_ = actual_js_state_;
    joint_states_pub_->unlockAndPublish();
  }
  if (base_state_pub_->trylock())
  {
    actual_base_state_.header.stamp = time;
    base_state_pub_->msg_ = actual_base_state_;
    base_state_pub_->unlockAndPublish();
  }

  // Try to lock the desired state using a mutex.
  std::unique_lock<std::timed_mutex> lock(mutex_, std::try_to_lock);
  // If locked and desired state received
  if (lock.owns_lock() && (!desired_state_.name.empty()))
  {
    for (size_t i = 1; i < desired_state_.name.size(); i++)
    {
      setDesiredJointState(desired_state_.name[i], desired_state_.position[i],
                           desired_state_.velocity[i], 0., desired_state_.effort[i]);
    }
    // If desired state haven't been received yet mantain the initial position
    // using impedance
  }
  else if (desired_state_.name.empty())
  {
    for (size_t i = 0; i < actual_js_state_.name.size(); i++)
    {
      desired_torque_[i] = initial_torque_[i] +
              gains_[i].computeCommand(initial_position_[i] - getActualJointPosition(i),
                                       -getActualJointVelocity(i), period);

     setDesiredJointState(actual_js_state_.name[i], initial_position_[i], 0., 0.,
                          desired_torque_[i]);
    }
  }
  else
  {
    ROS_WARN("Skipping received state");
  }
}

void SubscriberController::startingExtras(const ros::Time &time)
{
}

void SubscriberController::stoppingExtra(const ros::Time &time)
{
  for (size_t i = 0; i < getControlledJointNames().size(); ++i)
  {
    setDesiredJointState(i, getActualJointPosition(i), 0., 0., 0.);
  }
}

void SubscriberController::controlCb(const sensor_msgs::JointStateConstPtr &desired_state)
{
  // Start subscriber timer
  tp_->startTimer("subscriber");
  // Try to lock desired_state recursively using mutex to set the desired state
  std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
  if (lock.try_lock_for(std::chrono::milliseconds(ms_mutex_)))
  {
    desired_state_ = *desired_state;
  }
  // Stop subsciber timer
  tp_->stopTime("subscriber");
}
}
