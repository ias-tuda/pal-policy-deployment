/*
 * Copyright 2020 PAL Robotics SL. All Rights Reserved
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited,
 * unless it was supplied under the terms of a license agreement or
 * nondisclosure agreement with PAL Robotics SL. In this case it may not be
 * copied or disclosed except in accordance with the terms of that agreement.
 */

#include <subscriber_controller/subscriber_controller.h>

#include <algorithm>
#include <cmath>

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

  ROS_WARN_STREAM("subscriber_controller initial_position (held until first desired_state):");
  for (size_t i = 0; i < actual_js_state_.name.size(); i++)
    ROS_WARN_STREAM("  [" << i << "] " << actual_js_state_.name[i]
                    << "  q0=" << initial_position_[i]
                    << "  tau0=" << initial_torque_[i]);

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

  control_nh.param("debug_desired_state", debug_desired_state_, false);
  control_nh.param<std::string>("debug_joint_name", debug_joint_name_, "");
  control_nh.param<std::string>("debug_joint_prefix", debug_joint_prefix_, "");
  control_nh.param("debug_log_period", debug_log_period_, 1.0);
  if (debug_desired_state_)
  {
    ROS_WARN_STREAM("subscriber_controller desired_state debug enabled"
                    << " joint='" << (debug_joint_name_.empty() ? "<first received>" : debug_joint_name_)
                    << "' prefix='" << debug_joint_prefix_
                    << "' period=" << debug_log_period_ << "s"
                    << " controlled_joints=" << actual_js_state_.name.size());
  }

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

  // Publish at 100 Hz (every 20 cycles at 2000 Hz)
  if (++publish_counter_ % 10 == 0)
  {
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
  }

  // Try to lock the desired state using a mutex.
  std::unique_lock<std::timed_mutex> lock(mutex_, std::try_to_lock);
  // If locked and desired state received
  if (lock.owns_lock() && (!desired_state_.name.empty()))
  {
    if (!first_desired_state_applied_)
    {
      first_desired_state_applied_ = true;
      ROS_WARN_STREAM("subscriber_controller ACTIVE: first desired_state applied at ros_time="
                      << time.toSec()
                      << " msg_count=" << desired_messages_received_
                      << " joints=" << desired_state_.name.size());
      ROS_WARN_STREAM("subscriber_controller FIRST CMD vs ACTUAL (jump at policy start):");
      for (size_t i = 0; i < desired_state_.name.size(); i++)
      {
        auto actual_it = std::find(actual_js_state_.name.begin(),
                                   actual_js_state_.name.end(),
                                   desired_state_.name[i]);
        double q_actual = (actual_it != actual_js_state_.name.end())
            ? actual_js_state_.position[std::distance(actual_js_state_.name.begin(), actual_it)]
            : 0.0;
        double q_des = (i < desired_state_.position.size()) ? desired_state_.position[i] : 0.0;
        double jump = q_des - q_actual;
        ROS_WARN_STREAM("  " << desired_state_.name[i]
                        << "  q_actual=" << q_actual
                        << "  q_des=" << q_des
                        << "  jump=" << jump
                        << (std::fabs(jump) > 0.05 ? "  *** LARGE ***" : ""));
      }
    }

    for (size_t i = 0; i < desired_state_.name.size(); i++)
    {
      setDesiredJointState(desired_state_.name[i], desired_state_.position[i],
                           desired_state_.velocity[i], 0., desired_state_.effort[i]);
    }

    if (debug_desired_state_ &&
        (last_debug_log_time_.isZero() ||
         (time - last_debug_log_time_).toSec() >= debug_log_period_))
    {
      auto log_joint = [&](const std::string& joint_name) {
        auto desired_it = std::find(desired_state_.name.begin(), desired_state_.name.end(), joint_name);
        auto actual_it = std::find(actual_js_state_.name.begin(), actual_js_state_.name.end(), joint_name);
        if (desired_it == desired_state_.name.end())
        {
          ROS_WARN_STREAM("subscriber_controller debug: joint '" << joint_name
                          << "' not present in desired_state. desired names="
                          << desired_state_.name.size() << " controlled names="
                          << actual_js_state_.name.size());
          return;
        }
        if (actual_it == actual_js_state_.name.end())
        {
          ROS_WARN_STREAM("subscriber_controller debug: joint '" << joint_name
                          << "' present in desired_state but not controlled by subscriber_controller");
          return;
        }

        const size_t desired_idx = std::distance(desired_state_.name.begin(), desired_it);
        const size_t actual_idx = std::distance(actual_js_state_.name.begin(), actual_it);
        if (desired_idx < desired_state_.position.size() &&
            desired_idx < desired_state_.velocity.size() &&
            desired_idx < desired_state_.effort.size() &&
            actual_idx < actual_js_state_.position.size() &&
            actual_idx < actual_js_state_.velocity.size())
        {
          const double q_des = desired_state_.position[desired_idx];
          const double dq_des = desired_state_.velocity[desired_idx];
          const double tau_ff = desired_state_.effort[desired_idx];
          const double q_actual = actual_js_state_.position[actual_idx];
          const double dq_actual = actual_js_state_.velocity[actual_idx];
          ROS_WARN_STREAM("subscriber_controller debug joint='" << joint_name
                          << "' msg_count=" << desired_messages_received_
                          << " q_des=" << q_des
                          << " q_actual=" << q_actual
                          << " err=" << (q_des - q_actual)
                          << " dq_des=" << dq_des
                          << " dq_actual=" << dq_actual
                          << " tau_ff=" << tau_ff);
        }
        else
        {
          ROS_WARN_STREAM("subscriber_controller debug: malformed desired/actual JointState for joint '"
                          << joint_name << "' desired_idx=" << desired_idx
                          << " actual_idx=" << actual_idx
                          << " desired position/velocity/effort sizes="
                          << desired_state_.position.size() << "/"
                          << desired_state_.velocity.size() << "/"
                          << desired_state_.effort.size()
                          << " actual position/velocity sizes="
                          << actual_js_state_.position.size() << "/"
                          << actual_js_state_.velocity.size());
        }
      };

      if (!debug_joint_prefix_.empty())
      {
        size_t logged = 0;
        for (const auto& joint_name : desired_state_.name)
        {
          if (joint_name.compare(0, debug_joint_prefix_.size(), debug_joint_prefix_) == 0)
          {
            log_joint(joint_name);
            logged++;
          }
        }
        if (logged == 0)
        {
          ROS_WARN_STREAM("subscriber_controller debug: no desired_state joints match prefix '"
                          << debug_joint_prefix_ << "'");
        }
      }
      else
      {
        const std::string joint_name =
            debug_joint_name_.empty() ? desired_state_.name.front() : debug_joint_name_;
        log_joint(joint_name);
      }
      last_debug_log_time_ = time;
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
    desired_messages_received_++;
    if (debug_desired_state_ && desired_messages_received_ <= 5)
    {
      ROS_WARN_STREAM("subscriber_controller received desired_state #"
                      << desired_messages_received_
                      << " joints=" << desired_state_.name.size()
                      << " positions=" << desired_state_.position.size()
                      << " velocities=" << desired_state_.velocity.size()
                      << " efforts=" << desired_state_.effort.size()
                      << " first_joint="
                      << (desired_state_.name.empty() ? "<none>" : desired_state_.name.front()));
    }
  }
  // Stop subsciber timer
  tp_->stopTime("subscriber");
}
}
