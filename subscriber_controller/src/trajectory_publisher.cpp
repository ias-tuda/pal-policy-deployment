/*
 * Copyright 2020 PAL Robotics SL. All Rights Reserved
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited,
 * unless it was supplied under the terms of a license agreement or
 * nondisclosure agreement with PAL Robotics SL. In this case it may not be
 * copied or disclosed except in accordance with the terms of that agreement.
 */


#include <math.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/JointState.h>
#include <realtime_tools/realtime_publisher.h>
#include <pal_statistics/pal_statistics_macros.h>
#include <pal_utils/time_profiler.h>
#include <rbdl/addons/urdfreader/urdfreader.h>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>
#include <rbdl/Dynamics.h>

namespace subscriber_controller
{
struct ActuatorParameters
{
  double motor_torque_constant = 0.0;
  double reduction_ratio = 0.0;
};

class TrajectoryPublisher
{
public:
  bool load(ros::NodeHandle& nh, double dt)
  {
    dt_ = dt;

    std::vector<std::string> joint_names;
    std::vector<double> joint_position_min;
    std::vector<double> joint_position_max;
    std::vector<double> joint_vel_min;
    std::vector<double> joint_vel_max;
    std::vector<double> joint_damping;
    std::vector<double> joint_friction;
    std::vector<double> joint_max_effort;

    RigidBodyDynamics::Addons::URDFReadFromParamServer(
        &rbdl_model_, RigidBodyDynamics::FloatingBaseType::FixedBase,
        joint_names, joint_position_min, joint_position_max, joint_vel_min, joint_vel_max,
        joint_damping, joint_friction, joint_max_effort);

    state_sub_js_ = nh.subscribe("actual_js_state", 1, &TrajectoryPublisher::StateCb, this);

    state_sub_base_ =
        nh.subscribe("actual_base_state", 1, &TrajectoryPublisher::BaseStateCb, this);

    state_pub_.reset(new realtime_tools::RealtimePublisher<sensor_msgs::JointState>(
        nh, "desired_state", 1));

    while (!actual_joint_state_)
    {
      ROS_INFO_THROTTLE(0.1, "Receiving the actual state from joint states");
      ros::spinOnce();
    }

    // Reduce model and set initial position
    joints_state_.name = actual_joint_state_->name;
    joints_state_.position = actual_joint_state_->position;
    joints_state_.velocity = actual_joint_state_->velocity;
    joints_state_.effort = actual_joint_state_->effort;

    std::string controlled_joint_name = "arm_left_4_joint";

    {
      auto it = std::find(actual_joint_state_->name.begin(),
                          actual_joint_state_->name.end(), controlled_joint_name);
      if (it == actual_joint_state_->name.end())
      {
        ROS_ERROR_STREAM("Controlled joint " << controlled_joint_name
                                             << " not detected in joint states");
        loaded_ = false;
        return false;
      }
      controlled_joint_id_ = std::distance(actual_joint_state_->name.begin(), it);
    }

    {
      auto it = std::find(joint_names.begin(), joint_names.end(), controlled_joint_name);
      if (it == joint_names.end())
      {
        ROS_ERROR_STREAM("Controlled joint " << controlled_joint_name
                                             << " not detected in model joints");
        loaded_ = false;
        return false;
      }
      model_joint_id_ = std::distance(joint_names.begin(), it);
    }

    if (!nh.getParam("joints/" + controlled_joint_name + "/actuator_params/motor_torque_constant",
                     actuator_parameters_.motor_torque_constant))
    {
      ROS_ERROR_STREAM("Could not find motor torque constant for joint " << controlled_joint_name);
      return false;
    }
    if (!nh.getParam("joints/" + controlled_joint_name + "/actuator_params/reduction_ratio",
                     actuator_parameters_.reduction_ratio))
    {
      ROS_ERROR_STREAM("Could not find reduction ratio for joint " << controlled_joint_name);
      return false;
    }

    q_act_.resize(joint_names.size());
    q_zero_.resize(joint_names.size());
    tau_cmd_.resize(joint_names.size());

    q_act_.setZero();
    q_zero_.setZero();
    tau_cmd_.setZero();

    for(size_t i = 0; i < joints_state_.name.size(); i++)
    {
      auto it = std::find(joint_names.begin(),
                          joint_names.end(), joints_state_.name[i]);
      if (it == joint_names.end())
      {
        ROS_ERROR_STREAM("Joint name " << joints_state_.name[i]
                                             << " not detected in the model");
        loaded_ = false;
        return false;
      }
      map_controlled_model_.push_back(std::distance(joint_names.begin(), it));
    }

    controlled_joint_limits_ = std::make_pair(joint_position_min[model_joint_id_],
                                              joint_position_max[model_joint_id_]);

    tp_.reset(new pal_robot_tools::TimeProfiler);
    tp_->registerTimer("update");
    tp_->registerTimer("publisher");
    tp_->registerTimer("solve");
    REGISTER_VARIABLE("/introspection_data", "update_time",
                      tp_->getLastCycleTimePtr("update"), &registered_variables_);
    REGISTER_VARIABLE("/introspection_data", "publish_time",
                      tp_->getLastCycleTimePtr("publisher"), &registered_variables_);
    REGISTER_VARIABLE("/introspection_data", "publish_periodicity",
                      tp_->getPeriodicityPtr("publisher"), &registered_variables_);
    REGISTER_VARIABLE("/introspection_data", "remaining_time", &remaining_dt_,
                      &registered_variables_);
    loaded_ = true;

    pos_step_ = 5.e-3;
    ddr_.reset(new ddynamic_reconfigure::DDynamicReconfigure(nh));
    ddr_->RegisterVariable(&pos_step_, "/trajectory_publisher_pos_step", 0.0, 1.0);
    ddr_->PublishServicesTopics();

    joint_limit_tolerance_ = 0.3;
    direction_ = 1.0;

    desired_joint_pos_ = actual_joint_state_->position[controlled_joint_id_];

    return true;
  }

  void update()
  {
    tp_->startTimer("update");
    if (!loaded_)
    {
      ROS_ERROR("Trajectory Publisher should be loaded first");
    }

    for (size_t i = 0; i < joints_state_.name.size(); i++)
    {
      q_act_[map_controlled_model_[i]] = actual_joint_state_->position[i];
    }

    desired_joint_pos_ += direction_ * pos_step_;

    if (desired_joint_pos_ < (controlled_joint_limits_.first + joint_limit_tolerance_) ||
        desired_joint_pos_ > (controlled_joint_limits_.second - joint_limit_tolerance_))
    {
      direction_ *= -1;
      desired_joint_pos_ += direction_ * pos_step_;
    }

    q_act_[model_joint_id_] = desired_joint_pos_;

    RigidBodyDynamics::InverseDynamics(rbdl_model_, q_act_, q_zero_, q_zero_, tau_cmd_);

    double desired_joint_eff =
        tau_cmd_[model_joint_id_] /
        (actuator_parameters_.motor_torque_constant * actuator_parameters_.reduction_ratio);

    if (state_pub_->trylock() && !std::isnan(desired_joint_eff))
    {
      tp_->startTimer("publisher");
      joints_state_.header.stamp = ros::Time::now();
      joints_state_.position[controlled_joint_id_] = desired_joint_pos_;
      ROS_ERROR_STREAM(desired_joint_eff);
      joints_state_.effort[controlled_joint_id_] = desired_joint_eff;
      state_pub_->msg_ = joints_state_;
      state_pub_->unlockAndPublish();
      ROS_INFO_STREAM("State published ");
      tp_->stopTime("publisher");
    }
    tp_->stopTime("update");
    remaining_dt_ = dt_ - tp_->getLastCycleTime("update");
    ROS_INFO_STREAM("Default dt " << dt_ << ", Update time " << tp_->getLastCycleTime("update")
                                  << ", Remaining time " << remaining_dt_);
  }

  double getRemainingCycleTime()
  {
    return remaining_dt_;
  }


protected:
  void StateCb(const sensor_msgs::JointStateConstPtr& msg)
  {
    actual_joint_state_ = msg;
  }
  void BaseStateCb(const nav_msgs::Odometry& msg)
  {
    actual_base_state_ = msg;
  }

  bool loaded_;
  ros::Subscriber state_sub_js_;
  ros::Subscriber state_sub_base_;

  boost::shared_ptr<realtime_tools::RealtimePublisher<sensor_msgs::JointState>> state_pub_;

  sensor_msgs::JointState joints_state_;
  sensor_msgs::JointStateConstPtr actual_joint_state_;

  nav_msgs::Odometry actual_base_state_;

  size_t controlled_joint_id_;
  size_t model_joint_id_;
  std::pair<double, double> controlled_joint_limits_;

  pal_statistics::RegistrationsRAII registered_variables_;

  pal_robot_tools::TimeProfilerPtr tp_;

  RigidBodyDynamics::Model rbdl_model_;

  ddynamic_reconfigure::DDynamicReconfigurePtr ddr_;

  double dt_;
  double remaining_dt_;

  double joint_limit_tolerance_;
  double pos_step_;
  double direction_;
  double desired_joint_pos_;

  Eigen::VectorXd q_act_;
  Eigen::VectorXd q_zero_;
  Eigen::VectorXd tau_cmd_;

  std::vector<unsigned int> map_controlled_model_;

  ActuatorParameters actuator_parameters_;
};
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trajectory_publisher");
  ros::NodeHandle nh("subscriber_controller");
  double dt = 0.01;
  ROS_INFO_STREAM("Trajectory publisher dt is: " << dt);

  subscriber_controller::TrajectoryPublisher publisher;
  if (!publisher.load(nh, dt))
  {
    ROS_ERROR("Failed to load Trajectory Publisher");
    return (-1);
  }

  while (ros::ok())
  {
    ros::spinOnce();
    publisher.update();
    if (publisher.getRemainingCycleTime() > 0.0)
      ros::Duration(publisher.getRemainingCycleTime()).sleep();
    PUBLISH_STATISTICS("/introspection_data");
  }

  return (0);
}
