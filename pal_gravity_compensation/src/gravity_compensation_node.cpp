#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <urdf/model.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

double clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
}

}  // namespace

class GravityCompensationNode {
 public:
  GravityCompensationNode() : private_nh_("~") {
    private_nh_.param<std::string>("robot_description", description_param_, "robot_description");
    private_nh_.param<std::string>("base_link", base_link_, "base_link");
    private_nh_.param<std::string>("gravity_tip_link", gravity_tip_link_,
                                   "gripper_left_base_link");
    private_nh_.param<std::string>("input_topic", input_topic_,
                                   "/pal_grasp/desired_state_uncompensated");
    private_nh_.param<std::string>("state_topic", state_topic_,
                                   "/subscriber_controller/actual_js_state");
    private_nh_.param<std::string>("output_topic", output_topic_,
                                   "/subscriber_controller/desired_state");
    private_nh_.param<std::string>("preview_topic", preview_topic_,
                                   "/pal_grasp/desired_state_compensated");
    private_nh_.param<std::string>("controller_name", controller_name_,
                                   "subscriber_controller");
    private_nh_.param("state_timeout", state_timeout_, 0.25);
    private_nh_.param("rate", rate_, 200.0);
    if (!std::isfinite(state_timeout_) || state_timeout_ <= 0.0 ||
        !std::isfinite(rate_) || rate_ <= 0.0)
      throw std::runtime_error("state_timeout and rate must be positive");
    if (!private_nh_.getParam("arm_joint_names", arm_names_) || arm_names_.empty() ||
        !private_nh_.getParam("effort_limits", effort_limits_) ||
        effort_limits_.size() != arm_names_.size())
      throw std::runtime_error("arm_joint_names and matching effort_limits are required");
    std::unordered_set<std::string> unique_names(arm_names_.begin(), arm_names_.end());
    if (unique_names.size() != arm_names_.size() ||
        !std::all_of(effort_limits_.begin(), effort_limits_.end(),
                     [](double value) { return std::isfinite(value) && value > 0.0; }))
      throw std::runtime_error("invalid arm_joint_names or effort_limits");
    std::vector<double> gravity_base;
    if (!private_nh_.getParam("gravity_base", gravity_base) || gravity_base.size() != 3 ||
        !std::all_of(gravity_base.begin(), gravity_base.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("gravity_base must contain three finite values");

    std::string urdf_xml;
    if (!nh_.getParam(description_param_, urdf_xml))
      throw std::runtime_error("robot_description is unavailable");
    urdf::Model model;
    KDL::Tree tree;
    if (!model.initString(urdf_xml) || !kdl_parser::treeFromUrdfModel(model, tree) ||
        !tree.getChain(base_link_, gravity_tip_link_, chain_))
      throw std::runtime_error("cannot build gravity chain to " + gravity_tip_link_);
    for (const auto& segment : chain_.segments) {
      if (segment.getJoint().getType() != KDL::Joint::None)
        chain_names_.push_back(segment.getJoint().getName());
    }
    for (const auto& name : arm_names_) {
      const auto found = std::find(chain_names_.begin(), chain_names_.end(), name);
      if (found == chain_names_.end())
        throw std::runtime_error("gravity chain is missing " + name);
      arm_chain_indices_.push_back(std::distance(chain_names_.begin(), found));
    }
    gravity_solver_.reset(new KDL::ChainDynParam(
        chain_, KDL::Vector(gravity_base[0], gravity_base[1], gravity_base[2])));

    output_pub_ = nh_.advertise<sensor_msgs::JointState>(output_topic_, 1);
    preview_pub_ = nh_.advertise<sensor_msgs::JointState>(preview_topic_, 1);
    state_sub_ = nh_.subscribe(state_topic_, 1, &GravityCompensationNode::stateCallback, this);
    input_sub_ = nh_.subscribe(input_topic_, 1, &GravityCompensationNode::inputCallback, this);
    timer_ = nh_.createWallTimer(ros::WallDuration(1.0 / rate_),
                                 &GravityCompensationNode::publish, this);
    ROS_INFO("Gravity bridge: %s -> %s using %s", input_topic_.c_str(),
             output_topic_.c_str(), gravity_tip_link_.c_str());
  }

 private:
  void stateCallback(const sensor_msgs::JointState::ConstPtr& message) {
    if (message->position.size() != message->name.size() ||
        message->velocity.size() != message->name.size()) return;
    std::unordered_set<std::string> names(message->name.begin(), message->name.end());
    if (names.size() != message->name.size()) {
      ROS_ERROR_THROTTLE(1.0, "Measured JointState contains duplicate names");
      return;
    }
    state_ = *message;
    state_wall_time_ = ros::WallTime::now();
    have_state_ = true;
  }

  void inputCallback(const sensor_msgs::JointState::ConstPtr& message) {
    if (message->name.empty() || message->position.size() != message->name.size() ||
        message->velocity.size() != message->name.size() ||
        (!message->effort.empty() && message->effort.size() != message->name.size())) {
      ROS_ERROR_THROTTLE(1.0, "Invalid uncompensated JointState shape");
      return;
    }
    std::unordered_set<std::string> names(message->name.begin(), message->name.end());
    if (names.size() != message->name.size() ||
        !std::all_of(message->position.begin(), message->position.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::all_of(message->velocity.begin(), message->velocity.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::all_of(message->effort.begin(), message->effort.end(),
                     [](double value) { return std::isfinite(value) && value == 0.0; })) {
      ROS_ERROR_THROTTLE(1.0, "Uncompensated JointState contains duplicate or invalid values");
      return;
    }
    for (const auto& name : arm_names_) {
      if (!names.count(name)) {
        ROS_ERROR_THROTTLE(1.0, "Uncompensated JointState is missing %s", name.c_str());
        return;
      }
    }
    target_ = *message;
    have_target_ = true;
  }

  bool loadGains() {
    if (have_gains_) return true;
    if (ros::WallTime::now() < gain_retry_after_) return false;
    gain_retry_after_ = ros::WallTime::now() + ros::WallDuration(0.5);
    stiffness_.clear();
    damping_.clear();
    for (const auto& name : arm_names_) {
      const std::string ns = "/" + controller_name_ + "/joints/" + name + "/impedance_params";
      double stiffness, damping;
      if (!ros::param::get(ns + "/stiffness_factor", stiffness) ||
          !ros::param::get(ns + "/damping_factor", damping) ||
          !std::isfinite(stiffness) || stiffness <= 0.0 ||
          !std::isfinite(damping) || damping < 0.0) {
        ROS_WARN_THROTTLE(2.0, "Waiting for valid impedance gains at %s", ns.c_str());
        return false;
      }
      stiffness_.push_back(stiffness);
      damping_.push_back(damping);
    }
    have_gains_ = true;
    return true;
  }

  void publish(const ros::WallTimerEvent&) {
    if (!have_target_ || !have_state_ || !loadGains()) return;
    if ((ros::WallTime::now() - state_wall_time_).toSec() > state_timeout_) {
      ROS_WARN_THROTTLE(1.0, "Gravity bridge: measured joint state is stale");
      return;
    }
    std::unordered_map<std::string, std::size_t> state_indices;
    std::unordered_map<std::string, std::size_t> target_indices;
    for (std::size_t i = 0; i < state_.name.size(); ++i) state_indices[state_.name[i]] = i;
    for (std::size_t i = 0; i < target_.name.size(); ++i) target_indices[target_.name[i]] = i;
    KDL::JntArray q(chain_.getNrOfJoints());
    for (std::size_t i = 0; i < chain_names_.size(); ++i) {
      const auto found = state_indices.find(chain_names_[i]);
      if (found == state_indices.end() || !std::isfinite(state_.position[found->second])) {
        ROS_WARN_THROTTLE(1.0, "Gravity bridge: gravity chain state is incomplete");
        return;
      }
      q(i) = state_.position[found->second];
    }
    KDL::JntArray gravity(chain_.getNrOfJoints());
    if (gravity_solver_->JntToGravity(q, gravity) < 0) {
      ROS_ERROR_THROTTLE(1.0, "Gravity bridge: dynamics solve failed");
      return;
    }
    sensor_msgs::JointState output = target_;
    output.header.stamp = ros::Time::now();
    output.effort.assign(output.name.size(), 0.0);
    for (std::size_t i = 0; i < arm_names_.size(); ++i) {
      const std::size_t actual_index = state_indices.at(arm_names_[i]);
      const std::size_t target_index = target_indices.at(arm_names_[i]);
      const double tau_gravity = gravity(arm_chain_indices_[i]);
      const double tau_pd = stiffness_[i] *
          (output.position[target_index] - state_.position[actual_index]) +
          damping_[i] * (output.velocity[target_index] - state_.velocity[actual_index]);
      if (!std::isfinite(tau_gravity) || !std::isfinite(tau_pd)) {
        ROS_ERROR_THROTTLE(1.0, "Gravity bridge: non-finite torque");
        return;
      }
      const double limit = effort_limits_[i];
      const double bounded_gravity = clamp(tau_gravity, -limit, limit);
      const double budgeted_gravity = clamp(bounded_gravity, -limit - tau_pd,
                                            limit - tau_pd);
      output.effort[target_index] = clamp(budgeted_gravity, -limit, limit);
      if (std::abs(output.effort[target_index] + tau_pd) > limit ||
          std::abs(output.effort[target_index] - tau_gravity) > 1e-6)
        ROS_WARN_THROTTLE(1.0, "Gravity bridge: nominal total torque budget reached");
    }
    preview_pub_.publish(output);
    output_pub_.publish(output);
  }

  ros::NodeHandle nh_, private_nh_;
  ros::Subscriber state_sub_, input_sub_;
  ros::Publisher output_pub_, preview_pub_;
  ros::WallTimer timer_;
  std::string description_param_, base_link_, gravity_tip_link_;
  std::string input_topic_, state_topic_, output_topic_, preview_topic_, controller_name_;
  std::vector<std::string> arm_names_, chain_names_;
  std::vector<std::size_t> arm_chain_indices_;
  std::vector<double> effort_limits_, stiffness_, damping_;
  double state_timeout_, rate_;
  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainDynParam> gravity_solver_;
  sensor_msgs::JointState state_, target_;
  ros::WallTime state_wall_time_, gain_retry_after_;
  bool have_state_ = false, have_target_ = false, have_gains_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "gravity_compensation");
  try {
    GravityCompensationNode node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("Gravity bridge startup failed: %s", error.what());
    return 1;
  }
  return 0;
}
