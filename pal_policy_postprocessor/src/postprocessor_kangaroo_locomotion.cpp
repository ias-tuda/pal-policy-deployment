#include "pal_policy_postprocessor/postprocessor_kangaroo_locomotion.hpp"
#include "pal_policy_utils/utils.hpp"

#include <algorithm>
#include <unordered_set>

namespace pal_policy_postprocessor
{
using pal_policy_utils::parse_doubles;
using pal_policy_utils::split_string;

KangarooLocoPostprocessor::KangarooLocoPostprocessor()
: Postprocessor("kangaroo_loco_postprocessor")
{
  const std::string joint_names_str = this->declare_parameter<std::string>("joint_names", "");
  const std::string excluded_action_joints_str = this->declare_parameter<std::string>(
    "excluded_action_joints",
    "leg_left_femur_joint,leg_left_knee_joint,leg_right_femur_joint,leg_right_knee_joint");
  const std::string action_scale_str = this->declare_parameter<std::string>("action_scale", "");
  const std::string default_joint_pos_str = this->declare_parameter<std::string>(
    "default_joint_pos", "");

  joint_names_ = split_string(joint_names_str, ',');
  default_joint_pos_ = parse_doubles(default_joint_pos_str);
  action_scale_ = parse_doubles(action_scale_str);

  output_dim_ = this->declare_parameter<int64_t>("output_dimension", 0);

  // ERROR HANDLING -------------------------------
  if (joint_names_.empty()) {
    throw std::runtime_error("joint_names parameter is empty");
  }

  if (default_joint_pos_.size() != joint_names_.size()) {
    throw std::runtime_error(
            "default_joint_pos size mismatch: expected " + std::to_string(joint_names_.size()) +
            " (same as joint_names), got " + std::to_string(default_joint_pos_.size()));
  }

  if (output_dim_ == 0) {
    throw std::runtime_error("output_dimension is zero");
  }

  const auto excluded_action_joints = split_string(excluded_action_joints_str, ',');
  std::unordered_set<std::string> excluded_set(excluded_action_joints.begin(), excluded_action_joints.end());

  action_joint_indices_.clear();
  action_joint_names_.clear();
  action_joint_indices_.reserve(joint_names_.size());
  action_joint_names_.reserve(joint_names_.size());

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    if (excluded_set.find(joint_names_[i]) == excluded_set.end()) {
      action_joint_indices_.push_back(i);
      action_joint_names_.push_back(joint_names_[i]);
    }
  }

  if (action_joint_indices_.size() != static_cast<size_t>(output_dim_)) {
    throw std::runtime_error(
            "output_dimension mismatch with derived action joints: output_dimension=" +
            std::to_string(output_dim_) + ", derived_action_joints=" +
            std::to_string(action_joint_indices_.size()) +
            ". Check joint_names and excluded_action_joints.");
  }

  action_offset_aligned_.resize(static_cast<size_t>(output_dim_));
  action_scale_aligned_.resize(static_cast<size_t>(output_dim_));

  for (size_t i = 0; i < static_cast<size_t>(output_dim_); ++i) {
    const size_t joint_idx = action_joint_indices_[i];
    action_offset_aligned_[i] = default_joint_pos_[joint_idx];

    if (action_scale_.size() == static_cast<size_t>(output_dim_)) {
      action_scale_aligned_[i] = action_scale_[i];
    } else if (action_scale_.size() == joint_names_.size()) {
      action_scale_aligned_[i] = action_scale_[joint_idx];
    } else {
      throw std::runtime_error(
              "action_scale size mismatch: expected " + std::to_string(output_dim_) +
              " (action dimension) or " + std::to_string(joint_names_.size()) +
              " (joint_names size), got " + std::to_string(action_scale_.size()));
    }
  }

  // -------------------------------

  msg_action_.data.resize(output_dim_);
  last_.data.resize(output_dim_);

  RCLCPP_INFO(this->get_logger(), "Postprocessor action mapping (%ld outputs):", output_dim_);
  for (size_t i = 0; i < static_cast<size_t>(output_dim_); ++i) {
    RCLCPP_INFO(
      this->get_logger(),
      "  y[%zu] -> %s (joint_names[%zu]), cmd = %.6f + %.6f * y",
      i,
      action_joint_names_[i].c_str(),
      action_joint_indices_[i],
      action_offset_aligned_[i],
      action_scale_aligned_[i]);
  }

  pub_action_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/rl_controller/commands", 1);

  pub_last_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/locomotion_kangaroo/last_action", 1);
}


void KangarooLocoPostprocessor::postprocessAndPublish(
  const std::vector<double> & y)
{
  if (y.size() != static_cast<size_t>(output_dim_)) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Postprocessor got output of size %zu, expected %ld",
      y.size(), output_dim_);
    return;
  }

  for (size_t i = 0; i < static_cast<size_t>(output_dim_); ++i) {
    msg_action_.data[i] = action_offset_aligned_[i] + action_scale_aligned_[i] * y[i];
  }

  

  // publish desired joint positions

  pub_action_->publish(msg_action_);
  // publish last action (raw network output)
  last_.data = y;
  pub_last_->publish(last_);
}

}
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pal_policy_postprocessor::KangarooLocoPostprocessor>());
  rclcpp::shutdown();
  return 0;
}
