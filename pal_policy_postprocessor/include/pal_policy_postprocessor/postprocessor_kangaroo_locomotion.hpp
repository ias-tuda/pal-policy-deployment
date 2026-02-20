#pragma once

#include "pal_policy_postprocessor/postprocessor.hpp"

#include <vector>


namespace pal_policy_postprocessor
{
class KangarooLocoPostprocessor : public Postprocessor
{
public:
  explicit KangarooLocoPostprocessor();

protected:
  void postprocessAndPublish(const std::vector<double> & y) override;

private:
  template<class T>
  static constexpr const T & clamp(const T & v, const T & lo, const T & hi)
  {
    return v < lo ? lo : (hi < v ? hi : v);
  }

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_action_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_last_;

  std_msgs::msg::Float64MultiArray msg_action_;
  std_msgs::msg::Float64MultiArray last_;

  std::vector<double> default_joint_pos_, action_scale_, min_val_, max_val_;
  std::vector<double> action_offset_aligned_, action_scale_aligned_;
  std::vector<std::string> joint_names_, action_joint_names_;
  std::vector<size_t> action_joint_indices_;
  int64_t output_dim_;
};

} // namespace pal_policy_postprocessor
