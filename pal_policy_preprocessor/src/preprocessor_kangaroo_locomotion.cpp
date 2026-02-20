#include "pal_policy_preprocessor/preprocessor_kangaroo_locomotion.hpp"
#include "pal_policy_utils/utils.hpp"

#include <algorithm>
#include <array>
#include <sstream>

using std::placeholders::_1;

namespace pal_policy_preprocessor
{
using pal_policy_utils::parse_doubles;
using pal_policy_utils::split_string;

KangarooLocoPreprocessor::KangarooLocoPreprocessor()
  : Preprocessor("kangaroo_loco_preprocessor"), imu_orientation_(Eigen::Quaterniond::Identity())
{
  // --- DECLARE PARAMETERS ---
  const std::string joint_names_str = this->declare_parameter<std::string>("joint_names", "");
  const std::string default_joint_pos_str = this->declare_parameter<std::string>("default_joint_pos", "");
  const std::string observation_names_str = this->declare_parameter<std::string>("observation_names", "");

  js_topic_ = this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
  imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/imu");
  odom_topic_ = this->declare_parameter<std::string>("odom_topic", "/odom");
  cmd_topic_ = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
  last_action_topic_ =
      this->declare_parameter<std::string>("last_action_topic", "/locomotion_kangaroo/last_action");
  debug_print_observation_vector_ = this->declare_parameter<bool>("debug_print_observation_vector", false);
  debug_print_every_n_cycles_ = this->declare_parameter<int64_t>("debug_print_every_n_cycles", 50);

  // Debug: Log received parameter values
  RCLCPP_INFO(this->get_logger(), "Parameters received:");
  RCLCPP_INFO(this->get_logger(), "  joint_names: %s", joint_names_str.c_str());
  RCLCPP_INFO(this->get_logger(), "  default_joint_pos: %s", default_joint_pos_str.c_str());
  RCLCPP_INFO(this->get_logger(), "  observation_names: %s", observation_names_str.c_str());
  RCLCPP_INFO(this->get_logger(), "  Topics: js=%s, imu=%s, odom=%s, cmd=%s",
              js_topic_.c_str(), imu_topic_.c_str(), odom_topic_.c_str(), cmd_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  last_action_topic: %s", last_action_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  debug_print_observation_vector: %s",
              debug_print_observation_vector_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  debug_print_every_n_cycles: %ld", debug_print_every_n_cycles_);

  const int64_t input_dim = this->declare_parameter<int64_t>("input_dimension", 0);
  const int64_t output_dim = this->declare_parameter<int64_t>("output_dimension", 0);

  config_.joint_names = split_string(joint_names_str, ',');
  config_.default_joint_pos = parse_doubles(default_joint_pos_str);
  config_.observation_names = split_string(observation_names_str, ',');

  config_.input_dim = static_cast<size_t>(input_dim);
  config_.output_dim = static_cast<size_t>(output_dim);

  // --- ERROR HANDLING ---
  if (config_.joint_names.empty())
  {
    throw std::runtime_error("joint_names parameter is empty");
  }

  if (config_.default_joint_pos.empty())
  {
    throw std::runtime_error("default_joint_pos parameter is empty");
  }

  if (config_.joint_names.size() != config_.default_joint_pos.size())
  {
    throw std::runtime_error(
        "joint_names and default_joint_pos size mismatch: " + std::to_string(config_.joint_names.size()) + " vs " +
        std::to_string(config_.default_joint_pos.size()));
  }

  if (config_.input_dim == 0 || config_.output_dim == 0)
  {
    throw std::runtime_error("input_dimension or output_dimension is zero");
  }

  buildJointIndexMap();
  initializeObservationTerms();

  // Validate input dimension against actual observation terms
  size_t expected_input_dim = 0;
  for (const auto& term_name : config_.observation_names)
  {
    int dim = getObservationTermDim(term_name);
    if (dim > 0)
    {
      expected_input_dim += dim;
      RCLCPP_INFO(this->get_logger(), "Observation term '%s' contributes %d dimensions", term_name.c_str(), dim);
    }
  }

  if (config_.input_dim != static_cast<int>(expected_input_dim))
  {
    throw std::runtime_error("input_dimension mismatch: expected " + std::to_string(expected_input_dim) +
                             " (based on observation_names), got " + std::to_string(config_.input_dim));
  }

  // --- SUBSCRIPTIONS ---
  sub_js_ = this->create_subscription<sensor_msgs::msg::JointState>(
      js_topic_, 1, std::bind(&KangarooLocoPreprocessor::jsCB, this, _1));
    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic_, 1,
                                                              std::bind(&KangarooLocoPreprocessor::imuCB, this, _1));
  sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 1, std::bind(&KangarooLocoPreprocessor::odomCB, this, _1));
  sub_cmd_ = this->create_subscription<geometry_msgs::msg::Twist>(
      cmd_topic_, 1, std::bind(&KangarooLocoPreprocessor::cmdCB, this, _1));
  sub_last_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      last_action_topic_, 1, std::bind(&KangarooLocoPreprocessor::lastCB, this, _1));

  obs_.assign(static_cast<size_t>(config_.input_dim), 0.0);
  run(std::chrono::milliseconds(20));
}

void KangarooLocoPreprocessor::buildInputVector(std::vector<double>* obs) const
{
  std::lock_guard<std::mutex> lock(obs_mutex_);

  // Iterate in the configured order
  for (const auto& term_name : config_.observation_names)
  {
    const auto& term = obs_terms_.at(term_name);
    int idx = term->start_idx;

    // Copy current values for this term
    for (int i = 0; i < term->dim; ++i)
    {
      (*obs)[idx++] = term->current[i];
    }
  }

  maybeLogObservationVector(*obs);
}

// ---- SENSOR CALLBACKS ----

void KangarooLocoPreprocessor::jsCB(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(obs_mutex_);

  for (size_t i = 0; i < msg->name.size(); ++i)
  {
    auto it = joint_name_to_index_.find(msg->name[i]);
    if (it == joint_name_to_index_.end())
    {
      continue;
    }

    const size_t k = it->second;

    // Update observation terms
    if (obs_terms_.find("joint_pos") != obs_terms_.end())
    {
      obs_terms_["joint_pos"]->current[k] = msg->position[i] - config_.default_joint_pos[k];
    }

    if (msg->velocity.size() == msg->name.size() && obs_terms_.find("joint_vel") != obs_terms_.end())
    {
      obs_terms_["joint_vel"]->current[k] = msg->velocity[i];
    }
  }
}

void KangarooLocoPreprocessor::imuCB(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(obs_mutex_);

  // Store orientation for projected gravity calculation
  imu_orientation_ = Eigen::Quaterniond(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

  // Update base_ang_vel term
  if (obs_terms_.find("base_ang_vel") != obs_terms_.end())
  {
    auto& term = obs_terms_["base_ang_vel"];
    term->current[0] = msg->angular_velocity.x;
    term->current[1] = msg->angular_velocity.y;
    term->current[2] = msg->angular_velocity.z;
  }

  // Update base_lin_acc term
  if (obs_terms_.find("base_lin_acc") != obs_terms_.end())
  {
    auto& term = obs_terms_["base_lin_acc"];
    term->current[0] = msg->linear_acceleration.x;
    term->current[1] = msg->linear_acceleration.y;
    term->current[2] = msg->linear_acceleration.z;
  }

  // Update imu_projected_gravity term
  if (obs_terms_.find("imu_projected_gravity") != obs_terms_.end())
  {
    auto& term = obs_terms_["imu_projected_gravity"];

    Eigen::Vector3d gravity_world(0.0, 0.0, -1.0);
    Eigen::Vector3d gravity_body = imu_orientation_.conjugate() * gravity_world;

    term->current[0] = gravity_body.x();
    term->current[1] = gravity_body.y();
    term->current[2] = gravity_body.z();
  }
}

void KangarooLocoPreprocessor::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(obs_mutex_);

  if (obs_terms_.find("base_lin_vel") != obs_terms_.end())
  {
    auto& term = obs_terms_["base_lin_vel"];
    term->current[0] = msg->twist.twist.linear.x;
    term->current[1] = msg->twist.twist.linear.y;
    term->current[2] = msg->twist.twist.linear.z;
  }
}

void KangarooLocoPreprocessor::cmdCB(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(obs_mutex_);

  // Update observation term
  if (obs_terms_.find("command") != obs_terms_.end())
  {
    auto& term = obs_terms_["command"];
    term->current[0] = msg->linear.x;
    term->current[1] = msg->linear.y;
    term->current[2] = msg->angular.z;
  }
}

void KangarooLocoPreprocessor::lastCB(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(obs_mutex_);

  // Update observation term
  if (obs_terms_.find("actions") != obs_terms_.end())
  {
    auto& term = obs_terms_["actions"];
    const size_t n = std::min(static_cast<size_t>(term->dim), msg->data.size());
    for (size_t i = 0; i < n; ++i)
    {
      term->current[i] = msg->data[i];
    }
  }
}

// ---- HELPER FUNCTIONS ----

int KangarooLocoPreprocessor::getObservationTermDim(const std::string& term_name) const
{
  if (term_name == "base_ang_vel")
    return 3;
  if (term_name == "base_lin_vel")
    return 3;
  if (term_name == "base_lin_acc")
    return 3;
  if (term_name == "imu_projected_gravity")
    return 3;
  if (term_name == "command")
    return 3;
  if (term_name == "joint_pos")
    return static_cast<int>(config_.joint_names.size());
  if (term_name == "joint_vel")
    return static_cast<int>(config_.joint_names.size());
  if (term_name == "actions")
    return static_cast<int>(config_.output_dim);

  return -1;  // Unknown term
}

void KangarooLocoPreprocessor::buildJointIndexMap()
{
  for (size_t i = 0; i < config_.joint_names.size(); ++i)
  {
    joint_name_to_index_[config_.joint_names[i]] = i;
  }
}

void KangarooLocoPreprocessor::initializeObservationTerms()
{
  obs_terms_.clear();

  int current_idx = 0;
  for (const auto& term_name : config_.observation_names)
  {
    int dim = getObservationTermDim(term_name);
    if (dim < 0)
    {
      RCLCPP_FATAL(this->get_logger(),
                   "Unknown observation term: '%s'. Add it to getObservationTermDim() and implement its callback.",
                   term_name.c_str());
      rclcpp::shutdown();
      return;
    }
    auto term = std::make_shared<ObservationTerm>(term_name, dim);
    term->start_idx = current_idx;  // Set the starting index
    obs_terms_[term_name] = term;

    // Advance index by term dimension
    current_idx += dim;

    RCLCPP_INFO(this->get_logger(), "  Term '%s': dim=%d, start_idx=%d", term_name.c_str(), dim, term->start_idx);
  }
}

std::string KangarooLocoPreprocessor::getObservationElementLabel(
  const std::string & term_name,
  int local_idx) const
{
  if (term_name == "base_ang_vel") {
    static const std::array<const char *, 3> names = {{"x", "y", "z"}};
    return "base_ang_vel." + std::string(names[local_idx]);
  }
  if (term_name == "base_lin_vel") {
    static const std::array<const char *, 3> names = {{"x", "y", "z"}};
    return "base_lin_vel." + std::string(names[local_idx]);
  }
  if (term_name == "base_lin_acc") {
    static const std::array<const char *, 3> names = {{"x", "y", "z"}};
    return "base_lin_acc." + std::string(names[local_idx]);
  }
  if (term_name == "imu_projected_gravity") {
    static const std::array<const char *, 3> names = {{"x", "y", "z"}};
    return "imu_projected_gravity." + std::string(names[local_idx]);
  }
  if (term_name == "command") {
    static const std::array<const char *, 3> names = {{"lin_x", "lin_y", "ang_z"}};
    return "command." + std::string(names[local_idx]);
  }
  if ((term_name == "joint_pos" || term_name == "joint_vel") &&
      local_idx >= 0 &&
      static_cast<size_t>(local_idx) < config_.joint_names.size())
  {
    return term_name + "[" + config_.joint_names[static_cast<size_t>(local_idx)] + "]";
  }
  if (term_name == "actions") {
    return "actions[" + std::to_string(local_idx) + "]";
  }

  return term_name + "[" + std::to_string(local_idx) + "]";
}

std::string KangarooLocoPreprocessor::getObservationElementSource(const std::string & term_name) const
{
  if (term_name == "base_ang_vel" || term_name == "base_lin_acc" || term_name == "imu_projected_gravity") {
    return imu_topic_ + " (imuCB)";
  }
  if (term_name == "base_lin_vel") {
    return odom_topic_ + " (odomCB)";
  }
  if (term_name == "command") {
    return cmd_topic_ + " (cmdCB)";
  }
  if (term_name == "joint_pos" || term_name == "joint_vel") {
    return js_topic_ + " (jsCB)";
  }
  if (term_name == "actions") {
    return last_action_topic_ + " (lastCB)";
  }

  return "unknown";
}

void KangarooLocoPreprocessor::maybeLogObservationVector(const std::vector<double> & obs) const
{
  if (!debug_print_observation_vector_) {
    return;
  }

  if (debug_print_every_n_cycles_ <= 0) {
    return;
  }

  ++debug_cycle_counter_;
  if ((debug_cycle_counter_ % debug_print_every_n_cycles_) != 0) {
    return;
  }

  std::ostringstream out;
  out << "Observation dump (cycle=" << debug_cycle_counter_ << ", dim=" << obs.size() << "):\n";

  for (const auto & term_name : config_.observation_names)
  {
    const auto & term = obs_terms_.at(term_name);
    const std::string source = getObservationElementSource(term_name);

    for (int i = 0; i < term->dim; ++i)
    {
      const int global_idx = term->start_idx + i;
      const std::string label = getObservationElementLabel(term_name, i);
      out << "  [" << global_idx << "] " << label << " = " << obs[static_cast<size_t>(global_idx)]
          << " | source: " << source << "\n";
    }
  }

  RCLCPP_INFO(this->get_logger(), "%s", out.str().c_str());
}

}  // namespace pal_policy_preprocessor

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pal_policy_preprocessor::KangarooLocoPreprocessor>());
  rclcpp::shutdown();
  return 0;
}
