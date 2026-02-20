#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace pal_policy_inferer
{

struct ObservationTerm
{
  std::string name;
  int dim;
  std::vector<double> current;
  std::deque<std::vector<double>> history;

  ObservationTerm(const std::string & n, int d)
  : name(n), dim(d), current(static_cast<size_t>(d), 0.0)
  {
  }

  void initHistory(int history_length)
  {
    history.clear();
    for (int i = 0; i < history_length; ++i) {
      history.push_back(std::vector<double>(static_cast<size_t>(dim), 0.0));
    }
  }

  void updateHistory()
  {
    history.pop_front();
    history.push_back(current);
  }
};

struct PolicyConfig
{
  std::string run_path;
  std::vector<std::string> joint_names;
  std::vector<std::string> action_joint_names;
  std::vector<double> default_joint_pos;
  std::vector<double> action_offset;
  std::vector<double> action_scale;
  std::vector<double> joint_stiffness;
  std::vector<double> joint_damping;
  std::vector<std::string> observation_names;
  int obs_dim{0};
  int action_dim{0};
  int history_length{1};
  int single_obs_dim{0};
};

class KangarooMjlabMonolithicNode : public rclcpp::Node
{
public:
  KangarooMjlabMonolithicNode()
  : Node("pal_policy_inferer_kangaroo_mjlab_monolithic"),
    imu_orientation_{{1.0, 0.0, 0.0, 0.0}},
    env_(ORT_LOGGING_LEVEL_WARNING, "rl"),
    mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    const std::string onnx_model_path = this->declare_parameter<std::string>("onnx_model_path", "");
    if (onnx_model_path.empty()) {
      throw std::runtime_error("onnx_model_path parameter is empty");
    }

    joint_states_topic_ = this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
    imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/imu_sensor_broadcaster/imu");
    cmd_vel_topic_ = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    odom_topic_ = this->declare_parameter<std::string>("odom_topic", "/odom");
    command_topic_ = this->declare_parameter<std::string>("command_topic", "/rl_controller/commands");
    const double default_stiffness = this->declare_parameter<double>("default_stiffness", 0.0);
    const double default_damping = this->declare_parameter<double>("default_damping", 0.0);
    const std::string stiffness_override_str = this->declare_parameter<std::string>("joint_stiffness", "");
    const std::string damping_override_str = this->declare_parameter<std::string>("joint_damping", "");

    const std::string excluded_action_joints_str = this->declare_parameter<std::string>(
      "excluded_action_joints",
      "leg_left_femur_joint,leg_left_knee_joint,leg_right_femur_joint,leg_right_knee_joint");

    const int64_t frequency_hz = this->declare_parameter<int64_t>("frequency_hz", 50);
    if (frequency_hz <= 0) {
      throw std::runtime_error("frequency_hz must be > 0");
    }

    loadPolicyConfigFromOnnx(onnx_model_path, excluded_action_joints_str);
    initializeObservationTerms();

    action_.assign(static_cast<size_t>(config_.action_dim), 0.0);
    input_shape_ = {1, static_cast<int64_t>(config_.obs_dim)};

    session_.reset(new Ort::Session(env_, onnx_model_path.c_str(), Ort::SessionOptions{}));
    input_name_ = session_->GetInputNameAllocated(0, allocator_).release();
    output_name_ = session_->GetOutputNameAllocated(0, allocator_).release();

    if (config_.action_scale.size() != static_cast<size_t>(config_.action_dim)) {
      throw std::runtime_error(
              "action_scale size mismatch: expected " + std::to_string(config_.action_dim) +
              ", got " + std::to_string(config_.action_scale.size()));
    }

    stiffness_aligned_ = alignGains(
      stiffness_override_str.empty() ? config_.joint_stiffness : parseDoubles(stiffness_override_str),
      default_stiffness,
      "joint_stiffness");
    damping_aligned_ = alignGains(
      damping_override_str.empty() ? config_.joint_damping : parseDoubles(damping_override_str),
      default_damping,
      "joint_damping");

    action_joint_to_action_index_.clear();
    for (size_t i = 0; i < config_.action_joint_names.size(); ++i) {
      action_joint_to_action_index_[config_.action_joint_names[i]] = i;
    }

    qpos_action_.assign(static_cast<size_t>(config_.action_dim), 0.0);
    qvel_action_.assign(static_cast<size_t>(config_.action_dim), 0.0);
    tau_.assign(static_cast<size_t>(config_.action_dim), 0.0);

    pub_cmd_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(command_topic_, 1);

    sub_js_ = this->create_subscription<sensor_msgs::msg::JointState>(
      joint_states_topic_, 1,
      std::bind(&KangarooMjlabMonolithicNode::jsCB, this, std::placeholders::_1));
    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, 1,
      std::bind(&KangarooMjlabMonolithicNode::imuCB, this, std::placeholders::_1));
    sub_cmd_ = this->create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, 1,
      std::bind(&KangarooMjlabMonolithicNode::velCmdCB, this, std::placeholders::_1));

    if (obs_terms_.find("base_lin_vel") != obs_terms_.end()) {
      sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 1,
        std::bind(&KangarooMjlabMonolithicNode::odomCB, this, std::placeholders::_1));
      RCLCPP_INFO(this->get_logger(), "Subscribed to %s for base_lin_vel", odom_topic_.c_str());
    }

    msg_command_.data.assign(static_cast<size_t>(config_.action_dim), 0.0);

    const auto period = std::chrono::microseconds(1000);
    timer_ = this->create_wall_timer(
      period,
      std::bind(&KangarooMjlabMonolithicNode::timerCB, this));

    RCLCPP_INFO(this->get_logger(), "Monolithic inferer started. obs_dim=%d, action_dim=%d, history=%d",
      config_.obs_dim, config_.action_dim, config_.history_length);
    RCLCPP_INFO(this->get_logger(), "Publishing commands to %s", command_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Impedance effort enabled: tau = K*(action-qpos) + D*qvel");
  }

  ~KangarooMjlabMonolithicNode() override
  {
    if (input_name_) {
      allocator_.Free(input_name_);
      input_name_ = nullptr;
    }
    if (output_name_) {
      allocator_.Free(output_name_);
      output_name_ = nullptr;
    }
  }

private:
  static std::vector<std::string> splitString(const std::string & str, char delimiter)
  {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }

  static std::vector<double> parseDoubles(const std::string & str)
  {
    std::vector<double> values;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      values.push_back(std::stod(token));
    }
    return values;
  }

  static int getObservationTermDim(const std::string & term_name, const PolicyConfig & config)
  {
    if (term_name == "base_ang_vel") {return 3;}
    if (term_name == "base_lin_vel") {return 3;}
    if (term_name == "base_lin_acc") {return 3;}
    if (term_name == "imu_projected_gravity") {return 3;}
    if (term_name == "command") {return 3;}
    if (term_name == "joint_pos") {return static_cast<int>(config.joint_names.size());}
    if (term_name == "joint_vel") {return static_cast<int>(config.joint_names.size());}
    if (term_name == "actions") {return static_cast<int>(config.action_joint_names.size());}
    return -1;
  }

  static std::array<double, 4> quatMultiply(
    const std::array<double, 4> & a,
    const std::array<double, 4> & b)
  {
    const double aw = a[0];
    const double ax = a[1];
    const double ay = a[2];
    const double az = a[3];

    const double bw = b[0];
    const double bx = b[1];
    const double by = b[2];
    const double bz = b[3];

    return {{
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw}};
  }

  static std::array<double, 3> rotateWorldToBody(
    const std::array<double, 4> & q_body_to_world,
    const std::array<double, 3> & v_world)
  {
    const std::array<double, 4> q_conj = {{
      q_body_to_world[0],
      -q_body_to_world[1],
      -q_body_to_world[2],
      -q_body_to_world[3]}};

    const std::array<double, 4> vq = {{0.0, v_world[0], v_world[1], v_world[2]}};
    const std::array<double, 4> tmp = quatMultiply(q_conj, vq);
    const std::array<double, 4> res = quatMultiply(tmp, q_body_to_world);
    return {{res[1], res[2], res[3]}};
  }

  std::vector<double> alignGains(
    const std::vector<double> & source,
    double fallback,
    const std::string & gain_name) const
  {
    std::vector<double> aligned(static_cast<size_t>(config_.action_dim), fallback);

    if (source.empty()) {
      RCLCPP_WARN(
        this->get_logger(),
        "%s not provided; using scalar default %.6f for all %d action joints",
        gain_name.c_str(), fallback, config_.action_dim);
      return aligned;
    }

    if (source.size() == static_cast<size_t>(config_.action_dim)) {
      return source;
    }

    if (source.size() == config_.joint_names.size()) {
      for (size_t i = 0; i < static_cast<size_t>(config_.action_dim); ++i) {
        const std::string & joint_name = config_.action_joint_names[i];
        auto it = std::find(config_.joint_names.begin(), config_.joint_names.end(), joint_name);
        if (it == config_.joint_names.end()) {
          throw std::runtime_error("Cannot align " + gain_name + ": missing action joint '" + joint_name + "'");
        }
        const size_t full_idx = static_cast<size_t>(std::distance(config_.joint_names.begin(), it));
        aligned[i] = source[full_idx];
      }
      return aligned;
    }

    throw std::runtime_error(
            gain_name + " size mismatch: expected " + std::to_string(config_.action_dim) +
            " (action joints) or " + std::to_string(config_.joint_names.size()) +
            " (all joints), got " + std::to_string(source.size()));
  }

  void loadPolicyConfigFromOnnx(const std::string & model_path, const std::string & excluded_action_joints)
  {
    Ort::SessionOptions session_options;
    Ort::Session temp_session(env_, model_path.c_str(), session_options);

    Ort::ModelMetadata metadata = temp_session.GetModelMetadata();
    std::vector<Ort::AllocatedStringPtr> keys = metadata.GetCustomMetadataMapKeysAllocated(allocator_);

    std::map<std::string, std::string> metadata_map;
    for (auto & key_ptr : keys) {
      const std::string key = key_ptr.get();
      Ort::AllocatedStringPtr value_ptr = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator_);
      if (value_ptr) {
        metadata_map[key] = value_ptr.get();
      }
    }

    if (metadata_map.find("joint_names") == metadata_map.end()) {
      throw std::runtime_error("Missing 'joint_names' in ONNX metadata");
    }
    if (metadata_map.find("default_joint_pos") == metadata_map.end()) {
      throw std::runtime_error("Missing 'default_joint_pos' in ONNX metadata");
    }
    if (metadata_map.find("action_scale") == metadata_map.end()) {
      throw std::runtime_error("Missing 'action_scale' in ONNX metadata");
    }
    if (metadata_map.find("observation_names") == metadata_map.end()) {
      throw std::runtime_error("Missing 'observation_names' in ONNX metadata");
    }

    if (metadata_map.find("run_path") != metadata_map.end()) {
      config_.run_path = metadata_map["run_path"];
    }

    config_.joint_names = splitString(metadata_map["joint_names"], ',');
    config_.default_joint_pos = parseDoubles(metadata_map["default_joint_pos"]);
    config_.action_scale = parseDoubles(metadata_map["action_scale"]);
    config_.observation_names = splitString(metadata_map["observation_names"], ',');

    if (metadata_map.find("joint_stiffness") != metadata_map.end()) {
      config_.joint_stiffness = parseDoubles(metadata_map["joint_stiffness"]);
    }
    if (metadata_map.find("joint_damping") != metadata_map.end()) {
      config_.joint_damping = parseDoubles(metadata_map["joint_damping"]);
    }

    if (config_.joint_names.size() != config_.default_joint_pos.size()) {
      throw std::runtime_error("joint_names and default_joint_pos size mismatch");
    }

    const std::vector<std::string> excluded = splitString(excluded_action_joints, ',');
    std::unordered_set<std::string> excluded_set(excluded.begin(), excluded.end());

    config_.action_joint_names.clear();
    config_.action_offset.clear();
    for (size_t i = 0; i < config_.joint_names.size(); ++i) {
      if (excluded_set.find(config_.joint_names[i]) == excluded_set.end()) {
        config_.action_joint_names.push_back(config_.joint_names[i]);
        config_.action_offset.push_back(config_.default_joint_pos[i]);
      }
    }

    Ort::TypeInfo input_info = temp_session.GetInputTypeInfo(0);
    auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();

    Ort::TypeInfo output_info = temp_session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();

    if (input_dims.size() < 2 || output_dims.size() < 2) {
      throw std::runtime_error("Invalid model input/output rank; expected [1, N]/[1, M]");
    }

    config_.obs_dim = static_cast<int>(input_dims[1]);
    config_.action_dim = static_cast<int>(output_dims[1]);

    config_.single_obs_dim = 0;
    for (const auto & term_name : config_.observation_names) {
      const int dim = getObservationTermDim(term_name, config_);
      if (dim < 0) {
        throw std::runtime_error("Unknown observation term: '" + term_name + "'");
      }
      config_.single_obs_dim += dim;
    }

    if (config_.single_obs_dim <= 0 || config_.obs_dim % config_.single_obs_dim != 0) {
      throw std::runtime_error(
              "Observation dimension mismatch: obs_dim=" + std::to_string(config_.obs_dim) +
              ", single_obs_dim=" + std::to_string(config_.single_obs_dim));
    }
    config_.history_length = config_.obs_dim / config_.single_obs_dim;

    if (config_.action_dim != static_cast<int>(config_.action_joint_names.size())) {
      throw std::runtime_error(
              "Action dimension mismatch: model=" + std::to_string(config_.action_dim) +
              ", action_joint_names=" + std::to_string(config_.action_joint_names.size()));
    }
  }

  void initializeObservationTerms()
  {
    obs_terms_.clear();
    obs_term_order_.clear();

    for (const auto & term_name : config_.observation_names) {
      const int dim = getObservationTermDim(term_name, config_);
      auto term = std::make_shared<ObservationTerm>(term_name, dim);
      term->initHistory(config_.history_length);
      obs_terms_[term_name] = term;
      obs_term_order_.push_back(term_name);
    }
  }

  std::vector<float> buildObservationVector() const
  {
    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(config_.obs_dim));

    for (const auto & term_name : obs_term_order_) {
      const auto & term = obs_terms_.at(term_name);
      for (int t = 0; t < config_.history_length; ++t) {
        for (int i = 0; i < term->dim; ++i) {
          obs.push_back(static_cast<float>(term->history[static_cast<size_t>(t)][static_cast<size_t>(i)]));
        }
      }
    }

    return obs;
  }

  void updateAllHistory()
  {
    for (auto & pair : obs_terms_) {
      pair.second->updateHistory();
    }
  }

  void timerCB()
  {
    std::vector<float> obs_input;
    {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      updateAllHistory();
      obs_input = buildObservationVector();
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_, obs_input.data(), obs_input.size(), input_shape_.data(), input_shape_.size());

    const char * input_names[] = {input_name_};
    const char * output_names[] = {output_name_};

    auto outputs = session_->Run(
      Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    const float * y_f = outputs.front().GetTensorData<float>();

    for (int i = 0; i < config_.action_dim; ++i) {
      action_[static_cast<size_t>(i)] =
        config_.action_scale[static_cast<size_t>(i)] * y_f[i] + config_.action_offset[static_cast<size_t>(i)];
    }

    {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      auto it = obs_terms_.find("actions");
      if (it != obs_terms_.end()) {
        auto & actions_term = it->second;
        for (int i = 0; i < config_.action_dim; ++i) {
          actions_term->current[static_cast<size_t>(i)] = y_f[i];
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      for (int i = 0; i < config_.action_dim; ++i) {
        const size_t k = static_cast<size_t>(i);
        tau_[k] = stiffness_aligned_[k] * (action_[k] - qpos_action_[k]) + damping_aligned_[k] * qvel_action_[k];
      }
    }

    msg_command_.data = tau_;
    pub_cmd_->publish(msg_command_);
  }

  void jsCB(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);

    auto pos_it = obs_terms_.find("joint_pos");
    auto vel_it = obs_terms_.find("joint_vel");

    for (size_t k = 0; k < config_.joint_names.size(); ++k) {
      for (size_t i = 0; i < msg->name.size(); ++i) {
        if (msg->name[i] == config_.joint_names[k]) {
          if (pos_it != obs_terms_.end() && i < msg->position.size()) {
            pos_it->second->current[k] = msg->position[i] - config_.default_joint_pos[k];
          }
          if (vel_it != obs_terms_.end() && i < msg->velocity.size()) {
            vel_it->second->current[k] = msg->velocity[i];
          }
          break;
        }
      }
    }

    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto it = action_joint_to_action_index_.find(msg->name[i]);
      if (it == action_joint_to_action_index_.end()) {
        continue;
      }
      const size_t k = it->second;
      if (i < msg->position.size()) {
        qpos_action_[k] = msg->position[i];
      }
      if (i < msg->velocity.size()) {
        qvel_action_[k] = msg->velocity[i];
      }
    }
  }

  void imuCB(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);

    imu_orientation_ = {{
      msg->orientation.w,
      msg->orientation.x,
      msg->orientation.y,
      msg->orientation.z}};

    auto ang_it = obs_terms_.find("base_ang_vel");
    if (ang_it != obs_terms_.end()) {
      ang_it->second->current[0] = msg->angular_velocity.x;
      ang_it->second->current[1] = msg->angular_velocity.y;
      ang_it->second->current[2] = msg->angular_velocity.z;
    }

    auto acc_it = obs_terms_.find("base_lin_acc");
    if (acc_it != obs_terms_.end()) {
      acc_it->second->current[0] = msg->linear_acceleration.x;
      acc_it->second->current[1] = msg->linear_acceleration.y;
      acc_it->second->current[2] = msg->linear_acceleration.z;
    }

    auto grav_it = obs_terms_.find("imu_projected_gravity");
    if (grav_it != obs_terms_.end()) {
      const std::array<double, 3> gravity_world = {{0.0, 0.0, -1.0}};
      const std::array<double, 3> gravity_body = rotateWorldToBody(imu_orientation_, gravity_world);
      grav_it->second->current[0] = gravity_body[0];
      grav_it->second->current[1] = gravity_body[1];
      grav_it->second->current[2] = gravity_body[2];
    }
  }

  void velCmdCB(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    auto cmd_it = obs_terms_.find("command");
    if (cmd_it != obs_terms_.end()) {
      cmd_it->second->current[0] = msg->linear.x;
      cmd_it->second->current[1] = msg->linear.y;
      cmd_it->second->current[2] = msg->angular.z;
    }
  }

  void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    auto vel_it = obs_terms_.find("base_lin_vel");
    if (vel_it != obs_terms_.end()) {
      vel_it->second->current[0] = msg->twist.twist.linear.x;
      vel_it->second->current[1] = msg->twist.twist.linear.y;
      vel_it->second->current[2] = msg->twist.twist.linear.z;
    }
  }

  PolicyConfig config_;

  std::map<std::string, std::shared_ptr<ObservationTerm>> obs_terms_;
  std::vector<std::string> obs_term_order_;

  std::string joint_states_topic_;
  std::string imu_topic_;
  std::string cmd_vel_topic_;
  std::string odom_topic_;
  std::string command_topic_;

  std::array<double, 4> imu_orientation_;
  std::mutex observation_mutex_;

  std::vector<int64_t> input_shape_;
  std::vector<double> action_;
  std::vector<double> qpos_action_;
  std::vector<double> qvel_action_;
  std::vector<double> tau_;
  std::vector<double> stiffness_aligned_;
  std::vector<double> damping_aligned_;
  std::unordered_map<std::string, size_t> action_joint_to_action_index_;
  std_msgs::msg::Float64MultiArray msg_command_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::TimerBase::SharedPtr timer_;

  Ort::Env env_;
  Ort::MemoryInfo mem_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  char * input_name_{nullptr};
  char * output_name_{nullptr};
};

}  // namespace pal_policy_inferer

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pal_policy_inferer::KangarooMjlabMonolithicNode>());
  rclcpp::shutdown();
  return 0;
}
