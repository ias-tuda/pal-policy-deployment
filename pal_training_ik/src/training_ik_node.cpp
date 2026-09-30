#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <urdf/model.h>
#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>

namespace {

template <typename T>
T clampValue(T value, T lower, T upper) {
  return std::max(lower, std::min(value, upper));
}

Eigen::Quaterniond kdlQuaternion(const KDL::Rotation& rotation) {
  double x, y, z, w;
  rotation.GetQuaternion(x, y, z, w);
  return Eigen::Quaterniond(w, x, y, z).normalized();
}

Eigen::Vector3d rotationError(const Eigen::Quaterniond& current,
                              const Eigen::Quaterniond& desired) {
  Eigen::Quaterniond error = desired * current.conjugate();
  if (error.w() < 0.0) {
    error.coeffs() *= -1.0;
  }
  const Eigen::AngleAxisd angle_axis(error.normalized());
  if (!std::isfinite(angle_axis.angle()) || std::abs(angle_axis.angle()) < 1e-12) {
    return Eigen::Vector3d::Zero();
  }
  return angle_axis.axis() * angle_axis.angle();
}

}  // namespace

class TrainingIkNode {
 public:
  TrainingIkNode() : private_nh_("~") {
    loadParameters();
    loadRobotModel();

    joint_state_sub_ = nh_.subscribe(joint_state_topic_, 1,
                                     &TrainingIkNode::jointStateCallback, this);
    action_sub_ = nh_.subscribe(action_topic_, 1,
                                &TrainingIkNode::actionCallback, this);
    command_pub_ = nh_.advertise<sensor_msgs::JointState>(command_topic_, 1);
    preview_pub_ = nh_.advertise<sensor_msgs::JointState>(preview_topic_, 1);
    timer_ = nh_.createTimer(ros::Duration(1.0 / ik_rate_),
                             &TrainingIkNode::timerCallback, this);

    ROS_INFO_STREAM("Training IK ready: " << base_link_ << " -> " << tip_link_
                    << ", " << arm_joint_names_.size() << " joints at " << ik_rate_
                    << " Hz, publication " << (publish_commands_ ? "enabled" : "disabled"));
  }

 private:
  void loadParameters() {
    private_nh_.param<std::string>("robot_description", robot_description_param_,
                                   "robot_description");
    private_nh_.param<std::string>("base_link", base_link_, "base_link");
    private_nh_.param<std::string>("tip_link", tip_link_, "arm_left_7_link");
    std::string policy_node_name;
    private_nh_.param<std::string>("policy_node_name", policy_node_name, "");
    if (!policy_node_name.empty() && policy_node_name[0] != '/') {
      policy_node_name.insert(0, "/");
    }
    // Topics come from our own override, else the policy config, else the default,
    // so the policy config is the single place they are configured.
    const auto topic_param = [&](const std::string& name, const std::string& fallback) {
      std::string value;
      if (private_nh_.getParam(name, value) && !value.empty()) return value;
      if (!policy_node_name.empty() &&
          ros::param::get(policy_node_name + "/" + name, value) && !value.empty()) {
        return value;
      }
      return fallback;
    };
    action_topic_ = topic_param("action_topic", "/pal_policy/raw_actions");
    joint_state_topic_ = topic_param("joint_state_topic", "/joint_states");
    private_nh_.param<std::string>("command_topic", command_topic_,
                                   "/subscriber_controller/desired_state");
    private_nh_.param<std::string>("preview_topic", preview_topic_,
                                   "/pal_training_ik/desired_state");
    private_nh_.param("ik_rate", ik_rate_, 500.0);
    private_nh_.param("state_timeout", state_timeout_, 0.1);
    private_nh_.param("action_timeout", action_timeout_, 0.1);
    private_nh_.param("delta_position_scale", delta_position_scale_, 0.01);
    private_nh_.param("delta_orientation_scale", delta_orientation_scale_, 0.05);
    private_nh_.param("damping", damping_, 0.05);
    private_nh_.param("max_dq", max_dq_, 0.02);
    private_nh_.param("position_weight", position_weight_, 1.0);
    private_nh_.param("orientation_weight", orientation_weight_, 1.0);
    private_nh_.param("joint_limit_weight", joint_limit_weight_, 1.0);
    private_nh_.param("posture_weight", posture_weight_, 0.02);
    private_nh_.param("publish_commands", publish_commands_, false);
    private_nh_.param<std::string>("gripper_joint_name", gripper_joint_name_,
                                   "gripper_left_finger_joint");
    private_nh_.param("gripper_default_position", gripper_default_position_, 0.0);
    private_nh_.param("gripper_scale", gripper_scale_, 0.9);
    private_nh_.param("gripper_lower", gripper_lower_, 0.0);
    private_nh_.param("gripper_upper", gripper_upper_, 0.95);
    private_nh_.param("gripper_max_step", gripper_max_step_, 0.10);

    if (!private_nh_.getParam("arm_joint_names", arm_joint_names_) ||
        arm_joint_names_.empty()) {
      throw std::runtime_error("~arm_joint_names is required");
    }
    if (!private_nh_.getParam("posture_target", posture_target_) ||
        posture_target_.size() != arm_joint_names_.size()) {
      throw std::runtime_error("~posture_target must match ~arm_joint_names");
    }
    std::string start_config_path;
    bool auto_discovered_start_config = false;
    private_nh_.param<std::string>("start_config_path", start_config_path, "");
    if (start_config_path.empty()) {
      if (!policy_node_name.empty()) {
        ros::param::get(policy_node_name + "/start_config_path", start_config_path);
        if (start_config_path.empty()) {
          std::string policy_model_path;
          if (!ros::param::get(policy_node_name + "/model_path", policy_model_path) ||
              policy_model_path.empty()) {
            throw std::runtime_error("cannot find grasp policy model_path for start config");
          }
          const auto separator = policy_model_path.find_last_of("/\\");
          start_config_path = policy_model_path.substr(0, separator + 1) +
                              "deployment_start_config.json";
          auto_discovered_start_config = true;
        }
      }
    }
    if (auto_discovered_start_config && !std::ifstream(start_config_path).good()) {
      ROS_WARN("No deployment_start_config.json beside ONNX; using IK config posture");
      start_config_path.clear();
    }
    if (!start_config_path.empty()) {
      boost::property_tree::ptree start_config;
      boost::property_tree::read_json(start_config_path, start_config);
      if (start_config.get<int>("schema_version") != 1) {
        throw std::runtime_error("unsupported start config schema_version");
      }
      std::unordered_set<std::string> start_joint_names;
      for (const auto& item : start_config.get_child("joint_names")) {
        if (!start_joint_names.insert(item.second.get_value<std::string>()).second) {
          throw std::runtime_error("duplicate joint name in start config");
        }
      }
      const auto& positions = start_config.get_child("joint_positions");
      for (size_t i = 0; i < arm_joint_names_.size(); ++i) {
        const auto& name = arm_joint_names_[i];
        if (!start_joint_names.count(name)) {
          throw std::runtime_error("start config missing joint " + name);
        }
        posture_target_[i] = positions.get<double>(name);
        if (!std::isfinite(posture_target_[i])) {
          throw std::runtime_error("non-finite start position for " + name);
        }
      }
      if (!start_joint_names.count(gripper_joint_name_)) {
        throw std::runtime_error("start config missing joint " + gripper_joint_name_);
      }
      gripper_default_position_ = positions.get<double>(gripper_joint_name_);
      if (!std::isfinite(gripper_default_position_)) {
        throw std::runtime_error("non-finite start position for " + gripper_joint_name_);
      }
      ROS_INFO("Loaded IK posture from %s", start_config_path.c_str());
    }
    gripper_target_ = gripper_default_position_;
    std::vector<double> tcp_offset;
    if (!private_nh_.getParam("tcp_offset", tcp_offset) || tcp_offset.size() != 3) {
      throw std::runtime_error("~tcp_offset must contain x, y, z");
    }
    tcp_offset_ = KDL::Vector(tcp_offset[0], tcp_offset[1], tcp_offset[2]);
  }

  void loadRobotModel() {
    std::string urdf_xml;
    if (!nh_.getParam(robot_description_param_, urdf_xml)) {
      throw std::runtime_error("Unable to read /" + robot_description_param_);
    }
    urdf::Model model;
    if (!model.initString(urdf_xml)) {
      throw std::runtime_error("Unable to parse robot_description");
    }
    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(model, tree) ||
        !tree.getChain(base_link_, tip_link_, chain_)) {
      throw std::runtime_error("Unable to build KDL chain " + base_link_ + " -> " + tip_link_);
    }
    joint_lower_.resize(arm_joint_names_.size());
    joint_upper_.resize(arm_joint_names_.size());
    std::unordered_map<std::string, std::size_t> arm_indices;
    for (std::size_t i = 0; i < arm_joint_names_.size(); ++i) {
      arm_indices[arm_joint_names_[i]] = i;
    }
    std::size_t chain_index = 0;
    for (const auto& segment : chain_.segments) {
      const std::string name = segment.getJoint().getName();
      if (segment.getJoint().getType() == KDL::Joint::None) continue;
      chain_joint_names_.push_back(name);
      const auto arm = arm_indices.find(name);
      if (arm != arm_indices.end()) {
        arm_chain_indices_.push_back(chain_index);
        const urdf::JointConstSharedPtr joint = model.getJoint(name);
        if (!joint || !joint->limits) {
          throw std::runtime_error("No URDF limits for " + name);
        }
        joint_lower_[arm->second] = joint->limits->lower;
        joint_upper_[arm->second] = joint->limits->upper;
      }
      ++chain_index;
    }
    if (arm_chain_indices_.size() != arm_joint_names_.size()) {
      throw std::runtime_error("KDL chain does not contain all configured arm joints");
    }
    for (std::size_t i = 0; i < arm_chain_indices_.size(); ++i) {
      if (chain_joint_names_[arm_chain_indices_[i]] != arm_joint_names_[i]) {
        throw std::runtime_error("Configured arm joints are not in KDL chain order");
      }
    }
    fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(chain_));
    jac_solver_.reset(new KDL::ChainJntToJacSolver(chain_));
  }

  bool fillJointArray(const sensor_msgs::JointState& msg, KDL::JntArray& q) const {
    std::unordered_map<std::string, std::size_t> indices;
    for (std::size_t i = 0; i < msg.name.size(); ++i) indices[msg.name[i]] = i;
    q.resize(chain_joint_names_.size());
    for (std::size_t i = 0; i < chain_joint_names_.size(); ++i) {
      const auto found = indices.find(chain_joint_names_[i]);
      if (found == indices.end() || found->second >= msg.position.size()) return false;
      q(i) = msg.position[found->second];
    }
    return true;
  }

  bool tcpPose(const KDL::JntArray& q, Eigen::Vector3d& position,
               Eigen::Quaterniond& orientation) const {
    KDL::Frame tip;
    if (fk_solver_->JntToCart(q, tip) < 0) return false;
    const KDL::Frame tcp = tip * KDL::Frame(tcp_offset_);
    position = Eigen::Vector3d(tcp.p.x(), tcp.p.y(), tcp.p.z());
    orientation = kdlQuaternion(tcp.M);
    return true;
  }

  void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg) {
    KDL::JntArray candidate;
    if (!fillJointArray(*msg, candidate)) {
      ROS_WARN_THROTTLE(2.0, "Joint state is missing one or more configured arm joints");
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    latest_joint_state_ = *msg;
    q_ = candidate;
    state_received_ = true;
    state_time_ = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
  }

  void actionCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
    if (msg->data.size() != 7) {
      ROS_ERROR_THROTTLE(1.0, "Training IK requires exactly 7 raw policy actions; got %zu",
                         msg->data.size());
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_received_) {
      ROS_WARN_THROTTLE(1.0, "Ignoring policy action until joint state is available");
      return;
    }
    Eigen::Vector3d measured_position;
    Eigen::Quaterniond measured_orientation;
    if (!tcpPose(q_, measured_position, measured_orientation)) return;

    desired_position_ = measured_position + delta_position_scale_ *
        Eigen::Vector3d(msg->data[0], msg->data[1], msg->data[2]);
    const Eigen::Vector3d rotation_vector = delta_orientation_scale_ *
        Eigen::Vector3d(msg->data[3], msg->data[4], msg->data[5]);
    const double angle = rotation_vector.norm();
    Eigen::Quaterniond delta = Eigen::Quaterniond::Identity();
    if (angle > 1e-6) delta = Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle));
    desired_orientation_ = (delta * measured_orientation).normalized();

    // Matches pal_mjlab RateLimitedJointPositionAction: scale and offset, clip, then limit
    // the change from the last commanded target once per policy step. The last target
    // starts at the default position, as on an MJLab reset.
    const double requested = clampValue(gripper_default_position_ +
        gripper_scale_ * msg->data[6], gripper_lower_, gripper_upper_);
    gripper_target_ += clampValue(requested - gripper_target_,
                                  -gripper_max_step_, gripper_max_step_);
    action_received_ = true;
    action_time_ = ros::Time::now();
  }

  void timerCallback(const ros::TimerEvent&) {
    KDL::JntArray q;
    Eigen::Vector3d desired_position;
    Eigen::Quaterniond desired_orientation;
    double gripper_target;
    ros::Time state_time;
    ros::Time action_time;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!state_received_ || !action_received_) return;
      q = q_;
      desired_position = desired_position_;
      desired_orientation = desired_orientation_;
      gripper_target = gripper_target_;
      state_time = state_time_;
      action_time = action_time_;
    }
    const ros::Time now = ros::Time::now();
    if ((now - state_time).toSec() > state_timeout_) {
      ROS_WARN_THROTTLE(1.0, "Training IK stopped: joint state is stale");
      return;
    }
    if ((now - action_time).toSec() > action_timeout_) {
      ROS_WARN_THROTTLE(1.0, "Training IK stopped: policy action is stale");
      return;
    }

    Eigen::Vector3d current_position;
    Eigen::Quaterniond current_orientation;
    if (!tcpPose(q, current_position, current_orientation)) return;

    KDL::Jacobian kdl_jacobian(chain_.getNrOfJoints());
    if (jac_solver_->JntToJac(q, kdl_jacobian) < 0) return;
    const Eigen::Vector3d tcp_offset_base = current_orientation *
        Eigen::Vector3d(tcp_offset_.x(), tcp_offset_.y(), tcp_offset_.z());
    kdl_jacobian.changeRefPoint(KDL::Vector(tcp_offset_base.x(), tcp_offset_base.y(),
                                            tcp_offset_base.z()));

    const std::size_t n = arm_joint_names_.size();
    Eigen::MatrixXd jacobian(6, n);
    for (std::size_t i = 0; i < n; ++i) {
      jacobian.col(i) = kdl_jacobian.data.col(arm_chain_indices_[i]);
    }
    Eigen::MatrixXd normal = position_weight_ * position_weight_ *
        jacobian.topRows(3).transpose() * jacobian.topRows(3) +
        orientation_weight_ * orientation_weight_ *
        jacobian.bottomRows(3).transpose() * jacobian.bottomRows(3);
    Eigen::VectorXd rhs = position_weight_ * position_weight_ *
        jacobian.topRows(3).transpose() * (desired_position - current_position) +
        orientation_weight_ * orientation_weight_ *
        jacobian.bottomRows(3).transpose() *
            rotationError(current_orientation, desired_orientation);

    for (std::size_t i = 0; i < n; ++i) {
      double limit_residual = 0.0;
      const double arm_q = q(arm_chain_indices_[i]);
      if (arm_q > joint_upper_[i]) limit_residual = joint_upper_[i] - arm_q;
      if (arm_q < joint_lower_[i]) limit_residual = joint_lower_[i] - arm_q;
      if (limit_residual != 0.0) {
        normal(i, i) += joint_limit_weight_ * joint_limit_weight_;
        rhs(i) += joint_limit_weight_ * joint_limit_weight_ * limit_residual;
      }
      normal(i, i) += posture_weight_ * posture_weight_ + damping_ * damping_;
      rhs(i) += posture_weight_ * posture_weight_ * (posture_target_[i] - arm_q);
    }
    const Eigen::VectorXd dq = normal.ldlt().solve(rhs);
    if (!dq.allFinite()) {
      ROS_ERROR_THROTTLE(1.0, "Training IK solve returned non-finite values");
      return;
    }

    sensor_msgs::JointState command;
    command.header.stamp = now;
    command.name = arm_joint_names_;
    command.name.push_back(gripper_joint_name_);
    command.position.resize(n + 1);
    command.velocity.assign(n + 1, 0.0);
    command.effort.assign(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      command.position[i] = q(arm_chain_indices_[i]) +
          clampValue(dq(i), -max_dq_, max_dq_);
    }
    command.position[n] = gripper_target;

    preview_pub_.publish(command);
    if (publish_commands_) command_pub_.publish(command);
    ROS_DEBUG_THROTTLE(1.0, "Training IK error: position=%.4f rotation=%.4f",
                       (desired_position - current_position).norm(),
                       rotationError(current_orientation, desired_orientation).norm());
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber joint_state_sub_;
  ros::Subscriber action_sub_;
  ros::Publisher command_pub_;
  ros::Publisher preview_pub_;
  ros::Timer timer_;

  std::string robot_description_param_, base_link_, tip_link_;
  std::string action_topic_, joint_state_topic_, command_topic_, preview_topic_;
  std::vector<std::string> arm_joint_names_;
  std::vector<std::string> chain_joint_names_;
  std::vector<std::size_t> arm_chain_indices_;
  std::vector<double> posture_target_, joint_lower_, joint_upper_;
  std::string gripper_joint_name_;
  double ik_rate_, state_timeout_, action_timeout_;
  double delta_position_scale_, delta_orientation_scale_;
  double damping_, max_dq_, position_weight_, orientation_weight_;
  ros::Time last_target_time_;
  double joint_limit_weight_, posture_weight_;
  double gripper_default_position_, gripper_scale_, gripper_lower_, gripper_upper_;
  double gripper_max_step_;
  bool publish_commands_;

  KDL::Chain chain_;
  KDL::Vector tcp_offset_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_;

  std::mutex mutex_;
  sensor_msgs::JointState latest_joint_state_;
  KDL::JntArray q_;
  Eigen::Vector3d desired_position_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond desired_orientation_ = Eigen::Quaterniond::Identity();
  double gripper_target_ = 0.0;
  ros::Time state_time_, action_time_;
  bool state_received_ = false;
  bool action_received_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "training_ik");
  try {
    TrainingIkNode node;
    ros::spin();
  } catch (const std::exception& exception) {
    ROS_FATAL("Training IK failed to start: %s", exception.what());
    return 1;
  }
  return 0;
}
