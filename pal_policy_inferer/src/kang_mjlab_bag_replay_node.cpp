#include <ros/ros.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
// ros msgs
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <onnxruntime_cxx_api.h>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
// core libraries
#include <vector>
#include <algorithm>
#include <array>
#include <mutex>
#include <iostream>
#include <deque>
#include <sstream>
#include <map>
#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fs = boost::filesystem;
#define foreach BOOST_FOREACH

// ---- OBSERVATION TERM REGISTRY ----
// Each term has: name, dimension, current values, history buffer

struct ObservationTerm {
    std::string name;
    int dim;
    std::vector<double> current;
    std::deque<std::vector<double>> history;
    
    ObservationTerm(const std::string& n, int d) : name(n), dim(d), current(d, 0.0) {}
    
    void initHistory(int history_length) {
        history.clear();
        for (int i = 0; i < history_length; ++i) {
            history.push_back(std::vector<double>(dim, 0.0));
        }
    }
    
    void updateHistory() {
        history.pop_front();
        history.push_back(current);
    }
};

// ---- JOINTS TO EXCLUDE FROM ACTIONS ----
static const std::vector<std::string> excluded_action_joints = {
    "leg_left_femur_joint", "leg_left_knee_joint",
    "leg_right_femur_joint", "leg_right_knee_joint"
};

// ---- JOINTS FROZEN IN SOME POLICIES ----
static const std::vector<std::string> frozen_arm_joints = {
    "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint",
    "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"
};

// ---- POLICY CONFIGURATION ----
struct PolicyConfig {
    std::string run_path;
    std::vector<std::string> joint_names;
    std::vector<std::string> observation_joint_names;
    std::vector<std::string> action_joint_names;
    std::vector<double> default_joint_pos;
    std::vector<double> observation_default_joint_pos;
    std::vector<double> action_offset;
    std::vector<double> action_scale;
    std::vector<double> joint_stiffness;
    std::vector<double> joint_damping;
    std::vector<std::string> observation_names;
    int obs_dim;
    int action_dim;
    int history_length;
    int single_obs_dim;
};

// ---- GLOBALS ----
PolicyConfig config;

// Observation terms - dynamically populated based on metadata
std::map<std::string, std::shared_ptr<ObservationTerm>> obs_terms;
std::vector<std::string> obs_term_order;  // Maintains order from metadata

// IMU-derived values
Eigen::Quaterniond imu_orientation(1.0, 0.0, 0.0, 0.0);
bool received_imu = false;

std::mutex observation_mutex;

// ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rl");
std::unique_ptr<Ort::Session> sess;
Ort::MemoryInfo mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
std::vector<const char*> in_names, out_names;

std::vector<int64_t> input_shape;
std::vector<double> action;
std::vector<double> zero_action;

ros::Publisher pub_sub_cmd;
ros::Publisher pub_default_pos;
ros::Publisher pub_tau_computed;

// Impedance gains indexed by action_joint_names, loaded once in main
std::vector<double> impedance_ks_global;
std::vector<double> impedance_kd_global;

// ---- DEBUG PUBLISHERS ----
ros::Publisher pub_dbg_ang_vel;
ros::Publisher pub_dbg_lin_acc;
ros::Publisher pub_dbg_proj_grav;
ros::Publisher pub_dbg_proj_grav_raw;        // projected_gravity without 180deg correction
ros::Publisher pub_dbg_proj_grav_corrected;  // projected_gravity with 180deg correction
ros::Publisher pub_dbg_joint_pos;
ros::Publisher pub_dbg_joint_pos_absolute;   // joint_pos + default_joint_pos
ros::Publisher pub_dbg_joint_vel;
ros::Publisher pub_dbg_command;
ros::Publisher pub_dbg_actions_input;
ros::Publisher pub_dbg_actions;
ros::Publisher pub_dbg_actions_scaled;       // action_scale * y_f + default_joint_pos (= desired_state.position)

bool bag_replay_mode = false;
bool replay_actions_from_bag = true;

std_msgs::Float64MultiArray makeDebugMsg(const std::vector<double>& data) {
    std_msgs::Float64MultiArray msg;
    msg.data = data;
    return msg;
}

// ---- UTILITY FUNCTIONS ----

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<double> parseDoubles(const std::string& str) {
    std::vector<double> values;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        values.push_back(std::stod(token));
    }
    return values;
}

bool isExcludedJoint(const std::string& joint_name) {
    return std::find(excluded_action_joints.begin(), excluded_action_joints.end(), joint_name) 
           != excluded_action_joints.end();
}

bool hasObservationTerm(const std::string& term_name) {
    return obs_terms.find(term_name) != obs_terms.end();
}

bool usesImuObservation() {
    return hasObservationTerm("base_ang_vel") ||
           hasObservationTerm("base_lin_acc") ||
           hasObservationTerm("projected_gravity") ||
           hasObservationTerm("imu_projected_gravity");
}

bool usesJointStateObservation() {
    return hasObservationTerm("joint_pos") ||
           hasObservationTerm("joint_vel");
}

bool isFrozenArmJoint(const std::string& joint_name) {
    return std::find(frozen_arm_joints.begin(), frozen_arm_joints.end(), joint_name)
           != frozen_arm_joints.end();
}

std::vector<std::string> filterJoints(bool exclude_passive, bool exclude_frozen_arms) {
    std::vector<std::string> joints;
    for (const auto& joint_name : config.joint_names) {
        if (exclude_passive && isExcludedJoint(joint_name)) continue;
        if (exclude_frozen_arms && isFrozenArmJoint(joint_name)) continue;
        joints.push_back(joint_name);
    }
    return joints;
}

double getDefaultJointPos(const std::string& joint_name) {
    auto it = std::find(config.joint_names.begin(), config.joint_names.end(), joint_name);
    if (it == config.joint_names.end()) {
        ROS_WARN("Joint '%s' has no default position in metadata, using 0.0", joint_name.c_str());
        return 0.0;
    }
    size_t idx = static_cast<size_t>(std::distance(config.joint_names.begin(), it));
    if (idx >= config.default_joint_pos.size()) {
        ROS_WARN("Joint '%s' default position index out of range, using 0.0", joint_name.c_str());
        return 0.0;
    }
    return config.default_joint_pos[idx];
}

int calculateSingleObsDim(const std::vector<std::string>& observation_joint_names,
                          const std::vector<std::string>& action_joint_names) {
    int dim = 0;
    for (const auto& term_name : config.observation_names) {
        if (term_name == "base_ang_vel") dim += 3;
        else if (term_name == "base_lin_vel") dim += 3;
        else if (term_name == "base_lin_acc") dim += 3;
        else if (term_name == "projected_gravity") dim += 3;
        else if (term_name == "imu_projected_gravity") dim += 3;
        else if (term_name == "command") dim += 3;
        else if (term_name == "joint_pos") dim += static_cast<int>(observation_joint_names.size());
        else if (term_name == "joint_vel") dim += static_cast<int>(observation_joint_names.size());
        else if (term_name == "actions") dim += static_cast<int>(action_joint_names.size());
        else {
            ROS_WARN("Unknown observation term: %s, assuming dim=3", term_name.c_str());
            dim += 3;
        }
    }
    return dim;
}

// ---- OBSERVATION TERM DIMENSION CALCULATOR ----
// Returns the dimension for a given observation term name

int getObservationTermDim(const std::string& term_name) {
    if (term_name == "base_ang_vel") return 3;
    if (term_name == "base_lin_vel") return 3;
    if (term_name == "base_lin_acc") return 3;
    if (term_name == "projected_gravity") return 3;
    if (term_name == "imu_projected_gravity") return 3;
    if (term_name == "command") return 3;  // twist command: vx, vy, wz
    if (term_name == "joint_pos") return static_cast<int>(config.observation_joint_names.size());
    if (term_name == "joint_vel") return static_cast<int>(config.observation_joint_names.size());
    if (term_name == "actions") return static_cast<int>(config.action_joint_names.size());
    
    ROS_WARN("Unknown observation term: %s, assuming dim=3", term_name.c_str());
    return 3;
}

// ---- OBSERVATION TERM INITIALIZATION ----

void initializeObservationTerms() {
    obs_terms.clear();
    obs_term_order.clear();
    
    for (const auto& term_name : config.observation_names) {
        int dim = getObservationTermDim(term_name);
        auto term = std::make_shared<ObservationTerm>(term_name, dim);
        term->initHistory(config.history_length);
        obs_terms[term_name] = term;
        obs_term_order.push_back(term_name);
        
        ROS_INFO("  Term '%s': dim=%d", term_name.c_str(), dim);
    }
}

// ---- ONNX METADATA PARSING ----

bool loadPolicyConfig(const std::string& model_path) {
    Ort::SessionOptions session_options;
    Ort::Session temp_session(env, model_path.c_str(), session_options);
    
    Ort::ModelMetadata metadata = temp_session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<Ort::AllocatedStringPtr> keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    
    std::map<std::string, std::string> metadata_map;
    for (auto& key_ptr : keys) {
        std::string key = key_ptr.get();
        Ort::AllocatedStringPtr value_ptr = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
        if (value_ptr) {
            metadata_map[key] = value_ptr.get();
        }
    }
    
    // Parse required metadata
    if (metadata_map.find("run_path") != metadata_map.end()) {
        config.run_path = metadata_map["run_path"];
    }
    
    if (metadata_map.find("joint_names") == metadata_map.end()) {
        ROS_FATAL("Missing 'joint_names' in ONNX metadata");
        return false;
    }
    config.joint_names = splitString(metadata_map["joint_names"], ',');
    
    if (metadata_map.find("default_joint_pos") == metadata_map.end()) {
        ROS_FATAL("Missing 'default_joint_pos' in ONNX metadata");
        return false;
    }
    config.default_joint_pos = parseDoubles(metadata_map["default_joint_pos"]);
    
    if (metadata_map.find("action_scale") == metadata_map.end()) {
        ROS_FATAL("Missing 'action_scale' in ONNX metadata");
        return false;
    }
    config.action_scale = parseDoubles(metadata_map["action_scale"]);
    
    if (metadata_map.find("observation_names") == metadata_map.end()) {
        ROS_FATAL("Missing 'observation_names' in ONNX metadata");
        return false;
    }
    config.observation_names = splitString(metadata_map["observation_names"], ',');
    
    // Optional metadata
    if (metadata_map.find("joint_stiffness") != metadata_map.end()) {
        config.joint_stiffness = parseDoubles(metadata_map["joint_stiffness"]);
    }
    if (metadata_map.find("joint_damping") != metadata_map.end()) {
        config.joint_damping = parseDoubles(metadata_map["joint_damping"]);
    }
    
    // Get input/output dimensions
    Ort::TypeInfo input_info = temp_session.GetInputTypeInfo(0);
    auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    
    Ort::TypeInfo output_info = temp_session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();
    
    config.obs_dim = static_cast<int>(input_dims[1]);
    config.action_dim = static_cast<int>(output_dims[1]);

    // Build joint sets. Newer exports may provide explicit metadata. For older
    // exports, infer the frozen-arm variant only when the ONNX dimensions require it.
    if (metadata_map.find("observation_joint_names") != metadata_map.end()) {
        config.observation_joint_names = splitString(metadata_map["observation_joint_names"], ',');
    }
    if (metadata_map.find("action_joint_names") != metadata_map.end()) {
        config.action_joint_names = splitString(metadata_map["action_joint_names"], ',');
    }

    std::vector<std::string> legacy_obs_joints = filterJoints(false, false);
    std::vector<std::string> frozen_obs_joints = filterJoints(false, true);
    std::vector<std::string> legacy_action_joints = filterJoints(true, false);
    std::vector<std::string> frozen_action_joints = filterJoints(true, true);

    if (config.action_joint_names.empty()) {
        if (config.action_dim == static_cast<int>(frozen_action_joints.size())) {
            config.action_joint_names = frozen_action_joints;
            ROS_INFO("Inferred frozen-arm action set from ONNX action dim: %d", config.action_dim);
        } else {
            config.action_joint_names = legacy_action_joints;
        }
    }

    if (config.observation_joint_names.empty()) {
        int frozen_single = calculateSingleObsDim(frozen_obs_joints, config.action_joint_names);
        int legacy_single = calculateSingleObsDim(legacy_obs_joints, config.action_joint_names);
        if (frozen_single > 0 && config.obs_dim % frozen_single == 0) {
            config.observation_joint_names = frozen_obs_joints;
            ROS_INFO("Inferred frozen-arm observation joint set from ONNX obs dim: %d", config.obs_dim);
        } else if (legacy_single > 0 && config.obs_dim % legacy_single == 0) {
            config.observation_joint_names = legacy_obs_joints;
        } else {
            // Keep the old behavior so the final fatal message reports the mismatch.
            config.observation_joint_names = legacy_obs_joints;
        }
    }

    config.action_offset.clear();
    for (const auto& joint_name : config.action_joint_names) {
        config.action_offset.push_back(getDefaultJointPos(joint_name));
    }

    config.observation_default_joint_pos.clear();
    for (const auto& joint_name : config.observation_joint_names) {
        config.observation_default_joint_pos.push_back(getDefaultJointPos(joint_name));
    }
    
    // Calculate single observation dimension from term names
    config.single_obs_dim = 0;
    for (const auto& term_name : config.observation_names) {
        config.single_obs_dim += getObservationTermDim(term_name);
    }
    
    // Calculate history length
    if (config.obs_dim % config.single_obs_dim == 0) {
        config.history_length = config.obs_dim / config.single_obs_dim;
    } else {
        ROS_FATAL("Observation dimension %d is not divisible by single obs dim %d",
                  config.obs_dim, config.single_obs_dim);
        return false;
    }
    
    // Validate
    if (config.action_dim != static_cast<int>(config.action_joint_names.size())) {
        ROS_FATAL("Action dimension mismatch: ONNX=%d, inferred joints=%zu",
                 config.action_dim, config.action_joint_names.size());
        return false;
    }
    if (config.action_dim != static_cast<int>(config.action_scale.size())) {
        ROS_FATAL("Action scale dimension mismatch: ONNX=%d, action_scale=%zu",
                  config.action_dim, config.action_scale.size());
        return false;
    }
    
    return true;
}

// ---- BUILD OBSERVATION VECTOR (TERM-MAJOR FORMAT) ----

std::vector<float> buildObservationVector() {
    std::vector<float> obs;
    obs.reserve(config.obs_dim);
    
    // For each term in order, append all history timesteps
    for (const auto& term_name : obs_term_order) {
        auto& term = obs_terms[term_name];
        for (int t = 0; t < config.history_length; ++t) {
            for (int i = 0; i < term->dim; ++i) {
                obs.push_back(static_cast<float>(term->history[t][i]));
            }
        }
    }
    
    return obs;
}

// ---- UPDATE ALL HISTORY BUFFERS ----

void updateAllHistory() {
    for (auto& pair : obs_terms) {
        pair.second->updateHistory();
    }
}

void publishDefaultActionCommand() {
    sensor_msgs::JointState cmd_msg;
    cmd_msg.header.stamp = ros::Time::now();
    cmd_msg.name = config.action_joint_names;
    cmd_msg.position = config.action_offset;
    cmd_msg.velocity = zero_action;
    cmd_msg.effort = zero_action;
    pub_sub_cmd.publish(cmd_msg);
}

void publishPolicyInputActions(const ros::Time& stamp) {
    if (!pub_dbg_actions_input || obs_terms.find("actions") == obs_terms.end()) {
        return;
    }

    sensor_msgs::JointState js;
    js.header.stamp = stamp;
    js.name = config.action_joint_names;
    js.position = obs_terms["actions"]->current;
    pub_dbg_actions_input.publish(js);
}

// ---- POLICY INFERENCE ----

void runPolicyOnce(const ros::Time& stamp) {
    std::vector<float> obs_input;
    {
        std::lock_guard<std::mutex> lk(observation_mutex);
        
        // Update history with current observations
        updateAllHistory();
        
        // Build flattened observation vector
        obs_input = buildObservationVector();
    }

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(mem, obs_input.data(), obs_input.size(),
                                        input_shape.data(), input_shape.size());

    auto outputs = sess->Run(Ort::RunOptions{nullptr},
                             in_names.data(), &input_tensor, 1,
                             out_names.data(), 1);

    const float* y_f = outputs.front().GetTensorData<float>();

    for (int i = 0; i < config.action_dim; ++i) {
        action[i] = config.action_scale[i] * y_f[i] + config.action_offset[i];
    }
    
    // In live mode, actions is the previous raw network output.
    // In bag replay mode, actions must stay exactly as recorded in /policy_input/actions.
    if (!bag_replay_mode || !replay_actions_from_bag) {
        std::lock_guard<std::mutex> lk(observation_mutex);
        if (obs_terms.find("actions") != obs_terms.end()) {
            auto& actions_term = obs_terms["actions"];
            for (int i = 0; i < config.action_dim; ++i) {
                actions_term->current[i] = y_f[i];
            }
        }
    }

    sensor_msgs::JointState cmd_msg;
    cmd_msg.header.stamp = stamp;
    cmd_msg.name = config.action_joint_names;
    cmd_msg.position = action;
    cmd_msg.velocity = zero_action;
    cmd_msg.effort = zero_action;

    pub_sub_cmd.publish(cmd_msg);

    // default_joint_pos only belongs in /policy_input when the loaded policy uses it.
    if (pub_default_pos) {
        sensor_msgs::JointState def_msg;
        def_msg.header.stamp = stamp;
        def_msg.name = config.joint_names;
        def_msg.position = config.default_joint_pos;
        pub_default_pos.publish(def_msg);
    }

    // raw network output (before scale/offset)
    std::vector<double> raw_actions(config.action_dim);
    for (int i = 0; i < config.action_dim; ++i)
        raw_actions[i] = y_f[i];
    {
        sensor_msgs::JointState js;
        js.header.stamp = stamp;
        js.name = config.action_joint_names;
        js.position = raw_actions;
        pub_dbg_actions.publish(js);
    }
    publishPolicyInputActions(stamp);

    // scaled output = action_scale * y_f + default_joint_pos (identical to desired_state.position)
    {
        sensor_msgs::JointState js;
        js.header.stamp = stamp;
        js.name = config.action_joint_names;
        js.position = action;
        pub_dbg_actions_scaled.publish(js);
    }

    // Command mit 50Hz publishen (nicht nur bei neuer cmd_vel Message)
    if (obs_terms.find("command") != obs_terms.end()) {
        auto& c = obs_terms["command"]->current;
        geometry_msgs::Vector3Stamped v;
        v.header.stamp = stamp;
        v.vector.x = c.size() > 0 ? c[0] : 0.0;
        v.vector.y = c.size() > 1 ? c[1] : 0.0;
        v.vector.z = c.size() > 2 ? c[2] : 0.0;
        pub_dbg_command.publish(v);
    }
}

void timerCallback(const ros::TimerEvent&) {
    runPolicyOnce(ros::Time::now());
}

// ---- SENSOR CALLBACKS ----

void jsCB(const sensor_msgs::JointState::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    // Update joint_pos term
    if (obs_terms.find("joint_pos") != obs_terms.end()) {
        auto& term = obs_terms["joint_pos"];
        for (size_t k = 0; k < config.observation_joint_names.size(); ++k) {
            for (size_t i = 0; i < msg->name.size(); ++i) {
                if (msg->name[i] == config.observation_joint_names[k]) {
                    term->current[k] = msg->position[i] - config.observation_default_joint_pos[k];
                    break;
                }
            }
        }
        {
            sensor_msgs::JointState js;
            js.header.stamp = msg->header.stamp;
            js.name = config.observation_joint_names;
            js.position = term->current;
            if (pub_dbg_joint_pos) {
                pub_dbg_joint_pos.publish(js);
            }
        }

        // absolute = relative + default_joint_pos
        std::vector<double> abs_pos(config.observation_joint_names.size());
        for (size_t k = 0; k < config.observation_joint_names.size(); ++k)
            abs_pos[k] = term->current[k] + config.observation_default_joint_pos[k];
        {
            sensor_msgs::JointState js;
            js.header.stamp = msg->header.stamp;
            js.name = config.observation_joint_names;
            js.position = abs_pos;
            if (pub_dbg_joint_pos_absolute) {
                pub_dbg_joint_pos_absolute.publish(js);
            }
        }
    }

    // Update joint_vel term
    if (obs_terms.find("joint_vel") != obs_terms.end()) {
        auto& term = obs_terms["joint_vel"];
        for (size_t k = 0; k < config.observation_joint_names.size(); ++k) {
            for (size_t i = 0; i < msg->name.size(); ++i) {
                if (msg->name[i] == config.observation_joint_names[k]) {
                    term->current[k] = msg->velocity[i];
                    break;
                }
            }
        }
        {
            sensor_msgs::JointState js;
            js.header.stamp = msg->header.stamp;
            js.name = config.observation_joint_names;
            js.velocity = term->current;
            if (pub_dbg_joint_vel) {
                pub_dbg_joint_vel.publish(js);
            }
        }
    }

    // Compute tau = ks*(desired_pos - actual_pos) + kd*(-actual_vel)
    if (!impedance_ks_global.empty() && !action.empty()) {
        sensor_msgs::JointState tau_msg;
        tau_msg.header.stamp = msg->header.stamp;
        tau_msg.name = config.action_joint_names;
        tau_msg.effort.resize(config.action_joint_names.size(), 0.0);
        for (size_t k = 0; k < config.action_joint_names.size(); ++k) {
            for (size_t i = 0; i < msg->name.size(); ++i) {
                if (msg->name[i] == config.action_joint_names[k]) {
                    tau_msg.effort[k] = impedance_ks_global[k] * (action[k] - msg->position[i])
                                      + impedance_kd_global[k] * (-msg->velocity[i]);
                    break;
                }
            }
        }
        pub_tau_computed.publish(tau_msg);
    }
}

void imuCB(const sensor_msgs::Imu::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);

    // Store orientation for projected gravity calculation
    imu_orientation = Eigen::Quaterniond(
        msg->orientation.w,
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z
    );

    ROS_INFO_THROTTLE(1.0, "[IMU DEBUG] raw orientation  w=%.4f x=%.4f y=%.4f z=%.4f",
        msg->orientation.w, msg->orientation.x,
        msg->orientation.y, msg->orientation.z);
    ROS_INFO_THROTTLE(1.0, "[IMU DEBUG] raw ang_vel      x=%.4f y=%.4f z=%.4f",
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    ROS_INFO_THROTTLE(1.0, "[IMU DEBUG] raw lin_acc      x=%.4f y=%.4f z=%.4f",
        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

    // Update base_ang_vel term
    if (obs_terms.find("base_ang_vel") != obs_terms.end()) {
        auto& term = obs_terms["base_ang_vel"];
        term->current[0] = msg->angular_velocity.x;
        term->current[1] = msg->angular_velocity.y;
        term->current[2] = msg->angular_velocity.z;
        geometry_msgs::Vector3Stamped v;
        v.header.stamp = msg->header.stamp;
        v.vector.x = term->current[0];
        v.vector.y = term->current[1];
        v.vector.z = term->current[2];
        if (pub_dbg_ang_vel) {
            pub_dbg_ang_vel.publish(v);
        }
    }

    // Update base_lin_acc term
    if (obs_terms.find("base_lin_acc") != obs_terms.end()) {
        auto& term = obs_terms["base_lin_acc"];
        term->current[0] = msg->linear_acceleration.x;
        term->current[1] = msg->linear_acceleration.y;
        term->current[2] = msg->linear_acceleration.z;
        geometry_msgs::Vector3Stamped v;
        v.header.stamp = msg->header.stamp;
        v.vector.x = term->current[0];
        v.vector.y = term->current[1];
        v.vector.z = term->current[2];
        if (pub_dbg_lin_acc) {
            pub_dbg_lin_acc.publish(v);
        }
    }

    // Update projected_gravity term (gravity vector in body frame)
    static const Eigen::Quaterniond imu_correction(
        Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())  // 180° around X
        // or try UnitY() if X doesn't work
    );

    std::string grav_key = obs_terms.count("imu_projected_gravity") ?
                           "imu_projected_gravity" : "projected_gravity";
    if (obs_terms.find(grav_key) != obs_terms.end()) {
        auto& term = obs_terms[grav_key];

        Eigen::Vector3d gravity_world(0.0, 0.0, -1.0);
        Eigen::Quaterniond q_conj = imu_orientation.conjugate();

        // RAW: without 180deg correction
        Eigen::Vector3d gravity_body_raw = q_conj * gravity_world;
        if (pub_dbg_proj_grav_raw) {
            geometry_msgs::Vector3Stamped v;
            v.header.stamp = msg->header.stamp;
            v.vector.x = gravity_body_raw.x();
            v.vector.y = gravity_body_raw.y();
            v.vector.z = gravity_body_raw.z();
            pub_dbg_proj_grav_raw.publish(v);
        }

        // CORRECTED: with 180deg correction around X (debug/reference only)
        Eigen::Vector3d gravity_body_corrected = imu_correction * gravity_body_raw;
        if (pub_dbg_proj_grav_corrected) {
            geometry_msgs::Vector3Stamped v;
            v.header.stamp = msg->header.stamp;
            v.vector.x = gravity_body_corrected.x();
            v.vector.y = gravity_body_corrected.y();
            v.vector.z = gravity_body_corrected.z();
            pub_dbg_proj_grav_corrected.publish(v);
        }

        // MuJoCo actor observation imu_projected_gravity is +Z when upright.
        // Feed the raw body-frame gravity to the policy to match training.
        Eigen::Vector3d gravity_body = gravity_body_raw;

        ROS_INFO_THROTTLE(1.0, "[IMU DEBUG] imu_orientation.conjugate()  w=%.4f x=%.4f y=%.4f z=%.4f",
            q_conj.w(), q_conj.x(), q_conj.y(), q_conj.z());
        ROS_INFO_THROTTLE(1.0, "[IMU DEBUG] projected_gravity raw        x=%.4f y=%.4f z=%.4f",
            gravity_body_raw.x(), gravity_body_raw.y(), gravity_body_raw.z());
        ROS_INFO_THROTTLE(1.0, "[IMU DEBUG] projected_gravity corrected  x=%.4f y=%.4f z=%.4f  (norm=%.4f)",
            gravity_body_corrected.x(), gravity_body_corrected.y(), gravity_body_corrected.z(), gravity_body_corrected.norm());
        ROS_INFO_THROTTLE(1.0, "[IMU DEBUG] projected_gravity policy     x=%.4f y=%.4f z=%.4f  (norm=%.4f)",
            gravity_body.x(), gravity_body.y(), gravity_body.z(), gravity_body.norm());

        term->current[0] = gravity_body.x();
        term->current[1] = gravity_body.y();
        term->current[2] = gravity_body.z();
        {
            geometry_msgs::Vector3Stamped v;
            v.header.stamp = msg->header.stamp;
            v.vector.x = term->current[0];
            v.vector.y = term->current[1];
            v.vector.z = term->current[2];
            if (pub_dbg_proj_grav) {
                pub_dbg_proj_grav.publish(v);
            }
        }
    }

    received_imu = true;
    
    // Update base_lin_vel term if present (would need odometry, placeholder)
    if (obs_terms.find("base_lin_vel") != obs_terms.end()) {
        // This would typically come from odometry or state estimation
        // For now, leaving as zeros - could subscribe to /odom
        ROS_WARN_ONCE("base_lin_vel term present but not populated - need odometry source");
    }
}

void velCmdCB(const geometry_msgs::Twist::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    // Update command term
    if (obs_terms.find("command") != obs_terms.end()) {
        auto& term = obs_terms["command"];
        term->current[0] = msg->linear.x;
        term->current[1] = msg->linear.y;
        term->current[2] = msg->angular.z;
    }
}

// Optional: odometry callback for base_lin_vel
void odomCB(const nav_msgs::Odometry::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    if (obs_terms.find("base_lin_vel") != obs_terms.end()) {
        auto& term = obs_terms["base_lin_vel"];
        term->current[0] = msg->twist.twist.linear.x;
        term->current[1] = msg->twist.twist.linear.y;
        term->current[2] = msg->twist.twist.linear.z;
    }
}

// ---- BAG REPLAY CALLBACKS ----

void setVectorTermFromBag(const std::string& term_name,
                          const geometry_msgs::Vector3Stamped::ConstPtr& msg,
                          const ros::Publisher& pub) {
    if (!msg || obs_terms.find(term_name) == obs_terms.end()) {
        return;
    }

    std::lock_guard<std::mutex> lk(observation_mutex);
    auto& term = obs_terms[term_name];
    term->current[0] = msg->vector.x;
    term->current[1] = msg->vector.y;
    term->current[2] = msg->vector.z;
    if (pub) {
        pub.publish(*msg);
    }
}

void setJointPositionTermFromBag(const std::string& term_name,
                                 const sensor_msgs::JointState::ConstPtr& msg,
                                 const ros::Publisher& pub) {
    if (!msg || obs_terms.find(term_name) == obs_terms.end()) {
        return;
    }

    std::lock_guard<std::mutex> lk(observation_mutex);
    auto& term = obs_terms[term_name];
    const size_t n = std::min(term->current.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i) {
        term->current[i] = msg->position[i];
    }
    if (pub) {
        pub.publish(*msg);
    }
}

void setJointVelocityTermFromBag(const sensor_msgs::JointState::ConstPtr& msg) {
    if (!msg || obs_terms.find("joint_vel") == obs_terms.end()) {
        return;
    }

    std::lock_guard<std::mutex> lk(observation_mutex);
    auto& term = obs_terms["joint_vel"];
    const size_t n = std::min(term->current.size(), msg->velocity.size());
    for (size_t i = 0; i < n; ++i) {
        term->current[i] = msg->velocity[i];
    }
    if (pub_dbg_joint_vel) {
        pub_dbg_joint_vel.publish(*msg);
    }
}

void publishJointStateDebugOnly(const sensor_msgs::JointState::ConstPtr& msg,
                                const ros::Publisher& pub) {
    if (msg && pub) {
        pub.publish(*msg);
    }
}

bool replayBagInputs(const std::string& bag_path,
                     const std::string& trigger_topic,
                     double rate_scale,
                     bool loop_replay) {
    if (bag_path.empty()) {
        ROS_FATAL("Missing required private param _bag:=/path/to/file.bag");
        return false;
    }

    std::vector<std::string> topics;
    topics.push_back("/policy_input/base_ang_vel");
    topics.push_back("/policy_input/joint_pos_rel");
    topics.push_back("/policy_input/joint_pos_abs");
    topics.push_back("/policy_input/joint_vel");
    topics.push_back("/policy_input/actions");
    topics.push_back("/policy_input/command");
    topics.push_back("/policy_input/imu_projected_gravity");
    topics.push_back("/policy_input/projected_gravity");
    topics.push_back("/policy_input/base_lin_acc");

    rate_scale = std::max(0.01, rate_scale);

    do {
        rosbag::Bag bag;
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (const rosbag::BagException& e) {
            ROS_FATAL("Failed to open bag '%s': %s", bag_path.c_str(), e.what());
            return false;
        }

        rosbag::View view(bag, rosbag::TopicQuery(topics));
        ROS_INFO("Replaying policy inputs from %s", bag_path.c_str());
        ROS_INFO("Inference trigger topic: %s", trigger_topic.c_str());

        bool have_last_trigger_stamp = false;
        ros::Time last_trigger_stamp;
        size_t inference_count = 0;

        foreach (rosbag::MessageInstance const m, view) {
            if (!ros::ok()) {
                bag.close();
                return true;
            }

            const std::string topic = m.getTopic();

            if (topic == "/policy_input/base_ang_vel") {
                setVectorTermFromBag("base_ang_vel", m.instantiate<geometry_msgs::Vector3Stamped>(), pub_dbg_ang_vel);
            } else if (topic == "/policy_input/base_lin_acc") {
                setVectorTermFromBag("base_lin_acc", m.instantiate<geometry_msgs::Vector3Stamped>(), pub_dbg_lin_acc);
            } else if (topic == "/policy_input/command") {
                setVectorTermFromBag("command", m.instantiate<geometry_msgs::Vector3Stamped>(), pub_dbg_command);
            } else if (topic == "/policy_input/imu_projected_gravity") {
                setVectorTermFromBag("imu_projected_gravity", m.instantiate<geometry_msgs::Vector3Stamped>(), pub_dbg_proj_grav);
            } else if (topic == "/policy_input/projected_gravity") {
                setVectorTermFromBag("projected_gravity", m.instantiate<geometry_msgs::Vector3Stamped>(), pub_dbg_proj_grav);
            } else if (topic == "/policy_input/joint_pos_rel") {
                setJointPositionTermFromBag("joint_pos", m.instantiate<sensor_msgs::JointState>(), pub_dbg_joint_pos);
            } else if (topic == "/policy_input/joint_pos_abs") {
                publishJointStateDebugOnly(m.instantiate<sensor_msgs::JointState>(), pub_dbg_joint_pos_absolute);
            } else if (topic == "/policy_input/joint_vel") {
                setJointVelocityTermFromBag(m.instantiate<sensor_msgs::JointState>());
            } else if (topic == "/policy_input/actions") {
                setJointPositionTermFromBag("actions", m.instantiate<sensor_msgs::JointState>(), pub_dbg_actions_input);
            }

            if (topic == trigger_topic) {
                const ros::Time stamp = m.getTime();
                if (have_last_trigger_stamp) {
                    const double dt = (stamp - last_trigger_stamp).toSec() / rate_scale;
                    if (dt > 0.0 && dt < 1.0) {
                        ros::WallDuration(dt).sleep();
                    }
                }
                last_trigger_stamp = stamp;
                have_last_trigger_stamp = true;

                runPolicyOnce(ros::Time::now());
                ++inference_count;
            }
        }

        bag.close();
        ROS_INFO("Finished replay pass, ran %zu inferences", inference_count);
    } while (ros::ok() && loop_replay);

    return true;
}

// ---- MAIN ----

int main(int argc, char** argv) {
    ros::init(argc, argv, "pal_policy_bag_replay_node");
    ros::NodeHandle nh("~");
    
    // Get model parameter
    std::string model;
    nh.param<std::string>("model", model, "v44");

    // Get task parameter
    std::string task;
    nh.param<std::string>("task", task, "loco");

    std::string bag_path;
    nh.param<std::string>("bag", bag_path, "");

    std::string trigger_topic;
    nh.param<std::string>("trigger_topic", trigger_topic, "/policy_input/base_lin_acc");

    double replay_rate_scale;
    nh.param<double>("replay_rate_scale", replay_rate_scale, 1.0);

    bool loop_replay;
    nh.param<bool>("loop", loop_replay, false);

    nh.param<bool>("replay_actions_from_bag", replay_actions_from_bag, true);
    bag_replay_mode = true;
    
    std::string pkg_root = ros::package::getPath("pal_policy_deployer");
    std::string config_dir = pkg_root + "/models/" + task + "/" + task + "_" + model + "_mjlab/";
    
    ROS_INFO("Looking for ONNX model in: %s", config_dir.c_str());
    
    // Find ONNX file
    std::string model_path;
    bool found = false;
    for (auto& entry : fs::directory_iterator(config_dir)) {
        fs::path p = entry.path();
        if (fs::is_regular_file(p) && p.extension() == ".onnx") {
            model_path = entry.path().string();
            found = true;
            ROS_INFO("Found ONNX model: %s", model_path.c_str());
            break;
        }
    }
    if (!found) {
        ROS_FATAL("No ONNX model found in %s", config_dir.c_str());
        return 1;
    }
    
    // Load policy configuration from ONNX metadata
    if (!loadPolicyConfig(model_path)) {
        ROS_FATAL("Failed to load policy configuration from ONNX metadata");
        return 1;
    }
    
    // Print configuration
    ROS_INFO("=== Policy Configuration ===");
    ROS_INFO("Run path: %s", config.run_path.c_str());
    ROS_INFO("Observation dimension: %d (single: %d x history: %d)",
             config.obs_dim, config.single_obs_dim, config.history_length);
    ROS_INFO("Action dimension: %d", config.action_dim);
    ROS_INFO("Number of metadata joints: %zu", config.joint_names.size());
    ROS_INFO("Number of observation joints: %zu", config.observation_joint_names.size());
    ROS_INFO("Number of action joints: %zu", config.action_joint_names.size());
    
    std::stringstream obs_ss;
    for (size_t i = 0; i < config.observation_names.size(); ++i) {
        if (i > 0) obs_ss << ", ";
        obs_ss << config.observation_names[i];
    }
    ROS_INFO("Observation terms: %s", obs_ss.str().c_str());
    
    // Initialize observation terms
    ROS_INFO("Initializing observation terms:");
    initializeObservationTerms();
    
    // Initialize action vectors
    action.resize(config.action_dim, 0.0);
    zero_action.resize(config.action_dim, 0.0);
    
    // Set input shape
    input_shape = {1, static_cast<int64_t>(config.obs_dim)};
    
    // Create ONNX session
    sess.reset(new Ort::Session(env, model_path.c_str(), Ort::SessionOptions{}));
    
    in_names.push_back(sess->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release());
    out_names.push_back(sess->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release());
    
    ROS_INFO("Bag replay mode: no live sensor subscribers are used");
    
    // Publisher
    pub_sub_cmd = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);

    // /policy_input publishers are created only for terms used by the loaded policy.
    if (hasObservationTerm("base_ang_vel")) {
        pub_dbg_ang_vel = nh.advertise<geometry_msgs::Vector3Stamped>("/policy_input/base_ang_vel", 1);
    }
    if (hasObservationTerm("base_lin_acc")) {
        pub_dbg_lin_acc = nh.advertise<geometry_msgs::Vector3Stamped>("/policy_input/base_lin_acc", 1);
    }
    if (hasObservationTerm("projected_gravity")) {
        pub_dbg_proj_grav = nh.advertise<geometry_msgs::Vector3Stamped>("/policy_input/projected_gravity", 1);
    } else if (hasObservationTerm("imu_projected_gravity")) {
        pub_dbg_proj_grav = nh.advertise<geometry_msgs::Vector3Stamped>("/policy_input/imu_projected_gravity", 1);
    }
    if (hasObservationTerm("joint_pos")) {
        pub_dbg_joint_pos = nh.advertise<sensor_msgs::JointState>("/policy_input/joint_pos_rel", 1);
        pub_dbg_joint_pos_absolute = nh.advertise<sensor_msgs::JointState>("/policy_input/joint_pos_abs", 1);
    }
    if (hasObservationTerm("joint_vel")) {
        pub_dbg_joint_vel = nh.advertise<sensor_msgs::JointState>("/policy_input/joint_vel", 1);
    }
    if (hasObservationTerm("command")) {
        pub_dbg_command = nh.advertise<geometry_msgs::Vector3Stamped>("/policy_input/command", 1);
    }
    if (hasObservationTerm("actions")) {
        pub_dbg_actions_input = nh.advertise<sensor_msgs::JointState>("/policy_input/actions", 1);
    }
    pub_dbg_actions              = nh.advertise<sensor_msgs::JointState>("/policy_output/actions_raw",                     1);
    pub_dbg_actions_scaled       = nh.advertise<sensor_msgs::JointState>("/policy_output/actions_scaled",                  1);

    pub_tau_computed = nh.advertise<sensor_msgs::JointState>("/policy_output/tau_computed", 1);

    // Load impedance gains from param server
    impedance_ks_global.resize(config.action_joint_names.size(), 0.0);
    impedance_kd_global.resize(config.action_joint_names.size(), 0.0);
    {
        ros::NodeHandle nh_ctrl("subscriber_controller");
        for (size_t i = 0; i < config.action_joint_names.size(); ++i) {
            const std::string& jname = config.action_joint_names[i];
            nh_ctrl.getParam("joints/" + jname + "/impedance_params/stiffness_factor", impedance_ks_global[i]);
            nh_ctrl.getParam("joints/" + jname + "/impedance_params/damping_factor",   impedance_kd_global[i]);
            ROS_INFO("Impedance gains [%s]: ks=%.1f kd=%.3f", jname.c_str(), impedance_ks_global[i], impedance_kd_global[i]);
        }
    }

    ROS_INFO("Policy bag replay node started successfully!");

    if (!replayBagInputs(bag_path, trigger_topic, replay_rate_scale, loop_replay)) {
        return 1;
    }
    
    return 0;
}
