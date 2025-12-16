#include <ros/ros.h>
#include <ros/package.h>
// ros msgs
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <onnxruntime_cxx_api.h>
#include <boost/filesystem.hpp>
// core libraries
#include <vector>
#include <algorithm>
#include <array>
#include <mutex>
#include <iostream>
#include <deque>
#include <sstream>
#include <map>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fs = boost::filesystem;

// ---- OBSERVATION TERM DIMENSIONS (fixed structure) ----
// Term order: base_ang_vel, joint_pos, joint_vel, actions, command, base_lin_acc
static constexpr int DIM_BASE_ANG_VEL = 3;
static constexpr int DIM_COMMAND = 3;
static constexpr int DIM_BASE_LIN_ACC = 3;

// ---- JOINTS TO EXCLUDE FROM ACTIONS ----
static const std::vector<std::string> excluded_action_joints = {
    "leg_left_femur_joint", "leg_left_knee_joint",
    "leg_right_femur_joint", "leg_right_knee_joint"
};

// ---- ONNX METADATA STRUCTURE ----
struct PolicyConfig {
    std::string run_path;
    std::vector<std::string> joint_names;        // obs order (26 joints)
    std::vector<std::string> action_joint_names; // action order (22 joints, excludes femur/knee)
    std::vector<double> default_joint_pos;       // obs offset (26 values)
    std::vector<double> action_offset;           // action offset (22 values)
    std::vector<double> action_scale;            // action scale (22 values)
    std::vector<double> joint_stiffness;         // for potential PD control (22 values)
    std::vector<double> joint_damping;           // for potential PD control (22 values)
    std::vector<std::string> observation_names;
    int obs_dim;
    int action_dim;
    int history_length;
    int single_obs_dim;
};

// Global config populated from ONNX metadata
PolicyConfig config;

// Policy specific vars
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rl");
std::unique_ptr<Ort::Session> sess;
Ort::MemoryInfo mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
std::vector<const char*> in_names, out_names;

// ---- CURRENT OBSERVATION TERMS (latest values from callbacks) ----
std::vector<double> current_base_ang_vel(DIM_BASE_ANG_VEL, 0.0);
std::vector<double> current_joint_pos;  // size set from config
std::vector<double> current_joint_vel;  // size set from config
std::vector<double> current_last_action; // size set from config
std::vector<double> current_vel_cmd(DIM_COMMAND, 0.0);
std::vector<double> current_base_lin_acc(DIM_BASE_LIN_ACC, 0.0);

// ---- HISTORY BUFFERS (deque of vectors, oldest first) ----
std::deque<std::vector<double>> history_base_ang_vel;
std::deque<std::vector<double>> history_joint_pos;
std::deque<std::vector<double>> history_joint_vel;
std::deque<std::vector<double>> history_last_action;
std::deque<std::vector<double>> history_vel_cmd;
std::deque<std::vector<double>> history_base_lin_acc;

std::mutex observation_mutex;

std::vector<int64_t> input_shape;
std::vector<double> action;
std::vector<double> zero_action;

ros::Publisher pub_sub_cmd;

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

// ---- ONNX METADATA PARSING ----

bool loadPolicyConfig(const std::string& model_path) {
    // Load ONNX model to read metadata
    Ort::SessionOptions session_options;
    Ort::Session temp_session(env, model_path.c_str(), session_options);
    
    // Get model metadata
    Ort::ModelMetadata metadata = temp_session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Get all custom metadata keys
    std::vector<Ort::AllocatedStringPtr> keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    
    std::map<std::string, std::string> metadata_map;
    for (auto& key_ptr : keys) {
        std::string key = key_ptr.get();
        Ort::AllocatedStringPtr value_ptr = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
        if (value_ptr) {
            metadata_map[key] = value_ptr.get();
        }
    }
    
    // Parse metadata
    if (metadata_map.find("run_path") != metadata_map.end()) {
        config.run_path = metadata_map["run_path"];
    }
    
    if (metadata_map.find("joint_names") != metadata_map.end()) {
        config.joint_names = splitString(metadata_map["joint_names"], ',');
    } else {
        ROS_FATAL("Missing 'joint_names' in ONNX metadata");
        return false;
    }
    
    if (metadata_map.find("default_joint_pos") != metadata_map.end()) {
        config.default_joint_pos = parseDoubles(metadata_map["default_joint_pos"]);
    } else {
        ROS_FATAL("Missing 'default_joint_pos' in ONNX metadata");
        return false;
    }
    
    if (metadata_map.find("action_scale") != metadata_map.end()) {
        config.action_scale = parseDoubles(metadata_map["action_scale"]);
    } else {
        ROS_FATAL("Missing 'action_scale' in ONNX metadata");
        return false;
    }
    
    if (metadata_map.find("joint_stiffness") != metadata_map.end()) {
        config.joint_stiffness = parseDoubles(metadata_map["joint_stiffness"]);
    }
    
    if (metadata_map.find("joint_damping") != metadata_map.end()) {
        config.joint_damping = parseDoubles(metadata_map["joint_damping"]);
    }
    
    if (metadata_map.find("observation_names") != metadata_map.end()) {
        config.observation_names = splitString(metadata_map["observation_names"], ',');
    }
    
    // Build action joint names (exclude femur/knee joints)
    config.action_joint_names.clear();
    config.action_offset.clear();
    for (size_t i = 0; i < config.joint_names.size(); ++i) {
        if (!isExcludedJoint(config.joint_names[i])) {
            config.action_joint_names.push_back(config.joint_names[i]);
            config.action_offset.push_back(config.default_joint_pos[i]);
        }
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
    
    // Calculate single observation dimension and history length
    // Single obs = base_ang_vel(3) + joint_pos(N) + joint_vel(N) + actions(M) + command(3) + base_lin_acc(3)
    int num_obs_joints = static_cast<int>(config.joint_names.size());
    int num_action_joints = static_cast<int>(config.action_joint_names.size());
    config.single_obs_dim = DIM_BASE_ANG_VEL + num_obs_joints + num_obs_joints + 
                            num_action_joints + DIM_COMMAND + DIM_BASE_LIN_ACC;
    
    if (config.obs_dim % config.single_obs_dim == 0) {
        config.history_length = config.obs_dim / config.single_obs_dim;
    } else {
        ROS_WARN("Observation dimension %d is not divisible by single obs dim %d, assuming history_length=1",
                 config.obs_dim, config.single_obs_dim);
        config.history_length = 1;
    }
    
    // Validate dimensions
    if (config.action_dim != num_action_joints) {
        ROS_WARN("Action dimension mismatch: ONNX output=%d, expected=%d", 
                 config.action_dim, num_action_joints);
    }
    
    if (config.action_scale.size() != static_cast<size_t>(config.action_dim)) {
        ROS_FATAL("Action scale size (%zu) doesn't match action dimension (%d)",
                  config.action_scale.size(), config.action_dim);
        return false;
    }
    
    return true;
}

// ---- HISTORY MANAGEMENT ----

void initializeHistoryBuffers() {
    int num_obs_joints = static_cast<int>(config.joint_names.size());
    int num_action_joints = static_cast<int>(config.action_joint_names.size());
    
    for (int i = 0; i < config.history_length; ++i) {
        history_base_ang_vel.push_back(std::vector<double>(DIM_BASE_ANG_VEL, 0.0));
        history_joint_pos.push_back(std::vector<double>(num_obs_joints, 0.0));
        history_joint_vel.push_back(std::vector<double>(num_obs_joints, 0.0));
        history_last_action.push_back(std::vector<double>(num_action_joints, 0.0));
        history_vel_cmd.push_back(std::vector<double>(DIM_COMMAND, 0.0));
        history_base_lin_acc.push_back(std::vector<double>(DIM_BASE_LIN_ACC, 0.0));
    }
}

void updateHistory() {
    // Remove oldest
    history_base_ang_vel.pop_front();
    history_joint_pos.pop_front();
    history_joint_vel.pop_front();
    history_last_action.pop_front();
    history_vel_cmd.pop_front();
    history_base_lin_acc.pop_front();
    
    // Add newest
    history_base_ang_vel.push_back(current_base_ang_vel);
    history_joint_pos.push_back(current_joint_pos);
    history_joint_vel.push_back(current_joint_vel);
    history_last_action.push_back(current_last_action);
    history_vel_cmd.push_back(current_vel_cmd);
    history_base_lin_acc.push_back(current_base_lin_acc);
}

// Build the flattened observation in TERM-MAJOR format
// [A_t0, A_t1, ..., A_tN, B_t0, B_t1, ..., B_tN, ...]
std::vector<float> buildObservationVector() {
    std::vector<float> obs;
    obs.reserve(config.obs_dim);
    
    // Term A: base_ang_vel history
    for (int t = 0; t < config.history_length; ++t) {
        for (size_t i = 0; i < history_base_ang_vel[t].size(); ++i) {
            obs.push_back(static_cast<float>(history_base_ang_vel[t][i]));
        }
    }
    
    // Term B: joint_pos history
    for (int t = 0; t < config.history_length; ++t) {
        for (size_t i = 0; i < history_joint_pos[t].size(); ++i) {
            obs.push_back(static_cast<float>(history_joint_pos[t][i]));
        }
    }
    
    // Term C: joint_vel history
    for (int t = 0; t < config.history_length; ++t) {
        for (size_t i = 0; i < history_joint_vel[t].size(); ++i) {
            obs.push_back(static_cast<float>(history_joint_vel[t][i]));
        }
    }
    
    // Term D: last_action history
    for (int t = 0; t < config.history_length; ++t) {
        for (size_t i = 0; i < history_last_action[t].size(); ++i) {
            obs.push_back(static_cast<float>(history_last_action[t][i]));
        }
    }
    
    // Term E: vel_cmd history
    for (int t = 0; t < config.history_length; ++t) {
        for (size_t i = 0; i < history_vel_cmd[t].size(); ++i) {
            obs.push_back(static_cast<float>(history_vel_cmd[t][i]));
        }
    }
    
    // Term F: base_lin_acc history
    for (int t = 0; t < config.history_length; ++t) {
        for (size_t i = 0; i < history_base_lin_acc[t].size(); ++i) {
            obs.push_back(static_cast<float>(history_base_lin_acc[t][i]));
        }
    }
    
    return obs;
}

// ---- TIMER CALLBACK ----

void timerCallback(const ros::TimerEvent&) {
    std::vector<float> obs_input;
    {
        std::lock_guard<std::mutex> lk(observation_mutex);
        
        // Update history with current observations
        updateHistory();
        
        // Build flattened observation vector in term-major format
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
    
    // Update current_last_action for next timestep (raw network output)
    {
        std::lock_guard<std::mutex> lk(observation_mutex);
        for (int i = 0; i < config.action_dim; ++i) {
            current_last_action[i] = y_f[i];
        }
    }

    sensor_msgs::JointState cmd_msg;
    cmd_msg.header.stamp = ros::Time::now();
    cmd_msg.name = config.action_joint_names;
    cmd_msg.position = action;
    cmd_msg.velocity = zero_action;
    cmd_msg.effort = zero_action;

    pub_sub_cmd.publish(cmd_msg);
}

// ---- OBSERVATION CALLBACKS ----

void jsCB(const sensor_msgs::JointState::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    for (size_t k = 0; k < config.joint_names.size(); ++k) {
        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (msg->name[i] == config.joint_names[k]) {
                current_joint_pos[k] = msg->position[i] - config.default_joint_pos[k];
                current_joint_vel[k] = msg->velocity[i];
                break;
            }
        }
    }
}

void imuCB(const sensor_msgs::Imu::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    // base_ang_vel
    current_base_ang_vel[0] = msg->angular_velocity.x;
    current_base_ang_vel[1] = msg->angular_velocity.y;
    current_base_ang_vel[2] = msg->angular_velocity.z;

    // base_lin_acc
    current_base_lin_acc[0] = msg->linear_acceleration.x;
    current_base_lin_acc[1] = msg->linear_acceleration.y;
    current_base_lin_acc[2] = msg->linear_acceleration.z;
}

void velCmdCB(const geometry_msgs::Twist::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    current_vel_cmd[0] = msg->linear.x;
    current_vel_cmd[1] = msg->linear.y;
    current_vel_cmd[2] = msg->angular.z;
}

// ---- MAIN ----

int main(int argc, char** argv) {
    ros::init(argc, argv, "pal_policy_deployer");
    ros::NodeHandle nh("~");
    
    // Get model parameter
    std::string model;
    nh.param<std::string>("model", model, "v44");
    
    std::string pkg_root = ros::package::getPath("pal_policy_deployer");
    std::string config_dir = pkg_root + "/models/loco/loco_" + model + "_mjlab/";
    
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
    
    // Print loaded configuration
    ROS_INFO("=== Policy Configuration ===");
    ROS_INFO("Run path: %s", config.run_path.c_str());
    ROS_INFO("Observation dimension: %d (single: %d x history: %d)", 
             config.obs_dim, config.single_obs_dim, config.history_length);
    ROS_INFO("Action dimension: %d", config.action_dim);
    ROS_INFO("Number of observation joints: %zu", config.joint_names.size());
    ROS_INFO("Number of action joints: %zu", config.action_joint_names.size());
    
    if (!config.observation_names.empty()) {
        std::stringstream obs_ss;
        for (size_t i = 0; i < config.observation_names.size(); ++i) {
            if (i > 0) obs_ss << ", ";
            obs_ss << config.observation_names[i];
        }
        ROS_INFO("Observation terms: %s", obs_ss.str().c_str());
    }
    
    // Initialize current observation vectors with correct sizes
    int num_obs_joints = static_cast<int>(config.joint_names.size());
    int num_action_joints = static_cast<int>(config.action_joint_names.size());
    
    current_joint_pos.resize(num_obs_joints, 0.0);
    current_joint_vel.resize(num_obs_joints, 0.0);
    current_last_action.resize(num_action_joints, 0.0);
    
    // Initialize action vectors
    action.resize(num_action_joints, 0.0);
    zero_action.resize(num_action_joints, 0.0);
    
    // Initialize history buffers
    initializeHistoryBuffers();
    
    // Set input shape
    input_shape = {1, static_cast<int64_t>(config.obs_dim)};
    
    // Create session
    sess.reset(new Ort::Session(env, model_path.c_str(), Ort::SessionOptions{}));
    
    in_names.push_back(sess->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release());
    out_names.push_back(sess->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release());
    
    // Subscribers
    ros::Subscriber s1 = nh.subscribe("/joint_states", 1, jsCB);
    ros::Subscriber s2 = nh.subscribe("/base_imu", 1, imuCB);
    ros::Subscriber s3 = nh.subscribe("/cmd_vel", 1, velCmdCB);
    
    // Publisher
    pub_sub_cmd = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);
    
    // Timer at 50Hz
    ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCallback);
    
    ROS_INFO("Policy deployer started successfully!");
    
    ros::spin();
    
    return 0;
}