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
#include <functional>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <time.h>

#include <pal_statistics/pal_statistics_macros.h>

static const std::string POLICY_DEPLOYER_TOPIC = "pal_policy_deployer_statistics";

namespace fs = boost::filesystem;

// ---- OBSERVATION TERM REGISTRY ----

struct ObservationTerm {
    std::string name;
    int dim;
    std::vector<double> current;
    std::deque<std::vector<double>> history;
    ros::Time last_msg_time;
    double elapsed_time;  // Since sensor update
    
    ObservationTerm(const std::string& n, int d) : name(n), dim(d), current(d, 0.0), last_msg_time(0), elapsed_time(0.0) {
        const std::string full_name = "elapsed_time." + name;
        REGISTER_VARIABLE(POLICY_DEPLOYER_TOPIC, full_name, &elapsed_time, nullptr);
    }
    
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

// ---- AUTOREGRESSIVE MOTION STATE ----
// Holds the current command state, initialized from robot, then updated from ONNX outputs each step.

struct MotionState {
    // Current command (updated autoregressively from ONNX outputs)
    std::vector<double> joint_pos;      // [num_joints]
    std::vector<double> joint_vel;      // [num_joints]

    int num_joints = 0;
    int timestep = 0;
    int max_timestep = 0;
    bool loaded = false;

    // Get command vector: [joint_pos, joint_vel]
    std::vector<double> getCommand() const {
        std::vector<double> cmd;
        cmd.reserve(num_joints * 2);
        cmd.insert(cmd.end(), joint_pos.begin(), joint_pos.end());
        cmd.insert(cmd.end(), joint_vel.begin(), joint_vel.end());
        return cmd;
    }

    int commandDim() const { return num_joints * 2; }

    // Update from ONNX outputs after inference
    void updateFromOutputs(const float* out_joint_pos, int nj,
                           const float* out_joint_vel, int nj2) {
        for (int i = 0; i < nj && i < num_joints; ++i) {
            joint_pos[i] = out_joint_pos[i];
            joint_vel[i] = out_joint_vel[i];
        }
    }
};

// ---- JOINTS TO EXCLUDE FROM ACTIONS ----
static const std::vector<std::string> excluded_action_joints = {
    "leg_left_femur_joint", "leg_left_knee_joint",
    "leg_right_femur_joint", "leg_right_knee_joint"
};

// ---- POLICY CONFIGURATION ----
struct PolicyConfig {
    std::string run_path;
    std::string task;  // "loco" or "imitation"
    std::vector<std::string> joint_names;
    std::vector<std::string> action_joint_names;
    std::vector<double> default_joint_pos;
    std::vector<double> action_offset;
    std::vector<double> action_scale;
    std::vector<double> joint_stiffness;
    std::vector<double> joint_damping;
    std::vector<std::string> observation_names;
    int obs_dim;
    int action_dim;
    int history_length;
    int single_obs_dim;
    int num_inputs;
    int num_outputs;
    // Output indices (found by name)
    int out_idx_actions = -1;
    int out_idx_joint_pos = -1;
    int out_idx_joint_vel = -1;
};

// ---- GLOBALS ----
PolicyConfig config;
MotionState motion;

std::map<std::string, std::shared_ptr<ObservationTerm>> obs_terms;
std::vector<std::string> obs_term_order;

// IMU-derived values
Eigen::Quaterniond imu_orientation(1.0, 0.0, 0.0, 0.0);
bool imu_received = false;

// Robot state tracking for imitation initialization
sensor_msgs::JointState latest_joint_state;
bool joint_state_received = false;

std::mutex observation_mutex;

// ONNX Runtime
Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "rl");
std::unique_ptr<Ort::Session> sess;
Ort::MemoryInfo ort_mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
std::vector<const char*> in_names, out_names;

std::vector<int64_t> input_shape;
std::vector<int64_t> time_step_shape;
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

// ---- INITIALIZE MOTION FROM ROBOT STATE ----

bool initializeMotionFromRobot() {
    if (!joint_state_received) {
        ROS_WARN("Waiting for joint state to initialize motion...");
        return false;
    }

    motion.num_joints = static_cast<int>(config.joint_names.size());
    motion.joint_pos.resize(motion.num_joints, 0.0);
    motion.joint_vel.resize(motion.num_joints, 0.0);

    // Map current robot joint states to motion state
    for (int j = 0; j < motion.num_joints; ++j) {
        const std::string& joint_name = config.joint_names[j];
        for (size_t i = 0; i < latest_joint_state.name.size(); ++i) {
            if (latest_joint_state.name[i] == joint_name) {
                motion.joint_pos[j] = latest_joint_state.position[i];
                if (i < latest_joint_state.velocity.size()) {
                    motion.joint_vel[j] = latest_joint_state.velocity[i];
                }
                break;
            }
        }
    }

    // Get max_timestep from param or use default
    ros::NodeHandle nh("~");
    nh.param<int>("max_timestep", motion.max_timestep, 1000);

    motion.timestep = 0;
    motion.loaded = true;

    ROS_INFO("Motion initialized from robot state: %d joints, max_timestep=%d",
             motion.num_joints, motion.max_timestep);
    
    return true;
}

// ---- OBSERVATION TERM DIMENSION CALCULATOR ----

int getObservationTermDim(const std::string& term_name) {
    if (term_name == "base_ang_vel") return 3;
    if (term_name == "base_lin_vel") return 3;
    if (term_name == "base_lin_acc") return 3;
    if (term_name == "imu_projected_gravity") return 3;
    if (term_name == "joint_pos") return static_cast<int>(config.joint_names.size());
    if (term_name == "joint_vel") return static_cast<int>(config.joint_names.size());
    if (term_name == "actions") return static_cast<int>(config.action_joint_names.size());

    if (term_name == "command") {
        if (config.task == "imitation") {
            return motion.commandDim();
        } else {
            return 3;
        }
    }
    
    return -1;
}

// ---- OBSERVATION TERM INITIALIZATION ----
void initializeObservationTerms() {
    obs_terms.clear();
    obs_term_order.clear();
    
    for (const auto& term_name : config.observation_names) {
        int dim = getObservationTermDim(term_name);
        if (dim < 0) {
            ROS_FATAL("Unknown observation term: '%s'. Add it to getObservationTermDim().", 
                      term_name.c_str());
            ros::shutdown();
            return;
        }
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
    Ort::Session temp_session(ort_env, model_path.c_str(), session_options);
    
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
    
    if (metadata_map.find("run_path") != metadata_map.end())
        config.run_path = metadata_map["run_path"];
    
    if (metadata_map.find("joint_names") == metadata_map.end()) { ROS_FATAL("Missing 'joint_names'"); return false; }
    config.joint_names = splitString(metadata_map["joint_names"], ',');
    
    if (metadata_map.find("default_joint_pos") == metadata_map.end()) { ROS_FATAL("Missing 'default_joint_pos'"); return false; }
    config.default_joint_pos = parseDoubles(metadata_map["default_joint_pos"]);
    
    if (metadata_map.find("action_scale") == metadata_map.end()) { ROS_FATAL("Missing 'action_scale'"); return false; }
    config.action_scale = parseDoubles(metadata_map["action_scale"]);
    
    if (metadata_map.find("observation_names") == metadata_map.end()) { ROS_FATAL("Missing 'observation_names'"); return false; }
    config.observation_names = splitString(metadata_map["observation_names"], ',');
    
    if (metadata_map.find("joint_stiffness") != metadata_map.end())
        config.joint_stiffness = parseDoubles(metadata_map["joint_stiffness"]);
    if (metadata_map.find("joint_damping") != metadata_map.end())
        config.joint_damping = parseDoubles(metadata_map["joint_damping"]);
    
    // Build action joint names (exclude femur/knee passive joints)
    config.action_joint_names.clear();
    config.action_offset.clear();
    for (size_t i = 0; i < config.joint_names.size(); ++i) {
        if (!isExcludedJoint(config.joint_names[i])) {
            config.action_joint_names.push_back(config.joint_names[i]);
            config.action_offset.push_back(config.default_joint_pos[i]);
        }
    }
    
    // Enumerate inputs and outputs
    config.num_inputs = static_cast<int>(temp_session.GetInputCount());
    config.num_outputs = static_cast<int>(temp_session.GetOutputCount());
    
    // Get obs dimension from first input
    {
        Ort::TypeInfo info = temp_session.GetInputTypeInfo(0);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        config.obs_dim = static_cast<int>(tensor_info.GetShape()[1]);
    }
    
    // Find output indices by name
    for (int i = 0; i < config.num_outputs; ++i) {
        Ort::AllocatedStringPtr name = temp_session.GetOutputNameAllocated(i, allocator);
        std::string n = name.get();
        
        Ort::TypeInfo info = temp_session.GetOutputTypeInfo(i);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        auto dims = tensor_info.GetShape();
        
        std::stringstream shape_ss;
        for (size_t j = 0; j < dims.size(); j++) {
            if (j > 0) shape_ss << ",";
            shape_ss << dims[j];
        }
        ROS_INFO("  Output %d: '%s' [%s]", i, n.c_str(), shape_ss.str().c_str());
        
        if (n == "actions") {
            config.out_idx_actions = i;
            config.action_dim = static_cast<int>(dims[1]);
        } else if (n == "joint_pos") {
            config.out_idx_joint_pos = i;
        } else if (n == "joint_vel") {
            config.out_idx_joint_vel = i;
        }
    }
    
    for (int i = 0; i < config.num_inputs; ++i) {
        Ort::AllocatedStringPtr name = temp_session.GetInputNameAllocated(i, allocator);
        Ort::TypeInfo info = temp_session.GetInputTypeInfo(i);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        auto dims = tensor_info.GetShape();
        std::stringstream shape_ss;
        for (size_t j = 0; j < dims.size(); j++) {
            if (j > 0) shape_ss << ",";
            shape_ss << dims[j];
        }
        ROS_INFO("  Input %d: '%s' [%s]", i, name.get(), shape_ss.str().c_str());
    }
    
    // For imitation, defer single_obs_dim calculation until after motion is initialized
    // (because command dim depends on motion data)
    if (config.task != "imitation") {
        config.single_obs_dim = 0;
        for (const auto& term_name : config.observation_names) {
            config.single_obs_dim += getObservationTermDim(term_name);
        }
        
        if (config.obs_dim % config.single_obs_dim == 0) {
            config.history_length = config.obs_dim / config.single_obs_dim;
        } else {
            ROS_FATAL("Observation dimension %d not divisible by single obs dim %d",
                      config.obs_dim, config.single_obs_dim);
            return false;
        }
    }
    
    if (config.action_dim != static_cast<int>(config.action_joint_names.size())) {
        ROS_WARN("Action dim mismatch: ONNX=%d, expected=%zu",
                 config.action_dim, config.action_joint_names.size());
    }
    
    return true;
}

// ---- BUILD OBSERVATION VECTOR ----

std::vector<float> buildObservationVector() {
    std::vector<float> obs;
    obs.reserve(config.obs_dim);
    const auto obs_time = ros::Time::now();
    for (const auto& term_name : obs_term_order) {
        auto& term = obs_terms[term_name];
        const auto elapsed_duration = obs_time - term->last_msg_time;
        term->elapsed_time = elapsed_duration.toSec();

        for (int t = 0; t < config.history_length; ++t) {
            for (int i = 0; i < term->dim; ++i) {
                obs.push_back(static_cast<float>(term->history[t][i]));
            }
        }
    }
    return obs;
}

void updateAllHistory() {
    for (auto& pair : obs_terms) {
        pair.second->updateHistory();
    }
}

// ---- TIMER CALLBACK ----

void timerCallback(const ros::TimerEvent&) {
    // For imitation task, ensure motion is initialized from robot state
    if (config.task == "imitation" && !motion.loaded) {
        if (!initializeMotionFromRobot()) {
            return;  // Wait for joint state
        }
    }

    std::vector<float> obs_input;
    float time_step_val = 0.0f;
    {
        std::lock_guard<std::mutex> lk(observation_mutex);

        if (config.task == "imitation" && motion.loaded) {
            // Update command obs from current motion state
            if (obs_terms.find("command") != obs_terms.end()) {
                auto& term = obs_terms["command"];
                std::vector<double> cmd = motion.getCommand();
                for (int i = 0; i < term->dim && i < static_cast<int>(cmd.size()); ++i) {
                    term->current[i] = cmd[i];
                }
            }

            // Timestep for the model
            time_step_val = static_cast<float>(motion.timestep);
        }
        
        updateAllHistory();
        obs_input = buildObservationVector();
    }
    PUBLISH_STATISTICS(POLICY_DEPLOYER_TOPIC);

    // Build ONNX inputs
    std::vector<Ort::Value> input_tensors;
    
    Ort::Value obs_tensor =
        Ort::Value::CreateTensor<float>(ort_mem, obs_input.data(), obs_input.size(),
                                        input_shape.data(), input_shape.size());
    input_tensors.push_back(std::move(obs_tensor));

    // Add time_step input for imitation
    if (config.num_inputs > 1 && config.task == "imitation") {
        Ort::Value ts_tensor =
            Ort::Value::CreateTensor<float>(ort_mem, &time_step_val, 1,
                                            time_step_shape.data(), time_step_shape.size());
        input_tensors.push_back(std::move(ts_tensor));
    }

    // Run inference (request all outputs)
    auto outputs = sess->Run(Ort::RunOptions{nullptr},
                             in_names.data(), input_tensors.data(),
                             input_tensors.size(),
                             out_names.data(), out_names.size());

    // Extract actions
    const float* actions_data = outputs[config.out_idx_actions].GetTensorData<float>();

    for (int i = 0; i < config.action_dim; ++i) {
        action[i] = config.action_scale[i] * actions_data[i] + config.action_offset[i];
    }
    
    {
        std::lock_guard<std::mutex> lk(observation_mutex);

        // Update 'actions' obs term (raw network output, before scaling)
        if (obs_terms.find("actions") != obs_terms.end()) {
            auto& actions_term = obs_terms["actions"];
            actions_term->last_msg_time = ros::Time::now();
            for (int i = 0; i < config.action_dim; ++i) {
                actions_term->current[i] = actions_data[i];
            }
        }

        // For imitation: update motion state autoregressively from ONNX outputs
        if (config.task == "imitation" && motion.loaded) {
            if (config.out_idx_joint_pos >= 0 && config.out_idx_joint_vel >= 0) {
                const float* out_jp = outputs[config.out_idx_joint_pos].GetTensorData<float>();
                const float* out_jv = outputs[config.out_idx_joint_vel].GetTensorData<float>();

                motion.updateFromOutputs(out_jp, motion.num_joints,
                                        out_jv, motion.num_joints);
            }

            // Advance timestep
            motion.timestep++;
            if (motion.max_timestep > 0 && motion.timestep >= motion.max_timestep) {
                ROS_INFO_THROTTLE(5.0, "Motion reached max_timestep=%d, holding last frame", motion.max_timestep);
                motion.timestep = motion.max_timestep - 1;
            }
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

// ---- SENSOR CALLBACKS ----

void jsCB(const sensor_msgs::JointState::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    // Store latest for imitation initialization
    if (!joint_state_received) {
        latest_joint_state = *msg;
        joint_state_received = true;
    }
    
    if (obs_terms.find("joint_pos") != obs_terms.end()) {
        auto& term = obs_terms["joint_pos"];
        term->last_msg_time = msg->header.stamp;
        for (size_t k = 0; k < config.joint_names.size(); ++k) {
            for (size_t i = 0; i < msg->name.size(); ++i) {
                if (msg->name[i] == config.joint_names[k]) {
                    term->current[k] = msg->position[i] - config.default_joint_pos[k];
                    break;
                }
            }
        }
    }
    
    if (obs_terms.find("joint_vel") != obs_terms.end()) {
        auto& term = obs_terms["joint_vel"];
        term->last_msg_time = msg->header.stamp;
        for (size_t k = 0; k < config.joint_names.size(); ++k) {
            for (size_t i = 0; i < msg->name.size(); ++i) {
                if (msg->name[i] == config.joint_names[k]) {
                    term->current[k] = msg->velocity[i];
                    break;
                }
            }
        }
    }
}

void imuCB(const sensor_msgs::Imu::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    imu_orientation = Eigen::Quaterniond(
        msg->orientation.w,
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z
    );
    imu_received = true;
    
    if (obs_terms.find("base_ang_vel") != obs_terms.end()) {
        auto& term = obs_terms["base_ang_vel"];
        term->last_msg_time = msg->header.stamp;
        term->current[0] = msg->angular_velocity.x;
        term->current[1] = msg->angular_velocity.y;
        term->current[2] = msg->angular_velocity.z;
    }
    
    if (obs_terms.find("base_lin_acc") != obs_terms.end()) {
        auto& term = obs_terms["base_lin_acc"];
        term->last_msg_time = msg->header.stamp;
        term->current[0] = msg->linear_acceleration.x;
        term->current[1] = msg->linear_acceleration.y;
        term->current[2] = msg->linear_acceleration.z;
    }

    if (obs_terms.find("imu_projected_gravity") != obs_terms.end()) {
        auto& term = obs_terms["imu_projected_gravity"];
	term->last_msg_time = msg->header.stamp;
        Eigen::Vector3d gravity_world(0.0, 0.0, -1.0);
        Eigen::Vector3d gravity_body = imu_orientation.conjugate() * gravity_world;
        term->current[0] = gravity_body.x();
        term->current[1] = gravity_body.y();
        term->current[2] = gravity_body.z();
    }
    
    if (obs_terms.find("base_lin_vel") != obs_terms.end()) {
        ROS_WARN_ONCE("base_lin_vel term present but not populated from IMU - need odometry source");
    }
}

void velCmdCB(const geometry_msgs::Twist::ConstPtr& msg) {
    if (config.task == "imitation") return;

    std::lock_guard<std::mutex> lk(observation_mutex);
    
    if (obs_terms.find("command") != obs_terms.end()) {
        auto& term = obs_terms["command"];
        term->last_msg_time = ros::Time::now(); // Not really important to know the exact time it was generated
        term->current[0] = msg->linear.x;
        term->current[1] = msg->linear.y;
        term->current[2] = msg->angular.z;
    }
}

void odomCB(const nav_msgs::Odometry::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(observation_mutex);
    
    if (obs_terms.find("base_lin_vel") != obs_terms.end()) {
        auto& term = obs_terms["base_lin_vel"];
        term->last_msg_time = msg->header.stamp;
        term->current[0] = msg->twist.twist.linear.x;
        term->current[1] = msg->twist.twist.linear.y;
        term->current[2] = msg->twist.twist.linear.z;
    }
}

// ---- MAIN ----

int main(int argc, char** argv) {
    ros::init(argc, argv, "pal_policy_deployer");
    ros::NodeHandle nh("~");
    
    std::string model;
    nh.param<std::string>("model", model, "v44");

    std::string task;
    nh.param<std::string>("task", task, "loco");
    config.task = task;
    
    std::string pkg_root = ros::package::getPath("pal_policy_deployer");
    std::string config_dir = pkg_root + "/models/" + task + "/" + task + "_" + model + "_mjlab/";
    
    ROS_INFO("Looking for ONNX model in: %s", config_dir.c_str());
    ROS_INFO("Task: %s", task.c_str());
    
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

    // Load ONNX metadata
    if (!loadPolicyConfig(model_path)) {
        ROS_FATAL("Failed to load policy configuration from ONNX metadata");
        return 1;
    }

    // For imitation task, motion will be initialized from robot state on first timer callback
    if (task == "imitation") {
        // Set number of joints for motion (will be populated from robot state)
        motion.num_joints = static_cast<int>(config.joint_names.size());
        
        // Compute single_obs_dim now that we know joint count
        config.single_obs_dim = 0;
        for (const auto& term_name : config.observation_names) {
            config.single_obs_dim += getObservationTermDim(term_name);
        }
        if (config.obs_dim % config.single_obs_dim == 0) {
            config.history_length = config.obs_dim / config.single_obs_dim;
        } else {
            ROS_FATAL("Observation dimension %d not divisible by single obs dim %d",
                      config.obs_dim, config.single_obs_dim);
            return 1;
        }
    }
    
    // Print configuration
    ROS_INFO("=== Policy Configuration ===");
    ROS_INFO("Run path: %s", config.run_path.c_str());
    ROS_INFO("Task: %s", config.task.c_str());
    ROS_INFO("Observation dimension: %d (single: %d x history: %d)",
             config.obs_dim, config.single_obs_dim, config.history_length);
    ROS_INFO("Action dimension: %d (output index: %d)", config.action_dim, config.out_idx_actions);
    ROS_INFO("Joints: %zu obs, %zu action", config.joint_names.size(), config.action_joint_names.size());
    if (task == "imitation") {
        ROS_INFO("Motion: autoregressive from robot state, command_dim=%d",
                 motion.commandDim());
    }
    
    std::stringstream obs_ss;
    for (size_t i = 0; i < config.observation_names.size(); ++i) {
        if (i > 0) obs_ss << ", ";
        obs_ss << config.observation_names[i];
    }
    ROS_INFO("Observation terms: %s", obs_ss.str().c_str());
    
    ROS_INFO("Initializing observation terms:");
    initializeObservationTerms();
    
    action.resize(config.action_dim, 0.0);
    zero_action.resize(config.action_dim, 0.0);
    
    input_shape = {1, static_cast<int64_t>(config.obs_dim)};
    time_step_shape = {1, 1};
    
    // Create ONNX session
    sess.reset(new Ort::Session(ort_env, model_path.c_str(), Ort::SessionOptions{}));
    
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < sess->GetInputCount(); ++i)
        in_names.push_back(sess->GetInputNameAllocated(i, alloc).release());
    for (size_t i = 0; i < sess->GetOutputCount(); ++i)
        out_names.push_back(sess->GetOutputNameAllocated(i, alloc).release());
    
    // Subscribers
    ros::Subscriber s1 = nh.subscribe("/joint_states", 1, jsCB);
    ros::Subscriber s2 = nh.subscribe("/base_imu", 1, imuCB);
    ros::Subscriber s3 = nh.subscribe("/cmd_vel", 1, velCmdCB);
    
    ros::Subscriber s4;
    if (obs_terms.find("base_lin_vel") != obs_terms.end()) {
        s4 = nh.subscribe("/odom", 1, odomCB);
        ROS_INFO("Subscribed to /odom for base_lin_vel");
    }
    
    pub_sub_cmd = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);
    
    ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCallback);
    
    ROS_INFO("Policy deployer started successfully!");
    if (task == "imitation") {
        ROS_INFO("Waiting for joint state to initialize motion...");
    }
    
    ros::spin();
    
    return 0;
}
