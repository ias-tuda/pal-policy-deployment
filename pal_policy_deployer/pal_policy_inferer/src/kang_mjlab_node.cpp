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
#include <pal_statistics_msgs/StatisticsNames.h>
#include <pal_statistics_msgs/StatisticsValues.h>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>

// core libraries
#include <vector>
#include <algorithm>
#include <array>
#include <mutex>
#include <iostream>
#include <deque>
#include <sstream>
#include <map>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <time.h>

#include <pal_statistics/pal_statistics_macros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>


typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::JointState,
    sensor_msgs::Imu> SyncPolicy;

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

// The grasp observation contract is supplied by the selected policy profile.
std::vector<std::string> grasp_joint_names;
std::vector<std::string> grasp_arm_joint_names;
std::vector<std::string> grasp_observation_terms;
std::vector<double> grasp_default_joint_pos;
KDL::Chain grasp_chain;
std::unique_ptr<KDL::ChainFkSolverPos_recursive> grasp_fk;
std::vector<std::string> grasp_chain_joint_names;
KDL::Vector grasp_tcp_offset;
KDL::JntArray grasp_q;
std::vector<double> grasp_position;
std::vector<double> grasp_velocity;
std::vector<float> grasp_last_action;
std::vector<double> grasp_mock_object_b;
std::vector<double> grasp_goal_b;
double grasp_mock_yaw_b = 0.0;
double grasp_rate = 50.0;
double grasp_state_timeout = 0.1;
bool grasp_state_received = false;
ros::Time grasp_state_time;
ros::Publisher grasp_action_pub;
ros::Subscriber grasp_joint_state_sub;
std::mutex grasp_mutex;
std::string grasp_action_topic = "/pal_policy/raw_actions";

int graspTermDim(const std::string& term)
{
    if (term == "joint_pos_rel" || term == "joint_vel") return grasp_joint_names.size();
    if (term == "last_action") return grasp_last_action.size();
    if (term == "tcp_pos_b" || term == "object_pos_b" || term == "goal_pos_b") return 3;
    if (term == "object_yaw_b") return 2;
    if (term == "tcp_quat_b") return 4;
    return -1;
}

ros::Publisher pub_sub_cmd;

uint32_t names_version;
std::vector<std::string> pal_statistics_names;

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

void runInference()
{
    // For imitation task, ensure motion is initialized from robot state
    if (config.task == "imitation" && !motion.loaded) {
        if (!initializeMotionFromRobot()) {
            return;  // Wait for joint state
        }
    }

    std::vector<float> obs_input;
    float time_step_val = 0.0f;
    {
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

void timerCallback(const ros::TimerEvent&)
{
    //std::lock_guard<std::mutex> lk(observation_mutex);
    runInference();
}
// ---- SENSOR CALLBACKS ----

void jsCB(const sensor_msgs::JointState::ConstPtr& msg) 
{  
    //std::lock_guard<std::mutex> lk(observation_mutex);

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

void imuCB(const sensor_msgs::Imu::ConstPtr& msg) 
{    
    //std::lock_guard<std::mutex> lk(observation_mutex);

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

void velCmdCB(const geometry_msgs::Twist::ConstPtr& msg) 
{
    //std::lock_guard<std::mutex> lk(observation_mutex);

    if (config.task == "imitation") return;
    
    if (obs_terms.find("command") != obs_terms.end()) {
        auto& term = obs_terms["command"];
        term->last_msg_time = ros::Time::now(); // Not really important to know the exact time it was generated
        term->current[0] = msg->linear.x;
        term->current[1] = msg->linear.y;
        term->current[2] = msg->angular.z;
    }
}

void odomCB(const nav_msgs::Odometry::ConstPtr& msg) 
{
    //std::lock_guard<std::mutex> lk(observation_mutex);

    if (obs_terms.find("base_lin_vel") != obs_terms.end()) {
        auto& term = obs_terms["base_lin_vel"];
        term->last_msg_time = msg->header.stamp;
        term->current[0] = msg->twist.twist.linear.x;
        term->current[1] = msg->twist.twist.linear.y;
        term->current[2] = msg->twist.twist.linear.z;
    }
}

void syncedCB(const sensor_msgs::JointState::ConstPtr& js,
              const sensor_msgs::Imu::ConstPtr& imu) 
{    
    //std::lock_guard<std::mutex> lk(observation_mutex);

    jsCB(js);
    imuCB(imu);
    runInference();
}

void palStatisticsNamesCb(const pal_statistics_msgs::StatisticsNames::ConstPtr& msg)
{
    names_version = msg->names_version;
    pal_statistics_names = msg->names;
}

int indexVector(const std::vector<std::string> &container, const std::string &key)
{
    const auto it = std::find(container.cbegin(), container.cend(), key);
    if(it == container.cend())
    {
        throw std::runtime_error("Unable to find the key : " + key);
        return -1;
    }
    return std::distance(container.cbegin(), it);
}

void palStatisticsValuesCb(const pal_statistics_msgs::StatisticsValues::ConstPtr& msg) 
{
    if(pal_statistics_names.empty())
    {
        ROS_WARN("Skipping the loop as the names is not received yet!");
        return;
    }
    if (names_version != msg->names_version)
    {
        ROS_WARN("Skipping the reading as the pal statistics names and values version is different");
        return;
    }

    const auto quat_w_index = indexVector(pal_statistics_names, "base_imu_orientation_W");
    const auto quat_z_index = indexVector(pal_statistics_names, "base_imu_orientation_Z");
    const auto quat_y_index = indexVector(pal_statistics_names, "base_imu_orientation_Y");
    const auto quat_x_index = indexVector(pal_statistics_names, "base_imu_orientation_X");

    const auto ang_vel_x_ind = indexVector(pal_statistics_names, "base_imu_angular_vel_X");
    const auto ang_vel_y_ind = indexVector(pal_statistics_names, "base_imu_angular_vel_Y");
    const auto ang_vel_z_ind = indexVector(pal_statistics_names, "base_imu_angular_vel_Z");

    const auto lin_accel_x_ind = indexVector(pal_statistics_names, "base_imu_linear_acc_X");
    const auto lin_accel_y_ind = indexVector(pal_statistics_names, "base_imu_linear_acc_Y");
    const auto lin_accel_z_ind = indexVector(pal_statistics_names, "base_imu_linear_acc_Z");


    imu_orientation = Eigen::Quaterniond(
        msg->values[quat_w_index],
        msg->values[quat_x_index],
        msg->values[quat_y_index],
        msg->values[quat_z_index]
    );
    imu_received = true;
    
    if (obs_terms.find("base_ang_vel") != obs_terms.end()) {
        auto& term = obs_terms["base_ang_vel"];
        term->last_msg_time = msg->header.stamp;
        term->current[0] = msg->values[ang_vel_x_ind];
        term->current[1] = msg->values[ang_vel_y_ind];
        term->current[2] = msg->values[ang_vel_z_ind];
    }
    
    if (obs_terms.find("base_lin_acc") != obs_terms.end()) {
        auto& term = obs_terms["base_lin_acc"];
        term->last_msg_time = msg->header.stamp;
        term->current[0] = msg->values[lin_accel_x_ind];
        term->current[1] = msg->values[lin_accel_y_ind];
        term->current[2] = msg->values[lin_accel_z_ind];
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


    // Store latest data of the joint info
    if (obs_terms.find("joint_pos") != obs_terms.end() && obs_terms.find("joint_vel") != obs_terms.end()) {
        auto& pos_term = obs_terms["joint_pos"];
        pos_term->last_msg_time = msg->header.stamp;
        auto& vel_term = obs_terms["joint_vel"];
        vel_term->last_msg_time = msg->header.stamp;
        for (size_t k = 0; k < config.joint_names.size(); ++k) {
            const std::string pos_str = config.joint_names[k] + "_position";
            const std::string vel_str = config.joint_names[k] + "_velocity";
            const auto jnt_pos_index = indexVector(pal_statistics_names, pos_str);
            const auto jnt_vel_index = indexVector(pal_statistics_names, vel_str);
            pos_term->current[k] = msg->values[jnt_pos_index] - config.default_joint_pos[k];
            vel_term->current[k] = msg->values[jnt_vel_index];
        }
    }
    runInference();
}

void graspJointStateCb(const sensor_msgs::JointState::ConstPtr& msg)
{
    std::unordered_map<std::string, size_t> indices;
    for (size_t i = 0; i < msg->name.size(); ++i) indices[msg->name[i]] = i;

    std::lock_guard<std::mutex> lock(grasp_mutex);
    KDL::JntArray q(grasp_chain_joint_names.size());
    std::vector<double> positions(grasp_joint_names.size());
    std::vector<double> velocities(grasp_joint_names.size());
    for (size_t j = 0; j < grasp_chain_joint_names.size(); ++j) {
        const auto it = indices.find(grasp_chain_joint_names[j]);
        if (it == indices.end() || it->second >= msg->position.size()) {
            ROS_WARN_THROTTLE(2.0, "Grasp policy waiting for chain joint %s",
                              grasp_chain_joint_names[j].c_str());
            return;
        }
        q(j) = msg->position[it->second];
        if (!std::isfinite(q(j))) return;
    }
    for (size_t j = 0; j < grasp_joint_names.size(); ++j) {
        const auto it = indices.find(grasp_joint_names[j]);
        if (it == indices.end() || it->second >= msg->position.size()) {
            ROS_WARN_THROTTLE(2.0, "Grasp policy waiting for arm and gripper joints");
            return;
        }
        positions[j] = msg->position[it->second];
        velocities[j] = it->second < msg->velocity.size() ? msg->velocity[it->second] : 0.0;
        if (!std::isfinite(positions[j]) || !std::isfinite(velocities[j])) return;
    }
    grasp_q = q;
    grasp_position.swap(positions);
    grasp_velocity.swap(velocities);
    grasp_state_time = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    grasp_state_received = true;
}

bool buildGraspObservation(std::vector<float>& obs)
{
    KDL::Frame tip;
    if (grasp_fk->JntToCart(grasp_q, tip) < 0) return false;
    const KDL::Frame tcp = tip * KDL::Frame(grasp_tcp_offset);
    double x, y, z, w;
    tcp.M.GetQuaternion(x, y, z, w);
    obs.clear();
    obs.reserve(config.obs_dim);
    for (const auto& term : grasp_observation_terms) {
        if (term == "joint_pos_rel") {
            for (size_t j = 0; j < grasp_position.size(); ++j)
                obs.push_back(static_cast<float>(grasp_position[j] - grasp_default_joint_pos[j]));
        } else if (term == "joint_vel") {
            for (double value : grasp_velocity) obs.push_back(static_cast<float>(value));
        } else if (term == "last_action") {
            obs.insert(obs.end(), grasp_last_action.begin(), grasp_last_action.end());
        } else if (term == "tcp_pos_b") {
            obs.insert(obs.end(), {static_cast<float>(tcp.p.x()), static_cast<float>(tcp.p.y()),
                                   static_cast<float>(tcp.p.z())});
        } else if (term == "object_pos_b") {
            for (double value : grasp_mock_object_b) obs.push_back(static_cast<float>(value));
        } else if (term == "object_yaw_b") {
            obs.insert(obs.end(), {static_cast<float>(std::cos(grasp_mock_yaw_b)),
                                   static_cast<float>(std::sin(grasp_mock_yaw_b))});
        } else if (term == "goal_pos_b") {
            for (double value : grasp_goal_b) obs.push_back(static_cast<float>(value));
        } else if (term == "tcp_quat_b") {
            obs.insert(obs.end(), {static_cast<float>(w), static_cast<float>(x),
                                   static_cast<float>(y), static_cast<float>(z)});
        }
    }
    return obs.size() == static_cast<size_t>(config.obs_dim) &&
           std::all_of(obs.begin(), obs.end(), [](float v) { return std::isfinite(v); });
}

void graspInferenceTimer(const ros::TimerEvent&)
{
    std::lock_guard<std::mutex> lock(grasp_mutex);
    if (!grasp_state_received) return;
    const double state_age = (ros::Time::now() - grasp_state_time).toSec();
    if (state_age < 0.0 || state_age > grasp_state_timeout) {
        ROS_WARN_THROTTLE(1.0, "Grasp policy stopped: joint state is stale");
        return;
    }

    std::vector<float> obs;
    if (!buildGraspObservation(obs)) return;
    const int64_t shape[] = {1, config.obs_dim};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        ort_mem, obs.data(), obs.size(), shape, 2);
    auto outputs = sess->Run(Ort::RunOptions{nullptr}, in_names.data(), &input_tensor, 1,
                             out_names.data(), 1);
    if (!outputs[0].IsTensor() ||
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(config.action_dim)) {
        ROS_ERROR_THROTTLE(1.0, "Grasp policy returned an unexpected action shape");
        return;
    }
    const float* output = outputs[0].GetTensorData<float>();
    if (!std::all_of(output, output + config.action_dim,
                     [](float v) { return std::isfinite(v); })) {
        ROS_ERROR_THROTTLE(1.0, "Grasp policy returned non-finite actions");
        return;
    }
    std_msgs::Float64MultiArray action_msg;
    action_msg.data.assign(output, output + config.action_dim);
    grasp_action_pub.publish(action_msg);
    for (size_t j = 0; j < grasp_last_action.size(); ++j) grasp_last_action[j] = output[j];
}

bool loadGraspPolicyConfig(ros::NodeHandle& nh)
{
    if (!nh.getParam("joint_names", grasp_joint_names) || grasp_joint_names.empty() ||
        !nh.getParam("arm_joint_names", grasp_arm_joint_names) || grasp_arm_joint_names.empty() ||
        !nh.getParam("default_joint_pos", grasp_default_joint_pos) ||
        grasp_default_joint_pos.size() != grasp_joint_names.size() ||
        !nh.getParam("observation_terms", grasp_observation_terms) || grasp_observation_terms.empty() ||
        !nh.getParam("action_dim", config.action_dim) || config.action_dim <= 0) {
        ROS_ERROR("Grasp profile requires joint_names, arm_joint_names, matching default_joint_pos, observation_terms and positive action_dim");
        return false;
    }
    for (const auto& arm_joint : grasp_arm_joint_names) {
        if (std::find(grasp_joint_names.begin(), grasp_joint_names.end(), arm_joint) ==
            grasp_joint_names.end()) {
            ROS_ERROR("Arm joint %s is missing from joint_names", arm_joint.c_str());
            return false;
        }
    }
    if (!std::all_of(grasp_default_joint_pos.begin(), grasp_default_joint_pos.end(),
                     [](double v) { return std::isfinite(v); })) {
        ROS_ERROR("default_joint_pos must contain finite values");
        return false;
    }
    grasp_last_action.assign(config.action_dim, 0.0f);
    grasp_position.assign(grasp_joint_names.size(), 0.0);
    grasp_velocity.assign(grasp_joint_names.size(), 0.0);
    config.obs_dim = 0;
    for (const auto& term : grasp_observation_terms) {
        const int dim = graspTermDim(term);
        if (dim < 0) { ROS_ERROR("Unknown grasp observation term: %s", term.c_str()); return false; }
        config.obs_dim += dim;
    }
    if (!nh.getParam("mock_object_position_b", grasp_mock_object_b) ||
        !nh.getParam("goal_position_b", grasp_goal_b) ||
        grasp_mock_object_b.size() != 3 || grasp_goal_b.size() != 3) {
        ROS_ERROR("Grasp profile requires 3-value object and goal positions");
        return false;
    }
    nh.param("mock_object_yaw_b", grasp_mock_yaw_b, 0.0);
    nh.param("policy_rate", grasp_rate, 50.0);
    nh.param("state_timeout", grasp_state_timeout, 0.1);
    if (!std::isfinite(grasp_rate) || grasp_rate <= 0.0 ||
        !std::isfinite(grasp_state_timeout) || grasp_state_timeout <= 0.0 ||
        !std::isfinite(grasp_mock_yaw_b)) {
        ROS_ERROR("Grasp policy rate, timeout and object yaw must be finite; rate and timeout must be positive");
        return false;
    }
    return true;
}

bool initializeGraspPolicy(ros::NodeHandle& nh)
{
    std::string description_param, base_link, tip_link, joint_state_topic;
    std::vector<double> tcp_offset;
    nh.param<std::string>("robot_description", description_param, "robot_description");
    nh.param<std::string>("base_link", base_link, "base_link");
    nh.param<std::string>("tip_link", tip_link, "arm_left_7_link");
    nh.param<std::string>("joint_state_topic", joint_state_topic, "/joint_states");
    nh.param<std::string>("action_topic", grasp_action_topic, "/pal_policy/raw_actions");
    if (description_param.empty()) { ROS_ERROR("robot_description must name a parameter"); return false; }
    if (!nh.getParam("tcp_offset", tcp_offset)) tcp_offset = {0.0, 0.0, 0.148};
    if (tcp_offset.size() != 3) { ROS_ERROR("tcp_offset must have 3 values"); return false; }
    if (!std::all_of(tcp_offset.begin(), tcp_offset.end(), [](double v) { return std::isfinite(v); }) ||
        !std::all_of(grasp_mock_object_b.begin(), grasp_mock_object_b.end(),
                     [](double v) { return std::isfinite(v); }) ||
        !std::all_of(grasp_goal_b.begin(), grasp_goal_b.end(),
                     [](double v) { return std::isfinite(v); })) {
        ROS_ERROR("TCP offset, object position and goal position must be finite");
        return false;
    }
    grasp_tcp_offset = KDL::Vector(tcp_offset[0], tcp_offset[1], tcp_offset[2]);

    std::string xml;
    const std::string description_key = description_param[0] == '/' ? description_param : "/" + description_param;
    if (!ros::param::get(description_key, xml)) {
        ROS_ERROR("Could not load robot description %s", description_param.c_str()); return false;
    }
    urdf::Model urdf_model;
    KDL::Tree tree;
    if (!urdf_model.initString(xml) || !kdl_parser::treeFromUrdfModel(urdf_model, tree) ||
        !tree.getChain(base_link, tip_link, grasp_chain)) {
        ROS_ERROR("Could not build KDL chain %s -> %s", base_link.c_str(), tip_link.c_str());
        return false;
    }
    for (const auto& segment : grasp_chain.segments) {
        if (segment.getJoint().getType() != KDL::Joint::None)
            grasp_chain_joint_names.push_back(segment.getJoint().getName());
    }
    if (grasp_chain_joint_names.size() != grasp_chain.getNrOfJoints()) {
        ROS_ERROR("Could not enumerate the KDL arm-chain joints"); return false;
    }
    for (const auto& arm_joint_name : grasp_arm_joint_names) {
        if (std::find(grasp_chain_joint_names.begin(), grasp_chain_joint_names.end(),
                      arm_joint_name) == grasp_chain_joint_names.end()) {
            ROS_ERROR("KDL chain is missing %s", arm_joint_name.c_str()); return false;
        }
    }
    grasp_fk.reset(new KDL::ChainFkSolverPos_recursive(grasp_chain));

    Ort::TypeInfo input_info = sess->GetInputTypeInfo(0);
    Ort::TypeInfo output_info = sess->GetOutputTypeInfo(0);
    auto input_shape = input_info.GetTensorTypeAndShapeInfo().GetShape();
    auto output_shape = output_info.GetTensorTypeAndShapeInfo().GetShape();
    if (input_info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_shape.size() != 2 || (input_shape[0] != 1 && input_shape[0] != -1) ||
        input_shape[1] != config.obs_dim ||
        output_shape.size() != 2 || (output_shape[0] != 1 && output_shape[0] != -1) ||
        output_shape[1] != config.action_dim) {
        ROS_ERROR("Grasp ONNX must have one float [%d] input and one float [%d] output",
                  config.obs_dim, config.action_dim); return false;
    }
    grasp_joint_state_sub = nh.subscribe(joint_state_topic, 1, graspJointStateCb);
    grasp_action_pub = nh.advertise<std_msgs::Float64MultiArray>(grasp_action_topic, 1);
    ROS_INFO("Grasp policy enabled: %d observations -> %d raw actions at %.1f Hz",
             config.obs_dim, config.action_dim, grasp_rate);
    return true;
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
    
    std::string model_path;
    nh.param<std::string>("model_path", model_path, "");
    if (model_path.empty() && task != "grasp") {
        std::string pkg_root = ros::package::getPath("pal_policy_deployer");
        std::string config_dir = pkg_root + "/models/" + task + "/" + task + "_" + model + "_mjlab/";
        ROS_INFO("Looking for ONNX model in: %s", config_dir.c_str());
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
    }

    if (model_path.empty() || !fs::is_regular_file(model_path)) {
        ROS_FATAL("Set ~model_path to an existing ONNX file");
        return 1;
    }
    ROS_INFO("Loading %s policy: %s", task.c_str(), model_path.c_str());

    if (task == "grasp") {
        if (!loadGraspPolicyConfig(nh)) return 1;
    } else {
        if (!loadPolicyConfig(model_path)) {
            ROS_FATAL("Failed to load policy configuration from ONNX metadata");
            return 1;
        }
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
    if (task != "grasp") {
        ROS_INFO("=== Policy Configuration ===");
        ROS_INFO("Run path: %s", config.run_path.c_str());
        ROS_INFO("Task: %s", config.task.c_str());
        ROS_INFO("Observation dimension: %d (single: %d x history: %d)",
                 config.obs_dim, config.single_obs_dim, config.history_length);
        ROS_INFO("Action dimension: %d (output index: %d)", config.action_dim, config.out_idx_actions);
        ROS_INFO("Joints: %zu obs, %zu action", config.joint_names.size(), config.action_joint_names.size());
    }
    if (task == "imitation") {
        ROS_INFO("Motion: autoregressive from robot state, command_dim=%d",
                 motion.commandDim());
    }
    
    if (task != "grasp") {
        std::stringstream obs_ss;
        for (size_t i = 0; i < config.observation_names.size(); ++i) {
            if (i > 0) obs_ss << ", ";
            obs_ss << config.observation_names[i];
        }
        ROS_INFO("Observation terms: %s", obs_ss.str().c_str());
        ROS_INFO("Initializing observation terms:");
        initializeObservationTerms();
    }
    
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

    if (task == "grasp") {
        if (sess->GetInputCount() != 1 || sess->GetOutputCount() != 1 ||
            !initializeGraspPolicy(nh)) {
            ROS_FATAL("Failed to initialize grasp policy");
            return 1;
        }
    }
    
    // Unsynced subscribers
    ros::Subscriber s1, s2, s3;
    if (task != "grasp") {
        s1 = nh.subscribe("/introspection_data/names", 1, palStatisticsNamesCb);
        s2 = nh.subscribe("/introspection_data/values", 1, palStatisticsValuesCb);
        s3 = nh.subscribe("/cmd_vel", 1, velCmdCB);
    }
    ros::Subscriber s4;
    if (task != "grasp" && obs_terms.find("base_lin_vel") != obs_terms.end()) {
        s4 = nh.subscribe("/odom", 1, odomCB);
        ROS_INFO("Subscribed to /odom for base_lin_vel");
    }

    /// Synced subscribers
    // message_filters::Subscriber<sensor_msgs::JointState> js_sub(nh, "/joint_states", 1);
    // message_filters::Subscriber<sensor_msgs::Imu>        imu_sub(nh, "/base_imu", 1);

    // message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(1), js_sub, imu_sub);
    /// For the policy, messages with as max this interval are the effectively the same
    // sync.setMaxIntervalDuration(ros::Duration(0.01));
    // sync.registerCallback(&syncedCB);

    if (task != "grasp")
        pub_sub_cmd = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);

    // The inference is now handled at the same time as the observations
    //ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCallback);
    
    ROS_INFO("Policy deployer started successfully!");
    if (task == "imitation") {
        ROS_INFO("Waiting for joint state to initialize motion...");
    }
    
    ros::Rate rate(task == "grasp" ? grasp_rate : 50.0);
    while(ros::ok())
    {
        ros::spinOnce();
        if (task == "grasp") graspInferenceTimer(ros::TimerEvent());
        rate.sleep();
    }
    
    return 0;
}
