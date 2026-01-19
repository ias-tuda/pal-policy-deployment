#include <ros/ros.h>
#include <ros/package.h>

// ros msgs
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64MultiArray.h>

// onnx / filesystem
#include <onnxruntime_cxx_api.h>
#include <boost/filesystem.hpp>

// core libraries
#include <vector>
#include <mutex>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace fs = boost::filesystem;

// ---- OBSERVATION INFORMATIONS ----

static const std::vector<std::string> isaaclab_order {
  "pelvis_1_joint", 
  "leg_left_1_joint", "leg_right_1_joint", 
  "pelvis_2_joint", 
  "leg_left_2_joint", "leg_right_2_joint", 
  "arm_left_1_joint", "arm_right_1_joint", 
  "leg_left_3_joint", "leg_right_3_joint", 
  "arm_left_2_joint", "arm_right_2_joint", 
  "leg_left_length_joint", "leg_left_femur_joint", 
  "leg_right_length_joint", "leg_right_femur_joint", 
  "arm_left_3_joint", "arm_right_3_joint", 
  "leg_left_4_joint", "leg_left_knee_joint", 
  "leg_right_4_joint", "leg_right_knee_joint", 
  "arm_left_4_joint", "arm_right_4_joint", 
  "leg_left_5_joint", "leg_right_5_joint",
};

// Isaac Lab joint position offsets
static const std::vector<double> joint_pos_offset {
  0.0, 
  0.0, 0.0, 
  0.0, 
  0.05, 0.05, 
  0.24, -0.24, 
  0.0, 0.0, 
  1.32, 1.32, 
  0.6832, 1.1172, 
  0.6832, 1.1172, 
  1.57, -1.57, 
  -0.05, 2.2345, 
  -0.05, 2.2345, 
  0.8, 0.8, 
  0.0, 0.0
};


// ---- ACTION INFORMATIONS ----

static const std::vector<std::string> action_order {
  "pelvis_1_joint", 
  "leg_left_1_joint", "leg_right_1_joint", 
  "pelvis_2_joint", 
  "leg_left_2_joint", "leg_right_2_joint", 
  "arm_left_1_joint", "arm_right_1_joint", 
  "leg_left_3_joint", "leg_right_3_joint", 
  "arm_left_2_joint", "arm_right_2_joint", 
  "leg_left_length_joint", /*"leg_left_femur_joint",*/
  "leg_right_length_joint", /*"leg_right_femur_joint",*/ 
  "arm_left_3_joint", "arm_right_3_joint", 
  "leg_left_4_joint", /*"leg_left_knee_joint",*/  
  "leg_right_4_joint", /*"leg_right_knee_joint",*/  
  "arm_left_4_joint", "arm_right_4_joint", 
  "leg_left_5_joint", "leg_right_5_joint"
};

// offset in the order of joint_states joint names
static const std::vector<double> action_offset {
  0.0, 
  0.0, 0.0, 
  0.0, 
  0.05, 0.05, 
  0.24, -0.24, 
  0.0, 0.0, 
  1.32, 1.32, 
  0.6832, // 1.1172, 
  0.6832, // 1.1172, 
  1.57, -1.57, 
  -0.05, // 2.2345, 
  -0.05, // 2.2345, 
  0.8, 0.8, 
  0.0, 0.0
};


// 1 if this joint uses y_f, 0 if it’s an arm (fixed offset only)
static const std::vector<int> use_y = {
  1, 1, 1, 1, 1, 1,
  0, 0,
  1, 1,
  0, 0,
  1, 1,
  0, 0,
  1, 1,
  0, 0,
  1, 1
};

// Index in y_f if used, otherwise 0
static const std::vector<int> y_idx = {
   0, 1, 2, 3, 4, 5,
   0, 0,
   6, 7,
   0, 0,
   8, 9,
   0, 0,
   10, 11,
   0, 0,
   12, 13
};

const float action_scale = 0.25;

// Policy specific vars
const double period = 0.6;
double global_phase = 0.0;
ros::Time last_time;

// ONNX stuff
Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"rl");
std::unique_ptr<Ort::Session> sess;
Ort::MemoryInfo mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
std::vector<const char*> in_names, out_names;

ros::Publisher pub_sub_cmd;

std::vector<double> observation(77, 0.0);
std::mutex observation_mutex;

static constexpr int64_t input_shape[2] = {1, 77};
std::vector<double> action(22);
std::vector<double> zero_action(22, 0.0);


void timerCallback(const ros::TimerEvent&)
{
  // update phase and phase features
  ros::Time now = ros::Time::now();
  double dt = (now - last_time).toSec();
  last_time = now;
  if (dt < 0.0) dt = 0.0;

  global_phase += dt / period;
  global_phase = std::fmod(global_phase, 1.0);

  {
    std::lock_guard<std::mutex> lk(observation_mutex);
    observation[75] = std::sin(global_phase * 2.0 * M_PI);
    observation[76] = std::cos(global_phase * 2.0 * M_PI);
  }

  // copy obs under lock
  std::vector<float> obs_input;
  {
    std::lock_guard<std::mutex> lk(observation_mutex);
    obs_input.assign(observation.begin(), observation.end());   // 77 values
  }

  // run ONNX
  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(mem, obs_input.data(), obs_input.size(), input_shape, 2);

  auto outputs = sess->Run(Ort::RunOptions{nullptr},
                           in_names.data(),  &input_tensor, 1,
                           out_names.data(), 1);

  const float* y_f = outputs.front().GetTensorData<float>();

  // map policy outputs to joint positions
  for (size_t i = 0; i < 22; ++i) {
    action[i] = action_scale * y_f[y_idx[i]] * use_y[i] + action_offset[i];
  }
  // leave remaining joints at default zero_action if you want them fixed

  // store last action into observation[52..65]
  {
    std::lock_guard<std::mutex> lk(observation_mutex);
    for (size_t i = 0; i < 14; ++i) {
      observation[52 + i] = y_f[i];
    }
  }

  // publish joint state command
  sensor_msgs::JointState cmd_msg;
  cmd_msg.header.stamp = ros::Time::now();

  cmd_msg.name = action_order;
  cmd_msg.position = action;
  cmd_msg.velocity = zero_action;
  cmd_msg.effort = zero_action;

  pub_sub_cmd.publish(cmd_msg);
}


// ---- OBSERVATION CONSTRUCTION ----

void jsCB(const sensor_msgs::JointState::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(observation_mutex);

  for (size_t k = 0; k < isaaclab_order.size(); ++k) {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      if (msg->name[i] == isaaclab_order[k]) {
        observation[k]      = msg->position[i] - joint_pos_offset[k];
        observation[k + 26] = 0.05 * msg->velocity[i];
      }
    }
  }
}

// imu data (6)
void imuCB(const sensor_msgs::Imu::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(observation_mutex);

  // base_ang_vel
  observation[66] = msg->angular_velocity.x;
  observation[67] = msg->angular_velocity.y;
  observation[68] = msg->angular_velocity.z;

  // base_lin_acc
  observation[69] = msg->linear_acceleration.x;
  observation[70] = msg->linear_acceleration.y;
  observation[71] = msg->linear_acceleration.z;
}

// teleop_twist keyboard (3)
void velCmdCB(const geometry_msgs::Twist::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(observation_mutex);

  observation[72] = msg->linear.x;
  observation[73] = msg->linear.y;
  observation[74] = msg->angular.z;

  ROS_INFO("Velocity command: Vx: %f, Vy: %f, wz: %f",
           msg->linear.x, msg->linear.y, msg->angular.z);
}


int main(int argc,char** argv)
{
  ros::init(argc, argv, "pal_policy_deployer");
  ros::NodeHandle nh("~");
  
  // locate ONNX model
  std::string pkg_root = ros::package::getPath("pal_policy_deployer");
  std::string config_dir = pkg_root + "/models/loco/loco_v21/exported/";
  std::string model_path;
  bool found = false;

  for (auto& entry : fs::directory_iterator(config_dir)) {
    fs::path p = entry.path();
    if (fs::is_regular_file(p) && p.extension() == ".onnx") {
      model_path = p.string();
      found = true;
      ROS_INFO("Found ONNX model: %s", model_path.c_str());
      break;
    }
  }

  if (!found) {
    ROS_FATAL("No ONNX model found in %s", config_dir.c_str());
    return 1;
  }

  sess.reset(new Ort::Session(env, model_path.c_str(), Ort::SessionOptions{}));

  in_names.push_back(sess->GetInputNameAllocated (0, Ort::AllocatorWithDefaultOptions()).release());
  out_names.push_back(sess->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release());

  last_time = ros::Time::now();

  ros::Subscriber s1 = nh.subscribe("/joint_states", 1, jsCB);
  ros::Subscriber s2 = nh.subscribe("/base_imu", 1, imuCB);
  ros::Subscriber s3 = nh.subscribe("/cmd_vel", 1, velCmdCB);

  pub_sub_cmd = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);
  
  ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCallback);

  ros::spin();
}
