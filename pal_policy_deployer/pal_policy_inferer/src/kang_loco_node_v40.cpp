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
// is it all needed ??
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
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fs = boost::filesystem;

// ---- OBSERVATION INFORMATIONS ----

static const std::vector<std::string> obs_order {
  "pelvis_1_joint", "pelvis_2_joint",
  "arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint",
  "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint",
  "leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", 
  "leg_left_length_joint", 
  "leg_left_4_joint", "leg_left_5_joint",
  "leg_left_femur_joint", "leg_left_knee_joint", 
  "leg_right_1_joint", "leg_right_2_joint", "leg_right_3_joint", 
  "leg_right_length_joint",
  "leg_right_4_joint", "leg_right_5_joint",
  "leg_right_femur_joint", "leg_right_knee_joint"
};



static const std::vector<double> obs_offset {
  0.000, 0.000,
  0.240, 1.320, 1.570, 0.800, 
  -0.240, 1.320, -1.570, 0.800, 
  -0.012, 0.054, 0.040, 
  0.600, 
  -0.053, 0.000, 
  0.900, 1.800, 
  0.012, 0.054, -0.040, 
  0.600, 
  -0.053, 0.000, 
  0.900, 1.800
};

// ---- ACTION INFORMATIONS ----

static const std::vector<std::string> action_order {
  "pelvis_1_joint", "pelvis_2_joint",
  "arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint",
  "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint",
  "leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", 
  "leg_left_length_joint", 
  "leg_left_4_joint", "leg_left_5_joint",
  "leg_right_1_joint", "leg_right_2_joint", "leg_right_3_joint", 
  "leg_right_length_joint",
  "leg_right_4_joint", "leg_right_5_joint",
};
static const std::vector<double> action_offset {
  0.000, 0.000,
  0.240, 1.320, 1.570, 0.800, 
  -0.240, 1.320, -1.570, 0.800, 
  -0.012, 0.054, 0.040, 
  0.600, 
  -0.053, 0.000, 
  0.012, 0.054, -0.040, 
  0.600, 
  -0.053, 0.000
};
static const std::vector<double> action_scale {
  0.250, 0.250,
  0.250, 0.250, 0.239, 0.239,
  0.250, 0.250, 0.239, 0.239,
  0.200, 0.575, 0.347, 
  0.172, 
  1.167, 0.683,
  0.200, 0.575, 0.347, 
  0.172,
  1.167, 0.683
};

// Policy specific vars

Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"rl");
std::unique_ptr<Ort::Session> sess;
Ort::MemoryInfo mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault));
std::vector<const char*> in_names, out_names;

std::vector<double> observation(83, 0.0);
std::mutex observation_mutex;

static constexpr int64_t input_shape[2] = {1, 83};
std_msgs::Float64MultiArray cmd, last;

std::vector<double> action(22);
std::vector<double> zero_action(22, 0.0);

ros::Publisher pub_sub_cmd;

void timerCallback(const ros::TimerEvent&)
{
  std::vector<float> obs_input;
  {
    std::lock_guard<std::mutex> lk(observation_mutex);
    obs_input.assign(observation.begin(), observation.end());   // 86 values
  }

  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(mem, obs_input.data(), obs_input.size(), input_shape, 2);

  auto outputs = sess->Run(Ort::RunOptions{nullptr},
                           in_names.data(),  &input_tensor, 1,
                           out_names.data(), 1);

  const float* y_f = outputs.front().GetTensorData<float>();

  for (size_t i = 0; i < 22; ++i) {
    action[i] = action_scale[i] * y_f[i] + action_offset[i];
    observation[55+i] = y_f[i];
  }

  sensor_msgs::JointState cmd_msg;
  cmd_msg.header.stamp = ros::Time::now();

  cmd_msg.name = action_order;
  cmd_msg.position = action;
  cmd_msg.velocity = zero_action;
  cmd_msg.effort = zero_action;

  pub_sub_cmd.publish(cmd_msg);
}

// OBSERVATION CONSTRUCTION 

// joint pos and joint vel (26 + 26 = 52)
void jsCB(const sensor_msgs::JointState::ConstPtr& msg)
{
  for (size_t k = 0; k < obs_order.size(); ++k) {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      if (msg->name[i] == obs_order[k]) {
        observation[3+k] = msg->position[i] - obs_offset[k];
        observation[3+k+26] = msg->velocity[i];
        // ROS_INFO("Isaac Lab name -> %s | joint_state name -> %s", obs_order[k].c_str(), msg->name[i].c_str());
      }
    }
  }
}

// imu data (6)
void imuCB(const sensor_msgs::Imu::ConstPtr& msg)
{
    // base_ang_vel
    observation[0] = msg->angular_velocity.x;
    observation[1] = msg->angular_velocity.y;
    observation[2] = msg->angular_velocity.z;
    // ROS_INFO("Base angular velocity: x: %f, y: %f, z: %f\n", msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

    // base_lin_acc
    observation[80] = msg->linear_acceleration.x;
    observation[81] = msg->linear_acceleration.y;
    observation[82] = msg->linear_acceleration.z;
    // ROS_INFO("Base linear acceleration: x: %f, y: %f, z: %f\n", msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
}

// rosrun teleop_twist_keyboard teleop_twist_keyboard.py (3)
void velCmdCB(const geometry_msgs::Twist::ConstPtr& msg)
{
    observation[77] = msg->linear.x;
    observation[78] = msg->linear.y;
    observation[79] = msg->angular.z;

    ROS_INFO("Velocity command: Vx: %f, Vy: %f, wz: %f\n", msg->linear.x, msg->linear.y, msg->angular.z);
}


int main(int argc,char** argv)
{
  ros::init(argc,argv,"pal_policy_deployer"); ros::NodeHandle nh("~");
  
  std::string pkg_root = ros::package::getPath("pal_policy_deployer");
  std::string config_dir = pkg_root + "/models/loco/loco_v40_mjlab_temp/";
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
    ROS_FATAL("No ONNX model found in %s, and fallback '%s' doesn't exist.",
              config_dir.c_str(), model_path.c_str());
    return 1;
  }

  sess.reset(new Ort::Session(env,model_path.c_str(), Ort::SessionOptions{}));

  in_names .push_back(sess->GetInputNameAllocated (0, Ort::AllocatorWithDefaultOptions()).release());
  out_names.push_back(sess->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release());

  ros::Subscriber s1 = nh.subscribe("/joint_states", 1, jsCB);
  ros::Subscriber s2 = nh.subscribe("/base_imu", 1, imuCB);
  // ros::Subscriber s2 = nh.subscribe("/floating_base_pose_simulated", 1, baseStateCB);
  ros::Subscriber s3 = nh.subscribe("/cmd_vel", 1, velCmdCB);

  pub_sub_cmd = nh.advertise<sensor_msgs::JointState>("/subscriber_controller/desired_state", 1);
  
  ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCallback);

  ros::spin();
}
