#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>

class BaseImuFix {
public:
  BaseImuFix(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh), pnh_(pnh) {
    pnh_.param<std::string>("input_imu_topic", input_imu_topic_, "/base_imu");
    pnh_.param<std::string>("odom_topic", odom_topic_, "/floating_base_pose_simulated");
    pnh_.param<std::string>("output_imu_topic", output_imu_topic_, "/base_imu_fake");
    pnh_.param<double>("gravity", gravity_, 9.81);
    pnh_.param<bool>("use_odom_acceleration", use_odom_acceleration_, false);
    pnh_.param<double>("accel_alpha", accel_alpha_, 0.2);

    accel_alpha_ = std::max(0.0, std::min(1.0, accel_alpha_));

    pub_ = nh_.advertise<sensor_msgs::Imu>(output_imu_topic_, 10);
    imu_sub_ = nh_.subscribe(input_imu_topic_, 10, &BaseImuFix::imuCB, this);

    if (use_odom_acceleration_) {
      odom_sub_ = nh_.subscribe(odom_topic_, 50, &BaseImuFix::odomCB, this);
    }

    ROS_INFO_STREAM("[base_imu_fix] " << input_imu_topic_ << " -> " << output_imu_topic_);
    ROS_INFO_STREAM("[base_imu_fix] linear_acceleration = R_imu_world * ([odom_accel] - gravity)");
    ROS_INFO_STREAM("[base_imu_fix] use_odom_acceleration=" << (use_odom_acceleration_ ? "true" : "false"));
  }

private:
  void odomCB(const nav_msgs::Odometry::ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mutex_);

    const ros::Time stamp = msg->header.stamp;
    Eigen::Vector3d vel(msg->twist.twist.linear.x,
                        msg->twist.twist.linear.y,
                        msg->twist.twist.linear.z);

    if (!have_odom_) {
      last_odom_stamp_ = stamp;
      last_odom_vel_ = vel;
      have_odom_ = true;
      return;
    }

    const double dt = (stamp - last_odom_stamp_).toSec();
    if (dt > 1e-4 && dt < 0.2) {
      Eigen::Vector3d raw_accel = (vel - last_odom_vel_) / dt;
      odom_accel_world_ = accel_alpha_ * raw_accel + (1.0 - accel_alpha_) * odom_accel_world_;
      have_odom_accel_ = true;
    }

    last_odom_stamp_ = stamp;
    last_odom_vel_ = vel;
  }

  void imuCB(const sensor_msgs::Imu::ConstPtr& msg) {
    sensor_msgs::Imu fixed = *msg;

    const Eigen::Quaterniond q_world_imu(msg->orientation.w,
                                         msg->orientation.x,
                                         msg->orientation.y,
                                         msg->orientation.z);
    const Eigen::Quaterniond q_imu_world = q_world_imu.conjugate();

    Eigen::Vector3d accel_world(0.0, 0.0, 0.0);
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (use_odom_acceleration_ && have_odom_accel_) {
        accel_world = odom_accel_world_;
      }
    }

    const Eigen::Vector3d gravity_world(0.0, 0.0, -gravity_);
    const Eigen::Vector3d proper_accel_imu = q_imu_world * (accel_world - gravity_world);

    fixed.linear_acceleration.x = proper_accel_imu.x();
    fixed.linear_acceleration.y = proper_accel_imu.y();
    fixed.linear_acceleration.z = proper_accel_imu.z();

    fixed.linear_acceleration_covariance = msg->linear_acceleration_covariance;
    pub_.publish(fixed);

    ROS_INFO_THROTTLE(1.0,
      "[base_imu_fix] raw acc=(%.3f %.3f %.3f), fixed acc=(%.3f %.3f %.3f)",
      msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
      fixed.linear_acceleration.x, fixed.linear_acceleration.y, fixed.linear_acceleration.z);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber imu_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher pub_;

  std::string input_imu_topic_;
  std::string odom_topic_;
  std::string output_imu_topic_;
  double gravity_ = 9.81;
  bool use_odom_acceleration_ = false;
  double accel_alpha_ = 0.2;

  std::mutex mutex_;
  bool have_odom_ = false;
  bool have_odom_accel_ = false;
  ros::Time last_odom_stamp_;
  Eigen::Vector3d last_odom_vel_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d odom_accel_world_ = Eigen::Vector3d::Zero();
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "base_imu_fix");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  BaseImuFix node(nh, pnh);
  ros::spin();
  return 0;
}
