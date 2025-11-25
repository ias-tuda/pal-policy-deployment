#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

struct SpeedMode {
  double min_x, max_x;
  double min_y, max_y;
  double min_wz, max_wz;
};

class JoyTwistController {
public:
  JoyTwistController(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh) {
    // Params
    pnh_.param<std::string>("joy_topic", joy_topic_, "/joy");
    pnh_.param<std::string>("cmd_vel_topic", cmd_vel_topic_, "/cmd_vel");
    
    // Your specific axis/button mappings
    pnh_.param<int>("axis_vx_index", axis_vx_idx_, 1);        // Axis 1 -> linear.x
    pnh_.param<int>("axis_vy_index", axis_vy_idx_, 0);        // Axis 0 -> linear.y (inverted)
    pnh_.param<int>("axis_wz_index", axis_wz_idx_, 2);        // Axis 2 -> angular.z (inverted)
    pnh_.param<int>("btn_rt_index", btn_rt_idx_, 7);          // Button 7 -> RT
    pnh_.param<int>("btn_lt_index", btn_lt_idx_, 6);          // Button 6 -> LT
    
    pnh_.param<double>("deadzone", deadzone_, 0.1);           // Joystick deadzone
    pnh_.param<double>("display_rate", display_rate_, 10.0);  // Display update rate (Hz)

    // Speed modes
    // Mode 0: Normal (no triggers)
    modes_[0].min_x = -1.0; modes_[0].max_x = 1.0;
    modes_[0].min_y = -1.0; modes_[0].max_y = 1.0;
    modes_[0].min_wz = -0.5; modes_[0].max_wz = 0.5;
    
    // Mode 1: RT pressed
    modes_[1].min_x = -1.5; modes_[1].max_x = 2.0;
    modes_[1].min_y = -1.0; modes_[1].max_y = 1.0;
    modes_[1].min_wz = -0.7; modes_[1].max_wz = 0.7;
    
    // Mode 2: Both LT and RT pressed
    modes_[2].min_x = -2.0; modes_[2].max_x = 3.0;
    modes_[2].min_y = -1.0; modes_[2].max_y = 1.0;
    modes_[2].min_wz = -0.7; modes_[2].max_wz = 0.7;

    pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_vel_topic_, 1);
    sub_ = nh_.subscribe(joy_topic_, 10, &JoyTwistController::joyCB, this);

    last_display_time_ = ros::Time::now();

    ROS_INFO_STREAM("==============================================");
    ROS_INFO_STREAM("[joy_twist_controller] Started");
    ROS_INFO_STREAM("==============================================");
    ROS_INFO_STREAM("Subscribing to: " << joy_topic_);
    ROS_INFO_STREAM("Publishing to: " << cmd_vel_topic_);
    ROS_INFO("\nControl mapping:");
    ROS_INFO("  Axis 1       -> linear.x (Vx)");
    ROS_INFO("  Axis 0       -> linear.y (Vy) [INVERTED]");
    ROS_INFO("  Axis 2       -> angular.z (wz) [INVERTED]");
    ROS_INFO("  Button 7 (RT) -> Speed mode 1");
    ROS_INFO("  Button 6 (LT) + Button 7 (RT) -> Speed mode 2");
    ROS_INFO_STREAM("==============================================\n");
  }

private:
  // Apply deadzone and map joystick value from [-1,1] to output range [min, max]
  double mapAxis(double raw_value, double min_out, double max_out, bool invert = false) {
    // Invert if needed
    if (invert) {
      raw_value = -raw_value;
    }
    
    // Apply deadzone
    if (std::abs(raw_value) < deadzone_) {
      return 0.0;
    }
    
    // Map from [-1, 1] to [min_out, max_out]
    if (raw_value > 0) {
      return raw_value * max_out;
    } else {
      return raw_value * std::abs(min_out);
    }
  }

  // Create a visual bar showing current value within min/max range
  std::string createBar(double value, double min_val, double max_val, int width = 30) {
    double normalized = (value - min_val) / (max_val - min_val);
    normalized = std::max(0.0, std::min(1.0, normalized));
    
    int pos = static_cast<int>(normalized * width);
    
    std::string bar = "[";
    for (int i = 0; i < width; ++i) {
      if (i == pos) {
        bar += "|";
      } else if (i == 0 || i == width - 1) {
        bar += "|";
      } else {
        bar += "-";
      }
    }
    bar += "]";
    return bar;
  }

  void displayStatus(int mode, const geometry_msgs::Twist& twist) {
    const SpeedMode& current_mode = modes_[mode];
    
    // Clear screen (optional - comment out if you don't want screen clearing)
    // std::cout << "\033[2J\033[1;1H";
    
    ROS_INFO_STREAM("\n========== Velocity Command Status ==========");
    ROS_INFO_STREAM("Speed Mode: " << mode << " (LT:" << (mode >= 2 ? "ON" : "OFF") 
                   << " RT:" << (mode >= 1 ? "ON" : "OFF") << ")");
    ROS_INFO_STREAM("---------------------------------------------");
    
    // Linear X
    std::string bar_x = createBar(twist.linear.x, current_mode.min_x, current_mode.max_x);
    ROS_INFO_STREAM(std::fixed << std::setprecision(2)
                   << "Vx:  " << std::setw(6) << twist.linear.x 
                   << " m/s  " << bar_x);
    ROS_INFO_STREAM("     Range: [" << current_mode.min_x << ", " << current_mode.max_x << "]");
    
    // Linear Y
    std::string bar_y = createBar(twist.linear.y, current_mode.min_y, current_mode.max_y);
    ROS_INFO_STREAM(std::fixed << std::setprecision(2)
                   << "Vy:  " << std::setw(6) << twist.linear.y 
                   << " m/s  " << bar_y);
    ROS_INFO_STREAM("     Range: [" << current_mode.min_y << ", " << current_mode.max_y << "]");
    
    // Angular Z
    std::string bar_wz = createBar(twist.angular.z, current_mode.min_wz, current_mode.max_wz);
    ROS_INFO_STREAM(std::fixed << std::setprecision(2)
                   << "wz:  " << std::setw(6) << twist.angular.z 
                   << " rad/s " << bar_wz);
    ROS_INFO_STREAM("     Range: [" << current_mode.min_wz << ", " << current_mode.max_wz << "]");
    
    ROS_INFO_STREAM("=============================================\n");
  }

  void joyCB(const sensor_msgs::Joy::ConstPtr& msg) {
    // Defensive checks
    if (axis_vx_idx_ >= static_cast<int>(msg->axes.size()) ||
        axis_vy_idx_ >= static_cast<int>(msg->axes.size()) ||
        axis_wz_idx_ >= static_cast<int>(msg->axes.size()) ||
        btn_rt_idx_ >= static_cast<int>(msg->buttons.size()) ||
        btn_lt_idx_ >= static_cast<int>(msg->buttons.size())) {
      ROS_WARN_THROTTLE(2.0, "Joy message doesn't match expected axis/button indices.");
      return;
    }

    // Determine speed mode based on triggers
    bool rt_pressed = (msg->buttons[btn_rt_idx_] != 0);
    bool lt_pressed = (msg->buttons[btn_lt_idx_] != 0);
    
    int mode = 0;
    if (lt_pressed && rt_pressed) {
      mode = 2;  // Both triggers
    } else if (rt_pressed) {
      mode = 1;  // RT only
    }

    const SpeedMode& current_mode = modes_[mode];

    // Map joystick inputs to velocities
    geometry_msgs::Twist twist;
    
    // Axis 1 controls linear.x (NOT inverted)
    twist.linear.x = mapAxis(msg->axes[axis_vx_idx_], 
                             current_mode.min_x, current_mode.max_x, false);
    
    // Axis 0 controls linear.y (INVERTED)
    twist.linear.y = mapAxis(msg->axes[axis_vy_idx_], 
                             current_mode.min_y, current_mode.max_y, true);
    
    // Axis 2 controls angular.z (INVERTED)
    twist.angular.z = mapAxis(msg->axes[axis_wz_idx_], 
                              current_mode.min_wz, current_mode.max_wz, true);

    pub_.publish(twist);

    // Display status at specified rate
    ros::Time now = ros::Time::now();
    if ((now - last_display_time_).toSec() >= 1.0 / display_rate_) {
      displayStatus(mode, twist);
      last_display_time_ = now;
    }
  }

private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  std::string joy_topic_, cmd_vel_topic_;
  int axis_vx_idx_, axis_vy_idx_, axis_wz_idx_;
  int btn_rt_idx_, btn_lt_idx_;
  double deadzone_;
  double display_rate_;
  SpeedMode modes_[3];
  ros::Time last_display_time_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "joy_twist_controller");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  JoyTwistController node(nh, pnh);
  ros::spin();
  return 0;
}