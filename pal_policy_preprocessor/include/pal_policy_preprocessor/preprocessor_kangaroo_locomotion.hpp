/**
 * @file preprocessor_kangaroo_locomotion.hpp
 * @brief Preprocessor for Kangaroo robot locomotion policy observations
 * 
 * This file defines the KangarooLocoPreprocessor class which handles observation
 * data collection and formatting for locomotion policies. It provides a flexible
 * system for dynamically configuring observation terms and their ordering.
 */

#pragma once

#include "preprocessor.hpp"

#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Geometry>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace pal_policy_preprocessor
{

/**
 * @struct ObservationTerm
 * @brief Represents a single observation term with its metadata and current values
 * 
 * Each term corresponds to a specific type of sensor data (e.g., joint positions,
 * IMU data) and tracks its dimension, position in the observation vector, and
 * current values.
 */
struct ObservationTerm {
  std::string name;              ///< Name identifier for the observation term
  int dim;                       ///< Dimensionality of the observation term
  int start_idx;                 ///< Starting index in the flattened observation vector
  std::vector<double> current;   ///< Current values of the observation term
  
  /**
   * @brief Construct a new Observation Term
   * @param n Name of the observation term
   * @param d Dimension of the observation term
   */
  ObservationTerm(const std::string& n, int d) 
      : name(n), dim(d), start_idx(0), current(d, 0.0) {}
};

/**
 * @struct PolicyConfig
 * @brief Configuration parameters for the locomotion policy
 * 
 * Contains all the configuration needed to set up the preprocessor including
 * joint information, observation structure, and input/output dimensions.
 */
struct PolicyConfig
{
  std::vector<std::string> joint_names;        ///< Names of robot joints
  std::vector<double> default_joint_pos;       ///< Default positions for each joint
  std::vector<std::string> observation_names;  ///< Ordered list of observation terms to include

  int input_dim{0};   ///< Total input dimension (should match sum of observation term dims)
  int output_dim{0};  ///< Total output dimension (number of action outputs)
};

/**
 * @class KangarooLocoPreprocessor
 * @brief Preprocessor for Kangaroo robot locomotion observations
 * 
 * This class subscribes to various sensor topics (joint states, IMU, odometry, etc.)
 * and constructs observation vectors for locomotion policies. The observation structure
 * is dynamically configurable through ROS2 parameters, allowing flexible compositions
 * of different sensor data.
 * 
 * Supported observation terms:
 * - joint_pos: Joint positions (relative to default)
 * - joint_vel: Joint velocities
 * - actions: Previous action outputs
 * - command: Velocity commands (vx, vy, wz)
 * - base_ang_vel: Base angular velocity from IMU
 * - base_lin_acc: Base linear acceleration from IMU
 * - base_lin_vel: Base linear velocity from odometry
 * - imu_projected_gravity: Gravity vector projected into body frame
 */
class KangarooLocoPreprocessor : public Preprocessor
{
public:
  /**
   * @brief Construct a new Kangaroo Loco Preprocessor
   * 
   * Initializes the preprocessor by:
   * 1. Declaring and reading ROS2 parameters
   * 2. Validating configuration
   * 3. Building internal data structures
   * 4. Creating subscriptions to sensor topics
   * 
   * @throws std::runtime_error if configuration is invalid
   */
  explicit KangarooLocoPreprocessor();

protected:
  /**
   * @brief Build the observation vector from current sensor data
   * 
   * Constructs the flattened observation vector by copying data from all configured
   * observation terms in the specified order. Thread-safe.
   * 
   * @param obs Pointer to the output observation vector (must be pre-allocated)
   */
  void buildInputVector(std::vector<double> * obs) const override;

private:
  std::string getObservationElementLabel(const std::string & term_name, int local_idx) const;
  std::string getObservationElementSource(const std::string & term_name) const;
  void maybeLogObservationVector(const std::vector<double> & obs) const;

  /**
   * @brief Build mapping from joint names to indices
   * 
   * Creates an efficient lookup table for finding joint indices by name,
   * used in the joint state callback for O(1) access.
   */
  void buildJointIndexMap();
  
  /**
   * @brief Initialize all observation terms based on configuration
   * 
   * Creates ObservationTerm objects for each configured observation name,
   * calculates their positions in the observation vector, and validates
   * that all terms are recognized.
   * 
   * @throws Logs fatal error and shuts down if unknown term is encountered
   */
  void initializeObservationTerms();
  
  /**
   * @brief Get the dimension of a specific observation term
   * 
   * @param term_name Name of the observation term
   * @return int Dimension of the term, or -1 if unknown
   */
  int getObservationTermDim(const std::string& term_name) const;

  // ---- Sensor Callbacks ----
  
  /**
   * @brief Joint state callback
   * 
   * Updates joint_pos and joint_vel observation terms from incoming joint state
   * messages. Thread-safe.
   * 
   * @param msg Joint state message containing positions and velocities
   */
  void jsCB(const sensor_msgs::msg::JointState::SharedPtr msg);
  
  /**
   * @brief IMU callback
   * 
   * Updates base_ang_vel, base_lin_acc, and imu_projected_gravity observation
   * terms from incoming IMU messages. Thread-safe.
   * 
   * @param msg IMU message containing orientation, angular velocity, and linear acceleration
   */
  void imuCB(const sensor_msgs::msg::Imu::SharedPtr msg);
  
  /**
   * @brief Odometry callback
   * 
   * Updates base_lin_vel observation term from incoming odometry messages.
   * Thread-safe.
   * 
   * @param msg Odometry message containing linear velocity in twist
   */
  void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg);
  
  /**
   * @brief Velocity command callback
   * 
   * Updates command observation term from incoming twist messages.
   * Thread-safe.
   * 
   * @param msg Twist message containing linear and angular velocity commands
   */
  void cmdCB(const geometry_msgs::msg::Twist::SharedPtr msg);
  
  /**
   * @brief Last action callback
   * 
   * Updates actions observation term with the previous policy output.
   * Thread-safe.
   * 
   * @param msg Float array containing the last action values
   */
  void lastCB(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  // ---- Configuration ----
  PolicyConfig config_;  ///< Policy configuration parameters

  // ---- Topics / Debug ----
  std::string js_topic_, imu_topic_, odom_topic_, cmd_topic_, last_action_topic_;
  bool debug_print_observation_vector_{false};
  int64_t debug_print_every_n_cycles_{50};
  mutable int64_t debug_cycle_counter_{0};

  // ---- Observation Management ----
  std::map<std::string, std::shared_ptr<ObservationTerm>> obs_terms_;  ///< Map of all observation terms

  // ---- Joint Management ----
  std::unordered_map<std::string, size_t> joint_name_to_index_;  ///< Fast lookup: joint name -> index

  // ---- IMU State ----
  Eigen::Quaterniond imu_orientation_;  ///< Current IMU orientation for gravity projection

  // ---- Thread Safety ----
  mutable std::mutex obs_mutex_;  ///< Protects obs_terms_ and imu_orientation_ from concurrent access

  // ---- ROS2 Subscriptions ----
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;        ///< Joint states subscription
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;              ///< IMU subscription
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;           ///< Odometry subscription
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_;          ///< Velocity command subscription
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_last_;  ///< Last action subscription
};

} // pal_policy_preprocessor
