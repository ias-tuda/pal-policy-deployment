#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System Identification Data Collector for KANGAROO Robot
Collects step response, sinusoidal sweep, and ramp input data for sim-to-real analysis.

Compatible with Python 2.7 / ROS Melodic
"""

from __future__ import print_function, division

import rospy
import csv
import os
import math
import threading
from datetime import datetime
from sensor_msgs.msg import JointState


# Joint configurations from the MuJoCo XML
# Format: (name, lower_limit, upper_limit, default_pos, is_prismatic)
LEG_JOINTS = {
    # Left leg
    "leg_left_1_joint": (-0.261799, 0.698132, 0.0, False),
    "leg_left_2_joint": (-0.741765, 0.663225, 0.0, False),
    "leg_left_3_joint": (-0.471239, 0.471239, 0.0, False),
    "leg_left_length_joint": (0.131973, 0.714166, 0.4, True),
    "leg_left_4_joint": (-0.741765, 0.663225, 0.0, False),
    "leg_left_5_joint": (-0.471239, 0.471239, 0.0, False),
    # Right leg
    "leg_right_1_joint": (-0.698132, 0.261799, 0.0, False),
    "leg_right_2_joint": (-0.741765, 0.663225, 0.0, False),
    "leg_right_3_joint": (-0.471239, 0.471239, 0.0, False),
    "leg_right_length_joint": (0.131973, 0.714166, 0.4, True),
    "leg_right_4_joint": (-0.741765, 0.663225, 0.0, False),
    "leg_right_5_joint": (-0.471239, 0.471239, 0.0, False),
}


def clamp(value, min_val, max_val):
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


class SysIdDataCollector(object):
    def __init__(self):
        rospy.init_node("sysid_data_collector", anonymous=True)
        
        # Parameters
        self.rate_hz = 50  # Fixed at 50 Hz to match policy deployer
        self.output_dir = rospy.get_param("~output_dir", "/home/user/exchange/tmp/sysid_data")
        self.test_joint = rospy.get_param("~test_joint", "all")
        self.test_type = rospy.get_param("~test_type", "all")
        
        # Step test params
        self.step_amplitude = rospy.get_param("~step_amplitude", 0.2)
        self.step_hold_time = rospy.get_param("~step_hold_time", 2.0)
        self.step_settle_time = rospy.get_param("~step_settle_time", 1.0)
        
        # Sinusoid test params
        self.sin_frequencies = rospy.get_param("~sin_frequencies", [0.5, 1.0, 2.0, 3.0, 5.0])
        self.sin_amplitude = rospy.get_param("~sin_amplitude", 0.15)
        self.sin_cycles = rospy.get_param("~sin_cycles", 5)
        
        # Ramp test params
        self.ramp_velocity = rospy.get_param("~ramp_velocity", 0.1)
        self.ramp_range = rospy.get_param("~ramp_range", 0.3)
        
        # State
        self.current_joint_states = {}
        self.joint_states_lock = threading.Lock()
        self.data_records = []
        self.is_recording = False
        self.start_time = 0.0
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Publisher (same as policy deployer)
        self.cmd_pub = rospy.Publisher(
            "/subscriber_controller/desired_state",
            JointState,
            queue_size=1
        )
        
        # Subscriber
        self.js_sub = rospy.Subscriber(
            "/joint_states",
            JointState,
            self.joint_states_callback,
            queue_size=1
        )
        
        self.rate = rospy.Rate(self.rate_hz)
        
        rospy.loginfo("SysId Data Collector initialized")
        rospy.loginfo("  Output dir: %s", self.output_dir)
        rospy.loginfo("  Rate: 50 Hz (fixed)")
        rospy.loginfo("  Test joint: %s", self.test_joint)
        rospy.loginfo("  Test type: %s", self.test_type)

    def joint_states_callback(self, msg):
        """Store current joint states."""
        with self.joint_states_lock:
            for i, name in enumerate(msg.name):
                pos = msg.position[i] if i < len(msg.position) else 0.0
                vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                eff = msg.effort[i] if i < len(msg.effort) else 0.0
                self.current_joint_states[name] = {
                    "position": pos,
                    "velocity": vel,
                    "effort": eff,
                }

    def get_joint_state(self, joint_name):
        """Get current state for a joint."""
        with self.joint_states_lock:
            return self.current_joint_states.get(joint_name, {
                "position": 0.0, "velocity": 0.0, "effort": 0.0
            })

    def publish_command(self, joint_names, positions):
        """Publish position command to joints."""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names
        msg.position = positions
        msg.velocity = [0.0] * len(joint_names)
        msg.effort = [0.0] * len(joint_names)
        self.cmd_pub.publish(msg)

    def record_data(self, test_type, joint_name, cmd_pos, test_param):
        """Record a single data point."""
        if not self.is_recording:
            return
        
        state = self.get_joint_state(joint_name)
        timestamp = rospy.Time.now().to_sec() - self.start_time
        
        record = {
            "timestamp": timestamp,
            "test_type": test_type,
            "joint_name": joint_name,
            "cmd_position": cmd_pos,
            "meas_position": state["position"],
            "meas_velocity": state["velocity"],
            "meas_effort": state["effort"],
            "test_param": test_param,
        }
        self.data_records.append(record)

    def go_to_position(self, joint_name, target_pos, duration=2.0):
        """Smoothly move to a target position."""
        lower, upper, default, is_prismatic = LEG_JOINTS[joint_name]
        current_state = self.get_joint_state(joint_name)
        start_pos = current_state["position"]
        
        # Clamp target to limits
        target_pos = clamp(target_pos, lower, upper)
        
        steps = int(duration * self.rate_hz)
        for i in range(steps):
            if rospy.is_shutdown():
                return
            t = (i + 1) / float(steps)
            # Smooth interpolation (ease in-out)
            t_smooth = t * t * (3 - 2 * t)
            pos = start_pos + (target_pos - start_pos) * t_smooth
            self.publish_command([joint_name], [pos])
            self.rate.sleep()

    def run_step_test(self, joint_name):
        """Run step response test on a single joint."""
        lower, upper, default, is_prismatic = LEG_JOINTS[joint_name]
        rospy.loginfo("[STEP] Testing %s", joint_name)
        
        # Calculate safe step positions within limits
        mid_pos = (lower + upper) / 2.0
        range_available = (upper - lower) / 2.0
        step_size = min(self.step_amplitude, range_available * 0.4)
        
        pos_low = mid_pos - step_size
        pos_high = mid_pos + step_size
        
        test_param = "step_size=%.3f" % step_size
        rospy.loginfo("  Step from %.3f to %.3f", pos_low, pos_high)
        
        # Go to starting position
        self.go_to_position(joint_name, pos_low, duration=2.0)
        rospy.sleep(self.step_settle_time)
        
        # Start recording
        self.is_recording = True
        self.start_time = rospy.Time.now().to_sec()
        
        # Hold at low position (baseline)
        for _ in range(int(self.step_settle_time * self.rate_hz)):
            if rospy.is_shutdown():
                return
            self.publish_command([joint_name], [pos_low])
            self.record_data("step", joint_name, pos_low, test_param)
            self.rate.sleep()
        
        # Step up
        for _ in range(int(self.step_hold_time * self.rate_hz)):
            if rospy.is_shutdown():
                return
            self.publish_command([joint_name], [pos_high])
            self.record_data("step", joint_name, pos_high, test_param)
            self.rate.sleep()
        
        # Step down
        for _ in range(int(self.step_hold_time * self.rate_hz)):
            if rospy.is_shutdown():
                return
            self.publish_command([joint_name], [pos_low])
            self.record_data("step", joint_name, pos_low, test_param)
            self.rate.sleep()
        
        self.is_recording = False

    def run_sinusoid_test(self, joint_name):
        """Run sinusoidal sweep test on a single joint."""
        lower, upper, default, is_prismatic = LEG_JOINTS[joint_name]
        rospy.loginfo("[SINUSOID] Testing %s", joint_name)
        
        # Calculate safe amplitude within limits
        mid_pos = (lower + upper) / 2.0
        range_available = (upper - lower) / 2.0
        amplitude = min(self.sin_amplitude, range_available * 0.4)
        
        rospy.loginfo("  Center: %.3f, Amplitude: %.3f", mid_pos, amplitude)
        
        # Go to center position
        self.go_to_position(joint_name, mid_pos, duration=2.0)
        rospy.sleep(0.5)
        
        for freq in self.sin_frequencies:
            rospy.loginfo("  Frequency: %.2f Hz", freq)
            test_param = "freq=%.2fHz,amp=%.3f" % (freq, amplitude)
            
            period = 1.0 / freq
            duration = self.sin_cycles * period
            steps = int(duration * self.rate_hz)
            
            self.is_recording = True
            self.start_time = rospy.Time.now().to_sec()
            
            for i in range(steps):
                if rospy.is_shutdown():
                    return
                t = i / float(self.rate_hz)
                pos = mid_pos + amplitude * math.sin(2 * math.pi * freq * t)
                self.publish_command([joint_name], [pos])
                self.record_data("sinusoid", joint_name, pos, test_param)
                self.rate.sleep()
            
            self.is_recording = False
            
            # Brief pause between frequencies
            rospy.sleep(0.5)

    def run_ramp_test(self, joint_name):
        """Run ramp input test on a single joint (reveals stiction)."""
        lower, upper, default, is_prismatic = LEG_JOINTS[joint_name]
        rospy.loginfo("[RAMP] Testing %s", joint_name)
        
        # Calculate safe ramp range within limits
        mid_pos = (lower + upper) / 2.0
        range_available = (upper - lower) / 2.0
        ramp_half_range = min(self.ramp_range / 2.0, range_available * 0.4)
        
        pos_low = mid_pos - ramp_half_range
        pos_high = mid_pos + ramp_half_range
        
        ramp_duration = (2 * ramp_half_range) / self.ramp_velocity
        test_param = "vel=%.3f,range=%.3f" % (self.ramp_velocity, 2 * ramp_half_range)
        
        rospy.loginfo("  Ramp from %.3f to %.3f at %.3f rad/s", pos_low, pos_high, self.ramp_velocity)
        
        # Go to starting position
        self.go_to_position(joint_name, pos_low, duration=2.0)
        rospy.sleep(0.5)
        
        self.is_recording = True
        self.start_time = rospy.Time.now().to_sec()
        
        # Ramp up
        steps = int(ramp_duration * self.rate_hz)
        for i in range(steps):
            if rospy.is_shutdown():
                return
            t = i / float(self.rate_hz)
            pos = pos_low + self.ramp_velocity * t
            self.publish_command([joint_name], [pos])
            self.record_data("ramp_up", joint_name, pos, test_param)
            self.rate.sleep()
        
        # Brief hold at top
        for _ in range(int(0.5 * self.rate_hz)):
            if rospy.is_shutdown():
                return
            self.publish_command([joint_name], [pos_high])
            self.record_data("ramp_hold", joint_name, pos_high, test_param)
            self.rate.sleep()
        
        # Ramp down
        for i in range(steps):
            if rospy.is_shutdown():
                return
            t = i / float(self.rate_hz)
            pos = pos_high - self.ramp_velocity * t
            self.publish_command([joint_name], [pos])
            self.record_data("ramp_down", joint_name, pos, test_param)
            self.rate.sleep()
        
        self.is_recording = False

    def save_data(self, filename_prefix="sysid"):
        """Save collected data to CSV."""
        if not self.data_records:
            rospy.logwarn("No data to save!")
            return None
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, "%s_%s.csv" % (filename_prefix, timestamp_str))
        
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_s",
                "test_type",
                "joint_name",
                "cmd_position",
                "meas_position",
                "meas_velocity",
                "meas_effort",
                "test_param",
            ])
            
            for record in self.data_records:
                writer.writerow([
                    "%.6f" % record["timestamp"],
                    record["test_type"],
                    record["joint_name"],
                    "%.6f" % record["cmd_position"],
                    "%.6f" % record["meas_position"],
                    "%.6f" % record["meas_velocity"],
                    "%.6f" % record["meas_effort"],
                    record["test_param"],
                ])
        
        rospy.loginfo("Saved %d records to %s", len(self.data_records), filename)
        return filename

    def run_tests(self):
        """Run all configured tests."""
        # Wait for joint states
        rospy.loginfo("Waiting for joint states...")
        timeout = rospy.Time.now() + rospy.Duration(5.0)
        while not self.current_joint_states and rospy.Time.now() < timeout:
            rospy.sleep(0.1)
        
        if not self.current_joint_states:
            rospy.logerr("No joint states received! Is the robot running?")
            return
        
        rospy.loginfo("Received joint states for %d joints", len(self.current_joint_states))
        
        # Determine which joints to test
        if self.test_joint == "all":
            joints_to_test = list(LEG_JOINTS.keys())
        elif self.test_joint in LEG_JOINTS:
            joints_to_test = [self.test_joint]
        else:
            rospy.logerr("Unknown joint: %s", self.test_joint)
            rospy.loginfo("Available joints: %s", list(LEG_JOINTS.keys()))
            return
        
        # Determine which tests to run
        valid_tests = ["step", "sinusoid", "ramp", "all"]
        if self.test_type not in valid_tests:
            rospy.logerr("Unknown test type: %s", self.test_type)
            rospy.loginfo("Available types: %s", valid_tests)
            return
        
        if self.test_type == "all":
            tests_to_run = ["step", "sinusoid", "ramp"]
        else:
            tests_to_run = [self.test_type]
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Starting System Identification Tests")
        rospy.loginfo("Joints: %s", joints_to_test)
        rospy.loginfo("Tests: %s", tests_to_run)
        rospy.loginfo("=" * 50)
        
        # Run tests
        for joint_name in joints_to_test:
            if rospy.is_shutdown():
                break
            
            rospy.loginfo("\n>>> Testing joint: %s", joint_name)
            
            for test_type in tests_to_run:
                if rospy.is_shutdown():
                    break
                
                if test_type == "step":
                    self.run_step_test(joint_name)
                elif test_type == "sinusoid":
                    self.run_sinusoid_test(joint_name)
                elif test_type == "ramp":
                    self.run_ramp_test(joint_name)
                
                rospy.sleep(1.0)
        
        # Return to safe position
        rospy.loginfo("Returning to default positions...")
        for joint_name in joints_to_test:
            lower, upper, default, is_prismatic = LEG_JOINTS[joint_name]
            self.go_to_position(joint_name, default, duration=2.0)
        
        # Save data
        filename = self.save_data()
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("System Identification Tests Complete!")
        if filename:
            rospy.loginfo("Data saved to: %s", filename)
        rospy.loginfo("=" * 50)


def main():
    try:
        collector = SysIdDataCollector()
        
        # Small delay to ensure publishers/subscribers are connected
        rospy.sleep(1.0)
        
        collector.run_tests()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr("Error: %s", str(e))
        raise


if __name__ == "__main__":
    main()