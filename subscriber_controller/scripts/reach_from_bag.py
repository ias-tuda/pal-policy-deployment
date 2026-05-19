#!/usr/bin/env python3
"""
Read a target arm configuration from a rosbag and drive there via
subscriber_controller using linear interpolation (velocity clamped, 50 Hz).

Reads one snapshot of /subscriber_controller/desired_state (or any JointState
topic) from the bag at a given time offset, then interpolates from the current
actual position to that target.

Usage:
  rosrun subscriber_controller reach_from_bag.py \
    _bag:=/path/to/bag.bag \
    _bag_time:=5.0 \
    _max_joint_vel:=0.3 \
    _hold_duration:=10.0

Parameters:
  _bag            Path to rosbag file
  _bag_time       Time in seconds from bag start to sample target (default: last message)
  _topic          JointState topic to read from bag (default: /subscriber_controller/desired_state)
  _max_joint_vel  Velocity limit in rad/s (default: 0.3 — conservative)
  _rate_hz        Control rate (default: 50.0)
  _hold_duration  Seconds to hold at target after reaching it (0 = hold forever)
"""

import rospy
import rosbag
from sensor_msgs.msg import JointState

ARM_JOINTS = [
    "arm_left_1_joint",  "arm_left_2_joint",  "arm_left_3_joint",
    "arm_left_4_joint",  "arm_left_5_joint",  "arm_left_6_joint",
    "arm_left_7_joint",
    "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
    "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint",
    "arm_right_7_joint",
]


def read_target_from_bag(bag_path, topic, bag_time_offset):
    """Return dict {joint_name: position} sampled at bag_time_offset seconds."""
    last_msg   = None
    start_time = None
    best_msg   = None
    best_dt    = float("inf")

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            if start_time is None:
                start_time = t
            elapsed = (t - start_time).to_sec()
            last_msg = msg
            if bag_time_offset is not None:
                dt = abs(elapsed - bag_time_offset)
                if dt < best_dt:
                    best_dt  = dt
                    best_msg = msg

    chosen = best_msg if bag_time_offset is not None else last_msg
    if chosen is None:
        return None

    result = {}
    for name, pos in zip(chosen.name, chosen.position):
        if name in ARM_JOINTS:
            result[name] = pos
    return result


def main():
    rospy.init_node("reach_from_bag")

    bag_path      = rospy.get_param("~bag",           "")
    bag_time_raw  = rospy.get_param("~bag_time",      None)
    topic         = rospy.get_param("~topic",         "/subscriber_controller/desired_state")
    max_joint_vel = float(rospy.get_param("~max_joint_vel", 0.3))
    rate_hz       = float(rospy.get_param("~rate_hz",       50.0))
    hold_duration = float(rospy.get_param("~hold_duration", 0.0))

    if not bag_path:
        rospy.logfatal("[reach_from_bag] _bag parameter required")
        return

    bag_time = float(bag_time_raw) if bag_time_raw is not None else None

    # ── Read target from bag ──────────────────────────────────────
    rospy.loginfo("[reach_from_bag] Reading bag: %s", bag_path)
    rospy.loginfo("[reach_from_bag] Topic: %s  |  bag_time: %s",
                  topic, "last" if bag_time is None else "%.2f s" % bag_time)

    target = read_target_from_bag(bag_path, topic, bag_time)
    if not target:
        rospy.logfatal("[reach_from_bag] No arm joints found in bag on topic '%s'", topic)
        return

    rospy.loginfo("[reach_from_bag] Target positions from bag:")
    for j in ARM_JOINTS:
        if j in target:
            rospy.loginfo("  %-35s %+.4f", j, target[j])

    # ── Read current actual positions ─────────────────────────────
    rospy.loginfo("[reach_from_bag] Waiting for /joint_states...")
    js = rospy.wait_for_message("/joint_states", JointState)

    current = {}
    for name, pos in zip(js.name, js.position):
        if name in ARM_JOINTS:
            current[name] = pos

    joints = [j for j in ARM_JOINTS if j in target and j in current]

    rospy.logwarn("[reach_from_bag] === PLAN (max_joint_vel=%.2f rad/s) ===", max_joint_vel)
    rospy.logwarn("[reach_from_bag]   %-35s  %10s  %10s  %10s  %8s",
                  "Joint", "current", "target", "delta", "est. time")
    max_time = 0.0
    for j in joints:
        delta = target[j] - current[j]
        est   = abs(delta) / max_joint_vel
        max_time = max(max_time, est)
        rospy.logwarn("[reach_from_bag]   %-35s  %+10.4f  %+10.4f  %+10.4f  %6.2f s",
                      j, current[j], target[j], delta, est)
    rospy.logwarn("[reach_from_bag] Estimated travel time: %.2f s", max_time)
    rospy.logwarn("[reach_from_bag] Press Ctrl-C to abort. Starting in 2 s...")
    rospy.sleep(2.0)

    # ── Interpolate to target ─────────────────────────────────────
    pub      = rospy.Publisher("/subscriber_controller/desired_state",
                               JointState, queue_size=1)
    rate     = rospy.Rate(rate_hz)
    dt       = 1.0 / rate_hz
    max_d    = max_joint_vel * dt

    cmd = {j: current[j] for j in joints}

    rospy.logwarn("[reach_from_bag] Moving to target...")
    while not rospy.is_shutdown():
        reached = True
        for j in joints:
            delta = target[j] - cmd[j]
            if abs(delta) > max_d:
                cmd[j] += max_d * (1.0 if delta > 0 else -1.0)
                reached = False
            else:
                cmd[j] = target[j]

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name     = joints
        msg.position = [cmd[j] for j in joints]
        msg.velocity = [0.0] * len(joints)
        msg.effort   = [0.0] * len(joints)
        pub.publish(msg)

        if reached:
            break
        rate.sleep()

    # ── Hold at target ────────────────────────────────────────────
    rospy.logwarn("[reach_from_bag] Target reached. Holding...")
    rospy.logwarn("[reach_from_bag]   %-35s  %10s  %10s", "Joint", "target", "actual_now")

    js2 = rospy.wait_for_message("/joint_states", JointState)
    actual_now = {n: p for n, p in zip(js2.name, js2.position)}
    for j in joints:
        err = actual_now.get(j, 0.0) - target[j]
        rospy.logwarn("[reach_from_bag]   %-35s  %+10.4f  %+10.4f  err=%+.4f%s",
                      j, target[j], actual_now.get(j, 0.0), err,
                      "  !!!" if abs(err) > 0.05 else "")

    hold_msg = JointState()
    hold_msg.name     = joints
    hold_msg.position = [target[j] for j in joints]
    hold_msg.velocity = [0.0] * len(joints)
    hold_msg.effort   = [0.0] * len(joints)

    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        hold_msg.header.stamp = rospy.Time.now()
        pub.publish(hold_msg)
        elapsed = (rospy.Time.now() - t0).to_sec()
        if hold_duration > 0.0 and elapsed >= hold_duration:
            rospy.loginfo("[reach_from_bag] Done after %.1f s.", hold_duration)
            break
        rate.sleep()


if __name__ == "__main__":
    main()
