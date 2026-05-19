#!/usr/bin/env python3
"""
Sine sweep test for arm joints via subscriber_controller.

Holds current arm positions and adds a sinusoidal offset to all joints
of one arm (or both). Useful for diagnosing oscillations in impedance gains.

Examples:
  rosrun subscriber_controller sine_sweep_arms.py _side:=left
  rosrun subscriber_controller sine_sweep_arms.py _side:=right _amplitude:=0.05 _frequency:=1.0
  rosrun subscriber_controller sine_sweep_arms.py _side:=both _amplitude:=0.03 _frequency:=2.0 _duration:=10.0
"""

import math
import copy
import rospy
from sensor_msgs.msg import JointState

ARM_JOINTS = {
    "left":  ["arm_left_1_joint",  "arm_left_2_joint",  "arm_left_3_joint",
               "arm_left_4_joint",  "arm_left_5_joint",  "arm_left_6_joint",
               "arm_left_7_joint"],
    "right": ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
               "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint",
               "arm_right_7_joint"],
}


def main():
    rospy.init_node("sine_sweep_arms")

    side      = rospy.get_param("~side",      "left")   # left | right | both
    amplitude = float(rospy.get_param("~amplitude",  0.05))  # rad
    frequency = float(rospy.get_param("~frequency",  1.0))   # Hz
    rate_hz   = float(rospy.get_param("~rate_hz",   50.0))
    duration  = float(rospy.get_param("~duration",   0.0))   # 0 = run forever

    if side not in ("left", "right", "both"):
        rospy.logfatal("_side must be: left | right | both")
        return

    active_joints = ARM_JOINTS["left"] + ARM_JOINTS["right"] if side == "both" \
                    else ARM_JOINTS[side]
    sweep_joints  = set(active_joints)

    rospy.loginfo("[sine_sweep] Waiting for /joint_states...")
    js = rospy.wait_for_message("/joint_states", JointState)

    # Build desired state: all arm joints at current position
    all_arm = ARM_JOINTS["left"] + ARM_JOINTS["right"]
    center = {}
    for name, pos in zip(js.name, js.position):
        if name in all_arm:
            center[name] = pos

    missing = [j for j in all_arm if j not in center]
    if missing:
        rospy.logwarn("[sine_sweep] Joints not found in /joint_states: %s", missing)

    rospy.loginfo("[sine_sweep] side=%s  amplitude=%.3f rad  frequency=%.2f Hz", side, amplitude, frequency)
    rospy.loginfo("[sine_sweep] Center positions:")
    for j in all_arm:
        rospy.loginfo("  %-35s %+.4f %s", j, center.get(j, 0.0),
                      "<-- sine" if j in sweep_joints else "")

    pub = rospy.Publisher("/subscriber_controller/desired_state", JointState,
                          queue_size=1)
    rate  = rospy.Rate(rate_hz)
    t0    = rospy.Time.now()

    rospy.logwarn("[sine_sweep] Publishing. Ctrl-C to stop.")

    while not rospy.is_shutdown():
        t = (rospy.Time.now() - t0).to_sec()
        offset = amplitude * math.sin(2.0 * math.pi * frequency * t)

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        for j in all_arm:
            if j not in center:
                continue
            msg.name.append(j)
            msg.position.append(center[j] + (offset if j in sweep_joints else 0.0))
            msg.velocity.append(0.0)
            msg.effort.append(0.0)

        pub.publish(msg)

        rospy.loginfo_throttle(1.0,
            "[sine_sweep] t=%.2f s  offset=%+.4f rad", t, offset)

        if duration > 0.0 and t >= duration:
            rospy.loginfo("[sine_sweep] Done after %.1f s.", duration)
            break

        rate.sleep()


if __name__ == "__main__":
    main()
