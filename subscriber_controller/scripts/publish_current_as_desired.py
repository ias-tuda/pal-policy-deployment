#!/usr/bin/env python3
"""
Freeze the current JointState and publish it as subscriber_controller desired_state.

Usage examples:
  rosrun subscriber_controller publish_current_as_desired.py
  rosrun subscriber_controller publish_current_as_desired.py _joint:=arm_left_6_joint _position:=0.003
  rosrun subscriber_controller publish_current_as_desired.py _joint:=arm_left_6_joint _delta:=0.001

The script captures one message from source_topic, then publishes the same joint
names and positions at rate_hz. Optionally one joint can be overwritten.
"""

import copy

import rospy
from sensor_msgs.msg import JointState


def get_private_param(name, default):
    return rospy.get_param("~" + name, default)


def main():
    rospy.init_node("publish_current_as_desired")

    source_topic = get_private_param("source_topic", "/joint_states")
    target_topic = get_private_param("target_topic", "/subscriber_controller/desired_state")
    rate_hz = float(get_private_param("rate_hz", 50.0))
    duration = float(get_private_param("duration", 0.0))
    override_joint = get_private_param("joint", "")
    override_position = get_private_param("position", None)
    override_delta = get_private_param("delta", None)
    effort_mode = get_private_param("effort_mode", "copy")

    if override_position is not None and override_delta is not None:
        rospy.logerr("Use either _position:=... or _delta:=..., not both.")
        return

    rospy.loginfo("[publish_current_as_desired] Waiting for %s ...", source_topic)
    current = rospy.wait_for_message(source_topic, JointState)

    desired = JointState()
    desired.name = list(current.name)
    desired.position = list(current.position)
    desired.velocity = [0.0] * len(desired.name)

    if effort_mode == "copy" and len(current.effort) == len(current.name):
        desired.effort = list(current.effort)
    else:
        desired.effort = [0.0] * len(desired.name)

    if override_joint:
        if override_joint not in desired.name:
            rospy.logerr("Joint '%s' is not present in %s.", override_joint, source_topic)
            rospy.logerr("Available joints: %s", ", ".join(desired.name))
            return

        idx = desired.name.index(override_joint)
        old_position = desired.position[idx]
        if override_position is not None:
            desired.position[idx] = float(override_position)
        elif override_delta is not None:
            desired.position[idx] = old_position + float(override_delta)
        else:
            rospy.logwarn("_joint was set, but no _position or _delta was given. Keeping current position.")

        rospy.loginfo(
            "[publish_current_as_desired] Override %s: %.9f -> %.9f",
            override_joint,
            old_position,
            desired.position[idx],
        )

    pub = rospy.Publisher(target_topic, JointState, queue_size=1)
    rate = rospy.Rate(rate_hz)
    start_time = rospy.Time.now()
    count = 0

    rospy.loginfo(
        "[publish_current_as_desired] Publishing %d joints to %s at %.1f Hz",
        len(desired.name),
        target_topic,
        rate_hz,
    )
    rospy.loginfo("[publish_current_as_desired] effort_mode=%s", effort_mode)

    while not rospy.is_shutdown():
        msg = copy.deepcopy(desired)
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        count += 1

        rospy.loginfo_throttle(
            2.0,
            "[publish_current_as_desired] running | messages=%d | joints=%d",
            count,
            len(msg.name),
        )

        if duration > 0.0 and (rospy.Time.now() - start_time).to_sec() >= duration:
            rospy.loginfo("[publish_current_as_desired] Finished after %.3f seconds.", duration)
            break

        rate.sleep()


if __name__ == "__main__":
    main()
