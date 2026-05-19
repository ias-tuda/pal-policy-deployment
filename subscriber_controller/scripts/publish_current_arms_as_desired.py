#!/usr/bin/env python
"""
Publish the current left and right arm positions as subscriber_controller desired_state.

Use this when the subscriber_controller only claims the arm joints.

Examples:
  python src/subscriber_controller/scripts/publish_current_arms_as_desired.py
  python src/subscriber_controller/scripts/publish_current_arms_as_desired.py _delay:=3 _joint:=arm_left_6_joint _position:=0.003
  python src/subscriber_controller/scripts/publish_current_arms_as_desired.py _delay:=3 _joint:=arm_left_6_joint _delta:=0.01
"""

import copy

import rospy
from sensor_msgs.msg import JointState


ARM_PREFIXES = ("arm_left_", "arm_right_")


def main():
    rospy.init_node("publish_current_arms_as_desired")

    source_topic = rospy.get_param("~source_topic", "/joint_states")
    target_topic = rospy.get_param("~target_topic", "/subscriber_controller/desired_state")
    rate_hz = float(rospy.get_param("~rate_hz", 50.0))
    duration = float(rospy.get_param("~duration", 0.0))
    delay = float(rospy.get_param("~delay", 0.0))
    override_joint = rospy.get_param("~joint", "")
    override_position = rospy.get_param("~position", None)
    override_delta = rospy.get_param("~delta", None)

    if override_position is not None and override_delta is not None:
        rospy.logerr("Use either _position:=... or _delta:=..., not both.")
        return

    rospy.loginfo("[publish_current_arms_as_desired] Waiting for %s ...", source_topic)
    current = rospy.wait_for_message(source_topic, JointState)

    desired = JointState()
    for i, name in enumerate(current.name):
        if name.startswith(ARM_PREFIXES):
            desired.name.append(name)
            desired.position.append(current.position[i])
            desired.velocity.append(0.0)
            if len(current.effort) == len(current.name):
                desired.effort.append(current.effort[i])
            else:
                desired.effort.append(0.0)

    if not desired.name:
        rospy.logerr("No arm joints found in %s.", source_topic)
        return

    override_idx = None
    step_position = None
    if override_joint:
        if override_joint not in desired.name:
            rospy.logerr("Joint '%s' is not an arm joint in %s.", override_joint, source_topic)
            rospy.logerr("Available arm joints: %s", ", ".join(desired.name))
            return

        override_idx = desired.name.index(override_joint)
        initial_position = desired.position[override_idx]
        if override_position is not None:
            step_position = float(override_position)
        elif override_delta is not None:
            step_position = initial_position + float(override_delta)
        else:
            rospy.logwarn("_joint was set, but no _position or _delta was given. No step will be applied.")

    pub = rospy.Publisher(target_topic, JointState, queue_size=1)
    rate = rospy.Rate(rate_hz)
    start_time = rospy.Time.now()
    count = 0
    step_applied = False

    rospy.loginfo(
        "[publish_current_arms_as_desired] Publishing %d arm joints to %s at %.1f Hz",
        len(desired.name),
        target_topic,
        rate_hz,
    )
    for name, pos in zip(desired.name, desired.position):
        rospy.loginfo("  %-35s pos=%+.9f", name, pos)
    if override_idx is not None and step_position is not None:
        rospy.loginfo(
            "[publish_current_arms_as_desired] Step after %.3f s: %s %.9f -> %.9f",
            delay,
            override_joint,
            desired.position[override_idx],
            step_position,
        )

    while not rospy.is_shutdown():
        elapsed = (rospy.Time.now() - start_time).to_sec()
        if (
            not step_applied
            and override_idx is not None
            and step_position is not None
            and elapsed >= delay
        ):
            old_position = desired.position[override_idx]
            desired.position[override_idx] = step_position
            step_applied = True
            rospy.logwarn(
                "[publish_current_arms_as_desired] STEP applied: %s %.9f -> %.9f at t=%.3f s",
                override_joint,
                old_position,
                step_position,
                elapsed,
            )

        msg = copy.deepcopy(desired)
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        count += 1

        rospy.loginfo_throttle(
            2.0,
            "[publish_current_arms_as_desired] running | messages=%d | joints=%d",
            count,
            len(msg.name),
        )

        if duration > 0.0 and elapsed >= duration:
            rospy.loginfo("[publish_current_arms_as_desired] Finished after %.3f seconds.", duration)
            break

        rate.sleep()


if __name__ == "__main__":
    main()
