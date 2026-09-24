#!/usr/bin/env python3
"""
Move both Kangaroo arms from the current JointState to a target configuration
by publishing a smooth JointState stream to subscriber_controller/desired_state.

Example:
  rosrun subscriber_controller publish_arm_target_as_desired.py \
    _left:="[0, 0.75, 1.75, 1.45, -0.75, 0, 0]" \
    _right:="[0, 0.75, -1.75, 1.45, 0.75, 0, 0]" \
    _max_joint_vel:=0.5 \
    _hold_duration:=10.0
"""

import ast

import rospy
from sensor_msgs.msg import JointState


LEFT_ARM = [
    "arm_left_1_joint",
    "arm_left_2_joint",
    "arm_left_3_joint",
    "arm_left_4_joint",
    "arm_left_5_joint",
    "arm_left_6_joint",
    "arm_left_7_joint",
]

RIGHT_ARM = [
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "arm_right_5_joint",
    "arm_right_6_joint",
    "arm_right_7_joint",
]


def parse_target(param_name, default):
    raw = rospy.get_param("~" + param_name, default)
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    values = [float(v) for v in raw]
    if len(values) != 7:
        raise ValueError("_{} must contain exactly 7 values".format(param_name))
    return values


def publish_command(pub, joints, positions):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = list(joints)
    msg.position = [positions[j] for j in joints]
    msg.velocity = [0.0] * len(joints)
    msg.effort = [0.0] * len(joints)
    pub.publish(msg)


def log_impedance_gains(controller_name, joints):
    rospy.logwarn("[publish_arm_target] Loaded impedance gains from /%s:", controller_name)
    for joint in joints:
        stiffness_param = "/{}/joints/{}/impedance_params/stiffness_factor".format(
            controller_name, joint
        )
        damping_param = "/{}/joints/{}/impedance_params/damping_factor".format(
            controller_name, joint
        )

        if not rospy.has_param(stiffness_param) or not rospy.has_param(damping_param):
            rospy.logwarn("[publish_arm_target]   %-35s K=<missing> D=<missing>", joint)
            continue

        stiffness = rospy.get_param(stiffness_param)
        damping = rospy.get_param(damping_param)
        rospy.logwarn("[publish_arm_target]   %-35s K=%s D=%s", joint, stiffness, damping)


def log_final_error(source_topic, joints, target):
    rospy.logwarn("[publish_arm_target] Reading final %s snapshot for error report.",
                  source_topic)
    try:
        js = rospy.wait_for_message(source_topic, JointState, timeout=2.0)
    except rospy.ROSException as exc:
        rospy.logwarn("[publish_arm_target] Could not read final %s: %s",
                      source_topic, exc)
        return

    actual = {name: pos for name, pos in zip(js.name, js.position)}
    rospy.logwarn("[publish_arm_target] Final joint error: target - actual")
    max_abs_error = 0.0
    for joint in joints:
        if joint not in actual:
            rospy.logwarn("[publish_arm_target]   %-35s missing in %s",
                          joint, source_topic)
            continue

        error = target[joint] - actual[joint]
        max_abs_error = max(max_abs_error, abs(error))
        rospy.logwarn("[publish_arm_target]   %-35s target=%+.4f actual=%+.4f error=%+.4f",
                      joint, target[joint], actual[joint], error)

    rospy.logwarn("[publish_arm_target]   max_abs_error=%+.4f rad", max_abs_error)


def main():
    rospy.init_node("publish_arm_target_as_desired")

    source_topic = rospy.get_param("~source_topic", "/joint_states")
    target_topic = rospy.get_param("~target_topic", "/subscriber_controller/desired_state")
    rate_hz = float(rospy.get_param("~rate_hz", 50.0))
    max_joint_vel = float(rospy.get_param("~max_joint_vel", 0.5))
    hold_duration = float(rospy.get_param("~hold_duration", 0.0))
    final_error_after = float(rospy.get_param("~final_error_after", 2.0))
    start_delay = float(rospy.get_param("~start_delay", 2.0))
    controller_name = rospy.get_param("~controller_name", "subscriber_controller")
    log_gains = bool(rospy.get_param("~log_gains", True))

    left_target = parse_target("left", [0.0, 0.75, 1.75, 1.45, -0.75, 0.0, 0.0])
    right_target = parse_target("right", [0.0, 0.75, -1.75, 1.45, 0.75, 0.0, 0.0])

    joints = LEFT_ARM + RIGHT_ARM
    target = dict(zip(LEFT_ARM, left_target))
    target.update(dict(zip(RIGHT_ARM, right_target)))

    if log_gains:
        log_impedance_gains(controller_name, joints)

    rospy.loginfo("[publish_arm_target] Waiting for %s ...", source_topic)
    js = rospy.wait_for_message(source_topic, JointState)
    actual = {name: pos for name, pos in zip(js.name, js.position)}

    missing = [j for j in joints if j not in actual]
    if missing:
        rospy.logfatal("[publish_arm_target] Missing joints in %s: %s", source_topic, missing)
        return

    current = {j: actual[j] for j in joints}
    command = dict(current)
    max_delta = max(abs(target[j] - current[j]) for j in joints)
    estimated_time = max_delta / max_joint_vel if max_joint_vel > 0.0 else 0.0

    rospy.logwarn("[publish_arm_target] Target plan:")
    rospy.logwarn("[publish_arm_target]   max_joint_vel=%.3f rad/s estimated_time=%.2f s",
                  max_joint_vel, estimated_time)
    for j in joints:
        rospy.logwarn("[publish_arm_target]   %-35s current=%+.4f target=%+.4f delta=%+.4f",
                      j, current[j], target[j], target[j] - current[j])

    pub = rospy.Publisher(target_topic, JointState, queue_size=1)
    rate = rospy.Rate(rate_hz)

    rospy.logwarn("[publish_arm_target] Holding current arm pose for %.2f s before moving.",
                  start_delay)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < start_delay:
        publish_command(pub, joints, command)
        rate.sleep()

    rospy.logwarn("[publish_arm_target] Moving to target.")
    step = max_joint_vel / rate_hz
    while not rospy.is_shutdown():
        reached = True
        for j in joints:
            delta = target[j] - command[j]
            if abs(delta) > step:
                command[j] += step if delta > 0.0 else -step
                reached = False
            else:
                command[j] = target[j]

        publish_command(pub, joints, command)

        if reached:
            break
        rate.sleep()

    rospy.logwarn("[publish_arm_target] Target reached. Holding.")
    hold_start = rospy.Time.now()
    final_error_logged = False
    while not rospy.is_shutdown():
        publish_command(pub, joints, target)
        hold_elapsed = (rospy.Time.now() - hold_start).to_sec()
        if (not final_error_logged) and final_error_after >= 0.0 and hold_elapsed >= final_error_after:
            log_final_error(source_topic, joints, target)
            final_error_logged = True
        if hold_duration > 0.0 and (rospy.Time.now() - hold_start).to_sec() >= hold_duration:
            rospy.loginfo("[publish_arm_target] Done after hold_duration=%.2f s.", hold_duration)
            break
        rate.sleep()

    if not final_error_logged:
        log_final_error(source_topic, joints, target)


if __name__ == "__main__":
    main()
