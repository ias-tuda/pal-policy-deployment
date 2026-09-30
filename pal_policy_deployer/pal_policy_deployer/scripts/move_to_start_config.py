#!/usr/bin/env python3
"""Move selected controlled joints to a deployment_start_config.json pose."""

import json
import math
import threading
import time

import rospy
from sensor_msgs.msg import JointState


DEFAULT_JOINTS = ["arm_left_{}_joint".format(i) for i in range(1, 8)] + ["gripper_left_finger_joint"]


def load_targets(path, joint_names):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported start config schema_version")
    listed_names = config["joint_names"]
    if len(listed_names) != len(set(listed_names)):
        raise ValueError("duplicate names in start config")
    if not joint_names or len(joint_names) != len(set(joint_names)):
        raise ValueError("~joint_names must contain unique joints")
    positions = config["joint_positions"]
    velocities = config["joint_velocities"]
    targets = {}
    for name in joint_names:
        if name not in listed_names or name not in positions or name not in velocities:
            raise ValueError("start config is missing {}".format(name))
        position = float(positions[name])
        velocity = float(velocities[name])
        if not math.isfinite(position) or not math.isfinite(velocity):
            raise ValueError("non-finite start state for {}".format(name))
        if abs(velocity) > 1e-6:
            raise ValueError("start velocity for {} must be zero for a stationary pose".format(name))
        targets[name] = position
    return targets


class MeasuredState:
    def __init__(self, topic):
        self._lock = threading.Lock()
        self._positions = {}
        self._velocities = {}
        self._receipt_time = 0.0
        self._subscriber = rospy.Subscriber(topic, JointState, self._callback, queue_size=1)

    def _callback(self, message):
        if len(message.name) != len(message.position) or len(message.name) != len(message.velocity):
            return
        positions = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity))
        if not all(math.isfinite(value) for value in list(positions.values()) + list(velocities.values())):
            return
        with self._lock:
            self._positions = positions
            self._velocities = velocities
            self._receipt_time = time.monotonic()

    def read(self, joint_names, max_age):
        with self._lock:
            if time.monotonic() - self._receipt_time > max_age:
                raise RuntimeError("measured joint state is missing or stale")
            missing = [name for name in joint_names if name not in self._positions]
            if missing:
                raise RuntimeError("measured joint state is missing: {}".format(", ".join(missing)))
            return ({name: self._positions[name] for name in joint_names},
                    {name: self._velocities[name] for name in joint_names})


def publish_command(publisher, joint_names, positions, velocities):
    command = JointState()
    command.header.stamp = rospy.Time.now()
    command.name = joint_names
    command.position = [positions[name] for name in joint_names]
    command.velocity = [velocities[name] for name in joint_names]
    command.effort = [0.0] * len(joint_names)
    publisher.publish(command)


def positive_param(name, default):
    value = float(rospy.get_param("~" + name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("~{} must be positive and finite".format(name))
    return value


def ensure_exclusive_publisher(topic):
    code, message, system_state = rospy.get_master().getSystemState()
    if code != 1:
        raise RuntimeError("cannot inspect ROS publishers: {}".format(message))
    publishers = dict(system_state[0]).get(topic, [])
    others = [name for name in publishers if name != rospy.get_name()]
    if others:
        raise RuntimeError("other publishers on {}: {}".format(topic, ", ".join(others)))


def move():
    rospy.init_node("move_to_start_config")
    path = rospy.get_param("~start_config_path", "")
    if not path:
        raise ValueError("~start_config_path is required")
    joint_names = rospy.get_param("~joint_names", DEFAULT_JOINTS)
    if not isinstance(joint_names, list) or not all(isinstance(name, str) for name in joint_names):
        raise ValueError("~joint_names must be a list of joint names")
    targets = load_targets(path, joint_names)
    # Commanded but not required to settle, e.g. the jaws, which run their own position loop.
    unverified = rospy.get_param("~unverified_joint_names",
                                 [name for name in joint_names if name.startswith("gripper_")])
    if not isinstance(unverified, list) or not set(unverified) <= set(joint_names):
        raise ValueError("~unverified_joint_names must be a subset of ~joint_names")
    verified_names = [name for name in joint_names if name not in unverified]
    if not verified_names:
        raise ValueError("at least one joint must be verified")
    # Added to the commanded position only: without a gravity bridge, the PD error that holds
    # the start pose against gravity (the task model target_preload). Disabled when a
    # gravity bridge adds that torque instead.
    preload = [0.0] * len(joint_names)
    if rospy.get_param("~use_target_preload", True):
        preload = rospy.get_param("~target_preload", preload)
    if len(preload) != len(joint_names) or not all(math.isfinite(float(v)) for v in preload):
        raise ValueError("~target_preload must have one finite value per joint")
    preload = {name: float(value) for name, value in zip(joint_names, preload)}
    commanded = {name: targets[name] + preload[name] for name in joint_names}

    state_topic = rospy.get_param("~state_topic", "/subscriber_controller/actual_js_state")
    command_topic = rospy.get_param("~command_topic", "/pal_grasp/desired_state_uncompensated")
    rate_hz = positive_param("rate_hz", 50.0)
    max_speed = positive_param("max_speed", 0.15)
    min_duration = positive_param("min_duration", 2.0)
    tolerance = positive_param("position_tolerance", 0.05)
    velocity_tolerance = positive_param("velocity_tolerance", 0.05)
    state_timeout = positive_param("state_timeout", 0.25)
    settle_time = positive_param("settle_time", 0.5)
    settle_timeout = positive_param("settle_timeout", 15.0)
    tracking_error_limit = positive_param("tracking_error_limit", 0.5)
    startup_timeout = positive_param("startup_timeout", 15.0)

    measured = MeasuredState(state_topic)
    publisher = rospy.Publisher(command_topic, JointState, queue_size=1)
    rospy.loginfo("Waiting for controlled joint state on %s", state_topic)
    startup_deadline = time.monotonic() + startup_timeout
    while not rospy.is_shutdown() and time.monotonic() < startup_deadline:
        try:
            start, _ = measured.read(joint_names, state_timeout)
            break
        except RuntimeError:
            rospy.sleep(0.1)
    if rospy.is_shutdown():
        return
    if time.monotonic() >= startup_deadline:
        raise RuntimeError("timed out waiting for controlled joint state")
    while not rospy.is_shutdown() and publisher.get_num_connections() == 0 and time.monotonic() < startup_deadline:
        rospy.loginfo_throttle(2.0, "Waiting for a controller subscriber on %s", command_topic)
        rospy.sleep(0.1)
    if rospy.is_shutdown():
        return
    if publisher.get_num_connections() == 0:
        raise RuntimeError("timed out waiting for controller subscriber")
    ensure_exclusive_publisher(command_topic)

    max_delta = max(abs(targets[name] - start[name]) for name in joint_names)
    duration = max(min_duration, 1.5 * max_delta / max_speed)
    rospy.logwarn("Moving %d joints to %s over %.2f s; max change %.3f",
                  len(joint_names), path, duration, max_delta)
    for name in joint_names:
        rospy.loginfo("%s: %.4f -> %.4f", name, start[name], targets[name])
    rate = rospy.Rate(rate_hz)
    start_time = time.monotonic()
    last_publisher_check = start_time
    while not rospy.is_shutdown():
        elapsed = time.monotonic() - start_time
        u = min(1.0, elapsed / duration)
        blend = u * u * (3.0 - 2.0 * u)
        blend_speed = 6.0 * u * (1.0 - u) / duration
        expected = {name: start[name] + blend * (targets[name] - start[name])
                    for name in joint_names}
        # The jaws get their goal directly; their own position loop does the motion.
        positions = {name: commanded[name] if name in unverified else
                     start[name] + blend * (commanded[name] - start[name])
                     for name in joint_names}
        velocities = {name: 0.0 if name in unverified else
                      blend_speed * (commanded[name] - start[name])
                      for name in joint_names}
        actual, _ = measured.read(joint_names, state_timeout)
        worst_joint = max(verified_names, key=lambda name: abs(expected[name] - actual[name]))
        worst_tracking_error = abs(expected[worst_joint] - actual[worst_joint])
        if worst_tracking_error > tracking_error_limit:
            raise RuntimeError("{} tracking error {:.3f} exceeds {:.3f}".format(
                worst_joint, worst_tracking_error, tracking_error_limit))
        if publisher.get_num_connections() == 0:
            raise RuntimeError("controller subscriber disconnected")
        if time.monotonic() - last_publisher_check >= 1.0:
            ensure_exclusive_publisher(command_topic)
            last_publisher_check = time.monotonic()
        publish_command(publisher, joint_names, positions, velocities)
        if u >= 1.0:
            break
        rate.sleep()

    rospy.loginfo("Holding target while checking measured joint positions")
    settled_since = None
    settle_deadline = time.monotonic() + settle_timeout
    zero_velocities = {name: 0.0 for name in joint_names}
    error_joint = speed_joint = verified_names[0]
    worst_error = worst_velocity = float("nan")
    while not rospy.is_shutdown() and time.monotonic() < settle_deadline:
        actual, actual_velocity = measured.read(joint_names, state_timeout)
        error_joint = max(verified_names, key=lambda name: abs(targets[name] - actual[name]))
        speed_joint = max(verified_names, key=lambda name: abs(actual_velocity[name]))
        worst_error = abs(targets[error_joint] - actual[error_joint])
        worst_velocity = abs(actual_velocity[speed_joint])
        if worst_error <= tolerance and worst_velocity <= velocity_tolerance:
            if settled_since is None:
                settled_since = time.monotonic()
            elif time.monotonic() - settled_since >= settle_time:
                rospy.loginfo("Start pose verified: max joint error %.4f, max speed %.4f",
                              worst_error, worst_velocity)
                for name in unverified:
                    rospy.loginfo("%s (not verified): %.4f, target %.4f",
                                  name, actual[name], targets[name])
                return
        else:
            settled_since = None
        if publisher.get_num_connections() == 0:
            raise RuntimeError("controller subscriber disconnected")
        if time.monotonic() - last_publisher_check >= 1.0:
            ensure_exclusive_publisher(command_topic)
            last_publisher_check = time.monotonic()
        publish_command(publisher, joint_names, commanded, zero_velocities)
        rate.sleep()
    if not rospy.is_shutdown():
        raise RuntimeError(
            "start pose not reached within {:.1f} s: {} error {:.4f}, {} speed {:.4f}".format(
                settle_timeout, error_joint, worst_error, speed_joint, worst_velocity))


if __name__ == "__main__":
    try:
        move()
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        rospy.logfatal("Start pose move failed: %s", error)
        raise SystemExit(1)
