#!/usr/bin/env python3
"""Forward grasp JointState targets to position JointTrajectoryControllers.

The training IK and move_to_start_config.py publish sensor_msgs/JointState targets.
This node sends the latest target to each configured controller as a single-point
trajectory, so the motor position loops track it instead of a software PD. When the
input goes stale, it stops sending and the controllers hold their last point.
"""

import math
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_CONTROLLERS = {
    "left_arm_controller": ["arm_left_{}_joint".format(i) for i in range(1, 8)],
    "gripper_left_controller": ["gripper_left_finger_joint"],
}


def positive_param(name, default):
    value = float(rospy.get_param("~" + name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("~{} must be positive and finite".format(name))
    return value


class Bridge:
    def __init__(self):
        self._controllers = rospy.get_param("~controllers", DEFAULT_CONTROLLERS)
        if not isinstance(self._controllers, dict) or not self._controllers:
            raise ValueError("~controllers must map controller names to joint lists")
        self._time_from_start = rospy.Duration(positive_param("time_from_start", 0.05))
        self._input_timeout = positive_param("input_timeout", 0.1)
        self._lock = threading.Lock()
        self._latest = None
        self._receipt_time = 0.0
        self._stale_logged = True
        self._publishers = {
            name: rospy.Publisher("/{}/command".format(name), JointTrajectory, queue_size=1)
            for name in self._controllers}
        input_topic = rospy.get_param("~input_topic", "/pal_grasp/desired_state_uncompensated")
        self._subscriber = rospy.Subscriber(input_topic, JointState, self._callback, queue_size=1)
        self._timer = rospy.Timer(rospy.Duration(1.0 / positive_param("rate", 100.0)), self._send)
        rospy.loginfo("Forwarding %s to %s", input_topic,
                      ", ".join("/{}/command".format(name) for name in self._controllers))

    def _callback(self, message):
        if len(message.name) != len(message.position):
            rospy.logwarn_throttle(1.0, "Ignoring JointState with mismatched name and position")
            return
        positions = dict(zip(message.name, message.position))
        if not all(math.isfinite(value) for value in positions.values()):
            rospy.logwarn_throttle(1.0, "Ignoring JointState with non-finite positions")
            return
        with self._lock:
            self._latest = positions
            self._receipt_time = time.monotonic()

    def _send(self, _event):
        with self._lock:
            positions = self._latest
            age = time.monotonic() - self._receipt_time
        if positions is None or age > self._input_timeout:
            if not self._stale_logged and positions is not None:
                rospy.logwarn("Target input is stale; controllers hold their last point")
            self._stale_logged = True
            return
        if self._stale_logged:
            rospy.loginfo("Receiving targets; forwarding to controllers")
            self._stale_logged = False
        for name, joints in self._controllers.items():
            # Joints absent from this message (e.g. arm-only homing) leave that controller alone.
            if not all(joint in positions for joint in joints):
                continue
            trajectory = JointTrajectory()
            # A zero stamp starts the trajectory on receipt.
            trajectory.joint_names = list(joints)
            point = JointTrajectoryPoint()
            # Positions only, so the controller interpolates linearly instead of braking
            # to zero velocity at every streamed point.
            point.positions = [positions[joint] for joint in joints]
            point.time_from_start = self._time_from_start
            trajectory.points = [point]
            self._publishers[name].publish(trajectory)


def main():
    rospy.init_node("joint_state_to_trajectory")
    Bridge()
    rospy.spin()


if __name__ == "__main__":
    main()
