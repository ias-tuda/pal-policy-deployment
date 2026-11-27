#!/usr/bin/env python2
"""Upper-body goals from RViz: drag the hands and turn the torso, and the tracker follows.

Python 2: tf2_ros has no Python 3 build on Melodic.

Publishes a geometry_msgs/PoseArray on ~goal_topic (default /upper_body/goal) in base_link:
[left TCP, right TCP, torso] -- the goal the upper-body tracking policy reads
(config/tasks/upper-body-tracking.yaml). The markers start at the robot's current poses (from TF:
the tip links plus tcp_offset, and the torso link), so publishing them unchanged holds the robot
still.

  rosrun pal_policy_deployer upper_body_goal_markers.py
  # In RViz: add an InteractiveMarkers display on /upper_body_goal_markers/update.

~demo:=circle moves both hands on circles about their start poses instead (no RViz needed):
~demo_radius (m, 0.08) and ~demo_period (s, 8) -- a first test in the simulator.

Every goal position is clamped to ~max_offset (m, 0.35) from its start pose: a dragged marker
cannot send a hand across the room. Raise it once the policy is trusted.
"""

import math
import threading

import rospy
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker

NAMES = ("left", "right", "torso")


def _axis_control(name, mode, x, y, z):
    control = InteractiveMarkerControl()
    control.name = name
    control.interaction_mode = mode
    control.orientation = Quaternion(x=x, y=y, z=z, w=1.0)
    norm = math.sqrt(x * x + y * y + z * z + 1.0)
    control.orientation.x /= norm
    control.orientation.y /= norm
    control.orientation.z /= norm
    control.orientation.w /= norm
    return control


def _marker(name, pose, frame, translate):
    marker = InteractiveMarker()
    marker.header.frame_id = frame
    marker.name = name
    marker.description = name
    marker.pose = pose
    marker.scale = 0.15 if translate else 0.3
    sphere = Marker(type=Marker.SPHERE)
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.04
    sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = {
        "left": (0.2, 0.4, 1.0, 0.8),
        "right": (1.0, 0.3, 0.2, 0.8),
        "torso": (0.3, 0.9, 0.3, 0.8),
    }[name]
    visual = InteractiveMarkerControl(always_visible=True)
    visual.markers.append(sphere)
    marker.controls.append(visual)
    for axis, (x, y, z) in (("x", (1, 0, 0)), ("y", (0, 0, 1)), ("z", (0, 1, 0))):
        marker.controls.append(
            _axis_control("rotate_" + axis, InteractiveMarkerControl.ROTATE_AXIS, x, y, z))
        if translate:
            marker.controls.append(
                _axis_control("move_" + axis, InteractiveMarkerControl.MOVE_AXIS, x, y, z))
    return marker


class GoalMarkers:
    def __init__(self):
        self.base = rospy.get_param("~base_link", "base_link")
        self.tips = rospy.get_param("~tip_links", ["arm_left_7_link", "arm_right_7_link"])
        self.torso = rospy.get_param("~torso_link", "pelvis_2_link")
        self.tcp_offset = rospy.get_param("~tcp_offset", [0.0, 0.0, 0.148])
        self.max_offset = float(rospy.get_param("~max_offset", 0.35))
        self.demo = rospy.get_param("~demo", "")
        self.radius = float(rospy.get_param("~demo_radius", 0.08))
        self.period = float(rospy.get_param("~demo_period", 8.0))
        self.lock = threading.Lock()
        self.pub = rospy.Publisher(rospy.get_param("~goal_topic", "/upper_body/goal"),
                                   PoseArray, queue_size=1)

        self.start = self._current_poses()
        self.goal = [_copy(p) for p in self.start]
        self.t0 = rospy.Time.now()
        if self.demo not in ("", "circle"):
            raise ValueError("~demo must be empty or 'circle', not {!r}".format(self.demo))
        if not self.demo:
            self.server = InteractiveMarkerServer("upper_body_goal_markers")
            for name, pose in zip(NAMES, self.goal):
                self.server.insert(_marker(name, pose, self.base, name != "torso"), self._feedback)
            self.server.applyChanges()
        rospy.loginfo("Upper-body goal markers ready (%s)", self.demo or "RViz")

    def _current_poses(self):
        buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(buffer)
        poses = []
        for i, link in enumerate(self.tips + [self.torso]):
            t = buffer.lookup_transform(self.base, link, rospy.Time(0), rospy.Duration(10.0))
            q, p = t.transform.rotation, t.transform.translation
            pose = Pose()
            pose.orientation = q
            offset = self.tcp_offset if i < len(self.tips) else [0.0, 0.0, 0.0]
            dx, dy, dz = _rotate(q, offset)
            pose.position.x, pose.position.y, pose.position.z = p.x + dx, p.y + dy, p.z + dz
            poses.append(pose)
        return poses

    def _feedback(self, feedback):
        with self.lock:
            i = NAMES.index(feedback.marker_name)
            pose = _copy(feedback.pose)
            if NAMES[i] == "torso":
                pose.position = self.start[i].position  # orientation only
            self.goal[i] = pose

    def _clamped(self, i, pose):
        s = self.start[i].position
        d = [pose.position.x - s.x, pose.position.y - s.y, pose.position.z - s.z]
        n = math.sqrt(sum(v * v for v in d))
        if n > self.max_offset:
            k = self.max_offset / n
            pose = _copy(pose)
            pose.position.x, pose.position.y, pose.position.z = (
                s.x + d[0] * k, s.y + d[1] * k, s.z + d[2] * k)
        return pose

    def publish(self, _event=None):
        with self.lock:
            if self.demo == "circle":
                phase = 2.0 * math.pi * (rospy.Time.now() - self.t0).to_sec() / self.period
                for i, sign in ((0, 1.0), (1, -1.0)):
                    pose = _copy(self.start[i])
                    # A circle in the y-z plane, starting at the start pose (no jump).
                    pose.position.y += sign * self.radius * math.sin(phase)
                    pose.position.z += self.radius * (1.0 - math.cos(phase))
                    self.goal[i] = pose
            msg = PoseArray()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base
            msg.poses = [self._clamped(i, p) for i, p in enumerate(self.goal)]
        self.pub.publish(msg)


def _copy(pose):
    out = Pose()
    out.position.x, out.position.y, out.position.z = pose.position.x, pose.position.y, pose.position.z
    o = pose.orientation
    out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = o.x, o.y, o.z, o.w
    return out


def _rotate(q, v):
    """Rotate v by the unit quaternion q (geometry_msgs)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    vx, vy, vz = v
    # t = 2 q_vec x v; v' = v + w t + q_vec x t
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


if __name__ == "__main__":
    rospy.init_node("upper_body_goal_markers")
    markers = GoalMarkers()
    rospy.Timer(rospy.Duration(1.0 / float(rospy.get_param("~rate", 50.0))), markers.publish)
    rospy.spin()
