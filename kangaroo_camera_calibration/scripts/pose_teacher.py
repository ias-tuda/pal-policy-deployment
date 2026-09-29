#!/usr/bin/env python
"""Teach calibration seed poses in RViz with an interactive marker.

Drag the marker (the board volume on the gripper). On release the node solves IK, checks
the pose (workspace, board clearance, MoveIt collision meshes) and lets MoveIt plan a
collision-free path that also passes the workspace rules:
green = reachable and safe, red = rejected (reason in the marker text).
Right-click menu:
  Suggest pose           puts the marker on the next reachable, safe pose that shows the
                         board well and differs most from the recorded poses
  Move arm to marker     plans again from the current pose and executes (green only)
  Record current pose    stores the arm's actual joint positions (board must be detected)
  Delete last pose
  Reset marker to arm
Poses are written to collect/poses_file of the config after every change.
Suggestions need the board pose on the gripper: it is estimated from the detector's TF
whenever the board is detected and the arm is still, and kept in config/board_on_gripper.yaml.
"""

import os
import threading

import numpy as np
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import Point, Pose
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import ColorRGBA, Int32
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    InteractiveMarkerFeedback, Marker, MarkerArray)

from kangaroo_camera_calibration import config, geometry, suggest
from kangaroo_camera_calibration.arm import Arm, Kinematics
from kangaroo_camera_calibration.board import CharucoBoard
from kangaroo_camera_calibration.motion import Motion
from kangaroo_camera_calibration.safety import SafetyChecker


GREEN = ColorRGBA(0.1, 0.8, 0.1, 0.5)
RED = ColorRGBA(0.9, 0.1, 0.1, 0.5)
GREY = ColorRGBA(0.6, 0.6, 0.6, 0.5)


def pose_from_matrix(T):
    xyz, quat = geometry.to_xyz_quat(T)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    return pose


def matrix_from_pose(pose):
    p, o = pose.position, pose.orientation
    return geometry.from_xyz_quat([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])


class PoseTeacher(object):
    def __init__(self):
        cfg = config.load()
        arm_cfg = cfg["arm"]
        safety_cfg = cfg["safety"]
        self.poses_file = cfg["collect"]["poses_file"]
        self.board_file = os.path.join(config.package_path(), "config", "board_on_gripper.yaml")
        self.base_frame = arm_cfg["base_frame"]
        self.ee_frame = arm_cfg["ee_frame"]
        self.collect_cfg = cfg["collect"]
        self.board_size = CharucoBoard.from_params(cfg["board"]).size
        self.kin = Kinematics(arm_cfg["base_frame"], arm_cfg["ee_frame"])
        self.safety = SafetyChecker(self.kin, safety_cfg)
        self.volume = safety_cfg["board_volume"]
        self.arm = Arm(arm_cfg["controller"], self.kin.joint_names, arm_cfg["max_speed"])
        self.motion = Motion(self.arm, self.safety, cfg["moveit"], self.board_size, self.volume)
        self.lock = threading.Lock()
        self.moving = False
        self.target = None
        self.num_corners = 0
        self.poses = self.load_poses()
        self.published_labels = 0
        self.Y = self.load_board_pose()
        self.motion.set_board(self.Y)
        self.suggestions = None
        self.suggestion_index = 0
        self.tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf)
        info = rospy.wait_for_message(cfg["camera"]["info_topic"], CameraInfo, timeout=30.0)
        self.K = np.array(info.K).reshape(3, 3)
        self.image_size = (info.width, info.height)
        self.X0 = geometry.from_transform_msg(self.tf.lookup_transform(
            self.base_frame, cfg["camera"]["optical_frame"], rospy.Time(0), rospy.Duration(5.0)).transform)
        rospy.Subscriber("charuco_detector/num_corners", Int32, self.on_corners, queue_size=1)
        self.recorded_pub = rospy.Publisher("~recorded_poses", MarkerArray, queue_size=1, latch=True)

        self.server = InteractiveMarkerServer("pose_teacher")
        self.menu = MenuHandler()
        self.menu.insert("Suggest pose", callback=self.on_suggest)
        self.menu.insert("Move arm to marker", callback=self.on_move)
        self.menu.insert("Record current pose", callback=self.on_record)
        self.menu.insert("Delete last pose", callback=self.on_delete)
        self.menu.insert("Reset marker to arm", callback=self.on_reset)
        self.T_marker = self.kin.fk(self.arm.positions())
        self.update_marker(GREY, "drag me, then right-click")
        self.publish_recorded()
        rospy.loginfo("Pose teacher ready: %d poses in %s", len(self.poses), self.poses_file)

    # ------------------------------------------------------------------ board on gripper
    def load_board_pose(self):
        if not os.path.exists(self.board_file):
            return None
        with open(self.board_file) as stream:
            return np.array(yaml.safe_load(stream)["T_ee_board"])

    def update_board_pose(self):
        """Estimate the board pose on the gripper from the detector's TF (arm must be still).
        Returns True if the estimate changed noticeably."""
        if self.num_corners == 0:
            return False
        try:
            _, velocity = self.arm.state()
            if np.max(np.abs(velocity)) > 0.01:
                return False
            t = self.tf.lookup_transform(self.ee_frame, "charuco_board_detected", rospy.Time(0),
                                         rospy.Duration(1.0))
        except (tf2_ros.TransformException, RuntimeError) as error:
            rospy.logwarn("Board pose on the gripper not updated: %s", error)
            return False
        Y = geometry.from_transform_msg(t.transform)
        if self.Y is not None:
            change = geometry.inverse(self.Y).dot(Y)
            if np.linalg.norm(change[:3, 3]) < 0.005 and np.degrees(geometry.rotation_angle(change)) < 2.0:
                return False
        self.Y = Y
        self.motion.set_board(Y)
        with open(self.board_file, "w") as stream:
            yaml.safe_dump({"ee_frame": self.ee_frame, "T_ee_board": self.Y.tolist()}, stream,
                           default_flow_style=None)
        return True

    # ------------------------------------------------------------------ poses file
    def load_poses(self):
        if not os.path.exists(self.poses_file):
            return []
        with open(self.poses_file) as stream:
            data = yaml.safe_load(stream) or {}
        if data.get("joint_names", self.kin.joint_names) != self.kin.joint_names:
            raise ValueError("{} was taught for other joints".format(self.poses_file))
        return data.get("poses", [])

    def save_poses(self):
        with open(self.poses_file, "w") as stream:
            yaml.safe_dump({"joint_names": self.kin.joint_names, "poses": self.poses},
                           stream, default_flow_style=None)
        self.publish_recorded()

    def publish_recorded(self):
        # Stale labels are deleted by id; DELETEALL in a MarkerArray crashes RViz on Melodic.
        markers = MarkerArray()
        for i in range(len(self.poses), self.published_labels):
            m = Marker()
            m.header.frame_id = self.base_frame
            m.ns, m.id, m.action = "recorded", i, Marker.DELETE
            m.pose.orientation.w = 1.0
            markers.markers.append(m)
        self.published_labels = len(self.poses)
        for i, pose in enumerate(self.poses):
            T = self.kin.fk(pose["joints"])
            m = Marker()
            m.header.frame_id = self.base_frame
            m.ns, m.id, m.type = "recorded", i, Marker.TEXT_VIEW_FACING
            m.pose = pose_from_matrix(T)
            m.scale.z = 0.04
            m.color = ColorRGBA(0.2, 0.4, 1.0, 1.0)
            m.text = pose["name"]
            markers.markers.append(m)
        self.recorded_pub.publish(markers)

    # ------------------------------------------------------------------ marker
    def update_marker(self, color, text):
        marker = InteractiveMarker()
        marker.header.frame_id = self.base_frame
        marker.name = "board_target"
        marker.description = text
        marker.scale = 0.3
        marker.pose = pose_from_matrix(self.T_marker)

        body = InteractiveMarkerControl()
        body.always_visible = True
        body.interaction_mode = InteractiveMarkerControl.MENU
        volume = Marker()
        volume.type = Marker.CYLINDER
        volume.scale.x = volume.scale.y = 2.0 * self.volume["radius"]
        volume.scale.z = self.volume["z_max"] - self.volume["z_min"]
        volume.pose.position.z = 0.5 * (self.volume["z_max"] + self.volume["z_min"])
        volume.pose.orientation.w = 1.0
        volume.color = color
        axis = Marker()
        axis.type = Marker.ARROW
        axis.points = [Point(0, 0, 0), Point(0, 0, self.volume["z_max"] + 0.05)]
        axis.pose.orientation.w = 1.0
        axis.scale.x, axis.scale.y, axis.scale.z = 0.01, 0.02, 0.03
        axis.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        body.markers = [volume, axis]
        marker.controls.append(body)

        h = np.sqrt(0.5)
        for name, (w, x, y, z) in (("x", (h, h, 0, 0)), ("y", (h, 0, 0, h)), ("z", (h, 0, h, 0))):
            for mode, prefix in ((InteractiveMarkerControl.MOVE_AXIS, "move_"),
                                 (InteractiveMarkerControl.ROTATE_AXIS, "rotate_")):
                control = InteractiveMarkerControl()
                control.name = prefix + name
                control.orientation.w, control.orientation.x = w, x
                control.orientation.y, control.orientation.z = y, z
                control.interaction_mode = mode
                marker.controls.append(control)

        self.server.insert(marker, self.on_feedback)
        self.menu.apply(self.server, marker.name)
        self.server.applyChanges()

    def on_feedback(self, feedback):
        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return
        self.T_marker = matrix_from_pose(feedback.pose)
        self.evaluate()

    def evaluate(self):
        q_now = self.arm.positions()
        q = self.kin.ik(self.T_marker, q_now)
        if q is None:
            self.target = None
            self.update_marker(RED, "no IK solution within joint limits")
            return
        self.update_marker(GREY, "planning...")
        reason = self.motion.check_target(q)
        if reason is None:
            _, reason = self.motion.plan(q, allow_enter=True)
        if reason:
            self.target = None
            self.update_marker(RED, reason)
            return
        self.target = q
        self.update_marker(GREEN, "ok, safe plan found")

    # ------------------------------------------------------------------ menu
    def on_corners(self, message):
        self.num_corners = message.data

    def on_suggest(self, _feedback):
        with self.lock:
            if self.moving:
                rospy.logwarn("Wait until the arm stopped")
                return
            self.moving = True  # blocks moves while computing
        threading.Thread(target=self.suggest_next).start()

    def suggest_next(self):
        try:
            if self.update_board_pose():
                self.suggestions = None  # new board estimate: recompute
            if self.Y is None:
                rospy.logerr("Show the board to the camera once (arm still) so its pose on the "
                             "gripper is known; then suggest again")
                return
            q_now = self.arm.positions()
            if self.suggestions is None:
                self.update_marker(GREY, "computing suggestions...")
                seeds = [q_now] + [np.array(p["joints"]) for p in self.poses]
                found = suggest.generate(self.kin, self.motion.check_target, self.K, self.image_size, self.X0,
                                         self.Y, self.board_size, self.collect_cfg, seeds)
                existing = [geometry.inverse(self.X0).dot(self.kin.fk(q)).dot(self.Y)
                            for q in [q_now] + [p["joints"] for p in self.poses]]
                self.suggestions = suggest.order_by_novelty(found, existing, max_count=40)
                self.suggestion_index = 0
                rospy.loginfo("%d reachable poses show the board; offering the %d most novel",
                              len(found), len(self.suggestions))
            while self.suggestion_index < len(self.suggestions):
                s = self.suggestions[self.suggestion_index]
                self.suggestion_index += 1
                self.update_marker(GREY, "planning suggestion {}/{}...".format(
                    self.suggestion_index, len(self.suggestions)))
                if self.motion.plan(s.q, allow_enter=True)[0] is not None:
                    self.T_marker = self.kin.fk(s.q)
                    self.target = s.q
                    self.update_marker(GREEN, "suggestion {}/{}: {}".format(
                        self.suggestion_index, len(self.suggestions), s.description))
                    return
            self.suggestions = None
            self.update_marker(RED, "no more suggestions reachable from here; suggest again")
        finally:
            with self.lock:
                self.moving = False

    def on_move(self, _feedback):
        with self.lock:
            if self.moving:
                rospy.logwarn("Arm is already moving")
                return
            if self.target is None:
                rospy.logwarn("Marker pose is not reachable/safe; not moving")
                return
            self.moving = True
        threading.Thread(target=self.move, args=(self.target.copy(),)).start()

    def move(self, q):
        try:
            self.update_marker(GREY, "planning and moving...")
            ok = self.motion.move_to(q, allow_enter=True)
            self.T_marker = self.kin.fk(self.arm.positions())
            self.update_marker(GREEN if ok else RED, "arrived" if ok else "move failed")
        finally:
            with self.lock:
                self.moving = False

    def on_record(self, _feedback):
        if self.moving:
            rospy.logwarn("Wait until the arm stopped")
            return
        q = self.arm.positions()
        reason = self.motion.check_target(q)
        if reason:
            rospy.logerr("Not recording, the pose is unsafe for sampling: %s", reason)
            return
        if self.num_corners == 0:
            rospy.logerr("Not recording, the board is not detected (check ~image of charuco_detector)")
            return
        name = "seed_{}".format(len(self.poses))
        self.poses.append({"name": name, "joints": [round(float(v), 5) for v in q],
                           "corners": int(self.num_corners)})
        self.save_poses()
        rospy.loginfo("Recorded %s with %d corners -> %s", name, self.num_corners, self.poses_file)
        self.update_board_pose()
        self.suggestions = None  # re-rank novelty against the new pose

    def on_delete(self, _feedback):
        if self.poses:
            rospy.loginfo("Deleted %s", self.poses.pop()["name"])
            self.save_poses()

    def on_reset(self, _feedback):
        self.T_marker = self.kin.fk(self.arm.positions())
        self.evaluate()


if __name__ == "__main__":
    rospy.init_node("pose_teacher")
    teacher = PoseTeacher()
    rospy.on_shutdown(teacher.arm.stop)
    rospy.spin()
