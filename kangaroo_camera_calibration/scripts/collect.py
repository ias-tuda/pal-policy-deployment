#!/usr/bin/env python
"""Visit the taught seed poses plus random poses around them and record board observations.

For every seed: move there, detect the board, and estimate where the board sits on the
arm (with the nominal camera pose). Then draw random joint offsets around the seed and
keep only candidates that
  - pass the safety rules (front workspace, board clearance, joint limits),
  - are collision-free for MoveIt (URDF collision meshes, board attached),
  - are predicted to show the whole board, facing the camera, at a sensible distance.
Every move is planned by MoveIt and the plan is checked against the safety rules (see
motion.py). After each move the arm must be still; samples where the board is not
actually detected are dropped. Everything is written to <output_dir>/<camera>_<date>/dataset.yaml (+ images) after
every sample, so an interrupted run keeps its data.

Run with rosrun (it asks for confirmation on stdin):
  rosrun kangaroo_camera_calibration collect.py [--camera torso_front] [--dry-run] [--seeds seed_0,seed_1]
--dry-run plans every move (each from the end of the previous plan) but executes none.
--seeds collects only the named taught poses.
If no plan to a seed stays in front of the body (e.g. the arm has to reconfigure), the arm
goes via collect/home_pose (the hanging rest pose) and approaches from there.
Flags are plain arguments on purpose: _param:=value would stay on the parameter server.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import rospy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from kangaroo_camera_calibration import config, geometry, visibility
from kangaroo_camera_calibration.arm import Arm, Kinematics
from kangaroo_camera_calibration.board import CharucoBoard
from kangaroo_camera_calibration.motion import Motion
from kangaroo_camera_calibration.safety import SafetyChecker


class ImageCache(object):
    def __init__(self, topic):
        self.lock = threading.Lock()
        self.message = None
        self.bridge = CvBridge()
        rospy.Subscriber(topic, Image, self.on_image, queue_size=1, buff_size=2 ** 24)

    def on_image(self, message):
        with self.lock:
            self.message = message

    def wait_newer_than(self, stamp, timeout=5.0):
        deadline = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                message = self.message
            if message is not None and message.header.stamp > stamp:
                return message, self.bridge.imgmsg_to_cv2(message, "bgr8")
            rospy.sleep(0.02)
        raise RuntimeError("no new camera image")


def matrix_list(T):
    return [[round(float(v), 9) for v in row] for row in T]


class Collector(object):
    def __init__(self, config_path, camera, dry_run, seed_names=None):
        self.cfg = cfg = config.load(config_path, camera)
        self.dry_run = dry_run
        c = cfg["collect"]
        self.c = c
        self.rng = np.random.RandomState(c["random_seed"])
        self.board = CharucoBoard.from_params(cfg["board"])
        arm_cfg = cfg["arm"]
        self.base_frame, self.ee_frame = arm_cfg["base_frame"], arm_cfg["ee_frame"]
        self.kin = Kinematics(self.base_frame, self.ee_frame)
        self.safety = SafetyChecker(self.kin, cfg["safety"])
        self.arm = Arm(arm_cfg["controller"], self.kin.joint_names, arm_cfg["max_speed"])
        self.motion = Motion(self.arm, self.safety, cfg["moveit"], self.board.size,
                             cfg["safety"]["board_volume"])
        board_file = os.path.join(config.package_path(), "config", "board_on_gripper.yaml")
        self.Y_saved = None
        if os.path.exists(board_file):
            with open(board_file) as stream:
                self.Y_saved = np.array(yaml.safe_load(stream)["T_ee_board"])
        self.motion.set_board(self.Y_saved)
        with open(c["poses_file"]) as stream:
            taught = yaml.safe_load(stream) or {}
        if taught.get("joint_names") != self.kin.joint_names:
            raise ValueError("{} does not match joints {}".format(c["poses_file"], self.kin.joint_names))
        self.seeds = taught.get("poses", [])
        if seed_names:
            unknown = set(seed_names) - set(p["name"] for p in self.seeds)
            if unknown:
                raise ValueError("unknown seeds: {}".format(", ".join(sorted(unknown))))
            self.seeds = [p for p in self.seeds if p["name"] in seed_names]
        self.home = np.asarray(c["home_pose"], dtype=float)
        if not self.seeds:
            raise ValueError("no taught poses in {}; run the pose teacher first".format(c["poses_file"]))

        cam = cfg["camera"]
        info = rospy.wait_for_message(cam["info_topic"], CameraInfo, timeout=30.0)
        self.K, self.D = np.array(info.K).reshape(3, 3), np.array(info.D)
        self.image_size = (info.width, info.height)
        self.images = ImageCache(cam["image_topic"])
        self.tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf)
        self.X0 = self.lookup(self.base_frame, cam["optical_frame"])
        self.dataset = {
            "camera": {"name": cam["name"], "K": self.K.tolist(), "D": self.D.tolist(), "width": info.width,
                       "height": info.height, "optical_frame": cam["optical_frame"],
                       "mount_frame": cam["mount_frame"]},
            "base_frame": self.base_frame,
            "ee_frame": self.ee_frame,
            "board": cfg["board"],
            "joint_names": self.kin.joint_names,
            "T_base_optical_nominal": matrix_list(self.X0),
            "T_base_mount_nominal": matrix_list(self.lookup(self.base_frame, cam["mount_frame"])),
            "T_mount_optical": matrix_list(self.lookup(cam["mount_frame"], cam["optical_frame"])),
            "samples": [],
        }
        self.out_dir = os.path.join(c["output_dir"], "{}_{}".format(cam["name"], time.strftime("%Y-%m-%d_%H-%M-%S")))

    def lookup(self, parent, child, stamp=rospy.Time(0)):
        t = self.tf.lookup_transform(parent, child, stamp, rospy.Duration(3.0))
        return geometry.from_transform_msg(t.transform)

    # ------------------------------------------------------------------ prediction
    def predicted_visible(self, q, Y):
        """Reason string if the board would not be fully and frontally visible, else None."""
        T_cam_board = geometry.inverse(self.X0).dot(self.kin.fk(q)).dot(Y)
        return visibility.board_problem(T_cam_board, self.board.size, self.K, self.image_size, self.c)

    def candidates(self, seed, Y, count):
        offsets = np.asarray(self.c["joint_offsets"], dtype=float)
        found, attempts, rejected = [], 0, {}
        while len(found) < count and attempts < 200 * count:
            attempts += 1
            q = seed + self.rng.uniform(-offsets, offsets)
            reason = self.safety.check_pose(q) or \
                (self.predicted_visible(q, Y) if Y is not None else None) or \
                self.motion.collision(q)
            if reason:
                key = reason.split(" ")[0]
                rejected[key] = rejected.get(key, 0) + 1
                continue
            found.append(q)
        rospy.loginfo("%d candidates from %d draws; rejected: %s", len(found), attempts, rejected)
        return found

    # ------------------------------------------------------------------ motion + capture
    def move(self, q_to, allow_enter=False, allow_exit=False):
        """Plan (MoveIt + safety rules) and execute. False if no safe plan; raises if
        execution fails."""
        # A dry run plans every move from where the previous planned move ended.
        trajectory, reason = self.motion.plan(q_to, allow_enter=allow_enter, allow_exit=allow_exit,
                                              q_start=self.q_planned if self.dry_run else None)
        if trajectory is None:
            rospy.logwarn("Skipping move, no safe plan: %s", reason)
            return False
        if self.dry_run:
            self.q_planned = np.array(trajectory.points[-1].positions)
            return True
        if not self.arm.execute(trajectory):
            raise RuntimeError("arm move failed; stopping")
        return True

    def capture(self, name, q_expected):
        if self.dry_run:
            return None
        if not self.arm.wait_still(settle_time=self.c["settle_time"]):
            rospy.logwarn("%s: arm did not settle", name)
            return None
        q = self.arm.positions()
        if np.max(np.abs(q - q_expected)) > 0.05:
            rospy.logwarn("%s: arm is %.3f rad away from the target", name, np.max(np.abs(q - q_expected)))
        message, image = self.images.wait_newer_than(rospy.Time.now())
        detection = self.board.detect(image, self.K, self.D)
        if not self.board.is_valid(detection):
            count = len(detection.ids) if detection is not None else 0
            rospy.logwarn("%s: board not detected (%d corners)", name, count)
            return None
        if detection.reprojection_rms > self.c["max_pnp_rms"]:
            rospy.logwarn("%s: bad detection, PnP rms %.2f px", name, detection.reprojection_rms)
            return None
        try:
            T_base_ee = self.lookup(self.base_frame, self.ee_frame, message.header.stamp)
        except tf2_ros.TransformException:
            T_base_ee = self.lookup(self.base_frame, self.ee_frame)
        if not os.path.isdir(os.path.join(self.out_dir, "images")):
            os.makedirs(os.path.join(self.out_dir, "images"))
        image_file = os.path.join("images", name + ".png")
        cv2.imwrite(os.path.join(self.out_dir, image_file), image)
        self.dataset["samples"].append({
            "name": name,
            "joints": [round(float(v), 6) for v in q],
            "T_base_ee": matrix_list(T_base_ee),
            "ids": detection.ids.tolist(),
            "corners": [[round(float(u), 3), round(float(v), 3)] for u, v in detection.corners],
            "pnp_rms": round(detection.reprojection_rms, 3),
            "image": image_file,
        })
        with open(os.path.join(self.out_dir, "dataset.yaml"), "w") as stream:
            yaml.safe_dump(self.dataset, stream, default_flow_style=None)
        rospy.loginfo("%s: %d corners, PnP rms %.2f px (%d samples)", name, len(detection.ids),
                      detection.reprojection_rms, len(self.dataset["samples"]))
        T_cam_board = geometry.from_rvec_tvec(detection.rvec, detection.tvec)
        Y = geometry.inverse(T_base_ee).dot(self.X0).dot(T_cam_board)
        self.motion.set_board(Y)
        return Y

    # ------------------------------------------------------------------ main
    def run(self):
        start = self.arm.positions()
        self.q_planned = start
        rospy.logwarn("Collecting around %d seed poses, %d random samples each, max speed %.2f rad/s%s",
                      len(self.seeds), self.c["samples_per_seed"], self.arm.max_speed,
                      " (DRY RUN, arm will not move)" if self.dry_run else "")
        if not self.dry_run:
            sys.stdout.write("Keep the e-stop at hand. Type 'yes' to start: ")
            sys.stdout.flush()
            if sys.stdin.readline().strip() != "yes":
                rospy.loginfo("Aborted")
                return
        first = True
        for index, seed_pose in enumerate(self.seeds):
            seed = np.asarray(seed_pose["joints"], dtype=float)
            name = seed_pose["name"]
            reason = self.motion.check_target(seed)
            if reason:
                rospy.logerr("%s is unsafe (%s); skipping", name, reason)
                continue
            if not self.move(seed, allow_enter=first):
                rospy.loginfo("%s: trying via the home pose", name)
                if not (self.move(self.home, allow_enter=True, allow_exit=True)
                        and self.move(seed, allow_enter=True)):
                    rospy.logerr("%s: not reachable; skipping", name)
                    continue
            first = False
            Y = self.capture("{}_000".format(name), seed)
            if Y is None:
                if not self.dry_run or self.Y_saved is None:
                    rospy.logerr("%s: board not seen at the seed; skipping its random samples", name)
                    continue
                Y = self.Y_saved
            for k, q in enumerate(self.candidates(seed, Y, self.c["samples_per_seed"]), 1):
                if rospy.is_shutdown():
                    return
                if self.move(q):
                    self.capture("{}_{:03d}".format(name, k), q)
            self.move(seed)
        if self.c["return_to_start"] and not first:
            rospy.loginfo("Returning to the start pose")
            if not self.move(start, allow_exit=True):
                rospy.loginfo("Going to the home pose instead")
                self.move(self.home, allow_enter=True, allow_exit=True)
        rospy.loginfo("Done: %d samples in %s", len(self.dataset["samples"]), self.out_dir)
        if self.dataset["samples"]:
            rospy.loginfo("Solve with: rosrun kangaroo_camera_calibration solve.py %s",
                          os.path.join(self.out_dir, "dataset.yaml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect calibration samples")
    parser.add_argument("--camera", help="camera name from the config (default: default_camera)")
    parser.add_argument("--dry-run", action="store_true", help="plan and check moves, do not move")
    parser.add_argument("--seeds", help="comma-separated taught pose names (default: all)")
    parser.add_argument("--config", default=os.path.join(config.package_path(), "config", "calibration.yaml"))
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("calibration_collector")
    collector = Collector(args.config, args.camera, args.dry_run, args.seeds.split(",") if args.seeds else None)
    rospy.on_shutdown(collector.arm.stop)
    try:
        collector.run()
    except RuntimeError as error:
        collector.arm.stop()
        rospy.logfatal("%s", error)
        sys.exit(1)
