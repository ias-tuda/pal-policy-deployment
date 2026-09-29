"""Collision-free arm motion: MoveIt plans, our workspace rules check, the controller executes.

MoveIt (move_group on the robot) plans around the URDF collision meshes, with the ChArUco
board attached to the gripper as an inflated box. Every plan is then checked densely with
SafetyChecker (front-of-body workspace, board clearance) and replanned if it breaks a rule.
The accepted trajectory is slowed to max_speed and sent to the arm controller.
"""

import actionlib
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints, JointConstraint,
                             MoveGroupAction, MoveGroupGoal, MoveItErrorCodes)
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from kangaroo_camera_calibration import geometry


ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items() if isinstance(v, int)}


class Motion(object):
    def __init__(self, arm, safety, moveit_cfg, board_size, volume):
        self.arm = arm
        self.safety = safety
        self.cfg = moveit_cfg
        self.board_size = board_size
        self.volume = volume
        self.board = None
        self.set_board(None)
        self._validity = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        self._client = actionlib.SimpleActionClient("/move_group", MoveGroupAction)
        rospy.loginfo("Waiting for MoveIt /move_group")
        if not self._client.wait_for_server(rospy.Duration(15.0)):
            raise RuntimeError("MoveIt /move_group is not available")
        rospy.wait_for_service("/check_state_validity", 15.0)

    def set_board(self, T_ee_board):
        """Attach the board as a box at T_ee_board, or as the safety cylinder if unknown."""
        cfg = self.cfg
        margin = cfg["board_margin"]
        board = AttachedCollisionObject()
        board.link_name = cfg["ee_link"]
        board.touch_links = list(cfg["board_touch_links"])
        board.object.id = "calibration_board"
        board.object.header.frame_id = cfg["ee_link"]
        board.object.operation = CollisionObject.ADD
        if T_ee_board is None:
            v = self.volume
            T = geometry.from_xyz_rpy([0, 0, 0.5 * (v["z_min"] + v["z_max"])], [0, 0, 0])
            shape = SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                   dimensions=[v["z_max"] - v["z_min"], v["radius"]])
        else:
            w, h = self.board_size
            T = np.asarray(T_ee_board).dot(geometry.from_xyz_rpy([w / 2.0, h / 2.0, 0.0], [0, 0, 0]))
            shape = SolidPrimitive(type=SolidPrimitive.BOX,
                                   dimensions=[w + 2 * margin, h + 2 * margin, cfg["board_thickness"] + 2 * margin])
        xyz, quat = geometry.to_xyz_quat(T)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = xyz
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        board.object.primitives = [shape]
        board.object.primitive_poses = [pose]
        self.board = board

    def collision(self, q):
        """None if MoveIt finds no collision for joint positions q, else the colliding links."""
        request = GetStateValidityRequest()
        request.group_name = self.cfg["group"]
        request.robot_state.is_diff = True
        request.robot_state.joint_state = JointState(name=self.arm.joint_names, position=list(q))
        request.robot_state.attached_collision_objects = [self.board]
        response = self._validity(request)
        if response.valid:
            return None
        pairs = sorted(set("{}/{}".format(c.contact_body_1, c.contact_body_2) for c in response.contacts))
        return "collision " + (", ".join(pairs) if pairs else "(constraint)")

    def check_target(self, q):
        """Workspace/board rules and MoveIt collision for one pose."""
        return self.safety.check_pose(q) or self.collision(q)

    def _plan_once(self, q_goal, q_start=None):
        goal = MoveGroupGoal()
        r = goal.request
        r.group_name = self.cfg["group"]
        r.num_planning_attempts = 3
        r.allowed_planning_time = self.cfg["planning_time"]
        r.max_velocity_scaling_factor = self.cfg["velocity_scaling"]
        r.max_acceleration_scaling_factor = self.cfg["velocity_scaling"]
        r.start_state.is_diff = True
        r.start_state.attached_collision_objects = [self.board]
        if q_start is not None:
            r.start_state.joint_state = JointState(name=self.arm.joint_names, position=list(q_start))
        constraints = Constraints()
        for name, value in zip(self.arm.joint_names, q_goal):
            constraints.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(value), tolerance_above=0.005,
                tolerance_below=0.005, weight=1.0))
        r.goal_constraints = [constraints]
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self._client.send_goal(goal)
        if not self._client.wait_for_result(rospy.Duration(self.cfg["planning_time"] + 10.0)):
            self._client.cancel_goal()
            return None, "planning timed out"
        result = self._client.get_result()
        code = result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return None, "MoveIt: " + ERROR_NAMES.get(code, str(code))
        return result.planned_trajectory.joint_trajectory, None

    def plan(self, q_goal, allow_enter=False, allow_exit=False, q_start=None):
        """Plan to q_goal from the current pose (or q_start, for planning ahead without
        moving); returns (trajectory in arm joint order, None) or (None, reason)."""
        reason = self.safety.check_pose(q_goal) if not allow_exit else None
        if reason:
            return None, "target: " + reason
        for _ in range(self.cfg["replan_attempts"]):
            trajectory, reason = self._plan_once(q_goal, q_start)
            if trajectory is None:
                if reason.startswith("MoveIt: ") and "IN_COLLISION" in reason:
                    return None, reason
                continue
            index = [trajectory.joint_names.index(n) for n in self.arm.joint_names]
            for point in trajectory.points:
                point.positions = [point.positions[i] for i in index]
                point.velocities = [point.velocities[i] for i in index] if point.velocities else []
                point.accelerations = [point.accelerations[i] for i in index] if point.accelerations else []
                point.effort = []
            trajectory.joint_names = list(self.arm.joint_names)
            waypoints = [np.array(p.positions) for p in trajectory.points]
            reason = self.safety.check_waypoints(waypoints, allow_enter=allow_enter, allow_exit=allow_exit)
            if reason is None:
                return self._limit_speed(trajectory), None
        return None, reason

    def _limit_speed(self, trajectory):
        """Stretch time so no joint exceeds the configured max speed between waypoints."""
        points = trajectory.points
        factor = 1.0
        for a, b in zip(points, points[1:]):
            dt = (b.time_from_start - a.time_from_start).to_sec()
            if dt > 0:
                speed = np.max(np.abs(np.array(b.positions) - np.array(a.positions))) / dt
                factor = max(factor, speed / self.arm.max_speed)
        if factor > 1.0:
            for p in points:
                p.time_from_start = rospy.Duration(p.time_from_start.to_sec() * factor)
                p.velocities = [v / factor for v in p.velocities]
                p.accelerations = [a / factor ** 2 for a in p.accelerations]
        return trajectory

    def move_to(self, q_goal, allow_enter=False, allow_exit=False):
        """Plan, check and execute; returns True on success."""
        trajectory, reason = self.plan(q_goal, allow_enter, allow_exit)
        if trajectory is None:
            rospy.logwarn("No safe plan: %s", reason)
            return False
        return self.arm.execute(trajectory)
