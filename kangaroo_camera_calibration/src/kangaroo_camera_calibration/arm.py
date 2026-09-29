"""Left arm state, kinematics and trajectory execution through left_arm_controller."""

import threading

import actionlib
import numpy as np
import PyKDL
import rospy
from control_msgs.msg import (FollowJointTrajectoryAction, FollowJointTrajectoryGoal,
                              JointTrajectoryControllerState)
from kdl_parser_py.urdf import treeFromUrdfModel
from urdf_parser_py.urdf import URDF

from kangaroo_camera_calibration import geometry


class Kinematics(object):
    """KDL chain from base_frame to tip_frame, with the URDF joint limits."""

    def __init__(self, base_frame, tip_frame, limit_margin=0.05):
        robot = URDF.from_parameter_server()
        ok, tree = treeFromUrdfModel(robot, quiet=True)
        if not ok:
            raise RuntimeError("cannot build a KDL tree from robot_description")
        self.chain = tree.getChain(base_frame, tip_frame)
        joints = {j.name: j for j in robot.joints}
        self.joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getTypeName() != "None":
                self.joint_names.append(joint.getName())
        if not self.joint_names:
            raise RuntimeError("no movable joints between {} and {}".format(base_frame, tip_frame))
        self.lower = np.array([joints[n].limit.lower for n in self.joint_names]) + limit_margin
        self.upper = np.array([joints[n].limit.upper for n in self.joint_names]) - limit_margin
        self._fk = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self._ik = PyKDL.ChainIkSolverPos_LMA(self.chain)

    def within_limits(self, q):
        q = np.asarray(q)
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))

    def fk(self, q):
        frame = PyKDL.Frame()
        self._fk.JntToCart(self._to_kdl(q), frame)
        T = np.eye(4)
        for r in range(3):
            T[r, 3] = frame.p[r]
            for c in range(3):
                T[r, c] = frame.M[r, c]
        return T

    def ik(self, T, q_seed):
        """Joint positions reaching T near q_seed, or None (unreachable or outside limits)."""
        frame = PyKDL.Frame(
            PyKDL.Rotation(*T[:3, :3].ravel().tolist()), PyKDL.Vector(*T[:3, 3].tolist()))
        q_out = PyKDL.JntArray(len(self.joint_names))
        if self._ik.CartToJnt(self._to_kdl(q_seed), frame, q_out) < 0:
            return None
        q = np.array([q_out[i] for i in range(len(self.joint_names))])
        # LMA may wrap continuous-looking solutions by 2 pi; bring them back next to the seed.
        q = q_seed + (q - q_seed + np.pi) % (2.0 * np.pi) - np.pi
        error = geometry.inverse(self.fk(q)).dot(T)
        if np.linalg.norm(error[:3, 3]) > 1e-3 or geometry.rotation_angle(error) > 1e-2:
            return None
        return q if self.within_limits(q) else None

    def _to_kdl(self, q):
        array = PyKDL.JntArray(len(self.joint_names))
        for i, value in enumerate(q):
            array[i] = float(value)
        return array


class Arm(object):
    """Reads the arm state from the controller and executes planned trajectories."""

    def __init__(self, controller, joint_names, max_speed=0.3):
        self.joint_names = list(joint_names)
        self.max_speed = float(max_speed)
        self._lock = threading.Lock()
        self._state = None
        self._state_time = rospy.Time(0)
        self._state_sub = rospy.Subscriber("/{}/state".format(controller),
                                           JointTrajectoryControllerState, self._on_state,
                                           queue_size=1)
        self._client = actionlib.SimpleActionClient(
            "/{}/follow_joint_trajectory".format(controller), FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for /%s/follow_joint_trajectory", controller)
        if not self._client.wait_for_server(rospy.Duration(15.0)):
            raise RuntimeError("controller action server /{}/follow_joint_trajectory not available"
                               .format(controller))

    def _on_state(self, message):
        if set(self.joint_names) - set(message.joint_names):
            return
        index = [message.joint_names.index(n) for n in self.joint_names]
        positions = np.array([message.actual.positions[i] for i in index])
        velocities = np.array([message.actual.velocities[i] for i in index]) \
            if message.actual.velocities else np.zeros(len(index))
        with self._lock:
            self._state = (positions, velocities)
            self._state_time = rospy.Time.now()

    def state(self, max_age=0.5):
        deadline = rospy.Time.now() + rospy.Duration(5.0)
        while not rospy.is_shutdown():
            with self._lock:
                if self._state is not None and (rospy.Time.now() - self._state_time).to_sec() < max_age:
                    return self._state[0].copy(), self._state[1].copy()
            if rospy.Time.now() > deadline:
                raise RuntimeError("no fresh arm controller state")
            rospy.sleep(0.02)
        raise rospy.ROSInterruptException("shutdown")

    def positions(self):
        return self.state()[0]

    def execute(self, trajectory):
        """Blocking execution of a planned trajectory (arm joint order); True on success."""
        q_now = self.positions()
        if np.max(np.abs(np.array(trajectory.points[0].positions) - q_now)) > 0.05:
            rospy.logerr("Trajectory does not start at the current arm pose; not executing")
            return False
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        goal.trajectory.header.stamp = rospy.Time(0)  # start on receipt
        duration = trajectory.points[-1].time_from_start.to_sec()
        self._client.send_goal(goal)
        if not self._client.wait_for_result(rospy.Duration(duration + 10.0)):
            self._client.cancel_goal()
            rospy.logerr("Arm move timed out")
            return False
        result = self._client.get_result()
        if result is None or result.error_code != 0:
            rospy.logerr("Arm move failed: %s", result.error_string if result else "no result")
            return False
        return True

    def stop(self):
        self._client.cancel_all_goals()

    def wait_still(self, velocity_tolerance=0.01, settle_time=1.0, timeout=10.0):
        """Wait until all joint speeds stay below the tolerance for settle_time."""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        still_since = None
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            _, velocity = self.state()
            if np.max(np.abs(velocity)) < velocity_tolerance:
                still_since = still_since or rospy.Time.now()
                if (rospy.Time.now() - still_since).to_sec() >= settle_time:
                    return True
            else:
                still_since = None
            rospy.sleep(0.02)
        return False
