"""Workspace and board-clearance checks for arm poses and joint-space paths.

All positions are in the chain base frame (torso_link, x forward, y left, z up).
The ChArUco board is modelled as a cylinder around the gripper axis (tip frame z) so
the check holds for any mounting orientation of the board around the gripper.
"""

import numpy as np
import PyKDL

from kangaroo_camera_calibration import geometry


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip((points - a).dot(ab) / max(ab.dot(ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


class SafetyChecker(object):
    def __init__(self, kinematics, params):
        self.kin = kinematics
        self.p = params
        board = params["board_volume"]
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        levels = np.linspace(board["z_min"], board["z_max"], 4)
        ring = [[board["radius"] * np.cos(a), board["radius"] * np.sin(a)] for a in angles]
        self.tool_points = np.array([[x, y, z, 1.0] for z in levels for x, y in ring + [[0.0, 0.0]]])
        self._segment_index = {}
        for i in range(kinematics.chain.getNrOfSegments()):
            self._segment_index[kinematics.chain.getSegment(i).getName()] = i + 1
        for link in [params["elbow_link"], params["wrist_link"]] + \
                [l for pair in params["arm_segments"] for l in pair]:
            if link not in self._segment_index:
                raise ValueError("{} is not on the chain".format(link))
        self._fk = PyKDL.ChainFkSolverPos_recursive(kinematics.chain)

    def _link_position(self, q, link):
        frame = PyKDL.Frame()
        self._fk.JntToCart(self.kin._to_kdl(q), frame, self._segment_index[link])
        return np.array([frame.p[0], frame.p[1], frame.p[2]])

    def tool_points_in_base(self, q):
        return self.kin.fk(q).dot(self.tool_points.T).T[:, :3]

    def check_pose(self, q):
        """Return None if the pose is safe, else a reason string."""
        if not self.kin.within_limits(q):
            return "joint limits"
        w = self.p["workspace"]
        tool = self.tool_points_in_base(q)
        wrist = self._link_position(q, self.p["wrist_link"])
        elbow = self._link_position(q, self.p["elbow_link"])
        for name, points in (("board/gripper", tool), ("wrist", wrist[None, :])):
            if np.any(points[:, 0] < w["x_min"]):
                return "{} behind x_min={:.2f} (x={:.2f})".format(name, w["x_min"], points[:, 0].min())
            if np.any(points[:, 0] > w["x_max"]):
                return "{} beyond x_max={:.2f}".format(name, w["x_max"])
            if np.any(points[:, 1] < w["y_min"]) or np.any(points[:, 1] > w["y_max"]):
                return "{} outside y [{:.2f}, {:.2f}]".format(name, w["y_min"], w["y_max"])
            if np.any(points[:, 2] < w["z_min"]) or np.any(points[:, 2] > w["z_max"]):
                return "{} outside z [{:.2f}, {:.2f}]".format(name, w["z_min"], w["z_max"])
        if elbow[0] < w["elbow_x_min"]:
            return "elbow behind elbow_x_min={:.2f} (x={:.2f})".format(w["elbow_x_min"], elbow[0])
        clearance = self.p["board_arm_clearance"]
        for a_link, b_link in self.p["arm_segments"]:
            d = _segment_distance(tool, self._link_position(q, a_link), self._link_position(q, b_link))
            if d.min() < clearance:
                return "board {:.3f} m from arm segment {}-{}".format(d.min(), a_link, b_link)
        return None

    def check_waypoints(self, waypoints, allow_enter=False, allow_exit=False, resolution=0.02):
        """Check a joint-space trajectory densely (straight segments between waypoints).

        allow_enter: poses at the start may be unsafe (e.g. leaving the rest pose) as long as
        the path never leaves the safe region again once it has entered it. allow_exit is the
        reverse, for returning to the rest pose. Joint limits are always enforced.
        """
        waypoints = [np.asarray(q, float) for q in waypoints]
        poses = [waypoints[0]]
        for a, b in zip(waypoints, waypoints[1:]):
            steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / resolution)))
            poses.extend(a + s * (b - a) for s in np.linspace(0.0, 1.0, steps + 1)[1:])
        reasons = [self.check_pose(q) for q in poses]
        ok = [r is None for r in reasons]
        if all(ok):
            return None
        if any(r == "joint limits" for r in reasons):
            return "path violates joint limits"
        if True not in ok:
            # Entirely outside the front box (e.g. rest pose to rest pose): only for moves that
            # may both enter and exit it; MoveIt still checks collisions.
            return None if allow_enter and allow_exit else "no pose on the path is safe: " + reasons[-1]
        first_ok = ok.index(True)
        last_ok = len(ok) - 1 - ok[::-1].index(True)
        if first_ok > 0 and not allow_enter:
            return "start: " + reasons[0]
        if last_ok < len(ok) - 1 and not allow_exit:
            return "target: " + reasons[-1]
        if not all(ok[first_ok:last_ok + 1]):
            return "path leaves the safe region: " + next(r for r in reasons[first_ok:] if r)
        return None
