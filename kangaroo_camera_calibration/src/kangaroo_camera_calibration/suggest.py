"""Suggest reachable, safe arm poses that show the board well to the camera.

Board poses are laid out in the camera frame (image position, distance, tilts, roll about
the normal); each is mapped to a gripper pose with the estimated board-on-gripper transform,
solved with IK and kept if `check(q)` (safety rules + MoveIt collision) returns None. The result is ordered greedily so
that every next suggestion differs most from the taught poses and earlier suggestions.
"""

import numpy as np

from kangaroo_camera_calibration import geometry, visibility


class Suggestion(object):
    def __init__(self, q, T_cam_board, description):
        self.q = q
        self.T_cam_board = T_cam_board
        self.description = description


def _difference(A, B):
    """Board pose difference: 10 deg rotation ~ 5 cm translation ~ 1."""
    return np.degrees(geometry.rotation_angle(geometry.inverse(A).dot(B))) / 10.0 + \
        np.linalg.norm(A[:3, 3] - B[:3, 3]) / 0.05


def generate(kin, check, K, image_size, X0, Y, board_size, limits, ik_seeds,
             distances=(0.35, 0.45, 0.55), image_fractions=(0.3, 0.5, 0.7),
             rolls_deg=range(0, 360, 45), tilts_deg=(-25.0, 0.0, 25.0)):
    Y_inv = geometry.inverse(Y)
    found = []
    for distance in distances:
        for fu in image_fractions:
            for fv in image_fractions:
                pixel = (fu * image_size[0], fv * image_size[1])
                for roll in rolls_deg:
                    for tx in tilts_deg:
                        for ty in tilts_deg:
                            T_cb = visibility.facing_board_pose(
                                K, pixel, distance, board_size,
                                np.radians(tx), np.radians(ty), np.radians(roll))
                            if visibility.board_problem(T_cb, board_size, K, image_size, limits):
                                continue
                            T_ee = X0.dot(T_cb).dot(Y_inv)
                            for seed in ik_seeds:
                                q = kin.ik(T_ee, seed)
                                if q is not None and check(q) is None:
                                    found.append(Suggestion(q, T_cb, "{:.2f} m, image ({:.0f}, {:.0f}), "
                                                            "roll {:d}, tilt {:+.0f}/{:+.0f} deg".format(
                                                                distance, pixel[0], pixel[1], roll, tx, ty)))
                                    break
    return found


def order_by_novelty(suggestions, existing_board_poses, max_count=20):
    """Greedy farthest-first order relative to the existing (taught) board poses."""
    remaining = list(suggestions)
    chosen = list(existing_board_poses)
    ordered = []
    while remaining and len(ordered) < max_count:
        if chosen:
            scores = [min(_difference(s.T_cam_board, T) for T in chosen) for s in remaining]
            best = int(np.argmax(scores))
        else:
            best = 0
        ordered.append(remaining.pop(best))
        chosen.append(ordered[-1].T_cam_board)
    return ordered
