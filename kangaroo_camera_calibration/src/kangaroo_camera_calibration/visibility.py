"""Predict whether the board is fully and frontally visible for a given board pose."""

import numpy as np

from kangaroo_camera_calibration import geometry


def board_problem(T_cam_board, board_size, K, image_size, limits):
    """Reason string if the board would not be fully and frontally visible, else None.

    limits: dict with min_distance, max_distance (m), max_view_angle (deg), image_margin (px).
    """
    w, h = board_size
    center = T_cam_board.dot([w / 2.0, h / 2.0, 0.0, 1.0])[:3]
    distance = np.linalg.norm(center)
    if not limits["min_distance"] <= distance <= limits["max_distance"]:
        return "distance {:.2f} m".format(distance)
    normal = T_cam_board[:3, 2]
    angle = np.degrees(np.arccos(min(1.0, abs(normal.dot(center)) / distance)))
    if angle > limits["max_view_angle"] or normal.dot(center) > 0:
        return "view angle {:.0f} deg or board seen from behind".format(angle)
    corners = T_cam_board.dot(np.array([[0, 0, 0, 1], [w, 0, 0, 1], [w, h, 0, 1], [0, h, 0, 1]],
                                       dtype=float).T)[:3].T
    if np.any(corners[:, 2] <= 0.05):
        return "board behind camera"
    pixels = K.dot(corners.T).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    m = limits["image_margin"]
    if np.any(pixels < m) or np.any(pixels[:, 0] > image_size[0] - m) \
            or np.any(pixels[:, 1] > image_size[1] - m):
        return "board outside image"
    return None


def facing_board_pose(K, pixel, distance, board_size, tilt_x, tilt_y, roll):
    """Board pose in the camera frame: centre on the ray through `pixel` at `distance`,
    facing the camera, then rotated by roll (about its normal) and tilts (radians)."""
    ray = np.linalg.inv(K).dot([pixel[0], pixel[1], 1.0])
    ray /= np.linalg.norm(ray)
    z = -ray
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    T_center = np.eye(4)
    T_center[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T_center[:3, 3] = distance * ray
    T_center = T_center.dot(geometry.from_xyz_rpy([0, 0, 0], [tilt_x, tilt_y, roll]))
    return T_center.dot(geometry.from_xyz_rpy([-board_size[0] / 2.0, -board_size[1] / 2.0, 0.0],
                                              [0, 0, 0]))
