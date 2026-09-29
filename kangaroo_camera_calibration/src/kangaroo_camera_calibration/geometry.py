"""4x4 homogeneous transform helpers."""

import cv2
import numpy as np
from tf import transformations


def from_rvec_tvec(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def to_rvec_tvec(T):
    return cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0].reshape(3), T[:3, 3].copy()


def from_xyz_quat(xyz, quat_xyzw):
    T = transformations.quaternion_matrix(quat_xyzw)
    T[:3, 3] = xyz
    return T


def from_transform_msg(transform):
    t, q = transform.translation, transform.rotation
    return from_xyz_quat([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


def to_xyz_quat(T):
    return T[:3, 3].tolist(), transformations.quaternion_from_matrix(T).tolist()


def to_xyz_rpy(T):
    """Fixed-axis roll, pitch, yaw as used by URDF <origin rpy>."""
    return T[:3, 3].tolist(), list(transformations.euler_from_matrix(T, "sxyz"))


def from_xyz_rpy(xyz, rpy):
    T = transformations.euler_matrix(rpy[0], rpy[1], rpy[2], "sxyz")
    T[:3, 3] = xyz
    return T


def inverse(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T.dot(T[:3, 3])
    return Ti


def exp(delta):
    """Small rigid motion from [rx, ry, rz, tx, ty, tz]."""
    return from_rvec_tvec(delta[:3], delta[3:])


def rotation_angle(T):
    return float(np.arccos(np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))


def average(transforms):
    """Chordal mean of the rotations and arithmetic mean of the translations."""
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in transforms))
    R = U.dot(Vt)
    if np.linalg.det(R) < 0:
        R = U.dot(np.diag([1.0, 1.0, -1.0])).dot(Vt)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.mean([T_i[:3, 3] for T_i in transforms], axis=0)
    return T
