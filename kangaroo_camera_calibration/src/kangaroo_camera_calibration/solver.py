"""Eye-to-hand calibration: camera fixed on the torso, board fixed on the arm.

Unknowns:
  X = T_base_camera   camera optical frame in the base frame (torso_link)
  Y = T_ee_board      board frame in the arm link it is mounted on

Every sample i gives the forward kinematics A_i = T_base_ee and the observed ChArUco
corners u_ij. Both unknowns are refined jointly by minimising the reprojection error
    u_ij - project(K, D, inv(X) * A_i * Y * p_j)
with Levenberg-Marquardt, so no single-image PnP pose enters the final result.
"""

import cv2
import numpy as np

from kangaroo_camera_calibration import geometry


class Sample(object):
    def __init__(self, T_base_ee, ids, corners, name=""):
        self.T_base_ee = np.asarray(T_base_ee, dtype=np.float64)
        self.ids = np.asarray(ids, dtype=int)
        self.corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        self.name = name


class Result(object):
    def __init__(self, X, Y, rms, per_sample_rms, used, sigma_X, iterations):
        self.X = X
        self.Y = Y
        self.rms = rms                          # px, over all used corners
        self.per_sample_rms = per_sample_rms    # px, one per input sample (also rejected ones)
        self.used = used                        # bool mask of samples kept after rejection
        self.sigma_X = sigma_X                  # 1-sigma of X: [rot rad x3, trans m x3] (camera-frame perturbation)
        self.iterations = iterations


def initial_board_pose(samples, X, board_poses_in_camera):
    """Y from the nominal camera pose and single-image PnP board poses."""
    return geometry.average([geometry.inverse(s.T_base_ee).dot(X).dot(C)
                             for s, C in zip(samples, board_poses_in_camera)])


def project(samples, object_points, K, D, X, Y):
    X_inv = geometry.inverse(X)
    residuals = []
    for s in samples:
        rvec, tvec = geometry.to_rvec_tvec(X_inv.dot(s.T_base_ee).dot(Y))
        projected, _ = cv2.projectPoints(object_points[s.ids], rvec, tvec, K, D)
        residuals.append(projected.reshape(-1, 2) - s.corners)
    return residuals


def _per_sample_rms(residuals):
    return np.array([np.sqrt(np.mean(np.sum(r ** 2, axis=1))) for r in residuals])


def _levenberg_marquardt(samples, object_points, K, D, X, Y, max_iterations=200):
    def evaluate(delta):
        X_d = X.dot(geometry.exp(delta[:6]))
        Y_d = Y.dot(geometry.exp(delta[6:]))
        return np.concatenate(project(samples, object_points, K, D, X_d, Y_d)).ravel(), X_d, Y_d

    lam = 1e-3
    r, _, _ = evaluate(np.zeros(12))
    cost = r.dot(r)
    J = None
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        J = np.empty((r.size, 12))
        eps = 1e-7
        for k in range(12):
            step = np.zeros(12)
            step[k] = eps
            J[:, k] = (evaluate(step)[0] - r) / eps
        H = J.T.dot(J)
        g = J.T.dot(r)
        improved = False
        while lam < 1e10:
            delta = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            r_new, X_new, Y_new = evaluate(delta)
            cost_new = r_new.dot(r_new)
            if cost_new < cost:
                converged = (cost - cost_new) < 1e-12 * cost or np.linalg.norm(delta) < 1e-10
                X, Y, r, cost = X_new, Y_new, r_new, cost_new
                lam = max(lam / 10.0, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved or converged:
            break
    # Recompute the Jacobian at the optimum for the covariance.
    r, _, _ = evaluate(np.zeros(12))
    J = np.empty((r.size, 12))
    for k in range(12):
        step = np.zeros(12)
        step[k] = 1e-7
        J[:, k] = (evaluate(step)[0] - r) / 1e-7
    dof = max(1, r.size - 12)
    sigma2 = r.dot(r) / dof
    try:
        covariance = sigma2 * np.linalg.inv(J.T.dot(J))
        sigma_X = np.sqrt(np.diag(covariance)[:6])
    except np.linalg.LinAlgError:
        sigma_X = np.full(6, np.nan)
    return X, Y, sigma_X, iteration


def solve(samples, object_points, K, D, X0, Y0, outlier_factor=3.0, outlier_min_px=2.0):
    """Refine X and Y. Samples far worse than the median are dropped and the rest re-solved."""
    if len(samples) < 3:
        raise ValueError("need at least 3 samples, got {}".format(len(samples)))
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    used = np.ones(len(samples), dtype=bool)
    X, Y, iterations = X0, Y0, 0
    # Re-solve on the kept samples until the kept set no longer changes.
    for _ in range(5):
        kept = [s for s, u in zip(samples, used) if u]
        X, Y, sigma_X, more = _levenberg_marquardt(kept, object_points, K, D, X, Y)
        iterations += more
        per_sample = _per_sample_rms(project(samples, object_points, K, D, X, Y))
        threshold = max(outlier_min_px, outlier_factor * np.median(per_sample[used]))
        new_used = per_sample <= threshold
        if new_used.sum() < 3 or np.array_equal(new_used, used):
            break
        used = new_used
    residuals = project([s for s, u in zip(samples, used) if u], object_points, K, D, X, Y)
    stacked = np.concatenate(residuals)
    rms = float(np.sqrt(np.mean(np.sum(stacked ** 2, axis=1))))
    return Result(X, Y, rms, per_sample, used, sigma_X, iterations)
