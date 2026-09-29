#!/usr/bin/env python
"""Solve the torso camera extrinsics from a dataset recorded by collect.py.

  rosrun kangaroo_camera_calibration solve.py <dataset.yaml> [<dataset.yaml> ...] [--outlier-factor 3.0]

Several datasets (e.g. separate collect.py runs) are solved together; the result is written
next to the first one.

Writes calibration_result.yaml next to the dataset and prints the corrected URDF
<origin> of the camera mount frame in the base frame (the xacro:sensor_d435 origin).
Needs no running robot.
"""

import argparse
import os
import sys

import numpy as np
import yaml

from kangaroo_camera_calibration import geometry, solver
from kangaroo_camera_calibration.board import CharucoBoard, Detection


def fmt_origin(T):
    xyz, rpy = geometry.to_xyz_rpy(T)
    return '<origin xyz="{:.5f} {:.5f} {:.5f}" rpy="{:.5f} {:.5f} {:.5f}"/>'.format(*(xyz + rpy))


def describe_delta(T_new, T_old):
    delta = geometry.inverse(T_old).dot(T_new)
    return "{:.1f} mm, {:.2f} deg".format(1000.0 * np.linalg.norm(T_new[:3, 3] - T_old[:3, 3]),
                                          np.degrees(geometry.rotation_angle(delta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--outlier-factor", type=float, default=3.0,
                        help="drop samples whose rms exceeds this times the median (default 3)")
    args = parser.parse_args([a for a in sys.argv[1:] if not a.startswith("__")])

    datasets = []
    for path in args.datasets:
        with open(path) as stream:
            datasets.append(yaml.safe_load(stream))
    data = datasets[0]
    for other, path in zip(datasets[1:], args.datasets[1:]):
        if other["camera"]["mount_frame"] != data["camera"]["mount_frame"]:
            raise SystemExit("{} is for another camera".format(path))
        for key in ("board", "base_frame", "ee_frame"):
            if other[key] != data[key]:
                raise SystemExit("{} differs in {}".format(path, key))
        if not np.allclose(other["camera"]["K"], data["camera"]["K"]) or \
                not np.allclose(other["T_base_optical_nominal"], data["T_base_optical_nominal"], atol=1e-6):
            raise SystemExit("{} was recorded with other camera intrinsics or URDF".format(path))
        data["samples"] = data["samples"] + other["samples"]
    board = CharucoBoard.from_params(data["board"])
    K = np.array(data["camera"]["K"])
    D = np.array(data["camera"]["D"])
    X0 = np.array(data["T_base_optical_nominal"])
    T_mount_optical = np.array(data["T_mount_optical"])
    T_base_mount_nominal = np.array(data["T_base_mount_nominal"])

    samples, board_in_camera = [], []
    for s in data["samples"]:
        detection = Detection(np.array(s["ids"]), np.array(s["corners"], dtype=float), None, None)
        if not board.estimate_pose(detection, K, D):
            print("skipping {}: degenerate corners".format(s["name"]))
            continue
        samples.append(solver.Sample(s["T_base_ee"], s["ids"], s["corners"], s["name"]))
        board_in_camera.append(geometry.from_rvec_tvec(detection.rvec, detection.tvec))
    print("camera {}: {} samples, {} corners".format(data["camera"].get("name", data["camera"]["mount_frame"]),
                                                   len(samples), sum(len(s.ids) for s in samples)))

    Y0 = solver.initial_board_pose(samples, X0, board_in_camera)
    result = solver.solve(samples, board.object_points, K, D, X0, Y0, outlier_factor=args.outlier_factor)
    # Nominal camera pose with the board pose averaged from single-image PnP.
    rms_nominal = np.sqrt(np.mean(np.concatenate(
        [np.sum(r ** 2, axis=1) for r in solver.project(samples, board.object_points, K, D, X0, Y0)])))

    print("\nper-sample reprojection rms [px]:")
    for s, rms, used in zip(samples, result.per_sample_rms, result.used):
        print("  {:<14} {:6.2f}{}".format(s.name, rms, "" if used else "   <- rejected as outlier"))
    print("\nreprojection rms: {:.3f} px over {} samples (nominal camera pose: {:.2f} px)".format(
        result.rms, int(result.used.sum()), rms_nominal))
    sigma = result.sigma_X
    print("1-sigma of the camera pose: {:.2f} mm, {:.3f} deg".format(
        1000.0 * np.linalg.norm(sigma[3:]), np.degrees(np.linalg.norm(sigma[:3]))))

    T_base_mount = result.X.dot(geometry.inverse(T_mount_optical))
    print("\ncamera correction vs. current TF: {}".format(describe_delta(result.X, X0)))
    print("\nURDF origin of {} in {} (xacro:sensor_d435 <origin>):".format(
        data["camera"]["mount_frame"], data["base_frame"]))
    print("  current:    " + fmt_origin(T_base_mount_nominal))
    print("  calibrated: " + fmt_origin(T_base_mount))
    board_xyz, board_rpy = geometry.to_xyz_rpy(result.Y)
    print("\nboard origin in {}: xyz {} m, rpy {} rad".format(
        data["ee_frame"], np.round(board_xyz, 4).tolist(), np.round(board_rpy, 3).tolist()))

    out = os.path.join(os.path.dirname(os.path.abspath(args.datasets[0])), "calibration_result.yaml")
    xyz, rpy = geometry.to_xyz_rpy(T_base_mount)
    with open(out, "w") as stream:
        yaml.safe_dump({
            "parent_frame": data["base_frame"],
            "mount_frame": data["camera"]["mount_frame"],
            "mount_origin": {"xyz": xyz, "rpy": rpy},
            "T_base_optical": result.X.tolist(),
            "T_ee_board": result.Y.tolist(),
            "reprojection_rms_px": result.rms,
            "sigma_rot_rad_trans_m": result.sigma_X.tolist(),
            "samples_used": [s.name for s, u in zip(samples, result.used) if u],
            "samples_rejected": [s.name for s, u in zip(samples, result.used) if not u],
            "datasets": [os.path.abspath(p) for p in args.datasets],
        }, stream, default_flow_style=None)
    print("\nwritten to " + out)


if __name__ == "__main__":
    main()
