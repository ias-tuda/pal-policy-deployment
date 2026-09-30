#!/usr/bin/env python3
"""Check that the grasp policy's current observation lies in its training start distribution.

Run against a grasp inferer started with dry_run:=true, so last_action is still zero.
Exits 0 when every feature is inside [min - tol, max + tol] of
start_observation_distribution.json, 1 when a feature is outside, 2 on setup errors.
"""

import json
import math
import os
import sys
import threading
import time

import rospy
from std_msgs.msg import Float64MultiArray


STATS = ("min", "p01", "mean", "std", "p99", "max")


def load_distribution(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported start distribution schema_version")
    terms = []
    for name, stats in config["terms"].items():
        missing = [key for key in STATS if key not in stats]
        if missing:
            raise ValueError("term {} is missing {}".format(name, ", ".join(missing)))
        dims = {len(stats[key]) for key in STATS}
        if len(dims) != 1:
            raise ValueError("term {} has inconsistent statistic lengths".format(name))
        terms.append((name, {key: [float(v) for v in stats[key]] for key in STATS}))
    return terms


def feature_labels(term, dim, joint_names, joint_vel_names):
    if term in ("joint_pos_rel", "joint_target_rel") and len(joint_names) == dim:
        return list(joint_names)
    if term == "joint_vel" and len(joint_vel_names) == dim:
        return list(joint_vel_names)
    if term == "object_yaw_b" and dim == 2:
        return ["cos", "sin"]
    if term == "object_yaw_to_jaw" and dim == 2:
        return ["cos4d", "sin4d"]
    if term == "tcp_quat_b" and dim % 4 == 0:
        return ["{}{}".format(axis, i) for i in range(dim // 4) for axis in "wxyz"]
    if dim % 3 == 0 and term.endswith("_b"):
        return ["{}{}".format(axis, i) if dim > 3 else axis
                for i in range(dim // 3) for axis in "xyz"]
    return [str(i) for i in range(dim)]


def check_observation(observation, config_terms, distribution, joint_names, joint_vel_names,
                      tolerance, term_tolerances):
    """Return (rows, ok). Terms are matched by order; the training and config names differ."""
    if len(config_terms) != len(distribution):
        raise ValueError("config has {} observation terms but the distribution has {}".format(
            len(config_terms), len(distribution)))
    expected_dim = sum(len(stats["mean"]) for _, stats in distribution)
    if len(observation) != expected_dim:
        raise ValueError("observation has {} values but the distribution describes {}".format(
            len(observation), expected_dim))

    rows = []
    ok = True
    offset = 0
    for term, (train_name, stats) in zip(config_terms, distribution):
        dim = len(stats["mean"])
        tol = float(term_tolerances.get(term, term_tolerances.get(train_name, tolerance)))
        for i, label in enumerate(feature_labels(term, dim, joint_names, joint_vel_names)):
            value = observation[offset + i]
            low, high = stats["min"][i] - tol, stats["max"][i] + tol
            std = stats["std"][i]
            z = (value - stats["mean"][i]) / std if std > 0.0 else None
            if not low <= value <= high:
                status = "FAIL"
                ok = False
            elif not stats["p01"][i] <= value <= stats["p99"][i]:
                status = "warn"
            else:
                status = "ok"
            rows.append((status, "{}/{}".format(term, train_name), label, value,
                         stats["min"][i], stats["max"][i], tol, z))
        offset += dim
    return rows, ok


def print_rows(rows, verbose):
    header = "{:<5} {:<30} {:<24} {:>11} {:>11} {:>11} {:>7} {:>9}".format(
        "", "term (deploy/train)", "feature", "value", "train min", "train max", "tol", "z")
    print(header)
    print("-" * len(header))
    for status, term, label, value, low, high, tol, z in rows:
        if status == "ok" and not verbose:
            continue
        z_text = "{:9.1f}".format(z) if z is not None else "{:>9}".format("const")
        print("{:<5} {:<30} {:<24} {:11.5f} {:11.5f} {:11.5f} {:7.4f} {}".format(
            status, term, label, value, low, high, tol, z_text))


class ObservationCollector:
    def __init__(self, topic):
        self._lock = threading.Lock()
        self._samples = []
        self._subscriber = rospy.Subscriber(topic, Float64MultiArray, self._callback, queue_size=10)

    def _callback(self, message):
        with self._lock:
            self._samples.append(list(message.data))

    def collect(self, duration, timeout):
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown():
            with self._lock:
                if self._samples:
                    break
            if time.monotonic() > deadline:
                raise RuntimeError("no observation received within {:.1f} s".format(timeout))
            time.sleep(0.02)
        with self._lock:
            self._samples = []
        end = time.monotonic() + duration
        while not rospy.is_shutdown() and time.monotonic() < end:
            time.sleep(0.02)
        with self._lock:
            samples = list(self._samples)
        if not samples:
            raise RuntimeError("observations stopped during collection")
        if len({len(sample) for sample in samples}) != 1:
            raise RuntimeError("observation length changed during collection")
        return [sum(column) / len(samples) for column in zip(*samples)], len(samples)


def main():
    rospy.init_node("check_start_distribution", anonymous=True)
    policy_node = rospy.get_param("~policy_node", "/pal_policy_inferer_mjlab_node").rstrip("/")
    try:
        config_terms = rospy.get_param(policy_node + "/observation_terms")
        joint_names = rospy.get_param(policy_node + "/joint_names", [])
        joint_vel_names = rospy.get_param(policy_node + "/joint_vel_names", joint_names)
        topic = rospy.get_param("~observation_topic", rospy.get_param(
            policy_node + "/observation_topic", "/pal_policy/observation"))
        path = rospy.get_param("~distribution_path", "")
        if not path:
            model_path = rospy.get_param(policy_node + "/model_path")
            path = os.path.join(os.path.dirname(model_path), "start_observation_distribution.json")
        distribution = load_distribution(path)
    except (KeyError, OSError, ValueError) as error:
        rospy.logerr("Start distribution check setup failed: %s", error)
        return 2

    if not rospy.get_param(policy_node + "/dry_run", False):
        rospy.logwarn("%s is not in dry_run mode; last_action will not be the start value",
                      policy_node)
    tolerance = float(rospy.get_param("~tolerance", 0.005))
    term_tolerances = rospy.get_param("~term_tolerances", {})
    duration = float(rospy.get_param("~duration", 1.0))
    timeout = float(rospy.get_param("~timeout", 10.0))
    verbose = bool(rospy.get_param("~verbose", False))

    rospy.loginfo("Checking %s against %s", topic, path)
    try:
        observation, count = ObservationCollector(topic).collect(duration, timeout)
        rows, ok = check_observation(observation, config_terms, distribution, joint_names,
                                     joint_vel_names, tolerance, term_tolerances)
    except (RuntimeError, ValueError) as error:
        rospy.logerr("Start distribution check failed: %s", error)
        return 2

    print("Mean of {} observations; FAIL outside [min - tol, max + tol], warn outside [p01, p99]".format(count))
    print_rows(rows, verbose)
    failures = sum(1 for row in rows if row[0] == "FAIL")
    if ok:
        rospy.loginfo("PASS: observation is inside the training start distribution")
        return 0
    rospy.logerr("FAIL: %d of %d features are outside the training start distribution",
                 failures, len(rows))
    return 1


if __name__ == "__main__":
    sys.exit(main())
