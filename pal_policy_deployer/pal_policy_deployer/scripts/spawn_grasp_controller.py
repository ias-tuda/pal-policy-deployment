#!/usr/bin/env python3
"""Load grasp actuator gains from the selected model artifact, then start control.

PAL's DirectImpedanceControl is a plain joint PD sent as effort,
tau = stiffness_factor * (q_des - q) + damping_factor * (qd_des - qd) + tau_ff,
so MJLab's actuator stiffness and damping are copied without conversion.
"""

import math
import os
import re

import rospy
import yaml
from controller_manager_msgs.srv import LoadController, SwitchController, UnloadController


JOINTS = ["arm_left_{}_joint".format(index) for index in range(1, 8)]
CONTROLLER = "subscriber_controller"


def actuator_entries(value):
    if isinstance(value, dict):
        if all(key in value for key in ("target_names_expr", "stiffness", "damping")):
            yield value
        for child in value.values():
            for entry in actuator_entries(child):
                yield entry
    elif isinstance(value, list):
        for child in value:
            for entry in actuator_entries(child):
                yield entry


def load_gains(profile_path):
    with open(profile_path, encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    if not isinstance(profile, dict) or not profile.get("model_path"):
        raise ValueError("grasp profile has no model_path")
    model_path = profile["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(profile_path), model_path)
    env_path = os.path.join(os.path.dirname(model_path), "params", "env.yaml")
    # BaseLoader reads MJLab's Python-tagged training YAML as plain data.
    with open(env_path, encoding="utf-8") as stream:
        environment = yaml.load(stream, Loader=yaml.BaseLoader)
    entries = list(actuator_entries(environment))
    gains = {}
    for joint in JOINTS:
        matches = [entry for entry in entries
                   if any(re.fullmatch(pattern, joint)
                          for pattern in entry["target_names_expr"])]
        if len(matches) != 1:
            raise ValueError("{} has {} matching actuator entries in {}".format(
                joint, len(matches), env_path))
        stiffness = float(matches[0]["stiffness"])
        damping = float(matches[0]["damping"])
        if not math.isfinite(stiffness) or stiffness <= 0 or not math.isfinite(damping) or damping < 0:
            raise ValueError("invalid impedance gains for {} in {}".format(joint, env_path))
        gains[joint] = (stiffness, damping)
    return env_path, gains


def main():
    rospy.init_node("spawn_grasp_controller")
    env_path, gains = load_gains(rospy.get_param("~profile"))
    for joint, (stiffness, damping) in gains.items():
        namespace = "/{}/joints/{}/impedance_params".format(CONTROLLER, joint)
        rospy.set_param(namespace + "/stiffness_factor", stiffness)
        rospy.set_param(namespace + "/damping_factor", damping)
        rospy.loginfo("%s: stiffness=%g damping=%g", joint, stiffness, damping)
    rospy.loginfo("Loaded grasp impedance gains from %s", env_path)

    load_name = "/controller_manager/load_controller"
    switch_name = "/controller_manager/switch_controller"
    unload_name = "/controller_manager/unload_controller"
    for service in (load_name, switch_name, unload_name):
        rospy.wait_for_service(service, timeout=30.0)
    load = rospy.ServiceProxy(load_name, LoadController)
    switch = rospy.ServiceProxy(switch_name, SwitchController)
    unload = rospy.ServiceProxy(unload_name, UnloadController)
    if not load(CONTROLLER).ok:
        raise RuntimeError("could not load {}".format(CONTROLLER))
    if not switch([CONTROLLER], [], 2).ok:
        raise RuntimeError("could not start {}".format(CONTROLLER))
    rospy.loginfo("Started %s with grasp training gains", CONTROLLER)
    rospy.on_shutdown(lambda: unload(CONTROLLER))
    rospy.spin()


if __name__ == "__main__":
    main()
