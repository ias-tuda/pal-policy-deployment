#!/usr/bin/env python3
"""Load grasp actuator gains from the selected model artifact, then start control.

PAL's DirectImpedanceControl is a plain joint PD sent as effort,
tau = stiffness_factor * (q_des - q) + damping_factor * (qd_des - qd) + tau_ff,
so MJLab's actuator stiffness and damping are copied without conversion.
"""

import math
import os
import re

import rosnode
import rospy
import yaml
from task_config import resolve
from controller_manager_msgs.srv import (ListControllers, LoadController, SwitchController,
                                         UnloadController)


# Used when the config has no impedance_joint_names: the single-arm grasp robots.
DEFAULT_JOINTS = ["arm_left_{}_joint".format(index) for index in range(1, 8)]
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


def load_gains(config_path):
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config.get("model_path"):
        raise ValueError("deployment config has no model_path")
    joints = config.get("impedance_joint_names", DEFAULT_JOINTS)
    if not isinstance(joints, list) or not joints or not all(isinstance(j, str) for j in joints):
        raise ValueError("impedance_joint_names must be a non-empty list of joint names")
    model_path = config["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(config_path), model_path)
    env_path = os.path.join(os.path.dirname(model_path), "params", "env.yaml")
    # BaseLoader reads MJLab's Python-tagged training YAML as plain data.
    with open(env_path, encoding="utf-8") as stream:
        environment = yaml.load(stream, Loader=yaml.BaseLoader)
    entries = list(actuator_entries(environment))
    gains = {}
    for joint in joints:
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


def release(switch, unload, state):
    """Stop the controller if it runs, then unload it; controller_manager refuses to unload
    a running controller. The arm has no torque until the controller runs again."""
    if state == "running" and not switch([], [CONTROLLER], 2).ok:
        rospy.logwarn("could not stop %s", CONTROLLER)
    if not unload(CONTROLLER).ok:
        rospy.logwarn("could not unload %s", CONTROLLER)


def main():
    rospy.init_node("spawn_grasp_controller")
    config_path = rospy.get_param("~config", "") or resolve(
        rospy.get_param("~task", ""), rospy.get_param("~robot", ""))[1]
    env_path, gains = load_gains(config_path)
    for joint, (stiffness, damping) in gains.items():
        namespace = "/{}/joints/{}/impedance_params".format(CONTROLLER, joint)
        rospy.set_param(namespace + "/stiffness_factor", stiffness)
        rospy.set_param(namespace + "/damping_factor", damping)
        rospy.loginfo("%s: stiffness=%g damping=%g", joint, stiffness, damping)
    rospy.loginfo("Loaded grasp impedance gains from %s", env_path)

    load_name = "/controller_manager/load_controller"
    switch_name = "/controller_manager/switch_controller"
    unload_name = "/controller_manager/unload_controller"
    list_name = "/controller_manager/list_controllers"
    for service in (load_name, switch_name, unload_name, list_name):
        rospy.wait_for_service(service, timeout=30.0)
    load = rospy.ServiceProxy(load_name, LoadController)
    switch = rospy.ServiceProxy(switch_name, SwitchController)
    unload = rospy.ServiceProxy(unload_name, UnloadController)
    list_controllers = rospy.ServiceProxy(list_name, ListControllers)
    # The controller reads its gains and joint setup only when loaded, so a restart is needed
    # only when they changed. Stopping it leaves the arm without torque and it collapses, so
    # an unchanged controller from a previous run is reused and a changed one is replaced only
    # on request.
    params = rospy.get_param("/" + CONTROLLER)
    applied_name = "~applied_params"
    states = [c.state for c in list_controllers().controller if c.name == CONTROLLER]
    if states == ["running"] and rospy.get_param(applied_name, None) == params:
        rospy.loginfo("Reusing the running %s: its parameters are unchanged", CONTROLLER)
    else:
        if states:
            if states[0] == "running" and not rospy.get_param("~reload", False):
                raise RuntimeError(
                    "{} is still running from a previous run with other parameters. Restarting "
                    "it drops the arm, so support the arm and relaunch with "
                    "reload_controller:=true, or relaunch with the previous task.".format(CONTROLLER))
            rospy.logwarn("Reloading %s: the arm is unpowered until it restarts", CONTROLLER)
            release(switch, unload, states[0])
        if not load(CONTROLLER).ok:
            raise RuntimeError("could not load {}".format(CONTROLLER))
        if not switch([CONTROLLER], [], 2).ok:
            raise RuntimeError("could not start {}".format(CONTROLLER))
        rospy.set_param(applied_name, params)
    rospy.loginfo("Started %s with grasp training gains", CONTROLLER)
    with open(config_path, encoding="utf-8") as stream:
        expected = yaml.safe_load(stream).get("gravity_compensation")
    if expected is not None:
        running = "/gravity_compensation" in rosnode.get_node_names()
        if running != bool(expected):
            rospy.logwarn("the robot sets gravity_compensation: %s, but the gravity bridge is %s; "
                          "use the matching controllers launch file", expected,
                          "running" if running else "not running")
    # The controller stays running after shutdown to hold the arm; the next launch reuses it.
    rospy.spin()


if __name__ == "__main__":
    main()
