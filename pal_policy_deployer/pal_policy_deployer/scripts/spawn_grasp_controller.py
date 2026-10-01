#!/usr/bin/env python3
"""Set up the robot's joints with grasp actuator gains from the selected model artifact,
start control, and run the robot's gravity bridge.

The robot config names the joints under impedance control (impedance_joint_names), the jaws
under position control (position_joint_names) and, with gravity_compensation: true, the bridge
config (gravity_compensation_config), so one controllers launch file serves every robot.

PAL's DirectImpedanceControl is a plain joint PD sent as effort,
tau = stiffness_factor * (q_des - q) + damping_factor * (qd_des - qd) + tau_ff,
so MJLab's actuator stiffness and damping are copied without conversion.
"""

import math
import os
import re
import signal
import subprocess
import tempfile

import rosgraph
import rospy
import rospkg
import yaml
from task_config import resolve
from controller_manager_msgs.srv import (ListControllers, LoadController, SwitchController,
                                         UnloadController)


# Used when the config has no impedance_joint_names or position_joint_names: the single-arm
# grasp robots.
DEFAULT_JOINTS = ["arm_left_{}_joint".format(index) for index in range(1, 8)]
DEFAULT_POSITION_JOINTS = ["gripper_left_finger_joint"]
DEFAULT_GRAVITY_CONFIG = "$(find pal_gravity_compensation)/config/gravity_compensation.yaml"
CONTROLLER = "subscriber_controller"
BRIDGE = "gravity_compensation"

# PAL's per-joint launch files, as kangaroo_controller_configuration's arm and pelvis launch
# files include them: the actuator, safety and controller parameters of one joint.
JOINT_INCLUDE = """  <include file="$(find pal_local_joint_control)/launch/{kind}.launch">
    <arg name="joint_name" value="{joint}" />
    <arg name="controller_pkg_name" value="{package}" />
    <arg name="ns" value="{ns}" />
  </include>
"""


def joint_control_params(config):
    """The controller parameters of the robot's joints, which kangaroo_subscriber_controller.launch
    leaves in NoControl: arms and waist in DirectImpedanceControl, jaws in DirectPositionControl.
    roslaunch evaluates PAL's launch files, so the parameters are the ones they set."""
    includes = [JOINT_INCLUDE.format(kind="direct_impedance_control", joint=joint, ns=CONTROLLER,
                                     package="kangaroo_controller_configuration")
                for joint in config.get("impedance_joint_names", DEFAULT_JOINTS)]
    includes += [JOINT_INCLUDE.format(kind="direct_position_control", joint=joint, ns=CONTROLLER,
                                      package="pal_policy_deployer")
                 for joint in config.get("position_joint_names", DEFAULT_POSITION_JOINTS)]
    with tempfile.NamedTemporaryFile("w", suffix=".launch") as launch:
        launch.write("<launch>\n{}</launch>\n".format("".join(includes)))
        launch.flush()
        params = yaml.safe_load(subprocess.check_output(["roslaunch", "--dump-params", launch.name]))
    return params or {}


def find_path(path):
    """Expand $(find package) in a path; a relative path is relative to this package's config."""
    path = re.sub(r"\$\(find ([^)\s]+)\)", lambda match: rospkg.RosPack().get_path(match.group(1)),
                  path)
    if not os.path.isabs(path):
        path = os.path.join(rospkg.RosPack().get_path("pal_policy_deployer"), "config", path)
    return path


def check_no_commands(config):
    """Refuse to start while a policy or a home move still publishes joint targets. A policy
    left running while the bridge was down keeps integrating its deltas into a target the arm
    never follows; the new bridge or controller would then send the arm straight there."""
    topic = config.get("command_topic", "/pal_grasp/desired_state_uncompensated")
    publishers = [nodes for name, nodes in rosgraph.Master(rospy.get_name()).getSystemState()[0]
                  if name == topic]
    if publishers and publishers[0]:
        raise RuntimeError(
            "{} still publish joint targets on {}. Stop the policy or home move (supervisor "
            "~stop) before starting the controllers.".format(", ".join(publishers[0]), topic))


def start_gravity_bridge(config):
    """Start the gravity bridge with the robot's config; it adds the feedforward torque to the
    commands on its input_topic. Returns the process, which ends with this node."""
    path = find_path(config.get("gravity_compensation_config", DEFAULT_GRAVITY_CONFIG))
    with open(path, encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    if rospy.has_param("/" + BRIDGE):
        rospy.delete_param("/" + BRIDGE)
    rospy.set_param("/" + BRIDGE, params)
    rospy.loginfo("Starting the gravity bridge with %s", path)
    return subprocess.Popen(["rosrun", "pal_gravity_compensation", "gravity_compensation_node",
                             "__name:=" + BRIDGE])


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
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    check_no_commands(config)
    for name, value in joint_control_params(config).items():
        rospy.set_param(name, value)
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
    # A robot without gravity compensation runs no bridge: its policy commands the controller
    # directly, and the home move brings its own bridge (the robot's supervisor: home_launch).
    bridge = start_gravity_bridge(config) if config.get("gravity_compensation", True) else None
    # The controller stays running after shutdown to hold the arm; the next launch reuses it.
    # The bridge is required, as the node it replaces was in the launch files.
    rate = rospy.Rate(2.0)
    try:
        while not rospy.is_shutdown():
            if bridge is not None and bridge.poll() is not None:
                raise RuntimeError("the gravity bridge exited with code {}".format(bridge.returncode))
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        if bridge is not None and bridge.poll() is None:
            bridge.send_signal(signal.SIGINT)
            bridge.wait()


if __name__ == "__main__":
    main()
