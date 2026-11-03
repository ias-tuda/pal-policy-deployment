#!/usr/bin/env python3
"""Set up the robot's joints with grasp actuator gains from the selected model artifact,
start control, and run the robot's gravity bridge.

The robot config names the joints under impedance control (impedance_joint_names), the jaws
(position_joint_names) and, with gravity_compensation: true, the bridge config
(gravity_compensation_config), so one controllers launch file serves every robot.

The jaws run either in subscriber_controller's DirectPositionControl (~jaw_control: position)
or, with ~jaw_control: current, under PAL's current-based gripper controllers
(config/gripper_current_controllers.yaml). subscriber_controller then leaves them in NoControl
and a joint_state_to_trajectory.py instance forwards their targets from the command topic.

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
import threading
import time

import dynamic_reconfigure.client
import rosgraph
import rospy
import rospkg
import yaml
from task_config import resolve
from controller_manager_msgs.srv import (ListControllers, LoadController, SwitchController,
                                         UnloadController)
from pal_policy_deployer.srv import SetTask, SetTaskResponse
from sensor_msgs.msg import JointState


# Used when the config has no impedance_joint_names or position_joint_names: the single-arm
# grasp robots.
DEFAULT_JOINTS = ["arm_left_{}_joint".format(index) for index in range(1, 8)]
DEFAULT_POSITION_JOINTS = ["gripper_left_finger_joint"]
DEFAULT_GRAVITY_CONFIG = "$(find pal_gravity_compensation)/config/gravity_compensation.yaml"
CONTROLLER = "subscriber_controller"
BRIDGE = "gravity_compensation"
JAW_BRIDGE = "jaw_trajectory_bridge"
# PAL's current-based gripper controllers per jaw joint (jaw_control: current), and the
# position controllers of PAL's default_controllers.launch that also claim the jaws.
JAW_CURRENT_CONTROLLERS = {
    "gripper_left_finger_joint": "gripper_left_current_controller",
    "gripper_right_finger_joint": "gripper_right_current_controller",
}
JAW_CURRENT_CONFIG = "gripper_current_controllers.yaml"
# A task switch moves the impedance gains to the new model's in this time and number of steps
# (G2 and G3 differ by a factor of 10-20), and waits this long for a new gravity bridge.
GAIN_RAMP_TIME = 2.0  # s
GAIN_RAMP_STEPS = 20
BRIDGE_START_TIMEOUT = 5.0  # s
PAL_JAW_CONTROLLERS = ["gripper_left_controller", "gripper_right_controller"]

# PAL's per-joint launch files, as kangaroo_controller_configuration's arm and pelvis launch
# files include them: the actuator, safety and controller parameters of one joint.
JOINT_INCLUDE = """  <include file="$(find pal_local_joint_control)/launch/{kind}.launch">
    <arg name="joint_name" value="{joint}" />
    <arg name="controller_pkg_name" value="{package}" />
    <arg name="ns" value="{ns}" />
  </include>
"""


def jaw_joints(config):
    return config.get("position_joint_names", DEFAULT_POSITION_JOINTS)


def joint_control_params(config, jaw_control):
    """The controller parameters of the robot's joints, which kangaroo_subscriber_controller.launch
    leaves in NoControl: arms and waist in DirectImpedanceControl, jaws in DirectPositionControl
    (or left in NoControl for jaw_control current). roslaunch evaluates PAL's launch files, so
    the parameters are the ones they set."""
    includes = [JOINT_INCLUDE.format(kind="direct_impedance_control", joint=joint, ns=CONTROLLER,
                                     package="kangaroo_controller_configuration")
                for joint in config.get("impedance_joint_names", DEFAULT_JOINTS)]
    if jaw_control == "position":
        includes += [JOINT_INCLUDE.format(kind="direct_position_control", joint=joint,
                                          ns=CONTROLLER, package="pal_policy_deployer")
                     for joint in jaw_joints(config)]
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


def jaw_current_controllers(config):
    unknown = [joint for joint in jaw_joints(config) if joint not in JAW_CURRENT_CONTROLLERS]
    if unknown:
        raise ValueError("no current-based gripper controller for {}".format(", ".join(unknown)))
    return {JAW_CURRENT_CONTROLLERS[joint]: [joint] for joint in jaw_joints(config)}


def controller_states(list_controllers):
    return {c.name: c.state for c in list_controllers().controller}


def stop_controllers(switch, list_controllers, names):
    running = [name for name in names if controller_states(list_controllers).get(name) == "running"]
    if running:
        rospy.loginfo("Stopping %s", ", ".join(running))
        if not switch([], running, 2).ok:
            raise RuntimeError("could not stop {}".format(", ".join(running)))


def start_jaw_controllers(config, load, switch, list_controllers):
    """Start PAL's current-based gripper controllers for the robot's jaws, after stopping PAL's
    default position gripper controllers, which claim the same joints. Running ones are reused;
    they hold the jaw at its last target, also after this node ends."""
    controllers = jaw_current_controllers(config)
    with open(find_path(JAW_CURRENT_CONFIG), encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    stop_controllers(switch, list_controllers, PAL_JAW_CONTROLLERS)
    states = controller_states(list_controllers)
    for name in controllers:
        if name not in states:
            rospy.set_param("/" + name, params[name])
            if not load(name).ok:
                raise RuntimeError("could not load {}".format(name))
    start = [name for name in controllers if states.get(name) != "running"]
    if start and not switch(start, [], 2).ok:
        raise RuntimeError(
            "could not start {} next to {}; if it still claims the jaws in NoControl, drop "
            "them from its joints".format(", ".join(start), CONTROLLER))
    rospy.loginfo("Jaws on %s", ", ".join(controllers))
    return controllers


def start_jaw_bridge(controllers):
    """Forward the jaw targets to the current-based gripper controllers. Every command reaches
    subscriber_controller's input (the policy directly or through a gravity bridge, which
    passes the jaw positions on, and the home moves), so the bridge listens there. Returns the
    process, which ends with this node."""
    namespace = "/" + JAW_BRIDGE
    if rospy.has_param(namespace):
        rospy.delete_param(namespace)
    rospy.set_param(namespace + "/controllers", controllers)
    rospy.set_param(namespace + "/input_topic", "/{}/desired_state".format(CONTROLLER))
    return subprocess.Popen(["rosrun", "pal_policy_deployer", "joint_state_to_trajectory.py",
                             "__name:=" + JAW_BRIDGE])


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


def set_gain_params(gains):
    for joint, (stiffness, damping) in gains.items():
        namespace = "/{}/joints/{}/impedance_params".format(CONTROLLER, joint)
        rospy.set_param(namespace + "/stiffness_factor", stiffness)
        rospy.set_param(namespace + "/damping_factor", damping)


def other_publishers(topic, allowed):
    publishers = dict(rosgraph.Master(rospy.get_name()).getSystemState()[0]).get(topic, [])
    return [node for node in publishers if node not in allowed]


class TaskSwitcher(object):
    """Switches the running controllers to another task's robot and model without restarting
    subscriber_controller (~set_task). The robots that can share it differ only in the
    impedance gains, which DirectImpedanceControl takes live through dynamic_reconfigure, and
    in the gravity bridge, which runs for gravity_compensation: true only. The joints under
    impedance and position control must be the same; anything else needs a relaunch with
    reload_controller:=true.

    To the gravity bridge (G3 -> G2): start the bridge, give it the measured arm pose as its
    target so it holds the arm with the gravity torque, then ramp the gains down. From it
    (G2 -> G3): ramp the gains up, then stop the bridge; the controller holds the last
    compensated command, as after the home move of a robot without the bridge."""

    def __init__(self, config_path, config, gains, bridge):
        self._config_path = config_path
        self._config = config
        self._gains = gains
        self._bridge = bridge
        self._lock = threading.Lock()
        self._publish_state()
        rospy.Service("~set_task", SetTask, self._handle)

    def _publish_state(self):
        rospy.set_param("~config_path", self._config_path)
        rospy.set_param("~gravity_compensation", self._bridge is not None)

    def check_bridge(self):
        with self._lock:
            if self._bridge is not None and self._bridge.poll() is not None:
                raise RuntimeError("the gravity bridge exited with code {}".format(
                    self._bridge.returncode))

    def shutdown(self):
        with self._lock:
            self._stop_bridge()

    def _handle(self, request):
        with self._lock:
            try:
                message = self._switch(request.task)
            except Exception as error:  # Every failure goes back to the caller.
                rospy.logerr("set_task %s: %s", request.task, error)
                return SetTaskResponse(False, str(error))
        rospy.loginfo("set_task: %s", message)
        return SetTaskResponse(True, message)

    def _switch(self, task_name):
        task, config_path = resolve(task_name)
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        robot = task["robot"]
        for key, default in (("impedance_joint_names", DEFAULT_JOINTS),
                             ("position_joint_names", DEFAULT_POSITION_JOINTS)):
            if config.get(key, default) != self._config.get(key, default):
                raise RuntimeError(
                    "{} of {} differs from the running controllers; relaunch "
                    "kang_grasp_controllers.launch task:={} reload_controller:=true".format(
                        key, robot, task["name"]))
        new_bridge = config.get("gravity_compensation", True)
        if (new_bridge and self._bridge is not None
                and config.get("gravity_compensation_config", DEFAULT_GRAVITY_CONFIG)
                != self._config.get("gravity_compensation_config", DEFAULT_GRAVITY_CONFIG)):
            raise RuntimeError("{} needs another gravity bridge config; relaunch "
                               "kang_grasp_controllers.launch task:={} reload_controller:=true"
                               .format(robot, task["name"]))
        # Our own bridge publishes on the controller input; a policy or a home move must not.
        topics = {self._config.get("command_topic", "/pal_grasp/desired_state_uncompensated"),
                  config.get("command_topic", "/pal_grasp/desired_state_uncompensated"),
                  "/{}/desired_state".format(CONTROLLER)}
        for topic in topics:
            publishers = other_publishers(topic, ["/" + BRIDGE])
            if publishers:
                raise RuntimeError("{} still publish on {}; stop the policy or home move first"
                                   .format(", ".join(publishers), topic))
        _, gains = load_gains(config_path)
        steps = []
        started_bridge = new_bridge and self._bridge is None
        if started_bridge:
            self._start_holding_bridge(config)
            steps.append("started the gravity bridge")
        if gains != self._gains:
            try:
                self._ramp(self._gains, gains)
            except Exception:
                if started_bridge:
                    # Back to the previous gains, which hold the arm without the bridge.
                    self._stop_bridge()
                raise
            steps.append("ramped the gains in {:.0f} s".format(GAIN_RAMP_TIME))
        if not new_bridge and self._bridge is not None:
            self._stop_bridge()
            steps.append("stopped the gravity bridge")
        self._config_path, self._config, self._gains = config_path, config, gains
        self._publish_state()
        # A relaunch with this task then reuses the running controller.
        rospy.set_param("~applied_params", rospy.get_param("/" + CONTROLLER))
        return "{} ({}): {}".format(task["name"], robot, ", ".join(steps) or "controllers unchanged")

    def _start_holding_bridge(self, config):
        """Start the bridge and make it hold the measured arm pose with the gravity torque.
        The bridge publishes nothing until it has a target, so the arm keeps its last command
        until the bridge output is confirmed."""
        self._bridge = start_gravity_bridge(config)
        try:
            input_topic = rospy.get_param("/{}/input_topic".format(BRIDGE))
            output_topic = rospy.get_param("/{}/output_topic".format(BRIDGE))
            joints = rospy.get_param("/{}/arm_joint_names".format(BRIDGE))
            state = rospy.wait_for_message("/joint_states", JointState, timeout=2.0)
            positions = dict(zip(state.name, state.position))
            missing = [joint for joint in joints if joint not in positions]
            if missing:
                raise RuntimeError("/joint_states has no {}".format(", ".join(missing)))
            hold = JointState(name=joints, position=[positions[joint] for joint in joints],
                              velocity=[0.0] * len(joints))
            publisher = rospy.Publisher(input_topic, JointState, queue_size=1)
            try:
                deadline = time.monotonic() + BRIDGE_START_TIMEOUT
                while publisher.get_num_connections() == 0:
                    if time.monotonic() > deadline or self._bridge.poll() is not None:
                        raise RuntimeError("the gravity bridge did not subscribe to " + input_topic)
                    time.sleep(0.05)
                publisher.publish(hold)
                # The move to the start pose refuses other publishers on the bridge input.
                rospy.wait_for_message(output_topic, JointState, timeout=BRIDGE_START_TIMEOUT)
            finally:
                publisher.unregister()
        except Exception:
            self._stop_bridge()
            raise

    def _stop_bridge(self):
        bridge, self._bridge = self._bridge, None
        if bridge is not None and bridge.poll() is None:
            bridge.send_signal(signal.SIGINT)
            bridge.wait()

    def _ramp(self, start, end):
        """Move every joint's gains linearly from start to end through dynamic_reconfigure.
        On a failure the start gains are set back."""
        clients = {joint: dynamic_reconfigure.client.Client(
            "/{}/joints/{}/impedance_parameters".format(CONTROLLER, joint), timeout=5.0)
            for joint in end}

        def apply(gains):
            for joint, (stiffness, damping) in gains.items():
                clients[joint].update_configuration(
                    {"stiffness_factor": stiffness, "damping_factor": damping})
            set_gain_params(gains)

        try:
            for step in range(1, GAIN_RAMP_STEPS + 1):
                fraction = float(step) / GAIN_RAMP_STEPS
                apply({joint: tuple(a + fraction * (b - a) for a, b in zip(start[joint], end[joint]))
                       for joint in end})
                time.sleep(GAIN_RAMP_TIME / GAIN_RAMP_STEPS)
        except Exception as error:
            rospy.logerr("Gain ramp failed (%s); setting the previous gains back", error)
            apply(start)
            raise RuntimeError("gain ramp failed: {}".format(error))


def main():
    rospy.init_node("spawn_grasp_controller")
    config_path = rospy.get_param("~config", "") or resolve(
        rospy.get_param("~task", ""), rospy.get_param("~robot", ""))[1]
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    check_no_commands(config)
    jaw_control = rospy.get_param("~jaw_control", "position")
    if jaw_control not in ("position", "current"):
        raise ValueError("~jaw_control must be position or current, not {}".format(jaw_control))
    for name, value in joint_control_params(config, jaw_control).items():
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
    if jaw_control == "position":
        # DirectPositionControl in subscriber_controller and the current controllers would
        # both command the jaws.
        stop_controllers(switch, list_controllers, list(JAW_CURRENT_CONTROLLERS.values()))
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
    jaw_bridge = None
    if jaw_control == "current":
        jaw_bridge = start_jaw_bridge(start_jaw_controllers(config, load, switch,
                                                                    list_controllers))
    # A robot without gravity compensation runs no bridge: its policy commands the controller
    # directly, and the home move brings its own bridge (the robot's supervisor: home_launch).
    bridge = start_gravity_bridge(config) if config.get("gravity_compensation", True) else None
    switcher = TaskSwitcher(config_path, config, gains, bridge)
    # The controller stays running after shutdown to hold the arm; the next launch reuses it.
    # The bridge is required, as the node it replaces was in the launch files.
    rate = rospy.Rate(2.0)
    try:
        while not rospy.is_shutdown():
            switcher.check_bridge()
            if jaw_bridge is not None and jaw_bridge.poll() is not None:
                raise RuntimeError("the jaw trajectory bridge exited with code {}".format(
                    jaw_bridge.returncode))
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        switcher.shutdown()
        if jaw_bridge is not None and jaw_bridge.poll() is None:
            jaw_bridge.send_signal(signal.SIGINT)
            jaw_bridge.wait()


if __name__ == "__main__":
    main()
