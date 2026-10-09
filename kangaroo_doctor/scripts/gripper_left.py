#!/usr/bin/env python3
"""Open or close the left gripper, and tell whether it really moved.

  rosrun kangaroo_doctor gripper_left.py close
  rosrun kangaroo_doctor gripper_left.py open
  rosrun kangaroo_doctor gripper_left.py 0.5            # fraction closed: 0 open, 1 closed
  rosrun kangaroo_doctor gripper_left.py close --duration 3

It uses whichever controller owns gripper_left_finger_joint:
  - gripper_left_controller (default controllers): a FollowJointTrajectory goal.
  - subscriber_controller (pal_policy_deployer kang_grasp_controllers.launch): a ramp on the
    gravity bridge input, with the arm joints held at the targets the bridge sends now. Refused
    while a policy or a home move publishes there.
Open and closed are the gripper_open_pos / gripper_closed_pos of pal_policy_deployer's
config/robots/left_arm_gravcomp.yaml.

Exit code: 0 target reached, 1 error, 2 stopped partway (holding an object, or blocked),
3 the gripper did not move (a fault: see the printed diagnosis).
"""

import argparse
import sys

import actionlib
import rosgraph
import rospy
from control_msgs.msg import (FollowJointTrajectoryAction, FollowJointTrajectoryGoal,
                              FollowJointTrajectoryResult, JointTrajectoryControllerState)
from controller_manager_msgs.srv import ListControllers
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT = "gripper_left_finger_joint"
MOTOR_DIAGNOSTICS = "/Hardware/Motor/gripper_left_finger_motor"
OPEN, CLOSED = 0.048, 0.902
BRIDGE_INPUT = "/pal_grasp/desired_state_uncompensated"
BRIDGE_OUTPUT = "/subscriber_controller/desired_state"
RATE = 50.0
REACHED = 0.05         # rad from the target
NOT_MOVED = 0.2        # moved less than this fraction of the way: a fault, no object is that wide

EXIT_REACHED, EXIT_ERROR, EXIT_PARTWAY, EXIT_NOT_MOVED = 0, 1, 2, 3
TRAJECTORY_ERRORS = {v: k for k, v in vars(FollowJointTrajectoryResult).items()
                     if k.isupper() and isinstance(v, int)}
GOAL_STATES = {v: k for k, v in vars(actionlib.GoalStatus).items()
               if k.isupper() and isinstance(v, int)}


def target_position(value):
    if value == "open":
        return OPEN
    if value == "close":
        return CLOSED
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction closed must be in [0, 1]")
    return OPEN + fraction * (CLOSED - OPEN)


def gripper_state(controller):
    """(actual, desired) from the owning controller's state topics, so it works without
    joint_state_controller. desired is what the controller commands the motor now."""
    if controller == "gripper_left_controller":
        state = rospy.wait_for_message("/gripper_left_controller/state",
                                       JointTrajectoryControllerState, timeout=5.0)
        index = list(state.joint_names).index(JOINT)
        return state.actual.positions[index], state.desired.positions[index]
    actual = rospy.wait_for_message("/subscriber_controller/actual_js_state", JointState, timeout=5.0)
    desired = rospy.wait_for_message(BRIDGE_OUTPUT, JointState, timeout=5.0)
    return (actual.position[list(actual.name).index(JOINT)],
            desired.position[list(desired.name).index(JOINT)])


def gripper_effort():
    """Effort from /joint_states, nan without joint_state_controller."""
    try:
        state = rospy.wait_for_message("/joint_states", JointState, timeout=1.0)
        return state.effort[list(state.name).index(JOINT)]
    except (rospy.ROSException, ValueError, IndexError):
        return float("nan")


def drive_diagnostics():
    """'level message; key: value, ...' of the gripper drive from /diagnostics_agg, or None."""
    try:
        array = rospy.wait_for_message("/diagnostics_agg", DiagnosticArray, timeout=3.0)
    except rospy.ROSException:
        return None
    for status in array.status:
        if status.name == MOTOR_DIAGNOSTICS:
            level = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}.get(status.level, "?")
            keys = ("Driver state", "Error code", "Motor errors", "Board errors", "Emergency stop",
                    "Actual control mode", "Motor temperature")
            values = ", ".join("%s: %s" % (v.key, v.value) for v in status.values if v.key in keys)
            return "%s %s; %s" % (level, status.message.strip() or "(no message)", values)
    return None


def owner():
    rospy.wait_for_service("/controller_manager/list_controllers", timeout=10.0)
    for controller in rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)().controller:
        if controller.state != "running":
            continue
        if any(JOINT in resource.resources for resource in controller.claimed_resources):
            return controller.name
    return None


def via_trajectory_controller(target, duration):
    """Returns a description of the action result."""
    client = actionlib.SimpleActionClient("/gripper_left_controller/follow_joint_trajectory",
                                          FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        raise RuntimeError("gripper_left_controller action server not available")
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = [JOINT]
    goal.trajectory.points = [JointTrajectoryPoint(positions=[target], velocities=[0.0],
                                                   time_from_start=rospy.Duration(duration))]
    client.send_goal(goal)
    if not client.wait_for_result(rospy.Duration(duration + 3.0)):
        client.cancel_goal()
        return "no result after %.0f s" % (duration + 3.0)
    result = client.get_result()
    text = GOAL_STATES.get(client.get_state(), str(client.get_state()))
    if result is not None and result.error_code:
        text += " " + TRAJECTORY_ERRORS.get(result.error_code, str(result.error_code))
        if result.error_string:
            text += " (%s)" % result.error_string
    return text


def via_gravity_bridge(target, duration):
    publishers = [nodes for name, nodes in rosgraph.Master(rospy.get_name()).getSystemState()[0]
                  if name == BRIDGE_INPUT]
    if publishers and publishers[0]:
        raise RuntimeError("{} already publish on {}; stop the policy or home move first".format(
            ", ".join(publishers[0]), BRIDGE_INPUT))
    # The bridge repeats its last input, so its output holds the current arm targets.
    current = rospy.wait_for_message(BRIDGE_OUTPUT, JointState, timeout=5.0)
    names, positions = list(current.name), list(current.position)
    index = names.index(JOINT)
    start = positions[index]
    publisher = rospy.Publisher(BRIDGE_INPUT, JointState, queue_size=1)
    rospy.sleep(0.5)  # let the bridge connect
    steps = max(1, int(duration * RATE))
    rate = rospy.Rate(RATE)
    for step in range(1, steps + 1 + int(RATE // 2)):  # half a second more at the target
        command = list(positions)
        command[index] = start + (target - start) * min(1.0, float(step) / steps)
        message = JointState(name=names, position=command, velocity=[0.0] * len(names))
        message.header.stamp = rospy.Time.now()
        publisher.publish(message)
        rate.sleep()
    return "ramp sent"


def diagnose_not_moved(controller, result, desired, target):
    """Why the gripper did not move: printed for the user."""
    rospy.logerr("FAULT: the gripper did not move (it is not just holding an object).")
    rospy.logerr("  controller %s: %s", controller, result)
    if abs(desired - target) < REACHED:
        rospy.logerr("  the controller commands %.3f, but the motor stays put: the drive does not "
                     "follow (drive error, e-stop, current limit, or a jammed finger).", desired)
    else:
        rospy.logerr("  the controller itself commands %.3f, not the target: the goal was "
                     "aborted or never executed.", desired)
    drive = drive_diagnostics()
    rospy.logerr("  drive (slave 31): %s", drive or "no diagnostics on /diagnostics_agg")
    rospy.logerr("  fix: e-stop power cycle (hold 2-5 s, release, re-arm), then restart the "
                 "deployer. The kangaroo_doctor page walks you through it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", help="open, close, or the fraction closed in [0, 1]")
    parser.add_argument("--duration", type=float, default=2.0, help="motion time in s")
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("gripper_left", anonymous=True)
    try:
        target = target_position(args.command)
        controller = owner()
        before, _ = gripper_state(controller)
        rospy.loginfo("%s %.4f -> %.4f via %s", JOINT, before, target, controller)
        if controller == "gripper_left_controller":
            result = via_trajectory_controller(target, args.duration)
        elif controller == "subscriber_controller":
            result = via_gravity_bridge(target, args.duration)
        else:
            raise RuntimeError("no running controller owns {}".format(JOINT))
        rospy.sleep(0.5)
        after, desired = gripper_state(controller)
        rospy.loginfo("%s now %.4f (commanded %.4f, effort %.2f; %s)", JOINT, after, desired,
                      gripper_effort(), result)
    except (RuntimeError, ValueError, rospy.ROSException) as error:
        rospy.logerr("gripper_left: %s", error)
        return EXIT_ERROR
    if abs(after - target) <= REACHED:
        rospy.loginfo("OK: target reached")
        return EXIT_REACHED
    way = target - before
    if abs(way) > REACHED and (after - before) / way < NOT_MOVED:
        diagnose_not_moved(controller, result, desired, target)
        return EXIT_NOT_MOVED
    rospy.logwarn("stopped %.3f short of the target: holding an object (a cube stalls the jaw at "
                  "~0.48), or blocked", target - after)
    return EXIT_PARTWAY


if __name__ == "__main__":
    sys.exit(main())
