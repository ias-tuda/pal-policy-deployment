#!/usr/bin/env python3
"""Start the leg and pelvis position controllers and move them to a stored pose.

kang_grasp_controllers.launch runs this with legs:=true, so the legs stand at the pose the
grasp policies were set up with while the subscriber controller drives the arms. The pose file
holds, per controller, the joint names and positions (config/leg_poses/default.yaml).

The controllers are PAL's JointTrajectoryControllers; the launch file loads their parameters
from kangaroo_controller_configuration. A controller that already runs is reused, and the
controllers keep running after this node ends, so the legs stay held.

Manual use (dry run unless --execute):
  rosrun pal_policy_deployer restore_leg_pose.py
  rosrun pal_policy_deployer restore_leg_pose.py --execute
  rosrun pal_policy_deployer restore_leg_pose.py --pose my_pose.yaml --execute
"""

import argparse
import os
import re
import sys

import actionlib
import rospkg
import rospy
import yaml
from control_msgs.msg import (FollowJointTrajectoryAction, FollowJointTrajectoryGoal,
                              JointTrajectoryControllerState)
from controller_manager_msgs.srv import ListControllers, LoadController, SwitchController
from diagnostic_msgs.msg import DiagnosticArray
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_SPEED = 0.1  # rad/s for revolute joints
LENGTH_SPEED = 0.005  # m/s for leg_*_length_joint
MIN_DURATION = 3.0
# Default of --max-jump: a stored pose further than this (rad, or m for the length joints) from
# the current one is refused; it means a wrong pose file or changed encoder zeros.
MAX_JUMP = 0.3


def default_pose():
    return os.path.join(rospkg.RosPack().get_path("pal_policy_deployer"), "config", "leg_poses",
                        "default.yaml")


def actuators_not_running():
    """Leg and pelvis actuators whose state machine is not Running, from the diagnostics."""
    try:
        message = rospy.wait_for_message("/diagnostics_agg", DiagnosticArray, timeout=10.0)
    except rospy.ROSException:
        return ["(no /diagnostics_agg)"]
    stopped = []
    for status in message.status:
        if status.name.startswith(("/Hardware/Motor/leg_", "/Hardware/Motor/pelvis_")):
            state = dict((value.key, value.value) for value in status.values).get("Actuator SM")
            if state != "Running":
                stopped.append("{}={}".format(status.name.split("/")[-1], state))
    return stopped


def start_controllers(names):
    """Load and start the controllers that do not run yet; running ones are reused."""
    for service in ("load_controller", "switch_controller", "list_controllers"):
        rospy.wait_for_service("/controller_manager/" + service, timeout=30.0)
    load = rospy.ServiceProxy("/controller_manager/load_controller", LoadController)
    switch = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
    states = dict((c.name, c.state) for c in rospy.ServiceProxy(
        "/controller_manager/list_controllers", ListControllers)().controller)
    for name in names:
        if name not in states and not load(name).ok:
            raise RuntimeError("could not load {}; are its parameters loaded?".format(name))
    to_start = [name for name in names if states.get(name) != "running"]
    if to_start and not switch(to_start, [], 2).ok:
        raise RuntimeError("could not start {}".format(", ".join(to_start)))
    rospy.loginfo("Leg controllers running (%s started, others reused)",
                  ", ".join(to_start) or "none")


def plan_motion(pose, selected, max_jump):
    """Per controller the joint names and targets, and a duration from the largest distance.
    Joints that selected rejects keep the position their controller commands now, so they do
    not move."""
    plan, duration = {}, MIN_DURATION
    for controller, stored in pose["controllers"].items():
        state = rospy.wait_for_message("/{}/state".format(controller),
                                       JointTrajectoryControllerState, timeout=10.0)
        names = list(state.joint_names)
        if sorted(names) != sorted(stored["joints"]):
            raise RuntimeError("{} joints {} differ from the stored pose {}".format(
                controller, names, stored["joints"]))
        stored_target = dict(zip(stored["joints"], stored["positions"]))
        current = dict(zip(names, state.actual.positions))
        held = dict(zip(names, state.desired.positions))
        target = dict((name, stored_target[name] if selected(name) else held[name])
                      for name in names)
        for name in names:
            if not selected(name):
                rospy.loginfo("  %-24s %+.4f held", name, held[name])
                continue
            delta = target[name] - current[name]
            length = "length" in name
            rospy.loginfo("  %-24s %+.4f -> %+.4f (%+.4f %s)", name, current[name], target[name],
                          delta * (1000.0 if length else 1.0), "mm" if length else "rad")
            if abs(delta) > max_jump:
                raise RuntimeError("{} is {:.3f} from the stored pose (limit {}); move it closer "
                                   "first or raise --max-jump".format(name, delta, max_jump))
            duration = max(duration, abs(delta) / (LENGTH_SPEED if length else JOINT_SPEED))
        plan[controller] = (names, [target[name] for name in names])
    return plan, duration


def execute(plan, duration):
    clients = {}
    for controller in plan:
        client = actionlib.SimpleActionClient(
            "/{}/follow_joint_trajectory".format(controller), FollowJointTrajectoryAction)
        if not client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError("{} action server not available".format(controller))
        clients[controller] = client
    # One start time, so the legs and the pelvis move together.
    start = rospy.Time.now() + rospy.Duration(0.5)
    for controller, (names, positions) in plan.items():
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = start
        goal.trajectory.joint_names = names
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=positions, velocities=[0.0] * len(positions),
            time_from_start=rospy.Duration(duration))]
        clients[controller].send_goal(goal)
    try:
        for client in clients.values():
            client.wait_for_result(rospy.Duration(duration + 10.0))
    except rospy.ROSInterruptException:
        for client in clients.values():
            client.cancel_goal()
        raise
    failed = [name for name, client in clients.items()
              if client.get_state() != actionlib.GoalStatus.SUCCEEDED]
    if failed:
        raise RuntimeError("move did not succeed for {}".format(", ".join(failed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose", default="", help="pose file (default config/leg_poses/default.yaml)")
    parser.add_argument("--execute", action="store_true", help="move (otherwise only print the plan)")
    parser.add_argument("--yes", action="store_true", help="do not ask before moving (for launch files)")
    parser.add_argument("--no-start", action="store_true", help="expect the controllers to run already")
    parser.add_argument("--joints", default="", help="regex; only joints matching it move to the "
                        "stored pose, the others are held (e.g. 'length')")
    parser.add_argument("--max-jump", type=float, default=MAX_JUMP,
                        help="refuse when a joint is further than this from the stored pose")
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("restore_leg_pose")

    pose_path = args.pose or default_pose()
    with open(pose_path, encoding="utf-8") as stream:
        pose = yaml.safe_load(stream)
    rospy.loginfo("Leg pose from %s (captured %s)", pose_path, pose.get("captured"))

    try:
        stopped = actuators_not_running()
        if stopped:
            raise RuntimeError("actuators not running: {}".format(", ".join(stopped)))
        # A dry run changes nothing, so it needs the controllers running already.
        if args.execute and not args.no_start:
            start_controllers(list(pose["controllers"]))
        plan, duration = plan_motion(pose, lambda name: re.search(args.joints, name) is not None,
                                     args.max_jump)
        rospy.loginfo("Move duration %.1f s", duration)
        if not args.execute:
            rospy.loginfo("Dry run, nothing sent (add --execute)")
            return
        if not args.yes and input("Move legs and pelvis? [y/N] ").strip().lower() != "y":
            return
        execute(plan, duration)
        rospy.loginfo("Legs and pelvis at the stored pose")
    except (RuntimeError, rospy.ROSException) as error:
        rospy.logerr("restore_leg_pose: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
