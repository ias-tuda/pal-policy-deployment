#!/usr/bin/env python3
"""A simulated policy supervisor, tracker and arm on ROS (voice_bridge/sim.py), to try the voice
bridge's ROS side in the dev container without the robot: roslaunch voice_bridge mock_robot.launch.

Never on the robot's master. It registers as the supervisor and publishes /tf and the
controller's joint state; it refuses a master on another PC and one where a supervisor or the
controllers run. In every terminal of the mock: export ROS_MASTER_URI=http://localhost:11311

Change the simulation with rosparam (read every 0.5 s), for example
  rosparam set /pal_policy_supervisor/objects_present false
  rosparam set /pal_policy_supervisor/arm_offset 0.3
  rosparam set /pal_policy_supervisor/home_fails true
  rosparam set /pal_policy_supervisor/tracker_running false
The start poses, frames and tracker objects come from pal_policy_deployer's task, robot and
start config files, read as the bridge reads them.
"""
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import rosgraph  # noqa: E402
import rosgraph.network  # noqa: E402
import rospkg  # noqa: E402
import rospy  # noqa: E402
import yaml  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from pal_policy_deployer.srv import SetTask, SetTaskResponse  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger, TriggerResponse  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from voice_bridge.configs import load_task_info  # noqa: E402
from voice_bridge.sim import SimSupervisor, Timing, sim_tasks  # noqa: E402
from voice_bridge.tasks import load_tasks  # noqa: E402

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "voice_tasks.yaml")
JOINT_STATE_TOPIC = "/subscriber_controller/actual_js_state"
ACTIVE_OBJECTS_TOPIC = "/tracking/active_objects_cfg"
REAL_NODES = ("/pal_policy_supervisor", "/subscriber_controller", "/spawn_grasp_controller")


def refuse_next_to_the_robot():
    host = urlparse(rosgraph.get_master_uri()).hostname
    if not host or not rosgraph.network.is_local_address(host):
        raise SystemExit("mock_supervisor: the master {} is not on this PC; "
                         "export ROS_MASTER_URI=http://localhost:11311".format(host))
    try:
        publishers, subscribers, services = rosgraph.Master("/voice_bridge_mock_check").getSystemState()
    except Exception as error:
        raise SystemExit("mock_supervisor: no ROS master ({})".format(error))
    nodes = {node for _, names in publishers + subscribers + services for node in names}
    topics = {topic for topic, _ in publishers}
    found = sorted(nodes.intersection(REAL_NODES)) + ([JOINT_STATE_TOPIC] if JOINT_STATE_TOPIC in topics else [])
    if found:
        raise SystemExit("mock_supervisor: {} runs on this master; the mock would disturb the robot".format(
            ", ".join(found)))


class RosSink(object):
    """Publishes the simulation as the supervisor, the tracker and the controller do."""

    def __init__(self):
        self._state = rospy.Publisher("~state", String, queue_size=1, latch=True)
        self._task = rospy.Publisher("~task", String, queue_size=1, latch=True)
        self._tasks = rospy.Publisher("~tasks", String, queue_size=1, latch=True)
        self._readiness = rospy.Publisher("~readiness", String, queue_size=1, latch=True)
        self._log = rospy.Publisher("~log", String, queue_size=100)
        self._tf = rospy.Publisher("/tf", TFMessage, queue_size=10)
        self._joints = rospy.Publisher(JOINT_STATE_TOPIC, JointState, queue_size=1)
        self._active = rospy.Publisher(ACTIVE_OBJECTS_TOPIC, String, queue_size=1, latch=True)

    def on_state(self, state):
        self._state.publish(String(state))

    def on_task(self, data):
        self._task.publish(String(data))

    def on_tasks(self, data):
        self._tasks.publish(String(data))

    def on_readiness(self, data):
        self._readiness.publish(String(data))

    def on_log(self, line):
        rospy.loginfo("%s", line)
        self._log.publish(String(line))

    def on_active_objects(self, path):
        self._active.publish(String(path))

    def on_tf(self, frames):
        stamp = rospy.Time.now()
        transforms = []
        for frame in frames:
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "base_link"
            transform.child_frame_id = frame
            transform.transform.rotation.w = 1.0
            transforms.append(transform)
        self._tf.publish(TFMessage(transforms))

    def on_joints(self, names, positions):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = list(names)
        message.position = list(positions)
        self._joints.publish(message)


def mock_tasks(names):
    """SimTasks with the robot, start pose, frames and tracker objects of pal_policy_deployer's
    files. A task whose files do not load keeps the simulation's data; the bridge refuses it."""
    config_dir = os.path.join(rospkg.RosPack().get_path("pal_policy_deployer"), "config")
    tasks = sim_tasks(names)
    for name, task in tasks.items():
        try:
            with open(os.path.join(config_dir, "tasks", name + ".yaml")) as stream:
                task.robot = (yaml.safe_load(stream) or {}).get("robot", task.robot)
            info = load_task_info(config_dir, name)
        except Exception as error:
            rospy.logwarn("mock: %s does not load (%s); it keeps the simulation's data", name, error)
            continue
        task.start_pose = info.start_pose
        task.object_frames = info.object_frames
        task.tracker_objects = info.tracker_objects
    return tasks


def main():
    # Before init_node: a node of the same name would shut a real supervisor down.
    refuse_next_to_the_robot()
    rospy.init_node("pal_policy_supervisor")
    names = load_tasks(rospy.get_param("~tasks_file", TASKS_FILE)).names()
    tasks = mock_tasks(names)
    task = rospy.get_param("~task", names[0])
    sim = SimSupervisor(RosSink(), tasks, task, Timing(float(rospy.get_param("~scale", 1.0))))
    rospy.Service("~set_task", SetTask, lambda request: SetTaskResponse(*sim.set_task(request.task)))
    for name in ("start", "stop", "go_home"):
        rospy.Service("~" + name, Trigger, lambda _, name=name: TriggerResponse(*getattr(sim, name)()))
    rospy.on_shutdown(sim.close)
    rospy.loginfo("mock supervisor runs %s; tasks: %s", task, ", ".join(names))
    offset = 0.0
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        for name in ("objects_present", "home_fails", "tracker_running"):
            setattr(sim, name, bool(rospy.get_param("~" + name, getattr(sim, name))))
        new_offset = float(rospy.get_param("~arm_offset", offset))
        if new_offset != offset:
            sim.move_arm(new_offset)
            offset = new_offset
        rate.sleep()


if __name__ == "__main__":
    main()
