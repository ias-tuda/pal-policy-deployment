"""The bridge's ROS side: the supervisor, tracker and joint topics into a RobotState, and the
supervisor's services. It runs in the dev container next to the supervisor, where
pal_policy_deployer (SetTask, the task and robot files) and the models are."""
import os

import rospkg
import rospy
from pal_policy_deployer.srv import SetTask
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from .configs import load_task_info


class RosIO(object):

    def __init__(self, state, supervisor, joint_state_topic, active_objects_topic, service_wait=2.0):
        self._state = state
        self._supervisor = supervisor.rstrip("/")
        self._service_wait = service_wait
        self._config_dir = os.path.join(rospkg.RosPack().get_path("pal_policy_deployer"), "config")
        # A second publisher on the supervisor's log: both phone pages show its lines.
        self._log_pub = rospy.Publisher(self._topic("log"), String, queue_size=20)
        subscriptions = (
            ("state", state.on_state, 1), ("task", state.on_task, 1), ("tasks", state.on_tasks, 1),
            ("readiness", state.on_readiness, 1), ("log", state.on_log, 200))
        for name, callback, queue_size in subscriptions:
            rospy.Subscriber(self._topic(name), String, _data(callback), queue_size=queue_size)
        rospy.Subscriber(joint_state_topic, JointState, self._on_joints, queue_size=1)
        rospy.Subscriber("/tf", TFMessage, self._on_tf, queue_size=100)
        rospy.Subscriber(active_objects_topic, String, _data(state.on_active_objects), queue_size=1)

    def _topic(self, name):
        return self._supervisor + "/" + name

    def _on_joints(self, message):
        if len(message.position) == len(message.name):
            self._state.on_joints(message.name, message.position)

    def _on_tf(self, message):
        self._state.on_tf([transform.child_frame_id for transform in message.transforms])

    def set_task(self, task):
        name = self._topic("set_task")
        rospy.wait_for_service(name, timeout=self._service_wait)
        response = rospy.ServiceProxy(name, SetTask)(task)
        return response.success, response.message

    def start(self):
        name = self._topic("start")
        rospy.wait_for_service(name, timeout=self._service_wait)
        response = rospy.ServiceProxy(name, Trigger)()
        return response.success, response.message

    def publish_log(self, line):
        rospy.loginfo("%s", line)
        self._log_pub.publish(String(line))

    def task_info(self, task, robot=None, current=False):
        """Read on every call: the supervisor rewrites its configs on each switch and restart."""
        # ~start_config_path applies to the supervisor's startup task only; set_task deletes it.
        start = rospy.get_param(self._topic("start_config_path"), "") if current else ""
        return load_task_info(self._config_dir, task, robot or "", start)


def _data(callback):
    return lambda message: callback(message.data)
