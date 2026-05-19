#!/usr/bin/env python3
"""
Publish one constant left-arm reaching target and print basic distance debug.

This node intentionally does not sample poses or publish markers. It sends the
raw training-frame pose expected by the policy on /reaching/left_pose and uses
TF only for human-readable distance checks in ROS base_link coordinates.
"""

import math

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


def get_param(name, default):
    return rospy.get_param("~" + name, default)


def train_to_ros_position(x, y, z):
    # Convention used by the existing debug tools:
    # rviz_x = training_y, rviz_y = -training_x, rviz_z = training_z
    return y, -x, z


class SimpleLeftReachDebug:
    def __init__(self):
        self.publish_topic = get_param("publish_topic", "/reaching/left_pose")
        self.base_frame = get_param("base_frame", "imu_link")
        self.ee_frame = get_param("ee_frame", "arm_left_7_link")
        self.rate_hz = float(get_param("rate", 1.0))

        self.x = float(get_param("x", -0.53))
        self.y = float(get_param("y", -0.03))
        self.z = float(get_param("z", 0.05))
        self.qw = float(get_param("qw", 1.0))
        self.qx = float(get_param("qx", 0.0))
        self.qy = float(get_param("qy", 0.0))
        self.qz = float(get_param("qz", 0.0))

        self.pub = rospy.Publisher(self.publish_topic, PoseStamped, queue_size=1)
        self.listener = tf.TransformListener()

        self.last_policy_action = None
        self.last_desired_state = None
        rospy.Subscriber("/policy_output/actions_scaled", JointState, self.policy_action_cb, queue_size=1)
        rospy.Subscriber("/subscriber_controller/desired_state", JointState, self.desired_state_cb, queue_size=1)

        self.target_ros = train_to_ros_position(self.x, self.y, self.z)

        rospy.logwarn("=" * 70)
        rospy.logwarn("[simple_left_reach_debug] BASIC LEFT REACH TEST")
        rospy.logwarn("[simple_left_reach_debug] Publishing raw policy target:")
        rospy.logwarn(
            "  topic=%s frame_id=%s train_xyz=[%.3f %.3f %.3f] quat_wxyz=[%.3f %.3f %.3f %.3f]",
            self.publish_topic,
            self.base_frame,
            self.x,
            self.y,
            self.z,
            self.qw,
            self.qx,
            self.qy,
            self.qz,
        )
        rospy.logwarn(
            "[simple_left_reach_debug] For ROS/TF debug this target is approx xyz=[%.3f %.3f %.3f] in %s",
            self.target_ros[0],
            self.target_ros[1],
            self.target_ros[2],
            self.base_frame,
        )
        rospy.logwarn("[simple_left_reach_debug] EE TF: %s -> %s", self.base_frame, self.ee_frame)
        rospy.logwarn("=" * 70)

    def policy_action_cb(self, msg):
        self.last_policy_action = msg

    def desired_state_cb(self, msg):
        self.last_desired_state = msg

    def make_pose(self):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = self.z
        msg.pose.orientation.w = self.qw
        msg.pose.orientation.x = self.qx
        msg.pose.orientation.y = self.qy
        msg.pose.orientation.z = self.qz
        return msg

    def print_action_debug(self):
        if self.last_policy_action is None:
            rospy.logwarn_throttle(2.0, "[simple_left_reach_debug] No /policy_output/actions_scaled yet")
            return

        names = self.last_policy_action.name
        positions = self.last_policy_action.position
        left = [(n, p) for n, p in zip(names, positions) if n.startswith("arm_left_")]
        if not left:
            rospy.logwarn_throttle(2.0, "[simple_left_reach_debug] actions_scaled has no arm_left_* joints")
            return

        compact = " ".join(["%s=%+.3f" % (n.replace("arm_left_", "L"), p) for n, p in left])
        rospy.loginfo_throttle(2.0, "[simple_left_reach_debug] actions_scaled left: %s", compact)

    def print_distance_debug(self):
        try:
            self.listener.waitForTransform(
                self.base_frame,
                self.ee_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
            trans, rot = self.listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0,
                "[simple_left_reach_debug] No TF %s -> %s yet: %s",
                self.base_frame,
                self.ee_frame,
                exc,
            )
            return

        dx = self.target_ros[0] - trans[0]
        dy = self.target_ros[1] - trans[1]
        dz = self.target_ros[2] - trans[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        rospy.logwarn_throttle(
            1.0,
            "[simple_left_reach_debug] target_ros=[%+.3f %+.3f %+.3f] ee=[%+.3f %+.3f %+.3f] "
            "delta=[%+.3f %+.3f %+.3f] dist=%.3f m",
            self.target_ros[0],
            self.target_ros[1],
            self.target_ros[2],
            trans[0],
            trans[1],
            trans[2],
            dx,
            dy,
            dz,
            dist,
        )

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.pub.publish(self.make_pose())
            rospy.loginfo_throttle(
                2.0,
                "[simple_left_reach_debug] publishing %s at %.1f Hz",
                self.publish_topic,
                self.rate_hz,
            )
            self.print_distance_debug()
            self.print_action_debug()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("simple_left_reach_debug")
    SimpleLeftReachDebug().spin()
