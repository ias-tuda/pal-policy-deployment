#!/usr/bin/env python3

import math
import random

import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


LEFT_RANGES = {
    "x": (-0.75, -0.35),
    "y": (-0.30, 0.25),
    "z": (0.05, 0.32),
    "roll": (0.10, 0.80),
    "pitch": (-0.60, 0.10),
    "yaw": (0.00, 0.80),
}

RIGHT_RANGES = {
    "x": (0.35, 0.75),
    "y": (-0.30, 0.25),
    "z": (0.05, 0.32),
    "roll": (0.20, 0.90),
    "pitch": (-0.30, 0.30),
    "yaw": (-0.30, 0.50),
}


def uniform_pair(values):
    return random.uniform(float(values[0]), float(values[1]))


def quat_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qw, qx, qy, qz


def sample_pose(ranges, frame_id):
    x = uniform_pair(ranges["x"])
    y = uniform_pair(ranges["y"])
    z = uniform_pair(ranges["z"])
    roll = uniform_pair(ranges["roll"])
    pitch = uniform_pair(ranges["pitch"])
    yaw = uniform_pair(ranges["yaw"])
    qw, qx, qy, qz = quat_from_rpy(roll, pitch, yaw)

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = qw
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    return msg, (x, y, z, roll, pitch, yaw)


def transformed_preview_pose(msg, frame_id, axis_convention, offset):
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z
    if axis_convention == "mujoco_to_ros_base":
        x, y, z = y, -x, z
    elif axis_convention != "raw":
        rospy.logwarn_once(
            "Unknown preview axis convention '%s'. Use 'mujoco_to_ros_base' or 'raw'. Falling back to raw.",
            axis_convention,
        )

    out = PoseStamped()
    out.header.frame_id = frame_id
    out.header.stamp = msg.header.stamp
    out.pose.position.x = x + offset[0]
    out.pose.position.y = y + offset[1]
    out.pose.position.z = z + offset[2]
    out.pose.orientation = msg.pose.orientation
    return out


def make_pose_marker(msg, ns, marker_id, color):
    marker = Marker()
    marker.header = msg.header
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose = msg.pose
    marker.scale.x = 0.18
    marker.scale.y = 0.035
    marker.scale.z = 0.035
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.lifetime = rospy.Duration(0.0)
    return marker


def publish_preview(preview_pub, left_msg, right_msg, preview_frame_id,
                    preview_axis_convention, left_offset, right_offset):
    markers = MarkerArray()
    stamp = rospy.Time.now()
    if left_msg is not None:
        left_msg.header.stamp = stamp
        preview_msg = transformed_preview_pose(left_msg, preview_frame_id, preview_axis_convention, left_offset)
        markers.markers.append(make_pose_marker(preview_msg, "reaching_preview_left", 0, (1.0, 0.25, 0.25, 1.0)))
    if right_msg is not None:
        right_msg.header.stamp = stamp
        preview_msg = transformed_preview_pose(right_msg, preview_frame_id, preview_axis_convention, right_offset)
        markers.markers.append(make_pose_marker(preview_msg, "reaching_preview_right", 1, (0.25, 0.45, 1.0, 1.0)))
    preview_pub.publish(markers)


def publish_pose_commands(left_pub, right_pub, left_msg, right_msg, publish_rate, duration):
    rate = rospy.Rate(publish_rate)
    end_time = rospy.Time.now() + rospy.Duration(duration)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        stamp = rospy.Time.now()
        if left_msg is not None:
            left_msg.header.stamp = stamp
            left_pub.publish(left_msg)
        if right_msg is not None:
            right_msg.header.stamp = stamp
            right_pub.publish(right_msg)
        rate.sleep()


def wait_for_enter(prompt):
    try:
        input(prompt)
        return True
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    rospy.init_node("reaching_pose_command_sampler")

    side = rospy.get_param("~side", "both")
    frame_id = rospy.get_param("~frame_id", "imu_link")
    publish_rate = float(rospy.get_param("~publish_rate", 5.0))
    resample_min = float(rospy.get_param("~resample_min", 5.0))
    resample_max = float(rospy.get_param("~resample_max", 10.0))
    confirm = bool(rospy.get_param("~confirm", False))
    preview_frame_id = rospy.get_param("~preview_frame_id", frame_id)
    preview_axis_convention = rospy.get_param("~preview_axis_convention", "mujoco_to_ros_base")
    preview_root_offset = rospy.get_param("~preview_root_offset", [0.0, 0.0, 0.0])
    preview_left_offset = rospy.get_param("~preview_left_offset", [0.0, 0.15, -0.4])
    preview_right_offset = rospy.get_param("~preview_right_offset", [0.0, -0.15, -0.4])
    seed = int(rospy.get_param("~seed", 0))

    if seed != 0:
        random.seed(seed)

    if side not in ("left", "right", "both"):
        raise rospy.ROSException("~side must be 'left', 'right', or 'both'")
    if len(preview_root_offset) != 3:
        raise rospy.ROSException("~preview_root_offset must be [x, y, z]")
    if len(preview_left_offset) != 3:
        raise rospy.ROSException("~preview_left_offset must be [x, y, z]")
    if len(preview_right_offset) != 3:
        raise rospy.ROSException("~preview_right_offset must be [x, y, z]")

    preview_left_total_offset = [preview_root_offset[i] + preview_left_offset[i] for i in range(3)]
    preview_right_total_offset = [preview_root_offset[i] + preview_right_offset[i] for i in range(3)]

    left_pub = rospy.Publisher("/reaching/left_pose", PoseStamped, queue_size=1, latch=True)
    right_pub = rospy.Publisher("/reaching/right_pose", PoseStamped, queue_size=1, latch=True)
    preview_pub = rospy.Publisher("/reaching/preview_pose_markers", MarkerArray, queue_size=1, latch=True)

    if confirm:
        rospy.loginfo("Confirm mode enabled: sampled poses are previewed first; press Enter to publish them to the policy.")
        while not rospy.is_shutdown():
            left_msg = None
            right_msg = None
            hold_duration = random.uniform(resample_min, resample_max)
            if side in ("left", "both"):
                left_msg, values = sample_pose(LEFT_RANGES, frame_id)
                rospy.loginfo("preview left  x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f", *values)
            if side in ("right", "both"):
                right_msg, values = sample_pose(RIGHT_RANGES, frame_id)
                rospy.loginfo("preview right x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f", *values)

            publish_preview(preview_pub, left_msg, right_msg, preview_frame_id,
                            preview_axis_convention,
                            preview_left_total_offset,
                            preview_right_total_offset)
            if not wait_for_enter("Press Enter to send preview target to policy, Ctrl-C to stop... "):
                break
            rospy.loginfo("Publishing confirmed target for %.2f seconds", hold_duration)
            publish_pose_commands(left_pub, right_pub, left_msg, right_msg, publish_rate, hold_duration)
        return

    left_msg = None
    right_msg = None
    next_resample = rospy.Time.now()
    rate = rospy.Rate(publish_rate)

    rospy.loginfo("Sampling reaching pose commands: side=%s frame=%s resample=[%.2f, %.2f]s",
                  side, frame_id, resample_min, resample_max)

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        if now >= next_resample:
            if side in ("left", "both"):
                left_msg, values = sample_pose(LEFT_RANGES, frame_id)
                rospy.loginfo("left target x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f", *values)
            if side in ("right", "both"):
                right_msg, values = sample_pose(RIGHT_RANGES, frame_id)
                rospy.loginfo("right target x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f", *values)
            next_resample = now + rospy.Duration(random.uniform(resample_min, resample_max))

        stamp = rospy.Time.now()
        if left_msg is not None:
            left_msg.header.stamp = stamp
            left_pub.publish(left_msg)
        if right_msg is not None:
            right_msg.header.stamp = stamp
            right_pub.publish(right_msg)

        rate.sleep()


if __name__ == "__main__":
    main()
