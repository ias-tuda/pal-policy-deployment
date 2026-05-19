#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


DEFAULT_LEFT_BOX = [-0.75, -0.35, -0.30, 0.25, 0.05, 0.32]
DEFAULT_RIGHT_BOX = [0.35, 0.75, -0.30, 0.25, 0.05, 0.32]
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def corners_from_ranges(box):
    x0, x1, y0, y1, z0, z1 = box
    return [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]


def transform_corner(corner, axis_convention, root_offset):
    x, y, z = corner
    if axis_convention == "mujoco_to_ros_base":
        # MuJoCo debug command frame -> ROS base_link visual frame.
        # Training x separates left/right; ROS base_link uses y for left/right.
        x, y, z = y, -x, z
    elif axis_convention != "raw":
        rospy.logwarn_once(
            "Unknown ~axis_convention '%s'. Use 'mujoco_to_ros_base' or 'raw'. Falling back to raw.",
            axis_convention,
        )
    ox, oy, oz = root_offset
    return (x + ox, y + oy, z + oz)


def transformed_corners_from_ranges(box, axis_convention, root_offset):
    return [transform_corner(corner, axis_convention, root_offset) for corner in corners_from_ranges(box)]


def box_from_corners(corners):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]
    return [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]


def mujoco_box_to_ros_base_box(box):
    # Training frame -> ROS base_link visual frame.
    # In the trained MuJoCo task, x separates left/right. In ROS base_link,
    # left/right is the y axis, so map: ros_x=train_y, ros_y=-train_x, ros_z=train_z.
    x_min, x_max, y_min, y_max, z_min, z_max = box
    xs = [y_min, y_max]
    ys = [-x_min, -x_max]
    zs = [z_min, z_max]
    return [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]


def apply_offset(box, offset):
    x_min, x_max, y_min, y_max, z_min, z_max = box
    ox, oy, oz = offset
    return [x_min + ox, x_max + ox, y_min + oy, y_max + oy, z_min + oz, z_max + oz]


def make_cube(box, frame_id, ns, marker_id, color):
    x_min, x_max, y_min, y_max, z_min, z_max = box
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = 0.5 * (x_min + x_max)
    marker.pose.position.y = 0.5 * (y_min + y_max)
    marker.pose.position.z = 0.5 * (z_min + z_max)
    marker.pose.orientation.w = 1.0
    marker.scale.x = x_max - x_min
    marker.scale.y = y_max - y_min
    marker.scale.z = z_max - z_min
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.lifetime = rospy.Duration(0.0)
    return marker


def make_wire(box, frame_id, ns, marker_id, color):
    return make_wire_from_corners(corners_from_ranges(box), frame_id, ns, marker_id, color)


def make_wire_from_corners(corners, frame_id, ns, marker_id, color):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.015
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.lifetime = rospy.Duration(0.0)

    def p(x, y, z):
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

    for a, b in BOX_EDGES:
        marker.points.append(p(*corners[a]))
        marker.points.append(p(*corners[b]))

    return marker


def build_markers(frame_id, left_corners, right_corners):
    left_box = box_from_corners(left_corners)
    right_box = box_from_corners(right_corners)
    markers = MarkerArray()
    markers.markers.append(make_cube(left_box, frame_id, "reaching_sampling_box_left", 0, (0.95, 0.20, 0.20, 0.18)))
    markers.markers.append(make_cube(right_box, frame_id, "reaching_sampling_box_right", 1, (0.20, 0.45, 0.95, 0.18)))
    markers.markers.append(make_wire_from_corners(left_corners, frame_id, "reaching_sampling_box_left_wire", 2, (1.0, 0.05, 0.05, 1.0)))
    markers.markers.append(make_wire_from_corners(right_corners, frame_id, "reaching_sampling_box_right_wire", 3, (0.05, 0.35, 1.0, 1.0)))
    return markers


def print_box_corners(label, box, corners, frame_id, axis_convention):
    x0, x1, y0, y1, z0, z1 = box
    sep = "=" * 72
    print(sep)
    print("  SAMPLING BOX CORNERS  (axis_convention='%s')" % axis_convention)
    print(sep)
    print("  ── %s  [x=(%.2f, %.2f), y=(%.2f, %.2f), z=(%.2f, %.2f)]" % (
        label, x0, x1, y0, y1, z0, z1))
    print("     Corners in published frame  -> frame_id='%s' in RViz:" % frame_id)
    for i, (cx, cy, cz) in enumerate(corners):
        print("       [%d] (%+.4f, %+.4f, %+.4f)" % (i, cx, cy, cz))
    print(sep)
    print()


def main():
    rospy.init_node("reaching_sampling_box_publisher")
    frame_id = rospy.get_param("~frame_id", "pelvis_2_link")
    left_box = rospy.get_param("~left_box", DEFAULT_LEFT_BOX)
    right_box = rospy.get_param("~right_box", DEFAULT_RIGHT_BOX)
    axis_convention = rospy.get_param("~axis_convention", "mujoco_to_ros_base")
    root_offset = rospy.get_param("~root_offset", [0.0, 0.0, 0.0])
    left_offset = rospy.get_param("~left_offset", [0.0, 0.0, 0.0])
    right_offset = rospy.get_param("~right_offset", [0.0, 0.0, 0.0])
    rate_hz = rospy.get_param("~rate", 2.0)

    if len(root_offset) != 3:
        rospy.logwarn("~root_offset must be [x, y, z], got %s. Using [0, 0, 0].", root_offset)
        root_offset = [0.0, 0.0, 0.0]
    if len(left_offset) != 3:
        rospy.logwarn("~left_offset must be [x, y, z], got %s. Using [0, 0, 0].", left_offset)
        left_offset = [0.0, 0.0, 0.0]
    if len(right_offset) != 3:
        rospy.logwarn("~right_offset must be [x, y, z], got %s. Using [0, 0, 0].", right_offset)
        right_offset = [0.0, 0.0, 0.0]

    left_total_offset = [root_offset[i] + left_offset[i] for i in range(3)]
    right_total_offset = [root_offset[i] + right_offset[i] for i in range(3)]
    left_corners = transformed_corners_from_ranges(left_box, axis_convention, left_total_offset)
    right_corners = transformed_corners_from_ranges(right_box, axis_convention, right_total_offset)

    print_box_corners("pose_command_left", left_box, left_corners, frame_id, axis_convention)
    print_box_corners("pose_command_right", right_box, right_corners, frame_id, axis_convention)

    pub = rospy.Publisher("/reaching/sampling_boxes", MarkerArray, queue_size=1, latch=True)
    rate = rospy.Rate(rate_hz)
    rospy.loginfo("Publishing reaching sampling boxes in frame '%s'", frame_id)
    rospy.loginfo("axis_convention = %s", axis_convention)
    rospy.loginfo("root_offset [x y z] = %s", root_offset)
    rospy.loginfo("left_offset [x y z] = %s", left_offset)
    rospy.loginfo("right_offset [x y z] = %s", right_offset)
    rospy.loginfo("left_ranges  [x_min x_max y_min y_max z_min z_max] = %s", left_box)
    rospy.loginfo("right_ranges [x_min x_max y_min y_max z_min z_max] = %s", right_box)

    while not rospy.is_shutdown():
        pub.publish(build_markers(frame_id, left_corners, right_corners))
        rate.sleep()


if __name__ == "__main__":
    main()
