#!/usr/bin/env python3
"""Interactive reaching-pose sampler.

Loop:
  1. Press Enter  →  sample a random position inside the box(es), show as sphere
  2. Press Enter  →  confirm: publish PoseStamped goal; or type 's'+Enter to skip
  Ctrl-C at any point exits cleanly.

Box freeze:
  At startup the node listens to /tf and /tf_static once, resolves the transform
  freeze_frame <- frame_id, and stores all box corners in that fixed frame.
  No tf2_ros C-extension is used (Melodic Python3 compatible).
"""

import random
import threading
from collections import deque
import rospy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_LEFT_BOX  = [-0.75, -0.35, -0.30, 0.25, 0.05, 0.32]  # [x0,x1,y0,y1,z0,z1]
DEFAULT_RIGHT_BOX = [ 0.35,  0.75, -0.30, 0.25, 0.05, 0.32]

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ── pure-Python transform math ────────────────────────────────────────────────

def _quat_rotate(v, q):
    """Rotate vector v=(x,y,z) by unit quaternion q=(qx,qy,qz,qw)."""
    vx, vy, vz = v
    qx, qy, qz, qw = q
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx)


def _quat_mul(q2, q1):
    """Hamilton product q2 * q1, both (qx,qy,qz,qw)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (w2*x1 + x2*w1 + y2*z1 - z2*y1,
            w2*y1 - x2*z1 + y2*w1 + z2*x1,
            w2*z1 + x2*y1 - y2*x1 + z2*w1,
            w2*w1 - x2*x1 - y2*y1 - z2*z1)


def _tf_invert(t, q):
    """Invert transform (t, q): p_parent = R*p_child + t → p_child from p_parent."""
    qx, qy, qz, qw = q
    q_inv = (-qx, -qy, -qz, qw)
    t_inv = _quat_rotate((-t[0], -t[1], -t[2]), q_inv)
    return t_inv, q_inv


def _tf_compose(t2, q2, t1, q1):
    """Compose: apply T1 first then T2.  p_out = R2*(R1*p + t1) + t2."""
    t_out = tuple(a + b for a, b in zip(_quat_rotate(t1, q2), t2))
    q_out = _quat_mul(q2, q1)
    return t_out, q_out


def _tf_apply(pos, t, q):
    """Apply transform to a point: p_out = R*pos + t."""
    rx, ry, rz = _quat_rotate(pos, q)
    return (rx + t[0], ry + t[1], rz + t[2])


# ── minimal TF buffer (no C-extension needed) ─────────────────────────────────

class _TfBuffer:
    """Subscribes to /tf and /tf_static; resolves transforms via BFS."""

    def __init__(self):
        self._lock = threading.Lock()
        self._edges = {}   # (parent_frame, child_frame) → (trans, quat)
        self._sub_tf     = rospy.Subscriber("/tf",        TFMessage, self._cb)
        self._sub_static = rospy.Subscriber("/tf_static", TFMessage, self._cb)

    def _cb(self, msg):
        with self._lock:
            for t in msg.transforms:
                parent = t.header.frame_id.lstrip("/")
                child  = t.child_frame_id.lstrip("/")
                tr = (t.transform.translation.x,
                      t.transform.translation.y,
                      t.transform.translation.z)
                q  = (t.transform.rotation.x,
                      t.transform.rotation.y,
                      t.transform.rotation.z,
                      t.transform.rotation.w)
                self._edges[(parent, child)] = (tr, q)

    def lookup(self, target_frame, source_frame, timeout=3.0):
        """Return (trans, quat) for target←source, or None on failure."""
        target_frame = target_frame.lstrip("/")
        source_frame = source_frame.lstrip("/")
        if target_frame == source_frame:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            with self._lock:
                result = self._bfs(target_frame, source_frame)
            if result is not None:
                return result
            rospy.sleep(0.05)
        return None

    def _bfs(self, target, source):
        # BFS from source; accumulate T_{current}_{source} at each node.
        identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        queue   = deque([(source, identity)])
        visited = {source}

        while queue:
            current, acc_t, acc_q = (lambda x: (x[0], x[1][0], x[1][1]))(
                (queue[0][0], queue[0][1])
            )
            queue.popleft()

            for (parent, child), (tr, q) in self._edges.items():
                # child → parent (go up the tree)
                if child == current and parent not in visited:
                    new_t, new_q = _tf_compose(tr, q, acc_t, acc_q)
                    if parent == target:
                        return new_t, new_q
                    visited.add(parent)
                    queue.append((parent, (new_t, new_q)))

                # parent → child (go down the tree, use inverse)
                if parent == current and child not in visited:
                    inv_t, inv_q = _tf_invert(tr, q)
                    new_t, new_q = _tf_compose(inv_t, inv_q, acc_t, acc_q)
                    if child == target:
                        return new_t, new_q
                    visited.add(child)
                    queue.append((child, (new_t, new_q)))

        return None

    def shutdown(self):
        self._sub_tf.unregister()
        self._sub_static.unregister()


# ── geometry ──────────────────────────────────────────────────────────────────

def _pose_from_list(values, frame_id):
    if len(values) != 7:
        raise rospy.ROSException("Pose list must have 7 values: [x,y,z,qw,qx,qy,qz]")
    msg = PoseStamped()
    msg.header.frame_id    = frame_id
    msg.pose.position.x    = float(values[0])
    msg.pose.position.y    = float(values[1])
    msg.pose.position.z    = float(values[2])
    msg.pose.orientation.w = float(values[3])
    msg.pose.orientation.x = float(values[4])
    msg.pose.orientation.y = float(values[5])
    msg.pose.orientation.z = float(values[6])
    return msg


def _corners_from_box(box):
    x0, x1, y0, y1, z0, z1 = box
    return [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]


def _freeze_corners(box, freeze_t, freeze_q):
    return [_tf_apply(c, freeze_t, freeze_q) for c in _corners_from_box(box)]


def _sample_in_frozen_box(box, freeze_t, freeze_q):
    x0, x1, y0, y1, z0, z1 = box
    local = (random.uniform(x0, x1), random.uniform(y0, y1), random.uniform(z0, z1))
    return _tf_apply(local, freeze_t, freeze_q)


# ── marker builders ───────────────────────────────────────────────────────────

def _make_box_cube_from_corners(corners, frame_id, ns, marker_id, color):
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]; zs = [c[2] for c in corners]
    m = Marker()
    m.header.frame_id = frame_id; m.header.stamp = rospy.Time.now()
    m.ns, m.id        = ns, marker_id
    m.type = Marker.CUBE; m.action = Marker.ADD
    m.pose.position.x    = 0.5 * (min(xs) + max(xs))
    m.pose.position.y    = 0.5 * (min(ys) + max(ys))
    m.pose.position.z    = 0.5 * (min(zs) + max(zs))
    m.pose.orientation.w = 1.0
    m.scale.x = max(xs) - min(xs); m.scale.y = max(ys) - min(ys); m.scale.z = max(zs) - min(zs)
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.lifetime = rospy.Duration(0.0)
    return m


def _make_box_wire_from_corners(corners, frame_id, ns, marker_id, color):
    m = Marker()
    m.header.frame_id = frame_id; m.header.stamp = rospy.Time.now()
    m.ns, m.id        = ns, marker_id
    m.type = Marker.LINE_LIST; m.action = Marker.ADD
    m.pose.orientation.w = 1.0; m.scale.x = 0.012
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.lifetime = rospy.Duration(0.0)
    for a, b in BOX_EDGES:
        for cx, cy, cz in (corners[a], corners[b]):
            pt = Point(); pt.x, pt.y, pt.z = cx, cy, cz
            m.points.append(pt)
    return m


def _make_sphere(pos, frame_id, ns, marker_id, color, scale):
    x, y, z = pos
    m = Marker()
    m.header.frame_id = frame_id; m.header.stamp = rospy.Time.now()
    m.ns, m.id        = ns, marker_id
    m.type = Marker.SPHERE; m.action = Marker.ADD
    m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = z
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.lifetime = rospy.Duration(0.0)
    return m


# ── full marker array ─────────────────────────────────────────────────────────

def _build_markers(box_frame, side, left_corners, right_corners,
                   pending_left, pending_right):
    ma = MarkerArray()
    if left_corners is not None and side in ("left", "both"):
        ma.markers.append(_make_box_cube_from_corners(
            left_corners, box_frame, "reaching_box_left",      10, (0.95, 0.20, 0.20, 0.08)))
        ma.markers.append(_make_box_wire_from_corners(
            left_corners, box_frame, "reaching_box_left_wire", 11, (1.00, 0.05, 0.05, 0.85)))
    if right_corners is not None and side in ("right", "both"):
        ma.markers.append(_make_box_cube_from_corners(
            right_corners, box_frame, "reaching_box_right",      12, (0.20, 0.45, 0.95, 0.08)))
        ma.markers.append(_make_box_wire_from_corners(
            right_corners, box_frame, "reaching_box_right_wire", 13, (0.05, 0.35, 1.00, 0.85)))
    if pending_left is not None:
        ma.markers.append(_make_sphere(
            pending_left,  box_frame, "reaching_pending_left",  20, (1.0, 0.1, 0.1, 1.0), 0.055))
    if pending_right is not None:
        ma.markers.append(_make_sphere(
            pending_right, box_frame, "reaching_pending_right", 21, (0.1, 0.4, 1.0, 1.0), 0.055))
    return ma


# ── helpers ───────────────────────────────────────────────────────────────────

def _ask(prompt):
    try:
        ans = input(prompt)
        skipped = ans.strip().lower() in ("s", "skip")
        return True, skipped
    except (KeyboardInterrupt, EOFError):
        return False, False


def _publish_goal(pub, frozen_pos, ref_msg, goal_frame, tf_buf, box_frame,
                  rate_hz, duration_s):
    if box_frame != goal_frame:
        result = tf_buf.lookup(goal_frame, box_frame, timeout=1.0)
        if result is not None:
            local_pos  = _tf_apply(frozen_pos, result[0], result[1])
            publish_frame = goal_frame
        else:
            rospy.logwarn("Cannot transform goal to '%s'; publishing in '%s'.",
                          goal_frame, box_frame)
            local_pos     = frozen_pos
            publish_frame = box_frame
    else:
        local_pos     = frozen_pos
        publish_frame = goal_frame

    msg = PoseStamped()
    msg.header.frame_id  = publish_frame
    msg.pose.orientation = ref_msg.pose.orientation
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = local_pos

    rate     = rospy.Rate(rate_hz)
    end_time = rospy.Time.now() + rospy.Duration(duration_s)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("preview_confirm_reaching_pose")

    side         = rospy.get_param("~side",         "left")
    frame_id     = rospy.get_param("~frame_id",     "imu_link")
    freeze_frame = rospy.get_param("~freeze_frame", "odom")

    left_pose  = rospy.get_param("~left_pose",
                                 [-0.5374, -0.2050, 0.2348, 0.9297,  0.1353, -0.2128,  0.2686])
    right_pose = rospy.get_param("~right_pose",
                                 [ 0.5374, -0.2050, 0.2348, 0.9297, -0.1353, -0.2128, -0.2686])

    publish_rate     = float(rospy.get_param("~publish_rate",     5.0))
    publish_duration = float(rospy.get_param("~publish_duration", 3.0))
    preview_rate     = float(rospy.get_param("~preview_rate",     2.0))

    left_box_p  = rospy.get_param("~left_box",  DEFAULT_LEFT_BOX)
    right_box_p = rospy.get_param("~right_box", DEFAULT_RIGHT_BOX)

    if side not in ("left", "right", "both"):
        raise rospy.ROSException("~side must be: left | right | both")
    if len(left_box_p) != 6 or len(right_box_p) != 6:
        raise rospy.ROSException("~left_box / ~right_box must be [x0,x1,y0,y1,z0,z1]")

    # ── TF freeze ─────────────────────────────────────────────────────────────
    tf_buf = _TfBuffer()
    rospy.sleep(0.5)

    freeze = tf_buf.lookup(freeze_frame, frame_id, timeout=3.0)
    if freeze is not None:
        freeze_t, freeze_q = freeze
        box_frame = freeze_frame
        rospy.loginfo("Boxes frozen in '%s' (looked up from '%s')", freeze_frame, frame_id)
    else:
        freeze_t, freeze_q = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        box_frame = frame_id
        rospy.logwarn("Could not resolve '%s'←'%s' – boxes will move with IMU.",
                      freeze_frame, frame_id)

    # ── frozen corners ────────────────────────────────────────────────────────
    left_corners  = _freeze_corners(left_box_p,  freeze_t, freeze_q) if side in ("left",  "both") else None
    right_corners = _freeze_corners(right_box_p, freeze_t, freeze_q) if side in ("right", "both") else None

    left_ref_msg  = _pose_from_list(left_pose,  frame_id) if side in ("left",  "both") else None
    right_ref_msg = _pose_from_list(right_pose, frame_id) if side in ("right", "both") else None

    left_pub    = rospy.Publisher("/reaching/left_pose",            PoseStamped, queue_size=1, latch=True)
    right_pub   = rospy.Publisher("/reaching/right_pose",           PoseStamped, queue_size=1, latch=True)
    preview_pub = rospy.Publisher("/reaching/preview_pose_markers", MarkerArray, queue_size=1, latch=True)

    rospy.sleep(0.25)

    state = {"pending_left": None, "pending_right": None}

    def _repub(_evt=None):
        preview_pub.publish(_build_markers(
            box_frame, side, left_corners, right_corners,
            state["pending_left"], state["pending_right"],
        ))

    _repub()
    timer = rospy.Timer(rospy.Duration(1.0 / max(preview_rate, 0.2)), _repub)

    rospy.loginfo("=" * 60)
    rospy.loginfo("Side: %s  |  goal frame: %s  |  box frame: %s", side, frame_id, box_frame)
    if left_corners:
        rospy.loginfo("Left  box [x0 x1 y0 y1 z0 z1]: %s", left_box_p)
    if right_corners:
        rospy.loginfo("Right box [x0 x1 y0 y1 z0 z1]: %s", right_box_p)
    rospy.loginfo("RViz topic: /reaching/preview_pose_markers  (MarkerArray)")
    rospy.loginfo("Controls: Enter = sample/confirm   |   s+Enter = skip   |   Ctrl-C = quit")
    rospy.loginfo("=" * 60)

    sample_count = 0

    while not rospy.is_shutdown():

        ok, _ = _ask("\n[%d] Press Enter to sample (Ctrl-C = quit)... " % (sample_count + 1))
        if not ok or rospy.is_shutdown():
            break

        if left_corners is not None:
            state["pending_left"]  = _sample_in_frozen_box(left_box_p,  freeze_t, freeze_q)
        if right_corners is not None:
            state["pending_right"] = _sample_in_frozen_box(right_box_p, freeze_t, freeze_q)

        _repub()

        if state["pending_left"]:
            rospy.loginfo("  Sampled LEFT  [%s]: (%.4f, %.4f, %.4f)", box_frame, *state["pending_left"])
        if state["pending_right"]:
            rospy.loginfo("  Sampled RIGHT [%s]: (%.4f, %.4f, %.4f)", box_frame, *state["pending_right"])

        ok, skipped = _ask("  Enter = confirm as goal   |   s+Enter = skip... ")
        if not ok or rospy.is_shutdown():
            break
        if skipped:
            rospy.loginfo("  Skipped.")
            state["pending_left"] = state["pending_right"] = None
            _repub()
            continue

        if state["pending_left"] is not None and left_ref_msg is not None:
            rospy.loginfo("  Publishing LEFT  goal (%.2f s @ %.1f Hz)...", publish_duration, publish_rate)
            _publish_goal(left_pub,  state["pending_left"],  left_ref_msg,
                          frame_id, tf_buf, box_frame, publish_rate, publish_duration)

        if state["pending_right"] is not None and right_ref_msg is not None:
            rospy.loginfo("  Publishing RIGHT goal (%.2f s @ %.1f Hz)...", publish_duration, publish_rate)
            _publish_goal(right_pub, state["pending_right"], right_ref_msg,
                          frame_id, tf_buf, box_frame, publish_rate, publish_duration)

        sample_count += 1
        rospy.loginfo("  Goal #%d published.", sample_count)
        state["pending_left"] = state["pending_right"] = None
        _repub()

    timer.shutdown()
    tf_buf.shutdown()
    rospy.loginfo("Done. %d goal(s) published.", sample_count)


if __name__ == "__main__":
    main()
