#!/usr/bin/env python
"""Measure how fast the jaw follows the policy's jaw target.

Records the jaw target (preview topic of the inferer) and the measured jaw position
(joint states), live or from a bag. For each target ramp larger than min_stroke it prints the
commanded rate, the measured 10-90 % rate, the peak measured velocity and the 50 % lag. In
training the target moves by actions.gripper.rate_limit per 50 Hz step (0.03 rad -> 1.5 rad/s).

  rosrun pal_policy_deployer measure_jaw_speed.py            # live, Ctrl-C to print the report
  rosrun pal_policy_deployer measure_jaw_speed.py --bag run.bag --csv jaw.csv
"""
import argparse
import csv

import rospy
from sensor_msgs.msg import JointState


def ramps(cmd, min_stroke):
    """Split the target series [(t, q)] into monotonic runs of changing samples."""
    out, start, prev_dir = [], None, 0
    for i in range(1, len(cmd)):
        d = cmd[i][1] - cmd[i - 1][1]
        direction = (d > 1e-6) - (d < -1e-6)
        if direction != prev_dir:
            if start is not None:
                out.append((start, i - 1))
            start = i - 1 if direction else None
            prev_dir = direction
    if start is not None:
        out.append((start, len(cmd) - 1))
    return [(cmd[a], cmd[b]) for a, b in out if abs(cmd[b][1] - cmd[a][1]) >= min_stroke]


def crossing(meas, t_from, t_to, level, rising):
    for i in range(1, len(meas)):
        t0, q0 = meas[i - 1]
        t1, q1 = meas[i]
        if t1 < t_from or t0 > t_to:
            continue
        if (q0 < level <= q1) if rising else (q0 > level >= q1):
            return t0 + (t1 - t0) * (level - q0) / (q1 - q0)
    return None


def report(joint, cmd, meas, min_stroke, sim_rate):
    print("\n=== %s (training target rate %.2f rad/s) ===" % (joint, sim_rate))
    if not cmd or not meas:
        print("no data (target samples %d, measured samples %d)" % (len(cmd), len(meas)))
        return
    found = ramps(cmd, min_stroke)
    if not found:
        print("no target ramp of at least %.2f rad" % min_stroke)
    for n, ((ta, qa), (tb, qb)) in enumerate(found):
        # Measure until the next ramp starts, at most 2 s after this one ends.
        t_end = min(tb + 2.0, found[n + 1][0][0]) if n + 1 < len(found) else tb + 2.0
        stroke, rising = qb - qa, qb > qa
        levels = [qa + f * stroke for f in (0.1, 0.5, 0.9)]
        t10, t50, t90 = [crossing(meas, ta - 0.1, t_end, lv, rising) for lv in levels]
        window = [m for m in meas if ta <= m[0] <= t_end]
        vel = [abs(q1 - q0) / (t1 - t0) for (t0, q0), (t1, q1) in zip(window, window[1:]) if t1 > t0]
        end_q = window[-1][1] if window else float("nan")
        print("%s %.3f -> %.3f at t=%.2f s" % ("close" if rising else "open ", qa, qb, ta - cmd[0][0]))
        print("  target   : %.2f rad/s over %.3f s" % (abs(stroke) / (tb - ta), tb - ta))
        if t10 is not None and t90 is not None:
            print("  measured : %.2f rad/s (10-90 %%), 10-90 %% time %.3f s" %
                  (0.8 * abs(stroke) / (t90 - t10), t90 - t10))
        else:
            print("  measured : did not cross both 10 and 90 %% of the stroke (ended at %.3f, extreme %.3f)" %
                  (end_q, (max if rising else min)(m[1] for m in window)))
        if vel:
            print("  peak vel : %.2f rad/s (finite difference)" % max(vel))
        if t50 is not None:
            t50_cmd = ta + 0.5 * (tb - ta)
            print("  50 %% lag : %+.3f s behind the target" % (t50 - t50_cmd))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", help="read from this bag instead of live topics")
    parser.add_argument("--joints", nargs="+", default=["gripper_left_finger_joint"])
    parser.add_argument("--state-topic", default="/joint_states")
    parser.add_argument("--target-topic", default="/pal_policy/desired_state")
    parser.add_argument("--min-stroke", type=float, default=0.2, help="smallest target ramp in rad")
    parser.add_argument("--sim-rate", type=float, default=1.5, help="training target rate in rad/s")
    parser.add_argument("--csv", help="write all samples to this file")
    args = parser.parse_args(rospy.myargv()[1:])

    cmd = {j: [] for j in args.joints}
    meas = {j: [] for j in args.joints}

    def store(series, msg, t):
        for j in args.joints:
            if j in msg.name:
                series[j].append((t, msg.position[msg.name.index(j)]))

    if args.bag:
        import rosbag
        with rosbag.Bag(args.bag) as bag:
            for topic, msg, t in bag.read_messages(topics=[args.state_topic, args.target_topic]):
                stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else t.to_sec()
                store(meas if topic == args.state_topic else cmd, msg, stamp)
    else:
        rospy.init_node("measure_jaw_speed", anonymous=True)
        rospy.Subscriber(args.state_topic, JointState,
                         lambda m: store(meas, m, m.header.stamp.to_sec() or rospy.get_time()), queue_size=100)
        rospy.Subscriber(args.target_topic, JointState,
                         lambda m: store(cmd, m, m.header.stamp.to_sec() or rospy.get_time()), queue_size=100)
        rospy.loginfo("Recording %s; move the jaw, then press Ctrl-C", ", ".join(args.joints))
        rospy.spin()

    for j in args.joints:
        report(j, cmd[j], meas[j], args.min_stroke, args.sim_rate)

    if args.csv:
        with open(args.csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["joint", "series", "t", "position"])
            for j in args.joints:
                writer.writerows([j, "target", t, q] for t, q in cmd[j])
                writer.writerows([j, "measured", t, q] for t, q in meas[j])
        print("\nwrote %s" % args.csv)


if __name__ == "__main__":
    main()
