#!/usr/bin/env python3
"""Start, stop and home a grasp policy on ROS service calls, and report if it is ready to start.

Services (std_srvs/Trigger):
  ~start    run the deployer launch (~deploy_command); refused while a job runs or when not ready
  ~stop     interrupt the running job (policy or home move)
  ~go_home  stop the policy, then run move_to_start_config.py (~home_command)
Topics:
  ~state      std_msgs/String, latched: "idle", "policy" or "home"
  ~log        std_msgs/String, one message per output line of the jobs
  ~readiness  std_msgs/String, latched JSON {"ready", "reasons", "notes"}

While the policy is not running, an observer runs the inferer with dry_run:=true. It publishes
the observation only while the joint state and the object TF are fresh. The policy is ready
when a fresh observation is inside the training start distribution widened by ~ood_margins
(per profile term, else ~ood_margin): only serious out-of-distribution inputs block the start.
Features outside the strict check_start_distribution.py range are listed as notes.

While the policy runs, the supervisor stops it (it does not resume) when
  - an observation of ~running_ood_terms (default object_to_goal_b, the object pose relative to
    the fixed goal) lies for ~running_ood_frames consecutive frames outside the box spanned by
    the training start range and 0 (object at the goal), widened by ~running_ood_margin, or
  - no observation arrives for ~running_timeout s (the inferer skips steps while the joint
    state or the object TF is stale), or none within ~startup_timeout s of the start, or
  - the object TF stamps are off this host's clock (ObjectStampCheck), which also blocks the start.

The grasp controllers must already be running. The jobs inherit this node's environment.
"""
import collections
import json
import os
import shlex
import signal
import subprocess
import threading
import time

import rospy
import yaml
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger, TriggerResponse
from tf2_msgs.msg import TFMessage

from check_start_distribution import check_observation, load_distribution
from task_config import node_config, setting

STRICT_TOLERANCE = 0.005


class Job(object):
    """One process group whose output lines go to on_line."""

    def __init__(self, name, command, on_line, on_exit):
        self.name = name
        # A new session so that stopping reaches roslaunch and all of its nodes.
        self._process = subprocess.Popen(
            shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, start_new_session=True)
        threading.Thread(target=self._pump, args=(on_line, on_exit), daemon=True).start()

    def _pump(self, on_line, on_exit):
        for line in self._process.stdout:
            on_line(line.rstrip())
        on_exit(self, self._process.wait())

    def running(self):
        return self._process.poll() is None

    def stop(self, log):
        # SIGINT lets roslaunch shut its nodes down and unregister their publishers.
        for sig, wait in ((signal.SIGINT, 15.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(self._process.pid, sig)
            except ProcessLookupError:
                return
            try:
                self._process.wait(wait)
                return
            except subprocess.TimeoutExpired:
                log("{} did not exit after {}".format(self.name, sig.name))


class Readiness(object):
    """Checks the dry-run observations against the training start distribution."""

    def __init__(self, profile, distribution_path, topic, window, margin, margins):
        self._terms = profile["observation_terms"]
        self._joint_names = profile.get("joint_names", [])
        self._joint_vel_names = profile.get("joint_vel_names", self._joint_names)
        self._distribution = load_distribution(distribution_path)
        self._window = window
        self._margin = margin
        self._margins = margins
        self._lock = threading.Lock()
        self._samples = collections.deque()
        self.last_warning = (0.0, "")
        rospy.Subscriber(topic, Float64MultiArray, self._callback, queue_size=10)

    def _callback(self, msg):
        with self._lock:
            self._samples.append((time.monotonic(), list(msg.data)))

    def on_observer_line(self, line):
        if "WARN" in line or "ERROR" in line:
            self.last_warning = (time.monotonic(), line.split("]: ", 1)[-1])

    def evaluate(self):
        now = time.monotonic()
        with self._lock:
            while self._samples and now - self._samples[0][0] > self._window:
                self._samples.popleft()
            samples = [data for _, data in self._samples]
        if not samples:
            stamp, warning = self.last_warning
            reason = "no fresh observation"
            if warning and now - stamp < 3.0:
                reason += ": " + warning
            return False, [reason], []
        mean = [sum(column) / len(samples) for column in zip(*samples)]
        try:
            serious, _ = check_observation(mean, self._terms, self._distribution, self._joint_names,
                                           self._joint_vel_names, self._margin, self._margins)
            strict, _ = check_observation(mean, self._terms, self._distribution, self._joint_names,
                                          self._joint_vel_names, STRICT_TOLERANCE, {})
        except ValueError as error:
            return False, [str(error)], []

        def describe(row):
            _, term, label, value, low, high, tol, _ = row
            return "{} {}: {:.3f} outside [{:.3f}, {:.3f}]".format(
                term.split("/")[0], label, value, low - tol, high + tol)
        reasons = [describe(row) for row in serious if row[0] == "FAIL"]
        notes = [describe(row) for row in strict if row[0] == "FAIL" and describe(row) not in reasons]
        return not reasons, reasons, notes


class RunGuard(object):
    """Calls on_trip once when the running policy's object pose leaves the training range or
    its observations stop."""

    def __init__(self, terms, distribution, topic, ood_terms, margin, frames, timeout, startup_timeout):
        self._bounds = []  # (label, index, low, high)
        offset = 0
        for term, (_, stats) in zip(terms, distribution):
            dim = len(stats["mean"])
            for i in range(dim):
                if term in ood_terms:
                    label = "{} {}".format(term, "xyz"[i] if dim == 3 else i)
                    # From anywhere in the start range the object may move to the goal (0).
                    self._bounds.append((label, offset + i, min(stats["min"][i], 0.0) - margin,
                                         max(stats["max"][i], 0.0) + margin))
            offset += len(stats["mean"])
        missing = set(ood_terms) - set(terms)
        if missing:
            raise ValueError("~running_ood_terms not in the profile: {}".format(", ".join(sorted(missing))))
        self._frames = frames
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._lock = threading.Lock()
        self._on_trip = None
        rospy.Subscriber(topic, Float64MultiArray, self._callback, queue_size=10)

    def arm(self, on_trip):
        with self._lock:
            self._on_trip = on_trip
            self._started = time.monotonic()
            self._last_observation = None
            self._bad_frames = 0

    def disarm(self):
        with self._lock:
            self._on_trip = None

    def _trip_locked(self, reason):
        on_trip, self._on_trip = self._on_trip, None
        # Stopping blocks until roslaunch exits; keep it off the subscriber thread.
        threading.Thread(target=on_trip, args=(reason,), daemon=True).start()

    def _callback(self, msg):
        with self._lock:
            if self._on_trip is None:
                return
            self._last_observation = time.monotonic()
            out = ["{}: {:.3f} outside [{:.3f}, {:.3f}]".format(label, msg.data[index], low, high)
                   for label, index, low, high in self._bounds
                   if index < len(msg.data) and not low <= msg.data[index] <= high]
            self._bad_frames = self._bad_frames + 1 if out else 0
            if self._bad_frames >= self._frames:
                self._trip_locked("object out of range: " + "; ".join(out))

    def check_stale(self, last_warning):
        with self._lock:
            if self._on_trip is None:
                return
            now = time.monotonic()
            if self._last_observation is None:
                if now - self._started > self._startup_timeout:
                    self._trip_locked("no observation within {:.0f} s of the start{}".format(
                        self._startup_timeout, last_warning))
            elif now - self._last_observation > self._timeout:
                self._trip_locked("no observation for {:.1f} s{}".format(
                    now - self._last_observation, last_warning))

    def trip(self, reason):
        with self._lock:
            if self._on_trip is not None:
                self._trip_locked(reason)


class ObjectStampCheck(object):
    """Checks the object TF stamps against this host's clock. The inferer looks the object up
    at the latest common TF time and measures its age from that, so a camera stamping ahead of
    this host hands the policy a pose that old without tripping object_timeout."""

    def __init__(self, frame, max_future, max_age, window):
        self._frame = frame
        self._max_future = max_future
        self._max_age = max_age
        self._ages = collections.deque(maxlen=window)
        self._lock = threading.Lock()
        rospy.Subscriber("/tf", TFMessage, self._callback, queue_size=100)

    def _callback(self, msg):
        now = rospy.Time.now()
        for transform in msg.transforms:
            if transform.child_frame_id == self._frame:
                with self._lock:
                    self._ages.append((time.monotonic(), (now - transform.header.stamp).to_sec()))

    def problem(self):
        """A reason when the recent stamps are off, else None; no recent stamps is left to the
        observation timeouts."""
        with self._lock:
            ages = sorted(age for received, age in self._ages if time.monotonic() - received < 2.0)
        if not ages:
            return None
        age = ages[len(ages) // 2]
        if age < -self._max_future:
            return "{} TF is stamped {:.2f} s in the future; resync the camera clock".format(
                self._frame, -age)
        if age > self._max_age:
            return "{} TF arrives {:.2f} s after its stamp (limit {:.2f} s)".format(
                self._frame, age, self._max_age)
        return None


class Supervisor(object):
    """Runs at most one job (policy or home move) and the observer while the policy is off."""

    def __init__(self, commands, readiness, run_guard, stamp_check, require_ready):
        self._commands = commands
        self._readiness = readiness
        self._run_guard = run_guard
        self._stamp_check = stamp_check
        self._require_ready = require_ready
        self._lock = threading.Lock()
        self._job = None
        self._observer = None
        self._observer_exit = 0.0
        self._shutdown = False
        self._state_pub = rospy.Publisher("~state", String, queue_size=1, latch=True)
        self._log_pub = rospy.Publisher("~log", String, queue_size=100)
        self._readiness_pub = rospy.Publisher("~readiness", String, queue_size=1, latch=True)
        self._state_pub.publish(String("idle"))
        for name, callback in (("start", self.start), ("stop", self.stop), ("go_home", self.go_home)):
            rospy.Service("~" + name, Trigger, self._trigger(callback))
        # Wall-clock thread, not rospy.Timer, so the check also runs while use_sim_time is set.
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _trigger(self, callback):
        def handle(_):
            try:
                return TriggerResponse(True, callback())
            except RuntimeError as error:
                return TriggerResponse(False, str(error))
        return handle

    def _log(self, line):
        rospy.loginfo("%s", line)
        self._log_pub.publish(String(line))

    def _job_running(self):
        return self._job is not None and self._job.running()

    def _on_exit(self, job, code):
        self._log("=== {} exited with code {} ===".format(job.name, code))
        with self._lock:
            if job is self._job:
                self._run_guard.disarm()
                self._state_pub.publish(String("idle"))

    def _on_policy_line(self, line):
        self._log(line)
        self._readiness.on_observer_line(line)

    def _abort(self, job, reason):
        with self._lock:
            if job is not self._job or not job.running():
                return
            self._log("=== stopping policy: {} ===".format(reason))
            job.stop(self._log)

    def _on_observer_exit(self, job, code):
        self._log("observer exited with code {}".format(code))
        if job is self._observer:  # It exited by itself, not stopped for the policy.
            self._observer_exit = time.monotonic()

    def _launch(self, name):
        self._log("=== {} ===".format(name))
        job = Job(name, self._commands[name],
                  self._on_policy_line if name == "policy" else self._log, self._on_exit)
        self._job = job
        if name == "policy":
            self._run_guard.arm(lambda reason: self._abort(job, reason))
        self._state_pub.publish(String(name))

    def _stop_job_locked(self):
        self._run_guard.disarm()
        if self._job_running():
            self._log("stopping {}".format(self._job.name))
            self._job.stop(self._log)

    def _stop_observer_locked(self):
        observer, self._observer = self._observer, None
        if observer is not None and observer.running():
            observer.stop(self._log)

    def _evaluate_locked(self):
        if self._job_running():
            return False, ["{} running".format(self._job.name)], []
        ready, reasons, notes = self._readiness.evaluate()
        problem = self._stamp_check.problem() if self._stamp_check else None
        if problem:
            ready, reasons = False, reasons + [problem]
        return ready, reasons, notes

    def _loop(self):
        count = 0
        while not rospy.is_shutdown():
            stamp, warning = self._readiness.last_warning
            self._run_guard.check_stale(": " + warning if warning and time.monotonic() - stamp < 3.0 else "")
            problem = self._stamp_check.problem() if self._stamp_check else None
            if problem:
                self._run_guard.trip(problem)
            if count % 5 == 0:
                self._tick()
            count += 1
            time.sleep(0.1)

    def _tick(self):
        with self._lock:
            if self._shutdown:
                return
            policy_running = self._job_running() and self._job.name == "policy"
            observer_running = self._observer is not None and self._observer.running()
            # The observer's inferer has the policy's node name, so it runs only while the
            # policy does not; after a crash it restarts after 5 s.
            if not policy_running and not observer_running and time.monotonic() - self._observer_exit > 5.0:
                self._log("starting observer")
                self._observer = Job("observer", self._commands["observer"],
                                     self._readiness.on_observer_line, self._on_observer_exit)
            ready, reasons, notes = self._evaluate_locked()
        self._readiness_pub.publish(String(json.dumps(
            {"ready": ready, "reasons": reasons, "notes": notes})))

    def start(self):
        with self._lock:
            if self._job_running():
                raise RuntimeError("{} is already running; stop it first".format(self._job.name))
            ready, reasons, _ = self._evaluate_locked()
            if self._require_ready and not ready:
                raise RuntimeError("not ready: " + "; ".join(reasons))
            self._stop_observer_locked()
            self._launch("policy")
        return "policy started"

    def go_home(self):
        with self._lock:
            if self._job_running():
                if self._job.name == "home":
                    raise RuntimeError("already moving home")
                self._stop_job_locked()
            self._launch("home")
        return "moving home"

    def stop(self):
        with self._lock:
            self._stop_job_locked()
        return "stopped"

    def shutdown(self):
        with self._lock:
            self._shutdown = True
            self._stop_job_locked()
            self._stop_observer_locked()


def main():
    rospy.init_node("policy_supervisor")
    task, profile_path, profile = node_config()
    model_dir = os.path.dirname(profile["model_path"])
    start_config = (rospy.get_param("~start_config_path", "")
                    or os.path.join(model_dir, "deployment_start_config.json"))
    distribution_path = (rospy.get_param("~distribution_path", "")
                         or os.path.join(model_dir, "start_observation_distribution.json"))
    for path in (start_config, distribution_path):
        if not os.path.isfile(path):
            raise SystemExit("not found: {}".format(path))

    inferer = " ".join([
        "roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch",
        "profile:=" + shlex.quote(profile_path), "start_controllers:=false",
        "joint_state_topic:=/joint_states", rospy.get_param("~launch_args", "")])
    commands = {
        "policy": rospy.get_param("~deploy_command", inferer + " publish_commands:=true"),
        "observer": rospy.get_param("~observer_command", inferer + " dry_run:=true"),
        "home": rospy.get_param("~home_command", " ".join([
            "rosrun pal_policy_deployer move_to_start_config.py",
            "_start_config_path:=" + shlex.quote(start_config), rospy.get_param("~home_args", "")])),
    }
    for name, command in sorted(commands.items()):
        rospy.loginfo("%s: %s", name, command)

    observation_topic = rospy.get_param(
        "~observation_topic", profile.get("observation_topic", "/pal_policy/observation"))
    readiness = Readiness(
        profile, distribution_path, observation_topic,
        float(rospy.get_param("~observation_window", 0.5)),
        float(setting("ood_margin", 0.05, profile, task)), setting("ood_margins", {}, profile, task))
    run_guard = RunGuard(
        profile["observation_terms"], load_distribution(distribution_path), observation_topic,
        setting("running_ood_terms", ["object_to_goal_b"], profile, task),
        float(setting("running_ood_margin", 0.05, profile, task)),
        int(setting("running_ood_frames", 3, profile, task)),
        float(setting("running_timeout", 1.0, profile, task)),
        float(setting("startup_timeout", 20.0, profile, task)))
    # Median of the recent object TF stamp ages (receipt minus stamp) must lie in
    # [-~object_stamp_max_future, ~object_stamp_max_age]; mock objects have no TF to check.
    stamp_check = None
    if profile.get("object_frame"):
        stamp_check = ObjectStampCheck(
            profile["object_frame"], float(rospy.get_param("~object_stamp_max_future", 0.05)),
            float(rospy.get_param("~object_stamp_max_age", 0.5)),
            int(rospy.get_param("~object_stamp_window", 15)))
    supervisor = Supervisor(commands, readiness, run_guard, stamp_check,
                            bool(rospy.get_param("~require_ready", True)))
    rospy.on_shutdown(supervisor.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
