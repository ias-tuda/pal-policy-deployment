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

from check_start_distribution import check_observation, load_distribution

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


class Supervisor(object):
    """Runs at most one job (policy or home move) and the observer while the policy is off."""

    def __init__(self, commands, readiness, require_ready):
        self._commands = commands
        self._readiness = readiness
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
                self._state_pub.publish(String("idle"))

    def _on_observer_exit(self, job, code):
        self._log("observer exited with code {}".format(code))
        if job is self._observer:  # It exited by itself, not stopped for the policy.
            self._observer_exit = time.monotonic()

    def _launch(self, name):
        self._log("=== {} ===".format(name))
        self._job = Job(name, self._commands[name], self._log, self._on_exit)
        self._state_pub.publish(String(name))

    def _stop_job_locked(self):
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
        return self._readiness.evaluate()

    def _loop(self):
        while not rospy.is_shutdown():
            self._tick()
            time.sleep(0.5)

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
    profile_path = rospy.get_param("~profile")
    with open(profile_path) as stream:
        profile = yaml.safe_load(stream)
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

    readiness = Readiness(
        profile, distribution_path,
        rospy.get_param("~observation_topic", profile.get("observation_topic", "/pal_policy/observation")),
        float(rospy.get_param("~observation_window", 0.5)),
        float(rospy.get_param("~ood_margin", 0.05)), rospy.get_param("~ood_margins", {}))
    supervisor = Supervisor(commands, readiness, bool(rospy.get_param("~require_ready", True)))
    rospy.on_shutdown(supervisor.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
