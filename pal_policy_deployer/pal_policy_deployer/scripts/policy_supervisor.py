#!/usr/bin/env python3
"""Start, stop and home a grasp policy on ROS service calls, and report if it is ready to start.

Services (std_srvs/Trigger):
  ~start    run the deployer launch (~deploy_command); refused while a job runs or when not ready
  ~stop     interrupt the running job (policy or home move)
  ~go_home  stop the policy, then move to the start pose (~home_command)
  ~set_task (pal_policy_deployer/SetTask) switch to another task: the controllers of
            kang_grasp_controllers.launch first (~controller_service; they change the gains and
            start or stop the gravity bridge in place), then this node, then move to the new
            start pose. The object tracker (~tracker) switches before the controllers, to
            the task's tracker_objects; it is not waited for (see Tracker). A running policy
            or home move is stopped first; on a failure the previous task stays. The
            TASK_PARAMS overrides apply to the startup task only.
Topics:
  ~state      std_msgs/String, latched: "idle", "policy" or "home"
  ~task       std_msgs/String, latched JSON {"task", "robot", "config_path"}
  ~tasks      std_msgs/String, latched JSON [{"task", "robot"}], the tasks ~set_task accepts
  ~log        std_msgs/String, one message per output line of the jobs
  ~readiness  std_msgs/String, latched JSON {"ready", "reasons", "notes"}

While the policy is not running, an observer runs the inferer with dry_run:=true. It publishes
the observation only while the joint state and the object TF are fresh. The policy is ready
when a fresh observation is inside the training start distribution widened by ~ood_margins
(per observation term, else ~ood_margin): only serious out-of-distribution inputs block the start.
Features outside the strict check_start_distribution.py range are listed as notes.

While the policy runs, the supervisor stops it (it does not resume) when
  - an observation of ~running_ood_terms (default object_to_goal_b, the object pose relative to
    the fixed goal) lies for ~running_ood_frames consecutive frames outside the box spanned by
    the training start range and 0 (object at the goal), widened by ~running_ood_margin, or
    (for object_pos_b, the object pose in the base frame) by the training start ranges of
    object_pos_b and goal_pos_b, or
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
from pal_policy_deployer.srv import SetTask, SetTaskResponse
from tf2_msgs.msg import TFMessage
from tracking_ros.srv import Reregister, SwitchTask

from check_start_distribution import check_observation, load_distribution
from task_config import resolve, setting, switchable_tasks

STRICT_TOLERANCE = 0.005
# The task switch of kang_grasp_controllers.launch (spawn_grasp_controller.py).
CONTROLLER_SERVICE = "/spawn_grasp_controller/set_task"
# The FoundationPose tracker (tracking_ros); it runs outside this container on the same master.
TRACKER = "/tracking_fp_plus_so3_trt_node"
# A tracked object counts as seen when its TF arrived within this time.
TRACKER_FRESH = 1.0


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

    def __init__(self, config, distribution_path, topic, window, margin, margins):
        self._terms = config["observation_terms"]
        self._joint_names = config.get("joint_names", [])
        self._joint_vel_names = config.get("joint_vel_names", self._joint_names)
        self._distribution = load_distribution(distribution_path)
        self._window = window
        self._margin = margin
        self._margins = margins
        self._lock = threading.Lock()
        self._samples = collections.deque()
        self.last_warning = (0.0, "")
        self._subscriber = rospy.Subscriber(topic, Float64MultiArray, self._callback, queue_size=10)

    def close(self):
        self._subscriber.unregister()

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
        term_stats = dict(zip(terms, (stats for _, stats in distribution)))
        offset = 0
        for term, (_, stats) in zip(terms, distribution):
            dim = len(stats["mean"])
            for i in range(dim):
                if term in ood_terms:
                    label = "{} {}".format(term, "xyz"[i] if dim == 3 else i)
                    # From anywhere in the start range the object may move to the goal: 0 in
                    # object_to_goal_b, the goal's start range in object_pos_b.
                    low, high = stats["min"][i], stats["max"][i]
                    if term == "object_to_goal_b":
                        low, high = min(low, 0.0), max(high, 0.0)
                    elif term == "object_pos_b" and "goal_pos_b" in term_stats:
                        goal = term_stats["goal_pos_b"]
                        low, high = min(low, goal["min"][i]), max(high, goal["max"][i])
                    self._bounds.append((label, offset + i, low - margin, high + margin))
            offset += len(stats["mean"])
        missing = set(ood_terms) - set(terms)
        if missing:
            raise ValueError("~running_ood_terms not in the config: {}".format(", ".join(sorted(missing))))
        self._frames = frames
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._lock = threading.Lock()
        self._on_trip = None
        self._subscriber = rospy.Subscriber(topic, Float64MultiArray, self._callback, queue_size=10)

    def close(self):
        self._subscriber.unregister()

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
        self._subscriber = rospy.Subscriber("/tf", TFMessage, self._callback, queue_size=100)

    def close(self):
        self._subscriber.unregister()

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


class Tracker(object):
    """Switches the object tracker to a task's objects YAML, then re-registers the objects
    until their TF frames arrive. A task switch never waits for the objects: while the
    tracker's services are absent, the switch is retried in the background."""

    def __init__(self, namespace, log, first_wait, retry_period):
        self._switch_service = namespace + "/switch_task"
        self._reregister_service = namespace + "/trigger_reregister"
        self._log = log
        self._first_wait = first_wait
        self._retry_period = retry_period
        self._lock = threading.Lock()
        # One service call at a time, so that a background retry cannot reach the tracker
        # after a newer switch.
        self._call_lock = threading.Lock()
        self._generation = 0
        self._objects = None  # The objects YAML the tracker must run; None leaves it alone.
        self._frames = None  # Its TF frames once the tracker accepted it.
        self._accepted = 0.0
        self._next_try = 0.0
        self._attempts = 0
        self._seen = False
        self._last_tf = {}
        self._subscriber = rospy.Subscriber("/tf", TFMessage, self._callback, queue_size=100)
        threading.Thread(target=self._loop, daemon=True).start()

    def _callback(self, msg):
        now = time.monotonic()
        with self._lock:
            frames = self._frames or ()
            for transform in msg.transforms:
                if transform.child_frame_id in frames:
                    self._last_tf[transform.child_frame_id] = now

    def _fresh_locked(self):
        # TF that arrived in the first second after the switch can still be the old objects'.
        since = max(self._accepted + TRACKER_FRESH, time.monotonic() - TRACKER_FRESH)
        return bool(self._frames) and all(self._last_tf.get(frame, 0.0) > since
                                          for frame in self._frames)

    def _accept_locked(self, objects, frames):
        self._objects, self._frames = objects, list(frames)
        self._accepted = time.monotonic()
        self._next_try = self._accepted + self._first_wait
        self._attempts = 0
        self._seen = False
        self._last_tf = {}

    def switch(self, objects):
        """Switch the tracker to objects (an objects YAML, absolute or relative to the
        tracker's ROOT_DIR); None leaves the tracker as it is. Raise RuntimeError when the
        tracker refuses the YAML; it then keeps its current objects."""
        with self._lock:
            if objects is not None and objects == self._objects and self._fresh_locked():
                return "tracker keeps {}".format(objects)
            self._generation += 1
            self._objects, self._frames = objects, None
            self._next_try = 0.0
        if objects is None:
            return "tracker unchanged (the task has no tracker_objects)"
        with self._call_lock:
            try:
                rospy.wait_for_service(self._switch_service, timeout=2.0)
                response = rospy.ServiceProxy(self._switch_service, SwitchTask)(objects, False, 0.0)
            except (rospy.ROSException, rospy.ServiceException) as error:
                return "tracker not reachable ({}); the switch to {} is retried".format(error, objects)
            if not response.success:
                with self._lock:
                    self._objects = None
                raise RuntimeError("tracker: " + response.message)
            with self._lock:
                self._accept_locked(objects, response.tf_frames)
        return "tracker: {} ({})".format(objects, ", ".join(response.tf_frames))

    def status(self):
        """None when the tracker sees the task's objects or is left alone, else what it waits for."""
        with self._lock:
            if self._objects is None or self._seen:
                return None
            if self._frames is None:
                return "tracker not reachable; the switch to {} is retried".format(self._objects)
            missing = [frame for frame in self._frames
                       if time.monotonic() - self._last_tf.get(frame, 0.0) > TRACKER_FRESH]
            return "tracker waits for {} (re-registered {} times)".format(
                ", ".join(missing or self._frames), self._attempts)

    def _loop(self):
        while not rospy.is_shutdown():
            time.sleep(0.5)
            with self._lock:
                if self._frames is not None and not self._seen and self._fresh_locked():
                    self._seen = True
                    self._log("tracker sees {}".format(", ".join(self._frames)))
                if self._objects is None or self._seen or time.monotonic() < self._next_try:
                    continue
                generation, objects, frames = self._generation, self._objects, self._frames
            with self._call_lock:
                with self._lock:
                    if generation != self._generation:
                        continue
                try:
                    if frames is None:
                        rospy.wait_for_service(self._switch_service, timeout=0.5)
                        response = rospy.ServiceProxy(self._switch_service, SwitchTask)(
                            objects, False, 0.0)
                    else:
                        with self._lock:
                            self._attempts += 1
                            self._next_try = time.monotonic() + self._retry_period
                        self._log("tracker does not see {}; re-register".format(", ".join(frames)))
                        response = rospy.ServiceProxy(self._reregister_service, Reregister)(frames)
                except (rospy.ROSException, rospy.ServiceException) as error:
                    with self._lock:
                        # An absent tracker is polled more often than a re-register is repeated.
                        self._next_try = time.monotonic() + (self._retry_period if frames else 2.0)
                    if frames is not None:
                        self._log("tracker re-register failed: {}".format(error))
                    continue
                with self._lock:
                    if generation != self._generation:
                        continue
                    if frames is None and response.success:
                        self._accept_locked(objects, response.tf_frames)
                        self._log("tracker: {} ({})".format(objects, ", ".join(response.tf_frames)))
                    elif frames is None:
                        # The YAML stays invalid; a retry cannot help.
                        self._objects = None
                        self._log("tracker refuses {}: {}".format(objects, response.message))
                    elif not response.success:
                        self._log("tracker re-register refused: {}".format(response.message))

    def close(self):
        self._subscriber.unregister()


class Supervisor(object):
    """Runs at most one job (policy or home move) and the observer while the policy is off."""

    def __init__(self, setup, require_ready, controller_service, tracker=None):
        """tracker: (namespace, first_wait, retry_period) of the object tracker, or None.
        Raises RuntimeError when the tracker refuses the startup task's objects."""
        self._require_ready = require_ready
        self._controller_service = controller_service
        self._lock = threading.Lock()
        self._job = None
        self._observer = None
        self._observer_exit = 0.0
        self._shutdown = False
        self._state_pub = rospy.Publisher("~state", String, queue_size=1, latch=True)
        self._log_pub = rospy.Publisher("~log", String, queue_size=100)
        self._readiness_pub = rospy.Publisher("~readiness", String, queue_size=1, latch=True)
        self._task_pub = rospy.Publisher("~task", String, queue_size=1, latch=True)
        # The tasks set_task can switch to: the same joints as the startup task's robot.
        self._tasks_pub = rospy.Publisher("~tasks", String, queue_size=1, latch=True)
        self._tasks_pub.publish(String(json.dumps(
            [{"task": task, "robot": robot} for task, robot in switchable_tasks(setup.robot)])))
        self._tracker = Tracker(tracker[0], self._log, *tracker[1:]) if tracker else None
        if self._tracker:
            self._log(self._tracker.switch(setup.tracker_objects))
        self._use(setup)
        self._state_pub.publish(String("idle"))
        for name, callback in (("start", self.start), ("stop", self.stop), ("go_home", self.go_home)):
            rospy.Service("~" + name, Trigger, self._trigger(callback))
        rospy.Service("~set_task", SetTask, self._handle_set_task)
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

    def _use(self, setup):
        self._setup = setup
        self._commands = setup.commands
        self._readiness = setup.readiness
        self._run_guard = setup.run_guard
        self._stamp_check = setup.stamp_check
        rospy.set_param("~task", setup.task)
        rospy.set_param("~robot", setup.robot)
        self._task_pub.publish(String(json.dumps(
            {"task": setup.task, "robot": setup.robot, "config_path": setup.config_path})))

    def _handle_set_task(self, request):
        try:
            return SetTaskResponse(True, self.set_task(request.task))
        except Exception as error:  # Every failure goes back to the caller.
            self._log("set_task {} refused: {}".format(request.task, error))
            return SetTaskResponse(False, str(error))

    def set_task(self, task_name):
        """Switch the tracker, the controllers (kang_grasp_controllers.launch) and this node to
        another task, then move to its start pose. A running policy or home move is stopped
        first, as ~stop does; on a failure the previous task stays."""
        with self._lock:
            setup = load_task(task_name, "", False)
            # The controllers refuse to switch while a job still publishes commands.
            self._stop_job_locked()
            try:
                # The observer's inferer would see the controllers change under it.
                self._stop_observer_locked()
                # The tracker first: it needs the camera images, and a YAML it refuses leaves
                # the controllers alone. The new objects are not waited for.
                tracker = self._tracker.switch(setup.tracker_objects) if self._tracker else "no tracker"
            except Exception:
                setup.close()
                raise
            try:
                rospy.wait_for_service(self._controller_service, timeout=2.0)
                response = rospy.ServiceProxy(self._controller_service, SetTask)(task_name)
                if not response.success:
                    raise RuntimeError("controllers: " + response.message)
            except Exception:
                setup.close()
                if self._tracker:
                    try:
                        self._log("controllers refused; " + self._tracker.switch(
                            self._setup.tracker_objects))
                    except RuntimeError as error:
                        self._log("tracker did not go back: {}".format(error))
                raise
            old, self._observer_exit = self._setup, 0.0
            self._use(setup)
            old.close()
            for name in TASK_PARAMS:
                if rospy.has_param("~" + name):
                    rospy.delete_param("~" + name)
            self._log("=== task {} ({}); {}; controllers: {} ===".format(
                setup.task, setup.robot, tracker, response.message))
            self._launch("home")
        return "task {} ({}): {}; controllers: {}; moving home".format(
            setup.task, setup.robot, tracker, response.message)

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
            # set_task replaces the checks; use one task's checks per cycle.
            setup = self._setup
            stamp, warning = setup.readiness.last_warning
            setup.run_guard.check_stale(": " + warning if warning and time.monotonic() - stamp < 3.0 else "")
            problem = setup.stamp_check.problem() if setup.stamp_check else None
            if problem:
                setup.run_guard.trip(problem)
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
        tracker = self._tracker.status() if self._tracker else None
        if tracker:
            notes = [tracker] + notes
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


# Private params that hold one task's settings or files. They apply to the task loaded at
# startup only: set_task loads the next task from its files and drops them.
TASK_PARAMS = ("home_launch", "ood_margin", "ood_margins", "running_ood_terms",
               "running_ood_margin", "running_ood_frames", "running_timeout", "startup_timeout",
               "object_stamp_max_future", "object_stamp_max_age", "object_stamp_window",
               "start_config_path", "distribution_path", "deploy_command", "observer_command",
               "home_command", "observation_topic")


class TaskSetup(object):
    """The commands and checks of one task."""

    def __init__(self, task, robot, config_path, commands, readiness, run_guard, stamp_check,
                 tracker_objects):
        self.task = task
        self.robot = robot
        self.config_path = config_path
        self.commands = commands
        self.readiness = readiness
        self.run_guard = run_guard
        self.stamp_check = stamp_check
        self.tracker_objects = tracker_objects

    def close(self):
        for check in (self.readiness, self.run_guard, self.stamp_check):
            if check is not None:
                check.close()


def load_task(task_name, robot_name, overrides):
    """Build a TaskSetup. overrides: whether the TASK_PARAMS private params apply."""
    task, config_path = resolve(task_name, robot_name)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    def param(name, default):
        return rospy.get_param("~" + name, default) if overrides else default

    def task_setting(name, default):
        if overrides:
            return setting(name, default, config, task)
        for source in (task, config):
            if name in (source.get("supervisor") or {}):
                return source["supervisor"][name]
        return default

    model_dir = os.path.dirname(config["model_path"])
    start_config = (param("start_config_path", "")
                    or os.path.join(model_dir, "deployment_start_config.json"))
    distribution_path = (param("distribution_path", "")
                         or os.path.join(model_dir, "start_observation_distribution.json"))
    for path in (start_config, distribution_path):
        if not os.path.isfile(path):
            raise ValueError("not found: {}".format(path))

    inferer = " ".join([
        "roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch",
        "config:=" + shlex.quote(config_path), "start_controllers:=false",
        "joint_state_topic:=/joint_states", rospy.get_param("~launch_args", "")])
    # The move always loads the task's config, so it moves that robot's joints. A robot whose
    # controllers run without the gravity bridge names a launch file that brings the bridge for
    # the move (the robot's supervisor: home_launch).
    home_launch = task_setting("home_launch", "kang_grasp_gravcomp_move_to_start.launch")
    # A private param of the move (_position_tolerance:=0.1) is the launch arg of that name.
    args = [arg[1:] if arg.startswith("_") else arg
            for arg in shlex.split(rospy.get_param("~home_args", ""))]
    home = " ".join([
        "roslaunch pal_policy_deployer", home_launch, "config:=" + shlex.quote(config_path),
        "start_config_path:=" + shlex.quote(start_config)] + [shlex.quote(arg) for arg in args])
    commands = {
        "policy": param("deploy_command", inferer + " publish_commands:=true"),
        "observer": param("observer_command", inferer + " dry_run:=true"),
        "home": param("home_command", home),
    }
    for name, command in sorted(commands.items()):
        rospy.loginfo("%s: %s", name, command)

    observation_topic = param("observation_topic",
                              config.get("observation_topic", "/pal_policy/observation"))
    # The checks subscribe when built; close the ones built so far on a failure.
    checks = []
    try:
        checks.append(Readiness(
            config, distribution_path, observation_topic,
            float(rospy.get_param("~observation_window", 0.5)),
            float(task_setting("ood_margin", 0.05)), task_setting("ood_margins", {})))
        checks.append(RunGuard(
            config["observation_terms"], load_distribution(distribution_path), observation_topic,
            task_setting("running_ood_terms", ["object_to_goal_b"]),
            float(task_setting("running_ood_margin", 0.05)),
            int(task_setting("running_ood_frames", 3)),
            float(task_setting("running_timeout", 1.0)),
            float(task_setting("startup_timeout", 20.0))))
        # Median of the recent object TF stamp ages (receipt minus stamp) must lie in
        # [-object_stamp_max_future, object_stamp_max_age]; mock objects have no TF to check.
        stamp_check = None
        if config.get("object_frame"):
            stamp_check = ObjectStampCheck(
                config["object_frame"],
                float(task_setting("object_stamp_max_future", 0.05)),
                float(task_setting("object_stamp_max_age", 0.5)),
                int(task_setting("object_stamp_window", 15)))
    except Exception:
        for check in checks:
            check.close()
        raise
    return TaskSetup(task["name"], robot_name or task["robot"], config_path, commands,
                     checks[0], checks[1], stamp_check, task.get("tracker_objects"))


def main():
    rospy.init_node("policy_supervisor")
    tracker = rospy.get_param("~tracker", TRACKER)
    try:
        setup = load_task(rospy.get_param("~task", ""), rospy.get_param("~robot", ""), True)
        supervisor = Supervisor(
            setup, bool(rospy.get_param("~require_ready", True)),
            rospy.get_param("~controller_service", CONTROLLER_SERVICE),
            (tracker, float(rospy.get_param("~tracker_first_wait", 30.0)),
             float(rospy.get_param("~tracker_retry_period", 10.0))) if tracker else None)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
    rospy.on_shutdown(supervisor.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
