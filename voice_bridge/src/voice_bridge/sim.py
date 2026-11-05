"""A simulated policy supervisor, tracker and arm: for the tests, for voice_bridge_node.py --sim
on a PC without ROS, and behind mock_supervisor.py.

It does what the bridge relies on in policy_supervisor.py, with the same messages:
  - ~state, ~task, ~tasks and ~readiness (2 Hz; it pauses while set_task holds the lock);
  - set_task stops a running job, switches the tracker (which applies the new objects later),
    publishes ~task, starts the home move and returns before the move ends;
  - the home move logs "Start pose verified" (or "Start pose move failed") before it exits;
  - start is refused while a job runs or when not ready; a policy run leaves the arm away from
    the start pose, so the next command needs Go home first.
The test controls are the attributes below the "Test controls" comment.
"""
import json
import threading
import time

from .bridge import STOP_REQUESTED
from .configs import TaskInfo

ARM = ["arm_left_{}_joint".format(index) for index in range(1, 8)]
JAW = "gripper_left_finger_joint"
# The start pose of every voice task's model (deployment_start_config.json); the jaw differs.
ARM_START = [0.102, 1.452, 1.777, 1.670, 0.458, 0.404, -0.009]
JAW_START = {"left_arm_gravcomp": 0.048, "left_arm_nogravcomp": 0.0}
TRACKER_DIR = "custom-assets/kang-grasp-cube/"
# Task -> (robot, object frames, tracker objects YAML), from the task files.
KNOWN_TASKS = {
    "grasp-cube": ("left_arm_gravcomp", ["cube_1"], "grasp-cube.yaml"),
    "stack-cube": ("left_arm_gravcomp", ["cube_1", "cube_2"], "stack-cube.yaml"),
    "putintobox-s102-wx-a-h22-s1": ("left_arm_nogravcomp", ["cube_1", "box_1"], "box-cube.yaml"),
    "putballintobox-s15-pb-dr-s1": ("left_arm_nogravcomp", ["ball_1", "block_1"], "ball-block.yaml"),
    "mug-ball": ("left_arm_gravcomp", ["ball_1", "mug_1"], "mug-ball.yaml"),
    "mug-turn": ("left_arm_gravcomp", ["mug_1"], "turn-mug.yaml"),
    "mug-coaster": ("left_arm_gravcomp", ["mug_1"], "turn-mug.yaml"),
}


class SimTask(object):

    def __init__(self, name, robot, start_pose, object_frames, tracker_objects, config_path=None):
        self.name = name
        self.robot = robot
        self.start_pose = dict(start_pose)
        self.object_frames = list(object_frames)
        self.tracker_objects = tracker_objects
        self.config_path = config_path or "/sim/{}.yaml".format(name)


def sim_tasks(names):
    """SimTasks for task names: the known voice tasks as in their task files, others like
    grasp-cube."""
    tasks = {}
    for name in names:
        robot, frames, objects = KNOWN_TASKS.get(name, KNOWN_TASKS["grasp-cube"])
        pose = dict(zip(ARM, ARM_START))
        pose[JAW] = JAW_START[robot]
        tasks[name] = SimTask(name, robot, pose, frames, TRACKER_DIR + objects)
    return tasks


class Timing(object):
    """Durations in s, all multiplied by scale."""

    def __init__(self, scale=1.0, **overrides):
        self.switch = 0.5      # set_task's own time, lock held
        self.start = 0.3       # ~start stops the observer first, lock held
        self.tracker = 1.0     # until the tracker publishes the new task's objects
        self.home = 3.0
        self.policy = 20.0
        self.readiness = 0.5
        self.tf = 1.0 / 30
        self.joints = 0.02
        for name, value in overrides.items():
            if not hasattr(self, name):
                raise ValueError("unknown timing {}".format(name))
            setattr(self, name, value)
        for name in list(vars(self)):
            setattr(self, name, getattr(self, name) * scale)


class SimSupervisor(object):
    """sink: on_state, on_task, on_tasks, on_readiness, on_joints, on_tf, on_active_objects and
    on_log, as RobotState has them."""

    def __init__(self, sink, tasks, task, timing=None):
        self._sink = sink
        self._tasks = dict(tasks)
        self._timing = timing or Timing()
        self._lock = threading.RLock()        # the supervisor's lock
        self._world_lock = threading.RLock()  # tracker and arm, which run on their own
        self._task = self._tasks[task]
        self._job = None
        self._job_end = None
        self._joints = dict(self._task.start_pose)
        self._tracked = self._task.tracker_objects
        self._tracker_next = None  # (objects YAML, time it applies)
        self._frames = {}  # tracker objects YAML -> frames
        for sim_task in self._tasks.values():
            if sim_task.tracker_objects:
                self._frames.setdefault(sim_task.tracker_objects, sim_task.object_frames)
        # Test controls.
        self.objects_present = True
        self.tracker_running = True
        self.home_fails = False
        self.switch_refusal = None  # a message: the controllers refuse the switch
        self.ignore_set_task = False  # set_task never answers (a hung service call)
        self.log_stop_requests = False  # stop() logs STOP_REQUESTED, which the real one does not yet
        self._closed = threading.Event()
        self._sink.on_tasks(json.dumps([{"task": t.name, "robot": t.robot} for t in self._tasks.values()]))
        self._publish_task()
        self._sink.on_state("idle")
        self._sink.on_active_objects(self._active_path())
        for target in (self._supervisor_loop, self._world_loop):
            threading.Thread(target=target, daemon=True).start()

    def close(self):
        self._closed.set()

    # The supervisor's services.

    def set_task(self, name):
        if self.ignore_set_task:
            self._closed.wait()
            return False, "closed"
        with self._lock:
            if name not in self._tasks:
                return False, "[Errno 2] No such file or directory: '{}.yaml'".format(name)
            self._stop_job_locked(idle=False)
            time.sleep(self._timing.switch)
            if self.switch_refusal:
                return False, "controllers: " + self.switch_refusal
            new = self._tasks[name]
            tracker = self._switch_tracker(new)
            self._task = new
            self._publish_task()
            self._log("=== task {} ({}); {}; controllers: controllers unchanged ===".format(
                new.name, new.robot, tracker))
            self._launch("home")
            return True, "task {} ({}): {}; controllers: controllers unchanged; moving home".format(
                new.name, new.robot, tracker)

    def start(self):
        with self._lock:
            if self._job:
                return False, "{} is already running; stop it first".format(self._job)
            ready, reasons = self._evaluate_locked()
            if not ready:
                return False, "not ready: " + "; ".join(reasons)
            time.sleep(self._timing.start)
            self._launch("policy")
            return True, "policy started"

    def go_home(self):
        with self._lock:
            if self._job == "home":
                return False, "already moving home"
            self._stop_job_locked(idle=False)
            self._launch("home")
            return True, "moving home"

    def stop(self):
        with self._lock:
            if self.log_stop_requests:
                self._log(STOP_REQUESTED)
            self._stop_job_locked()
            return True, "stopped"

    def task_info(self, name, robot=None, current=False):
        task = self._tasks[name]
        return TaskInfo(dict(task.start_pose), list(task.object_frames), task.tracker_objects)

    def move_arm(self, offset):
        """Put the arm offset rad away from the start pose (joint 1)."""
        with self._world_lock:
            self._joints[ARM[0]] = self._task.start_pose[ARM[0]] + offset

    # Inside.

    def _publish_task(self):
        self._sink.on_task(json.dumps({"task": self._task.name, "robot": self._task.robot,
                                       "config_path": self._task.config_path}))

    def _log(self, line):
        self._sink.on_log(line)

    def _launch(self, name):
        self._log("=== {} ===".format(name))
        self._job = name
        self._job_end = time.monotonic() + (self._timing.home if name == "home" else self._timing.policy)
        self._sink.on_state(name)

    def _stop_job_locked(self, idle=True):
        """idle: publish "idle". The real supervisor does so only for ~stop: after set_task and
        go_home the stopped job's exit finds the next job already running."""
        if self._job:
            self._log("stopping {}".format(self._job))
            if self._job == "policy":
                self.move_arm(0.5)  # A stopped policy leaves the arm where it was.
            self._exit_locked(-2, idle)

    def _exit_locked(self, code, idle=True):
        self._log("=== {} exited with code {} ===".format(self._job, code))
        self._job = None
        self._job_end = None
        if idle:
            self._sink.on_state("idle")

    def _end_job_locked(self):
        if self._job == "home":
            if self.home_fails:
                self._log("[FATAL] [{:.3f}]: Start pose move failed: start pose not reached within "
                          "15.0 s: arm_left_2_joint error 0.2000, arm_left_2_joint speed 0.0000".format(time.time()))
                self._exit_locked(1)
                return
            with self._world_lock:
                self._joints = dict(self._task.start_pose)
            self._log("[INFO] [{:.3f}]: Start pose verified: max joint error 0.0123, max speed 0.0010".format(
                time.time()))
            self._exit_locked(0)
        else:
            self.move_arm(0.5)  # The policy ends away from the start pose.
            self._exit_locked(0)

    def _switch_tracker(self, task):
        if task.tracker_objects is None:
            return "tracker unchanged (the task has no tracker_objects)"
        if not self.tracker_running:
            return "tracker not reachable (sim); the switch to {} is retried".format(task.tracker_objects)
        with self._world_lock:
            if task.tracker_objects == self._tracked:
                return "tracker keeps {}".format(task.tracker_objects)
            self._tracker_next = (task.tracker_objects, time.monotonic() + self._timing.tracker)
        return "tracker: {} ({})".format(task.tracker_objects, ", ".join(task.object_frames))

    def _active_path(self):
        return "/sim/tracker/" + self._tracked if self._tracked else ""

    def _objects_seen(self, task):
        with self._world_lock:
            return (self.objects_present and self.tracker_running
                    and (task.tracker_objects is None or self._tracked == task.tracker_objects))

    def _evaluate_locked(self):
        if self._job:
            return False, ["{} running".format(self._job)]
        if not self._objects_seen(self._task):
            return False, ["no fresh observation: Grasp policy waiting for object TF {}".format(
                self._task.object_frames[0] if self._task.object_frames else "-")]
        with self._world_lock:
            error = max(abs(self._joints[name] - self._task.start_pose[name]) for name in ARM)
        if error > 0.15:
            return False, ["joint_pos_rel outside the start range"]
        return True, []

    def _supervisor_loop(self):
        next_tick = time.monotonic()
        while not self._closed.is_set():
            with self._lock:
                if self._job and time.monotonic() >= self._job_end:
                    self._end_job_locked()
                if time.monotonic() >= next_tick:
                    ready, reasons = self._evaluate_locked()
                    self._sink.on_readiness(json.dumps({"ready": ready, "reasons": reasons, "notes": []}))
                    next_tick = time.monotonic() + self._timing.readiness
            time.sleep(0.005)

    def _world_loop(self):
        next_tf = next_joints = time.monotonic()
        while not self._closed.is_set():
            now = time.monotonic()
            with self._world_lock:
                if self._tracker_next and now >= self._tracker_next[1]:
                    self._tracked = self._tracker_next[0]
                    self._tracker_next = None
                    self._sink.on_active_objects(self._active_path())
                frames = self._frames.get(self._tracked, [])
                publish_tf = self.objects_present and self.tracker_running and frames and now >= next_tf
                joints = dict(self._joints) if now >= next_joints else None
            if publish_tf:
                self._sink.on_tf(frames)
                next_tf = now + self._timing.tf
            if joints is not None:
                self._sink.on_joints(list(joints), list(joints.values()))
                next_joints = now + self._timing.joints
            time.sleep(0.002)


class SimIO(object):
    """The bridge's io for a SimSupervisor whose sink is the bridge's RobotState."""

    def __init__(self, sim, state):
        self._sim = sim
        self._state = state

    def set_task(self, task):
        return self._sim.set_task(task)

    def start(self):
        return self._sim.start()

    def publish_log(self, line):
        self._state.on_log(line)

    def task_info(self, task, robot=None, current=False):
        return self._sim.task_info(task, robot, current)
