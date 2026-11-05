"""What the bridge knows about the robot: the latest supervisor, tracker and joint messages.

ros_io (or sim) feeds it; every update wakes the threads in wait(). The fields are read with
the lock held, inside wait() predicates or in snapshot().
"""
import collections
import json
import threading
import time


class RobotState(object):

    def __init__(self, clock=time.monotonic, log_size=1000):
        self._clock = clock
        self._cond = threading.Condition()  # An RLock: predicates may call the methods below.
        self.state = None           # ~state: "idle", "policy" or "home"
        self.task = None            # ~task: {"task", "robot", "config_path"}
        self.tasks = None           # ~tasks: the task names ~set_task accepts
        self.readiness = None       # ~readiness: {"ready", "reasons", "notes"}
        self.readiness_time = None
        self.joints = {}            # joint name -> position
        self.joints_time = None
        self.tf_time = {}           # TF child frame -> receive time
        self.active_objects = None  # the tracker's objects YAML; "" while it is idle
        self._log = collections.deque(maxlen=log_size)  # (seq, line)
        self._seq = 0

    def now(self):
        return self._clock()

    def age(self, stamp):
        return None if stamp is None else self._clock() - stamp

    def _update(self, change):
        with self._cond:
            change()
            self._cond.notify_all()

    def on_state(self, state):
        self._update(lambda: setattr(self, "state", state))

    def on_task(self, data):
        task = _json(data, dict)
        if task is not None and isinstance(task.get("task"), str):
            self._update(lambda: setattr(self, "task", task))

    def on_tasks(self, data):
        tasks = _json(data, list)
        if tasks is not None:
            names = [entry["task"] for entry in tasks if isinstance(entry, dict) and "task" in entry]
            self._update(lambda: setattr(self, "tasks", names))

    def on_readiness(self, data):
        readiness = _json(data, dict)
        if readiness is None:
            return

        def change():
            self.readiness = readiness
            self.readiness_time = self._clock()
        self._update(change)

    def on_joints(self, names, positions):
        def change():
            self.joints.update(zip(names, positions))
            self.joints_time = self._clock()
        self._update(change)

    def on_tf(self, frames):
        def change():
            stamp = self._clock()
            for frame in frames:
                self.tf_time[frame] = stamp
        self._update(change)

    def on_active_objects(self, path):
        self._update(lambda: setattr(self, "active_objects", path))

    def on_log(self, line):
        def change():
            self._seq += 1
            self._log.append((self._seq, line))
        self._update(change)

    def log_mark(self):
        """The number of the latest log line; log_since(mark) returns the lines after it."""
        with self._cond:
            return self._seq

    def log_since(self, mark):
        with self._cond:
            return [line for seq, line in self._log if seq > mark]

    def task_name(self):
        with self._cond:
            return self.task["task"] if self.task else None

    def current(self):
        """(state, task name), read together."""
        with self._cond:
            return self.state, (self.task["task"] if self.task else None)

    def wait(self, predicate, timeout):
        """Wait until predicate() is true or timeout s passed; return its last value."""
        with self._cond:
            return self._cond.wait_for(predicate, timeout)

    def snapshot(self):
        with self._cond:
            return {
                "state": self.state,
                "task": dict(self.task) if self.task else None,
                "tasks": list(self.tasks) if self.tasks is not None else None,
                "readiness": dict(self.readiness) if self.readiness else None,
                "readiness_age": self.age(self.readiness_time),
                "joints": dict(self.joints),
                "joints_age": self.age(self.joints_time),
                "tf_age": {frame: self.age(stamp) for frame, stamp in self.tf_time.items()},
                "active_objects": self.active_objects,
            }


def _json(data, kind):
    try:
        value = json.loads(data)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, kind) else None
