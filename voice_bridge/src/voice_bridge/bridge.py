"""Start a voice task on the policy supervisor.

start_task() checks that the robot may start, switches the task when it is not the current one,
waits for the verified start pose and a ready observation with the task's objects in view, then
starts the policy. One voice command switches and starts. The checks are here, in code, not in
the LLM prompt.

- It refuses while a policy or a home move runs: a voice command never stops anything. STOP is
  on the phone pages and the e-stop.
- It starts only from the start pose: the joints the home move drives must be within
  start_tolerance of the current task's start pose and of the new task's. So the home move of
  set_task hardly moves the arm. From anywhere else, tap Go home on a phone page first.
- Before set_task it waits for a ~readiness newer than the command: the supervisor publishes it
  only while its lock is free, so no start or set_task of a phone page is under way.
- The phone pages may act at any time; the last command wins. When the task or the state
  changes during a command without the bridge, the command ends as "overridden".
- Every outcome goes to the supervisor's ~log as a "[voice] ..." line; both phone pages show it.
"""
import os
import threading

from .configs import TaskInfo  # noqa: F401 (part of the io interface)

VOICE_PREFIX = "[voice]"
# policy_supervisor.py logs these lines; move_to_start_config.py prints the Start pose ones.
TASK_SWITCHED = "=== task {} ("
TASK_ANY = "=== task "
HOME_STARTED = "=== home ==="
HOME_EXITED = "=== home exited with code"
STOPPING = "stopping "
# A STOP while nothing runs logs nothing in policy_supervisor.py. With this line in its stop()
# the bridge also notices a STOP while it waits for the objects (see the README).
STOP_REQUESTED = "=== stop requested ==="
HOME_VERIFIED = "Start pose verified"
HOME_FAILED = "Start pose move failed"
UNSURE = ("start_unknown",)  # Codes that do not mean the command was refused.


class Settings(object):
    """Limits in s and rad. The defaults suit policy_supervisor.py's timing."""

    def __init__(self, **overrides):
        self.readiness_wait = 1.5         # ~readiness comes at 2 Hz while the supervisor's lock is free
        self.joint_timeout = 0.5
        self.tf_timeout = 0.5             # the tracker publishes at about 30 Hz
        self.start_tolerance = 0.1        # the home move's position_tolerance (home_args)
        self.ignore_joints = ["gripper"]  # joints whose names contain one of these are not checked
        self.switch_timeout = 60.0
        self.confirm_timeout = 3.0
        self.home_timeout = 45.0
        self.idle_timeout = 10.0
        self.observer_delay = 0.6         # a readiness older than this after the home move is stale
        self.quiet_wait = 2.0             # without a switch, the whole wait for the objects
        self.ready_timeout = 15.0         # after a switch, the wait for the objects
        self.start_timeout = 35.0         # ~start stops the observer first: up to 25 s
        for name, value in overrides.items():
            if not hasattr(self, name):
                raise ValueError("unknown setting {}".format(name))
            setattr(self, name, value)

    def total(self):
        """About the longest a command can take, in s."""
        return (self.readiness_wait + self.switch_timeout + 2 * self.confirm_timeout + self.home_timeout
                + self.idle_timeout + self.ready_timeout + self.start_timeout)


class Result(object):

    def __init__(self, ok, code, message, task=None):
        self.ok = ok
        self.code = code
        self.message = message
        self.task = task

    def to_dict(self):
        return {"ok": self.ok, "code": self.code, "message": self.message, "task": self.task}


class CallTimeout(Exception):
    pass


class VoiceBridge(object):
    """io: set_task(task) and start() -> (success, message), publish_log(line),
    task_info(task, robot, current) -> TaskInfo (robot None: the task's own; current: the task
    is the supervisor's current one)."""

    def __init__(self, state, io, tasks, settings=None):
        self._state = state
        self._io = io
        self._tasks = tasks
        self._settings = settings or Settings()
        self._op = threading.Lock()
        self._pending = None  # A service call that outlived its timeout; no new call until it ends.
        self.last_result = None

    def start_task(self, name):
        """Run one voice command to its end and return the Result."""
        voice_task = self._tasks.find(name)
        if voice_task is None:
            return self._finish(Result(False, "unknown_task", "unbekannte Aufgabe {!r}".format(name)))
        task = voice_task.task
        if not self._op.acquire(False):
            return self._finish(Result(False, "busy", "ein anderer Sprachbefehl läuft noch", task))
        try:
            if self._pending is not None and self._pending.is_alive():
                result = Result(False, "busy", "ein früherer Aufruf beim Supervisor hängt noch", task)
            else:
                result = self._run(task)
        except Exception as error:  # A bug or a broken config must not end the server thread.
            result = Result(False, "error", "interner Fehler: {}".format(error), task)
        finally:
            self._op.release()
        return self._finish(result)

    def _run(self, task):
        settings = self._settings
        begin = self._state.now()
        mark = self._state.log_mark()
        refusal = self._check_start(task, begin)
        if refusal:
            return refusal
        if self._state.task_name() != task:
            refusal = self._switch(task, mark)
            if refusal:
                return refusal
            since, timeout = self._state.now() + settings.observer_delay, settings.ready_timeout
        else:
            since, timeout = begin, settings.quiet_wait
        snap = self._state.snapshot()
        robot = snap["task"].get("robot") if snap["task"] and snap["task"]["task"] == task else None
        try:
            info = self._io.task_info(task, robot, True)
        except Exception as error:
            return Result(False, "config_error", "kann die Konfiguration von {} nicht lesen: {}".format(
                task, error), task)
        refusal = self._wait_ready(task, info, mark, since, timeout)
        if refusal:
            return refusal
        return self._start(task, info, mark, since)

    def _check_start(self, task, begin):
        settings = self._settings
        fresh = self._state.wait(lambda: self._state.readiness_time is not None
                                 and self._state.readiness_time > begin, settings.readiness_wait)
        if not fresh:
            return Result(False, "supervisor_busy", "der Supervisor antwortet nicht oder ist beschäftigt "
                          "(läuft gerade ein Start oder Wechsel?)", task)
        snap = self._state.snapshot()
        if snap["task"] is None or snap["state"] is None:
            return Result(False, "supervisor_down", "der Supervisor hat noch keinen Zustand gemeldet", task)
        reasons = (snap["readiness"] or {}).get("reasons") or []
        if snap["state"] == "policy" or "policy running" in reasons:
            return Result(False, "policy_running",
                          "es läuft schon eine Policy; erst STOP auf einer Handy-Seite", task)
        if snap["state"] == "home" or "home running" in reasons:
            return Result(False, "home_running", "der Arm fährt gerade in die Startposition", task)
        if snap["state"] != "idle":
            return Result(False, "supervisor_down", "unbekannter Zustand {!r}".format(snap["state"]), task)
        if snap["tasks"] is not None and task not in snap["tasks"]:
            return Result(False, "not_switchable",
                          "der Supervisor kann nicht zu {} wechseln".format(task), task)
        current = snap["task"]
        try:
            info = self._io.task_info(current["task"], current.get("robot"), True)
            target = info if task == current["task"] else self._io.task_info(task, None, False)
        except Exception as error:
            return Result(False, "config_error", "kann die Startpose nicht lesen: {}".format(error), task)
        problem = self._pose_problem(info, snap)
        if problem:
            return Result(False, problem[0], problem[1], task)
        if target is not info:
            joint, error = start_pose_error(target.start_pose, snap["joints"], settings.ignore_joints)
            if joint is None or error is None or error > settings.start_tolerance:
                return Result(False, "other_start_pose", "die Startpose von {} ist eine andere ({}{}); "
                              "per Sprache nur zwischen Aufgaben mit derselben Startpose".format(
                                  task, joint or "keine Gelenke",
                                  " um {:.2f} rad".format(error) if error is not None else ""), task)
        return None

    def _pose_problem(self, info, snap):
        """(code, message) when the arm is not at info's start pose, else None."""
        age = snap["joints_age"]
        if age is None or age > self._settings.joint_timeout:
            return "no_joint_state", "keine aktuellen Gelenkwinkel"
        joint, error = start_pose_error(info.start_pose, snap["joints"], self._settings.ignore_joints)
        if joint is None:
            return "config_error", "die Startpose hat keine prüfbaren Gelenke"
        if error is None:
            return "no_joint_state", "kein Gelenkwinkel für {}".format(joint)
        if error > self._settings.start_tolerance:
            return "not_at_start", ("der Arm steht nicht in der Startposition ({} um {:.2f} rad daneben); "
                                    "erst Go home auf einer Handy-Seite".format(joint, error))
        return None

    def _switch(self, task, mark):
        settings = self._settings
        self._say(task, "wechsle die Aufgabe")
        try:
            ok, message = self._call(self._io.set_task, (task,), settings.switch_timeout)
        except CallTimeout:
            return Result(False, "switch_timeout", "set_task antwortet nicht innerhalb von {:.0f} s".format(
                settings.switch_timeout), task)
        except Exception as error:
            return Result(False, "switch_failed", "set_task fehlgeschlagen: {}".format(error), task)
        if not ok:
            return Result(False, "switch_refused", "Wechsel abgelehnt: {}".format(message), task)
        # The supervisor publishes ~task before set_task returns.
        if not self._state.wait(lambda: self._state.task_name() == task, settings.confirm_timeout):
            return self._overridden(task)
        return self._wait_home(task, mark)

    def _wait_home(self, task, mark):
        settings = self._settings
        outcome = [None]

        def ended():
            outcome[0] = self._home_outcome(task, mark)
            return outcome[0] is not None
        if not self._state.wait(ended, settings.home_timeout):
            return Result(False, "home_timeout", "Startposition nicht innerhalb von {:.0f} s erreicht".format(
                settings.home_timeout), task)
        kind, line = outcome[0]
        if kind == "overridden":
            return self._overridden(task, line)
        if kind == "stopped":
            # A ~stop ends in idle (the stopped job's exit); a phone's set_task publishes another
            # ~task instead, after it has switched the tracker and the controllers.
            self._state.wait(lambda: self._state.task_name() != task or self._state.current()[0] == "idle",
                             settings.switch_timeout)
            if self._state.task_name() != task:
                return self._overridden(task)
            return Result(False, "home_stopped", "die Fahrt in die Startposition wurde gestoppt", task)
        if kind == "failed":
            return Result(False, "home_failed",
                          "Startposition nicht erreicht: {}".format(line.strip()), task)
        # Verified; the home move exits next and the state turns idle.
        if not self._state.wait(lambda: self._state.current()[0] != "home", settings.idle_timeout):
            return Result(False, "home_timeout", "die Fahrt in die Startposition endet nicht", task)
        if self._state.current() != ("idle", task):
            return self._overridden(task)
        return None

    def _home_outcome(self, task, mark):
        """(kind, line) once the home move of the bridge's set_task ended or was overridden,
        else None. Lines before the supervisor's "=== task <task> (" line are another client's."""
        own = TASK_SWITCHED.format(task)
        switched = home = False
        for line in self._state.log_since(mark):
            if line.startswith(VOICE_PREFIX):
                continue
            text = line.strip()
            if not switched:
                if text.startswith(own):
                    switched = True
                elif text.startswith(STOPPING) or text.startswith(TASK_ANY):
                    return "overridden", line  # A phone page switched or stopped at the same time.
                continue
            if text == HOME_STARTED:
                home = True
            elif not home:
                continue
            elif HOME_VERIFIED in line:
                return "verified", line
            elif HOME_FAILED in line or text.startswith(HOME_EXITED):
                return "failed", line
            elif text.startswith(STOPPING) or text == STOP_REQUESTED:
                return "stopped", line
        state, current = self._state.current()
        if current != task or state == "policy":
            return "overridden", None
        return None

    def _outside_stop(self, mark):
        """A STOP or a stopped job after mark, which the bridge did not cause, else None."""
        for line in self._state.log_since(mark):
            text = line.strip()
            if not line.startswith(VOICE_PREFIX) and (text == STOP_REQUESTED or text.startswith(STOPPING)):
                return line
        return None

    def _wait_ready(self, task, info, mark, since, timeout):
        settings = self._settings
        problem = [None]

        def ended():
            if self._state.current() != ("idle", task) or self._outside_stop(mark):
                return True
            problem[0] = self._ready_problem(task, info, since)
            return problem[0] is None
        quiet = min(settings.quiet_wait, timeout)
        ready = self._state.wait(ended, quiet)
        if not ready and timeout > quiet:
            self._say(task, "warte: {}".format(problem[0]))
            ready = self._state.wait(ended, timeout - quiet)
        if self._outside_stop(mark):
            return Result(False, "stopped", "STOP auf einer Handy-Seite; nicht gestartet", task)
        if self._state.current() != ("idle", task):
            return self._overridden(task)
        if not ready:
            return Result(False, "not_ready", "nach {:.0f} s nicht bereit: {}".format(timeout, problem[0]), task)
        return None

    def _ready_problem(self, task, info, since):
        """Why the task may not start yet, or None."""
        snap = self._state.snapshot()
        pose = self._pose_problem(info, snap)
        if pose:
            return pose[1]
        missing = [frame for frame in info.object_frames
                   if snap["tf_age"].get(frame) is None or snap["tf_age"][frame] > self._settings.tf_timeout]
        if missing:
            return "ich sehe {} nicht".format(", ".join(missing))
        if (info.tracker_objects and snap["active_objects"] is not None
                and not same_objects(snap["active_objects"], info.tracker_objects)):
            return "der Tracker verfolgt noch nicht die Objekte von {}".format(task)
        readiness, age = snap["readiness"], snap["readiness_age"]
        if readiness is None or age is None or self._state.now() - age < since:
            return "warte auf eine neue Bereitschaftsmeldung des Supervisors"
        if not readiness.get("ready"):
            return "nicht bereit: {}".format("; ".join(readiness.get("reasons") or []) or "ohne Grund")
        return None

    def _start(self, task, info, mark, since):
        settings = self._settings
        if self._state.current() != ("idle", task):
            return self._overridden(task)
        if self._outside_stop(mark):
            return Result(False, "stopped", "STOP auf einer Handy-Seite; nicht gestartet", task)
        problem = self._ready_problem(task, info, since)
        if problem:
            return Result(False, "not_ready", "nicht bereit: {}".format(problem), task)
        try:
            ok, message = self._call(self._io.start, (), settings.start_timeout)
        except CallTimeout:
            return Result(False, "start_unknown", "keine Antwort von ~start nach {:.0f} s; die Policy "
                          "startet eventuell noch, Roboter beobachten".format(settings.start_timeout), task)
        except Exception as error:
            return Result(False, "start_failed", "start fehlgeschlagen: {}".format(error), task)
        if not ok:
            return Result(False, "start_refused", "Start abgelehnt: {}".format(message), task)
        self._state.wait(lambda: self._state.current()[0] != "idle", settings.confirm_timeout)
        if self._state.task_name() != task:
            return self._overridden(task)
        return Result(True, "started", "gestartet", task)

    def _overridden(self, task, line=None):
        state, current = self._state.current()
        if line:
            message = "eine Handy-Seite hat gleichzeitig gewechselt oder gestoppt; {} ist nicht gestartet".format(task)
        elif current != task:
            message = "die Aufgabe wurde von außen gewechselt (jetzt {})".format(current)
        elif state == "policy":
            message = "eine Policy wurde von außen gestartet"
        else:
            message = "der Zustand wurde von außen geändert ({})".format(state)
        return Result(False, "overridden", message, task)

    def _call(self, function, args, timeout):
        """function(*args) in a thread; CallTimeout when it has not returned after timeout s.
        rospy service calls have no timeout of their own."""
        box = {}

        def run():
            try:
                box["value"] = function(*args)
            except Exception as error:
                box["error"] = error
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self._pending = thread
            raise CallTimeout()
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _say(self, task, text):
        line = "{} {}: {}".format(VOICE_PREFIX, task, text) if task else "{} {}".format(VOICE_PREFIX, text)
        try:
            self._io.publish_log(line)
        except Exception:  # The log is for people; a failed line must not end a command.
            pass

    def _finish(self, result):
        if result.ok or result.code in UNSURE:
            self._say(result.task, result.message)
        else:
            self._say(None, "{} abgelehnt: {}".format(result.task, result.message) if result.task
                      else "abgelehnt: {}".format(result.message))
        self.last_result = result
        return result

    def busy(self):
        return self._op.locked() or (self._pending is not None and self._pending.is_alive())

    def tasks(self):
        """The voice tasks; available: the supervisor can switch to it (None: not known yet)."""
        switchable = self._state.snapshot()["tasks"]
        return [dict(task.to_dict(), available=None if switchable is None else task.task in switchable)
                for task in self._tasks.tasks]

    def status(self):
        snap = self._state.snapshot()
        readiness = snap["readiness"] or {}
        status = {
            "state": snap["state"],
            "task": snap["task"]["task"] if snap["task"] else None,
            "ready": readiness.get("ready"),
            "reasons": readiness.get("reasons", []),
            "readiness_age": snap["readiness_age"],
            "busy": self.busy(),
            "at_start": None,
            "start_error": None,
            "objects": {},
            "last_result": self.last_result.to_dict() if self.last_result else None,
        }
        if snap["task"]:
            try:
                info = self._io.task_info(snap["task"]["task"], snap["task"].get("robot"), True)
            except Exception:
                return status
            joint, error = start_pose_error(info.start_pose, snap["joints"], self._settings.ignore_joints)
            if error is not None and snap["joints_age"] is not None \
                    and snap["joints_age"] <= self._settings.joint_timeout:
                status["at_start"] = error <= self._settings.start_tolerance
                status["start_error"] = {"joint": joint, "error": error}
            status["objects"] = {frame: snap["tf_age"].get(frame) for frame in info.object_frames}
        return status


def start_pose_error(start_pose, joints, ignore):
    """(joint, error) of the checked joint farthest from the start pose; (joint, None) when a
    checked joint has no position; (None, None) when no joint is checked."""
    worst, worst_error = None, -1.0
    for name in sorted(start_pose):
        if any(part in name for part in ignore):
            continue
        if name not in joints:
            return name, None
        error = abs(float(joints[name]) - float(start_pose[name]))
        if error > worst_error:
            worst, worst_error = name, error
    return (worst, worst_error) if worst is not None else (None, None)


def same_objects(active, tracker_objects):
    """Whether the tracker's active objects YAML (absolute) is the task's (relative to the
    tracker's ROOT_DIR, or absolute)."""
    active = os.path.normpath(active)
    wanted = os.path.normpath(tracker_objects)
    return active == wanted or active.endswith(os.sep + wanted.lstrip(os.sep))
