"""A bridge on a fast simulated robot, for the tests."""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from voice_bridge.bridge import Settings, VoiceBridge  # noqa: E402
from voice_bridge.sim import SimIO, SimSupervisor, Timing, sim_tasks  # noqa: E402
from voice_bridge.state import RobotState  # noqa: E402
from voice_bridge.tasks import VoiceTask, VoiceTasks  # noqa: E402

VOICE_TASKS = ["grasp-cube", "stack-cube", "putintobox-s102-wx-a-h22-s1", "putballintobox-s15-pb-dr-s1",
               "mug-ball", "mug-turn", "mug-coaster"]


def fast_settings(**overrides):
    """The Settings scaled to Timing(scale=0.1)."""
    values = dict(readiness_wait=0.5, joint_timeout=0.2, tf_timeout=0.2, switch_timeout=3.0,
                  confirm_timeout=0.5, home_timeout=3.0, idle_timeout=1.0, observer_delay=0.06,
                  quiet_wait=0.3, ready_timeout=1.5, start_timeout=2.0)
    values.update(overrides)
    return Settings(**values)


class Rig(object):
    """A RobotState, a SimSupervisor feeding it and a VoiceBridge on both. timing: Timing
    overrides before the 0.1 scale; edit_tasks(tasks) may change the SimTasks first."""

    def __init__(self, task="grasp-cube", sim_task_names=None, settings=None, edit_tasks=None, **timing):
        self.state = RobotState()
        values = dict(tf=0.3, joints=0.2)
        values.update(timing)
        tasks = sim_tasks(sim_task_names or VOICE_TASKS)
        if edit_tasks:
            edit_tasks(tasks)
        self.sim = SimSupervisor(self.state, tasks, task, Timing(scale=0.1, **values))
        voice_tasks = VoiceTasks([VoiceTask(name, "description of " + name) for name in VOICE_TASKS])
        self.bridge = VoiceBridge(self.state, SimIO(self.sim, self.state), voice_tasks, settings or fast_settings())
        self.state.wait(lambda: self.state.readiness is not None and self.state.joints_time is not None
                        and self.state.readiness.get("ready"), 2.0)

    def close(self):
        self.sim.close()

    def voice_lines(self):
        return [line for line in self.state.log_since(0) if line.startswith("[voice]")]

    def supervisor_lines(self):
        return [line for line in self.state.log_since(0) if not line.startswith("[voice]")]

    def in_thread(self, function, *args):
        box = {}
        thread = threading.Thread(target=lambda: box.setdefault("result", function(*args)))
        thread.start()
        return thread, box

    def start_in_thread(self, task):
        return self.in_thread(self.bridge.start_task, task)

    def wait_state(self, state, timeout=3.0):
        return self.state.wait(lambda: self.state.state == state, timeout)
