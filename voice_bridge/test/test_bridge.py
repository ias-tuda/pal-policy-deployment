"""The voice bridge against the simulated supervisor (voice_bridge/sim.py)."""
import time
import unittest

from helpers import Rig, fast_settings

from voice_bridge.bridge import VoiceBridge, same_objects, start_pose_error
from voice_bridge.state import RobotState
from voice_bridge.tasks import VoiceTask, VoiceTasks


class StartTask(unittest.TestCase):

    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.close()

    def test_switches_and_starts_in_one_command(self):
        result = self.rig.bridge.start_task("stack-cube")
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.code, "started")
        self.assertEqual(self.rig.state.current(), ("policy", "stack-cube"))
        lines = self.rig.voice_lines()
        self.assertIn("[voice] stack-cube: wechsle die Aufgabe", lines)
        self.assertEqual(lines[-1], "[voice] stack-cube: gestartet")
        self.assertFalse(any(line.startswith("stopping ") for line in self.rig.supervisor_lines()))

    def test_switches_across_groups(self):
        result = self.rig.bridge.start_task("putintobox-s102-wx-a-h22-s1")
        self.assertTrue(result.ok, result.message)

    def test_starts_the_current_task_without_a_switch(self):
        result = self.rig.bridge.start_task("grasp-cube")
        self.assertTrue(result.ok, result.message)
        self.assertFalse(any(line.startswith("=== task") for line in self.rig.state.log_since(0)))

    def test_task_names_ignore_case(self):
        self.assertTrue(self.rig.bridge.start_task(" Grasp-Cube ").ok)

    def test_refuses_an_unknown_task(self):
        result = self.rig.bridge.start_task("make-coffee")
        self.assertEqual(result.code, "unknown_task")
        self.assertEqual(self.rig.voice_lines()[-1], "[voice] abgelehnt: unbekannte Aufgabe 'make-coffee'")

    def test_refuses_while_a_policy_runs(self):
        self.assertTrue(self.rig.bridge.start_task("grasp-cube").ok)
        result = self.rig.bridge.start_task("stack-cube")
        self.assertEqual(result.code, "policy_running")
        self.assertEqual(self.rig.state.current(), ("policy", "grasp-cube"))  # Nothing was stopped.
        self.assertTrue(self.rig.voice_lines()[-1].startswith("[voice] stack-cube abgelehnt: "))

    def test_refuses_while_the_arm_moves_home(self):
        self.rig.sim.move_arm(0.3)
        self.rig.sim.go_home()
        self.assertEqual(self.rig.bridge.start_task("stack-cube").code, "home_running")

    def test_refuses_away_from_the_start_pose(self):
        self.rig.sim.move_arm(0.3)
        time.sleep(0.1)
        result = self.rig.bridge.start_task("stack-cube")
        self.assertEqual(result.code, "not_at_start")
        self.assertIn("arm_left_1_joint", result.message)
        self.assertEqual(self.rig.state.task_name(), "grasp-cube")  # No switch.

    def test_after_a_run_the_arm_must_go_home_first(self):
        self.assertTrue(self.rig.bridge.start_task("grasp-cube").ok)
        self.rig.sim.stop()
        time.sleep(0.1)
        self.assertEqual(self.rig.bridge.start_task("grasp-cube").code, "not_at_start")
        self.rig.sim.go_home()
        self.assertTrue(self.rig.wait_state("home") and self.rig.wait_state("idle"))
        time.sleep(0.1)
        self.assertTrue(self.rig.bridge.start_task("grasp-cube").ok)

    def test_after_a_switch_refuses_without_the_objects(self):
        self.rig.sim.objects_present = False
        time.sleep(0.3)
        result = self.rig.bridge.start_task("stack-cube")
        self.assertEqual(result.code, "not_ready")
        self.assertIn("cube_1", result.message)
        self.assertTrue(any(line.startswith("[voice] stack-cube: warte: ") for line in self.rig.voice_lines()))
        self.assertEqual(self.rig.state.current(), ("idle", "stack-cube"))  # Switched, not started.

    def test_without_a_switch_refuses_missing_objects_at_once(self):
        self.rig.sim.objects_present = False
        time.sleep(0.3)
        begin = time.monotonic()
        result = self.rig.bridge.start_task("grasp-cube")
        self.assertEqual(result.code, "not_ready")
        self.assertLess(time.monotonic() - begin, 1.2)  # quiet_wait 0.3 s, not ready_timeout
        self.assertFalse(any(": warte: " in line for line in self.rig.voice_lines()))

    def test_after_a_switch_starts_when_the_objects_appear_in_time(self):
        self.rig.sim.objects_present = False
        thread, box = self.rig.start_in_thread("stack-cube")
        time.sleep(0.6)
        self.rig.sim.objects_present = True
        thread.join(5)
        self.assertTrue(box["result"].ok, box["result"].message)

    def test_waits_for_the_tracker_to_take_the_new_objects(self):
        # grasp-cube and putintobox share cube_1: its TF alone does not show the new objects.
        rig = Rig(tracker=8.0)  # 0.8 s
        try:
            result = rig.bridge.start_task("putintobox-s102-wx-a-h22-s1")
            self.assertTrue(result.ok, result.message)
            self.assertTrue(rig.state.active_objects.endswith("box-cube.yaml"))
        finally:
            rig.close()

    def test_an_arm_moved_while_waiting_is_not_started(self):
        self.rig.sim.objects_present = False
        thread, box = self.rig.start_in_thread("stack-cube")
        time.sleep(0.7)
        self.rig.sim.move_arm(0.13)
        self.rig.sim.objects_present = True
        thread.join(5)
        self.assertEqual(box["result"].code, "not_ready")
        self.assertIn("Startposition", box["result"].message)

    def test_a_failed_home_move_refuses(self):
        self.rig.sim.home_fails = True
        result = self.rig.bridge.start_task("stack-cube")
        self.assertEqual(result.code, "home_failed")
        self.assertIn("Start pose move failed", result.message)

    def test_a_refused_switch_refuses(self):
        self.rig.sim.switch_refusal = "a node other than the bridge still publishes"
        result = self.rig.bridge.start_task("stack-cube")
        self.assertEqual(result.code, "switch_refused")
        self.assertIn("still publishes", result.message)

    def test_a_hung_set_task_times_out_and_blocks_the_next_command(self):
        rig = Rig(settings=fast_settings(switch_timeout=0.3))
        try:
            rig.sim.ignore_set_task = True
            self.assertEqual(rig.bridge.start_task("stack-cube").code, "switch_timeout")
            self.assertEqual(rig.bridge.start_task("mug-turn").code, "busy")
        finally:
            rig.close()

    def test_a_slow_start_is_not_reported_as_refused(self):
        rig = Rig(start=20.0, settings=fast_settings(start_timeout=0.5))  # ~start takes 2 s
        try:
            result = rig.bridge.start_task("grasp-cube")
            self.assertEqual(result.code, "start_unknown")
            self.assertNotIn("abgelehnt", rig.voice_lines()[-1])
            self.assertTrue(rig.wait_state("policy"))
        finally:
            rig.close()

    def test_a_phone_stop_during_the_home_move_refuses(self):
        thread, box = self.rig.start_in_thread("stack-cube")
        self.assertTrue(self.rig.wait_state("home"))
        self.rig.sim.stop()
        thread.join(5)
        self.assertEqual(box["result"].code, "home_stopped")

    def test_a_second_command_while_one_runs_is_busy(self):
        self.rig.sim.objects_present = False
        thread, box = self.rig.start_in_thread("stack-cube")
        time.sleep(0.1)
        self.assertEqual(self.rig.bridge.start_task("mug-turn").code, "busy")
        thread.join(5)

    def test_refuses_a_task_the_supervisor_cannot_switch_to(self):
        rig = Rig(sim_task_names=["grasp-cube", "stack-cube"])
        try:
            self.assertEqual(rig.bridge.start_task("mug-turn").code, "not_switchable")
        finally:
            rig.close()

    def test_refuses_a_switch_to_another_start_pose(self):
        def edit(tasks):
            tasks["mug-turn"].start_pose["arm_left_4_joint"] += 0.5
        rig = Rig(edit_tasks=edit)
        try:
            result = rig.bridge.start_task("mug-turn")
            self.assertEqual(result.code, "other_start_pose")
            self.assertIn("arm_left_4_joint", result.message)
            self.assertEqual(rig.state.task_name(), "grasp-cube")
        finally:
            rig.close()

    def test_status_and_tasks(self):
        status = self.rig.bridge.status()
        self.assertEqual(status["state"], "idle")
        self.assertEqual(status["task"], "grasp-cube")
        self.assertTrue(status["at_start"])
        self.assertIn("cube_1", status["objects"])
        tasks = self.rig.bridge.tasks()
        self.assertEqual(len(tasks), 7)
        self.assertTrue(all(task["available"] for task in tasks))


class PhonePages(unittest.TestCase):
    """The phone pages act at any time; the last command wins, and the bridge says so."""

    def test_a_phone_switch_during_the_home_move_overrides(self):
        rig = Rig(switch=20.0)  # set_task holds the lock 2 s, as a real switch with a gain ramp
        try:
            thread, box = rig.start_in_thread("stack-cube")
            self.assertTrue(rig.wait_state("home", 5))
            rig.sim.set_task("mug-turn")
            thread.join(10)
            self.assertEqual(box["result"].code, "overridden")
            self.assertIn("mug-turn", box["result"].message)
        finally:
            rig.close()

    def test_a_phone_switch_while_waiting_for_objects_overrides(self):
        rig = Rig()
        try:
            rig.sim.objects_present = False
            thread, box = rig.start_in_thread("stack-cube")
            time.sleep(0.7)
            rig.sim.set_task("mug-turn")
            thread.join(5)
            self.assertEqual(box["result"].code, "overridden")
        finally:
            rig.close()

    def test_a_voice_command_during_a_phone_start_stops_nothing(self):
        rig = Rig(start=10.0)  # ~start holds the lock 1 s while it stops the observer
        try:
            phone, _ = rig.in_thread(rig.sim.start)
            time.sleep(0.1)
            result = rig.bridge.start_task("stack-cube")
            phone.join(5)
            self.assertIn(result.code, ("supervisor_busy", "policy_running"))
            self.assertEqual(rig.state.current(), ("policy", "grasp-cube"))
            self.assertFalse(any(line.startswith("stopping ") for line in rig.supervisor_lines()))
        finally:
            rig.close()

    def test_a_voice_command_during_a_phone_switch_stops_nothing(self):
        rig = Rig(switch=10.0)  # set_task holds the lock 1 s
        try:
            phone, _ = rig.in_thread(rig.sim.set_task, "mug-turn")
            time.sleep(0.1)
            result = rig.bridge.start_task("stack-cube")
            phone.join(5)
            self.assertIn(result.code, ("supervisor_busy", "home_running"))
            self.assertFalse(any(line.startswith("stopping ") for line in rig.supervisor_lines()))
        finally:
            rig.close()

    def test_a_phone_stop_while_waiting_for_objects_is_seen_when_the_supervisor_logs_it(self):
        rig = Rig()
        try:
            rig.sim.log_stop_requests = True
            rig.sim.objects_present = False
            thread, box = rig.start_in_thread("stack-cube")
            time.sleep(0.7)
            rig.sim.stop()
            rig.sim.objects_present = True
            thread.join(5)
            self.assertEqual(box["result"].code, "stopped")
            self.assertEqual(rig.state.current(), ("idle", "stack-cube"))
        finally:
            rig.close()


class WithoutSupervisor(unittest.TestCase):

    def test_refuses_when_the_supervisor_is_silent(self):
        rig = Rig()
        try:
            rig.sim.close()
            time.sleep(0.2)
            self.assertEqual(rig.bridge.start_task("stack-cube").code, "supervisor_busy")
        finally:
            rig.close()

    def test_refuses_before_any_message(self):
        lines = []

        class IO(object):
            def publish_log(self, line):
                lines.append(line)
        bridge = VoiceBridge(RobotState(), IO(), VoiceTasks([VoiceTask("grasp-cube", "d")]), fast_settings())
        self.assertEqual(bridge.start_task("grasp-cube").code, "supervisor_busy")
        self.assertEqual(len(lines), 1)


class Helpers(unittest.TestCase):

    def test_start_pose_error(self):
        pose = {"arm_left_1_joint": 0.1, "arm_left_2_joint": 1.0, "gripper_left_finger_joint": 0.048}
        joints = {"arm_left_1_joint": 0.15, "arm_left_2_joint": 0.7, "gripper_left_finger_joint": 0.0}
        joint, error = start_pose_error(pose, joints, ["gripper"])
        self.assertEqual(joint, "arm_left_2_joint")
        self.assertAlmostEqual(error, 0.3)
        self.assertEqual(start_pose_error(pose, {"arm_left_1_joint": 0.1}, ["gripper"]), ("arm_left_2_joint", None))
        self.assertEqual(start_pose_error({"gripper_left_finger_joint": 0.0}, joints, ["gripper"]), (None, None))

    def test_same_objects(self):
        wanted = "custom-assets/kang-grasp-cube/stack-cube.yaml"
        self.assertTrue(same_objects("/home/u/tracker/custom-assets/kang-grasp-cube/stack-cube.yaml", wanted))
        self.assertTrue(same_objects("/abs/stack-cube.yaml", "/abs/stack-cube.yaml"))
        self.assertFalse(same_objects("/home/u/tracker/custom-assets/kang-grasp-cube/my-stack-cube.yaml", wanted))
        self.assertFalse(same_objects("", wanted))


if __name__ == "__main__":
    unittest.main()
