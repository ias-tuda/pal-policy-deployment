"""Reading a task's start pose, frames and tracker objects from the pal_policy_deployer files."""
import json
import os
import shutil
import tempfile
import unittest

import helpers  # noqa: F401 (the import path)

from voice_bridge.bridge import start_pose_error
from voice_bridge.configs import DEFAULT_JOINTS, deployment_config, load_task_info, task_info

try:
    import yaml
except ImportError:
    yaml = None

ARM_START = [0.102, 1.452, 1.777, 1.670, 0.458, 0.404, -0.009]
ROBOT = {"joint_names": DEFAULT_JOINTS, "object_frame": "object", "supervisor": {"ood_margin": 0.05}}
TASK = {"robot": "left_arm_gravcomp", "model": {"model_path": "/models/m/model.onnx"},
        "object_frame": "cube_1", "base_object_frame": "cube_2", "track_base_object": True,
        "tracker_objects": "custom-assets/kang-grasp-cube/stack-cube.yaml", "supervisor": {"object_timeout": 4.0}}


def full_start_config():
    """A start config as the models have it: all 44 joints, not only the ones the home move drives."""
    names = (["arm_left_{}_joint".format(i) for i in range(1, 8)] + ["arm_right_{}_joint".format(i) for i in range(1, 8)]
             + ["gripper_left_finger_joint", "gripper_right_finger_joint", "pelvis_1_joint", "pelvis_2_joint"]
             + ["leg_{}_{}".format(side, i) for side in ("left", "right") for i in range(1, 13)])
    positions = {name: 0.6 for name in names}
    positions.update(zip(DEFAULT_JOINTS, ARM_START + [0.048]))
    return {"schema_version": 1, "joint_names": names, "joint_positions": positions,
            "joint_velocities": {name: 0.0 for name in names}}


class TaskInfoFromFiles(unittest.TestCase):

    def test_the_deployment_config_is_the_robot_with_the_task_on_top(self):
        config = deployment_config(TASK, ROBOT)
        self.assertEqual(config["model_path"], "/models/m/model.onnx")
        self.assertEqual(config["object_frame"], "cube_1")
        self.assertEqual(config["supervisor"], {"ood_margin": 0.05, "object_timeout": 4.0})
        self.assertNotIn("tracker_objects", config)

    def test_only_the_joints_the_home_move_drives(self):
        info = task_info(TASK, ROBOT, full_start_config())
        self.assertEqual(sorted(info.start_pose), sorted(DEFAULT_JOINTS))
        self.assertEqual(info.object_frames, ["cube_1", "cube_2"])
        self.assertEqual(info.tracker_objects, "custom-assets/kang-grasp-cube/stack-cube.yaml")
        # The right arm and the legs elsewhere do not matter; the controller reports them too.
        joints = {name: 1.5 for name in full_start_config()["joint_positions"]}
        joints.update(zip(DEFAULT_JOINTS, ARM_START + [0.0]))
        joint, error = start_pose_error(info.start_pose, joints, ["gripper"])
        self.assertLess(error, 1e-9)

    def test_a_base_object_counts_only_when_tracked(self):
        task = dict(TASK, track_base_object=False)
        self.assertEqual(task_info(task, ROBOT, full_start_config()).object_frames, ["cube_1"])

    def test_a_joint_missing_in_the_start_config(self):
        start = full_start_config()
        del start["joint_positions"]["arm_left_3_joint"]
        with self.assertRaises(ValueError):
            task_info(TASK, ROBOT, start)

    @unittest.skipUnless(yaml, "needs PyYAML")
    def test_load_from_files(self):
        folder = tempfile.mkdtemp()
        try:
            for kind in ("tasks", "robots", "model"):
                os.makedirs(os.path.join(folder, kind))
            task = dict(TASK, model={"model_path": os.path.join(folder, "model", "model.onnx")})
            with open(os.path.join(folder, "tasks", "stack-cube.yaml"), "w") as stream:
                yaml.safe_dump(task, stream)
            with open(os.path.join(folder, "robots", "left_arm_gravcomp.yaml"), "w") as stream:
                yaml.safe_dump(ROBOT, stream)
            with open(os.path.join(folder, "model", "deployment_start_config.json"), "w") as stream:
                json.dump(full_start_config(), stream)
            info = load_task_info(folder, "stack-cube")
            self.assertEqual(len(info.start_pose), 8)
            other = os.path.join(folder, "other.json")
            start = full_start_config()
            start["joint_positions"]["arm_left_1_joint"] = 0.9
            with open(other, "w") as stream:
                json.dump(start, stream)
            self.assertEqual(load_task_info(folder, "stack-cube", "", other).start_pose["arm_left_1_joint"], 0.9)
            with self.assertRaises(OSError):
                load_task_info(folder, "mug-turn")
        finally:
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
