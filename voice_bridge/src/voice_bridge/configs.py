"""What the bridge checks for a task, read from pal_policy_deployer's task, robot and start
config files the way task_config.py and move_to_start_config.py read them. No ROS needed."""
import json
import os
import collections

# start_pose: joint -> position of the joints the home move drives (the robot's joint_names);
# object_frames: the TF frames the policy observes; tracker_objects: the task's tracker objects
# YAML, or None.
TaskInfo = collections.namedtuple("TaskInfo", "start_pose object_frames tracker_objects")

TASK_ONLY = ("robot", "instruction", "scene", "name", "record_task", "tracker_objects")  # As task_config.py.
DEFAULT_JOINTS = ["arm_left_{}_joint".format(index) for index in range(1, 8)] + ["gripper_left_finger_joint"]


def deployment_config(task, robot):
    """The robot with the task's keys on top, as task_config.deployment_config writes it."""
    config = dict(robot)
    overrides = dict(task.get("model") or {})
    overrides.update((key, value) for key, value in task.items() if key not in TASK_ONLY and key != "model")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = dict(config[key], **value)
        else:
            config[key] = value
    return config


def task_info(task, robot, start):
    """TaskInfo from the parsed task file, robot file and deployment_start_config.json. The start
    config lists every joint of the robot; the home move drives only the robot's joint_names."""
    config = deployment_config(task, robot)
    names = config.get("joint_names") or DEFAULT_JOINTS
    positions = start.get("joint_positions") or {}
    missing = [name for name in names if name not in positions]
    if missing:
        raise ValueError("the start config has no {}".format(", ".join(missing)))
    frames = [config["object_frame"]] if config.get("object_frame") else []
    if config.get("base_object_frame") and config.get("track_base_object"):
        frames.append(config["base_object_frame"])
    return TaskInfo({name: float(positions[name]) for name in names}, frames, task.get("tracker_objects"))


def load_task_info(config_dir, task_name, robot_name="", start_config_path=""):
    """TaskInfo of a task in config_dir/tasks/ with its robot (or robot_name, a name or path)
    and the start config beside the model (or start_config_path)."""
    task = _load(_path(config_dir, "tasks", task_name))
    robot_name = robot_name or task.get("robot", "")
    if not robot_name:
        raise ValueError("task {} names no robot".format(task_name))
    robot = _load(_path(config_dir, "robots", robot_name))
    model_path = deployment_config(task, robot).get("model_path")
    if not start_config_path and not model_path:
        raise ValueError("task {} has no model_path".format(task_name))
    start_path = start_config_path or os.path.join(os.path.dirname(model_path), "deployment_start_config.json")
    with open(start_path, encoding="utf-8") as stream:
        start = json.load(stream)
    return task_info(task, robot, start)


def _path(config_dir, kind, name):
    if name.endswith(".yaml") or os.sep in name:
        return os.path.expanduser(name)
    return os.path.join(config_dir, kind, name + ".yaml")


def _load(path):
    import yaml  # Only here, so that the tests of the other functions run without PyYAML.
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("{} is not a mapping".format(path))
    return data
