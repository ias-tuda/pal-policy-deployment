#!/usr/bin/env python3
"""Resolve a task name to the deployment config of a run.

A robot (config/robots/<robot>.yaml) is a robot part and how it is driven: joints, topics,
gravity compensation. A task (config/tasks/<task>.yaml) is what the robot is asked to do with
it: the model, the instruction and scene, where the object and the goal are. Several tasks can
share one robot. A node takes ~task (a name or a path) and optionally ~robot (a name or path
that replaces the task's robot).

The deployment config is the robot with the task's keys on top: `model:` is lifted to the top
level, `supervisor:` is merged key by key, every other key overrides. It is written to a file,
so that launch files and nodes, which take one config path, need no changes. In a shell:
P=$(rosrun pal_policy_deployer task_config.py <task>)
"""
import os
import sys
import tempfile

import rospkg
import rospy
import yaml

CONFIG = os.path.join(rospkg.RosPack().get_path("pal_policy_deployer"), "config")
TASK_ONLY = ("robot", "instruction", "scene", "name", "record_task")  # Not copied into the deployment config.


def _path(kind, name):
    if name.endswith(".yaml") or os.sep in name:
        return os.path.expanduser(name)
    return os.path.join(CONFIG, kind, name + ".yaml")


def _load(path):
    with open(path) as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("{} is not a mapping".format(path))
    return data


def load_task(name):
    task = _load(_path("tasks", name))
    task["name"] = os.path.basename(name[:-5] if name.endswith(".yaml") else name)
    return task


def deployment_config(task, robot_path):
    """Write the robot with the task's keys on top to a file and return its path."""
    config = _load(robot_path)
    overrides = dict(task.get("model") or {})
    overrides.update((key, value) for key, value in task.items()
                     if key not in TASK_ONLY and key != "model")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = dict(config[key], **value)
        else:
            config[key] = value
    directory = os.path.join(tempfile.gettempdir(), "pal_policy_configs")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "{}__{}.yaml".format(
        task["name"], os.path.basename(robot_path)[:-5]))
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as stream:
        stream.write("# Generated from {} and task {}. Do not edit.\n".format(
            robot_path, task["name"]))
        yaml.safe_dump(config, stream, default_flow_style=None)
    os.replace(stream.name, path)
    return path


def resolve(task_name="", robot_name=""):
    """Return (task, deployment config path) for a task name and optionally another robot."""
    if not task_name:
        raise ValueError("give a task (config/tasks/)")
    task = load_task(task_name)
    robot_name = robot_name or task.get("robot", "")
    if not robot_name:
        raise ValueError("task {} names no robot (config/robots/)".format(task["name"]))
    return task, deployment_config(task, _path("robots", robot_name))


def node_config():
    """(task, config_path, config) from the node's ~task and ~robot params."""
    task, config_path = resolve(rospy.get_param("~task", ""), rospy.get_param("~robot", ""))
    with open(config_path) as stream:
        return task, config_path, yaml.safe_load(stream)


def setting(name, default, config, task=None):
    """A supervisor setting: the ROS param, else the task's, else the config's, else default."""
    if rospy.has_param("~" + name):
        return rospy.get_param("~" + name)
    for source in (task or {}, config):
        if name in (source.get("supervisor") or {}):
            return source["supervisor"][name]
    return default


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: task_config.py <task> [robot]  (prints the deployment config path)")
    print(resolve(*sys.argv[1:])[1])
