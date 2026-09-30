"""Resolve a task name to its task file and profile.

config/tasks/<task>.yaml names a profile (config/profiles/<profile>.yaml, the model contract),
the instruction and the scene. Several tasks can share one profile. A node takes ~task (a name
or a path) and optionally ~profile (a name or path that replaces the task's profile).
"""
import os

import rospkg
import rospy
import yaml

CONFIG = os.path.join(rospkg.RosPack().get_path("pal_policy_deployer"), "config")


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


def resolve(task_name="", profile_name=""):
    """Return (task or None, profile_path) for a task name and/or an explicit profile."""
    task = load_task(task_name) if task_name else None
    profile_name = profile_name or (task or {}).get("profile", "")
    if not profile_name:
        raise ValueError("give a task (config/tasks/) or a profile (config/profiles/)")
    return task, _path("profiles", profile_name)


def node_config():
    """(task or None, profile_path, profile) from the node's ~task and ~profile params."""
    task, profile_path = resolve(rospy.get_param("~task", ""), rospy.get_param("~profile", ""))
    with open(profile_path) as stream:
        return task, profile_path, yaml.safe_load(stream)


def setting(name, default, profile, task=None):
    """A supervisor setting: the ROS param, else the task's, else the profile's, else default."""
    if rospy.has_param("~" + name):
        return rospy.get_param("~" + name)
    for source in (task or {}, profile):
        if name in (source.get("supervisor") or {}):
            return source["supervisor"][name]
    return default
