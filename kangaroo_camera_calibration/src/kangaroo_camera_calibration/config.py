"""Load config/calibration.yaml (or ~config) for one camera.

Resolves $(find kangaroo_camera_calibration), selects cameras/<camera> as cfg["camera"]
(with its name under "name") and fills {camera} in collect/poses_file.
"""

import os

import rospkg
import rospy
import yaml

PACKAGE = "kangaroo_camera_calibration"


def package_path():
    return rospkg.RosPack().get_path(PACKAGE)


def load(path=None, camera=None):
    if path is None:
        path = rospy.get_param("~config", os.path.join(package_path(), "config", "calibration.yaml"))
    with open(path) as stream:
        cfg = yaml.safe_load(stream.read().replace("$(find {})".format(PACKAGE), package_path()))
    camera = camera or rospy.get_param("~camera", "") or cfg["default_camera"]
    if camera not in cfg["cameras"]:
        raise ValueError("unknown camera '{}'; configured: {}".format(camera, ", ".join(sorted(cfg["cameras"]))))
    cfg["camera"] = dict(cfg["cameras"][camera], name=camera)
    cfg["collect"]["poses_file"] = cfg["collect"]["poses_file"].format(camera=camera)
    return cfg
