#!/usr/bin/env python3
"""Record each policy run of policy_supervisor as one episode for oopsie-data.com, and label it.

The episodes are raw: a rosbag plus JSON sidecars. A converter writes the oopsie-data HDF5
format from them later. While recording is enabled, every run of the supervisor's policy job
is one episode, from the state "policy" until the state leaves it (STOP, guard trip, exit).

Layout under ~record_dir:
  <YYYY-MM-DD>/<episode_id>/                labelled or not yet labelled episodes
  excluded/<reason>/<YYYY-MM-DD>/<episode_id>/  not the policy's fault (e.g. tracking lost)
  discarded/<YYYY-MM-DD>/<episode_id>/      bad recordings
Each episode dir: episode.bag, meta.json, label.json (after labelling), params.yaml (the whole
parameter server at the start, with robot_description and the controller gains), policy_env.yaml
(the model's params/env.yaml) and calibration_<camera>.yaml (the latest camera calibration).

Services:
  ~set_enabled  std_srvs/SetBool
Topics:
  ~label   std_msgs/String JSON in, for the latest episode:
           {"episode", "outcome", "side_effect_category": [...], "severity", "tags": [...]},
           {"episode", "exclude": <EXCLUDE_REASONS>} or {"episode", "discard": true};
           {"add_tag": <text>} adds an own failure tag to <record_dir>/tags.json. Tags are
           details beyond the oopsie-data taxonomy; the converter maps them to "other" and
           additional_notes.
  ~status  std_msgs/String JSON, latched, 1 Hz: enabled, recording, the latest episode, the
           instruction and scene, the counts for the instruction and the last message
Params ~instruction and ~scene (a dict, e.g. {arrangement: A}) change rarely, so they are set in
the launch file, not on the page. oopsie-data counts a different instruction or scene layout
as a distinct task.

The recorder also logs gaps in ~observation_topic (e.g. object tracking lost), because
oopsie-data excludes episodes with missing frames unless the failure came first.
"""
import datetime
import glob
import json
import os
import re
import shlex
import shutil
import socket
import threading
import time

import rosgraph
import rospkg
import rospy
import yaml
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool, SetBoolResponse

from policy_supervisor import Job

OUTCOMES = ["success", "success_suboptimal", "success_side_effect", "failure"]
CATEGORIES = ["reaching", "grasp", "manipulation", "sequencing_semantic", "collision", "hardware",
              "not_attempted", "other"]
SEVERITIES = ["low", "medium", "catastrophic"]
EXCLUDE_REASONS = ["tracking_lost", "hardware", "setup_error", "other"]
TAXONOMY_SCHEMA = "oopsiedata_taxonomy_schema_v2"
DEFAULT_TOPICS = [
    "/joint_states", "/tf", "/tf_static", "/pal_policy/observation", "/pal_policy/raw_actions",
    "/pal_grasp/desired_state_uncompensated", "/pal_policy/desired_state",
    "/torso_front_rgbd_camera/color/image_raw/compressed",
    "/torso_front_rgbd_camera/color/camera_info", "/pal_policy_supervisor/log"]


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (IOError, ValueError):
        return None


def classify_stop(reason):
    """The kind of a supervisor stop reason, see RunGuard and ObjectStampCheck."""
    if reason.startswith("no observation"):
        return "tracking_lost"  # Stale object TF (or joint state): the inferer skips steps.
    if reason.startswith("object out of range"):
        return "object_out_of_range"
    if " TF " in reason:
        return "tf_stamp"
    return "guard"


class Recorder(object):
    def __init__(self, root, supervisor, profile_path, topics, instruction, scene, operator,
                 observation_topic, gap_threshold, max_duration, calibration_camera):
        self._root = root
        self._profile_path = profile_path
        with open(profile_path) as stream:
            self._profile_text = stream.read()
        self._profile = yaml.safe_load(self._profile_text)
        self._topics = topics
        self._instruction = instruction
        self._scene = scene
        self._operator = operator
        self._gap_threshold = gap_threshold
        self._max_duration = max_duration
        self._calibration_camera = calibration_camera
        self._lock = threading.RLock()
        self._enabled = bool(rospy.get_param("~enabled", False))
        self._state = None
        self._current = None  # The episode being recorded: dict with "job", "dir", "meta", ...
        self._finishing = None  # The episode whose bag is being closed.
        self._latest = None   # The latest finished episode: {"dir", "meta", "label"}
        self._message = ""
        self._counts = {}
        self._tags_path = os.path.join(root, "tags.json")
        self._tags = read_json(self._tags_path) or []
        self._status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        rospy.Service("~set_enabled", SetBool, self._set_enabled)
        rospy.Subscriber("~label", String, self._on_label, queue_size=10)
        rospy.Subscriber(supervisor + "/log", String, self._on_log, queue_size=100)
        rospy.Subscriber(observation_topic, Float64MultiArray, self._on_observation, queue_size=10)
        rospy.Subscriber(supervisor + "/state", String, self._on_state, queue_size=10)
        self._recount()
        threading.Thread(target=self._loop, daemon=True).start()

    # --- inputs ---

    def _set_enabled(self, request):
        with self._lock:
            self._enabled = request.data
            self._say("recording {}".format("enabled" if request.data else "disabled"))
        self._publish()
        return SetBoolResponse(True, "recording " + ("on" if request.data else "off"))

    def _on_log(self, msg):
        line = msg.data
        # The first stop line of a run wins: a guard trip or an operator stop comes before the
        # exit line.
        if line.startswith("=== stopping policy: ") or line == "stopping policy" or (
                line.startswith("=== policy exited with code")):
            with self._lock:
                episode = self._current or self._finishing
                if episode is not None and episode["stop_line"] is None:
                    episode["stop_line"] = line

    def _on_observation(self, _):
        now = time.monotonic()
        with self._lock:
            episode = self._current
            if episode is None:
                return
            last = episode["last_observation"]
            if last is None:
                episode["meta"]["first_observation_s"] = round(now - episode["start"], 3)
                # The policy's topics exist from now on.
                threading.Thread(target=self._check_topics, args=(episode,), daemon=True).start()
            elif now - last > self._gap_threshold:
                episode["meta"]["observation_gaps"].append(
                    {"start_s": round(last - episode["start"], 3), "length_s": round(now - last, 3)})
            episode["last_observation"] = now

    def _on_state(self, msg):
        with self._lock:
            previous, self._state = self._state, msg.data
            # Not on the first (latched) state: a run already going is not recorded halfway.
            if msg.data == "policy" and previous not in (None, "policy") and self._enabled:
                self._start()
            elif msg.data != "policy" and self._current is not None:
                episode, self._current = self._current, None
                episode["end"] = time.monotonic()
                self._finishing = episode
                threading.Thread(target=self._finish, args=(episode,), daemon=True).start()
        self._publish()

    # --- recording ---

    def _start(self):
        now = datetime.datetime.now()
        episode_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        session = now.strftime("%Y-%m-%d")
        directory = os.path.join(self._root, session, episode_id)
        os.makedirs(directory)
        model_path = self._profile.get("model_path", "")
        meta = {
            "episode_id": episode_id,
            "session": session,
            "timestamp": time.time(),
            "operator_name": self._operator,
            "language_instruction": self._instruction,
            "scene": dict(self._scene),
            "profile_path": self._profile_path,
            "profile": self._profile_text,
            "model_path": model_path,
            "policy_id": "{}/{}".format(os.path.basename(os.path.dirname(model_path)),
                                        os.path.basename(model_path)),
            "control_freq": float(self._profile.get("policy_rate", 50.0)),
            "topics": self._topics,
            "missing_topics": [],  # Checked at the first observation, see _check_topics.
            "hostname": socket.gethostname(),
            "observation_gaps": [],
            "first_observation_s": None,
        }
        write_json(os.path.join(directory, "meta.json"), meta)
        command = "rosbag record --lz4 -O {} {}".format(
            shlex.quote(os.path.join(directory, "episode.bag")),
            " ".join(shlex.quote(topic) for topic in self._topics))
        job = Job("rosbag", command, self._on_bag_line, lambda job, code: None)
        self._current = {"job": job, "dir": directory, "meta": meta, "start": time.monotonic(),
                         "last_observation": None, "truncated": False, "stop_line": None}
        self._say("recording episode " + episode_id)
        threading.Thread(target=self._save_context, args=(directory, model_path), daemon=True).start()

    def _check_topics(self, episode):
        published = set(topic for topic, _ in rospy.get_published_topics())
        missing = [topic for topic in self._topics if topic not in published]
        with self._lock:
            episode["meta"]["missing_topics"] = missing
        if missing:
            self._say("no publisher for: " + ", ".join(missing), warn=True)

    def _on_bag_line(self, line):
        if "WARN" in line or "ERROR" in line:
            rospy.logwarn("rosbag: %s", line)

    def _save_context(self, directory, model_path):
        """Files the robot and policy profiles for oopsie-data need; none is essential."""
        try:
            # Not rospy.get_param("/"): once the node has logged, rospy's parameter cache
            # answers it with only /rosout_disable_topics_generation.
            params = rosgraph.Master(rospy.get_name()).getParam("/")
            with open(os.path.join(directory, "params.yaml"), "w") as stream:
                yaml.safe_dump(params, stream, default_flow_style=None)
        except (rosgraph.MasterException, IOError, yaml.YAMLError) as error:
            rospy.logwarn("could not save params: %s", error)
        env = os.path.join(os.path.dirname(model_path), "params", "env.yaml")
        if os.path.isfile(env):
            shutil.copy(env, os.path.join(directory, "policy_env.yaml"))
        try:
            data = rospkg.RosPack().get_path("kangaroo_camera_calibration")
            results = sorted(glob.glob(os.path.join(
                data, "data", self._calibration_camera + "_*", "calibration_result.yaml")))
            if results:
                shutil.copy(results[-1], os.path.join(
                    directory, "calibration_{}.yaml".format(self._calibration_camera)))
        except rospkg.ResourceNotFound:
            pass

    def _stop_bag(self, episode):
        if episode["job"].running():
            episode["job"].stop(rospy.logwarn)

    def _finish(self, episode):
        """Stops the bag and completes meta.json; the log lines arrive meanwhile."""
        self._stop_bag(episode)
        if episode["last_observation"] is None:
            self._check_topics(episode)
        with self._lock:
            if self._finishing is episode:
                self._finishing = None
            end, last = episode["end"], episode["last_observation"]
            meta = episode["meta"]
            line = episode["stop_line"] or ""
            if line.startswith("=== stopping policy: "):
                # Without the terminal colour codes of the forwarded inferer lines.
                reason = re.sub(r"\x1b\[[0-9;]*m", "", line[len("=== stopping policy: "):]).rstrip(" =")
                meta["stop_reason_kind"] = classify_stop(reason)
            elif line == "stopping policy":
                reason, meta["stop_reason_kind"] = "stopped by operator", "operator"
            elif line:
                reason, meta["stop_reason_kind"] = line.strip(" ="), "exited"
            else:
                reason, meta["stop_reason_kind"] = "unknown", "unknown"
            meta["stop_reason"] = reason
            meta["duration_s"] = round(end - episode["start"], 3)
            meta["end_timestamp"] = meta["timestamp"] + meta["duration_s"]
            meta["truncated"] = episode["truncated"]
            if last is not None and end - last > self._gap_threshold:
                meta["trailing_gap_s"] = round(end - last, 3)
            # Gaps the policy recovered from; a trailing gap before the stop is separate.
            meta["has_gaps"] = bool(meta["observation_gaps"])
            meta["too_short"] = meta["duration_s"] < 1.0
            write_json(os.path.join(episode["dir"], "meta.json"), meta)
            self._latest = {"dir": episode["dir"], "meta": meta, "label": None}
            self._say("episode {} done ({:.1f} s): {}".format(
                meta["episode_id"], meta["duration_s"], reason))
        self._publish()

    # --- labels ---

    def _on_label(self, msg):
        try:
            data = json.loads(msg.data)
            with self._lock:
                if "add_tag" in data:
                    self._add_tag(data["add_tag"])
                    self._publish()
                    return
                latest = self._latest
                if latest is None or data.get("episode") != latest["meta"]["episode_id"]:
                    raise ValueError("not the latest episode: {}".format(data.get("episode")))
                self._apply_label(latest, data)
        except (ValueError, TypeError, AttributeError, OSError) as error:
            self._say("label failed: {}".format(error), warn=True)
        self._recount()
        self._publish()

    def _add_tag(self, tag):
        tag = " ".join(str(tag).split())
        if not tag or len(tag) > 40:
            raise ValueError("a tag needs 1 to 40 characters")
        if tag not in self._tags:
            self._tags.append(tag)
            write_json(self._tags_path, self._tags)
            self._say("new tag: " + tag)

    def _apply_label(self, latest, data):
        meta = latest["meta"]
        label = {"annotator_name": self._operator, "labelled_at": time.time()}
        if data.get("discard"):
            destination = os.path.join(self._root, "discarded", meta["session"])
            label["discarded"] = True
        elif data.get("exclude"):
            if data["exclude"] not in EXCLUDE_REASONS:
                raise ValueError("unknown exclude reason {}".format(data["exclude"]))
            destination = os.path.join(self._root, "excluded", data["exclude"], meta["session"])
            label["excluded"] = data["exclude"]
        else:
            outcome = data.get("outcome")
            if outcome not in OUTCOMES:
                raise ValueError("unknown outcome {}".format(outcome))
            categories = list(data.get("side_effect_category") or [])
            severity = data.get("severity")
            # Categories describe what went wrong, the severity how bad a failure was.
            if outcome in ("success", "success_suboptimal"):
                categories = []
            if outcome != "failure":
                severity = None
            unknown = set(categories) - set(CATEGORIES)
            if unknown:
                raise ValueError("unknown categories {}".format(", ".join(sorted(unknown))))
            if severity is not None and severity not in SEVERITIES:
                raise ValueError("unknown severity {}".format(severity))
            tags = [] if outcome == "success" else list(data.get("tags") or [])
            unknown = set(tags) - set(self._tags)
            if unknown:
                raise ValueError("unknown tags {}".format(", ".join(sorted(unknown))))
            label.update({
                "outcome": outcome,
                "success": 1.0 if outcome.startswith("success") else 0.0,
                "side_effect_category": categories,
                "severity": severity,
                "tags": tags,
                "taxonomy_schema": TAXONOMY_SCHEMA,
            })
            destination = os.path.join(self._root, meta["session"])
        target = os.path.join(destination, meta["episode_id"])
        if target != latest["dir"]:
            if not os.path.isdir(destination):
                os.makedirs(destination)
            shutil.move(latest["dir"], target)
            latest["dir"] = target
        write_json(os.path.join(target, "label.json"), label)
        latest["label"] = label
        self._say("episode {}: {}".format(meta["episode_id"], label.get("outcome") or (
            "excluded ({})".format(label["excluded"]) if "excluded" in label else "discarded")))

    def _recount(self):
        """Episodes and failures per instruction over all kept (not excluded) episodes."""
        counts = {}
        for meta_path in glob.glob(os.path.join(self._root, "20*", "*", "meta.json")):
            meta = read_json(meta_path)
            if meta is None:
                continue
            label = read_json(os.path.join(os.path.dirname(meta_path), "label.json"))
            count = counts.setdefault(meta.get("language_instruction", ""),
                                      {"episodes": 0, "failures": 0, "unlabelled": 0})
            if label is None:
                count["unlabelled"] += 1
            else:
                count["episodes"] += 1
                count["failures"] += label.get("success") == 0.0
        with self._lock:
            self._counts = counts

    # --- status ---

    def _say(self, message, warn=False):
        (rospy.logwarn if warn else rospy.loginfo)("%s", message)
        with self._lock:
            self._message = time.strftime("%H:%M:%S ") + message

    def _loop(self):
        while not rospy.is_shutdown():
            with self._lock:
                episode = self._current
                if (episode is not None and not episode["truncated"]
                        and time.monotonic() - episode["start"] > self._max_duration):
                    episode["truncated"] = True
                    self._say("episode longer than {:.0f} s: recording stopped".format(
                        self._max_duration), warn=True)
                    threading.Thread(target=self._stop_bag, args=(episode,), daemon=True).start()
            self._publish()
            time.sleep(1.0)

    def _publish(self):
        with self._lock:
            episode = None
            if self._current is not None:
                meta = self._current["meta"]
                episode = {"id": meta["episode_id"], "recording": True,
                           "elapsed_s": round(time.monotonic() - self._current["start"], 1),
                           "gaps": len(meta["observation_gaps"]),
                           "missing_topics": meta["missing_topics"]}
            elif self._latest is not None:
                meta = self._latest["meta"]
                episode = {key: meta.get(key) for key in (
                    "stop_reason", "stop_reason_kind", "duration_s", "has_gaps", "trailing_gap_s",
                    "too_short", "truncated", "missing_topics", "language_instruction", "scene")}
                episode.update({"id": meta["episode_id"], "recording": False,
                                "gaps": len(meta["observation_gaps"]),
                                "label": self._latest["label"]})
            status = {
                "enabled": self._enabled, "recording": self._current is not None,
                "episode": episode, "instruction": self._instruction, "scene": self._scene,
                "counts": self._counts.get(self._instruction,
                                           {"episodes": 0, "failures": 0, "unlabelled": 0}),
                "message": self._message, "options": {
                    "outcomes": OUTCOMES, "categories": CATEGORIES, "severities": SEVERITIES,
                    "exclude": EXCLUDE_REASONS, "tags": self._tags}}
        self._status_pub.publish(String(json.dumps(status)))

    def shutdown(self):
        with self._lock:
            episodes = [episode for episode in (self._current, self._finishing) if episode]
        for episode in episodes:
            self._stop_bag(episode)


def main():
    rospy.init_node("episode_recorder")
    root = os.path.expanduser(rospy.get_param("~record_dir", "/catkin_ws/ias-nas/home/oopsie-data"))
    if not os.path.isdir(root):
        os.makedirs(root)
    profile_path = rospy.get_param("~profile")
    with open(profile_path) as stream:
        profile = yaml.safe_load(stream)
    policy_rate = float(profile.get("policy_rate", 50.0))
    recorder = Recorder(
        root, rospy.get_param("~supervisor", "/pal_policy_supervisor"), profile_path,
        rospy.get_param("~topics", DEFAULT_TOPICS),
        rospy.get_param("~instruction", "Grasp the cube and move it to the goal"),
        rospy.get_param("~scene", {}),
        rospy.get_param("~operator", "operator1"),
        rospy.get_param("~observation_topic", profile.get("observation_topic", "/pal_policy/observation")),
        # A gap is a pause of more than this many policy periods.
        float(rospy.get_param("~gap_periods", 3)) / policy_rate,
        float(rospy.get_param("~max_duration", 600.0)),
        rospy.get_param("~calibration_camera", "torso_front"))
    rospy.on_shutdown(recorder.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
