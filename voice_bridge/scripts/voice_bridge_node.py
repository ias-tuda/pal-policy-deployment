#!/usr/bin/env python3
"""HTTP/JSON API for the voice agent to start the voice tasks (see the package README).

  roslaunch voice_bridge voice_bridge.launch              next to the supervisor (dev container)
  python3 voice_bridge_node.py --sim [--port 8010]        without ROS, on a simulated robot

With --sim, POST /sim changes the simulation and acts as the phone pages do, for example
{"objects_present": false}, {"move_arm": 0.3}, {"go_home": true}, {"stop": true},
{"set_task": "mug-turn"}, {"start": true}, {"home_fails": true}, {"tracker_running": false}.
"""
import argparse
import os
import sys
import threading

# Without catkin (--sim on a laptop) the package is beside this script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from voice_bridge.bridge import Settings, VoiceBridge  # noqa: E402
from voice_bridge.http_api import make_server  # noqa: E402
from voice_bridge.state import RobotState  # noqa: E402
from voice_bridge.tasks import load_tasks  # noqa: E402

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "voice_tasks.yaml")


def run_ros():
    import rospy
    from voice_bridge.ros_io import RosIO

    rospy.init_node("voice_bridge")
    tasks = load_tasks(rospy.get_param("~tasks_file", TASKS_FILE))
    state = RobotState()
    io = RosIO(state, rospy.get_param("~supervisor", "/pal_policy_supervisor"),
               rospy.get_param("~joint_state_topic", "/subscriber_controller/actual_js_state"),
               rospy.get_param("~active_objects_topic", "/tracking/active_objects_cfg"))
    settings = Settings(**rospy.get_param("~settings", {}))
    bridge = VoiceBridge(state, io, tasks, settings)
    host, port = rospy.get_param("~host", "127.0.0.1"), int(rospy.get_param("~port", 8010))
    server = make_server(bridge, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rospy.on_shutdown(server.shutdown)
    rospy.loginfo("voice bridge on http://%s:%d for %s", host, port, ", ".join(tasks.names()))
    rospy.loginfo("voice bridge settings: %s; a command takes at most about %.0f s", vars(settings), settings.total())
    rospy.spin()


class PrintingState(RobotState):
    """Prints the log lines, as the phone pages would show them."""

    def on_log(self, line):
        print(line, flush=True)
        RobotState.on_log(self, line)


def run_sim(args):
    from voice_bridge.sim import SimIO, SimSupervisor, Timing, sim_tasks

    tasks = load_tasks(args.tasks)
    state = PrintingState()
    sim = SimSupervisor(state, sim_tasks(tasks.names()), tasks.names()[0], Timing(args.scale))
    bridge = VoiceBridge(state, SimIO(sim, state), tasks)

    def control(body):
        answers = {}
        for name in ("objects_present", "tracker_running", "home_fails", "switch_refusal"):
            if name in body:
                setattr(sim, name, body[name])
                answers[name] = body[name]
        if "move_arm" in body:
            sim.move_arm(float(body["move_arm"]))
            answers["move_arm"] = body["move_arm"]
        for name in ("stop", "go_home", "start"):
            if body.get(name):
                answers[name] = getattr(sim, name)()
        if body.get("set_task"):  # As the task page does: it returns before the home move ends.
            threading.Thread(target=sim.set_task, args=(body["set_task"],), daemon=True).start()
            answers["set_task"] = body["set_task"]
        return answers

    server = make_server(bridge, args.host, args.port, {"/sim": control})
    print("simulated robot; voice bridge on http://{}:{}".format(args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="simulated robot, no ROS")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--tasks", default=TASKS_FILE, help="voice task file")
    parser.add_argument("--scale", type=float, default=1.0, help="simulated durations times this")
    # ROS arguments (__name:=, _param:=, remaps) are rospy's.
    args = parser.parse_args([arg for arg in sys.argv[1:] if ":=" not in arg])
    if args.sim:
        run_sim(args)
    else:
        run_ros()


if __name__ == "__main__":
    main()
