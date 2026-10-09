#!/usr/bin/env python
"""Beginner helper for kangaroo-5c: which EtherCAT slave fails, and the recovery ladder.

  web     (default) page on http://<robot>:8001 with a body map of the motors, the diagnosis
          and buttons for the recovery steps
  status  print the slaves by motor name and what to do next
  fix     guided: restart deployer -> e-stop power cycle -> give up
  homing  run full_homing; if REJECTED, restart ros_bringup and retry
  gripper open|close the left gripper (gripper_left.py), and tell whether it really moved

Runs on the robot (deployed with pal_deploy), or from the container over ssh.

Usage: rosrun kangaroo_doctor kangaroo_doctor.py [web|status|fix|homing|gripper open|close] [--port 8001]
"""
from __future__ import print_function

import argparse
import collections
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
except ImportError:  # Python 2, as on the robot
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn

try:
    input = raw_input  # Python 2
except NameError:
    pass

ROBOT = "pal@kangaroo-5c"
ON_ROBOT = socket.gethostname().startswith("kangaroo")

# slave_position from /opt/pal/ferrum/share/pal_deployer_cfg/config/device/motor/*.cpf
SLAVES = {
    0: "coupler",
    1: "leg_right_1", 2: "leg_right_2", 3: "leg_right_3", 4: "leg_right_4",
    5: "leg_right_5", 6: "leg_right_length",
    7: "leg_left_1", 8: "leg_left_3", 9: "leg_left_2", 10: "leg_left_5",
    11: "leg_left_4", 12: "leg_left_length",
    13: "coupler",
    14: "arm_right_1", 15: "arm_right_2", 16: "arm_right_3", 17: "arm_right_4",
    18: "arm_right_5", 19: "arm_right_6", 20: "arm_right_7",
    21: "gripper_right_finger",
    22: "pelvis_1", 23: "pelvis_2",
    24: "arm_left_1", 25: "arm_left_2", 26: "arm_left_3", 27: "arm_left_4",
    28: "arm_left_5", 29: "arm_left_6", 30: "arm_left_7",
    31: "gripper_left_finger",
}
COUPLERS = {0, 13}  # GX-JC06 junctions, they stay PREOP

DEPLOYER_LOG = "~/.ros/log/latest/deployer.pal.log/current"

# Like the pal_restart_deployer alias, but the alias' `pkill -f deployer` also kills the shell
# that runs it (its command line contains "deployer"), so the rest never runs. [d] avoids that.
RESTART_DEPLOYER = ("pal-stop deployer; "
                    "pkill -9 -f '[d]eployer-gnulinux|[p]al_deployer.sh'; "
                    "pal-stop ros_bringup; sleep 5; pal-start ros_bringup; sleep 5; "
                    "pal-start deployer")
RESTART_BRINGUP = "pal-stop ros_bringup; sleep 5; pal-start ros_bringup"
FULL_HOMING = "rosrun play_motion run_motion full_homing"
GRIPPER = "rosrun kangaroo_doctor gripper_left.py %s"
# Exit codes of gripper_left.py: (result shown, advice)
GRIPPER_RESULTS = {
    0: ("done", ""),
    2: ("stopped partway", "The gripper stopped partway: it holds an object, or something blocks it."),
    3: ("DID NOT MOVE", "The left gripper did not move: a fault, not an object. "
                        "Fix: e-stop power cycle (the page restarts the deployer after it)."),
}

# (regex in deployer log / dmesg, what it means, what to do)
KNOWN_ERRORS = [
    (r"Failed to reserve master|Master already in use",
     "an old deployer was holding the EtherCAT bus",
     "Restart deployer (it kills leftover deployers)"),
    (r"is not either a transmission_name or a joint name",
     "the robot was brought up with the wrong robot_type",
     "check /tmp/robot_type.yaml, bring the robot up as full_7dof_gripper"),
    (r"Ankle Transmission.*outside the range",
     "an ankle actuator's absolute encoder is out of range (leg_right_5's encoder is known to jump)",
     "Tap leg_*_4/5 and check 'Actual absolute position' (about -0.05 at home). Move the ankle by "
     "hand or re-zero it in the jig (~/kangaroo_legs_zero_reset.sh), then Restart deployer"),
    (r"older than 0\.5",
     "joint states are stale (joint_state_controller not running)",
     "Restart deployer (pal-start deployer alone does not start it)"),
]


# ---------------------------------------------------------------- running commands

_NOISE = re.compile(r"cannot set terminal process group|no job control|Using packages and overlays")


def _quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def _wrap(cmd):
    # Interactive bash: the ROS env and pal-start/pal-stop are set up in ~/.bashrc.
    if ON_ROBOT:
        return ["bash", "-ic", cmd]
    return ["ssh", "-o", "ConnectTimeout=5", ROBOT, "bash -ic " + _quote(cmd)]


def _decode(data):
    # The couplers' names are Shift-JIS.
    return data.decode("utf-8", "replace")


def _popen(cmd):
    # Own session, no terminal: bash -i would otherwise take over the terminal of the shell
    # that started this script, and that shell then stops this script ("Stopped"). close_fds:
    # a child must not keep the page's port open after the page exits.
    return subprocess.Popen(_wrap(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=open(os.devnull), preexec_fn=os.setsid, close_fds=True)


def run(cmd, timeout=30):
    """Run on the robot, return stdout+stderr without the bash -i noise."""
    p = _popen(cmd)
    timer = threading.Timer(timeout, p.kill)
    timer.start()
    try:
        out = _decode(p.communicate()[0])
    finally:
        timer.cancel()
    return "\n".join(l for l in out.splitlines() if not _NOISE.search(l))


def run_live(cmd, on_line):
    """Run on the robot, call on_line for each output line, return the exit code."""
    p = _popen(cmd)
    for raw in iter(p.stdout.readline, b""):
        line = _decode(raw).rstrip()
        if not _NOISE.search(line):
            on_line(line)
    return p.wait()


# ---------------------------------------------------------------- diagnosis

def read_slaves():
    """[(pos, state, flag)] from `ethercat slaves`."""
    slaves = []
    for line in run("ethercat slaves").splitlines():
        m = re.match(r"\s*\d+\s+\d+:(\d+)\s+(\S+)\s+(\S)\s", line)
        if m:
            slaves.append((int(m.group(1)), m.group(2), m.group(3)))
    return slaves


def deployer_procs():
    """[(pid, ppid)] of running deployer-gnulinux processes."""
    procs = []
    for line in run("ps -eo pid,ppid,cmd | grep [d]eployer-gnulinux").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            procs.append((int(parts[0]), int(parts[1])))
    return procs


def known_log_errors():
    text = run("tail -300 %s 2>/dev/null; dmesg 2>/dev/null | tail -40" % DEPLOYER_LOG)
    return [(why, fix) for pat, why, fix in KNOWN_ERRORS if re.search(pat, text)]


def deployer_crash():
    """The error that put the running deployer into exception state, or None.

    The log keeps earlier starts of this boot; each start prints "mlockall done.".
    """
    text = run("tail -400 %s 2>/dev/null" % DEPLOYER_LOG)
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if "mlockall done." in l]
    lines = lines[starts[-1]:] if starts else lines
    for i, line in enumerate(lines):
        if "switching to exception state" in line:
            reason = lines[i + 1] if i + 1 < len(lines) else line
            return re.sub(r"\x1b\[[0-9;]*m|^[\d.]+ \[ *ERROR *\]\[\w+\]\s*", "", reason).strip()
    return None


def robot_type():
    m = re.search(r"robot_type:\s*(\S+)", run("cat /tmp/robot_type.yaml 2>/dev/null"))
    return m.group(1) if m else None


def diagnose(restarted_recently=False):
    """Check the bus and the deployer. Returns a dict (also what the page shows).

    next: the step to take, "restart" | "estop" | "wait" | "help" | None (healthy)
    """
    slaves = read_slaves()
    procs = deployer_procs()
    problems = []

    def problem(level, text, fix=""):
        problems.append({"level": level, "text": text, "fix": fix})

    seen = {pos: (state, flag) for pos, state, flag in slaves}
    table = []
    for pos in sorted(SLAVES):
        state, flag = seen.get(pos, ("", ""))
        if pos not in seen:
            status = "missing"
        elif flag == "E":
            status = "error"
        elif pos in COUPLERS:
            status = "coupler"
        elif state == "OP":
            status = "ok"
        else:
            status = "idle"
        table.append({"pos": pos, "name": SLAVES[pos], "state": state, "flag": flag,
                      "status": status})

    motors = [s for s in table if s["pos"] not in COUPLERS]
    missing = [s for s in table if s["status"] == "missing"]
    errored = [s for s in table if s["status"] == "error"]
    all_op = all(s["status"] == "ok" for s in motors)
    orphans = [pid for pid, ppid in procs if ppid == 1]
    crash = deployer_crash() if procs else None
    nxt = None

    if not slaves:
        problem("error", "Could not read `ethercat slaves` (robot unreachable, or EtherCAT "
                         "master not loaded).", "Check the robot is on; reboot it if this stays.")
        nxt = "help"
    else:
        if missing:
            problem("error", "Not on the bus: " + ", ".join(
                "%d %s" % (s["pos"], s["name"]) for s in missing) + ".",
                "Everything behind a missing slave is cut off: check the e-stop is released "
                "and armed, then cables/power.")
            nxt = "estop"
        for s in errored:
            problem("error", "Slave %d (%s) is in error (%s E)." % (s["pos"], s["name"], s["state"]),
                    "Usually the drive froze. An e-stop power cycle resets it." if restarted_recently
                    else "Restart deployer first; if it stays, do the e-stop power cycle.")
        if errored and not nxt:
            nxt = "estop" if restarted_recently else "restart"
        if orphans:
            problem("error", "A leftover deployer (pid %s) holds the bus." %
                    ", ".join(map(str, orphans)), "Restart deployer kills it.")
            nxt = nxt or "restart"
        if not procs:
            problem("warn", "The deployer is not running, so the motors stay off (PREOP).",
                    "Restart deployer.")
            nxt = nxt or "restart"
        elif not all_op and not orphans and not errored and not missing:
            if restarted_recently:
                problem("warn", "The deployer is starting, motors not all on yet.", "Wait a bit.")
                nxt = nxt or "wait"
            else:
                problem("error", "The deployer runs but did not switch the motors on.",
                        "Restart deployer; see the log hints below.")
                nxt = nxt or "restart"

    if crash:
        # The motors stay on (OP), but the controller manager does not answer.
        known = [(why, fix) for pat, why, fix in KNOWN_ERRORS if re.search(pat, crash)]
        problem("error", "The deployer stopped with an error while starting: %s" % crash,
                known[0][1] if known else "Restart deployer; if it happens again, read the "
                                          "deployer log.")
        nxt = nxt or ("help" if known and "Ankle" in crash else "restart")

    hints = []
    if problems:
        # Only when something is wrong: the logs keep old errors around.
        hints = [{"text": why, "fix": fix} for why, fix in known_log_errors()
                 if not any(p["fix"] == fix for p in problems)]
    rtype = robot_type()
    if rtype and rtype != "full_7dof_gripper":
        problem("warn", "robot_type is %s (arms/grippers missing from the URDF)." % rtype,
                "Bring the robot up as full_7dof_gripper if you use the arms.")

    healthy = (bool(slaves) and all_op and not errored and not missing and not orphans
               and bool(procs) and not crash)
    return {"time": time.strftime("%H:%M:%S"), "slaves": table, "problems": problems,
            "hints": hints, "healthy": healthy, "next": None if healthy else nxt,
            "deployer_running": bool(procs), "robot_type": rtype}


# ---------------------------------------------------------------- robot diagnostics

# Shown when a motor is tapped / in the CLI; the drive's own view, which the bus state misses.
DRIVE_KEYS = ("Driver state", "Error code", "Board errors", "Motor errors", "Gate driver errors",
              "Emergency stop", "Actual control mode", "Board status", "Motor status",
              "Motor temperature", "Board temperature", "Actual position",
              "Actual absolute position")


def _unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v.strip()


def read_robot_diagnostics():
    """What the web commander shows, from /diagnostics_agg. None if nothing is published.

    `rostopic echo` does not always print valid YAML here (multi-line values), so this reads
    the lines it needs.
    """
    out = run("rostopic echo -n1 /diagnostics_agg", timeout=8)
    statuses, cur, key = [], None, None
    for line in out.splitlines():
        m = re.match(r"^    (level|name|message): ?(.*)$", line)
        if m:
            if m.group(1) == "level":
                level = m.group(2).strip()
                cur = {"level": int(level) if level.isdigit() else 2, "name": "", "message": "",
                       "values": {}}
                statuses.append(cur)
            elif cur is not None:
                value = _unquote(m.group(2))
                if m.group(1) == "message":
                    # Drives pad and join several errors with runs of spaces.
                    value = "; ".join(re.split(r"\s{2,}", value))
                cur[m.group(1)] = value
            continue
        m = re.match(r"^        (key|value): ?(.*)$", line)
        if m and cur is not None:
            if m.group(1) == "key":
                key = _unquote(m.group(2))
            elif key is not None:
                cur["values"][key] = _unquote(m.group(2))
                key = None
    if not statuses:
        return None
    names = [st["name"] for st in statuses]
    motors = {}
    for st in statuses:
        m = re.match(r"^/Hardware/Motor/(\w+)_motor$", st["name"])
        if m:
            motors[m.group(1)] = {"level": st["level"], "message": st["message"],
                                  "values": [[k, st["values"][k]] for k in DRIVE_KEYS
                                             if k in st["values"]]}
    # Non-OK entries without children, like the web commander's red/yellow rows.
    others = [{"level": st["level"], "name": st["name"], "message": st["message"]}
              for st in statuses
              if st["level"] in (1, 2) and not st["name"].startswith("/Hardware/Motor/")
              and not any(n.startswith(st["name"] + "/") for n in names)]
    emergency = {}
    for st in statuses:
        if st["name"] == "/Hardware/Emergency":
            emergency = st["values"]
    # PAL's key really is "Is Emergency Waiting Rearmy".
    rearm = [v for k, v in emergency.items() if "Rearm" in k]
    return {"estop": emergency.get("Emergency Stop On") == "True",
            "rearm": bool(rearm) and rearm[0] == "True",
            "motors": motors, "others": others}


def apply_robot_diagnostics(d, diag):
    """Merge drive errors and the e-stop into a diagnose() result."""
    d = dict(d)
    if not diag:
        d["robot"] = None
        return d
    d["robot"] = {"estop": diag["estop"], "rearm": diag["rearm"], "others": diag["others"]}
    problems = list(d["problems"])
    slaves = []
    for s in d["slaves"]:
        s = dict(s)
        drive = diag["motors"].get(s["name"])
        if drive:
            s["drive"] = drive
            # Level 3 is stale: the deployer is down and the values are old.
            if drive["level"] == 2 and d.get("deployer_running") and s["status"] in ("ok", "idle"):
                s["status"] = "drive_error"
        slaves.append(s)
    d["slaves"] = slaves
    if diag["estop"]:
        problems.insert(0, {"level": "error", "text": "The e-stop is pressed: the motors have no power.",
                            "fix": "Release it (twist until it pops up), then press the blinking red "
                                   "or yellow button to re-arm."})
        d["next"] = "rearm"
    elif diag["rearm"]:
        problems.insert(0, {"level": "error", "text": "The robot waits to be re-armed.",
                            "fix": "Press the blinking red or yellow button on the robot."})
        d["next"] = "rearm"
    else:
        bad_drives = [s for s in slaves if s["status"] == "drive_error"]
        by_message = collections.OrderedDict()
        for s in bad_drives:
            by_message.setdefault(s["drive"]["message"] or "error", []).append(s["name"])
        for message, names in by_message.items():
            who = ", ".join(names) if len(names) <= 4 else "%d motors" % len(names)
            fix = "Restart deployer; if it stays, do the e-stop power cycle."
            if "Emergency stop" in message:
                fix = ("The drives still hold the e-stop although it is released: Restart "
                       "deployer to re-enable them.")
            problems.append({"level": "error", "text": "Drive error on %s: %s" % (who, message),
                             "fix": fix})
        if bad_drives:
            d["next"] = d.get("next") or "restart"
    d["problems"] = problems
    d["healthy"] = d["healthy"] and not diag["estop"] and not diag["rearm"] and \
        not any(s["status"] == "drive_error" for s in slaves)
    if d["healthy"]:
        d["next"] = None
    return d


def diagnose_full(restarted_recently=False):
    return apply_robot_diagnostics(diagnose(restarted_recently), read_robot_diagnostics())


# ---------------------------------------------------------------- CLI

USE_COLOR = sys.stdout.isatty()


def c(text, code):
    return "\033[%sm%s\033[0m" % (code, text) if USE_COLOR else text


def ok(t): return c(t, "32")
def bad(t): return c(t, "31;1")
def warn(t): return c(t, "33")
def bold(t): return c(t, "1")


def ask(question, default=True):
    try:
        a = input("%s %s " % (question, "[Y/n]" if default else "[y/N]")).strip().lower()
    except EOFError:
        return default
    return default if not a else a.startswith("y")


def countdown(text, seconds):
    for s in range(seconds, 0, -1):
        sys.stdout.write("\r  %s: %2d s " % (text, s))
        sys.stdout.flush()
        time.sleep(1)
    print()


def print_report(d, slaves=True):
    if slaves:
        print(bold("EtherCAT slaves"))
        color = {"ok": ok, "coupler": ok, "idle": warn, "error": bad, "missing": bad,
                 "drive_error": bad}
        for s in d["slaves"]:
            name = s["name"] + (" (stays PREOP)" if s["status"] == "coupler" else "")
            tag = "%-5s %s" % (s["state"], s["flag"]) if s["state"] else "missing"
            if s["status"] in ("error", "missing"):
                tag += "   <-- " + s["status"].upper()
            if s["status"] == "drive_error":
                tag += "   <-- DRIVE: " + s["drive"]["message"]
            print("  %2d  %-26s %s" % (s["pos"], name, color[s["status"]](tag)))
        print()
    print(bold("Diagnosis"))
    for p in d["problems"]:
        print("  " + (bad if p["level"] == "error" else warn)(p["text"]))
        if p["fix"]:
            print("  -> " + p["fix"])
    for h in d["hints"]:
        print(warn("  Log says (may be from an earlier run): " + h["text"]))
        print("  -> " + h["fix"])
    robot = d.get("robot")
    if robot and robot["others"]:
        print(bold("  Other robot diagnostics (as in the web commander):"))
        for o in robot["others"]:
            print("    " + (bad if o["level"] == 2 else warn)("%s: %s" % (o["name"], o["message"])))
    if d["healthy"]:
        print(ok("  All motors on (OP), no errors, deployer running. Good to go."))
    print()


def cli_restart():
    print(bold(">>> Restart deployer") + " (stop, kill leftovers, restart ros_bringup + deployer, ~30 s)")
    run_live(RESTART_DEPLOYER, lambda l: print("    " + l))


def wait_until_healthy(timeout=60):
    sys.stdout.write("Waiting for all motors to switch on ")
    end = time.time() + timeout
    while time.time() < end:
        if diagnose_full(restarted_recently=True)["healthy"]:
            print(ok(" ok"))
            return True
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(3)
    print(bad(" not reached"))
    return False


def cmd_status(args):
    d = diagnose_full()
    print_report(d)
    return 0 if d["healthy"] else 1


def cmd_fix(args):
    d = diagnose_full()
    print_report(d)
    if d["healthy"]:
        return 0
    print(bold("Step 1/2: restart the deployer"))
    if not ask("Restart the deployer now?"):
        return 1
    cli_restart()
    if wait_until_healthy():
        return 0
    print_report(diagnose_full(restarted_recently=True))

    print(bold("Step 2/2: e-stop power cycle") + " (resets a frozen drive)")
    for step in ESTOP_STEPS:
        print("  " + step)
    input("  Press Enter once the button has stopped blinking... ")
    countdown("Letting drives boot", 20)
    cli_restart()
    if wait_until_healthy():
        return 0
    print_report(diagnose_full(restarted_recently=True))
    print(bad("Still failing after a deployer restart and an e-stop power cycle."))
    print("Note which slave(s) fail above, then reboot the robot or ask PAL support.")
    return 1


def homing_result(output):
    """True/False, or None when the output says neither."""
    if re.search(r"REJECTED|ABORTED|PREEMPTED|LOST|[Ff]ailed", output):
        return False
    if "SUCCEEDED" in output:
        return True
    return None


def cmd_homing(args):
    d = diagnose_full()
    if not d["healthy"]:
        print_report(d, slaves=False)
        print(bad("Fix the problems above first (run `fix`)."))
        return 1
    for attempt in (1, 2):
        print(warn("The robot will MOVE to the home pose."))
        if not ask("Run full_homing?", default=False):
            return 1
        lines = []
        run_live(FULL_HOMING, lambda l: (lines.append(l), print("    " + l)))
        result = homing_result("\n".join(lines))
        if result:
            print(ok("full_homing done."))
            return 0
        if result is None:
            print(warn("Could not tell from the output whether homing worked; check the robot."))
            return 1
        print(bad("full_homing was rejected or failed."))
        if attempt == 2:
            break
        print("Usual fix: restart ros_bringup (play_motion lives there).")
        if not ask("Restart ros_bringup now?"):
            return 1
        run_live(RESTART_BRINGUP, lambda l: print("    " + l))
        countdown("Waiting for bringup", 15)
    print(bad("Still rejected. Run `fix` (restart deployer), then homing again."))
    return 1


def cmd_gripper(args):
    if args.arg not in ("open", "close"):
        print(bad("usage: kangaroo_doctor.py gripper open|close"))
        return 1
    code = run_live(GRIPPER % args.arg, lambda l: print("    " + l))
    result, advice = GRIPPER_RESULTS.get(code, ("FAILED", ""))
    print((ok if code == 0 else warn if code == 2 else bad)("gripper %s: %s" % (args.arg, result)))
    if advice:
        print(advice)
    return code


ESTOP_STEPS = [
    "1. Press the e-stop and hold it 2-5 s.",
    "2. Release the e-stop (twist it until it pops back up).",
    "3. Re-arm: press the blinking red or yellow button on the robot.",
]


# ---------------------------------------------------------------- web

PAGE = u"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kangaroo Doctor</title>
<style>
:root{--bg:#111;--card:#1c1c1c;--fg:#eee;--muted:#999;--ok:#1a7f37;--idle:#8a6d00;--err:#c62828;
--coupler:#2d3b2f;--btn:#333;--blue:#1f5fbf}
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--fg);max-width:900px;margin:auto}
h1{font-size:1.2em;margin:0 0 12px;display:flex;justify-content:space-between;align-items:baseline}
h1 small{font-size:.65em;color:var(--muted);font-weight:400}
#banner{padding:14px;border-radius:10px;margin-bottom:12px;font-size:1.1em;font-weight:600}
#banner.ok{background:var(--ok)}#banner.bad{background:var(--err)}#banner.warn{background:var(--idle)}#banner.off{background:#333}
.card{background:var(--card);border-radius:10px;padding:12px;margin-bottom:12px}
.card h2{font-size:.95em;margin:0 0 8px;color:var(--muted);font-weight:600}
.problem{margin:0 0 10px}.problem b{display:block}.problem span{color:#ccc}
.problem.error b{color:#ff8a80}.problem.warn b{color:#f5b041}
.hint{color:#f5b041;font-size:.9em;margin-top:6px}
button{border:0;border-radius:10px;color:#fff;font-size:1.05em;font-weight:600;padding:16px;cursor:pointer;background:var(--btn)}
button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--blue)}button.danger{background:var(--err)}
#actions{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}
#actions button small{display:block;font-weight:400;font-size:.75em;color:#ddd;margin-top:3px}
#next{width:100%;margin-bottom:12px;padding:20px;font-size:1.2em}
#body{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.part h3{font-size:.85em;margin:0 0 6px;color:var(--muted);font-weight:600}
.tiles{display:flex;flex-wrap:wrap;gap:4px}
.tile{flex:1 0 46px;min-width:46px;padding:6px 4px;border-radius:6px;text-align:center;font-size:.8em;line-height:1.2}
.tile b{display:block;font-size:1.1em}
.tile.ok{background:var(--ok)}.tile.idle{background:var(--idle)}.tile.coupler{background:var(--coupler);color:#aaa}
.tile{cursor:pointer}.tile.sel{outline:2px solid #fff}
.tile.drive_error{background:#7a1f1f}
.tile.error,.tile.missing{background:var(--err);animation:pulse 1s infinite alternate}
@keyframes pulse{to{filter:brightness(1.4)}}
.full{grid-column:1 / -1}
.legend{font-size:.8em;color:var(--muted);margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}
pre{background:#000;padding:8px;font-size:11px;max-height:40vh;overflow:auto;white-space:pre-wrap;word-break:break-all;margin:0}
#estop ol{padding-left:20px;margin:0 0 10px}#estop li{margin-bottom:8px;font-size:1.05em}
#estop .row{display:flex;gap:8px;flex-wrap:wrap}
.hidden{display:none}
#motor table{border-collapse:collapse;width:100%;font-size:.85em}
#motor td{padding:3px 6px;border-top:1px solid #2a2a2a;vertical-align:top}
#motor td:first-child{color:var(--muted);white-space:nowrap}
.diag{padding:6px 0;border-top:1px solid #2a2a2a;font-size:.9em}
.diag b{margin-right:6px}.diag.l2 b{color:#ff8a80}.diag.l1 b{color:#f5b041}
.grp{font-size:.75em;color:var(--muted);margin:12px 0 2px;letter-spacing:.05em}
.ctl{display:flex;align-items:center;gap:10px;padding:8px 0;border-top:1px solid #2a2a2a;flex-wrap:wrap}
.ctl .dot{width:11px;height:11px;border-radius:50%;flex:none}
.dot.running{background:#4cc46a}.dot.stopped,.dot.initialized{background:#888}
.dot.configured{border:2px solid #666}
.ctl .info{flex:1 1 220px;min-width:0}.ctl .info b{word-break:break-all}
.ctl .info div{font-size:.78em;color:var(--muted);word-break:break-word}
.ctl .btns{display:flex;gap:6px}.ctl button{padding:9px 14px;font-size:.9em}
button.go{background:var(--ok)}
details summary{cursor:pointer;color:var(--muted);font-size:.85em;margin-top:10px}
.muted{color:var(--muted);font-size:.85em}
@media (max-width:520px){#body{grid-template-columns:1fr}}
</style></head><body>
<h1>Kangaroo Doctor <small id="updated">connecting...</small></h1>
<div id="banner" class="off">checking...</div>
<button id="next" class="primary hidden"></button>

<div id="estop" class="card hidden">
  <h2>E-STOP POWER CYCLE (resets a frozen drive)</h2>
  <ol id="estop-steps"></ol>
  <div class="row">
    <button id="estop-done" class="primary">I re-armed it</button>
    <button onclick="showEstop(false)">Cancel</button>
  </div>
  <div id="estop-wait" class="muted"></div>
</div>

<div id="problems" class="card hidden"><h2>WHAT IS WRONG</h2><div id="problem-list"></div></div>

<div class="card"><h2>MOTORS (robot's right on the left, like facing the robot)</h2>
  <div id="body"></div>
  <div id="motor" class="hidden"></div>
  <div class="muted" style="margin-top:6px">Tap a motor for its drive details.</div>
  <div class="legend"><i style="background:var(--ok)"></i>on (OP)<i style="background:var(--idle)"></i>off (PREOP)
  <i style="background:var(--err)"></i>bus error / missing<i style="background:#7a1f1f"></i>drive error<i style="background:var(--coupler)"></i>junction (always PREOP)</div>
</div>

<div class="card"><h2>ACTIONS</h2>
  <div id="actions">
    <button data-job="restart_deployer">Restart deployer<small>fixes most problems, ~30 s</small></button>
    <button data-estop>E-stop power cycle<small>if a motor stays in error</small></button>
    <button data-job="full_homing" class="danger">Full homing<small>robot MOVES to home pose</small></button>
    <button data-job="restart_bringup">Restart ros_bringup<small>if full homing is rejected</small></button>
    <button data-job="gripper_open">Open left gripper<small>checks that it really moves</small></button>
    <button data-job="gripper_close">Close left gripper<small>empty: must close fully</small></button>
  </div>
</div>

<div id="others" class="card hidden"><h2>OTHER ROBOT DIAGNOSTICS (as in the web commander)</h2><div id="other-list"></div></div>

<div class="card"><h2>CONTROLLERS <span id="ctl-note" class="muted"></span></h2>
  <div id="ctl"></div>
</div>

<div id="jobcard" class="card hidden"><h2 id="jobtitle">OUTPUT</h2><pre id="joblog"></pre></div>

<script>
const PARTS=[["Right arm",[14,15,16,17,18,19,20,21]],["Left arm",[24,25,26,27,28,29,30,31]],
  ["Pelvis",[22,23],1],["Right leg",[1,2,3,4,5,6]],["Left leg",[7,9,8,11,10,12]],["Junctions",[0,13],1]];
const NEXT={restart:["Restart deployer","restart_deployer"],estop:["Start e-stop power cycle","estop"],rearm:null,
  wait:null,help:null};
const CONFIRM={full_homing:"The robot will MOVE to the home pose. Is the area clear?",
  restart_deployer:"Restart the deployer? Motors switch off for ~30 s.",
  restart_bringup:"Restart ros_bringup?"};
let lastJob=null,estopTimer=null;
const $=id=>document.getElementById(id);
function el(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
function short(n){return n.replace(/^(arm|leg)_(left|right)_/,'').replace(/^gripper_(left|right)_finger$/,'grip')
  .replace(/^pelvis_/,'').replace('length','len').replace('coupler','junc');}
async function startJob(job){
  if(CONFIRM[job]&&!confirm(CONFIRM[job]))return;
  try{const r=await (await fetch('/job/'+job,{method:'POST'})).json();if(!r.success)alert(r.message);}
  catch(e){alert('request failed: '+e);}
  poll();
}
document.querySelectorAll('[data-job]').forEach(b=>b.onclick=()=>startJob(b.dataset.job));
document.querySelector('[data-estop]').onclick=()=>showEstop(true);
function showEstop(on){$('estop').classList.toggle('hidden',!on);if(on)$('estop').scrollIntoView({behavior:'smooth'});
  clearInterval(estopTimer);$('estop-wait').textContent='';$('estop-done').disabled=false;}
$('estop-done').onclick=()=>{
  let s=20;$('estop-done').disabled=true;
  const tick=()=>{$('estop-wait').textContent='Letting the drives boot: '+s+' s, then the deployer restarts.';
    if(s--<=0){clearInterval(estopTimer);showEstop(false);fetch('/job/restart_deployer?after_estop=1',{method:'POST'}).then(poll);}};
  tick();estopTimer=setInterval(tick,1000);
};
let selected=null,lastD=null,lastJ=null;
const LEVEL=['OK','WARN','ERROR','STALE'];
function renderMotor(d){
  const box=$('motor'),s=d.slaves.find(x=>x.pos===selected);
  box.classList.toggle('hidden',!s);if(!s)return;box.innerHTML='';
  const t=el('table'),row=(k,v)=>{const r=el('tr');r.appendChild(el('td',k));r.appendChild(el('td',v));t.appendChild(r);};
  row('Motor','#'+s.pos+' '+s.name);
  row('EtherCAT',s.state?s.state+' '+(s.flag==='E'?'ERROR flag':'ok'):'missing from the bus');
  if(s.drive){row('Drive',LEVEL[s.drive.level]+(s.drive.message?': '+s.drive.message:''));
    s.drive.values.forEach(([k,v])=>row(k,v));}
  else row('Drive','no diagnostics for this slave');
  box.appendChild(t);box.style.marginTop='10px';
}
function renderOthers(d){
  const list=$('other-list'),o=(d.robot&&d.robot.others)||[];
  $('others').classList.toggle('hidden',!o.length);list.innerHTML='';
  o.forEach(x=>{const e=el('div',null,'diag l'+x.level);e.appendChild(el('b',LEVEL[x.level]));
    e.appendChild(document.createTextNode(x.name.replace(/^\//,'')+(x.message?': '+x.message:'')));list.appendChild(e);});
}
function render(d,job){
  lastD=d;lastJ=job;
  $('updated').textContent='updated '+d.time;
  const b=$('banner');
  const errs=d.slaves.filter(s=>s.status==='error'||s.status==='missing'||s.status==='drive_error');
  const rb=d.robot,estop=rb&&(rb.estop||rb.rearm);
  if(job&&job.running){b.className='warn';b.textContent='Running: '+job.title+' ...';}
  else if(d.healthy){b.className='ok';b.textContent='All motors on, no errors. Good to go.';}
  else if(estop){b.className='bad';b.textContent=d.problems[0].text+' '+d.problems[0].fix;}
  else if(errs.length){b.className='bad';b.textContent=errs.map(s=>s.name+' '+({missing:'missing',drive_error:'drive error'}[s.status]||'error')).join(', ');}
  else{b.className=d.problems.some(p=>p.level==='error')?'bad':'warn';b.textContent=d.problems.length?d.problems[0].text:'Not ready';}
  // The suggested next step as one big button.
  const n=$('next'),busy=job&&job.running,nx=!busy&&NEXT[d.next];
  n.classList.toggle('hidden',!nx);
  if(nx){n.textContent='Next step: '+nx[0];n.onclick=()=>nx[1]==='estop'?showEstop(true):startJob(nx[1]);}
  document.querySelectorAll('#actions button').forEach(x=>x.disabled=!!busy);
  const pl=$('problem-list');pl.innerHTML='';
  d.problems.forEach(p=>{const e=el('div',null,'problem '+p.level);e.appendChild(el('b',p.text));
    if(p.fix)e.appendChild(el('span','\\u2192 '+p.fix));pl.appendChild(e);});
  d.hints.forEach(h=>pl.appendChild(el('div','Log (may be from an earlier run): '+h.text+' \\u2192 '+h.fix,'hint')));
  $('problems').classList.toggle('hidden',!pl.children.length);
  const bySlave={};d.slaves.forEach(s=>bySlave[s.pos]=s);
  const body=$('body');body.innerHTML='';
  PARTS.forEach(([title,ids,full])=>{
    const part=el('div',null,'part'+(full?' full':''));part.appendChild(el('h3',title));
    const tiles=el('div',null,'tiles');
    ids.forEach(i=>{const s=bySlave[i];if(!s)return;const t=el('div',null,'tile '+s.status);
      t.title=i+' '+s.name+': '+(s.state?s.state+' '+s.flag:'missing')+(s.drive?' \u2014 '+(s.drive.message||'ok'):'');
      if(i===selected)t.classList.add('sel');
      t.onclick=()=>{selected=selected===i?null:i;render(lastD,lastJ);};
      t.appendChild(el('b',short(s.name)));t.appendChild(el('span','#'+i));tiles.appendChild(t);});
    part.appendChild(tiles);body.appendChild(part);});
  renderMotor(d);renderOthers(d);
}
function renderJob(job){
  if(!job)return;
  $('jobcard').classList.remove('hidden');
  $('jobtitle').textContent=(job.title+(job.running?' (running)':' \\u2014 '+job.result)).toUpperCase();
  const log=$('joblog'),bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-20;
  log.textContent=job.lines.join('\\n');if(bottom)log.scrollTop=log.scrollHeight;
  if(lastJob&&lastJob.running&&!job.running&&job.advice){alert(job.advice);if(job.estop)showEstop(true);}
  lastJob=job;
}
// Controllers (controller manager): start/stop/load/unload.
let ctlKey=null,ctlData=null,sensorsOpen=false;
const CTL_LABEL={start:'Start',stop:'Stop',load:'Load',load_start:'Load + start',unload:'Unload'};
function joints(c){return c.resources.map(r=>r.replace(/_joint$/,'')).join(', ');}
function conflictsOf(name){
  const me=ctlData.loaded.find(c=>c.name===name);if(!me||!me.resources.length)return[];
  return ctlData.loaded.filter(c=>c.name!==name&&c.state==='running'&&c.resources.some(r=>me.resources.indexOf(r)>=0)).map(c=>c.name);
}
async function ctlAction(action,name){
  const me=ctlData.loaded.find(c=>c.name===name);
  let q=null;
  if(action==='stop')q='Stop '+name+'?'+(me&&me.resources.length?'\\n\\nIts joints are no longer controlled: they may go limp or drop ('+joints(me)+').':'');
  if(action==='start'){const k=conflictsOf(name);q='Start '+name+'?'+(k.length?'\\n\\nThis also stops '+k.join(', ')+' (same joints).':'');}
  if(action==='load_start')q='Load and start '+name+'?\\n\\nRunning controllers on the same joints are stopped.';
  if(q&&!confirm(q))return;
  try{const r=await (await fetch('/controller/'+action+'/'+name,{method:'POST'})).json();if(!r.success)alert(r.message);}
  catch(e){alert('request failed: '+e);}
  poll();
}
function ctlRow(c,state,actions,busy){
  const row=el('div',null,'ctl');row.appendChild(el('span',null,'dot '+state));
  const info=el('div',null,'info');info.appendChild(el('b',c.name));
  info.appendChild(el('div',(c.type?c.type.replace(/^.*[\/:]/,'')+' \u00b7 ':'')+state));
  if(c.resources&&c.resources.length)info.appendChild(el('div',c.resources.length+' joints: '+joints(c)));
  row.appendChild(info);
  const btns=el('div',null,'btns');
  actions.forEach(a=>{const b=el('button',CTL_LABEL[a],a==='start'||a==='load_start'?'go':a==='stop'?'danger':'');
    b.disabled=busy;b.onclick=()=>ctlAction(a,c.name);btns.appendChild(b);});
  row.appendChild(btns);return row;
}
function renderControllers(d,busy){
  const box=$('ctl');
  if(!d){box.textContent='loading...';return;}
  const key=JSON.stringify([d,busy]);if(key===ctlKey)return;ctlKey=key;ctlData=d;
  box.innerHTML='';
  if(!d.ok){box.appendChild(el('div','Controller manager not reachable (deployer down?): '+d.error,'muted'));return;}
  const order={running:0,stopped:1,initialized:1};
  const main=d.loaded.filter(c=>c.resources.length).sort((a,b)=>order[a.state]-order[b.state]||a.name.localeCompare(b.name));
  const sensors=d.loaded.filter(c=>!c.resources.length);
  const add=(title,rows)=>{if(!rows.length)return;box.appendChild(el('div',title,'grp'));rows.forEach(r=>box.appendChild(r));};
  add('MOTION CONTROLLERS (control joints)',main.map(c=>ctlRow(c,c.state,c.state==='running'?['stop']:['start','unload'],busy)));
  add('CONFIGURED, NOT LOADED',d.configured.map(n=>ctlRow({name:n,resources:[]},'configured',['load_start','load'],busy)));
  if(sensors.length){
    const det=el('details');det.open=sensorsOpen;det.ontoggle=()=>{sensorsOpen=det.open;};
    const sr=sensors.filter(c=>c.state==='running').length;
    det.appendChild(el('summary','State/sensor controllers ('+sr+'/'+sensors.length+' running)'));
    sensors.forEach(c=>det.appendChild(ctlRow(c,c.state,c.state==='running'?['stop']:['start','unload'],busy)));
    box.appendChild(det);
  }
  const run=d.loaded.filter(c=>c.state==='running').length;
  $('ctl-note').textContent=run+' running, '+(d.loaded.length-run)+' stopped, '+d.configured.length+' not loaded';
}
async function poll(){
  try{const s=await (await fetch('/status')).json();render(s.diagnosis,s.job);renderJob(s.job);
    renderControllers(s.controllers,!!(s.job&&s.job.running));}
  catch(e){$('banner').className='off';$('banner').textContent='page server unreachable';}
}
setInterval(poll,2000);poll();
</script></body></html>
"""

ESTOP_HTML = "".join("<li>%s</li>" % s.split(". ", 1)[1] for s in ESTOP_STEPS)
PAGE = PAGE.replace('<ol id="estop-steps"></ol>', '<ol id="estop-steps">%s</ol>' % ESTOP_HTML)


# ---------------------------------------------------------------- controllers

CONTROLLER_MANAGER = "/controller_manager"
NOT_CONTROLLERS = {"pal_robot_info"}  # has a ~type param, but is no controller
_CONTROLLER_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def read_controllers():
    """Loaded controllers (from the controller manager) and configured ones not loaded yet."""
    import yaml  # only needed here; on the robot it comes with ROS
    out = run("rosservice call %s/list_controllers '{}'" % CONTROLLER_MANAGER, timeout=10)
    try:
        listed = yaml.safe_load(out)["controller"] or []
    except Exception:
        lines = out.strip().splitlines()
        return {"ok": False, "error": lines[-1] if lines else "no answer", "loaded": [],
                "configured": []}
    loaded = []
    for ctl in listed:
        resources = sorted({r for claim in (ctl.get("claimed_resources") or [])
                            for r in (claim.get("resources") or [])})
        loaded.append({"name": ctl["name"], "state": ctl["state"], "type": ctl["type"],
                       "resources": resources})
    names = {ctl["name"] for ctl in loaded}
    configured = []
    for line in run("rosparam list", timeout=10).splitlines():
        m = re.match(r"^/([^/]+)/type$", line.strip())
        if m and m.group(1) not in names and m.group(1) not in NOT_CONTROLLERS:
            configured.append(m.group(1))
    return {"ok": True, "error": "", "loaded": loaded, "configured": sorted(configured)}


def conflicts(controllers, name):
    """Running controllers that claim a joint of `name`: starting it must stop them."""
    mine = [c for c in controllers["loaded"] if c["name"] == name]
    if not mine or not mine[0]["resources"]:
        return []
    joints = set(mine[0]["resources"])
    return [c["name"] for c in controllers["loaded"]
            if c["name"] != name and c["state"] == "running" and joints & set(c["resources"])]


def _names(names):
    return "[" + ", ".join("'%s'" % n for n in names) + "]"


def cm_load(name):
    return "rosservice call %s/load_controller \"name: '%s'\"" % (CONTROLLER_MANAGER, name)


def cm_unload(name):
    return "rosservice call %s/unload_controller \"name: '%s'\"" % (CONTROLLER_MANAGER, name)


def cm_switch(start, stop):
    return ("rosservice call %s/switch_controller \"{start_controllers: %s, stop_controllers: %s, "
            "strictness: 2}\"" % (CONTROLLER_MANAGER, _names(start), _names(stop)))


# ---------------------------------------------------------------- web backend

class Doctor(object):
    """Diagnoses in the background, so many open pages don't each run `ethercat`.

    Runs one action (job) at a time; its output is shown on the page.
    """
    JOBS = {
        "restart_deployer": ("Restart deployer", RESTART_DEPLOYER),
        "restart_bringup": ("Restart ros_bringup", RESTART_BRINGUP),
        "full_homing": ("Full homing", FULL_HOMING),
        "gripper_open": ("Open left gripper", GRIPPER % "open"),
        "gripper_close": ("Close left gripper", GRIPPER % "close"),
    }
    CONTROLLER_ACTIONS = ("start", "stop", "load", "load_start", "unload")

    def __init__(self):
        self.diagnosis = None
        self.controllers = None
        self.robot_diag = None
        self.job = None
        self.last_restart = 0.0
        self.lock = threading.Lock()
        for loop in (self._diagnose_loop, self._controllers_loop, self._robot_diag_loop):
            t = threading.Thread(target=loop)
            t.daemon = True
            t.start()

    def _diagnose_loop(self):
        while True:
            try:
                self.diagnosis = diagnose(restarted_recently=time.time() - self.last_restart < 300)
            except Exception as error:  # keep the page alive
                self.diagnosis = {"time": time.strftime("%H:%M:%S"), "slaves": [], "hints": [],
                                  "problems": [{"level": "error", "text": "diagnosis failed: %s" % error,
                                                "fix": ""}],
                                  "healthy": False, "next": None}
            time.sleep(2)

    def _robot_diag_loop(self):
        while True:
            try:
                self.robot_diag = read_robot_diagnostics()
            except Exception:
                self.robot_diag = None
            time.sleep(2)

    def _controllers_loop(self):
        while True:
            self.refresh_controllers()
            time.sleep(3)

    def refresh_controllers(self):
        try:
            self.controllers = read_controllers()
        except Exception as error:
            self.controllers = {"ok": False, "error": str(error), "loaded": [], "configured": []}
        return self.controllers

    # -- jobs

    def _start_job(self, name, title, steps):
        """steps(job) runs the commands (via self._cmd) and returns True/False/None."""
        with self.lock:
            if self.job and self.job["running"]:
                return {"success": False, "message": self.job["title"] + " is still running"}
            self.job = {"name": name, "title": title, "lines": [], "running": True,
                        "result": "", "advice": ""}
            if name == "restart_deployer":
                self.last_restart = time.time()
        t = threading.Thread(target=self._run, args=(self.job, steps))
        t.daemon = True
        t.start()
        return {"success": True, "message": title + " started"}

    def _run(self, job, steps):
        try:
            result = steps(job)
        except Exception as error:
            job["lines"].append("failed: %s" % error)
            result = False
        if job["name"] == "full_homing":
            job["result"] = {True: "done", False: "REJECTED / failed", None: "unclear"}[result]
            if result is False:
                job["advice"] = ("Full homing was rejected. Usual fix: Restart ros_bringup, wait "
                                 "~15 s, then try Full homing again. If it is rejected again: "
                                 "Restart deployer.")
            elif result is None:
                job["advice"] = "Could not tell from the output whether homing worked; check the robot."
        elif job["name"].startswith("gripper"):
            job["result"], job["advice"] = GRIPPER_RESULTS.get(job.get("code"), ("FAILED", ""))
            job["estop"] = job.get("code") == 3  # the page opens the e-stop steps
        else:
            job["result"] = "done" if result else "FAILED"
        if job["name"] == "restart_deployer":
            self.last_restart = time.time()
        if job["name"].startswith("controller"):
            self.refresh_controllers()
        job["running"] = False

    @staticmethod
    def _cmd(job, cmd):
        """Run one command for a job, return its output."""
        job["lines"].append("$ " + cmd)
        lines = []
        code = run_live(cmd, lambda l: (lines.append(l), job["lines"].append(l)))
        job["code"] = code
        return "\n".join(lines)

    def start(self, name):
        if name not in self.JOBS:
            return {"success": False, "message": "unknown job " + name}
        if name == "full_homing" and not (self.diagnosis or {}).get("healthy"):
            return {"success": False, "message": "Fix the problems first: full homing needs "
                                                 "all motors on."}
        title, cmd = self.JOBS[name]
        if name == "full_homing":
            steps = lambda job: homing_result(self._cmd(job, cmd))
        else:
            steps = lambda job: (self._cmd(job, cmd), job["code"] == 0)[1]
        return self._start_job(name, title, steps)

    def controller(self, action, name):
        if action not in self.CONTROLLER_ACTIONS or not _CONTROLLER_NAME.match(name or ""):
            return {"success": False, "message": "bad request"}

        def service_ok(output):
            return bool(re.search(r"\bok: True", output))

        def explain_failure(job):
            # The controller manager only answers ok: False; the reason is in the deployer log.
            out = run("tail -200 %s 2>/dev/null | grep -iE 'error|warn' | grep -v 'Unknown packet' "
                      "| tail -8" % DEPLOYER_LOG)
            if out.strip():
                job["lines"].append("--- deployer log (last errors/warnings) ---")
                job["lines"].extend(out.splitlines())

        def steps(job):
            ok = True
            if action in ("load", "load_start"):
                ok = service_ok(self._cmd(job, cm_load(name)))
            if ok and action in ("start", "load_start"):
                stop = conflicts(self.refresh_controllers(), name)
                if stop:
                    job["lines"].append("(stopping %s: they use the same joints)" % ", ".join(stop))
                ok = service_ok(self._cmd(job, cm_switch([name], stop)))
            if action == "stop":
                ok = service_ok(self._cmd(job, cm_switch([], [name])))
            if action == "unload":
                ok = service_ok(self._cmd(job, cm_unload(name)))
            if not ok:
                explain_failure(job)
            return ok

        titles = {"start": "Start", "stop": "Stop", "load": "Load", "load_start": "Load + start",
                  "unload": "Unload"}
        return self._start_job("controller_" + action, "%s %s" % (titles[action], name), steps)

    def status(self):
        diagnosis = self.diagnosis or {"time": "-", "slaves": [], "problems": [], "hints": [],
                                       "healthy": False, "next": None}
        if self.diagnosis:
            diagnosis = apply_robot_diagnostics(diagnosis, self.robot_diag)
        ctrls = self.controllers
        if ctrls and ctrls["ok"]:
            jsc = [c for c in ctrls["loaded"] if c["name"] == "joint_state_controller"]
            if not jsc or jsc[0]["state"] != "running":
                diagnosis = dict(diagnosis, problems=diagnosis["problems"] + [{
                    "level": "warn", "text": "joint_state_controller is not running: no joint "
                                             "states, full homing will fail.",
                    "fix": "Restart deployer (pal-start deployer alone does not start it)."}])
        return {"diagnosis": diagnosis, "controllers": ctrls, "job": self.job}


def make_handler(doctor):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body, content_type="application/json"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._reply(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/status":
                self._reply(200, json.dumps(doctor.status()))
            else:
                self._reply(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            m = re.match(r"/job/(\w+)", self.path)
            c = re.match(r"/controller/(\w+)/(\w+)$", self.path)
            if m:
                self._reply(200, json.dumps(doctor.start(m.group(1))))
            elif c:
                self._reply(200, json.dumps(doctor.controller(c.group(1), c.group(2))))
            else:
                self._reply(404, json.dumps({"error": "not found"}))

        def log_message(self, fmt, *args):
            if self.command == "POST":
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return Handler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def cmd_web(args):
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(Doctor()))
    print("serving on http://%s:%d" % (socket.gethostname(), args.port))
    server.serve_forever()


def main():
    global ROBOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="web", choices=["web", "status", "fix", "homing", "gripper"])
    ap.add_argument("arg", nargs="?", help="gripper: open or close")
    ap.add_argument("--port", type=int, default=8001, help="web page port (default %(default)s)")
    ap.add_argument("--robot", default=ROBOT, help="ssh target when not on the robot (default %(default)s)")
    # rosrun/roslaunch may append __name:= and similar.
    args, _ = ap.parse_known_args()
    ROBOT = args.robot
    return {"web": cmd_web, "status": cmd_status, "fix": cmd_fix, "homing": cmd_homing,
            "gripper": cmd_gripper}[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\naborted")
        sys.exit(130)
