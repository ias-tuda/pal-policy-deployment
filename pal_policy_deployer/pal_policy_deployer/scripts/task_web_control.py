#!/usr/bin/env python3
"""Phone page to switch the grasp task (policy_supervisor ~set_task) and to stop the arm.

One button per task of the supervisor's ~tasks, grouped by robot (G2 with the gravity bridge,
G3 without); a tap switches the tracker, the controllers and the supervisor to that task, and
the arm moves to its start pose. STOP calls the supervisor's ~stop: the controller holds the
last target. The page also shows whether the supervisor, the controllers and the tracker run
the current task. It does not check the object positions; policy_web_control does that.
Re-register tracking makes the tracker detect all objects of its task again, from a fresh
segmentation mask, for example when it tracks the wrong object.

Runs in the dev container beside the supervisor; policy_web_control (port 8000) stays the page
to start policies and label episodes.

Usage: rosrun pal_policy_deployer task_web_control.py [_port:=8003]
         [_supervisor:=/pal_policy_supervisor] [_controllers:=/spawn_grasp_controller]
         [_tracker:=/tracking_fp_plus_so3_trt_node] [_tracker_objects_topic:=/tracking/active_objects_cfg]
         [_hidden_tasks:="[a, b]"]
"""
import collections
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import rosgraph
import rospy
import yaml
from controller_manager_msgs.srv import ListControllers
from std_msgs.msg import String
from std_srvs.srv import Trigger
from pal_policy_deployer.srv import SetTask
from tracking_ros.srv import Reregister

from task_config import load_task

# The two controller groups that switch in place; other robots show their own name.
GROUPS = {"left_arm_gravcomp": "G2 · left_arm_gravcomp (gravity bridge)",
          "left_arm_nogravcomp": "G3 · left_arm_nogravcomp (no bridge)"}
# Tasks the page leaves out; set_task still accepts them from the command line.
HIDDEN_TASKS = ["putintobox-s101-wx-a-s1", "pininsert-s12-in-t-s2", "mug-upright"]
BRIDGE = "/gravity_compensation"
SUBSCRIBER_CONTROLLER = "subscriber_controller"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task Switch</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;padding:0 16px 16px;background:#111;color:#eee}
#top{position:sticky;top:0;background:#111;padding:12px 0 8px;z-index:1}
h1{font-size:1.2em;margin:0 0 8px}
#stop{display:block;width:100%;padding:28px;font-size:1.6em;font-weight:700;border:0;border-radius:10px;
color:#fff;background:#c62828}
#task{padding:12px;border-radius:8px;background:#222;margin:12px 0;font-size:1.1em}
.box{padding:10px 12px;border-radius:8px;background:#222;margin-bottom:12px;font-size:.95em}
.check{margin:2px 0}.ok{color:#4cc46a}.bad{color:#f06b6b}.muted{color:#aaa;font-size:.9em}
h2{font-size:1em;margin:16px 0 6px;color:#ccc}
.row{display:flex;flex-wrap:wrap;gap:6px}
.row button{flex:1 1 40%;padding:16px 8px;font-size:1em;border:0;border-radius:8px;color:#fff;
background:#333;word-break:break-all}
.row button.active{background:#1f5fbf;font-weight:700}
#reregister{display:block;width:100%;padding:18px;margin:0 0 12px;font-size:1.2em;font-weight:600;border:0;
border-radius:10px;color:#fff;background:#7b3fb5}
button:disabled{opacity:.4}
#reply{white-space:pre-wrap}#reply.bad{background:#4a1515}#reply:empty{display:none}
pre{background:#000;padding:8px;font-size:11px;max-height:30vh;overflow:auto;white-space:pre-wrap;word-break:break-all}
</style></head><body>
<div id="top"><h1>Kangaroo task switch</h1><button id="stop" onclick="stop()">STOP</button></div>
<div id="task">connecting...</div>
<button id="reregister" onclick="reregister()">Re-register tracking</button>
<div id="checks" class="box"></div>
<div id="reply" class="box"></div>
<div id="tasks"></div>
<pre id="log"></pre>
<script>
let status=null,switching=null,tasksKey=null;
function el(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
function reply(ok,text){const r=document.getElementById('reply');r.className='box'+(ok?'':' bad');r.textContent=text;}
async function post(path,body){
  try{const r=await fetch('/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
    const j=await r.json();reply(j.success,j.message);}
  catch(e){reply(false,'request failed: '+e);}
}
async function stop(){await post('stop');poll();}
async function reregister(){
  // A fresh registration can make the tracked pose jump; ask only while the arm moves.
  if(status&&status.state!=='idle'&&!confirm(status.state+' is running. Re-register tracking?'))return;
  reply(true,'re-registering...');await post('reregister');poll();
}
async function switchTo(name){
  // The switch stops a running policy or home move; ask only then.
  if(status&&status.state!=='idle'&&!confirm(status.state+' is running. Stop it and switch to '+name+'?'))return;
  switching=name;render();reply(true,'switching to '+name+'...');
  await post('set_task',{task:name});
  switching=null;poll();
}
function render(){
  const s=status;if(!s)return;
  const t=s.task;
  document.getElementById('task').textContent=(t?'Task: '+t.task+' ('+(s.groups[t.robot]||t.robot)+')':'Task: unknown')+
    (s.state&&s.state!=='idle'?'   \\u2014 '+s.state.toUpperCase()+' running':'');
  const checks=document.getElementById('checks');checks.innerHTML='';
  s.checks.forEach(c=>{const d=el('div',null,'check');d.appendChild(el('span',(c.ok?'\\u2713 ':'\\u2717 ')+c.name+': ',c.ok?'ok':'bad'));
    d.appendChild(el('span',c.detail));checks.appendChild(d);});
  document.getElementById('reregister').disabled=!s.tracker_running;
  if(!s.can_switch)checks.appendChild(el('div','Switch needs the supervisor and the controllers (kang_grasp_controllers.launch).','muted'));
  // Rebuild the buttons only on change, so that a tap is not lost to a rebuild.
  const key=JSON.stringify([s.tasks,t&&t.task,s.can_switch,switching]);
  if(key!==tasksKey){
    tasksKey=key;const box=document.getElementById('tasks');box.innerHTML='';
    const robots=[];s.tasks.forEach(x=>{if(robots.indexOf(x.robot)<0)robots.push(x.robot);});
    robots.forEach(r=>{
      box.appendChild(el('h2',s.groups[r]||r));
      const row=el('div',null,'row');
      s.tasks.filter(x=>x.robot===r).forEach(x=>{
        const b=el('button',switching===x.task?'switching...':x.task,t&&x.task===t.task?'active':'');
        b.disabled=!s.can_switch||switching!==null;b.onclick=()=>switchTo(x.task);row.appendChild(b);});
      box.appendChild(row);
    });
    if(!s.tasks.length)box.appendChild(el('div','no task list from the supervisor','muted'));
  }
  const log=document.getElementById('log');log.textContent=s.log.join('\\n');log.scrollTop=log.scrollHeight;
}
async function poll(){
  try{status=await (await fetch('/status')).json();render();}
  catch(e){document.getElementById('task').textContent='server unreachable';}
}
setInterval(poll,1000);poll();
</script></body></html>
"""


class TaskClient(object):
    """The supervisor's topics and services, and a 1 Hz check of what runs."""

    def __init__(self, supervisor, controllers, tracker, tracker_objects_topic, hidden_tasks):
        self._supervisor = supervisor
        self._hidden_tasks = set(hidden_tasks)
        self._controllers = controllers
        self._tracker = tracker
        self.state = None
        self.task = None
        self.tasks = []
        self.log = collections.deque(maxlen=20)
        self._readiness_stamp = 0.0
        self._tracker_objects = None
        self._checks = []
        self._can_switch = False
        self._tracker_running = False
        self._task_objects = {}  # task -> tracker_objects, from the task files
        self._lock = threading.Lock()
        self._set_task = rospy.ServiceProxy(supervisor + "/set_task", SetTask)
        self._stop = rospy.ServiceProxy(supervisor + "/stop", Trigger)
        self._reregister = rospy.ServiceProxy(tracker + "/trigger_reregister", Reregister)
        self._list_controllers = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
        self._master = rosgraph.Master(rospy.get_name())
        rospy.Subscriber(supervisor + "/state", String, self._on_state, queue_size=1)
        rospy.Subscriber(supervisor + "/task", String, self._on_task, queue_size=1)
        rospy.Subscriber(supervisor + "/tasks", String, self._on_tasks, queue_size=1)
        rospy.Subscriber(supervisor + "/log", String, self._on_log, queue_size=100)
        rospy.Subscriber(supervisor + "/readiness", String, self._on_readiness, queue_size=1)
        rospy.Subscriber(tracker_objects_topic, String, self._on_tracker_objects, queue_size=1)
        threading.Thread(target=self._loop, daemon=True).start()

    def _on_state(self, msg):
        self.state = msg.data

    def _on_task(self, msg):
        self.task = json.loads(msg.data)

    def _on_tasks(self, msg):
        self.tasks = [entry for entry in json.loads(msg.data) if entry["task"] not in self._hidden_tasks]

    def _on_log(self, msg):
        self.log.append(time.strftime("%H:%M:%S ") + msg.data)

    def _on_readiness(self, _):
        self._readiness_stamp = time.time()

    def _on_tracker_objects(self, msg):
        self._tracker_objects = msg.data

    def _loop(self):
        while not rospy.is_shutdown():
            checks, can_switch = self._check()
            with self._lock:
                self._checks, self._can_switch = checks, can_switch
            time.sleep(1.0)

    def _check(self):
        try:
            publishers, subscribers, services = self._master.getSystemState()
        except Exception as error:  # The master is gone or busy; report it on every line.
            return [{"name": "ROS master", "ok": False, "detail": str(error)}], False
        nodes = set(node for _, names in publishers + subscribers + services for node in names)
        service_names = set(name for name, _ in services)
        task = self.task

        # The supervisor publishes ~readiness at 2 Hz.
        supervisor_ok = (self._supervisor + "/set_task" in service_names
                         and time.time() - self._readiness_stamp < 2.0)
        checks = [{"name": "Supervisor", "ok": supervisor_ok,
                   "detail": ("runs" if supervisor_ok else "not running")
                   + (", " + self.state if supervisor_ok and self.state else "")}]
        controllers_service = self._controllers + "/set_task" in service_names
        self._tracker_running = self._tracker + "/trigger_reregister" in service_names
        checks.append(self._check_controllers(controllers_service, nodes, task))
        checks.append(self._check_tracker(nodes, task))
        return checks, supervisor_ok and controllers_service

    def _check_controllers(self, service, nodes, task):
        check = {"name": "Controllers", "ok": False}
        if not service:
            check["detail"] = "spawn_grasp_controller not running"
            return check
        try:
            controllers = self._list_controllers()
            running = any(c.name == SUBSCRIBER_CONTROLLER and c.state == "running"
                          for c in controllers.controller)
        except (rospy.ServiceException, rospy.ROSException) as error:
            check["detail"] = "controller manager: {}".format(error)
            return check
        if not running:
            check["detail"] = SUBSCRIBER_CONTROLLER + " not running"
            return check
        config_path = rospy.get_param(self._controllers + "/config_path", "")
        if not task or config_path != task["config_path"]:
            check["detail"] = "on another task ({})".format(config_path or "unknown")
            return check
        try:
            with open(config_path) as stream:
                bridge_needed = yaml.safe_load(stream).get("gravity_compensation", True)
        except (OSError, yaml.YAMLError) as error:
            check["detail"] = "config: {}".format(error)
            return check
        bridge = BRIDGE in nodes
        # A G3 home move brings its own bridge of that name.
        if bridge_needed != bridge and not (bridge and self.state == "home"):
            check["detail"] = "gravity bridge {}".format(
                "missing" if bridge_needed else "runs, but the task has none")
            return check
        check["ok"] = True
        check["detail"] = "{}, {}".format(task["task"], "gravity bridge" if bridge_needed else "no bridge")
        return check

    def _check_tracker(self, nodes, task):
        check = {"name": "Tracker", "ok": False}
        if not task:
            check["detail"] = "no task"
            return check
        name = task["task"]
        if name not in self._task_objects:
            try:
                self._task_objects[name] = load_task(name).get("tracker_objects")
            except (OSError, ValueError, yaml.YAMLError):
                self._task_objects[name] = None
        objects = self._task_objects[name]
        if not objects:
            check["ok"] = True
            check["detail"] = "not used by this task"
            return check
        if self._tracker not in nodes:
            check["detail"] = "not running"
            return check
        # The tracker reports the absolute path; the task names it relative to ROOT_DIR.
        active = self._tracker_objects or ""
        if not (active == objects or active.endswith("/" + objects.lstrip("/"))):
            check["detail"] = "runs {}, task needs {}".format(active or "unknown objects", objects)
            return check
        check["ok"] = True
        check["detail"] = "runs " + objects.split("/")[-1]
        return check

    def status(self):
        with self._lock:
            checks, can_switch = list(self._checks), self._can_switch
        groups = {entry["robot"]: GROUPS.get(entry["robot"], entry["robot"]) for entry in self.tasks}
        return {"state": self.state, "task": self.task, "tasks": self.tasks, "groups": groups,
                "checks": checks, "can_switch": can_switch, "tracker_running": self._tracker_running,
                "log": list(self.log)}

    def set_task(self, body):
        task = body.get("task") or ""
        if task not in [entry["task"] for entry in self.tasks]:
            return {"success": False, "message": "unknown task: {}".format(task)}
        try:
            # Blocks until the switch is done: the job stops, the gains ramp (2 s).
            response = self._set_task(task)
        except (rospy.ServiceException, rospy.ROSException) as error:
            return {"success": False, "message": "set_task failed: {}".format(error)}
        return {"success": response.success, "message": response.message}

    def reregister(self):
        try:
            # An empty list re-registers every object of the tracker's current task.
            response = self._reregister([])
        except (rospy.ServiceException, rospy.ROSException) as error:
            return {"success": False, "message": "re-register failed: {}".format(error)}
        return {"success": response.success, "message": "tracker: " + response.message}

    def stop(self):
        try:
            response = self._stop()
        except (rospy.ServiceException, rospy.ROSException) as error:
            return {"success": False, "message": "stop failed: {}".format(error)}
        return {"success": response.success, "message": response.message}


def make_handler(client):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body, content_type="application/json"):
            data = body.encode()
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
                self._reply(200, json.dumps(client.status()))
            else:
                self._reply(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            name = self.path.lstrip("/")
            if name in ("stop", "reregister"):
                self._reply(200, json.dumps(getattr(client, name)()))
                return
            if name != "set_task":
                self._reply(404, json.dumps({"error": "not found"}))
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode() or "{}")
            except ValueError:
                self._reply(400, json.dumps({"error": "bad json"}))
                return
            self._reply(200, json.dumps(client.set_task(body)))

        def log_message(self, fmt, *args):
            if not self.path.startswith("/status"):
                rospy.loginfo("%s " + fmt, self.address_string(), *args)

    return Handler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    rospy.init_node("task_web_control")
    port = rospy.get_param("~port", 8003)
    # Bind to ROS_IP (the robot-network address), not 0.0.0.0: VS Code's dev container port
    # forwarding holds 127.0.0.1:<port> on the host, which blocks a 0.0.0.0 bind of that port.
    host = rospy.get_param("~host", os.environ.get("ROS_IP") or "0.0.0.0")
    client = TaskClient(rospy.get_param("~supervisor", "/pal_policy_supervisor"),
                        rospy.get_param("~controllers", "/spawn_grasp_controller"),
                        rospy.get_param("~tracker", "/tracking_fp_plus_so3_trt_node"),
                        rospy.get_param("~tracker_objects_topic", "/tracking/active_objects_cfg"),
                        rospy.get_param("~hidden_tasks", HIDDEN_TASKS))
    server = ThreadingHTTPServer((host, port), make_handler(client))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rospy.loginfo("serving on http://%s:%d", host, port)
    rospy.spin()
    server.shutdown()


if __name__ == "__main__":
    main()
