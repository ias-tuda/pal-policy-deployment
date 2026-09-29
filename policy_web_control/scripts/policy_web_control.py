#!/usr/bin/env python
"""Phone-friendly web page for pal_policy_deployer's policy_supervisor.

The page's buttons call the supervisor's ~start, ~stop and ~go_home services; the page shows
its ~state, ~readiness and ~log topics and disables Start while the supervisor reports that
the policy is not ready. This node talks to the policy only through ROS.

Usage: rosrun policy_web_control policy_web_control.py [_port:=8000] [_supervisor:=/pal_policy_supervisor]
"""
import collections
import json
import threading
import time

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
except ImportError:  # Python 2, as on the robot
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Policy Control</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#111;color:#eee}
h1{font-size:1.2em;margin:0 0 12px}
#state{padding:12px;border-radius:8px;background:#222;margin-bottom:12px;font-size:1.1em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#ready{padding:10px 12px;border-radius:8px;margin-bottom:12px;font-size:.95em}
#ready.ok{background:#12351f}#ready.bad{background:#4a1515}
#ready ul{margin:6px 0 0;padding-left:18px}#ready li.note{color:#aaa}
button{display:block;width:100%;padding:22px;margin:0 0 12px;font-size:1.3em;font-weight:600;
border:0;border-radius:10px;color:#fff}
#start{background:#1a7f37}#go_home{background:#1f5fbf}#stop{background:#c62828;padding:34px;font-size:1.6em}
button:disabled{opacity:.4}
pre{background:#000;padding:8px;font-size:11px;height:40vh;overflow:auto;white-space:pre-wrap;word-break:break-all}
</style></head><body>
<h1>Kangaroo policy control</h1>
<div id="state">connecting...</div>
<button id="stop" onclick="send('stop')">STOP</button>
<button id="start" onclick="send('start')">Start policy</button>
<button id="go_home" onclick="send('go_home')">Go home</button>
<!-- Below the buttons: its height changes with the reasons, which must not move them. -->
<div id="ready"></div>
<pre id="log"></pre>
<script>
async function send(cmd){
  try{const r=await fetch('/'+cmd,{method:'POST'});const j=await r.json();if(!j.success)alert(j.message);}
  catch(e){alert('request failed: '+e);}
  poll();
}
async function poll(){
  try{
    const s=await (await fetch('/status')).json();
    document.getElementById('state').textContent=s.state==='idle'?'idle':('RUNNING: '+s.state);
    const r=s.readiness,box=document.getElementById('ready');
    document.getElementById('start').disabled=s.state!=='idle'||!r.ready;
    box.className=r.ready?'ok':'bad';
    box.innerHTML='';
    const head=document.createElement('b');head.textContent=r.ready?'Ready to start':'Not ready';box.appendChild(head);
    const ul=document.createElement('ul');
    r.reasons.forEach(t=>{const li=document.createElement('li');li.textContent=t;ul.appendChild(li);});
    r.notes.forEach(t=>{const li=document.createElement('li');li.className='note';li.textContent='(ok) '+t;ul.appendChild(li);});
    if(ul.children.length)box.appendChild(ul);
    const log=document.getElementById('log'),bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-20;
    log.textContent=s.log.join('\\n');if(bottom)log.scrollTop=log.scrollHeight;
  }catch(e){document.getElementById('state').textContent='server unreachable';}
}
setInterval(poll,1000);poll();
</script></body></html>
"""


class SupervisorClient(object):
    def __init__(self, supervisor):
        self.state = "no supervisor"
        self.log = collections.deque(maxlen=300)
        self._readiness = None
        self._readiness_stamp = 0.0
        self._services = {name: rospy.ServiceProxy(supervisor + "/" + name, Trigger)
                          for name in ("start", "stop", "go_home")}
        rospy.Subscriber(supervisor + "/state", String, self._on_state, queue_size=1)
        rospy.Subscriber(supervisor + "/log", String, self._on_log, queue_size=100)
        rospy.Subscriber(supervisor + "/readiness", String, self._on_readiness, queue_size=1)

    def _on_state(self, msg):
        self.state = msg.data

    def _on_readiness(self, msg):
        self._readiness = json.loads(msg.data)
        self._readiness_stamp = time.time()

    def _on_log(self, msg):
        self.log.append(time.strftime("%H:%M:%S ") + msg.data)

    def call(self, name):
        try:
            response = self._services[name]()
        except (rospy.ServiceException, rospy.ROSException) as error:
            return {"success": False, "message": "{} failed: {}".format(name, error)}
        return {"success": response.success, "message": response.message}

    def status(self):
        readiness = self._readiness
        # The supervisor publishes at 2 Hz; an old message means it is gone.
        if readiness is None or time.time() - self._readiness_stamp > 2.0:
            readiness = {"ready": False, "reasons": ["no readiness from the supervisor"], "notes": []}
        return {"state": self.state, "readiness": readiness, "log": list(self.log)}


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
            if name not in ("start", "stop", "go_home"):
                self._reply(404, json.dumps({"error": "not found"}))
                return
            self._reply(200, json.dumps(client.call(name)))

        def log_message(self, fmt, *args):
            if not self.path.startswith("/status"):
                rospy.loginfo("%s " + fmt, self.address_string(), *args)

    return Handler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    rospy.init_node("policy_web_control")
    port = rospy.get_param("~port", 8000)
    client = SupervisorClient(rospy.get_param("~supervisor", "/pal_policy_supervisor"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(client))
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    rospy.loginfo("serving on http://0.0.0.0:%d", port)
    rospy.spin()
    server.shutdown()


if __name__ == "__main__":
    main()
