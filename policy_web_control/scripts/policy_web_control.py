#!/usr/bin/env python
"""Phone-friendly web page for pal_policy_deployer's policy_supervisor.

The page's buttons call the supervisor's ~start, ~stop and ~go_home services; the page shows
its ~state, ~readiness and ~log topics and disables Start while the supervisor reports that
the policy is not ready. This node talks to the policy only through ROS.

With pal_policy_deployer's episode_recorder running, the page also switches recording on and
off and labels the latest episode for oopsie-data.

Usage: rosrun policy_web_control policy_web_control.py [_port:=8000] [_supervisor:=/pal_policy_supervisor]
         [_recorder:=/episode_recorder]
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
from std_srvs.srv import SetBool, Trigger

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
.box{padding:10px 12px;border-radius:8px;margin-bottom:12px;background:#222;font-size:.95em}
.box:empty{display:none}
.row{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.row button,.chip{display:inline-block;width:auto;flex:1 1 auto;padding:12px 10px;margin:0;font-size:1em;
font-weight:500;background:#333;border-radius:8px}
.row button.on{background:#1f5fbf}.row .ok.on{background:#1a7f37}.row .bad.on{background:#c62828}
#recbtn{padding:14px;font-size:1.1em;background:#444}#recbtn.on{background:#c62828}
.warn{color:#f5b041}.good{color:#4cc46a}.muted{color:#aaa;font-size:.9em}.reason{font-weight:700}
.save{background:#1a7f37!important;font-weight:700!important;padding:16px!important;margin:10px 0 6px!important}
.row button small{display:block;font-size:.72em;font-weight:400;color:#ccc;margin-top:2px}
.row button.tag{flex:0 1 auto;padding:10px 12px}
.step{font-weight:600;margin:12px 0 2px}
.add{display:flex;gap:6px;margin:6px 0}
.add input{flex:1;min-width:0;padding:10px;font-size:1em;border-radius:8px;border:1px solid #555;background:#111;color:#eee}
.add button{width:auto;padding:10px 14px;margin:0;font-size:1em;background:#444}
button.link{background:none;padding:8px 0;margin:4px 0;font-size:.9em;font-weight:400;color:#8ab4f8;text-align:left}
</style></head><body>
<h1>Kangaroo policy control</h1>
<div id="state">connecting...</div>
<button id="stop" onclick="send('stop')">STOP</button>
<button id="start" onclick="send('start')">Start policy</button>
<button id="go_home" onclick="send('go_home')">Go home</button>
<!-- Below the buttons: their heights change, which must not move the buttons. -->
<div id="label" class="box"></div>
<div id="rec" class="box"></div>
<div id="ready"></div>
<pre id="log"></pre>
<button id="reregister" class="link" onclick="if(confirm('Re-register tracking?'))send('reregister')">Re-register tracking</button>
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
    const rec=s.recorder;
    if(rec&&rec.recording&&rec.episode)document.getElementById('state').textContent+='   \\u25cf REC '+mmss(rec.episode.elapsed_s);
    renderRec(rec);
  }catch(e){document.getElementById('state').textContent='server unreachable';}
}
// Episode recording and labelling for oopsie-data (episode_recorder.py).
const NAMES={success:'Success',success_suboptimal:'Suboptimal',success_side_effect:'Side-effect',
  failure:'Failure',reaching:'Reaching',grasp:'Grasp',manipulation:'Manipulation',
  sequencing_semantic:'Sequencing',collision:'Collision',hardware:'Hardware',
  not_attempted:'Not attempted',other:'Other',low:'Low',medium:'Medium',catastrophic:'Catastrophic',
  tracking_lost:'Tracking lost',setup_error:'Setup error',discard:'Discard'};
const OUT_HINT={success:'task done cleanly',success_suboptimal:'done, but slow / retries',
  success_side_effect:'done, but something else went wrong',failure:'task not done'};
const CAT_HINT={reaching:'before contact',grasp:'at contact',manipulation:'after contact',
  sequencing_semantic:'wrong order / object',collision:'hit something',hardware:'caused by the policy',
  not_attempted:'did not try',other:'anything else'};
const SEV_HINT={low:'just retry',medium:'needed a reset',catastrophic:'dangerous / damage'};
const EX_HINT={tracking_lost:'FoundationPose, not the policy',hardware:'not caused by the policy',
  setup_error:'our config / scene',other:''};
let recKey=null,labelKey=null,relabel=null,pick=null,paint=()=>{};
function el(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
// The object's distance to the goal (episode_recorder.py), within threshold_m counts as reached.
function goalText(g,live){
  const cm=m=>(100*m).toFixed(1)+' cm';
  if(live)return 'Cube to goal: '+cm(g.final_m)+' (closest '+cm(g.min_m)+')';
  return 'Cube to goal: final '+cm(g.final_m)+', closest '+cm(g.min_m)+', '+g.within_s.toFixed(1)+' s within '+cm(g.threshold_m);
}
function goalClass(g){return g.final_m<g.threshold_m?'reason good':g.min_m<g.threshold_m?'reason warn':'reason';}
function mmss(s){s=Math.floor(s||0);return Math.floor(s/60)+':'+String(s%60).padStart(2,'0');}
async function post(path,body){
  try{const r=await fetch('/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();if(!j.success)alert(j.message);}
  catch(e){alert('request failed: '+e);}
  poll();
}
function button(text,cls,onclick){const b=el('button',text,cls);b.onclick=onclick;return b;}
function renderRec(rec){
  const box=document.getElementById('rec');
  // Rebuild only on change, so that a tap is not lost to a rebuild.
  const key=rec?JSON.stringify([rec.enabled,rec.instruction,rec.scene,rec.counts,rec.message]):'none';
  if(key!==recKey){
    recKey=key;box.innerHTML='';
    if(!rec){box.appendChild(el('span','no episode recorder','muted'));}
    else{
      const b=button('Recording episodes: '+(rec.enabled?'ON':'OFF'),rec.enabled?'on':'',()=>post('record',{enabled:!rec.enabled}));
      b.id='recbtn';box.appendChild(b);
      // Instruction and scene are launch file settings.
      const scene=Object.keys(rec.scene).map(k=>k+' '+rec.scene[k]).join(', ');
      box.appendChild(el('div','Task: '+rec.instruction+(scene?' ('+scene+')':''),'muted'));
      const c=rec.counts;
      box.appendChild(el('div',c.episodes+' labelled ('+c.failures+' failures), '+c.unlabelled+' unlabelled; at most 200 count per task','muted'));
      if(rec.message)box.appendChild(el('div',rec.message,'muted'));
    }
  }
  renderLabel(rec);
}
function renderLabel(rec){
  const box=document.getElementById('label'),ep=rec&&rec.episode;
  if(!ep){labelKey=null;box.innerHTML='';return;}
  if(ep.recording){
    labelKey=null;box.innerHTML='';
    box.appendChild(el('b','\\u25cf Recording '+ep.id+'  '+mmss(ep.elapsed_s),'warn'));
    if(ep.goal_distance)box.appendChild(el('div',goalText(ep.goal_distance,true),goalClass(ep.goal_distance)));
    if(ep.gaps)box.appendChild(el('div',ep.gaps+' observation gap(s) so far','warn'));
    if(ep.missing_topics.length)box.appendChild(el('div','not published: '+ep.missing_topics.join(', '),'warn'));
    return;
  }
  const o=rec.options,open=!ep.label||relabel===ep.id;
  // Keep the operator's picks while the panel is open; rebuild when a new tag arrives.
  const key=JSON.stringify([ep.id,open,ep.label,o.tags]);
  if(key===labelKey)return;
  labelKey=key;box.innerHTML='';
  if(!open){
    box.appendChild(el('span','Last '+ep.id+': '+summary(ep.label)+' ','muted'));
    const b=button('Relabel','',()=>{relabel=ep.id;pick=null;poll();});
    b.style.cssText='display:inline;width:auto;padding:6px 10px;margin:0;font-size:.9em;background:#444';
    box.appendChild(b);return;
  }
  if(!pick||pick.id!==ep.id)pick={id:ep.id,outcome:null,cats:[],tags:[],severity:null,exclude:false};
  const done=body=>{relabel=null;post('label',Object.assign({episode:ep.id},body));};
  box.appendChild(el('b','Label episode '+ep.id+' ('+(ep.duration_s||0).toFixed(1)+' s)'));
  box.appendChild(el('div','Stopped: '+ep.stop_reason,'reason'+(ep.stop_reason_kind==='tracking_lost'?' warn':'')));
  const g=ep.goal_distance;
  if(g){
    box.appendChild(el('div',goalText(g,false),goalClass(g)));
    box.appendChild(el('div',g.final_m<g.threshold_m?'Ended at the goal (training counts this as success).':
      g.min_m<g.threshold_m?'Reached the goal but left it again (dropped?).':'Never within '+(100*g.threshold_m).toFixed(0)+' cm of the goal.','muted'));
  }
  if(ep.stop_reason_kind==='tracking_lost')box.appendChild(el('div','Tracking lost. Caused by the policy (cube knocked away, hidden by the arm)? Then label a failure. Otherwise: Exclude, tracking lost.','muted'));
  if(ep.has_gaps)box.appendChild(el('div',ep.gaps+' observation gap(s) during the run (missing frames)','warn'));
  if(ep.too_short)box.appendChild(el('div','shorter than 1 s: oopsie-data rejects it','warn'));
  if(ep.truncated)box.appendChild(el('div','recording stopped at the max duration','warn'));
  if(ep.missing_topics&&ep.missing_topics.length)box.appendChild(el('div','not recorded: '+ep.missing_topics.join(', '),'warn'));
  const chip=(v,cls,hint,onclick)=>{const b=button(NAMES[v]||v,cls,onclick);if(hint)b.appendChild(el('small',hint));b.dataset.v=v;return b;};
  const toggle=(list,v)=>{const i=list.indexOf(v);if(i<0)list.push(v);else list.splice(i,1);paint();};
  // Step 1: outcome.
  const outs=el('div',null,'row');
  o.outcomes.forEach(v=>outs.appendChild(chip(v,v.startsWith('success')?'ok':'bad',OUT_HINT[v],()=>{pick.outcome=v;paint();})));
  // Step 2: what went wrong (taxonomy categories), and own tags.
  const what=el('div'),whatHead=el('div',null,'step'),cats=el('div',null,'row'),tags=el('div',null,'row');
  o.categories.forEach(v=>cats.appendChild(chip(v,'',CAT_HINT[v],()=>toggle(pick.cats,v))));
  o.tags.forEach(v=>tags.appendChild(chip(v,'tag','',()=>toggle(pick.tags,v))));
  const add=el('div',null,'add'),input=el('input');input.placeholder='own tag, e.g. releases at goal';input.maxLength=40;
  const addTag=()=>{const t=input.value.trim().replace(/ +/g,' ');if(!t)return;
    if(pick.tags.indexOf(t)<0)pick.tags.push(t);post('label',{add_tag:t});};
  input.onkeydown=e=>{if(e.key==='Enter')addTag();};
  add.appendChild(input);add.appendChild(button('+ Add','',addTag));
  [whatHead,cats,el('div','Own tags','muted'),tags,add].forEach(e=>what.appendChild(e));
  // Step 3: severity, for failures only.
  const how=el('div'),sev=el('div',null,'row');
  o.severities.forEach(v=>sev.appendChild(chip(v,'',SEV_HINT[v],()=>{pick.severity=pick.severity===v?null:v;paint();})));
  how.appendChild(el('div','3. How bad?','step'));how.appendChild(sev);
  const body=()=>{const f=pick.outcome==='failure',side=f||pick.outcome==='success_side_effect';
    return {outcome:pick.outcome,side_effect_category:side?pick.cats:[],severity:f?pick.severity:null,
            tags:pick.outcome==='success'?[]:pick.tags.filter(t=>o.tags.indexOf(t)>=0)};};
  const save=button('Save','save',()=>done(body()));
  // Not the policy's fault: collapsed, it is rare.
  const exLink=button("Not the policy's fault? Exclude or discard","link",()=>{pick.exclude=!pick.exclude;paint();});
  const ex=el('div',null,'row');
  o.exclude.forEach(v=>ex.appendChild(chip(v,'',EX_HINT[v],()=>{if(confirm('Exclude '+ep.id+': '+NAMES[v]+'?'))done({exclude:v});})));
  ex.appendChild(chip('discard','','bad recording',()=>{if(confirm('Discard '+ep.id+'?'))done({discard:true});}));
  [el('div','1. How did it end?','step'),outs,what,how,save,exLink,ex].forEach(e=>box.appendChild(e));
  paint=()=>{
    const mark=(row,on)=>row.querySelectorAll('button').forEach(b=>b.classList.toggle('on',on(b.dataset.v)));
    const f=pick.outcome==='failure',side=pick.outcome==='success_side_effect';
    mark(outs,v=>v===pick.outcome);mark(cats,v=>pick.cats.indexOf(v)>=0);
    mark(tags,v=>pick.tags.indexOf(v)>=0);mark(sev,v=>v===pick.severity);
    what.style.display=f||side||pick.outcome==='success_suboptimal'?'':'none';
    cats.style.display=f||side?'':'none';
    whatHead.textContent=f?'2. What went wrong? (any that apply)':side?'2. What was the side-effect?':'2. What was suboptimal? (optional)';
    how.style.display=f?'':'none';
    save.disabled=!pick.outcome;
    save.textContent=pick.outcome?'Save: '+summary(body()):'Save';
    ex.style.display=pick.exclude?'':'none';
  };
  paint();
}
function summary(l){
  if(l.excluded)return 'excluded: '+NAMES[l.excluded];
  if(l.discarded)return 'discarded';
  return [NAMES[l.outcome]].concat((l.side_effect_category||[]).map(c=>NAMES[c]),l.tags||[],
    l.severity?[NAMES[l.severity]]:[]).join(' \\u00b7 ');
}
setInterval(poll,1000);poll();
</script></body></html>
"""


class EmptyRequest(rospy.AnyMsg):
    """Raw request for tracking_ros/Reregister, whose type is not installed here.

    Assumes the request is a single string field; all-default is one empty string (4 zero bytes).
    """
    def __init__(self):
        super(EmptyRequest, self).__init__()
        self._buff = b"\x00\x00\x00\x00"


class AnyService(object):
    """Service type wildcard: empty request, raw reply. Used where the real type is not installed."""
    _type = "*"
    _md5sum = "*"
    _request_class = EmptyRequest
    _response_class = rospy.AnyMsg


class SupervisorClient(object):
    def __init__(self, supervisor, recorder, reregister):
        self.state = "no supervisor"
        self.log = collections.deque(maxlen=300)
        self._readiness = None
        self._readiness_stamp = 0.0
        self._services = {name: rospy.ServiceProxy(supervisor + "/" + name, Trigger)
                          for name in ("start", "stop", "go_home")}
        self._services["reregister"] = rospy.ServiceProxy(reregister, AnyService)
        rospy.Subscriber(supervisor + "/state", String, self._on_state, queue_size=1)
        rospy.Subscriber(supervisor + "/log", String, self._on_log, queue_size=100)
        rospy.Subscriber(supervisor + "/readiness", String, self._on_readiness, queue_size=1)
        self._recorder = None
        self._recorder_stamp = 0.0
        self._set_recording = rospy.ServiceProxy(recorder + "/set_enabled", SetBool)
        self._recorder_pubs = {name: rospy.Publisher(recorder + "/" + name, String, queue_size=10)
                               for name in ("label",)}
        rospy.Subscriber(recorder + "/status", String, self._on_recorder, queue_size=1)

    def _on_recorder(self, msg):
        self._recorder = json.loads(msg.data)
        self._recorder_stamp = time.time()

    def _on_state(self, msg):
        self.state = msg.data

    def _on_readiness(self, msg):
        self._readiness = json.loads(msg.data)
        self._readiness_stamp = time.time()

    def _on_log(self, msg):
        self.log.append(time.strftime("%H:%M:%S ") + msg.data)

    def call(self, name):
        try:
            if name == "reregister":
                self._services[name](EmptyRequest())
                return {"success": True, "message": "reregister called"}
            response = self._services[name]()
        except (rospy.ServiceException, rospy.ROSException) as error:
            return {"success": False, "message": "{} failed: {}".format(name, error)}
        return {"success": response.success, "message": response.message}

    def record(self, body):
        try:
            response = self._set_recording(bool(body.get("enabled")))
        except (rospy.ServiceException, rospy.ROSException) as error:
            return {"success": False, "message": "record failed: {}".format(error)}
        return {"success": response.success, "message": response.message}

    def to_recorder(self, name, body):
        """label is a topic: the recorder's ~status shows the result."""
        if self.recorder_status() is None:
            return {"success": False, "message": "no episode recorder"}
        self._recorder_pubs[name].publish(String(json.dumps(body)))
        return {"success": True, "message": name + " sent"}

    def recorder_status(self):
        # The recorder publishes at 1 Hz.
        if self._recorder is None or time.time() - self._recorder_stamp > 3.0:
            return None
        return self._recorder

    def status(self):
        readiness = self._readiness
        # The supervisor publishes at 2 Hz; an old message means it is gone.
        if readiness is None or time.time() - self._readiness_stamp > 2.0:
            readiness = {"ready": False, "reasons": ["no readiness from the supervisor"], "notes": []}
        return {"state": self.state, "readiness": readiness, "log": list(self.log),
                "recorder": self.recorder_status()}


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
            if name in ("start", "stop", "go_home", "reregister"):
                self._reply(200, json.dumps(client.call(name)))
                return
            if name not in ("record", "label"):
                self._reply(404, json.dumps({"error": "not found"}))
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode() or "{}")
            except ValueError:
                self._reply(400, json.dumps({"error": "bad json"}))
                return
            if name == "record":
                self._reply(200, json.dumps(client.record(body)))
            else:
                self._reply(200, json.dumps(client.to_recorder(name, body)))

        def log_message(self, fmt, *args):
            if not self.path.startswith("/status"):
                rospy.loginfo("%s " + fmt, self.address_string(), *args)

    return Handler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    rospy.init_node("policy_web_control")
    port = rospy.get_param("~port", 8000)
    client = SupervisorClient(rospy.get_param("~supervisor", "/pal_policy_supervisor"),
                              rospy.get_param("~recorder", "/episode_recorder"),
                              rospy.get_param("~reregister_service",
                                              "/tracking_fp_plus_so3_trt_node/trigger_reregister"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(client))
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    rospy.loginfo("serving on http://0.0.0.0:%d", port)
    rospy.spin()
    server.shutdown()


if __name__ == "__main__":
    main()
