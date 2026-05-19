#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
read_onnx_meta.py  --  ONNX-Modell Inspektion
===============================================

Verwendung (gleiche Syntax wie rosrun):
  rosrun pal_policy_deployer read_onnx_meta.py _model:=v61 _task:=loco
  rosrun pal_policy_deployer read_onnx_meta.py /pfad/zum/model.onnx

Findet das gleiche Modell wie:
  rosrun pal_policy_inferer pal_policy_inferer_mjlab_node _model:=v61 _task:=loco

Modellpfad-Logik (identisch zu kang_mjlab_node.cpp):
  <pal_policy_deployer>/models/<task>/<task>_<model>_mjlab/*.onnx
"""
import sys
import os
import glob
import struct

# ------------------------------------------------------------------ #
# Argument-Parsing  (_model:=v61 _task:=loco  oder  /pfad/model.onnx)
# ------------------------------------------------------------------ #
model_arg = 'v61'
task_arg  = 'loco'
explicit_path = None

for arg in sys.argv[1:]:
    if arg.startswith('_model:='):
        model_arg = arg.split(':=', 1)[1]
    elif arg.startswith('_task:='):
        task_arg = arg.split(':=', 1)[1]
    elif arg.endswith('.onnx') or os.path.isfile(arg):
        explicit_path = arg

# ------------------------------------------------------------------ #
# Modellpfad bestimmen  (gleiche Logik wie kang_mjlab_node.cpp)
# ------------------------------------------------------------------ #
script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_root   = os.path.join(script_dir, '..')  # pal_policy_deployer/

if explicit_path:
    model_path = explicit_path
else:
    config_dir = os.path.join(pkg_root, 'models', task_arg,
                              task_arg + '_' + model_arg + '_mjlab')
    matches = sorted(glob.glob(os.path.join(config_dir, '*.onnx')))
    if not matches:
        print("FEHLER: Kein ONNX-Modell gefunden in: " + config_dir)
        sys.exit(1)
    model_path = matches[-1]

print("=" * 70)
print("Modell : " + os.path.abspath(model_path))
print("Task   : " + task_arg + "   Model-Version: " + model_arg)
print("=" * 70)

# ------------------------------------------------------------------ #
# Protobuf-Parser (kein onnx/onnxruntime noetig)
# ------------------------------------------------------------------ #
def read_varint(data, pos):
    result = 0
    shift  = 0
    while True:
        b = data[pos] if isinstance(data[pos], int) else ord(data[pos])
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos

def read_length_delimited(data, pos):
    length, pos = read_varint(data, pos)
    return data[pos:pos + length], pos + length

def parse_string_entry(data):
    pos = 0
    key = val = None
    while pos < len(data):
        try:
            tag, pos = read_varint(data, pos)
            fn = tag >> 3
            wt = tag & 0x7
            if wt == 2:
                s, pos = read_length_delimited(data, pos)
                try:    s = s.decode('utf-8')
                except: s = repr(s)
                if fn == 1: key = s
                elif fn == 2: val = s
            else:
                break
        except:
            break
    return key, val

def parse_onnx_metadata(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    meta = {}
    pos  = 0
    while pos < len(data):
        try:
            tag, pos = read_varint(data, pos)
            fn = tag >> 3
            wt = tag & 0x7
            if wt == 2:
                chunk, pos = read_length_delimited(data, pos)
                if fn == 14:  # metadata_props
                    k, v = parse_string_entry(chunk)
                    if k is not None:
                        meta[k] = v
            elif wt == 0: _, pos = read_varint(data, pos)
            elif wt == 1: pos += 8
            elif wt == 5: pos += 4
            else: break
        except:
            break
    return meta

# ------------------------------------------------------------------ #
# Onnxruntime fuer I/O-Shapes (optional, falls installiert)
# ------------------------------------------------------------------ #
def get_io_shapes(model_path):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path,
               providers=['CPUExecutionProvider'])
        inputs  = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
        outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
        return inputs, outputs
    except ImportError:
        return None, None
    except Exception as e:
        print("  [onnxruntime Fehler: " + str(e) + "]")
        return None, None

# ------------------------------------------------------------------ #
# Dimensionen aus Metadaten berechnen
# ------------------------------------------------------------------ #
EXCLUDED_JOINTS = {
    'leg_left_femur_joint', 'leg_left_knee_joint',
    'leg_right_femur_joint', 'leg_right_knee_joint',
}

def compute_dims(meta):
    joint_names    = meta.get('joint_names', '').split(',') if meta.get('joint_names') else []
    obs_names      = meta.get('observation_names', '').split(',') if meta.get('observation_names') else []
    action_scale   = meta.get('action_scale', '').split(',') if meta.get('action_scale') else []
    default_pos    = meta.get('default_joint_pos', '').split(',') if meta.get('default_joint_pos') else []

    n_joints = len(joint_names)
    action_joints = [j for j in joint_names if j.strip() not in EXCLUDED_JOINTS]
    n_actions = len(action_scale) if action_scale else len(action_joints)

    term_dims = {}
    single_obs_dim = 0
    for term in obs_names:
        t = term.strip()
        if t in ('base_ang_vel', 'base_lin_vel', 'base_lin_acc',
                 'projected_gravity', 'imu_projected_gravity', 'command'):
            d = 3
        elif t in ('joint_pos', 'joint_vel'):
            d = n_joints
        elif t == 'actions':
            d = n_actions
        else:
            d = 3
        term_dims[t] = d
        single_obs_dim += d

    return {
        'n_joints':      n_joints,
        'n_actions':     n_actions,
        'action_joints': action_joints,
        'obs_terms':     obs_names,
        'term_dims':     term_dims,
        'single_obs_dim':single_obs_dim,
    }

# ------------------------------------------------------------------ #
# Ausgabe
# ------------------------------------------------------------------ #
meta = parse_onnx_metadata(model_path)
if not meta:
    print("Keine Metadaten gefunden!")
    sys.exit(1)

dims = compute_dims(meta)

# --- I/O Shapes via onnxruntime ---
io_inputs, io_outputs = get_io_shapes(model_path)

action_scale_vals = meta.get('action_scale', '').split(',')       if meta.get('action_scale')       else []
default_pos_all   = meta.get('default_joint_pos', '').split(',')  if meta.get('default_joint_pos')  else []
joint_names_all   = meta.get('joint_names', '').split(',')        if meta.get('joint_names')        else []
stiffness_list    = meta.get('joint_stiffness', '').split(',')    if meta.get('joint_stiffness')    else []
damping_list      = meta.get('joint_damping', '').split(',')      if meta.get('joint_damping')      else []

# Offset fuer Action-Joints = ihr default_joint_pos
action_offset_vals = []
for name in dims['action_joints']:
    n = name.strip()
    try:
        idx = [j.strip() for j in joint_names_all].index(n)
        action_offset_vals.append(float(default_pos_all[idx]) if idx < len(default_pos_all) else 0.0)
    except (ValueError, IndexError):
        action_offset_vals.append(0.0)

# I/O tensor shapes
obs_dim_total = '?'
act_dim_total = '?'
history_len   = '?'
if io_inputs:
    obs_dim_total = io_inputs[0][1][1]  if len(io_inputs[0][1])  > 1 else io_inputs[0][1][0]
    act_dim_total = io_outputs[0][1][1] if len(io_outputs[0][1]) > 1 else io_outputs[0][1][0]
    if dims['single_obs_dim']:
        history_len = obs_dim_total // dims['single_obs_dim']

def hdr(title):
    print("")
    print("=" * 80)
    print("  " + title)
    print("=" * 80)

def sep():
    print("-" * 80)

sources = {
    'base_ang_vel':          ('/base_imu',          'angular_velocity'),
    'base_lin_acc':          ('/base_imu',          'linear_acceleration'),
    'imu_projected_gravity': ('/base_imu',          'R_x(180) * q_conj * [0,0,-1]'),
    'projected_gravity':     ('/base_imu',          'R_x(180) * q_conj * [0,0,-1]'),
    'command':               ('/cmd_vel',           'linear.x, linear.y, angular.z'),
    'joint_pos':             ('/joint_states',      'position[i] - default_pos[i]'),
    'joint_vel':             ('/joint_states',      'velocity[i]'),
    'actions':               ('intern',             'y_f vom letzten Timestep'),
    'base_lin_vel':          ('/odom',              '!! NICHT IMPLEMENTIERT !!'),
}

# ================================================================== #
hdr("PIPELINE")
print("")
print("  {:<25s}  {:<24s}  {:>3s}  {}".format(
    "Quelle (Topic)", "Observation Term", "Dim", "Berechnung"))
sep()
for term in dims['obs_terms']:
    t         = term.strip()
    d         = dims['term_dims'].get(t, '?')
    topic, calc = sources.get(t, ('?', '?'))
    print("  {:<25s}  {:<24s}  {:>3s}  {}".format(topic, t, str(d), calc))
sep()
print("  {:<25s}  {:<24s}  {:>3d}  pro Timestep".format("", "TOTAL", dims['single_obs_dim']))
print("  {:<25s}  {:<24s}  {:>3s}  history".format("", "x", str(history_len)))
print("  {:<25s}  {:<24s}  {:>3s}  Input-Tensor".format("", "=", str(obs_dim_total)))
print("")
print("                                       |")
print("                                    [NETZ]")
print("                                       |")
print("                                       v   y_f  ({} dim, ~[-1, 1])".format(dims['n_actions']))
print("")
print("  desired_pos[i] = action_scale[i] * y_f[i] + default_joint_pos[i]")
print("  Output-Tensor: {} dim".format(act_dim_total))
print("")
print("                                       |")
print("                                       v")
print("  /subscriber_controller/desired_state")
print("       .position = desired_pos    (s.o.)")
print("       .velocity = 0              (kein velocity feedforward)")
print("       .effort   = 0              (kein torque feedforward)")
print("")
print("                                       |")
print("                                       v")
print("  DirectImpedanceControl  (Hardware, 2000 Hz)")
print("  tau[i] = stiffness[i] * (desired_pos[i] - actual_pos[i])")
print("         + damping[i]   * (0              - actual_vel[i])")
print("         + 0")

# ================================================================== #
hdr("OBSERVATION TERMS  ({} dim/timestep  x  history {}  =  {} dim Input)".format(
    dims['single_obs_dim'], history_len, obs_dim_total))
print("")
print("  {:>2s}  {:<25s}  {:>3s}   {}".format("#", "Term", "Dim", "Quelle / Berechnung"))
sep()
for idx, term in enumerate(dims['obs_terms']):
    t            = term.strip()
    d            = dims['term_dims'].get(t, '?')
    topic, calc  = sources.get(t, ('?', '?'))
    print("  {:>2d}  {:<25s}  {:>3s}   {:<18s}  {}".format(idx + 1, t, str(d), topic, calc))
sep()
print("      {:<25s}  {:>3d}".format("TOTAL", dims['single_obs_dim']))

# ================================================================== #
hdr("ACTION JOINTS  ({} dim)  ->  desired_state.position".format(dims['n_actions']))
print("")
print("  desired_pos[i] = action_scale[i] * y_f[i] + default_joint_pos[i]")
print("  tau[i] = stiffness[i]*(desired_pos[i]-actual_pos[i]) + damping[i]*(-actual_vel[i])")
print("")
print("  {:>2s}  {:<30s}  {:>6s}  {:>8s}  {:>10s}  {:>8s}".format(
    "#", "Joint", "scale", "def_pos", "stiffness", "damping"))
sep()
for i, name in enumerate(dims['action_joints']):
    n        = name.strip()
    scale    = float(action_scale_vals[i].strip()) if i < len(action_scale_vals) else 0.0
    offset_v = action_offset_vals[i]
    try:
        jidx    = [j.strip() for j in joint_names_all].index(n)
        stiff   = float(stiffness_list[jidx].strip()) if jidx < len(stiffness_list) else 0.0
        damp    = float(damping_list[jidx].strip())   if jidx < len(damping_list)   else 0.0
        stiff_s = "{:10.1f}".format(stiff)
        damp_s  = "{:8.1f}".format(damp)
    except (ValueError, IndexError):
        stiff_s = "         -"
        damp_s  = "       -"
    print("  {:>2d}  {:<30s}  {:>6.3f}  {:>8.4f}  {}  {}".format(
        i, n, scale, offset_v, stiff_s, damp_s))

# ================================================================== #
hdr("OBSERVATION JOINTS  ({} dim)  ->  joint_pos / joint_vel".format(dims['n_joints']))
print("")
print("  joint_pos_input[i] = /joint_states.position[i] - default_joint_pos[i]")
print("")
print("  {:>2s}  {:<30s}  {:>8s}  {:>10s}  {:>8s}  {}".format(
    "#", "Joint", "def_pos", "stiffness", "damping", "Action?"))
sep()
for i, name in enumerate(joint_names_all):
    n      = name.strip()
    defpos = float(default_pos_all[i].strip()) if i < len(default_pos_all) else 0.0
    stiff  = stiffness_list[i].strip()         if i < len(stiffness_list)  else '-'
    damp   = damping_list[i].strip()           if i < len(damping_list)    else '-'
    try:    stiff_s = "{:10.1f}".format(float(stiff))
    except: stiff_s = "{:>10s}".format('-')
    try:    damp_s  = "{:8.1f}".format(float(damp))
    except: damp_s  = "{:>8s}".format('-')
    act_s  = "KEIN AKTOR" if n in EXCLUDED_JOINTS else "ja"
    print("  {:>2d}  {:<30s}  {:>8.4f}  {}  {}  {}".format(
        i, n, defpos, stiff_s, damp_s, act_s))

# ================================================================== #
hdr("SONSTIGE METADATEN")
known = {'joint_names', 'default_joint_pos', 'action_scale', 'observation_names',
         'joint_stiffness', 'joint_damping', 'run_path'}
if meta.get('run_path'):
    print("  run_path     : " + meta['run_path'])
for k, v in meta.items():
    if k not in known:
        print("  " + k + " : " + v[:100])
print("")
