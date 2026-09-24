#!/usr/bin/env python3
"""
sim_policy.py  --  Simuliert Policy-Output mit y_f=0 (Netz-Nullausgabe)
========================================================================

Publisht die default_joint_pos aus den ONNX-Metadaten als desired_state.
Das entspricht exakt dem was die Policy sendet wenn y_f=0 (kein Netzwerk-Output).

Verwendung:
  rosrun subscriber_controller sim_policy.py _model:=v61 _task:=loco
"""

import sys
import os
import glob
import rospy
from sensor_msgs.msg import JointState

# ------------------------------------------------------------------ #
# Argumente  (_model:=v61  _task:=loco)
# ------------------------------------------------------------------ #
model_arg = 'v61'
task_arg  = 'loco'
for arg in sys.argv[1:]:
    if arg.startswith('_model:='):
        model_arg = arg.split(':=', 1)[1]
    elif arg.startswith('_task:='):
        task_arg = arg.split(':=', 1)[1]

# ------------------------------------------------------------------ #
# Modellpfad  (gleiche Logik wie kang_mjlab_node.cpp)
# ------------------------------------------------------------------ #
import rospkg
pkg_root   = rospkg.RosPack().get_path('pal_policy_deployer')

config_dir = os.path.join(pkg_root, 'models', task_arg,
                          task_arg + '_' + model_arg + '_mjlab')
matches = sorted(glob.glob(os.path.join(config_dir, '*.onnx')))
if not matches:
    print('[sim_policy] FEHLER: Kein ONNX-Modell gefunden in: ' + config_dir)
    sys.exit(1)
model_path = matches[-1]

# ------------------------------------------------------------------ #
# Protobuf-Parser (kein onnxruntime noetig) -- aus read_onnx_meta.py
# ------------------------------------------------------------------ #
def read_varint(data, pos):
    result, shift = 0, 0
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
            fn, wt = tag >> 3, tag & 0x7
            if wt == 2:
                s, pos = read_length_delimited(data, pos)
                try:    s = s.decode('utf-8')
                except: s = repr(s)
                if fn == 1:   key = s
                elif fn == 2: val = s
            else:
                break
        except:
            break
    return key, val

def parse_onnx_metadata(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    meta, pos = {}, 0
    while pos < len(data):
        try:
            tag, pos = read_varint(data, pos)
            fn, wt = tag >> 3, tag & 0x7
            if wt == 2:
                chunk, pos = read_length_delimited(data, pos)
                if fn == 14:
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
# Metadaten parsen
# ------------------------------------------------------------------ #
meta = parse_onnx_metadata(model_path)
if not meta:
    print('[sim_policy] FEHLER: Keine Metadaten im Modell gefunden')
    sys.exit(1)

if 'joint_names' not in meta or 'default_joint_pos' not in meta:
    print('[sim_policy] FEHLER: joint_names oder default_joint_pos fehlen in Metadaten')
    sys.exit(1)

EXCLUDED = {'leg_left_femur_joint', 'leg_left_knee_joint',
            'leg_right_femur_joint', 'leg_right_knee_joint'}

joint_names     = [j.strip() for j in meta['joint_names'].split(',')]
default_pos_all = [float(v.strip()) for v in meta['default_joint_pos'].split(',')]

action_joint_names = []
action_default_pos = []
for i, name in enumerate(joint_names):
    if name not in EXCLUDED:
        action_joint_names.append(name)
        action_default_pos.append(default_pos_all[i])

# ------------------------------------------------------------------ #
# ROS Node
# ------------------------------------------------------------------ #
rospy.init_node('sim_policy')

pub = rospy.Publisher('/subscriber_controller/desired_state', JointState, queue_size=1)

rospy.loginfo('=' * 70)
rospy.loginfo('[sim_policy] Modell : %s', os.path.basename(model_path))
rospy.loginfo('[sim_policy] Action joints: %d   (excluded: femur/knee)', len(action_joint_names))
rospy.loginfo('[sim_policy] default_joint_pos (= was gesendet wird):')
for i, name in enumerate(action_joint_names):
    rospy.loginfo('  %-35s  default_pos=%+.4f', name, action_default_pos[i])
rospy.loginfo('=' * 70)

rate  = rospy.Rate(50)
count = 0

while not rospy.is_shutdown():
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name     = action_joint_names
    msg.position = action_default_pos
    msg.velocity = [0.0] * len(action_joint_names)
    msg.effort   = [0.0] * len(action_joint_names)

    pub.publish(msg)
    count += 1

    rospy.loginfo_throttle(5.0,
        '[sim_policy] publishing default_joint_pos @ 50Hz | count=%d', count)

    rate.sleep()
