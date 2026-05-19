#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_obs.py - Policy Pipeline Visualisierung
=============================================

Pipeline-Uebersicht:

  SENSOREN (raw)                POLICY INPUT              POLICY OUTPUT
  -----------------             ----------------------    ------------------------------
  /joint_states.position  --->  joint_pos_relative        actions (raw network output)
                                = pos - default_pos
  /joint_states.velocity  --->  joint_vel                 actions_scaled
                                                          = scale * raw + default_pos
  /base_imu.angular_vel   --->  base_ang_vel               = desired_state.position
  /base_imu.linear_acc    --->  base_lin_acc
  /base_imu.orientation   --->  projected_gravity_raw     CONTROLLER INPUT
                                  = q_conj * [0,0,-1]     ------------------------------
                          --->  projected_gravity_corrected /subscriber_controller/
                                  = R_x(180) * raw               desired_state.position
  /cmd_vel                --->  command [vx, vy, wz]             = actions_scaled


Ordnerstruktur der Ausgabe:
  <bag>/
    1_sensor_raw/               Rohe Sensordaten wie sie ankommen
    2_policy_input/             Was die Policy als Eingabe sieht
    3_policy_output/            Was die Policy ausgibt
    4_controller_input/         Was an den subscriber_controller geht
    5_comparison/               Vergleiche (z.B. raw vs corrected gravity)
"""

import sys
import os
import glob
import rosbag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.4,
})

# ------------------------------------------------------------------ #
# Argumente
#   python plot_obs.py [bag.bag] [_group:=leg_left]
#   _group:=leg_left        -> PDF nur mit leg_left + gruppenunabhaengigen Plots
#   _group:=leg_left,arm_left -> mehrere Gruppen
#   _group:=all             -> alle Gruppen (= komplettes PDF)
#   kein _group             -> PDF mit allen Plots
# ------------------------------------------------------------------ #
script_dir = os.path.dirname(os.path.abspath(__file__))

bag_path   = None
pdf_filter = None   # None = alle Plots, sonst set von Gruppen

ALL_GROUPS = {'pelvis', 'arm_left', 'arm_right', 'leg_left', 'leg_right'}

for arg in sys.argv[1:]:
    if arg.startswith('_group:='):
        val = arg.split(':=', 1)[1].strip()
        if val == 'all':
            pdf_filter = ALL_GROUPS
        else:
            pdf_filter = set(g.strip() for g in val.split(','))
    elif arg.endswith('.bag') or os.path.isfile(arg):
        bag_path = arg

if bag_path is None:
    bags = sorted(glob.glob(os.path.join(script_dir, '*.bag')), key=os.path.getmtime)
    if not bags:
        print("Kein .bag File gefunden in: " + script_dir)
        sys.exit(1)
    bag_path = bags[-1]

print("Loading bag: " + bag_path)
if pdf_filter is not None:
    print("PDF Gruppen-Filter: " + str(sorted(pdf_filter)))
else:
    print("PDF: alle Plots")

# ------------------------------------------------------------------ #
# Ausgabe-Ordner
# ------------------------------------------------------------------ #
bag_name = os.path.splitext(os.path.basename(bag_path))[0]
out_base  = os.path.join(os.path.dirname(os.path.abspath(bag_path)), bag_name)

dirs = {
    'sensor':     os.path.join(out_base, '1_sensor_raw'),
    'pol_in':     os.path.join(out_base, '2_policy_input'),
    'pol_out':    os.path.join(out_base, '3_policy_output'),
    'ctrl_in':    os.path.join(out_base, '4_torques'),
    'sub_ctrl':   os.path.join(out_base, '5_subscriber_controller'),
    'ft':         os.path.join(out_base, '6_hardware'),
    'compare':    os.path.join(out_base, '7_comparison'),
}
for d in dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

# PDF oeffnen
_pdf_suffix = '_' + '_'.join(sorted(pdf_filter)) if pdf_filter is not None else '_all'
pdf_path = os.path.join(out_base, 'report' + _pdf_suffix + '.pdf')
_pdf = PdfPages(pdf_path)
print("PDF wird gespeichert nach: " + pdf_path)

# Titelseite
_fig_title = plt.figure(figsize=(22, 10))
_fig_title.patch.set_facecolor('#1a252f')
_fig_title.text(0.5, 0.65, os.path.basename(bag_path),
                ha='center', va='center', fontsize=28, fontweight='bold',
                color='white', transform=_fig_title.transFigure,
                wrap=True)
_fig_title.text(0.5, 0.48, os.path.dirname(os.path.abspath(bag_path)),
                ha='center', va='center', fontsize=13,
                color='#95a5a6', transform=_fig_title.transFigure)
_groups_str = 'Gruppen: ' + ', '.join(sorted(pdf_filter)) if pdf_filter is not None else 'Alle Gruppen'
_fig_title.text(0.5, 0.35, _groups_str,
                ha='center', va='center', fontsize=16,
                color='#3498db', transform=_fig_title.transFigure)
_pdf.savefig(_fig_title, bbox_inches='tight', facecolor=_fig_title.get_facecolor())
plt.close(_fig_title)

# ------------------------------------------------------------------ #
# Joint Namen
# ------------------------------------------------------------------ #
JOINT_NAMES = [
    'pelvis_1_joint', 'pelvis_2_joint',
    'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
    'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint',
    'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
    'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint',
    'leg_left_1_joint', 'leg_left_2_joint', 'leg_left_3_joint', 'leg_left_length_joint',
    'leg_left_4_joint', 'leg_left_5_joint', 'leg_left_femur_joint', 'leg_left_knee_joint',
    'leg_right_1_joint', 'leg_right_2_joint', 'leg_right_3_joint', 'leg_right_length_joint',
    'leg_right_4_joint', 'leg_right_5_joint', 'leg_right_femur_joint', 'leg_right_knee_joint',
]
EXCLUDED = {'leg_left_femur_joint', 'leg_left_knee_joint',
            'leg_right_femur_joint', 'leg_right_knee_joint'}
ACTION_NAMES = [n for n in JOINT_NAMES if n not in EXCLUDED]

JOINT_GROUPS = [
    ('pelvis',    [n for n in JOINT_NAMES if 'pelvis'    in n]),
    ('arm_left',  [n for n in JOINT_NAMES if 'arm_left'  in n]),
    ('arm_right', [n for n in JOINT_NAMES if 'arm_right' in n]),
    ('leg_left',  [n for n in JOINT_NAMES if 'leg_left'  in n]),   # inkl. femur/knee (Observation!)
    ('leg_right', [n for n in JOINT_NAMES if 'leg_right' in n]),   # inkl. femur/knee (Observation!)
]
ACTION_GROUPS = [
    ('pelvis',    [n for n in ACTION_NAMES if 'pelvis'    in n]),
    ('arm_left',  [n for n in ACTION_NAMES if 'arm_left'  in n]),
    ('arm_right', [n for n in ACTION_NAMES if 'arm_right' in n]),
    ('leg_left',  [n for n in ACTION_NAMES if 'leg_left'  in n]),
    ('leg_right', [n for n in ACTION_NAMES if 'leg_right' in n]),
]

# ------------------------------------------------------------------ #
# Daten einlesen
# ------------------------------------------------------------------ #
def empty_ts():
    return {'t': [], 'v': [], 'names': None}

def has_data(d, min_cols=1):
    v = d['v']
    return len(d['t']) > 0 and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= min_cols

data = {
    # raw sensor
    'imu':                          empty_ts(),
    'joint_pos_raw':                empty_ts(),
    'joint_vel_raw':                empty_ts(),
    'cmd_vel':                      empty_ts(),
    # policy input (processed)
    'pol_joint_pos_rel':            empty_ts(),  # relativ zu default_pos
    'pol_joint_pos_abs':            empty_ts(),  # absolut
    'pol_joint_vel':                empty_ts(),
    'pol_base_ang_vel':             empty_ts(),
    'pol_base_lin_acc':             empty_ts(),
    'pol_proj_grav':                empty_ts(),  # was policy bekommt (mit Korrektur)
    'pol_proj_grav_raw':            empty_ts(),  # ohne Korrektur
    'pol_proj_grav_corrected':      empty_ts(),  # mit Korrektur (explizit)
    'pol_command':                  empty_ts(),
    # policy output
    'pol_actions_raw':              empty_ts(),  # raw network output y_f
    'pol_actions_scaled':           empty_ts(),  # scale*y_f + default_pos
    # controller input / output
    'desired_state':                empty_ts(),
    'joint_effort_raw':             empty_ts(),  # /joint_states.effort = tau gemessen
    'joint_torque_states':          empty_ts(),  # /joint_torque_states = Hardware-Torques
    'tau_computed':                 empty_ts(),  # berechnet: ks*(des-act) + kd*(-vel)
    # FT sensors
    'ft_left':                      empty_ts(),  # left_ankle_ft  [fx,fy,fz, tx,ty,tz]
    'ft_right':                     empty_ts(),  # right_ankle_ft [fx,fy,fz, tx,ty,tz]
    # subscriber_controller actual state
    'actual_js_pos':                empty_ts(),  # actual_js_state.position
    'actual_js_vel':                empty_ts(),  # actual_js_state.velocity
    'actual_js_effort':             empty_ts(),  # actual_js_state.effort = desired_torque_
    'actual_base_pos':              empty_ts(),  # actual_base_state.pose.position
    'actual_base_orient':           empty_ts(),  # actual_base_state.pose.orientation (rpy)
    'actual_base_vel_lin':          empty_ts(),  # actual_base_state.twist.linear
    'actual_base_vel_ang':          empty_ts(),  # actual_base_state.twist.angular
}

# Float64MultiArray topics (msg.data) -- Fallback fuer alte Bags
TOPIC_MAP = {}

# Vector3Stamped topics (msg.vector.x/y/z)
TOPIC_MAP_V3 = {
    '/policy_debug/base_ang_vel':               'pol_base_ang_vel',
    '/policy_debug/base_lin_acc':               'pol_base_lin_acc',
    '/policy_debug/projected_gravity':          'pol_proj_grav',
    '/policy_debug/projected_gravity_raw':      'pol_proj_grav_raw',
    '/policy_debug/projected_gravity_corrected':'pol_proj_grav_corrected',
    '/policy_debug/command':                    'pol_command',
}

# JointState topics (msg.position + msg.name)
TOPIC_MAP_JS = {
    '/policy_debug/joint_pos':                  'pol_joint_pos_rel',
    '/policy_debug/joint_pos_absolute':         'pol_joint_pos_abs',
    '/policy_debug/actions':                    'pol_actions_raw',
    '/policy_debug/actions_scaled':             'pol_actions_scaled',
    '/policy_input/joint_pos_rel':              'pol_joint_pos_rel',
    '/policy_input/joint_pos_abs':              'pol_joint_pos_abs',
    '/policy_output/actions_raw':               'pol_actions_raw',
    '/policy_output/actions_scaled':            'pol_actions_scaled',
}

# JointState topics where data is in msg.velocity field
TOPIC_MAP_JS_VEL = {
    '/policy_debug/joint_vel':                  'pol_joint_vel',
    '/policy_input/joint_vel':                  'pol_joint_vel',
}

all_topics = (['/joint_states', '/joint_torque_states',
               '/base_imu', '/base_imu_fake', '/cmd_vel',
               '/subscriber_controller/desired_state',
               '/subscriber_controller/actual_js_state',
               '/subscriber_controller/actual_base_state',
               '/ft_sensor_estimator/left_ankle_ft',
               '/ft_sensor_estimator/right_ankle_ft']
              + list(TOPIC_MAP.keys()) + list(TOPIC_MAP_JS.keys()) + list(TOPIC_MAP_V3.keys()))

bag = rosbag.Bag(bag_path)
def get_ts(msg, t):
    """Header-Timestamp wenn vorhanden, sonst Bag-Empfangszeit."""
    try:
        s = msg.header.stamp.to_sec()
        if s > 0:
            return s
    except AttributeError:
        pass
    return t.to_sec()

for topic, msg, t in bag.read_messages(topics=all_topics):
    ts = get_ts(msg, t)
    if topic in ('/base_imu', '/base_imu_fake'):
        data['imu']['t'].append(ts)
        data['imu']['v'].append([
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
        ])
    elif topic == '/joint_states':
        if data['joint_pos_raw']['names'] is None:
            data['joint_pos_raw']['names']   = list(msg.name)
            data['joint_vel_raw']['names']   = list(msg.name)
            data['joint_effort_raw']['names'] = list(msg.name)
        data['joint_pos_raw']['t'].append(ts)
        data['joint_pos_raw']['v'].append(list(msg.position))
        data['joint_vel_raw']['t'].append(ts)
        data['joint_vel_raw']['v'].append(list(msg.velocity))
        if msg.effort:
            data['joint_effort_raw']['t'].append(ts)
            data['joint_effort_raw']['v'].append(list(msg.effort))
    elif topic == '/joint_torque_states':
        if data['joint_torque_states']['names'] is None:
            data['joint_torque_states']['names'] = list(msg.name)
        data['joint_torque_states']['t'].append(ts)
        data['joint_torque_states']['v'].append(list(msg.effort))
    elif topic == '/ft_sensor_estimator/left_ankle_ft':
        data['ft_left']['t'].append(ts)
        data['ft_left']['v'].append([
            msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ])
    elif topic == '/ft_sensor_estimator/right_ankle_ft':
        data['ft_right']['t'].append(ts)
        data['ft_right']['v'].append([
            msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ])
    elif topic == '/cmd_vel':
        data['cmd_vel']['t'].append(ts)
        data['cmd_vel']['v'].append([msg.linear.x, msg.linear.y, msg.angular.z])
    elif topic == '/subscriber_controller/desired_state':
        if data['desired_state']['names'] is None:
            data['desired_state']['names'] = list(msg.name)
        data['desired_state']['t'].append(ts)
        data['desired_state']['v'].append(list(msg.position))
    elif topic == '/subscriber_controller/actual_js_state':
        if data['actual_js_pos']['names'] is None:
            data['actual_js_pos']['names']    = list(msg.name)
            data['actual_js_vel']['names']    = list(msg.name)
            data['actual_js_effort']['names'] = list(msg.name)
        data['actual_js_pos']['t'].append(ts)
        data['actual_js_pos']['v'].append(list(msg.position))
        data['actual_js_vel']['t'].append(ts)
        data['actual_js_vel']['v'].append(list(msg.velocity))
        if msg.effort:
            data['actual_js_effort']['t'].append(ts)
            data['actual_js_effort']['v'].append(list(msg.effort))
    elif topic == '/subscriber_controller/actual_base_state':
        import tf.transformations as tft
        data['actual_base_pos']['t'].append(ts)
        data['actual_base_pos']['v'].append([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        q = msg.pose.pose.orientation
        rpy = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        data['actual_base_orient']['t'].append(ts)
        data['actual_base_orient']['v'].append(list(rpy))
        data['actual_base_vel_lin']['t'].append(ts)
        data['actual_base_vel_lin']['v'].append([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])
        data['actual_base_vel_ang']['t'].append(ts)
        data['actual_base_vel_ang']['v'].append([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ])
    elif topic in TOPIC_MAP:
        key = TOPIC_MAP[topic]
        data[key]['t'].append(ts)
        data[key]['v'].append(list(msg.data))
    elif topic in TOPIC_MAP_V3:
        key = TOPIC_MAP_V3[topic]
        data[key]['t'].append(ts)
        data[key]['v'].append([msg.vector.x, msg.vector.y, msg.vector.z])
    elif topic in TOPIC_MAP_JS:
        key = TOPIC_MAP_JS[topic]
        if data[key]['names'] is None and msg.name:
            data[key]['names'] = list(msg.name)
        data[key]['t'].append(ts)
        data[key]['v'].append(list(msg.position))
bag.close()

# ------------------------------------------------------------------ #
# Impedance Gains laden
# Quelle: kangaroo_controller_configuration/local_joint_control/impedance_control/
# Ein yaml pro Joint mit stiffness_factor und damping_factor
# ------------------------------------------------------------------ #
IMPEDANCE_YAML_DIR = '/catkin_ws/install/share/kangaroo_controller_configuration/' \
                     'config/local_joint_control/impedance_control'

def load_impedance_gains(yaml_dir):
    gains = {}  # {joint_name: {'ks': float, 'kd': float}}
    if not os.path.isdir(yaml_dir):
        print("  [WARN] impedance_control Verzeichnis nicht gefunden: " + yaml_dir)
        return gains
    for fname in os.listdir(yaml_dir):
        if not fname.endswith('.yaml'):
            continue
        joint_name = fname.replace('.yaml', '')
        fpath = os.path.join(yaml_dir, fname)
        ks, kd = 0.0, 0.0
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('stiffness_factor'):
                    try:    ks = float(line.split(':')[1].strip())
                    except: pass
                elif line.startswith('damping_factor'):
                    try:    kd = float(line.split(':')[1].strip())
                    except: pass
        gains[joint_name] = {'ks': ks, 'kd': kd}
    return gains

impedance_gains = load_impedance_gains(IMPEDANCE_YAML_DIR)
print("  Impedance Gains geladen: " + str(len(impedance_gains)) + " Joints")
for jn in sorted(impedance_gains.keys()):
    g = impedance_gains[jn]
    print("    {:<35s}  ks={:8.1f}  kd={:7.3f}".format(jn, g['ks'], g['kd']))

# ------------------------------------------------------------------ #
# tau_computed berechnen
# tau[i] = ks * (desired_pos[i] - actual_pos[i]) + kd * (-actual_vel[i])
# desired_pos  = /policy_debug/actions_scaled  (ACTION_NAMES Reihenfolge)
# actual_pos   = /joint_states.position
# actual_vel   = /joint_states.velocity
# Interpolation: actions_scaled auf joint_states Zeitachse interpolieren
# ------------------------------------------------------------------ #
def compute_tau(data_actions_scaled, data_pos, data_vel, gains, action_names, js_names):
    """Berechnet tau_computed auf der joint_states Zeitachse."""
    if not data_actions_scaled['t'] or not data_pos['t']:
        return empty_ts()

    t_js   = np.array(data_pos['t'])
    v_pos  = data_pos['v']    # shape (T_js, N_js)
    v_vel  = data_vel['v']    # shape (T_js, N_js)
    t_act  = np.array(data_actions_scaled['t'])
    v_act  = data_actions_scaled['v']  # shape (T_act, N_act)

    tau_out = np.zeros((len(t_js), len(action_names)))

    for ai, jname in enumerate(action_names):
        if jname not in gains:
            continue
        ks = gains[jname]['ks']
        kd = gains[jname]['kd']

        # desired_pos interpolieren auf joint_states Zeitachse
        if ai < v_act.shape[1]:
            des = np.interp(t_js, t_act, v_act[:, ai])
        else:
            continue

        # actual_pos / vel aus joint_states
        if jname in js_names:
            ji = js_names.index(jname)
            act_pos = v_pos[:, ji]
            act_vel = v_vel[:, ji]
        else:
            continue

        tau_out[:, ai] = ks * (des - act_pos) + kd * (-act_vel)

    result = empty_ts()
    result['t']     = list(t_js)
    result['v']     = tau_out
    result['names'] = action_names
    return result

# Zeitachse normalisieren
all_t0 = [d['t'][0] for d in data.values() if d['t']]
t0 = min(all_t0) if all_t0 else 0.0
for d in data.values():
    d['t'] = [t - t0 for t in d['t']]
    d['v'] = np.array(d['v']) if d['v'] else np.empty((0, 0))

# Fallback: falls Namen nicht aus Message kamen (altes Bag-Format Float64MultiArray)
for key in ['pol_joint_pos_rel', 'pol_joint_pos_abs', 'pol_joint_vel']:
    if data[key]['names'] is None:
        data[key]['names'] = JOINT_NAMES
for key in ['pol_actions_raw', 'pol_actions_scaled']:
    if data[key]['names'] is None:
        data[key]['names'] = ACTION_NAMES

# tau_computed berechnen (nach Normalisierung, da numpy arrays benoetigt)
if has_data(data['pol_actions_scaled']) and has_data(data['joint_pos_raw']):
    js_names = data['joint_pos_raw']['names'] or []
    data['tau_computed'] = compute_tau(
        data['pol_actions_scaled'], data['joint_pos_raw'], data['joint_vel_raw'],
        impedance_gains, ACTION_NAMES, js_names)
    print("  tau_computed: " + str(len(data['tau_computed']['t'])) + " Zeitschritte berechnet")
else:
    print("  [SKIP] tau_computed: actions_scaled oder joint_states fehlen")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ------------------------------------------------------------------ #
# Plot-Helfer
# ------------------------------------------------------------------ #
GROUP_INDEPENDENT = {'imu', 'ang_vel', 'lin_acc', 'cmd_vel', 'gravity', 'command',
                     'actual_base_pos', 'actual_base_orient',
                     'actual_base_vel_lin', 'actual_base_vel_ang',
                     'ft_left_ankle', 'ft_right_ankle'}

def _plot_in_pdf(name, folder):
    """Prueft ob dieser Plot in die PDF soll."""
    # Comparison-Plots nie in PDF
    if 'comparison' in folder or '7_comparison' in folder:
        return False
    # Gruppenunabhaengige Plots immer rein
    for key in GROUP_INDEPENDENT:
        if key in name:
            return True
    # Kein Filter: alle Plots
    if pdf_filter is None:
        return True
    # Gruppenfilter
    for grp in pdf_filter:
        if name.endswith('_' + grp):
            return True
    return False

def pdf_section_separator(title, subtitle=''):
    """Fuegt eine Trennseite zwischen Sections ein."""
    fig_sep = plt.figure(figsize=(22, 5))
    fig_sep.patch.set_facecolor('#2c3e50')
    fig_sep.text(0.5, 0.55, title,
                 ha='center', va='center', fontsize=42, fontweight='bold',
                 color='white', transform=fig_sep.transFigure)
    if subtitle:
        fig_sep.text(0.5, 0.28, subtitle,
                     ha='center', va='center', fontsize=16,
                     color='#95a5a6', transform=fig_sep.transFigure)
    _pdf.savefig(fig_sep, bbox_inches='tight', facecolor=fig_sep.get_facecolor())
    plt.close(fig_sep)

def save_svg(fig, folder, name):
    path = os.path.join(folder, name + '.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    print("  Saved: " + path)
    if _plot_in_pdf(name, folder):
        _pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_formula(fig, formula):
    """Fuegt eine Formel-Zeile als Untertitel unter den Suptitle."""
    fig.text(0.5, 0.01, 'Formel:  ' + formula,
             ha='center', va='bottom', fontsize=11,
             style='italic', color='dimgray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.8))


def plot_xyz(d, title, ylabel, folder, fname, labels=('x', 'y', 'z'), note='', formula=''):
    if not has_data(d, 3):
        print("  [SKIP] " + fname)
        return
    fig, ax = plt.subplots(figsize=(22, 5))
    fig.subplots_adjust(bottom=0.12 if formula else 0.08)
    ax.plot(d['t'], d['v'][:, 0], label=labels[0], color='tab:red')
    ax.plot(d['t'], d['v'][:, 1], label=labels[1], color='tab:green')
    ax.plot(d['t'], d['v'][:, 2], label=labels[2], color='tab:blue')
    full_title = title + ('\n' + note if note else '')
    ax.set_title(full_title)
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.legend()
    ax.grid(True)
    if formula:
        add_formula(fig, formula)
    save_svg(fig, folder, fname)


def plot_joint_groups(d, groups, title_prefix, unit, folder, file_prefix, note='', formula=''):
    if not has_data(d):
        print("  [SKIP] " + file_prefix)
        return
    names = d['names'] or []
    for group_name, group_joints in groups:
        fig, axes = plt.subplots(len(group_joints), 1,
                                 figsize=(22, 4 * len(group_joints)), sharex=True)
        if len(group_joints) == 1:
            axes = [axes]
        fig.subplots_adjust(bottom=0.08 if formula else 0.05)
        full_title = title_prefix + ' - ' + group_name + ' ' + unit
        if note:
            full_title += '\n' + note
        fig.suptitle(full_title, fontsize=14)
        for ax, jname in zip(axes, group_joints):
            if jname in names:
                idx = names.index(jname)
                if idx < d['v'].shape[1]:
                    ax.plot(d['t'], d['v'][:, idx],
                            color=colors[idx % len(colors)], label=jname)
            ax.set_ylabel(jname, fontsize=9)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.legend(loc='upper right')
            ax.grid(True)
        axes[-1].set_xlabel('Zeit [s]')
        if formula:
            add_formula(fig, formula)
        save_svg(fig, folder, file_prefix + '_' + group_name)


def plot_xyz_compare(d1, d2, title, ylabel, folder, fname, label1='A', label2='B', note='', formula=''):
    """Vergleich zweier XYZ-Signale nebeneinander."""
    if not has_data(d1, 3) or not has_data(d2, 3):
        print("  [SKIP compare] " + fname)
        return
    fig, axes = plt.subplots(3, 1, figsize=(22, 10), sharex=True)
    fig.subplots_adjust(bottom=0.08 if formula else 0.05)
    fig.suptitle(title + ('\n' + note if note else ''), fontsize=14)
    for i, (axis_label, color1, color2) in enumerate(
            zip(['x', 'y', 'z'], ['tab:red', 'tab:green', 'tab:blue'],
                                  ['salmon', 'lightgreen', 'lightblue'])):
        axes[i].plot(d1['t'], d1['v'][:, i], color=color1, label=label1 + ' ' + axis_label)
        axes[i].plot(d2['t'], d2['v'][:, i], color=color2, label=label2 + ' ' + axis_label,
                     linestyle='--')
        axes[i].set_ylabel(axis_label + '  ' + ylabel)
        axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        axes[i].legend(loc='upper right')
        axes[i].grid(True)
    axes[-1].set_xlabel('Zeit [s]')
    if formula:
        add_formula(fig, formula)
    save_svg(fig, folder, fname)


def plot_joint_compare(d1, d2, groups, title_prefix, unit, folder, file_prefix,
                       label1='A', label2='B', note='', formula=''):
    """Vergleich zweier Joint-Signale pro Gruppe."""
    if not has_data(d1) or not has_data(d2):
        print("  [SKIP compare] " + file_prefix)
        return
    names1 = d1['names'] or []
    names2 = d2['names'] or []
    for group_name, group_joints in groups:
        fig, axes = plt.subplots(len(group_joints), 1,
                                 figsize=(22, 4 * len(group_joints)), sharex=True)
        if len(group_joints) == 1:
            axes = [axes]
        fig.subplots_adjust(bottom=0.08 if formula else 0.05)
        full_title = title_prefix + ' - ' + group_name + ' ' + unit
        if note:
            full_title += '\n' + note
        fig.suptitle(full_title, fontsize=14)
        for ax, jname in zip(axes, group_joints):
            if jname in names1:
                idx = names1.index(jname)
                if idx < d1['v'].shape[1]:
                    ax.plot(d1['t'], d1['v'][:, idx], label=label1, color='tab:blue')
            if jname in names2:
                idx = names2.index(jname)
                if idx < d2['v'].shape[1]:
                    ax.plot(d2['t'], d2['v'][:, idx], label=label2,
                            color='tab:orange', linestyle='--')
            ax.set_ylabel(jname, fontsize=9)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.legend(loc='upper right')
            ax.grid(True)
        axes[-1].set_xlabel('Zeit [s]')
        if formula:
            add_formula(fig, formula)
        save_svg(fig, folder, file_prefix + '_' + group_name)


# ================================================================== #
# 1_sensor_raw  -  Rohe Sensordaten
# ================================================================== #
pdf_section_separator('SENSOR RAW DATA', '/joint_states  |  /base_imu  |  /cmd_vel')
print("\n=== 1_sensor_raw ===")

plot_xyz(data['imu'], 'IMU - Angular Velocity (raw)',
         'rad/s', dirs['sensor'], '01_imu_ang_vel',
         note='Direkter IMU-Output, noch keine Verarbeitung',
         formula='output = /base_imu.angular_velocity  (unveraendert)')

d_lin = {'t': data['imu']['t'],
         'v': data['imu']['v'][:, 3:] if has_data(data['imu'], 6) else np.empty((0,0))}
plot_xyz(d_lin, 'IMU - Linear Acceleration (raw)',
         'm/s^2', dirs['sensor'], '02_imu_lin_acc',
         note='Direkter IMU-Output, noch keine Verarbeitung',
         formula='output = /base_imu.linear_acceleration  (unveraendert)')

plot_xyz(data['cmd_vel'], 'cmd_vel (raw)',
         '', dirs['sensor'], '03_cmd_vel', labels=('vx', 'vy', 'wz'),
         note='Joystick-Input, wird direkt als command an Policy weitergegeben',
         formula='output = [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z]')

plot_joint_groups(data['joint_pos_raw'], JOINT_GROUPS,
                  'joint_pos (raw /joint_states)', '[rad]',
                  dirs['sensor'], '04_joint_pos',
                  note='Absolute Gelenkposition aus /joint_states',
                  formula='output = /joint_states.position  (absolut, unveraendert)')

plot_joint_groups(data['joint_vel_raw'], JOINT_GROUPS,
                  'joint_vel (raw /joint_states)', '[rad/s]',
                  dirs['sensor'], '05_joint_vel',
                  note='Gelenkgeschwindigkeit aus /joint_states',
                  formula='output = /joint_states.velocity  (unveraendert)')

# ================================================================== #
# 2_policy_input  -  Was die Policy als Eingabe sieht
# ================================================================== #
pdf_section_separator('POLICY INPUT', 'joint_pos_rel  |  joint_vel  |  projected_gravity  |  command  |  base_ang_vel  |  base_lin_acc')
print("\n=== 2_policy_input ===")

plot_xyz(data['pol_base_ang_vel'],
         'Policy Input - base_ang_vel', 'rad/s',
         dirs['pol_in'], '01_base_ang_vel',
         note='= IMU angular_velocity direkt (keine Transformation)',
         formula='base_ang_vel = /base_imu.angular_velocity')

plot_xyz(data['pol_base_lin_acc'],
         'Policy Input - base_lin_acc', 'm/s^2',
         dirs['pol_in'], '02_base_lin_acc',
         note='= IMU linear_acceleration direkt (keine Transformation)',
         formula='base_lin_acc = /base_imu.linear_acceleration')

plot_xyz(data['pol_proj_grav'],
         'Policy Input - projected_gravity (mit 180deg X-Korrektur)',
         '', dirs['pol_in'], '03_projected_gravity',
         note='Was die Policy tatsaechlich als projected_gravity bekommt',
         formula='proj_grav = R_x(180deg) * q_imu_conj * [0, 0, -1]')

plot_xyz(data['pol_command'],
         'Policy Input - command [vx, vy, wz]',
         '', dirs['pol_in'], '04_command', labels=('vx', 'vy', 'wz'),
         note='= cmd_vel direkt weitergegeben',
         formula='command = [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z]')

plot_joint_groups(data['pol_joint_pos_rel'], JOINT_GROUPS,
                  'Policy Input - joint_pos (relativ zu default_pos)', '[rad]',
                  dirs['pol_in'], '05_joint_pos_relative',
                  note='Policy sieht NICHT die absolute Position!',
                  formula='joint_pos_input = /joint_states.position[i] - default_joint_pos[i]')

plot_joint_groups(data['pol_joint_vel'], JOINT_GROUPS,
                  'Policy Input - joint_vel', '[rad/s]',
                  dirs['pol_in'], '06_joint_vel',
                  note='Gelenkgeschwindigkeit direkt als Policy-Input',
                  formula='joint_vel_input = /joint_states.velocity  (unveraendert)')

# ================================================================== #
# 3_policy_output  -  Was die Policy ausgibt
# ================================================================== #
pdf_section_separator('POLICY OUTPUT', 'actions raw (y_f)  |  actions scaled (desired_pos)')
print("\n=== 3_policy_output ===")

plot_joint_groups(data['pol_actions_raw'], ACTION_GROUPS,
                  'Policy Output - actions raw (Netzwerk-Output y_f)', '[dimensionslos]',
                  dirs['pol_out'], '01_actions_raw',
                  note='Roher Netzwerk-Output, typisch in [-1, 1]. Noch keine Einheit.',
                  formula='y_f[i] = network(obs)   ->   Abweichung von default_pos in Einheiten von action_scale')

plot_joint_groups(data['pol_actions_scaled'], ACTION_GROUPS,
                  'Policy Output - desired_pos (= actions_scaled = desired_state.position)', '[rad]',
                  dirs['pol_out'], '02_actions_scaled',
                  note='Absolute Zielposition fuer den Controller. Identisch mit desired_state.position.',
                  formula='desired_pos[i] = action_scale[i] * y_f[i] + default_joint_pos[i]   (Bsp: leg_length = 0.172 * y_f + 0.600)')

# ================================================================== #
# 4_torques  -  Torques
# ================================================================== #
pdf_section_separator('TORQUES', 'tau berechnet (ks*(des-act) + kd*(-vel))')
print("\n=== 4_torques ===")

# tau_computed: Gains pro Gruppe als Info in der Formel
def gains_note(group_joints, gains):
    parts = []
    for jn in group_joints:
        if jn in gains:
            parts.append('{}: ks={:.0f} kd={:.1f}'.format(
                jn.replace('_joint',''), gains[jn]['ks'], gains[jn]['kd']))
    return '  |  '.join(parts)

for group_name, group_joints in ACTION_GROUPS:
    gnote = gains_note(group_joints, impedance_gains)
    d_tau = data['tau_computed']
    if not has_data(d_tau):
        print("  [SKIP] 02_tau_computed_" + group_name)
        continue
    fig, axes = plt.subplots(len(group_joints), 1,
                             figsize=(22, 4 * len(group_joints)), sharex=True)
    if len(group_joints) == 1:
        axes = [axes]
    fig.suptitle('tau BERECHNET [Nm] - ' + group_name +
                 '\n(theoretisch aus Impedance-Formel, NICHT gemessen)', fontsize=14)
    fig.subplots_adjust(bottom=0.10)
    names = d_tau['names'] or []
    for ax, jname in zip(axes, group_joints):
        if jname in names:
            idx = names.index(jname)
            if idx < d_tau['v'].shape[1]:
                ax.plot(d_tau['t'], d_tau['v'][:, idx],
                        color='tab:red', label=jname)
        ks = impedance_gains.get(jname, {}).get('ks', 0.0)
        kd = impedance_gains.get(jname, {}).get('kd', 0.0)
        ax.set_ylabel(jname, fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title('{}: ks={:.1f}  kd={:.3f}  ->  tau = {:.1f}*(des-act) + {:.3f}*(-vel)'.format(
            jname, ks, kd, ks, kd), fontsize=9, loc='left')
    axes[-1].set_xlabel('Zeit [s]')
    add_formula(fig,
        'tau[i] = ks[i]*(desired_pos[i]-actual_pos[i]) + kd[i]*(-actual_vel[i])'
        '   Gains: kangaroo_controller_configuration/impedance_control/')
    save_svg(fig, dirs['ctrl_in'], '02_tau_computed_' + group_name)

# ================================================================== #
# 5_subscriber_controller
# ================================================================== #
pdf_section_separator('SUBSCRIBER CONTROLLER - INPUT',
                      '/subscriber_controller/desired_state  (von Policy)')
print("\n=== 5_subscriber_controller — INPUT ===")

plot_joint_groups(data['desired_state'], ACTION_GROUPS,
                  '[CONTROLLER INPUT]  desired_state.position', '[rad]',
                  dirs['sub_ctrl'], '01_INPUT_desired_pos',
                  note='Was die Policy an den Subscriber Controller schickt',
                  formula='desired_pos[i] = action_scale[i]*y_f[i] + default_joint_pos[i]   velocity=0   effort=0')

pdf_section_separator('SUBSCRIBER CONTROLLER - OUTPUT',
                      '/subscriber_controller/actual_js_state  |  actual_base_state')
print("\n=== 5_subscriber_controller — OUTPUT ===")

plot_joint_groups(data['actual_js_pos'], JOINT_GROUPS,
                  '[CONTROLLER OUTPUT]  actual_js_state.position', '[rad]',
                  dirs['sub_ctrl'], '02_OUTPUT_actual_pos',
                  note='Gemessene Ist-Position vom Subscriber Controller',
                  formula='output = /subscriber_controller/actual_js_state.position')

plot_joint_groups(data['actual_js_vel'], JOINT_GROUPS,
                  '[CONTROLLER OUTPUT]  actual_js_state.velocity', '[rad/s]',
                  dirs['sub_ctrl'], '03_OUTPUT_actual_vel',
                  note='Gemessene Ist-Geschwindigkeit vom Subscriber Controller',
                  formula='output = /subscriber_controller/actual_js_state.velocity')

plot_xyz(data['actual_base_pos'],
         '[CONTROLLER OUTPUT]  actual_base_state.position', '[m]',
         dirs['sub_ctrl'], '05_OUTPUT_base_pos',
         labels=('x', 'y', 'z'),
         note='Base-Position aus Estimator',
         formula='output = /subscriber_controller/actual_base_state.pose.position')

plot_xyz(data['actual_base_orient'],
         '[CONTROLLER OUTPUT]  actual_base_state.orientation (RPY)', '[rad]',
         dirs['sub_ctrl'], '06_OUTPUT_base_orient',
         labels=('roll', 'pitch', 'yaw'),
         note='Base-Orientierung (Euler RPY) aus Estimator',
         formula='rpy = euler_from_quaternion(actual_base_state.pose.orientation)')

plot_xyz(data['actual_base_vel_lin'],
         '[CONTROLLER OUTPUT]  actual_base_state.twist.linear', '[m/s]',
         dirs['sub_ctrl'], '07_OUTPUT_base_vel_lin',
         labels=('vx', 'vy', 'vz'),
         note='Geschaetzte lineare Base-Geschwindigkeit',
         formula='output = /subscriber_controller/actual_base_state.twist.linear')

plot_xyz(data['actual_base_vel_ang'],
         '[CONTROLLER OUTPUT]  actual_base_state.twist.angular', '[rad/s]',
         dirs['sub_ctrl'], '08_OUTPUT_base_vel_ang',
         labels=('wx', 'wy', 'wz'),
         note='Geschaetzte angulare Base-Geschwindigkeit',
         formula='output = /subscriber_controller/actual_base_state.twist.angular')

# Vergleich desired_state vs actual_js_pos
plot_joint_compare(
    data['desired_state'], data['actual_js_pos'],
    ACTION_GROUPS,
    'desired_state vs actual_js_pos', '[rad]',
    dirs['sub_ctrl'], '09_IO_desired_vs_actual',
    label1='desired (Policy)',
    label2='actual (gemessen)',
    note='Tracking-Fehler: Differenz zwischen Soll- und Ist-Position',
    formula='error[i] = desired_state.position[i] - actual_js_state.position[i]')

# ------------------------------------------------------------------
# WAS DIE HARDWARE IN POLICY-MODUS BEKOMMT
# Hardware-Befehl: position=desired_state.position, velocity=0, effort=0
# Resultat: tau = ks*(desired-actual) + kd*(-vel)  (Impedance Control)
# ------------------------------------------------------------------
print("\n=== 5_subscriber_controller -- HARDWARE COMMAND (Policy Mode) ===")

for group_name, group_joints in ACTION_GROUPS:
    d_tau = data['tau_computed']
    d_des = data['desired_state']
    if not has_data(d_tau) or not has_data(d_des):
        print("  [SKIP] 10_HARDWARE_policy_mode_" + group_name)
        continue
    names_tau = d_tau['names'] or []
    names_des = d_des['names'] or []

    fig, axes = plt.subplots(len(group_joints), 1,
                             figsize=(22, 4 * len(group_joints)), sharex=True)
    if len(group_joints) == 1:
        axes = [axes]
    fig.suptitle(
        '[HARDWARE BEKOMMT - POLICY MODUS]  tau_computed  [Nm]  -  ' + group_name + '\n'
        'In Policy-Modus: effort=0, velocity=0 -> Impedance Control berechnet Drehmoment',
        fontsize=14)
    fig.subplots_adjust(bottom=0.10)

    for ax, jname in zip(axes, group_joints):
        # tau_computed (= was hardware wirklich bekommt)
        if jname in names_tau:
            idx = names_tau.index(jname)
            if idx < d_tau['v'].shape[1]:
                ax.plot(d_tau['t'], d_tau['v'][:, idx],
                        color='tab:red', linewidth=1.5, label='tau_computed (Hardware)')
        # desired_pos als sekundaere y-Achse (Referenz)
        ax2 = ax.twinx()
        if jname in names_des:
            idx = names_des.index(jname)
            if idx < d_des['v'].shape[1]:
                ax2.plot(d_des['t'], d_des['v'][:, idx],
                         color='tab:blue', linewidth=1.0, linestyle='--',
                         alpha=0.6, label='desired_pos (Policy)')
        ax2.set_ylabel('[rad]', fontsize=8, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

        ks = impedance_gains.get(jname, {}).get('ks', 0.0)
        kd = impedance_gains.get(jname, {}).get('kd', 0.0)
        ax.set_ylabel(jname, fontsize=9)
        ax.set_title(
            '{}: ks={:.1f}  kd={:.3f}  ->  tau = {:.1f}*(des-act) + {:.3f}*(-vel)'
            .format(jname, ks, kd, ks, kd), fontsize=8, loc='left')
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax.grid(True)

        # Kombinierte Legende
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Zeit [s]')
    add_formula(fig,
        '[POLICY MODUS]  setDesiredJointState(pos=desired, vel=0, effort=0)  '
        '->  tau[i] = ks[i]*(desired[i]-actual[i]) + kd[i]*(-actual_vel[i])'
        '   |   Gains: impedance_control/*.yaml')
    save_svg(fig, dirs['sub_ctrl'], '10_HARDWARE_policy_mode_' + group_name)

# ================================================================== #
# 6_hardware  -  Gemessene Hardware-Daten
# ================================================================== #
pdf_section_separator('HARDWARE', 'joint_torque_states  |  joint_states.effort  |  FT Sensoren')
print("\n=== 6_hardware ===")

def plot_joint_groups_with_tau(d, groups, title_prefix, unit, folder, file_prefix, note='', formula=''):
    """Wie plot_joint_groups, aber mit tau_computed als Vergleichskurve."""
    if not has_data(d):
        print("  [SKIP] " + file_prefix)
        return
    names = d['names'] or []
    names_tau = data['tau_computed']['names'] or []
    for group_name, group_joints in groups:
        fig, axes = plt.subplots(len(group_joints), 1,
                                 figsize=(22, 4 * len(group_joints)), sharex=True)
        if len(group_joints) == 1:
            axes = [axes]
        full_title = title_prefix + ' - ' + group_name + ' ' + unit
        if note:
            full_title += '\n' + note
        fig.suptitle(full_title, fontsize=14)
        fig.subplots_adjust(bottom=0.10)
        for ax, jname in zip(axes, group_joints):
            if jname in names:
                idx = names.index(jname)
                if idx < d['v'].shape[1]:
                    ax.plot(d['t'], d['v'][:, idx],
                            color=colors[idx % len(colors)], label=jname + ' (gemessen)')
            if has_data(data['tau_computed']) and jname in names_tau:
                idx = names_tau.index(jname)
                if idx < data['tau_computed']['v'].shape[1]:
                    ax.plot(data['tau_computed']['t'], data['tau_computed']['v'][:, idx],
                            color='tab:red', linestyle='--', linewidth=1.2,
                            label='tau_computed')
            ax.set_ylabel(jname, fontsize=9)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.legend(loc='upper right')
            ax.grid(True)
        axes[-1].set_xlabel('Zeit [s]')
        if formula:
            add_formula(fig, formula)
        save_svg(fig, folder, file_prefix + '_' + group_name)

plot_joint_groups_with_tau(data['joint_torque_states'], JOINT_GROUPS,
                  'Hardware Torques  (/joint_torque_states)  vs tau_computed', '[Nm]',
                  dirs['ft'], '01_joint_torque_states',
                  note='Blau=gemessen (/joint_torque_states)  Rot gestrichelt=berechnet (tau_computed)',
                  formula='tau_computed[i] = ks[i]*(desired[i]-actual[i]) + kd[i]*(-vel[i])')

plot_joint_groups_with_tau(data['joint_effort_raw'], JOINT_GROUPS,
                  'joint_states.effort  (/joint_states)  vs tau_computed', '[Nm]',
                  dirs['ft'], '02_joint_states_effort',
                  note='Blau=gemessen (/joint_states.effort)  Rot gestrichelt=berechnet (tau_computed)',
                  formula='tau_computed[i] = ks[i]*(desired[i]-actual[i]) + kd[i]*(-vel[i])')

def plot_ft(d, title, folder, fname):
    if not has_data(d, 6):
        print("  [SKIP] " + fname)
        return
    fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True)
    fig.suptitle(title, fontsize=14)
    axes[0].plot(d['t'], d['v'][:, 0], label='fx', color='tab:red')
    axes[0].plot(d['t'], d['v'][:, 1], label='fy', color='tab:green')
    axes[0].plot(d['t'], d['v'][:, 2], label='fz', color='tab:blue')
    axes[0].set_ylabel('Force [N]')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(d['t'], d['v'][:, 3], label='tx', color='tab:red')
    axes[1].plot(d['t'], d['v'][:, 4], label='ty', color='tab:green')
    axes[1].plot(d['t'], d['v'][:, 5], label='tz', color='tab:blue')
    axes[1].set_ylabel('Torque [Nm]')
    axes[1].set_xlabel('Zeit [s]')
    axes[1].legend()
    axes[1].grid(True)
    save_svg(fig, folder, fname)

plot_ft(data['ft_left'],
        'FT Sensor - Left Ankle  (/ft_sensor_estimator/left_ankle_ft)',
        dirs['ft'], '03_ft_left_ankle')

plot_ft(data['ft_right'],
        'FT Sensor - Right Ankle  (/ft_sensor_estimator/right_ankle_ft)',
        dirs['ft'], '04_ft_right_ankle')

# ================================================================== #
# 7_comparison  -  Vergleiche
# ================================================================== #
print("\n=== 7_comparison ===")

plot_xyz_compare(
    data['pol_proj_grav_raw'], data['pol_proj_grav_corrected'],
    'projected_gravity: raw vs corrected (180deg X-Rotation)',
    '', dirs['compare'], '01_gravity_raw_vs_corrected',
    label1='raw (q_conj * [0,0,-1])',
    label2='corrected (R_x(180) * raw)',
    note='Vergleich zeigt den Effekt der 180deg IMU-Montagekorrektur',
    formula='raw = q_imu_conj * [0,0,-1]   |   corrected = R_x(180deg) * raw')

plot_joint_compare(
    data['pol_joint_pos_rel'], data['pol_joint_pos_abs'],
    JOINT_GROUPS,
    'joint_pos: relativ (Policy-Input) vs absolut', '[rad]',
    dirs['compare'], '02_joint_pos_rel_vs_abs',
    label1='relativ (Policy-Input)',
    label2='absolut (/joint_states)',
    note='Policy sieht nur relative Positionen',
    formula='absolut = relativ + default_joint_pos   |   relativ = absolut - default_joint_pos')

plot_joint_compare(
    data['pol_actions_scaled'], data['desired_state'],
    ACTION_GROUPS,
    'actions_scaled vs desired_state (Sanity Check)', '[rad]',
    dirs['compare'], '03_actions_scaled_vs_desired_state',
    label1='actions_scaled',
    label2='desired_state.position',
    note='Diese beiden muessen identisch sein - Sanity Check fuer die Pipeline',
    formula='desired_pos[i] = scale[i]*y_f[i] + default_pos[i]   ->   tau[i] = stiffness[i]*(desired_pos[i]-actual_pos[i]) + damping[i]*(-actual_vel[i])')

# ================================================================== #
print("\nFertig! Ausgabe in: " + out_base)
print("  1_sensor_raw/              Rohe Sensordaten")
print("  2_policy_input/            Policy-Eingaben (verarbeitet)")
print("  3_policy_output/           Policy-Ausgaben (raw + skaliert)")
print("  4_torques/                 desired_state + tau gemessen + tau berechnet")
print("  5_subscriber_controller/   actual_js_state + actual_base_state + desired vs actual + HARDWARE tau (policy mode)")
print("  6_hardware/                joint_torque_states + joint_states.effort + FT Sensoren")
print("  7_comparison/              Vergleiche (gravity, joint_pos, sanity check)")

_pdf.close()
print("")
print("PDF gespeichert: " + pdf_path)
if pdf_filter is not None:
    print("  Enthaelt: gruppenunabhaengige Plots + Gruppen " + str(sorted(pdf_filter)))
