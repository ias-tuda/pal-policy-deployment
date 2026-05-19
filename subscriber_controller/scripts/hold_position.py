#!/usr/bin/env python3
"""
Replicates the subscriber_controller fallback behavior in Python.
Excludes femur/knee joints like the deployer (kang_mjlab_node.cpp).

=============================================================================
REGELGESETZ
=============================================================================

Dieses Script berechnet den effort-Feedforward und schickt ihn zusammen mit
der eingefrorenen Startposition als desired_state an den subscriber_controller.
Der subscriber_controller leitet das weiter an DirectImpedanceControl (Hardware).

Das vollstaendige Regelgesetz ergibt sich aus zwei ueberlagerten Ebenen:

  tau_motor = [DirectImpedanceControl]  +  [effort feedforward (dieses Script)]

  tau_motor = ks  * (q_init - q_act)           <- Impedance: Positionsfehler
            + kd_imp * (0 - dq_act)            <- Impedance: Daempfung
            + tau_init                          <- Gravity Compensation (gemessen beim Start)
            + kp * (q_init - q_act)             <- PID: Proportionalanteil
            + ki * integral(q_init - q_act) dt  <- PID: Integralanteil (mit i_clamp)
            + kd_pid * (-dq_act)                <- PID: Differentialanteil

Vereinfacht (gleiche Terme zusammengefasst):

  tau_motor = (ks + kp) * (q_init - q_act)
            + (kd_imp + kd_pid) * (-dq_act)
            + ki * integral(q_init - q_act) dt
            + tau_init

Variablen:
  ks       = stiffness_factor    (aus kangaroo_controller_configuration/impedance_control/*.yaml)
  kd_imp   = damping_factor      (aus kangaroo_controller_configuration/impedance_control/*.yaml)
  kp,ki,kd_pid = PID gains       (aus subscriber_controller/config/kangaroo_subscriber_controller.yaml)
  tau_init = initialer Torque    (gemessen beim Start von loadEtras, beinhaltet Gravity Comp)
  q_init   = initiale Position   (gemessen beim Start von loadEtras)
  q_act    = aktuelle Position   (Sensor)
  dq_act   = aktuelle Geschwindigkeit (Sensor)

Hinweis: DirectImpedanceControl laeuft im Hardware-Interface bei 2000 Hz.
         Dieses Script laeuft als ROS-Callback (~50-100 Hz) und berechnet
         nur den effort-Feedforward. Die Impedance-Ebene ist immer aktiv.
=============================================================================
"""
import rospy
from sensor_msgs.msg import JointState

EXCLUDED_JOINTS = {
    "leg_left_femur_joint",
    "leg_left_knee_joint",
    "leg_right_femur_joint",
    "leg_right_knee_joint",
}

initial_positions = None
initial_torques   = None
pid_gains         = {}   # {name: {p, i, d, i_clamp}}
i_terms           = {}   # integrator state per joint
last_time         = None


def load_pid_gains():
    gains_param = rospy.get_param('/subscriber_controller/gains', {})
    for name, g in gains_param.items():
        if name not in EXCLUDED_JOINTS:
            pid_gains[name] = {
                'p':       float(g.get('p', 0.0)),
                'i':       float(g.get('i', 0.0)),
                'd':       float(g.get('d', 0.0)),
                'i_clamp': float(g.get('i_clamp', 0.0)),
            }
            i_terms[name] = 0.0
    rospy.loginfo('[hold_position] Loaded PID gains for %d joints', len(pid_gains))
    for name in sorted(pid_gains.keys()):
        g = pid_gains[name]
        rospy.loginfo('  %-35s p=%7.1f  i=%5.1f  d=%5.1f  i_clamp=%5.1f',
                      name, g['p'], g['i'], g['d'], g['i_clamp'])


def compute_pid(name, pos_error, vel, dt):
    g = pid_gains.get(name, {'p': 0, 'i': 0, 'd': 0, 'i_clamp': 0})

    # integrator with anti-windup clamp
    i_terms[name] += g['i'] * pos_error * dt
    if g['i_clamp'] > 0:
        i_terms[name] = max(-g['i_clamp'], min(g['i_clamp'], i_terms[name]))

    return g['p'] * pos_error + i_terms[name] + g['d'] * (-vel)


def callback(msg):
    global initial_positions, initial_torques, last_time

    now = rospy.Time.now()

    if initial_positions is None:
        initial_positions = {}
        initial_torques   = {}
        for i, name in enumerate(msg.name):
            if name not in EXCLUDED_JOINTS:
                initial_positions[name] = msg.position[i]
                initial_torques[name]   = msg.effort[i]
        last_time = now
        rospy.loginfo('=' * 70)
        rospy.loginfo('[hold_position] Captured initial state (%d joints):', len(initial_positions))
        for name in sorted(initial_positions.keys()):
            rospy.loginfo('  %-35s pos=%+.4f  init_torque=%+.4f',
                          name, initial_positions[name], initial_torques[name])
        rospy.loginfo('=' * 70)
        return

    dt = (now - last_time).to_sec()
    last_time = now
    if dt <= 0:
        return

    actual_map = {msg.name[i]: (msg.position[i], msg.velocity[i])
                  for i in range(len(msg.name))}

    out = JointState()
    out.header = msg.header

    for name, init_pos in initial_positions.items():
        actual_pos, actual_vel = actual_map.get(name, (init_pos, 0.0))
        pos_error = init_pos - actual_pos

        pid_correction = compute_pid(name, pos_error, actual_vel, dt)
        effort = initial_torques[name] + pid_correction

        out.name.append(name)
        out.position.append(init_pos)
        out.velocity.append(0.0)
        out.effort.append(effort)

        rospy.logdebug('%-35s  err=%+.4f  vel=%+.4f  pid=%+.4f  eff=%+.4f',
                       name, pos_error, actual_vel, pid_correction, effort)

    rospy.loginfo_throttle(2.0,
        '[hold_position] running | length_left=%+.4f  length_right=%+.4f',
        actual_map.get('leg_left_length_joint',  (0,))[0],
        actual_map.get('leg_right_length_joint', (0,))[0])

    pub.publish(out)


rospy.init_node('hold_position')
pub = rospy.Publisher('/subscriber_controller/desired_state', JointState, queue_size=1)

load_pid_gains()

rospy.Subscriber('/subscriber_controller/actual_js_state', JointState, callback)
rospy.loginfo('[hold_position] Waiting for actual_js_state...')
rospy.spin()
