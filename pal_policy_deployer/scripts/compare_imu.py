#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vergleicht base_ang_vel und base_lin_acc zwischen zwei Rosbags (Sim vs. Real).
Zeigt Mittelwert, Stdabw. und Side-by-Side Plots.

Verwendung:
  python compare_imu.py <sim.bag> <real.bag>
  python compare_imu.py          # auto: sucht *sim*.bag und *real*.bag
"""
import sys
import os
import glob
import rosbag
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 1.2,
    'grid.alpha': 0.4,
})

script_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Bag-Pfade ermitteln
# ------------------------------------------------------------------ #
if len(sys.argv) == 3:
    bag_sim  = sys.argv[1]
    bag_real = sys.argv[2]
else:
    sims  = sorted(glob.glob(os.path.join(script_dir, '*sim*.bag')),  key=os.path.getmtime)
    reals = sorted(glob.glob(os.path.join(script_dir, '*real*.bag')), key=os.path.getmtime)
    if not sims or not reals:
        print("Keine *sim*.bag oder *real*.bag gefunden.")
        print("Verwendung: python compare_imu.py <sim.bag> <real.bag>")
        sys.exit(1)
    bag_sim  = sims[-1]
    bag_real = reals[-1]

print("Sim:  " + bag_sim)
print("Real: " + bag_real)

out_dir = os.path.join(script_dir, 'compare_imu')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# ------------------------------------------------------------------ #
# Daten einlesen
# ------------------------------------------------------------------ #
def read_imu(bag_path):
    t, ang, acc = [], [], []
    bag = rosbag.Bag(bag_path)
    for _, msg, ts in bag.read_messages(topics=['/base_imu']):
        t.append(ts.to_sec())
        ang.append([msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z])
        acc.append([msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z])
    bag.close()
    t   = np.array(t)
    ang = np.array(ang)
    acc = np.array(acc)
    if len(t):
        t -= t[0]
    return t, ang, acc

print("Lese Sim-Bag...")
t_sim,  ang_sim,  acc_sim  = read_imu(bag_sim)
print("Lese Real-Bag...")
t_real, ang_real, acc_real = read_imu(bag_real)

if not len(t_sim) or not len(t_real):
    print("Keine /base_imu Daten gefunden!")
    sys.exit(1)

# ------------------------------------------------------------------ #
# Statistik
# ------------------------------------------------------------------ #
AXES = ['X', 'Y', 'Z']

def print_stats(label, data):
    print("\n  " + label)
    for i, ax in enumerate(AXES):
        print("    {:s}: mean={:+.4f}  std={:.4f}  min={:+.4f}  max={:+.4f}".format(
            ax, np.mean(data[:,i]), np.std(data[:,i]),
            np.min(data[:,i]),  np.max(data[:,i])))

print("\n=== base_ang_vel [rad/s] ===")
print_stats("Sim",  ang_sim)
print_stats("Real", ang_real)

print("\n=== base_lin_acc [m/s^2] ===")
print_stats("Sim",  acc_sim)
print_stats("Real", acc_real)

# Mittelwerte fuer Zusammenfassung
print("\n=== Mittelwert-Vergleich (Roboter steht still) ===")
print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "", "ang_x", "ang_y", "ang_z", "acc_x", "acc_y", "acc_z"))
print("{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
    "Sim",
    np.mean(ang_sim[:,0]),  np.mean(ang_sim[:,1]),  np.mean(ang_sim[:,2]),
    np.mean(acc_sim[:,0]),  np.mean(acc_sim[:,1]),  np.mean(acc_sim[:,2])))
print("{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
    "Real",
    np.mean(ang_real[:,0]), np.mean(ang_real[:,1]), np.mean(ang_real[:,2]),
    np.mean(acc_real[:,0]), np.mean(acc_real[:,1]), np.mean(acc_real[:,2])))
print("{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
    "Diff",
    np.mean(ang_real[:,0]) - np.mean(ang_sim[:,0]),
    np.mean(ang_real[:,1]) - np.mean(ang_sim[:,1]),
    np.mean(ang_real[:,2]) - np.mean(ang_sim[:,2]),
    np.mean(acc_real[:,0]) - np.mean(acc_sim[:,0]),
    np.mean(acc_real[:,1]) - np.mean(acc_sim[:,1]),
    np.mean(acc_real[:,2]) - np.mean(acc_sim[:,2])))

# ------------------------------------------------------------------ #
# Plots
# ------------------------------------------------------------------ #
# Sim = Blau-Familie (solid), Real = Orange-Familie (dashed)
SIM_COLOR  = '#2471a3'   # blau
REAL_COLOR = '#e67e22'   # orange
AXIS_COLORS = {
    'X': {'sim': '#1a5276', 'real': '#d35400'},  # dunkelblau / dunkelorange
    'Y': {'sim': '#2980b9', 'real': '#e67e22'},  # mittelblau / orange
    'Z': {'sim': '#7fb3d3', 'real': '#f0b27a'},  # hellblau   / hellorange
}

def save_svg(fig, name):
    path = os.path.join(out_dir, name + '.svg')
    fig.savefig(path, format='svg', bbox_inches='tight')
    print("Gespeichert: " + path)
    plt.close(fig)

def plot_compare(t_s, data_s, t_r, data_r, title, ylabel_unit, filename):
    fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=False)
    fig.suptitle(title, fontsize=15)
    for i, ax_name in enumerate(AXES):
        c_sim  = AXIS_COLORS[ax_name]['sim']
        c_real = AXIS_COLORS[ax_name]['real']
        axes[i].plot(t_s, data_s[:,i], color=c_sim,  alpha=0.9, linewidth=1.4,
                     label='Sim ' + ax_name)
        axes[i].plot(t_r, data_r[:,i], color=c_real, alpha=0.9, linewidth=1.4,
                     linestyle='--', label='Real ' + ax_name)
        axes[i].axhline(np.mean(data_s[:,i]), color=c_sim,  linestyle=':',  linewidth=2.0,
                        label='mean Sim={:+.4f}'.format(np.mean(data_s[:,i])))
        axes[i].axhline(np.mean(data_r[:,i]), color=c_real, linestyle='-.', linewidth=2.0,
                        label='mean Real={:+.4f}'.format(np.mean(data_r[:,i])))
        axes[i].set_ylabel(ax_name + ' ' + ylabel_unit)
        axes[i].legend(loc='upper right', ncol=2)
        axes[i].grid(True)
    axes[-1].set_xlabel('Zeit [s]')
    save_svg(fig, filename)

# --- base_ang_vel ---
plot_compare(t_sim, ang_sim, t_real, ang_real,
             'base_ang_vel [rad/s]  --  Sim (blau) vs. Real (orange)',
             '[rad/s]', '01_base_ang_vel_sim_vs_real')

# --- base_lin_acc ---
plot_compare(t_sim, acc_sim, t_real, acc_real,
             'base_lin_acc [m/s^2]  --  Sim (blau) vs. Real (orange)',
             '[m/s^2]', '02_base_lin_acc_sim_vs_real')

print("\nFertig! Plots in: " + out_dir)
