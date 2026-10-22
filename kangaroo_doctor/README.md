# kangaroo_doctor

A page and a command-line tool for when kangaroo-5c doesn't come up. They show which EtherCAT
slave fails (by motor name) and walk you through the usual fixes:

1. **Restart deployer.** This fixes most problems: a deployer that isn't running, a leftover
   deployer still holding the bus, or motors that never switched on.
2. **E-stop power cycle**, if a motor stays in error. This resets a frozen drive: hold the e-stop
   for 2–5 s, release it, then press the blinking red or yellow button to re-arm.
3. **Restart ros_bringup**, if `full_homing` is rejected. Then home again.

## Web page

```bash
rosrun kangaroo_doctor kangaroo_doctor.py            # on the robot
```

Open `http://kangaroo-5c:8001` (`--port` changes the port). The page shows:

- the Kangaroo icon, in the browser tab and next to the title. Its cross shows the state: green
  healthy, red problem, amber starting or an action runs, gray page unreachable.

- a body map of all motors, colored by state. A motor is red when its EtherCAT state is bad,
  and dark red when the bus looks fine but the drive itself reports an error. Tap a motor to see
  its drive details: error code, board and motor errors, e-stop, temperatures, positions.
- the e-stop state: pressed, or waiting to be re-armed
- other non-OK entries from `/diagnostics_agg`, the data the web commander shows
- what is wrong, and a **Next step** button
- the buttons Restart deployer, E-stop power cycle, Full homing, Restart ros_bringup and
  Open/Close left gripper, with the output of the running action
- the controllers of the controller manager: running, stopped, and configured but not loaded
  (from the `~type` params). You can start, stop, load and unload them.
  - **Start** also stops the running controllers that use the same joints. The confirmation
    names them.
  - **Stop** warns that the joints go limp.
  - If an action fails, the output shows the last errors from the deployer log. The controller
    manager itself only answers `ok: False`.
  - State and sensor controllers (no joints) are folded away under "State/sensor controllers".
  - The page warns when `joint_state_controller` is not running, because full homing then fails.

The page has no authentication. Anyone on the robot's network can restart the deployer and
make the robot move (full homing).

## Command line

```bash
rosrun kangaroo_doctor kangaroo_doctor.py status   # slaves by motor name + what to do
rosrun kangaroo_doctor kangaroo_doctor.py fix      # guided: restart deployer, then e-stop
rosrun kangaroo_doctor kangaroo_doctor.py homing   # full_homing; if rejected, restart ros_bringup
rosrun kangaroo_doctor kangaroo_doctor.py gripper close   # or open; runs gripper_left.py on the robot
```

The script also runs in the dev container. There it runs its commands on the robot over ssh as
`pal@kangaroo-5c` (`--robot` changes the target).

## Left gripper

```bash
rosrun kangaroo_doctor gripper_left.py close        # or open, or 0.5 (fraction closed)
rosrun kangaroo_doctor gripper_left.py close --duration 3
```

It moves `gripper_left_finger_joint` with the controller that owns it: `gripper_left_controller`
(a trajectory goal) or `subscriber_controller` (a ramp on the gravity bridge input; refused
while a policy publishes there). It then reports how far the gripper got:

| Exit code | Page shows      | Meaning |
|-----------|-----------------|---------|
| 0         | done            | target reached (within 0.05 rad) |
| 2         | stopped partway | holding an object or blocked (a cube stalls the jaw at ~0.48) |
| 3         | DID NOT MOVE    | a fault: it moved less than 20 % of the way |
| 1         | FAILED          | no controller owns the joint, or topics are missing |

With exit code 3, the output says whether the controller still commands the target, so the
drive is not following (drive error, e-stop, jammed finger), or the controller itself gave
up (aborted goal and its error code). It also shows the drive diagnostics of slave 31.
To fix it, try these in order, and close the gripper again after each to check:

1. **E-stop power cycle.** After a gripper close shows DID NOT MOVE, the page opens the e-stop
   steps and restarts the deployer once you re-arm.
2. **Reboot** the robot.
3. With the default controllers running (`gripper_left_controller`), **detach the gripper**,
   which kills its motor, reattach it, then do the e-stop power cycle.
Close the gripper while it is empty to check it: it must close fully.

## Deploy to the robot

```bash
cd /catkin_ws && catkin_make && source devel/setup.bash
rosrun pal_deploy deploy.py --package kangaroo_doctor kangaroo-5c   # user pal
```

### Start at boot

kangaroo-5c starts the page at boot as a PAL startup app. `pal_startup.sh` also loads
`~/.pal/pal_startup` after PAL's config. `/home` is persistent, so this needs no root and no
`rw`/`chroot`. Two files on the robot (user pal):

`~/.pal/pal_startup/apps/kangaroo_doctor.yaml`:
```yaml
rosrun: "kangaroo_doctor kangaroo_doctor.py"
dependencies: []
timeout: 20
```

`~/.pal/pal_startup/control/kangaroo_doctor.yaml` (its own category: every list under
`/pal_startup/control/` is started, so PAL's `config/control/kangaroo.yaml` stays untouched):
```yaml
- kangaroo_doctor
```

With no dependencies, it starts about 1 s after the startup manager does, before the deployer
is up. The startup manager reads this config only when it starts, so changes take effect at the
next boot. After that, `pal-start kangaroo_doctor` and `pal-stop kangaroo_doctor` control it, and
the log is in `~/.ros/log/latest/kangaroo_doctor.pal.log/current`. To undo, delete the two
files. Two pages can't share port 8001, so stop a page you started by hand first.

The page doesn't use ROS itself, so it keeps running while the deployer and ros_bringup restart.

## Notes

- **Restart deployer** runs the steps of the `pal_restart_deployer` alias, not the alias itself.
  The alias runs `pkill -f deployer`, which also kills the shell running it when that shell's
  command line contains "deployer". The rest of the restart then never runs.
- The deployer can stop with an error while starting (Orocos "exception state"), for example
  when an ankle encoder reads out of range. All motors then still show OP, but the controller
  manager doesn't answer, and `change_controllers` times out waiting for it. The doctor reads
  the deployer log since the last start (`mlockall done.`) and reports the error.
- Slaves 0 and 13 are GX-JC06 junctions. They always stay PREOP.
- The slave-to-motor map is in `SLAVES` in the script. It comes from `slave_position` in
  `/opt/pal/ferrum/share/pal_deployer_cfg/config/device/motor/*.cpf` on the robot.
