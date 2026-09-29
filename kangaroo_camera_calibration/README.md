# kangaroo_camera_calibration

Calibrates where an RGB-D camera sits on the torso, using a ChArUco board on the left gripper.
Result: the corrected `<origin>` of the camera's `xacro:sensor_d435` in
`kangaroo_description/urdf/torso/torso.urdf.xacro`. Cameras: `torso_front`, `waist_front`
(`cameras:` in `config/calibration.yaml`).

## Hardware

`hardware/`: print `charuco_board.pdf` at 100 % (check a 30 mm square with a caliper), glue it
into the 3D-printed `charuco_mount.stl` (source: `.scad`) and screw the mount onto the left gripper.

## Steps

```bash
roslaunch kangaroo_camera_calibration teach.launch camera:=torso_front
```
In RViz, right-click the marker: *Suggest pose* (or drag it) → *Move arm to marker* → check the
board detection image → *Record current pose*. Teach 6–8 seeds spread over the workspace
(left/right, high/low, near/far); they are saved in `config/poses/<camera>.yaml`.

```bash
rosrun kangaroo_camera_calibration collect.py --camera torso_front --dry-run   # plans only
rosrun kangaroo_camera_calibration collect.py --camera torso_front             # moves, asks 'yes'
rosrun kangaroo_camera_calibration solve.py data/<camera>_<date>/dataset.yaml [more datasets]
```
Put the printed `<origin>` into the xacro on the robot and restart the robot bringup.

## Safety

Moves are planned by MoveIt with the board attached, then checked against a front-of-body
workspace and board clearance (`safety:` in the config). The table is not modelled; keep the
e-stop at hand.

## Known limitation

The left arm's kinematics are off by about 1° (joint zero offsets; `output_offset` is 0).
Across the workspace this leaves ~4 mm RMS between camera and arm, and the camera result
partly absorbs the arm error.
