# Kangaroo Policy Deployment

ROS packages to run mjlab RL policies on PAL's Kangaroo robot, in simulation and on hardware.

## Packages

| Package | What it does |
| --- | --- |
| [`pal_policy_deployer`](pal_policy_deployer/README.md) | Runs the ONNX policy (inferer, preprocessor, launch files). Start here. |
| [`pal_training_ik`](pal_training_ik/README.md) | Turns the grasp policy's 7 raw actions into arm and gripper joint targets (500 Hz IK). |
| [`pal_gravity_compensation`](pal_gravity_compensation/README.md) | Adds gravity feedforward torques to the arm commands. Not used with position control. |
| [`subscriber_controller`](subscriber_controller/README.md) | 2 kHz ROS controller that sends the received commands to the motors. |
| [`kangaroo-pal-physics-simulator`](kangaroo-pal-physics-simulator/README.md) | PAL Physics Simulator for Kangaroo. Test here before using the real robot. |
| [`kangaroo_camera_calibration`](kangaroo_camera_calibration/README.md) | Calibrates the torso/waist camera poses with a ChArUco board on the left arm. |

## How the grasp pipeline fits together

```
policy (pal_policy_deployer)
   │  /pal_policy/raw_actions
   ▼
IK (pal_training_ik)
   │  /pal_grasp/desired_state_uncompensated
   ▼
gravity compensation (pal_gravity_compensation)
   │  /subscriber_controller/desired_state
   ▼
subscriber_controller  ──►  robot or simulator
```

With position control, `joint_state_to_trajectory.py` replaces the last two stages and
forwards the IK targets to `left_arm_controller` and `gripper_left_controller`.

## Build

```bash
cd /catkin_ws
catkin_make
source devel/setup.bash
```

## Quick start: grasp policy

1. Simulation only, start the simulator (never against the robot's master):
   ```bash
   roslaunch kangaroo_pal_physics_simulator kangaroo_pal_physics_simulator.launch
   ```
2. Start the controllers and move the left arm to the start pose:
   ```bash
   roslaunch pal_policy_deployer kang_grasp_controllers.launch
   rosrun pal_policy_deployer move_to_start_config.py _start_config_path:=<path/to/deployment_start_config.json>
   # position control (motor position loops instead of the software PD; stop subscriber_controller first):
   roslaunch pal_policy_deployer kang_grasp_position_controllers.launch
   rosrun pal_policy_deployer move_to_start_config.py _start_config_path:=<path/to/deployment_start_config.json> _state_topic:=/joint_states
   ```
   If the move fails with "other publishers on /pal_grasp/desired_state_uncompensated", a
   stopped policy left stale nodes; remove them with `rosnode cleanup`.
3. Check that the robot's first observation matches the training data:
   ```bash
   roslaunch pal_policy_deployer kang_grasp_check_start.launch
   # simulation:
   roslaunch pal_policy_deployer kang_grasp_check_start.launch joint_state_topic:=/simulator/joint_states
   ```
4. Run the policy and send its commands to the robot:
   ```bash
   roslaunch pal_policy_deployer kang_grasp_deployer.launch start_controllers:=false
   # simulation:
   roslaunch pal_policy_deployer kang_grasp_deployer.launch start_controllers:=false joint_state_topic:=/simulator/joint_states
   ```

To try it without moving the robot, pass `publish_commands:=false`. You can then watch the
outputs on `/pal_policy/raw_actions` and `/pal_training_ik/desired_state`.

For locomotion, model setup and all options, see the
[`pal_policy_deployer` README](pal_policy_deployer/README.md).

## Repository history

This repository combines several repos, and each one keeps its full git history.
`kangaroo-pal-physics-simulator`, `pal_policy_deployer` and `subscriber_controller`
came from separate repos under `humanoids-lab/kangaroo/pal` on the IAS GitLab.
Use `git log -- <folder>` to see the history of one package.


# Clear old subscriber controller

```
rosrun controller_manager controller_manager stop subscriber_controller && rosrun controller_manager controller_manager unload subscriber_controller
```