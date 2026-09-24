# pal_policy_deployer

ROS package to deploy mjlab RL policies on KANGAROO.

## Setup

Use ONNX 1.17.0 and kangaroo_robot package with the louis/gains-rl branch.

Add your ONNX model following this structure:
```
pal_policy_deployer/pal_policy_deployer/models/{task}/{task}_v{version}_mjlab/
```

Example for locomotion v60:
```
models/loco/loco_v60_mjlab/
```

## Usage

Launch the deployer:
```bash
roslaunch pal_policy_deployer kang_rl_deployer.launch model:=v60 task:=loco lab:=mjlab
```

Send velocity commands:
```bash
rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"
```

## Kangaroo grasp policy

The grasp mode reuses the MJLab inference node and sends its seven raw actions to
`pal_training_ik`. The object pose is currently mocked in the robot base frame in
`pal_policy_deployer/config/kang_grasp_mock.yaml`.

Start inference and IK. By default the IK targets go to the subscriber controller:

```bash
roslaunch pal_policy_deployer kang_grasp_deployer.launch
```

To disable forwarding and only inspect actions on `/pal_policy/raw_actions` and IK joint
targets on `/pal_training_ik/desired_state`:

```bash
roslaunch pal_policy_deployer kang_grasp_deployer.launch publish_commands:=false
```

The grasp profile is the one place that sets `joint_state_topic` and `action_topic`; the
inferer loads it and the IK node reads both from the inferer. The profile targets the
robot's `/joint_states`. In simulation, override it with the simulator's 500 Hz topic:
`joint_state_topic:=/simulator/joint_states`.

Pass `profile:=/path/to/another_grasp_profile.yaml` to select a different
grasp checkpoint. The grasp profile lists the joint order, observation order, action
dimension, model path, frames, and temporary object/goal observations. The node reads
`deployment_start_config.json` beside the selected ONNX file and uses its positions
and velocities as the baselines for relative joint observations. If that file is absent,
the profile must supply `default_joint_pos` and zero velocity is assumed. Set
`start_config_path` in the profile to use a different JSON file. For a new
grasp checkpoint, update the profile values; the node checks
the resulting observation and action dimensions against the ONNX model at startup.
The terms currently supported are `joint_pos_rel`, `joint_vel`, `last_action`,
`tcp_pos_b`, `object_pos_b`, `object_yaw_b`, `goal_pos_b`, and `tcp_quat_b`.
The object position and yaw remain fixed test inputs until a tracked pose source
is connected.

The grasp launch uses the same JSON for the IK arm posture target and gripper default.
It finds the file beside the profile's ONNX model. For a JSON file elsewhere, pass
`start_config_path:=/path/to/deployment_start_config.json` to the launch file so both
nodes use it.

When starting the grasp controllers, the launch reads the arm actuator stiffness and
damping directly from `params/env.yaml` beside the selected ONNX model. It sets the
seven left-arm impedance parameters before loading `subscriber_controller`; missing or
ambiguous training gains stop controller startup. The currently selected s43 artifact
specifies stiffness 49.94 and damping 3.179 for left-arm joints 1-2, and stiffness
26.177 and damping 1.666 for joints 3-7. PAL's `DirectImpedanceControl` uses these values
as a plain joint PD (τ = K·e + D·ė + τ_ff), so no conversion is needed. See
[What the Kangaroo local joint control types do](../subscriber_controller/README.md#what-the-kangaroo-local-joint-control-types-do).

The left gripper runs in `DirectPositionControl`. The IK's gripper target goes to the
motor's own position loop, not to an effort PD with the training gains (stiffness 20,
damping 0.5). PAL's default for the gripper is `NoControl`, which ignores all gripper
commands. The files in `config/local_joint_control/` provide the motor name and a
2 rad/s command rate limit, about 0.4 s for a 0.75 rad move. The policy itself can ask
for up to 5 rad/s; raise `limiter_max_rate` to allow that.

In training, pal_mjlab's `RateLimitedJointPositionAction` computes the gripper target as
`clip(0.0475 + 0.9 * action, 0, 0.95)`. It then limits the change from the last commanded
target to 0.1 rad once per 50 Hz policy step, starting from 0.0475 after each reset.
`pal_training_ik` reproduces this. The training gripper was a MuJoCo position actuator
with kp 20, kv 0.5 and force limited to 20. Here the motor's own position loop replaces it.

### Move to the start pose first

The start config does not move the robot when the deployer starts. To move the left arm
explicitly, start the grasp controllers in one terminal, then run the move script in
another:

```bash
roslaunch pal_policy_deployer kang_grasp_controllers.launch
rosrun pal_policy_deployer move_to_start_config.py \
  _start_config_path:=/catkin_ws/src/models/2026-09-23_16-53-33_grasp-kangaroo-grippers_s43/deployment_start_config.json
```

The script sends a smooth 50 Hz joint target through the gravity bridge to
`/subscriber_controller/desired_state`
and exits successfully only after the controller's measured arm joints remain within
0.05 rad of the JSON targets and below 0.05 rad/s for 0.5 s. It fails if feedback is stale, the controller
disconnects, tracking error grows too large, or the pose cannot be reached. By default
it moves the seven left arm joints. Use `_joint_names:="[...]"` to select a different
set of joints that the subscriber controller reports in `actual_js_state`.

### Check the start observation

With the arm at the start pose and the controllers still running, check that the
policy's first observation lies in the checkpoint's training start distribution
(`start_observation_distribution.json` beside the ONNX model):

```bash
roslaunch pal_policy_deployer kang_grasp_check_start.launch
```

This starts the inferer with `dry_run:=true`, so it publishes the observation
vector on `/pal_policy/observation` but runs no inference, which keeps `last_action`
at zero. `check_start_distribution.py` averages 1 s of observations and compares each
feature with the training `[min - tolerance, max + tolerance]`. It prints each
failing feature, plus a warning for any feature outside `[p01, p99]`, then exits with 0 on
pass, 1 on failure and 2 on setup errors. Terms are matched by order, and their dimensions
must match. Change the tolerance with `tolerance:=0.01`. To set a tolerance for one
term, run the script directly with, for example, `_term_tolerances:="{joint_pos_rel: 0.05}"`.

After the script succeeds, leave the controllers running and start the policy without
spawning a second controller:

```bash
roslaunch pal_policy_deployer kang_grasp_deployer.launch start_controllers:=false
```
