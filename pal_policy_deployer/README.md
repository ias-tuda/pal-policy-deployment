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
impedance parameters of the profile's `impedance_joint_names` (default: the seven left-arm
joints) before loading `subscriber_controller`; missing or ambiguous training gains stop
controller startup. The currently selected s43 artifact
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
it moves the seven left arm joints and reopens the left gripper; gripper joints are commanded
but not required to settle (`_unverified_joint_names`). Use `_joint_names:="[...]"` to select a different
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

## Kangaroo upper-body grasp policy

`config/kang_grasp_upperbody.yaml` deploys `Mjlab-Grasp-UpperBody-Standing-ArmV36`
(`model_2999.onnx`). The policy controls the waist, both 7-DoF arms and both jaws. It
reads 92 observations and writes 18 actions. It uses `action_mode: delta_joint_position`.
At each 50 Hz step the inferer adds `action_scale * clip(action, -1, 1)` to the joint
target and clamps the result to `[target_lower, target_upper]`. The inferer computes
`action_scale` as `scale * scale_multiplier` from `actions.joint_pos` in
`params/env.yaml` beside the model, so a retrained speed takes effect without editing the
profile. A profile that still sets `action_scale` is rejected. The inferer publishes the
targets directly, without `pal_training_ik` and without a gravity bridge. The policy was
trained without gravity feedforward, so the PD error of the targets holds the arms up.
The first target is the measured position plus `target_preload`. This is the gravity
offset that the policy starts from in training, so the pose holds at handover. The
object pose is mocked at the training start mean.

The controllers launch puts the waist and both arms under `DirectImpedanceControl` with
the `env.yaml` gains: 150/8 for the waist, 100/6.35 for arm joints 1-4 and 26.177/1.666
for arm joints 5-7. Both jaws use `DirectPositionControl`. The legs stay at `NoControl`,
as in the single-arm setup. In training, stiff position control held the legs, so
something else must keep the robot standing. Stop any other controller that controls the
waist or the arms first, or the controller switch fails.

```bash
# 1. Controllers
roslaunch pal_policy_deployer kang_grasp_upperbody_controllers.launch
# 2. Move to the start pose with whole-tree gravity compensation (bridge stops on exit)
roslaunch pal_policy_deployer kang_grasp_upperbody_move_to_start.launch \
  start_config_path:=/path/to/model_dir/deployment_start_config.json
# 3. Check the first observation against the training start distribution
roslaunch pal_policy_deployer kang_grasp_check_start.launch \
  profile:=$(rospack find pal_policy_deployer)/config/kang_grasp_upperbody.yaml \
  joint_state_topic:=/joint_states
# 4. Inspect targets on /pal_policy/desired_state, then send them to the controller
roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch \
  start_controllers:=false joint_state_topic:=/joint_states
roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch \
  start_controllers:=false joint_state_topic:=/joint_states publish_commands:=true
```

Without `publish_commands:=true`, the targets go only to `/pal_policy/desired_state`.

The move to the start pose runs `pal_gravity_compensation` with
`config/gravity_compensation_upperbody.yaml`. Its `gravity_tip_link` is `base_link`, so the
bridge adds the gravity torque of every link below `base_link` to the waist and both arms,
and the move commands the start pose itself. When the move ends, the bridge stops and the
controller holds the last compensated command. At the start pose, the gravity torque divided
by the stiffness matches the training `target_preload` to within about 2 mrad, so the torque
changes little when the policy takes over.

## Kangaroo left-arm grasp policy with gravity compensation

`config/kang_grasp_leftarm_gc.yaml` deploys
`2026-09-29_07-44-46_grasp-la-gc-v36-s50-nt-smooth-wide-patient-lean-ft8xbpht1p_s123`. The policy
controls the left arm and the left jaw only. It reads 38 observations (`joint_pos_rel`,
`joint_vel`, `last_action`, `joint_target_rel`, `jaw_to_object_b`, `object_yaw_to_jaw`,
`object_to_goal_b`) and writes 8 actions: 7
arm deltas in `delta_joint_position` mode and one binary jaw action. The policy was trained
with gravity compensation: the start `joint_target_rel` mean is about zero, so
`target_preload` is zero. The inferer publishes to `/pal_grasp/desired_state_uncompensated`,
and the left-arm gravity bridge started by `kang_grasp_controllers.launch` adds the gravity
torque while the policy runs. The waist and the right arm stay at `NoControl`. The object
pose comes from the `cube_1` TF frame. The jaw point for `jaw_to_object_b` is the fingertip
midpoint, from the `jaw_mid_*` table in `arm_left_7_link` measured in MJLab.

```bash
P=$(rospack find pal_policy_deployer)/config/kang_grasp_leftarm_gc.yaml
M=/catkin_ws/src/models/kangaroo-grasp-cube/2026-09-29_07-44-46_grasp-la-gc-v36-s50-nt-smooth-wide-patient-lean-ft8xbpht1p_s123
# 1. Left-arm controllers with the env.yaml gains, plus the left-arm gravity bridge
roslaunch pal_policy_deployer kang_grasp_controllers.launch profile:=$P
# 2. Move to the start pose through the gravity bridge
rosrun pal_policy_deployer move_to_start_config.py \
  _start_config_path:=$M/deployment_start_config.json
# 3. Check the first observation against the training start distribution
roslaunch pal_policy_deployer kang_grasp_check_start.launch profile:=$P joint_state_topic:=/joint_states
# 4. Inspect targets on /pal_policy/desired_state, then send them to the controller
roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch profile:=$P \
  start_controllers:=false joint_state_topic:=/joint_states
roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch profile:=$P \
  start_controllers:=false joint_state_topic:=/joint_states publish_commands:=true
```

To run step 2 and the last command of step 4 in one go (skips step 3; the policy starts only
if the move succeeds):

```bash
rosrun pal_policy_deployer move_and_deploy.sh $M/deployment_start_config.json profile:=$P \
  start_controllers:=false joint_state_topic:=/joint_states publish_commands:=true
```

Step 4 uses the upper-body deployer launch because it runs the inferer without
`pal_training_ik`. The profile uses the named ONNX file, which carries the jaw open and
closed positions.

### Start, stop and home through ROS

`policy_supervisor.py` runs the move to the start pose and the policy on ROS service calls, so
other tools (for example the phone page in [`policy_web_control`](../policy_web_control/README.md))
need no shell. Start the controllers of step 1 first, then:

```bash
roslaunch pal_policy_deployer kang_grasp_supervisor.launch profile:=$P
rosservice call /pal_policy_supervisor/go_home   # stop the policy, run move_to_start_config.py
rosservice call /pal_policy_supervisor/start     # step 4 with publish_commands:=true
rosservice call /pal_policy_supervisor/stop      # Ctrl-C to the running job
```

Only one job runs at a time; `start` is refused while one runs.

Instead of the step 3 check, `start` is refused while the policy is not ready. While the policy
is not running, the supervisor runs the inferer with `dry_run:=true`, which publishes an
observation only while the joint state and the object TF are fresh. The policy is ready when the
last 0.5 s of observations lie inside the training `[min, max]` widened by the `ood_margins` in
`kang_grasp_supervisor.launch` (0.15 rad for joints, 3 cm for the object terms). Only such
serious misses block the start; features outside the strict step 3 range are listed as notes.
The result is published as JSON on `~readiness` (latched, 2 Hz). Pass `require_ready:=false`
to start regardless.

While the policy runs, the supervisor stops it (the policy does not resume by itself) when the
object pose leaves the training range or the inputs stop:

- `object_to_goal_b` lies for 3 consecutive frames outside the box spanned by the training start
  range and 0 (object at the goal), widened by 5 cm. The object may move anywhere between its
  start positions and the goal.
- No observation arrives for 1 s. The inferer skips steps while the joint state or the object TF
  is stale, and would otherwise continue once they return.

The reason is written to `~log`. The limits are parameters in `kang_grasp_supervisor.launch`.
The node publishes `idle`, `policy` or `home` on `~state` (latched) and each output line of the
jobs on `~log`. Without `start_config_path:=`, it uses `deployment_start_config.json` beside
the profile's `model_path`. Pass extra deployer args with `launch_args:="..."` and extra
`move_to_start_config.py` params with `home_args:="..."`. The default
`home_args:="_position_tolerance:=0.1"` accepts a start pose that stays up to 0.1 rad off.
