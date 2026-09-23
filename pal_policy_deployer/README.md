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

Start inference and IK with joint-command forwarding disabled:

```bash
roslaunch pal_policy_deployer kang_grasp_deployer.launch
```

Inspect actions on `/pal_policy/raw_actions` and IK joint targets on
`/pal_training_ik/desired_state`. To forward targets to the subscriber controller:

```bash
roslaunch pal_policy_deployer kang_grasp_deployer.launch publish_commands:=true
```

The simulation launch uses `/simulator/joint_states` for the 500 Hz IK loop. On hardware,
override it with the robot's high-rate joint-state topic, for example
`joint_state_topic:=/joint_states`.

Pass `profile:=/path/to/another_grasp_profile.yaml` to select a different
grasp checkpoint. The grasp profile lists the joint order, default pose, observation order, action
dimension, model path, frames, and temporary object/goal observations. For a new
grasp checkpoint, update those values in `kang_grasp_mock.yaml`; the node checks
the resulting observation and action dimensions against the ONNX model at startup.
The terms currently supported are `joint_pos_rel`, `joint_vel`, `last_action`,
`tcp_pos_b`, `object_pos_b`, `object_yaw_b`, `goal_pos_b`, and `tcp_quat_b`.
The object position and yaw remain fixed test inputs until a tracked pose source
is connected.
