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