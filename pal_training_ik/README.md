# pal_training_ik

This node reproduces the action side of mjlab 1.6.0's `DifferentialIKAction` used by
`Mjlab-Grasp-Pal-Kangaroo-Grippers`.

It consumes seven raw policy outputs on `/pal_policy/raw_actions` as a
`std_msgs/Float64MultiArray`. The first six values are a Cartesian delta and the seventh
is the gripper action. Each new action is anchored to the currently measured TCP pose.
At 500 Hz the node recomputes damped least-squares IK and publishes arm and gripper joint
targets as `sensor_msgs/JointState`.

Computed commands are always available on `/pal_training_ik/desired_state` for inspection.
Only `publish_commands:=true` forwards them to the
`pal_gravity_compensation` bridge, which sends compensated commands to
`/subscriber_controller/desired_state`.

Publication is disabled by default. Start a dry run with:

```bash
roslaunch pal_training_ik kangaroo_left_grasp.launch
```

After checking the diagnostics and output topic, enable commands with:

```bash
roslaunch pal_training_ik kangaroo_left_grasp.launch publish_commands:=true
```

The manipulation policy deployer must publish its unscaled seven-element ONNX output on
`/pal_policy/raw_actions`; it must not publish those values directly to the subscriber
controller.
