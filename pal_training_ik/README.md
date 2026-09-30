# pal_training_ik

IK node for the older end-effector grasp policy (`Mjlab-Grasp-Pal-Kangaroo-Grippers`, the flat
configs `legacy/cube` and `legacy/mock`). It is not used by the current left-arm and upper-body policies.

- Input: `/pal_policy/raw_actions` (`std_msgs/Float64MultiArray`), seven raw policy outputs:
  a six-value Cartesian delta and the gripper action.
- Output: arm and gripper joint targets at 500 Hz on `/pal_training_ik/desired_state`
  (`sensor_msgs/JointState`). Each action is anchored to the measured TCP pose.
- `publish_commands:=true` also forwards the targets to the `pal_gravity_compensation` bridge.
  It is off by default.

```bash
roslaunch pal_training_ik kangaroo_left_grasp.launch                        # dry run
roslaunch pal_training_ik kangaroo_left_grasp.launch publish_commands:=true # send commands
```

The deployer must publish the unscaled ONNX output on `/pal_policy/raw_actions`, never directly
to the subscriber controller. `kang_grasp_deployer.launch` starts this node together with the
inferer.
