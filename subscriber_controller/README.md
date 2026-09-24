# Subscriber controller

This example was developed for the [Memmo project](https://www.memmo-project.eu/) to run the MPC controller in a separate, isolated, and more powerful PC. 

The subscriber controller is a ROS controller that runs at 2Khz. It publishes the actual state of the robot `actual_base_state`, `actual_js_state` and sends the reference received by an external publisher to the motors. 

At the beginning, when no references are received, it reads the actual state of the robot and tries to mantain it using either position control or impedance control, depending on the control mode.

This controller could be started with the different local joint control modes developed by PAL:
- *InertiaShapingEffortAnalyticDOBControl / InertiaShapingEffortSimpleDOBControl / InertiaShapingDifferentialEffortAnalyticDOBControl*: Torque control developed by PAL Robotics to use in the real robot.
- *DirectPositionControl*: Position Control.
- *DirectTorqueControl*: Torque Control used in simulation.
- *DirectImpedanceControl*: Joint PD computed in software and sent as effort (see below).
- *DirectEffortControl / DifferentialEffortControl*: Current Control.
- *NoControl*: No control. The joint is only read; its commands are dropped.

## What the Kangaroo local joint control types do

PAL ships these types as binaries (`/opt/pal/ferrum/include/pal_local_joint_control`).
The notes below were confirmed in the PAL physics simulator (2026-09-24) from the
`/introspection_data` values of a running `subscriber_controller`.

| Type | Launch name | Command written to the motor | Motor mode |
| --- | --- | --- | --- |
| `DirectImpedanceControl` | `direct_impedance_control` | effort τ = K·(q_des − q) + D·(q̇_des − q̇) + τ_ff | 2 (effort) |
| `DirectPositionControl` | `direct_position_control` | q_des, rate-limited by `limiter_max_rate` (rad/s) | 0 (position) |
| `NoControl` | `no_control` | none: the joint handle is read-only | unchanged |

**`DirectImpedanceControl` is a plain joint PD controller, despite its name.** There is
no Cartesian impedance and no scaling:

- K is `/subscriber_controller/joints/<joint>/impedance_params/stiffness_factor`.
- D is `damping_factor`, and `integral_factor` adds an I term.
- τ_ff is the `effort` field of `desired_state`, which the gravity bridge fills with
  gravity feedforward.

For example, `arm_left_4_joint` with K = 26.177 and D = 1.666 logged
`scaled_position_error` = 26.177 × `position_error` and
`scaled_velocity_error` = 1.666 × `velocity_error`, and
`local_control_command` = their sum + `desired_joint_torque`.
So MJLab's actuator `stiffness` and `damping` can be copied to K and D unchanged, which
`pal_policy_deployer/scripts/spawn_grasp_controller.py` does.

**The grippers need `DirectPositionControl`.** For `full_7dof_gripper`, PAL's
`local_joint_control/launcher.launch` sets both `gripper_*_finger_joint` to
`NoControl`, so a gripper target in `desired_state` has no effect. In position mode the
motor runs its own position PD. In the simulator this is `pid_gripper_left_finger_joint`,
with P 50, D 1 and no I. It stops about 0.04 rad short of the target, for example
0.086 for a 0.0475 target. This matches the real robot, where PAL drives the gripper
with a position `JointTrajectoryController`. `kang_grasp_controllers.launch` switches the
left gripper to `DirectPositionControl`, with its config in
`pal_policy_deployer/config/local_joint_control/`.

The URDF limits the gripper joints to effort 0.1. Controllers that clamp to URDF limits,
such as `effort_controllers/JointPositionController`, therefore cannot drive the gripper
with an effort PD. The training effort limit is 20.

## Examples:

### Position Control
To just control the left_arm in position control start it as follows:
```
$ roslaunch talos_gazebo talos_gazebo.launch
```
```
$ roslaunch talos_controller_configuration_gazebo default_controllers.launch
```
Wait until the robot reaches the default pose, and then stop the default controllers and then launch:
```
$ roslaunch subscriber_controller subscriber_controller.launch simulation:=true arm_right_control_type:=no_control legs_control_type:=no_control torso_control_type:=no_control arm_left_control_type:=direct_position_control
```
This package also contains an example of a trajectory publisher that sends references for the `arm_left_4_joint` from one limit to the another.
```
$ rosrun subscriber_controller trajectory_publisher
```

### Torque Control
To start the subscriber controller in torque control:
```
$ roslaunch talos_pal_physics_simulator talos_pal_physics_simulator_with_actuators.launch
```
```
$ roslaunch talos_controller_configuration default_controllers.launch
```
Wait until the robot reach the default pose, and stop it. Then start:
```
$ roslaunch subscriber_controller subscriber_controller.launch simulation:=true
```
