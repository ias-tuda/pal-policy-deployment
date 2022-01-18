# Subscriber controller

This example was developed for the [Memmo project](https://www.memmo-project.eu/) to run the MPC controller in a separate and more powerful PC. 

The subscriber controller is a ROS controller that runs at 2Khz. It publishes the actual state of the robot `actual_base_state`, `actual_js_state` and send to the motors the reference received by an external publisher. 

If no reference is received, it sends the previous one. 

At the beginning, when no references have been yet received, it reads the actual state of the robot and tries to mantain it using either position control or impedance control, depending on the control mode.

This controller could be started with the different local control modes developed by PAL:
- *InertiaShapingEffortAnalyticDOBControl / InertiaShapingEffortSimpleDOBControl / InertiaShapingDifferentialEffortAnalyticDOBControl*: Torque control developed by PAL Robotics to use in the real robot.
- *DirectPositionControl*: Position Control.
- *DirectTorqueControl*: Torque Control used in simulation
- *DirectEffortControl / DifferentialEffortControl*: Current Control.
- *NoControl*: No control.

For example to just control the left_arm in position control start it as follows:
```
$ roslaunch talos_gazebo talos_gazebo.launch
```
```
$ roslaunch talos_controller_configuration_gazebo default_controllers.launch
```
Wait until the robot reach the default pose, and stop it. Then start:
```
$ roslaunch subscriber_controller subscriber_controller.launch simulation:=true arm_right_control_type:=no_control legs_control_type:=no_control torso_control_type:=no_control arm_left_control_type:=direct_position_control
```
This package also contains an example of a publisher that sends positions for the `arm_left_4_joint` from limit to limit.
```
$ rosrun subscriber_controller trajectory_publisher
```

To start for example the subscriber controller in torque control:
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
