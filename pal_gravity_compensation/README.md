# pal_gravity_compensation

`gravity_compensation_node` subscribes to
`/pal_grasp/desired_state_uncompensated`, fills the arm effort array with KDL
gravity feedforward, and publishes `/subscriber_controller/desired_state`.
It preserves target positions and velocities. Both the training IK node and
the start-pose mover can send commands through this bridge.

The model chain extends to the left gripper base to include its mass. The
configured gravity vector assumes a level `base_link`. The bridge reads the
loaded impedance gains and bounds feedforward effort to 50 Nm for arm joints
1-2 and 25 Nm for joints 3-7. It also budgets for estimated PD effort and
warns if the requested position exceeds that budget. Final actuator limits
remain the responsibility of the lower-level controller.

Run the bridge separately with:

```bash
roslaunch pal_gravity_compensation gravity_compensation.launch
```

The launch file loads `config/gravity_compensation.yaml`. It expects a loaded
`robot_description` and the subscriber controller's impedance gain parameters.
