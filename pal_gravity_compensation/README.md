# pal_gravity_compensation

Bridge node that adds gravity feedforward torque to arm commands.

- Subscribes to `/pal_grasp/desired_state_uncompensated`.
- Publishes `/subscriber_controller/desired_state` with KDL gravity torque in the effort array.
  Target positions and velocities are unchanged.
- Needs a loaded `robot_description` and the subscriber controller's impedance gains.
- Limits the feedforward to 50 Nm (arm joints 1-2) and 25 Nm (joints 3-7) and warns when the
  requested position exceeds the effort budget. The controller enforces the final actuator limits.

`kang_grasp_controllers.launch` starts it. To run it alone:

```bash
roslaunch pal_gravity_compensation gravity_compensation.launch
```

Config: `config/gravity_compensation.yaml`. The upper-body move to the start pose uses
`pal_policy_deployer/config/gravity_compensation_upperbody.yaml`.
