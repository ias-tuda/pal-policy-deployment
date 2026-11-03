# pal_policy_deployer

Deploys mjlab RL policies (ONNX) on Kangaroo: locomotion and grasping.

- [Robots and tasks](#robots-and-tasks)
- [Grasp robots](#grasp-robots)
- [Start, stop and home through ROS (supervisor)](#start-stop-and-home-through-ros-supervisor)
- [Manual step-by-step run](#manual-step-by-step-run)
- [Add a new model](#add-a-new-model)
- [Locomotion](#locomotion)
- [Command reference](#command-reference)

Requires ONNX Runtime 1.17.0 and the `kangaroo_robot` package on the `louis/gains-rl` branch.
The robot type must be `full_7dof_gripper` (`cat /tmp/robot_type.yaml`).

## Robots and tasks

```
config/robots/<name>.yaml   a robot part and how it is driven: joints, topics, limits, tcp and
                            jaw geometry, gravity_compensation: true/false
config/tasks/<name>.yaml    what the robot is asked to do with it: robot, model (path and
                            contract), instruction, scene, object frame, goal, safety thresholds
```

Each task has one robot, and several tasks can share a robot. The task file name is also the
recording folder, `<record_dir>/<task>/<date>/<episode>/`. A task's keys sit on top of its
robot's (`model:` is lifted to the top level, `supervisor:` is merged key by key). The result is
the deployment config that the nodes read.

| Task | Robot | What it does |
| --- | --- | --- |
| `grasp-cube` | `left_arm_gravcomp` | Grasp the cube and move it to the goal. Current default. |
| `grasp-cube-upperbody` | `upper_body_nogravcomp` | The same with the waist, both arms and both jaws. |
| `stack-cube` | `left_arm_gravcomp` | Stack the cube (`cube_1`) on a second cube (`cube_2`). Constant goal, see the task file. |

`config/legacy/` holds the older end-effector policy (`cube`, `mock`: IK and a mocked object
pose) as flat config files without a task. Pass them as `config:=<path>`.

```bash
roslaunch pal_policy_deployer kang_grasp_supervisor.launch task:=grasp-cube-upperbody
```

`robot:=`, `instruction:=` and `arrangement:=` override the task for one run. A setup that
oopsie-data should count separately (other instruction or arrangement) is a new task file.

The deployment config as a file, for the manual commands below:

```bash
P=$(rosrun pal_policy_deployer task_config.py grasp-cube)   # prints /tmp/pal_policy_configs/<task>__<robot>.yaml
```

## Grasp robots

| | `left_arm_gravcomp` | `both_arms_gravcomp` | `upper_body_nogravcomp` |
| --- | --- | --- | --- |
| Driven joints | 7 left-arm joints and left jaw | both arms, both jaws | waist, both arms, both jaws |
| Move to start pose | `kang_grasp_gravcomp_move_to_start.launch` | `kang_grasp_gravcomp_move_to_start.launch` | `kang_grasp_upperbody_move_to_start.launch` |
| Policy launch | `kang_grasp_upperbody_deployer.launch` | `kang_grasp_upperbody_deployer.launch` | `kang_grasp_upperbody_deployer.launch` |
| Gravity | the bridge adds the torque (`pal_gravity_compensation`) | the two-arm bridge (`config/gravity_compensation_both_arms.yaml`) | none: the PD error holds the arms |

`kang_grasp_controllers.launch task:=<task>` starts the controllers of every robot: the robot
config names the joints (`impedance_joint_names`, `position_joint_names`) and, with
`gravity_compensation: true`, the gravity bridge (`gravity_compensation_config`), and
`spawn_grasp_controller.py` sets them up. A model must match its robot: joint set and order,
and gravity compensation (`target_preload`).

Common to both:

- Arm joints run in `DirectImpedanceControl` (a joint PD) with the stiffness and damping read
  from `params/env.yaml` beside the ONNX model.
- Jaws (`jaw_control:=current`, the default) run under PAL's current-based gripper controllers
  (`config/gripper_current_controllers.yaml`, `gripper_*_current_controller`). The jaws stay in
  `NoControl` in `subscriber_controller`, and a `joint_state_to_trajectory.py` node
  (`jaw_trajectory_bridge`) forwards their targets from `/subscriber_controller/desired_state`.
  PAL's `gripper_*_controller` are stopped first. `jaw_control:=position` puts the jaws back in
  `DirectPositionControl`, whose motor position loop raises `MotorForceSensorFail` when the
  object blocks the fingers. Switching between the two needs `reload_controller:=true`.
  `spawn_grasp_controller.py` sets these before the controllers start.
- The inferer reads `deployment_start_config.json` beside the ONNX model for the start pose.
- `start_observation_distribution.json` beside the model is the training start distribution
  that the readiness and start checks compare against.
- The policy outputs deltas (`delta_joint_position`). The inferer adds them to the joint
  target and clamps to the soft limits of the robot.

## Start, stop and home through ROS (supervisor)

`policy_supervisor.py` runs the move to the start pose and the policy on service calls, so the
phone page and other tools need no shell. Start the controllers first, then:

```bash
roslaunch pal_policy_deployer kang_grasp_controllers.launch task:=grasp-cube
roslaunch pal_policy_deployer kang_grasp_supervisor.launch task:=grasp-cube
rosservice call /pal_policy_supervisor/go_home   # stop the policy, move to the start pose
rosservice call /pal_policy_supervisor/start     # run the policy
rosservice call /pal_policy_supervisor/stop
```

One job runs at a time. `kang_grasp_supervisor.launch` also starts `episode_recorder.py`
(`record:=false` to skip it).

**Switch the task without a relaunch.** One command; a running policy or home move is
stopped first, as `~stop` does:

```bash
rosservice call /pal_policy_supervisor/set_task mug-upright
```

The supervisor calls `/spawn_grasp_controller/set_task` of the running
`kang_grasp_controllers.launch`, loads the new task and moves to its start pose. The controllers
change in place, without a `subscriber_controller` restart: the impedance gains ramp to the new
model's in 2 s through `dynamic_reconfigure`, and the gravity bridge starts (it first holds the
measured arm pose) or stops. This covers the robots with the same joints, `left_arm_gravcomp`
and `left_arm_nogravcomp`; other joints need a relaunch with `reload_controller:=true`. Hand-set
supervisor params (`~ood_margin`, `~start_config_path`, ...) apply to the startup task only.
`episode_recorder.py` keeps its startup task, so run with `record:=false` when you switch.

Step by step: [TASK_SWITCH.md](TASK_SWITCH.md).

**Task page.** `rosrun pal_policy_deployer task_web_control.py` (in the container, beside the
supervisor) serves `http://<this PC's IP>:8003`: one button per task of the supervisor's `~tasks`
(G2 and G3), a fixed STOP (`~stop`) and whether the supervisor, the controllers (the task's
config and gravity bridge) and the tracker (the task's objects YAML) run the current task. A
tap switches at once; it asks first only while a policy or home move runs. **Re-register tracking**
(`trigger_reregister`, all objects of the tracker's task) repairs a tracker on the wrong object. Object positions
are not checked there; `policy_web_control` (port 8000) still starts the policy.

The FoundationPose tracker switches first, to the task's `tracker_objects` (an objects YAML of
the tracker, relative to its `ROOT_DIR`); the supervisor also switches it at startup. The tracker
runs outside the container on the same ROS master; `tracking_ros` here holds only its `.srv`
files, which must stay identical to the tracker's. A YAML the tracker refuses refuses the whole
switch. The switch does not wait for the objects: if the tracker is not reachable, the switch
is retried every 2 s; when the new objects have no TF after `~tracker_first_wait` (30 s), they are
re-registered every `~tracker_retry_period` (10 s) until they arrive. `~readiness` shows what the
tracker waits for in its notes. A task without `tracker_objects` (`pininsert-s12-in-t-s2`,
`mug-handover`) leaves the tracker as it is; `tracker:=` (empty) leaves it alone for every task.

**Readiness.** While idle, the supervisor runs the inferer with `dry_run:=true`, which
publishes an observation only while the joint state and the object TF are fresh. `start` is
refused unless the last 0.5 s of observations lie inside the training `[min, max]` widened by
the `ood_margins`. The result is on `~readiness` (JSON, 2 Hz). `require_ready:=false` skips
the check.

**Stops while running.** The supervisor stops the policy when `object_to_goal_b` is outside the
training range (plus `running_ood_margin`) for `running_ood_frames` frames, or when no
observation arrives for `running_timeout` seconds. The reason is written to `~log`.

**Safety thresholds** are in the task's `supervisor:` block (`ood_margins`, `running_ood_*`)
and the robot's (`running_timeout`, `startup_timeout`). A ROS param overrides both.

**Topics.** `~state` (`idle`, `policy`, `home`), `~readiness`, `~log`.

**Launch args.** `start_config_path:=` (default: `deployment_start_config.json` beside the
model), `launch_args:="..."` (extra args for the deployer launch), `home_args:="..."` (extra
args for the home move launch file, default `_position_tolerance:=0.1`).

## Manual step-by-step run

For debugging without the supervisor. Shown for `grasp-cube`; for `grasp-cube-upperbody` use
the launch files in the [table above](#grasp-robots).

```bash
P=$(rosrun pal_policy_deployer task_config.py grasp-cube)
M=<model folder with deployment_start_config.json>

# 1. Controllers (env.yaml gains) and the gravity bridge
roslaunch pal_policy_deployer kang_grasp_controllers.launch task:=grasp-cube
# 2. Move to the start pose through the gravity bridge
roslaunch pal_policy_deployer kang_grasp_gravcomp_move_to_start.launch config:=$P start_config_path:=$M/deployment_start_config.json
# 3. Check the first observation against the training start distribution
roslaunch pal_policy_deployer kang_grasp_check_start.launch config:=$P joint_state_topic:=/joint_states
# 4. Run the policy: first without commands (targets on /pal_policy/desired_state), then with
roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch config:=$P start_controllers:=false joint_state_topic:=/joint_states
roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch config:=$P start_controllers:=false joint_state_topic:=/joint_states publish_commands:=true
```

Steps 2 and 4 in one go (the policy starts only if the move succeeds):

```bash
rosrun pal_policy_deployer move_and_deploy.sh $M/deployment_start_config.json config:=$P \
  start_controllers:=false joint_state_topic:=/joint_states publish_commands:=true
```

- `move_to_start_config.py` succeeds when the arm joints are within 0.05 rad of the targets and
  slower than 0.05 rad/s for 0.5 s. Pass `_position_tolerance:=0.1` to accept a pose that is
  up to 0.1 rad off.
- `move_to_start_config.py` keeps the TCP (`tip_link` + `tcp_offset` in `base_link`) above
  `_min_tcp_z:=-0.05`, so the move does not sweep through the table. It checks the straight
  joint path with MoveIt's `/compute_fk` (needs `move_group`) and, where the path dips, bends
  it upwards: one smooth move, no stop. A start or target pose below the limit lowers the limit
  to it. `_check_tcp_height:=false` skips the check (for example without `move_group`).
- `check_start_distribution.py` exits with 0 (pass), 1 (feature outside the training range) or
  2 (setup error). `tolerance:=0.01` widens the range.
- Position control instead of the software PD: stop `subscriber_controller`, then use
  `kang_grasp_position_controllers.launch` and `_state_topic:=/joint_states` on the move.

## Add a new model

1. Put the ONNX model in `/catkin_ws/src/open_lab/oopsie-data/ros/pal-policy-deployment/models/<name>/`, together with `params/env.yaml`,
   `deployment_start_config.json` and `start_observation_distribution.json`.
2. Copy the nearest task in `config/tasks/`. Set `robot:` (the robot part the model was trained
   for), `instruction:`, and under `model:` the `model_path`, `observation_terms`, `action_dim`
   and `target_preload`. The inferer checks the observation and action dimensions against the
   ONNX model at startup.
3. Set the object frame, the goal and the `supervisor:` thresholds for the new setup.
4. Run `task:=<new task>`.

A new robot part or drive mode (other joints, with or without gravity compensation) is a new
file in `config/robots/`.

Supported observation terms: `joint_pos_rel`, `joint_vel`, `last_action`, `joint_target_rel`,
`tcp_pos_b`, `tcp_quat_b`, `object_pos_b`, `object_yaw_b`, `object_yaw_zero_b` (a ball: always (1, 0)), `object_tilt_zero_b` (a ball: always (0, 0)), `object_rot6d_b`, `goal_pos_b`,
`jaw_to_object_b`, `object_yaw_to_jaw`, `object_to_goal_b`. `jaw_to_object_b` has 3 values per
jaw of `jaw_joints` (jaw i on `tip_links[i]`; `jaw_joint` for one jaw); `object_yaw_to_jaw`
uses the first jaw. `joint_target_rel` is the commanded target minus the default pose, also for
`joint_position` models (pal_mjlab `commanded_target_rel`).

## Locomotion

Put the ONNX model in `pal_policy_deployer/models/{task}/{task}_v{version}_mjlab/`
(for example `models/loco/loco_v60_mjlab/`), then:

```bash
roslaunch pal_policy_deployer kang_rl_deployer.launch model:=v60 task:=loco lab:=mjlab
rostopic pub -1 /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## Command reference

| Launch / script | Use |
| --- | --- |
| `kang_grasp_supervisor.launch` | Supervisor and recorder (`task:=`). The normal entry point. |
| `kang_grasp_controllers.launch` | Controllers and gravity bridge of the task's robot (`task:=` or `config:=`). |
| `kang_grasp_position_controllers.launch` | Position control of the left arm and jaw. |
| `kang_grasp_gravcomp_move_to_start.launch` | Move the config's joints to the start pose through the running gravity bridge. |
| `kang_grasp_upperbody_move_to_start.launch` | Move to the start pose with whole-body gravity compensation. |
| `kang_grasp_check_start.launch` | Compare the first observation with the training distribution. |
| `kang_grasp_upperbody_deployer.launch` | Inferer without IK (`publish_commands:=true` sends commands). |
| `kang_grasp_deployer.launch` | Inferer and `pal_training_ik` (older end-effector policy). |
| `kang_rl_deployer.launch` | Locomotion. |
| `move_to_start_config.py` | Smooth move to `deployment_start_config.json`. |
| `move_and_deploy.sh` | Move to the start pose, then deploy. |
