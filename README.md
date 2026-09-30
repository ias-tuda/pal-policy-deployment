# Kangaroo Policy Deployment

ROS packages to run mjlab RL policies on PAL's Kangaroo robot, in simulation and on hardware.

- [Packages](#packages)
- [Setup](#setup)
- [Run a grasp task on the robot](#run-a-grasp-task-on-the-robot)
- [Run in simulation](#run-in-simulation)
- [Troubleshooting](#troubleshooting)

## Packages

| Package | What it does |
| --- | --- |
| [`pal_policy_deployer`](pal_policy_deployer/README.md) | Runs the ONNX policy: inferer, robots and tasks, supervisor, recorder. **Start here.** |
| [`policy_web_control`](policy_web_control/README.md) | Phone page to start, stop and home the policy and to label recorded episodes. |
| [`pal_training_ik`](pal_training_ik/README.md) | IK for the older end-effector grasp policy (7 raw actions to joint targets). |
| [`pal_gravity_compensation`](pal_gravity_compensation/README.md) | Adds gravity feedforward torque to the arm commands. |
| [`subscriber_controller`](subscriber_controller/README.md) | 2 kHz controller that sends the received commands to the motors. |
| [`kangaroo-pal-physics-simulator`](kangaroo-pal-physics-simulator/README.md) | PAL Physics Simulator. Test here before the real robot. |
| [`kangaroo_camera_calibration`](kangaroo_camera_calibration/README.md) | Calibrates the torso camera poses with a ChArUco board. |

Data flow of the left-arm grasp policy:

```
policy_supervisor ──► inferer (pal_policy_deployer)
                         │ /pal_grasp/desired_state_uncompensated
                         ▼
                 pal_gravity_compensation
                         │ /subscriber_controller/desired_state
                         ▼
                 subscriber_controller ──► robot or simulator
```

The upper-body policy publishes straight to the controller (no gravity bridge). The older
end-effector policy adds `pal_training_ik` between the inferer and the bridge.

## Setup

```bash
cd /catkin_ws && catkin_make && source devel/setup.bash
```

ONNX Runtime 1.17.0 (once per machine):

```bash
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar xzf onnxruntime-linux-x64-1.17.0.tgz && sudo mv onnxruntime-linux-x64-1.17.0 /opt/onnxruntime
echo /opt/onnxruntime/lib | sudo tee /etc/ld.so.conf.d/onnxruntime.conf && sudo ldconfig
```

## Run a grasp task on the robot

A *task* (`pal_policy_deployer/config/tasks/`) picks a *robot* (the robot part and how it is
driven), the model, the instruction and the scene. Its name is also the recording folder. See
[Robots and tasks](pal_policy_deployer/README.md#robots-and-tasks).

| Where | Step |
| --- | --- |
| Robot | 1. Start the default controller and run `full_homing`. |
| Robot | 2. On the 5m PC: reset `ntpd` (should not be required anymore - only if the supervisor complains about clocks being off). |
| Robot | 3. Kill the default controller. |
| Robot | 4. Start the web page: `rosrun policy_web_control policy_web_control.py` |
| Tracker repo | 5. Start the FoundationPose tracker (publishes the `cube_1` TF). |
| Dev container | 6. `roslaunch pal_policy_deployer kang_grasp_controllers.launch task:=grasp-cube` |
| Dev container | 7. `roslaunch pal_policy_deployer kang_grasp_supervisor.launch task:=grasp-cube` |
| Dev container | 8. Start RViz. |
| Phone | 9. Open `http://<control PC IP>:8000`: **Go home**, turn recording on, **Start policy**, label the episode. |

Another task: change `task:=` in steps 6 and 7 (for example `grasp-cube-upperbody`).
Episodes are saved under `/catkin_ws/ias-nas/home/oopsie-data/<task>/<date>/<episode_id>/`.

## Run in simulation

```bash
roslaunch kangaroo_pal_physics_simulator kangaroo_pal_physics_simulator.launch   # never against the robot's master
roslaunch pal_policy_deployer kang_grasp_controllers.launch task:=grasp-cube
roslaunch pal_policy_deployer kang_grasp_supervisor.launch task:=grasp-cube
```

The simulator's joint states are on `/simulator/joint_states`. The robot files use `/joint_states`,
so launch files that run the inferer take `joint_state_topic:=/simulator/joint_states`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Move fails with "other publishers on /pal_grasp/desired_state_uncompensated" | A stopped policy left stale nodes: `rosnode cleanup`. |
| An old subscriber controller is still loaded | `rosrun controller_manager controller_manager stop subscriber_controller && rosrun controller_manager controller_manager unload subscriber_controller` |
| Start is disabled on the page | The page lists why (object not tracked, joints out of the training range). `require_ready:=false` on the supervisor launch skips the check. |

## Repository history

This repository combines several repos with their full history (`kangaroo-pal-physics-simulator`,
`pal_policy_deployer` and `subscriber_controller` came from `humanoids-lab/kangaroo/pal` on the
IAS GitLab). Use `git log -- <folder>` for one package.
