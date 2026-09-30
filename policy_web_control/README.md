# policy_web_control

Phone page to start, stop and home a grasp policy and to label recorded episodes. It talks to
the policy only through ROS: the buttons call the services of
[`policy_supervisor`](../pal_policy_deployer/README.md#start-stop-and-home-through-ros-supervisor)
and the page shows its `~state`, `~readiness` and `~log` topics.

- [Usage](#usage)
- [Recording episodes](#recording-episodes)
- [Deploy to the robot](#deploy-to-the-robot)

## Usage

Start the supervisor first, then:

```bash
rosrun policy_web_control policy_web_control.py
```

Open `http://<control PC IP>:8000` on a phone on the robot's WiFi. Params: `_port:=`,
`_supervisor:=` (the supervisor's node name), `_recorder:=` (the recorder's node name).

| Button | Service |
| --- | --- |
| STOP | `/pal_policy_supervisor/stop` |
| Start policy | `/pal_policy_supervisor/start` (disabled while the policy is not ready; the page lists why) |
| Go home | `/pal_policy_supervisor/go_home` |

The page has no authentication. Anyone on the robot's network can move the arm.

## Recording episodes

`kang_grasp_supervisor.launch` starts `episode_recorder.py` (`record:=false` to skip). On the
page:

1. Switch "Recording episodes" to ON. Each **Start policy** run is then one rosbag episode under
   `/catkin_ws/ias-nas/home/oopsie-data/<task>/<date>/<episode_id>/`.
2. After the run, label it: outcome, categories and severity (the oopsie-data taxonomy),
   **Exclude** (not the policy's fault, for example tracking lost) or **Discard** (bad recording).

The instruction and the scene come from the task file
(`pal_policy_deployer/config/tasks/<task>.yaml`); the page only shows them. The conversion to
the oopsie-data HDF5 format is a separate, later step.

## Deploy to the robot

The page runs on the robot's control PC (kangaroo-5c). PAL's deploy script copies what
`install()` in `CMakeLists.txt` declares to the robot's persistent storage:

```bash
cd /catkin_ws && catkin_make && source devel/setup.bash
rosrun pal_deploy deploy.py --package policy_web_control kangaroo-5c   # user pal
```

Restart the running page (`rosnode kill /policy_web_control`, then start it again) to serve
the new version. Deploy `pal_policy_deployer` the same way after changes to it.

The robot's filesystem is not persistent: changes outside the deployed packages are lost on
restart. To change files persistently, get root on the robot, run `rw` and `chroot /ro`, and
run `exit` and `ro` when done.

**Start the page at boot (optional).** Edit the startup files on the control PC
(`roscd pal_startup_base`, read-write mode as above):

1. Add `config/apps/policy_web_control.yaml`:
   ```yaml
   rosrun: "policy_web_control policy_web_control.py"
   dependencies: []
   timeout: 20
   ```
2. Add `- policy_web_control` to `config/control/tu_darmstadt.yaml`.

If an app in the startup list fails, the robot refuses to work, so add the page only when it
runs reliably.
