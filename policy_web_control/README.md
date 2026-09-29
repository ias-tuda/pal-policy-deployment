# policy_web_control

Phone-friendly web page to start, stop and home a grasp policy. The page's buttons call the
`std_srvs/Trigger` services of `pal_policy_deployer`'s
[`policy_supervisor`](../pal_policy_deployer/README.md#start-stop-and-home-through-ros), and
the page shows its `~state`, `~readiness` and `~log` topics. Start is disabled while
the supervisor reports that the policy is not ready, and the page lists the reasons. This package talks to the policy only through ROS.

| Button | Service |
| --- | --- |
| STOP | `/pal_policy_supervisor/stop` |
| Start policy | `/pal_policy_supervisor/start` |
| Go home | `/pal_policy_supervisor/go_home` |

## Usage

Start the supervisor first, then:

```bash
rosrun policy_web_control policy_web_control.py
```

Open `http://<this PC's IP>:8000` on a phone on the robot's WiFi. Use `_port:=` to change
the port and `_supervisor:=` to change the supervisor's node name.

The page has no authentication. Anyone on the robot's network can move the arm.

## Recording episodes for oopsie-data

With `pal_policy_deployer`'s `episode_recorder.py` running (started by
`kang_grasp_supervisor.launch`, `record:=true`), the page switches recording on and off and labels
the latest episode with quick options:
outcome, categories and severity (the oopsie-data taxonomy), or Exclude (not the policy's fault,
e.g. object tracking lost) / Discard. While recording is on, each Start policy run is one rosbag
episode under `/catkin_ws/ias-nas/home/oopsie-data/<date>/<episode_id>/`. The conversion to the
oopsie-data HDF5 format is a separate, later step. The task instruction and the object
arrangement rarely change, so they are launch args of `kang_grasp_supervisor.launch`
(`instruction:=`, `arrangement:=`); the page only shows them. Use `_recorder:=` to change the recorder's node name.

## Deploying to the robot (kangaroo-5c)

The page runs on the robot's control PC. PAL's deploy script copies the package's installed files
(only what `install()` in `CMakeLists.txt` declares) to the robot's persistent storage. From the
dev container:

```bash
cd /catkin_ws
catkin_make
source devel/setup.bash
rosrun pal_deploy deploy.py --package policy_web_control kangaroo-5c   # user pal
```

Then restart the running page on the robot, so that it serves the new version
(`rosnode kill /policy_web_control`, then start it again as in [Usage](#usage)). Otherwise the
new version is there after the next robot restart.

The robot's filesystem is not persistent: changes outside the deployed packages are lost when the
robot restarts. To change files persistently, get root on the robot, then `rw` and `chroot /ro`.
When you are done, `exit` and `ro`.

### Starting the page at boot (optional)

The startup files are in `roscd pal_startup_base` / `config`. Change them **on the control PC**,
also for apps that run on the multimedia PC. In read-write mode (see above):

1. Add `config/apps/policy_web_control.yaml`:
   ```yaml
   rosrun: "policy_web_control policy_web_control.py"
   dependencies: []
   timeout: 20
   ```
2. Add `- policy_web_control` to `config/control/tu_darmstadt.yaml` (our file for custom changes).

If an app in the startup list fails, the webcommander shows an error and the robot refuses to
work. So add the page only when it runs reliably.
