# Task switch: one base, every task

Start the controllers, the supervisor and the tracker once, then switch between the left-arm
tasks with one tap (or one command). A switch changes the tracker's objects, the controller
gains and the gravity bridge in place: the arm holds, nothing is relaunched.

- [What runs where](#what-runs-where)
- [Start everything](#start-everything)
- [The task page](#the-task-page)
- [Run a policy](#run-a-policy)
- [Commands](#commands)
- [Tasks](#tasks)
- [Shut down](#shut-down)
- [Troubleshooting](#troubleshooting)

## What runs where

| Where | What | Started by |
| --- | --- | --- |
| Robot (kangaroo-5c) | PAL's controllers, `policy_web_control` (port 8000) | you, step 1 |
| Host PC, outside Docker | FoundationPose tracker (`/tracking_fp_plus_so3_trt_node`) | you, step 2 |
| Dev container | `kang_grasp_controllers.launch`, `kang_grasp_supervisor.launch`, task page (port 8003) | you, step 3 |

All of them use the robot's master (`http://kangaroo-5c:11311`). The container shares the host
network, so the supervisor calls the tracker's services directly.

```
task page ──set_task──► policy_supervisor ──1. switch_task──► tracker (host)
                               │
                               ├──2. set_task──► spawn_grasp_controller ──► gains, gravity bridge
                               │
                               └──3. home move to the new task's start pose
```

## Start everything

### 1. Robot

1. Start the default controller and run `full_homing`:
   `roslaunch kangaroo_controller_configuration default_controllers.launch`
2. Kill the default controller.
3. Start the policy page: `rosrun policy_web_control policy_web_control.py`

### 2. Tracker (host PC)

Start the FoundationPose tracker from the tracker repo, in a terminal on the host:

```bash
cd pal/tracker
pixi shell -e noetic                      # the Noetic shell
pixi run -e noetic ros1-tracking-build    # build the ROS1 workspace
source "$ROS1_WS_DIR/devel/setup.bash"
roslaunch tracking_bringup kangaroo_tracking.launch \
  objects_cfg:=$ROOT_DIR/custom-assets/kang-grasp-cube/grasp-cube.yaml
```

Any objects YAML works; the supervisor switches the tracker to the task's objects at startup.
`grasp-cube.yaml` is a good default.

### 3. Dev container

Run `source /catkin_ws/devel/setup.bash` in every terminal first. Keep all three running.

```bash
# Terminal 1: controllers
roslaunch pal_policy_deployer kang_grasp_controllers.launch task:=grasp-cube

# Terminal 2: supervisor, after terminal 1 shows the controllers running
roslaunch pal_policy_deployer kang_grasp_supervisor.launch task:=grasp-cube record:=false

# Terminal 3: task page
rosrun pal_policy_deployer task_web_control.py
```

- Start **one** supervisor only. A second one with the same name stops both
  ("new node registered with same name").
- `record:=false`: the episode recorder keeps its startup task and does not follow a switch.
- Do not add `reload_controller:=true` unless you mean it: it restarts `subscriber_controller`
  and **the arm drops**. Support the arm first.

Check that the base is up:

```bash
rosservice list | grep set_task
# /pal_policy_supervisor/set_task
# /spawn_grasp_controller/set_task
```

## The task page

Open `http://<container PC IP>:8003` on a phone in the robot's network.

```
┌─────────────────────────────┐
│ STOP                        │  fixed at the top
├─────────────────────────────┤
│ Task: grasp-cube (G2 ...)   │
│ [ Re-register tracking ]    │
│ ✓ Supervisor: runs, idle    │
│ ✓ Controllers: grasp-cube,  │
│   gravity bridge            │
│ ✓ Tracker: runs grasp-cube  │
├─────────────────────────────┤
│ G2 · gravity bridge         │
│ [grasp-cube] [mug-ball] ... │
│ G3 · no bridge              │
│ [pickball-...] [...]        │
├─────────────────────────────┤
│ last 20 supervisor log lines│
└─────────────────────────────┘
```

| Element | What it does |
| --- | --- |
| **STOP** | Stops the policy or home move (`~stop`). The controller holds the last target. |
| **Task buttons** | A tap switches to that task: tracker, controllers, supervisor, then the arm moves to the start pose. The active task is blue. While a policy or home move runs, the page asks first. |
| **Re-register tracking** | The tracker detects all objects of its task again from a fresh segmentation mask. Use it when the tracker follows the wrong object. Clear other objects from the view first. |
| **Supervisor** | ✓ when the supervisor runs. |
| **Controllers** | ✓ when `subscriber_controller` runs the current task's config, with the gravity bridge for G2 and without it for G3. |
| **Tracker** | ✓ when the tracker runs the task's objects YAML. It does not check that the objects are seen or where they are. |

The task buttons are disabled while the supervisor or the controllers are down.

## Run a policy

The task page switches tasks; `policy_web_control` runs the policy.

1. Switch to the task on the task page. Wait until the arm is at the start pose.
2. Open `http://<robot control PC IP>:8000`.
3. Wait for **Ready to start**. It needs a fresh object pose inside the training start range;
   the page lists the reasons when it is not ready.
4. Tap **Start policy**. **STOP** on either page stops it.
5. Tap **Go home** before the next run.

## Commands

The pages call these services; use them from any container terminal.

```bash
rosservice call /pal_policy_supervisor/set_task mug-ball      # switch task
rosservice call /pal_policy_supervisor/go_home
rosservice call /pal_policy_supervisor/start
rosservice call /pal_policy_supervisor/stop
rosservice call /tracking_fp_plus_so3_trt_node/trigger_reregister "objects: []"

rostopic echo /pal_policy_supervisor/log                       # new lines only, not latched
rostopic echo -n 1 /pal_policy_supervisor/task
rostopic echo -n 1 /pal_policy_supervisor/readiness
rosparam get /subscriber_controller/joints/arm_left_1_joint/impedance_params/stiffness_factor
rosnode list | grep gravity_compensation
```

| Group | `arm_left_1` stiffness | gravity bridge |
| --- | --- | --- |
| G2 `left_arm_gravcomp` | 100.0 | runs |
| G3 `left_arm_nogravcomp` | 998.79 | absent |

## Tasks

| Task | Group | Tracker objects (`custom-assets/kang-grasp-cube/`) |
| --- | --- | --- |
| grasp-cube | G2 | grasp-cube.yaml (`cube_1`) |
| mug-ball\* | G2 | mug-ball.yaml (`mug_1`, `ball_1`) |
| mug-turn\*, mug-turn-p1-old\*, mug-coaster\*, mug-upright\* | G2 | turn-mug.yaml (`mug_1`) |
| stack-cube | G2 | stack-cube.yaml (`cube_1`, `cube_2`) |
| pickandplace-cube\*, pickandplace-\* | G3 | grasp-cube.yaml (`cube_1`) |
| pickball-s81-wb-jw-s2 | G3 | ball_60.yaml (`ball_1`) |
| putballintobox-s15-pb-dr-s1 | G3 | ball-block.yaml (`ball_1`, `block_1`) |
| putintobox-s102-wx-a-h22-s1 (putintobox-s101-wx-a-s1\*) | G3 | box-cube.yaml (`box_1`, `cube_1`) |
| stack-cube-siwei\* | G3 | stack-cube.yaml (`cube_1`, `cube_2`) |
| pininsert-s12-in-t-s2\* | G3 | none: the tracker is left as it is |

\* Not on the task page (`HIDDEN_TASKS` in `task_web_control.py`, or
`_hidden_tasks:="[a, b]"`). `set_task` from the command line still accepts them. Since
2026-11-04 the page also hides the mug tasks (no mug), the cube pick-and-place tasks
(P1-P5) and stack-cube-siwei.

The buttons show a name from `TASK_LABELS` in `task_web_control.py`, with the task name below
it. The cube pick-and-place variants are P1-P5: P1 pickandplace-cube, P2
pickandplace-dep1-t-c-s2, P3 pickandplace-dep1-wc-c-s1, P4 pickandplace-s711-wj-jo-s2 (P3
trained longer), P5 pickandplace-wc-abs-c-s78 (absolute joint actions).

The tracker objects are the task file's `tracker_objects:` key. A new task gets a button when
its robot drives the same joints as the running one (G2 and G3 do); restart the supervisor to
pick it up.

## Shut down

1. **STOP** on a page (or `rosservice call /pal_policy_supervisor/stop`).
2. Ctrl-C the task page and the supervisor.
3. Ctrl-C the controllers. `subscriber_controller` keeps running and holds the arm.
4. To power down the arm, support it first, then stop the controller on the robot.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Service [/pal_policy_supervisor/set_task] is not available` | The supervisor does not run. Start terminal 2; check with `rosnode list \| grep supervisor`. |
| Supervisor stops right after start, log says "new node registered with same name" | Two supervisors. Ctrl-C all supervisor launches (`ps aux \| grep kang_grasp_supervisor`), start one. |
| Task page shows no buttons | The supervisor started before `~tasks` existed. Restart the supervisor. |
| Task page: `Address already in use` | Another program holds the port, often a VS Code dev container forward on `127.0.0.1`. The page binds `ROS_IP` by default to avoid that; otherwise use `_host:=<IP>` or `_port:=<free port>`. Open the page at the `ROS_IP` address, not `localhost`. |
| Switch refused: "... still publish" | A node other than the bridge publishes arm commands (a manual launch). Stop it, switch again. |
| Switch refused: "tracker: ..." | The tracker rejects the task's objects YAML (a missing file or mesh). Nothing changed; fix `tracker_objects:` in the task file. |
| Tracker ✗ "not running" | Start the tracker on the host. A switch still works; the supervisor retries the tracker every 2 s. |
| Not ready: "waiting for object TF ..." | The tracker lost the object. Put it in view and tap **Re-register tracking**. After a switch the supervisor re-registers by itself (after 30 s, then every 10 s) until it sees the objects. |
| Tracker follows the wrong object | Remove the other objects from the view, tap **Re-register tracking**. |
| Controllers ✗ "on another task" | The controllers missed a switch. Switch to the task again. |
| `kang_grasp_controllers.launch` refuses: "still running ... with other parameters" | `subscriber_controller` runs another robot's settings. Start with the task it runs (see `rosparam get /spawn_grasp_controller/config_path`) and switch from there, or support the arm and use `reload_controller:=true`. |
| `/tmp/robot_type.yaml` missing | `python3 /opt/pal/ferrum/lib/kangaroo_description/robot_type full_7dof_gripper` (lost when the container restarts). |
| New code not active | Restart the node that changed: the supervisor and the task page load their code at startup. Run `catkin_make` after `.srv` changes. |
