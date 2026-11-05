# voice_bridge

A localhost HTTP/JSON API for the voice agent: one call starts a voice task. The bridge checks
in code that the robot may start, switches the task through the policy supervisor when needed,
waits for the start pose and the objects, and starts the policy.

```
voice agent ──HTTP 127.0.0.1:8010──► voice_bridge ──ROS──► pal_policy_supervisor (set_task, start)
                                          │
                                          └── "[voice] ..." lines on /pal_policy_supervisor/log,
                                              shown by the task page (8002) and the policy page (8000)
```

It needs the task switch (`~set_task`, `pal_policy_deployer/SetTask`, see
`pal_policy_deployer/TASK_SWITCH.md`). It runs in the dev container next to the supervisor
(Python 3.6 or later).

## Run

```bash
# Dev container, after the controllers and the supervisor (TASK_SWITCH.md, step 3)
roslaunch voice_bridge voice_bridge.launch

curl -s localhost:8010/tasks
curl -s -X POST localhost:8010/start_task -d '{"task": "stack-cube"}'
curl -s localhost:8010/status
```

## API

| Request | Answer |
| --- | --- |
| `GET /tasks` | `[{"task", "description", "aliases", "available"}]`: the voice tasks; `available` is false when the supervisor cannot switch to the task |
| `GET /status` | `state`, `task`, `ready`, `reasons`, `at_start`, `start_error`, `objects` (TF age per frame), `busy`, `last_result` |
| `POST /start_task` `{"task": "<name>"}` | `{"ok", "code", "message", "task"}` when the policy runs or the command is refused |

A refused command is HTTP 200 with `"ok": false`. HTTP 4xx is a malformed request; 403 a request
from a web page (it has an `Origin` header). `/start_task` blocks until the end: about 10 s with a
switch, at most about 2.5 min (the bridge logs the exact sum at startup). Give the client a read
timeout of 180 s.

## What one command does

1. **Checks**, refused with a `code` and nothing moves:
   - `supervisor_busy`: no `~readiness` newer than the command within 1.5 s. The supervisor
     publishes it only while its lock is free, so a start or switch from a phone page is under way,
     or the supervisor is down.
   - `policy_running`, `home_running`: a voice command never stops anything. STOP is on the phone pages.
   - `not_switchable`: the task is not in the supervisor's `~tasks`.
   - `not_at_start`: a joint the home move drives (the robot's `joint_names`, without the jaw) is more
     than 0.1 rad from the current task's start pose (`deployment_start_config.json`). Tap Go home
     on a phone page first.
   - `other_start_pose`: the new task's start pose is another one. Voice switches only between
     tasks with the same start pose, so the home move of the switch hardly moves the arm.
   - `busy`: another voice command runs, or an earlier service call hangs.
2. **Switch**, when the task is not the current one: `~set_task`, then the home move until
   `Start pose verified` (`switch_refused`, `switch_timeout`, `home_failed`, `home_stopped`,
   `home_timeout`).
3. **Wait** until the arm is still at the start pose, all object frames of the task are fresh on
   `/tf` (0.5 s), the tracker runs the task's objects YAML (`/tracking/active_objects_cfg`) and a
   new `~readiness` says ready: up to 15 s after a switch, 2 s without one (`not_ready`, with what
   is missing).
4. **Start**: `~start` (`start_refused`), then `started`. `start_unknown`: `~start` did not answer
   within 35 s; the policy may still start.

When a phone page switches, starts or homes during a command, the command ends with
`overridden`: the last command wins. A phone STOP is seen during the home move (`home_stopped`).

The limits are `Settings` in `src/voice_bridge/bridge.py`; set them in the launch file's
`settings` param, for example `{ready_timeout: 20.0}`.

## For the supervisor (optional, policy_supervisor.py)

- A STOP while nothing runs logs nothing, so the bridge cannot see a STOP while it waits for the
  objects after a switch, and then starts. With `self._log("=== stop requested ===")` as the first
  line of `stop()` (inside the lock), the bridge ends such a command with `stopped`.
- Between the bridge's last `~readiness` and its `~set_task` call (milliseconds) a phone page can
  still start a policy, which the switch would then stop. A `~set_task_if_idle` service that
  refuses while a job runs (same `SetTask` type) closes this; the bridge would call it instead.

## Voice tasks

`config/voice_tasks.yaml` lists the tasks the voice agent may start, one per behavior, with a
description and aliases for the LLM. A new task: add its task file to pal_policy_deployer
(it gets a button on the task page; restart the supervisor), then add an entry here and restart
the bridge.

## Without the robot

```bash
# Any PC, no ROS: a simulated robot (src/voice_bridge/sim.py); needs PyYAML
python3 scripts/voice_bridge_node.py --sim
curl -s -X POST localhost:8010/sim -d '{"objects_present": false}'   # also move_arm, go_home, stop, set_task, start, home_fails

# Dev container, ROS side: a simulated supervisor, tracker and arm, on a master of its own.
# In every terminal of this example:
export ROS_MASTER_URI=http://localhost:11311
roscore &
roslaunch voice_bridge mock_robot.launch
roslaunch voice_bridge voice_bridge.launch
rosparam set /pal_policy_supervisor/objects_present false

# Tests (no ROS)
cd test && python3 -m unittest
```

`mock_supervisor.py` registers as `/pal_policy_supervisor` and publishes `/tf` and the joint
state. It refuses a master on another PC and one where a supervisor or the controllers run.

## Notes

- No authentication: keep `host` at 127.0.0.1 unless the voice agent runs on another PC.
- The bridge's log lines start with `[voice]`; its own log parser skips them.
