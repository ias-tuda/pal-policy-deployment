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
