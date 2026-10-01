#!/usr/bin/env bash
# Move to the start pose, then start the grasp policy once the move has succeeded.
# Usage: move_and_deploy.sh START_CONFIG_PATH [roslaunch args for kang_grasp_upperbody_deployer.launch]
# The two steps run in sequence because the move needs to be the only command publisher.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 START_CONFIG_PATH [deployer launch args...]" >&2
  exit 2
fi
start_config_path=$1
shift

# A previous move launch leaves its config (joint_names, ...) on the parameter server.
rosparam delete /move_to_start_config 2>/dev/null || true
rosrun pal_policy_deployer move_to_start_config.py _start_config_path:="$start_config_path"
exec roslaunch pal_policy_deployer kang_grasp_upperbody_deployer.launch "$@"
