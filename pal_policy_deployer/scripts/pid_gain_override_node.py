#!/usr/bin/env python3

import os
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
import rclpy


class PidGainOverrideNode(Node):
    def __init__(self) -> None:
        super().__init__("pid_gain_override_node")

        default_pid_yaml = os.path.join(
            get_package_share_directory("kangaroo_mujoco"),
            "config",
            "rl_pids.yaml",
        )

        self.declare_parameter("target_node_name", "/mujoco_node")
        self.declare_parameter("pid_yaml_path", default_pid_yaml)
        self.declare_parameter("retry_period_sec", 0.5)
        self.declare_parameter("max_retries", 120)

        self.target_node_name = self.get_parameter("target_node_name").value
        self.pid_yaml_path = self.get_parameter("pid_yaml_path").value
        self.retry_period_sec = float(self.get_parameter("retry_period_sec").value)
        self.max_retries = int(self.get_parameter("max_retries").value)

        self.retry_count = 0
        self.pending_future = None
        set_parameters_service = f"{self.target_node_name.rstrip('/')}/set_parameters"
        self.client = self.create_client(SetParameters, set_parameters_service)

        self.get_logger().info(
            f"PID override node started. target='{self.target_node_name}', file='{self.pid_yaml_path}'"
        )

        self.timer = self.create_timer(self.retry_period_sec, self._try_override)

    def _load_pid_parameters(self) -> list[Parameter]:
        with open(self.pid_yaml_path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        if not isinstance(data, dict) or len(data) == 0:
            raise RuntimeError("PID YAML is empty or invalid")

        if "/**" in data:
            root = data["/**"]
        elif self.target_node_name in data:
            root = data[self.target_node_name]
        else:
            root = next(iter(data.values()))

        if "ros__parameters" not in root:
            raise RuntimeError("Could not find 'ros__parameters' in PID YAML")

        ros_params = root["ros__parameters"]
        if "pid_gains" not in ros_params:
            raise RuntimeError("Could not find 'pid_gains' in PID YAML")

        flattened: dict[str, Any] = {}
        self._flatten("pid_gains", ros_params["pid_gains"], flattened)

        parameters: list[Parameter] = []
        for name, value in flattened.items():
            if isinstance(value, bool):
                parameters.append(Parameter(name, Parameter.Type.BOOL, value))
            elif isinstance(value, int):
                parameters.append(Parameter(name, Parameter.Type.INTEGER, value))
            elif isinstance(value, float):
                parameters.append(Parameter(name, Parameter.Type.DOUBLE, value))
            else:
                raise RuntimeError(
                    f"Unsupported PID value type for '{name}': {type(value).__name__}"
                )

        return parameters

    def _flatten(self, prefix: str, value: Any, out: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                self._flatten(next_prefix, nested_value, out)
            return

        out[prefix] = value

    def _shutdown(self) -> None:
        self.timer.cancel()
        self.destroy_node()
        rclpy.shutdown()

    def _try_override(self) -> None:
        if self.pending_future is not None:
            return

        if self.max_retries >= 0 and self.retry_count >= self.max_retries:
            self.get_logger().error(
                f"Reached max retries ({self.max_retries}) waiting for parameter service on '{self.target_node_name}'"
            )
            self._shutdown()
            return

        self.retry_count += 1

        if not self.client.wait_for_service(timeout_sec=0.0):
            self.get_logger().warn(
                f"Waiting for parameter service on '{self.target_node_name}' (attempt {self.retry_count})"
            )
            return

        try:
            parameters = self._load_pid_parameters()
        except Exception as exc:
            self.get_logger().error(f"Failed to load PID YAML: {exc}")
            self._shutdown()
            return

        self.get_logger().info(
            f"Applying {len(parameters)} PID parameters to '{self.target_node_name}'"
        )

        request = SetParameters.Request()
        request.parameters = [param.to_parameter_msg() for param in parameters]

        self.pending_future = self.client.call_async(request)
        self.pending_future.add_done_callback(self._on_set_done)

    def _on_set_done(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Set-parameters call failed: {exc}")
            self._shutdown()
            return

        results = response.results

        failed_reasons = [result.reason for result in results if not result.successful]
        if failed_reasons:
            self.get_logger().error("PID override failed for one or more parameters:")
            for reason in failed_reasons[:10]:
                self.get_logger().error(f"  - {reason}")
            self._shutdown()
            return

        self.get_logger().info("PID override applied successfully.")
        self._shutdown()


def main() -> None:
    rclpy.init()
    node = PidGainOverrideNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
