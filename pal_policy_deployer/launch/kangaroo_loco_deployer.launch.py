from dataclasses import dataclass

import os
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction, SetLaunchConfiguration, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from controller_manager.launch_utils import generate_load_controller_launch_description

from launch_pal.arg_utils import LaunchArgumentsBase


# Definition of opaque function
def _build_metadata_map(context, *args, **kwargs) -> list[SetLaunchConfiguration]:
    import onnx
    onnx_model_path = LaunchConfiguration("onnx_model_path").perform(context)
    model = onnx.load(onnx_model_path)

    graph = model.graph

    input_tensor = graph.input[0].type.tensor_type
    output_tensor = graph.output[0].type.tensor_type

    input_shape = tuple(dim.dim_value for dim in input_tensor.shape.dim)
    output_shape = tuple(dim.dim_value for dim in output_tensor.shape.dim)

    input_shape = input_shape[-1]
    output_shape = output_shape[-1]

    metadata_params = []
    for prop in model.metadata_props:
        metadata_params.append(SetLaunchConfiguration(name = prop.key, value = prop.value))

    metadata_params.append(SetLaunchConfiguration(name = "input_dimension", value = str(input_shape)))
    metadata_params.append(SetLaunchConfiguration(name = "output_dimension", value = str(output_shape)))
    return metadata_params 

# Path definitions
DEFAULT_MODEL_PATH = PathJoinSubstitution([
    FindPackageShare("pal_policy_deployer"),
    "models",
    "kangaroo",
    "locomotion",
    "v62.onnx",
])

# Launch file 
@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):
    use_sim_time: DeclareLaunchArgument = DeclareLaunchArgument(
        name="use_sim_time",
        default_value="true",
        description="Use simulation clock.",
    )

    onnx_model_path: DeclareLaunchArgument = DeclareLaunchArgument(
        name="onnx_model_path",
        default_value=DEFAULT_MODEL_PATH,
        description="Absolute path to the ONNX model file.",
    )

    preprocessor_params: DeclareLaunchArgument = DeclareLaunchArgument(
        name="preprocessor_params",
        default_value="params.yaml",
        description="Path to the preprocessor parameters file.",
    )

    use_ros1_style_monolithic_inferer: DeclareLaunchArgument = DeclareLaunchArgument(
        name="use_ros1_style_monolithic_inferer",
        default_value="false",
        description="Run ROS1-style monolithic inferer node instead of modular preprocessor+inferer+postprocessor pipeline.",
    )

    enable_pid_override: DeclareLaunchArgument = DeclareLaunchArgument(
        name="enable_pid_override",
        default_value="true",
        description="Apply PID gains programmatically to MuJoCo node after startup.",
    )

    mujoco_node_name: DeclareLaunchArgument = DeclareLaunchArgument(
        name="mujoco_node_name",
        default_value="/mujoco_node",
        description="Target node name whose PID parameters will be updated.",
    )

    pid_override_yaml: DeclareLaunchArgument = DeclareLaunchArgument(
        name="pid_override_yaml",
        default_value=PathJoinSubstitution([
            FindPackageShare("kangaroo_mujoco"),
            "config",
            "rl_pids.yaml",
        ]),
        description="YAML file with PID gains to override on the MuJoCo node.",
    )

    # postprocessor_params: DeclareLaunchArgument = DeclareLaunchArgument(
    #     name="postprocessor_params",
    #     default_value="",
    #     description="Path to the postprocessor parameters file.",
    # )

def declare_actions(launch_description: LaunchDescription, launch_args: LaunchArguments):
    PREPROCESSOR_PARAMS = PathJoinSubstitution([
        FindPackageShare("pal_policy_preprocessor"),
        "config",
        "kangaroo",
        "locomotion",
        LaunchConfiguration("preprocessor_params"),
    ])

    # POSTPROCESSOR_PARAMS = PathJoinSubstitution([
    # FindPackageShare("pal_policy_postprocessor"),
    #     "config",
    #     "kangaroo",
    #     "locomotion",
    #     "params.yaml",
    # ])
    
    launch_description.add_action(OpaqueFunction(function=_build_metadata_map))

    pkg_share_folder = get_package_share_directory('pal_policy_deployer')

    pid_override_node = Node(
        package="pal_policy_deployer",
        executable="pid_gain_override_node.py",
        name="pid_gain_override_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_pid_override")),
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "target_node_name": LaunchConfiguration("mujoco_node_name"),
            "pid_yaml_path": LaunchConfiguration("pid_override_yaml"),
        }],
    )

    modular_policy_nodes = GroupAction([
        Node(
            package="pal_policy_preprocessor",
            executable="pal_policy_preprocessor_kangaroo_locomotion",
            name="pal_policy_preprocessor_kangaroo_locomotion",
            output="screen",
            parameters=[
                PREPROCESSOR_PARAMS,
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "input_dimension": ParameterValue(LaunchConfiguration("input_dimension"), value_type=int),
                    "output_dimension": ParameterValue(LaunchConfiguration("output_dimension"), value_type=int),
                    "joint_names": LaunchConfiguration("joint_names"),
                    "default_joint_pos": LaunchConfiguration("default_joint_pos"),
                    "observation_names": LaunchConfiguration("observation_names"),
                }
            ],
        ),
        Node(
            package="pal_policy_inferer",
            executable="pal_policy_inferer_node",
            name="pal_policy_inferer",
            output="screen",
            parameters=[{
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "onnx_model_path": LaunchConfiguration("onnx_model_path"),
                "input_dimension": ParameterValue(LaunchConfiguration("input_dimension"), value_type=int),
                "output_dimension": ParameterValue(LaunchConfiguration("output_dimension"), value_type=int),
            }],
        ),
        Node(
            package="pal_policy_postprocessor",
            executable="pal_policy_postprocessor_kangaroo_locomotion",
            name="pal_policy_postprocessor",
            output="screen",
            parameters=[
                # POSTPROCESSOR_PARAMS,
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "output_dimension": ParameterValue(LaunchConfiguration("output_dimension"), value_type=int),
                    "joint_names": LaunchConfiguration("joint_names"),
                    "action_scale": LaunchConfiguration("action_scale"),
                    "default_joint_pos": LaunchConfiguration("default_joint_pos"),
                }
            ],
        ),
    ], condition=UnlessCondition(LaunchConfiguration("use_ros1_style_monolithic_inferer")))

    monolithic_policy_node = Node(
        package="pal_policy_inferer",
        executable="pal_policy_inferer_kangaroo_mjlab_monolithic_node",
        name="pal_policy_inferer_kangaroo_mjlab_monolithic",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_ros1_style_monolithic_inferer")),
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "onnx_model_path": LaunchConfiguration("onnx_model_path"),
            "joint_states_topic": "/joint_states",
            "imu_topic": "/imu_sensor_broadcaster/imu",
            "cmd_vel_topic": "/cmd_vel",
            "odom_topic": "/odom",
            "command_topic": "/rl_controller/commands",
        }],
    )

    launch_description.add_action(modular_policy_nodes)
    launch_description.add_action(monolithic_policy_node)
    launch_description.add_action(pid_override_node)

    # RL controller: spawn after PID override exits (if enabled)
    rl_controller_after_override = GroupAction(
        [generate_load_controller_launch_description(
            controller_name='rl_controller',
            controller_params_file=os.path.join(
                pkg_share_folder,
                'config', 'kangaroo_rl_controller.yaml'))
         ],
        forwarding=False)

    launch_description.add_action(
        RegisterEventHandler(
            OnProcessExit(
                target_action=pid_override_node,
                on_exit=[rl_controller_after_override],
            ),
            condition=IfCondition(LaunchConfiguration("enable_pid_override")),
        )
    )

    # Fallback: spawn immediately when PID override is disabled
    rl_controller_direct = GroupAction(
        [generate_load_controller_launch_description(
            controller_name='rl_controller',
            controller_params_file=os.path.join(
                pkg_share_folder,
                'config', 'kangaroo_rl_controller.yaml'))
         ],
        forwarding=False,
        condition=UnlessCondition(LaunchConfiguration("enable_pid_override")))

    launch_description.add_action(rl_controller_direct)


def generate_launch_description():
    ld = LaunchDescription()

    launch_arguments = LaunchArguments()
    launch_arguments.add_to_launch_description(ld)

    declare_actions(ld, launch_arguments)

    return ld

