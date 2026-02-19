from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = LaunchConfiguration("config")
    ctrl_config = LaunchConfiguration("ctrl_config")
    planner_config = LaunchConfiguration("planner_config")
    log_folder = LaunchConfiguration("logging_sub_folder")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config"),
            DeclareLaunchArgument("ctrl_config", default_value="default"),
            DeclareLaunchArgument("planner_config", default_value="default"),
            DeclareLaunchArgument("logging_sub_folder", default_value="default"),
            Node(
                package="mm_run",
                executable="mpc_ros",
                name="controller_mpc",
                output="screen",
                arguments=[
                    "--config",
                    config,
                    "--ctrl_config",
                    ctrl_config,
                    "--planner_config",
                    planner_config,
                    "--logging_sub_folder",
                    log_folder,
                ],
            ),
        ]
    )
