from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = LaunchConfiguration("config")
    log_folder = LaunchConfiguration("logging_sub_folder")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config"),
            DeclareLaunchArgument("logging_sub_folder", default_value="default"),
            Node(
                package="mm_run",
                executable="sim_ros",
                name="sim_ros",
                output="screen",
                parameters=[{"use_sim_time": True}],
                arguments=[
                    "--config",
                    config,
                    "--GUI",
                    "--logging_sub_folder",
                    log_folder,
                ],
            ),
        ]
    )
