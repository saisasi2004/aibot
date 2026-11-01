"""View the robot model in RViz and move every controllable joint with sliders.

    ros2 launch aibot_description display.launch.py
    ros2 launch aibot_description display.launch.py simple_collision:=false gui:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("aibot_description")
    gui = LaunchConfiguration("gui")

    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            PathJoinSubstitution([pkg, "urdf", "aibot.urdf.xacro"]),
            " lock_legs:=", LaunchConfiguration("lock_legs"),
            " simple_collision:=", LaunchConfiguration("simple_collision"),
        ]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true",
                              description="joint_state_publisher_gui sliders (false: all joints at 0)"),
        DeclareLaunchArgument("lock_legs", default_value="true",
                              description="legs fixed at the README upright pose"),
        DeclareLaunchArgument("simple_collision", default_value="true",
                              description="convex-hull collision meshes instead of visual meshes"),
        DeclareLaunchArgument("rviz_config",
                              default_value=PathJoinSubstitution([pkg, "rviz", "view_robot.rviz"])),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}], output="screen"),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
             condition=IfCondition(gui)),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             condition=UnlessCondition(gui)),
        Node(package="rviz2", executable="rviz2", arguments=["-d", LaunchConfiguration("rviz_config")],
             output="screen"),
    ])
