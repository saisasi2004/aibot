"""Gazebo Harmonic simulation of the fixed-base robot with ros2_control.

    ros2 launch aibot_gazebo sim.launch.py               # GUI
    ros2 launch aibot_gazebo sim.launch.py gui:=false    # headless server
    ros2 launch aibot_gazebo sim.launch.py rviz:=true
    grasping:=true is set by aibot_bringup/teleop.launch.py together with grasp_manager_node.

Startup order: gz server -> robot_state_publisher -> spawn robot (ros_gz_sim create)
-> joint_state_broadcaster -> the six JointGroupPositionControllers.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

GROUP_CONTROLLERS = [
    "waist_controller", "head_controller",
    "left_arm_controller", "right_arm_controller",
    "left_hand_controller", "right_hand_controller",
]
# Graspable models in worlds/pick_table.sdf. Single source of truth for the DetachableJoint
# plugins (URDF), their ROS<->gz bridges (below) and grasp_manager_node (via aibot_bringup).
GRASP_OBJECTS = ["cube_small", "cube_small_2", "cube_blue", "cube_yellow", "box_small", "block_flat",
                 "cylinder_can", "mug", "ball"]
# Static containers (not graspable): poses are bridged too, for evaluation and the scene manager.
CONTAINERS = ["bin", "tray"]
SIDES = ["left", "right"]


def generate_launch_description() -> LaunchDescription:
    desc_share = get_package_share_directory("aibot_description")
    gz_share = get_package_share_directory("aibot_gazebo")
    controllers_file = os.path.join(gz_share, "config", "controllers.yaml")

    gui = LaunchConfiguration("gui")
    grasping = LaunchConfiguration("grasping")
    use_sim_time = {"use_sim_time": True}

    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            os.path.join(desc_share, "urdf", "aibot.urdf.xacro"),
            " sim_mode:=gazebo",
            " controllers_file:=", controllers_file,
            # DetachableJoint plugins only with grasping:=true. They weld every object to both
            # hands when the robot spawns (gz-sim 8 behaviour) and grasp_manager_node, started by
            # aibot_bringup, releases them; without it the arms would be locked together.
            " grasp_objects:=", PythonExpression(["'", ",".join(GRASP_OBJECTS), "' if '", grasping, "' == 'true' else 'none'"]),
            " simple_collision:=", LaunchConfiguration("simple_collision"),
            " head_camera:=", LaunchConfiguration("head_camera"),
            " camera_width:=", LaunchConfiguration("camera_width"),
            " camera_height:=", LaunchConfiguration("camera_height"),
            " camera_rate:=", LaunchConfiguration("camera_rate"),
            " camera_mount_pitch:=", LaunchConfiguration("camera_mount_pitch"),
        ]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])),
        launch_arguments={
            # -r: start running; -s: server only when gui:=false
            # -s --headless-rendering: server only, cameras render through EGL without a window
            "gz_args": [LaunchConfiguration("world"), " -r -v ", LaunchConfiguration("gz_verbosity"),
                        PythonExpression(["'' if '", gui, "' == 'true' else ' -s --headless-rendering'"])],
            "on_exit_shutdown": "true",
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[{"robot_description": robot_description}, use_sim_time],
    )

    spawn_robot = Node(
        package="ros_gz_sim", executable="create", output="screen",
        arguments=["-topic", "robot_description", "-name", "aibot", "-allow_renaming", "false"],
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        parameters=[{"config_file": os.path.join(gz_share, "config", "gz_bridge.yaml")}, use_sim_time],
    )
    # DetachableJoint topics, one set per (hand, object): attach/detach ROS->gz, state gz->ROS.
    # Object poses: each object's PosePublisher (/model/<obj>/pose, named Pose_V) is bridged and
    # remapped onto the single ROS topic /aibot/object_poses (tf2_msgs/TFMessage).
    grasp_topics = []
    for side in SIDES:
        for obj in GRASP_OBJECTS:
            base = f"/aibot/grasp/{side}/{obj}"
            grasp_topics += [f"{base}/attach@std_msgs/msg/Empty]gz.msgs.Empty",
                             f"{base}/detach@std_msgs/msg/Empty]gz.msgs.Empty",
                             f"{base}/state@std_msgs/msg/String[gz.msgs.StringMsg"]
    grasp_topics += [f"/model/{obj}/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V" for obj in GRASP_OBJECTS + CONTAINERS]
    # teleporting objects for /aibot/scene/reset (scene_manager_node)
    grasp_topics.append("/world/pick_table/set_pose@ros_gz_interfaces/srv/SetEntityPose")
    grasp_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="grasp_bridge", output="screen",
        arguments=grasp_topics, parameters=[use_sim_time],
        remappings=[(f"/model/{obj}/pose", "/aibot/object_poses") for obj in GRASP_OBJECTS + CONTAINERS],
        condition=IfCondition(grasping),
    )

    spawner_args = ["--controller-manager", "/controller_manager", "--controller-manager-timeout", "60"]
    jsb_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["joint_state_broadcaster", *spawner_args],
    )
    group_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=[*GROUP_CONTROLLERS, *spawner_args],
    )

    rviz = Node(
        package="rviz2", executable="rviz2", output="screen", condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", os.path.join(desc_share, "rviz", "view_robot.rviz")], parameters=[use_sim_time],
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=os.path.join(gz_share, "worlds", "pick_table.sdf")),
        DeclareLaunchArgument("gui", default_value="true", description="Gazebo GUI (false: server only)"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("simple_collision", default_value="true"),
        DeclareLaunchArgument("gz_verbosity", default_value="3"),
        DeclareLaunchArgument("head_camera", default_value="true", description="head RGB-D camera sensor"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument("camera_rate", default_value="10"),
        DeclareLaunchArgument("camera_mount_pitch", default_value="0.0", description="extra camera tilt (rad, + = down)"),
        DeclareLaunchArgument("grasping", default_value="false",
                              description="load DetachableJoint grasp plugins (needs grasp_manager_node)"),

        # model:// lookups: our object models, and package://aibot_description/... meshes, which
        # Gazebo resolves by searching for an "aibot_description" folder on this path.
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(gz_share, "models")),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.dirname(desc_share)),

        gazebo,
        robot_state_publisher,
        bridge,
        grasp_bridge,
        spawn_robot,
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[jsb_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[group_spawner])),
        rviz,
    ])
