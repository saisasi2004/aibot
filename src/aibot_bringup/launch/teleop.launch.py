"""Webcam teleoperation of the STAR1 upper body in Gazebo: one command for everything.

    ros2 launch aibot_bringup teleop.launch.py
    ros2 launch aibot_bringup teleop.launch.py mirror:=true use_ik:=false camera_id:=0

Arguments
    mode         mimic  mimic (webcam drives the robot) | ai (Physical AI: skills, perception, planner)
    detector     tabletop  perception: tabletop (depth) | yolo_world | color_blob
    llm_backend  hybrid  models: hybrid (OpenRouter, then local Ollama) | openrouter | ollama (see aibot_ai/config/ai.yaml)
    mirror       false  true: the robot moves like your mirror image (your right arm -> its left)
    use_ik       false  true: wrist-position IK with elbow hint instead of direct angle mapping
    camera_id    0      /dev/video<N>
    pose_model   full   full | lite (lite is a few ms faster, less accurate in depth)
    calibrate    true   hold a T-pose for 3 s before the robot starts following you
    gazebo_gui   true   Gazebo window (false: headless server)
    rviz         true   RViz with robot, grasp zones and webcam debug image
    camera_gui   true   PyQt5 camera window with status and controls
    webcam       true   false: no webcam_node (tests feeding synthetic landmarks)

Started: Gazebo + robot + controllers (aibot_gazebo/sim.launch.py), webcam_node (+GUI),
retarget_node, grasp_manager_node, RViz, and the Physical AI side: mode_manager_node (single
owner of /aibot/mode, switches at runtime from the GUI), skill_server_node, scene_manager_node,
perception_node and planner_node (chat: GUI panel in AI mode, or `ros2 run aibot_ai chat`).
"""
import importlib.util
import os
from pathlib import Path

from aibot_ai.backends.secrets import load_dotenv
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _grasp_objects(gz_share: str) -> list[str]:
    """The graspable model list is defined once, in aibot_gazebo/launch/sim.launch.py."""
    spec = importlib.util.spec_from_file_location("aibot_sim_launch", os.path.join(gz_share, "launch", "sim.launch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.GRASP_OBJECTS)


def generate_launch_description() -> LaunchDescription:
    # OPENROUTER_API_KEY from the environment, or the git-ignored .env at the workspace root
    # (the launch file is symlinked from src/, so that root is found from its real path; the
    # current directory covers non-symlink installs). Children inherit the environment.
    for env_file in (Path(os.path.realpath(__file__)).parents[3] / ".env", Path.cwd() / ".env"):
        if load_dotenv(env_file):
            break
    bringup = get_package_share_directory("aibot_bringup")
    teleop = get_package_share_directory("aibot_teleop")
    gz_share = get_package_share_directory("aibot_gazebo")
    desc = get_package_share_directory("aibot_description")
    params = os.path.join(bringup, "config", "teleop_params.yaml")
    ai_params = os.path.join(get_package_share_directory("aibot_ai"), "config", "ai.yaml")

    def arg(name: str, typ: type) -> ParameterValue:
        return ParameterValue(LaunchConfiguration(name), value_type=typ)

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "sim.launch.py")),
        launch_arguments={"gui": LaunchConfiguration("gazebo_gui"), "rviz": "false", "grasping": "true"}.items(),
    )

    # Kinematics only (mimic IK mode, skill IK); same model as in Gazebo, without the sim tags.
    robot_description = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", os.path.join(desc, "urdf", "aibot.urdf.xacro")]), value_type=str)

    webcam = Node(
        package="aibot_teleop", executable="webcam_node", name="webcam_node", output="screen",
        condition=IfCondition(LaunchConfiguration("webcam")),
        parameters=[params, {"camera_id": arg("camera_id", int), "pose_model": LaunchConfiguration("pose_model"),
                             "gui": arg("camera_gui", bool),
                             # started in Physical AI: the camera stays closed until the switch to mimic
                             "start_enabled": ParameterValue(PythonExpression(["'", LaunchConfiguration("mode"), "' != 'ai'"]), value_type=bool)}],
    )
    retarget = Node(
        package="aibot_teleop", executable="retarget_node", name="retarget_node", output="screen",
        parameters=[os.path.join(teleop, "config", "retarget.yaml"), params,
                    {"mirror": arg("mirror", bool), "use_ik": arg("use_ik", bool),
                     "calibrate_on_startup": arg("calibrate", bool), "robot_description": robot_description,
                     "mode_required": True}],
    )
    mode_manager = Node(
        package="aibot_ai", executable="mode_manager_node", name="mode_manager_node", output="screen",
        parameters=[ai_params, {"initial_mode": LaunchConfiguration("mode")}],
    )
    scene_manager = Node(
        package="aibot_ai", executable="scene_manager_node", name="scene_manager_node", output="screen",
        parameters=[ai_params],
    )
    perception = Node(
        package="aibot_ai", executable="perception_node", name="perception_node", output="screen",
        parameters=[ai_params, {"use_sim_time": True, "detector": LaunchConfiguration("detector"),
                               "vlm_backend": LaunchConfiguration("llm_backend")}],
    )
    skill_server = Node(
        package="aibot_ai", executable="skill_server_node", name="skill_server_node", output="screen",
        parameters=[ai_params, {"robot_description": robot_description}],
    )
    # Wall clock for every node that does not need simulation time: a use_sim_time node handles
    # every /clock message (one per physics step), which kept four idle Python nodes at 100% CPU.
    # Only perception needs it (TF at the camera image's stamp).
    planner = Node(
        package="aibot_ai", executable="planner_node", name="planner_node", output="screen",
        parameters=[ai_params, {"llm_backend": LaunchConfiguration("llm_backend")}],  # key: env / .env
    )
    grasp = Node(
        package="aibot_teleop", executable="grasp_manager_node", name="grasp_manager_node", output="screen",
        parameters=[params, {"objects": _grasp_objects(gz_share)}],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log", condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", os.path.join(bringup, "rviz", "teleop.rviz")], parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="mimic", description="start in mimic or ai mode"),
        DeclareLaunchArgument("detector", default_value="tabletop", description="tabletop | yolo_world | color_blob"),
        DeclareLaunchArgument("llm_backend", default_value="hybrid", description="models: hybrid | openrouter | ollama"),
        DeclareLaunchArgument("mirror", default_value="false", description="robot mirrors you (your right -> its left)"),
        DeclareLaunchArgument("use_ik", default_value="false", description="wrist IK instead of direct angle mapping"),
        DeclareLaunchArgument("camera_id", default_value="0", description="/dev/video<N>"),
        DeclareLaunchArgument("pose_model", default_value="full", description="MediaPipe pose model: full | lite"),
        DeclareLaunchArgument("calibrate", default_value="true", description="T-pose calibration on startup"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("camera_gui", default_value="true"),
        DeclareLaunchArgument("webcam", default_value="true"),
        sim, webcam, retarget, grasp, mode_manager, skill_server, scene_manager, perception, planner, rviz,
    ])
