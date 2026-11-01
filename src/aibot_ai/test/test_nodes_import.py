"""Every node module must at least import (a syntax error in a node is otherwise only found
when the launch file starts it and the process dies)."""
import importlib

import pytest

NODES = ["mode_manager_node", "skill_server_node", "scene_manager_node", "perception_node", "planner_node", "chat_cli"]


@pytest.mark.parametrize("name", NODES)
def test_node_module_imports(name):
    mod = importlib.import_module(f"aibot_ai.{name}")
    assert callable(getattr(mod, "main"))


def test_teleop_modules_import():
    for name in ("camera_gui", "grasp_manager_node", "retarget_node", "webcam_node"):
        importlib.import_module(f"aibot_teleop.{name}")


@pytest.mark.parametrize("name,cls", [("planner_node", "PlannerNode"), ("perception_node", "PerceptionNode")])
def test_node_constructs(name, cls, monkeypatch):
    """Declaring parameters etc. happens in the constructor: an import alone misses those crashes."""
    import rclpy
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    mod = importlib.import_module(f"aibot_ai.{name}")
    rclpy.init()
    try:
        getattr(mod, cls)().destroy_node()
    finally:
        rclpy.try_shutdown()


def test_every_subpackage_is_installed_by_a_plain_build():
    """find_packages skips a directory without __init__.py: it works with --symlink-install, then
    a plain colcon build leaves the node dying on import."""
    import aibot_ai
    from pathlib import Path
    root = Path(aibot_ai.__file__).resolve().parent
    missing = [d.name for d in root.iterdir() if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py")) and not (d / "__init__.py").exists()]
    assert not missing
