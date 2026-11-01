from glob import glob

from setuptools import find_packages, setup

package_name = "aibot_ai"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        # YOLO-World weights (downloaded by scripts/setup_ai.sh; optional detector)
        ("share/" + package_name + "/models", glob("models/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Gampa Sai Sasivardhan",
    maintainer_email="gampassv2004@gmail.com",
    description="Physical AI mode for aibot: mode manager, perception, skills, LLM planner.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mode_manager_node = aibot_ai.mode_manager_node:main",
            "skill_server_node = aibot_ai.skill_server_node:main",
            "scene_manager_node = aibot_ai.scene_manager_node:main",
            "perception_node = aibot_ai.perception_node:main",
            "planner_node = aibot_ai.planner_node:main",
            "chat = aibot_ai.chat_cli:main",
        ],
    },
)
