from glob import glob

from setuptools import find_packages, setup

package_name = "aibot_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        # MediaPipe models are downloaded by scripts/setup_env.sh; rebuild after downloading.
        ("share/" + package_name + "/models", glob("models/*.task")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Gampa Sai Sasivardhan",
    maintainer_email="gampassv2004@gmail.com",
    description="Webcam (MediaPipe) teleoperation of the STAR1 upper body: landmarks, retargeting, grasping.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "webcam_node = aibot_teleop.webcam_node:main",
            "retarget_node = aibot_teleop.retarget_node:main",
            "grasp_manager_node = aibot_teleop.grasp_manager_node:main",
        ],
    },
)
