"""scene_manager_node: random, reachable tabletop layouts from a fixed set of model slots.

Service  /aibot/scene/reset   aibot_msgs/ResetScene {seed, count, types, containers}

Why fixed slots: the gz DetachableJoint plugins are declared per (hand, model) when the URDF is
generated, so models cannot be spawned later and still be graspable. Every slot in GRASP_OBJECTS
(and every container) always exists in the world; a reset teleports a random subset onto the
table and parks the rest on the floor behind the robot, out of every camera view.

Mechanism: gz-sim's /world/pick_table/set_pose service (UserCommands system), bridged by
ros_gz_bridge as ros_gz_interfaces/srv/SetEntityPose (see aibot_gazebo/launch/sim.launch.py).
Held objects are released first (grasp manager), otherwise a weld would drag them along; the
perception memory is cleared afterwards (it described the previous scene).
"""
from __future__ import annotations

import contextlib
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_srvs.srv import Trigger

from aibot_msgs.srv import ResetScene

from .geometry.layout import Placement, choose_slots, resting_height, sample_layout


class SceneManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_manager_node")
        p = self.declare_parameter
        default_cfg = Path(get_package_share_directory("aibot_gazebo")) / "config" / "objects.yaml"
        self.cfg = yaml.safe_load(Path(p("objects_file", str(default_cfg)).value).read_text())
        self.world: str = p("world", "pick_table").value
        self.default_count: int = p("default_count", 9).value
        self.settle_time: float = p("settle_time", 1.5).value
        self.call_timeout: float = p("call_timeout", 2.0).value
        self.cb = ReentrantCallbackGroup()
        self.set_pose = self.create_client(SetEntityPose, f"/world/{self.world}/set_pose", callback_group=self.cb)
        self.release = self.create_client(Trigger, "/aibot/grasp/release_all", callback_group=self.cb)
        self.clear_perception = self.create_client(Trigger, "/aibot/perception/clear", callback_group=self.cb)
        self.create_service(ResetScene, "/aibot/scene/reset", self._srv_reset, callback_group=self.cb)
        self.get_logger().info(f"scene manager: {len(self.cfg['slots']['graspable'])} object slots, "
                               f"{len(self.cfg['slots']['containers'])} containers")

    def _call(self, client, req):
        if not client.wait_for_service(timeout_sec=self.call_timeout):
            return None
        done = threading.Event()
        fut = client.call_async(req)
        fut.add_done_callback(lambda _f: done.set())
        return fut.result() if done.wait(self.call_timeout) else None

    def _teleport(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        req = SetEntityPose.Request()
        req.entity = Entity(name=name, type=Entity.MODEL)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, z
        req.pose.orientation.z, req.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        res = self._call(self.set_pose, req)
        return bool(res and res.success)

    def plan(self, seed: int, count: int, types: list[str], containers: list[str]) -> tuple[list[Placement], list[str]]:
        """(table placements, parked slots) for a request. Pure apart from the config."""
        rng = np.random.default_rng(seed)
        graspable = self.cfg["slots"]["graspable"]
        conts = self.cfg["slots"]["containers"]
        unknown = [c for c in containers if c not in conts]
        if unknown:
            raise ValueError(f"unknown containers {unknown}; known: {sorted(conts)}")
        chosen = {s: graspable[s] for s in choose_slots(graspable, count, types, rng)}
        chosen |= {c: conts[c] for c in containers}
        placements = sample_layout(chosen, self.cfg["models"], self.cfg["table"], rng)
        parked = [s for s in list(graspable) + list(conts) if s not in chosen]
        return placements, parked

    def _srv_reset(self, req: ResetScene.Request, res: ResetScene.Response) -> ResetScene.Response:
        t0 = time.monotonic()
        try:
            # no count: the full table, tray included; test scripts ask for fewer and get the bin
            containers = list(req.containers) or (["bin"] if req.count else list(self.cfg["slots"]["containers"]))
            placements, parked = self.plan(req.seed, req.count or self.default_count, list(req.types), containers)
        except (ValueError, RuntimeError) as exc:
            res.success, res.message = False, str(exc)
            return res
        self._call(self.release, Trigger.Request())           # never teleport a welded object
        models = {**self.cfg["slots"]["graspable"], **self.cfg["slots"]["containers"]}
        ox, oy = self.cfg["parking"]["origin"]
        failed = []
        for i, slot in enumerate(parked):                     # park first: frees the table
            z = resting_height(self.cfg["models"][models[slot]]) + 0.001
            if not self._teleport(slot, ox - i * self.cfg["parking"]["spacing"], oy, z, 0.0):
                failed.append(slot)
        for pl in placements:
            if not self._teleport(pl.slot, pl.x, pl.y, pl.z, pl.yaw):
                failed.append(pl.slot)
        time.sleep(self.settle_time)
        # a new scene: the robot's object memory describes the old one (ghosts for ~20 s otherwise);
        # perception is optional here (a mimic-only bringup has none), so a short wait
        if self.clear_perception.wait_for_service(timeout_sec=0.3):
            self._call(self.clear_perception, Trigger.Request())
        res.placed = [pl.slot for pl in placements]
        res.success = not failed
        res.message = (f"seed {req.seed}: placed {', '.join(res.placed)} in {time.monotonic() - t0:.1f} s"
                       if not failed else f"set_pose failed for {failed} (is the set_pose bridge running?)")
        self.get_logger().info(res.message)
        return res


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SceneManagerNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(KeyboardInterrupt):
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
