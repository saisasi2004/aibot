#!/usr/bin/env python3
"""Phase-3 acceptance check: /aibot/scene/reset yields stable, non-penetrating, on-table layouts.

    ros2 launch aibot_bringup teleop.launch.py webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false
    python3 scripts/check_scene_reset.py --seeds 20 --count 5

Ground-truth poses (/aibot/object_poses) are used here for EVALUATION only. Per seed:
settle drift over 2 s < 2 mm, resting height within 3 mm of the model's (no penetration, not
lying on its side), tilt < 5 deg, inside the table, parked slots off the table.
"""
from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from aibot_ai.geometry.layout import resting_height
from aibot_msgs.msg import GraspStatus
from aibot_msgs.srv import ResetScene


class Checker(Node):
    def __init__(self) -> None:
        super().__init__("scene_reset_checker")
        self.poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.lock = threading.Lock()
        self.ready = False
        self.create_subscription(TFMessage, "/aibot/object_poses", self._on_poses, 50)
        self.create_subscription(GraspStatus, "/aibot/grasp/status", lambda m: setattr(self, "ready", m.ready), 5)
        self.reset = self.create_client(ResetScene, "/aibot/scene/reset")

    def _on_poses(self, m: TFMessage) -> None:
        with self.lock:
            for t in m.transforms:
                p, q = t.transform.translation, t.transform.rotation
                self.poses[t.child_frame_id.split("::")[-1]] = (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))

    def snapshot(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        with self.lock:
            return {k: (v[0].copy(), v[1].copy()) for k, v in self.poses.items()}


def tilt_deg(q: np.ndarray) -> float:
    x, y, z, w = q
    zz = 1 - 2 * (x * x + y * y)               # world-z component of the body z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()
    cfg = yaml.safe_load((Path(get_package_share_directory("aibot_gazebo")) / "config" / "objects.yaml").read_text())
    models = {**cfg["slots"]["graspable"], **cfg["slots"]["containers"]}
    t = cfg["table"]
    rclpy.init()
    n = Checker()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not n.reset.wait_for_service(timeout_sec=60):
        raise SystemExit("/aibot/scene/reset not available")
    t_wait = time.monotonic()
    while not n.ready and time.monotonic() - t_wait < 60:
        time.sleep(0.2)
    print(f"grasp manager ready: {n.ready} (all {2 * len(cfg['slots']['graspable'])} hand/object pairs detached)")
    worst = {"drift": 0.0, "dz": 0.0, "tilt": 0.0}
    failures = 0
    for seed in range(args.seeds):
        fut = n.reset.call_async(ResetScene.Request(seed=seed, count=args.count))
        t_call = time.monotonic()
        while not fut.done() and time.monotonic() - t_call < 60:
            time.sleep(0.02)
        if not fut.done():
            print(f"seed {seed:2d}: reset TIMED OUT (60 s)")
            failures += 1
            continue
        r = fut.result()
        if not r.success:
            print(f"seed {seed:2d}: reset FAILED: {r.message}")
            failures += 1
            continue
        time.sleep(0.5)
        a = n.snapshot()
        time.sleep(2.0)
        b = n.snapshot()
        problems = []
        for slot in models:
            if slot not in a or slot not in b:
                problems.append(f"{slot}: no pose")
                continue
            pa, pb = a[slot][0], b[slot][0]
            drift = float(np.linalg.norm(pb - pa))
            if slot in r.placed:
                dz = abs(pb[2] - (t["top_z"] + resting_height(cfg["models"][models[slot]])))
                tilt = tilt_deg(b[slot][1])
                worst["drift"], worst["dz"], worst["tilt"] = max(worst["drift"], drift), max(worst["dz"], dz), max(worst["tilt"], tilt)
                on_table = t["x"][0] <= pb[0] <= t["x"][1] and t["y"][0] <= pb[1] <= t["y"][1]
                if drift > 0.002 or dz > 0.003 or tilt > 5.0 or not on_table:
                    problems.append(f"{slot}: drift {drift * 1000:.1f} mm, dz {dz * 1000:.1f} mm, tilt {tilt:.1f} deg, on table {on_table}")
            elif pb[2] > 0.5:
                problems.append(f"{slot}: parked slot still on the table")
        failures += bool(problems)
        print(f"seed {seed:2d}: {'OK ' if not problems else 'BAD'} placed {', '.join(r.placed)}"
              + ("" if not problems else "\n    " + "\n    ".join(problems)))
    print(f"\nSUMMARY: {args.seeds - failures}/{args.seeds} seeds stable; worst over placed objects: drift "
          f"{worst['drift'] * 1000:.2f} mm (< 2), resting-height error {worst['dz'] * 1000:.2f} mm (< 3), "
          f"tilt {worst['tilt']:.2f} deg (< 5)")
    print("PASS" if failures == 0 and n.ready else "FAIL")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
