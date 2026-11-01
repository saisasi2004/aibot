#!/usr/bin/env python3
"""Phase-4 evaluation: head-camera perception vs simulator ground truth over randomised scenes.

    ros2 launch aibot_bringup teleop.launch.py mode:=ai webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false [detector:=yolo_world]
    python3 scripts/eval_perception.py --scenes 30 --csv /tmp/perception.csv

Ground truth (/aibot/object_poses) is used ONLY here, to score the detector. Matching: each
ground-truth object to the nearest unused detection within 5 cm (x-y). Reports recall,
precision, 3D centroid error (mean/p95, all and within 0.6 m of the camera; containers x-y
only, their model origin is their bottom), label and colour accuracy, detector/fusion latency.
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from aibot_msgs.action import ExecuteSkill
from aibot_msgs.msg import DetectedObjectArray
from aibot_msgs.srv import ResetScene

LABEL = {"box": None, "cylinder": None, "sphere": "ball", "bin": "bin", "tray": "tray"}


def expected_label(name: str, m: dict) -> str:
    if m["shape"] == "box":
        sx, sy, sz = m["size"]
        if sz < 0.6 * min(sx, sy):
            return "block"
        return "cube" if abs(sx - sy) < 0.2 * max(sx, sy) and 0.7 < sz / ((sx + sy) / 2) < 1.35 else "box"
    if m["shape"] == "cylinder":
        return "can" if m["length"] / (2 * m["radius"]) > 1.5 else "mug"
    return LABEL[m["shape"]]


class Eval(Node):
    def __init__(self) -> None:
        super().__init__("perception_eval")
        self.gt: dict[str, np.ndarray] = {}
        self.det: DetectedObjectArray | None = None
        self.create_subscription(TFMessage, "/aibot/object_poses", self._on_gt, 50)
        self.create_subscription(DetectedObjectArray, "/aibot/perception/objects", lambda m: setattr(self, "det", m), 5)
        self.reset = self.create_client(ResetScene, "/aibot/scene/reset")
        self.detect = self.create_client(Trigger, "/aibot/perception/detect_now")
        self.clear = self.create_client(Trigger, "/aibot/perception/clear")
        self.skills = ActionClient(self, ExecuteSkill, "/aibot/skills/execute")
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)

    def _on_gt(self, m: TFMessage) -> None:
        for t in m.transforms:
            p = t.transform.translation
            self.gt[t.child_frame_id.split("::")[-1]] = np.array([p.x, p.y, p.z])

    def call(self, client, req, timeout=60.0):
        fut = client.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        return fut.result() if fut.done() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--csv", default="/tmp/perception_eval.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load((Path(get_package_share_directory("aibot_gazebo")) / "config" / "objects.yaml").read_text())
    slot_model = {**cfg["slots"]["graspable"], **cfg["slots"]["containers"]}
    rclpy.init()
    n = Eval()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    for c in (n.reset, n.detect, n.clear):
        if not c.wait_for_service(timeout_sec=60):
            raise SystemExit(f"{c.srv_name} not available (start the bringup with mode:=ai)")
    n.skills.wait_for_server(timeout_sec=30)
    gh = n.skills.send_goal_async(ExecuteSkill.Goal(skill="home"))      # head to the scan pose
    while not gh.done():
        time.sleep(0.05)
    rf = gh.result().get_result_async()
    while not rf.done():
        time.sleep(0.05)
    print("home:", rf.result().result.reason_code)
    rows, errs, errs_near, n_gt, n_match, n_det, lab_ok, col_ok, lat_det, lat_fus, wall = [], [], [], 0, 0, 0, 0, 0, [], [], []
    for seed in range(args.scenes):
        r = n.call(n.reset, ResetScene.Request(seed=1000 + seed, count=args.count))
        if r is None or not r.success:
            print(f"scene {seed}: reset failed {r.message if r else 'timeout'}")
            continue
        n.call(n.clear, Trigger.Request())
        for _ in range(3):                                   # a few frames for memory smoothing
            t0 = time.monotonic()
            n.call(n.detect, Trigger.Request(), timeout=10)
            wall.append(time.monotonic() - t0)
        det = n.det
        cam = n.tf.lookup_transform("world", "head_camera_optical_frame", rclpy.time.Time()).transform.translation
        cam = np.array([cam.x, cam.y, cam.z])
        lat_det.append(det.detect_ms)
        lat_fus.append(det.fusion_ms)
        objs = list(det.objects)
        n_det += len(objs)
        used = set()
        for slot in r.placed:
            model = cfg["models"][slot_model[slot]]
            g = n.gt[slot].copy()
            container = model["shape"] in ("bin", "tray")
            if container:
                g[2] += model["size"][2] / 2
            n_gt += 1
            cands = [(np.linalg.norm(np.array([o.pose.position.x, o.pose.position.y]) - g[:2]), i) for i, o in enumerate(objs) if i not in used]
            d, i = min(cands) if cands else (np.inf, -1)
            ok = d < 0.05
            e = lab = col = None
            if ok:
                used.add(i)
                n_match += 1
                o = objs[i]
                p = np.array([o.pose.position.x, o.pose.position.y, o.pose.position.z])
                e = float(np.linalg.norm((p - g)[:2])) if container else float(np.linalg.norm(p - g))
                errs.append(e)
                if np.linalg.norm(g - cam) < 0.6:
                    errs_near.append(e)
                lab = o.label == expected_label(slot, model)
                col = o.color == model["color"] or (model["color"] in ("gray", "silver") and o.color in ("gray", "silver"))
                lab_ok += lab
                col_ok += col
            rows.append([seed, slot, ok, f"{e:.4f}" if e is not None else "", objs[i].label if ok else "", expected_label(slot, model),
                         objs[i].color if ok else "", model["color"], f"{np.linalg.norm(g - cam):.3f}"])
        print(f"scene {seed:2d}: {sum(1 for row in rows if row[0] == seed and row[2])}/{len(r.placed)} found, "
              f"{len(objs)} detections, det {det.detect_ms:.0f} ms fus {det.fusion_ms:.0f} ms")
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "slot", "matched", "pos_err_m", "label", "expected_label", "color", "expected_color", "dist_to_camera_m"])
        w.writerows(rows)
    e, en = np.array(errs), np.array(errs_near)
    print(f"\nDETECTOR {det.detector}: scenes {args.scenes}, objects {n_gt}")
    print(f"  recall {n_match / n_gt:.1%}  precision {n_match / max(n_det, 1):.1%}")
    print(f"  position error: mean {e.mean() * 100:.2f} cm, p95 {np.percentile(e, 95) * 100:.2f} cm (all matched)")
    if len(en):
        print(f"  position error within 0.6 m of camera: mean {en.mean() * 100:.2f} cm, p95 {np.percentile(en, 95) * 100:.2f} cm (n={len(en)})")
    print(f"  label accuracy {lab_ok / max(n_match, 1):.1%}, colour accuracy {col_ok / max(n_match, 1):.1%}")
    print(f"  latency per frame: detector {np.mean(lat_det):.0f} ms (p95 {np.percentile(lat_det, 95):.0f}), "
          f"depth fusion {np.mean(lat_fus):.0f} ms (p95 {np.percentile(lat_fus, 95):.0f}); detect_now round trip "
          f"{np.mean(wall) * 1000:.0f} ms")
    print(f"  csv: {args.csv}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
