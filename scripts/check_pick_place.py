#!/usr/bin/env python3
"""Phase-5 acceptance: pick + place over seeded random scenes, scored with simulator ground truth.

    ros2 launch aibot_bringup teleop.launch.py mode:=ai webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false
    python3 scripts/check_pick_place.py --relation in --scenes 30 --csv /tmp/pick_place_in.csv
    python3 scripts/check_pick_place.py --relation on|left_of|at_region --scenes 10

Per scene: reset (seed) -> scan -> pick(a seeded choice among the detected graspable objects)
-> place by --relation:
    in         place_in(the detected bin)
    on         place on another detected flat-topped object
    left_of    place left_of another detected object
    at_region  place at a seeded region (front_left, back_right, ...)
The robot only uses its own perception. Ground truth (/aibot/object_poses) is used ONLY here:
to name which model the robot chose and to judge the final pose:
    in         object centre inside the bin's inner footprint, its bottom >= 1 cm below the rim
    on         object centre over the reference's footprint, resting on its top (+-2 cm)
    left_of    object further +y than the reference by more than both half-widths, on the table
    at_region  object centre within 10 cm of the region point, on the table
Also logged: reason codes, skill durations, and other objects displaced by more than 2 cm.
Pass: success rate >= --target (0.8).
"""
from __future__ import annotations

import argparse
import csv
import math
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
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from aibot_msgs.srv import SetMode

from aibot_ai.geometry.place import regions
from aibot_msgs.action import ExecuteSkill
from aibot_msgs.msg import DetectedObjectArray, PlaceTarget
from aibot_msgs.srv import ResetScene

FLAT = ("cube", "box", "block", "can")
REGIONS = ("front_left", "front_right", "back_left", "back_right", "center", "left", "right", "front", "back")


class Check(Node):
    def __init__(self) -> None:
        super().__init__("pick_place_check")
        self.gt: dict[str, np.ndarray] = {}
        self.det: DetectedObjectArray | None = None
        self.create_subscription(TFMessage, "/aibot/object_poses", self._on_gt, 50)
        self.create_subscription(DetectedObjectArray, "/aibot/perception/objects", lambda m: setattr(self, "det", m), 5)
        self.reset = self.create_client(ResetScene, "/aibot/scene/reset")
        self.skills = ActionClient(self, ExecuteSkill, "/aibot/skills/execute")
        self.stop = self.create_client(Trigger, "/aibot/stop")
        self.set_mode = self.create_client(SetMode, "/aibot/set_mode")
        self.cmd_times: list[float] = []                   # arrival of arm/waist commands
        for c in ("waist_controller", "left_arm_controller", "right_arm_controller"):
            self.create_subscription(Float64MultiArray, f"/{c}/commands",
                                     lambda _m: self.cmd_times.append(time.monotonic()), 50)

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

    def skill(self, goal: ExecuteSkill.Goal, timeout: float = 150.0):
        t0 = time.monotonic()
        gh = self.skills.send_goal_async(goal)
        while not gh.done():
            time.sleep(0.02)
        rf = gh.result().get_result_async()
        while not rf.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        if not rf.done():
            return None, time.monotonic() - t0
        return rf.result().result, time.monotonic() - t0


def stop_test(n: "Check", trials: int, how: str) -> None:
    """Start a pick, interrupt it mid-motion (STOP button service or a switch to mimic), and
    measure from the request until the last arm/waist command (requirement: <= 100 ms)."""
    lat = []
    for k in range(trials):
        n.call(n.set_mode, SetMode.Request(mode="ai"), timeout=30)
        n.call(n.reset, ResetScene.Request(seed=900 + k, count=3))
        n.skill(ExecuteSkill.Goal(skill="scan"))
        objs = [o for o in (n.det.objects if n.det else []) if o.label not in ("bin", "tray")]
        if not objs:
            continue
        gh = n.skills.send_goal_async(ExecuteSkill.Goal(skill="pick", object_id=objs[0].id))
        time.sleep(4.0)                                    # well inside the approach motion
        moving = len(n.cmd_times) and time.monotonic() - n.cmd_times[-1] < 0.1
        t0 = time.monotonic()
        if how == "stop":
            n.stop.call_async(Trigger.Request())
        else:
            n.set_mode.call_async(SetMode.Request(mode="mimic"))
        time.sleep(1.5)
        after = [t for t in n.cmd_times if t >= t0]
        last = (max(after) - t0) if after else 0.0
        lat.append(last)
        res = gh.result().get_result_async() if gh.done() else None
        while res is not None and not res.done():
            time.sleep(0.05)
        code = res.result().result.reason_code if res is not None else "?"
        print(f"  {how} {k}: was moving {bool(moving)}, last command {last * 1000:.0f} ms after the request, skill -> {code}")
    if lat:
        print(f"{how}: last command after request: max {max(lat) * 1000:.0f} ms, mean {np.mean(lat) * 1000:.0f} ms "
              f"(n={len(lat)}, budget 100 ms) -> {'PASS' if max(lat) <= 0.1 else 'FAIL'}")
    n.call(n.set_mode, SetMode.Request(mode="ai"), timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="in", choices=("in", "on", "left_of", "at_region"))
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--count", type=int, default=4, help="graspable objects per scene")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--csv", default="/tmp/pick_place.csv")
    ap.add_argument("--stop-test", type=int, default=0, metavar="N",
                    help="instead: N interrupted picks each for STOP and for a switch to mimic")
    args = ap.parse_args()
    cfg = yaml.safe_load((Path(get_package_share_directory("aibot_gazebo")) / "config" / "objects.yaml").read_text())
    slot_model = {**cfg["slots"]["graspable"], **cfg["slots"]["containers"]}
    table, models = cfg["table"], cfg["models"]
    region_xy = regions(table)

    def height(slot: str) -> float:
        m = models[slot_model[slot]]
        return m["size"][2] if m["shape"] in ("box", "bin", "tray") else m.get("length", 2 * m.get("radius", 0.0))

    def half_width(slot: str) -> float:
        m = models[slot_model[slot]]
        return 0.5 * max(m["size"][:2]) if m["shape"] in ("box", "bin", "tray") else m["radius"]

    def centre(slot: str) -> np.ndarray:
        """GT model origin -> geometric centre (containers have their origin at the bottom)."""
        g = n.gt[slot].copy()
        if models[slot_model[slot]]["shape"] in ("bin", "tray"):
            g[2] += height(slot) / 2
        return g

    rclpy.init()
    n = Check()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not n.reset.wait_for_service(timeout_sec=90) or not n.skills.wait_for_server(timeout_sec=60):
        raise SystemExit("scene reset / skill server not available (start the bringup with mode:=ai)")
    time.sleep(2.0)
    if args.stop_test:
        for how in ("stop", "mimic"):
            stop_test(n, args.stop_test, how)
        rclpy.try_shutdown()
        return
    rows, ok_count, done = [], 0, 0
    for k in range(args.scenes):
        seed = args.seed0 + k
        rng = np.random.default_rng(seed)
        containers = ["bin"] if args.relation == "in" else []
        r = n.call(n.reset, ResetScene.Request(seed=seed, count=args.count, containers=containers))
        if r is None or not r.success:
            print(f"scene {k:2d}: reset failed ({r.message if r else 'timeout'})")
            continue
        res, t_scan = n.skill(ExecuteSkill.Goal(skill="scan"))
        objs = list(n.det.objects) if n.det else []
        gt0 = {s: centre(s) for s in r.placed}

        def slot_of(o) -> str:
            p = np.array([o.pose.position.x, o.pose.position.y])
            return min(r.placed, key=lambda s: np.linalg.norm(gt0[s][:2] - p))

        if res is None or not res.success:
            rows.append({"scene": k, "seed": seed, "relation": args.relation, "scan": res.reason_code if res else "TIMEOUT",
                         "success": False, "message": f"scan failed: {res.message if res else 'timeout'}"})
            done += 1
            print(f"scene {k:2d}: FAIL scan {rows[-1]['message'][:160]}", flush=True)
            continue
        graspable = [o for o in objs if o.label not in ("bin", "tray") and not o.held_by]
        row = {"scene": k, "seed": seed, "relation": args.relation, "scan": res.reason_code if res else "TIMEOUT",
               "target": "", "target_slot": "", "reference": "", "pick": "", "pick_s": "", "place": "", "place_s": "",
               "success": False, "final_err": "", "displaced": "", "message": ""}
        if not graspable:
            row["message"] = "nothing graspable detected"
            rows.append(row)
            done += 1
            print(f"scene {k:2d}: FAIL nothing graspable detected")
            continue
        tgt = graspable[int(rng.integers(len(graspable)))]
        slot = slot_of(tgt)
        row.update(target=tgt.id, target_slot=slot)
        # choose the reference / place target from what the robot perceives
        ref = ref_slot = None
        if args.relation == "in":
            bins = [o for o in objs if o.label == "bin"]
            ref = bins[0] if bins else None
        elif args.relation in ("on", "left_of"):
            others = [o for o in graspable if o.id != tgt.id and (args.relation != "on" or o.label in FLAT)]
            ref = others[int(rng.integers(len(others)))] if others else None
        region = REGIONS[int(rng.integers(len(REGIONS)))] if args.relation == "at_region" else ""
        if args.relation != "at_region" and ref is None:
            row["message"] = f"no reference for {args.relation} detected"
            rows.append(row)
            done += 1
            print(f"scene {k:2d}: FAIL {row['message']}")
            continue
        if ref is not None:
            ref_slot = slot_of(ref)
            row["reference"] = ref.id
        res, t_pick = n.skill(ExecuteSkill.Goal(skill="pick", object_id=tgt.id))
        row.update(pick=res.reason_code if res else "TIMEOUT", pick_s=f"{t_pick:.1f}")
        if res is None or not res.success:
            row["message"] = res.message if res else "pick timed out"
        else:
            if args.relation == "in":
                goal = ExecuteSkill.Goal(skill="place_in", object_id=ref.id)
            else:
                goal = ExecuteSkill.Goal(skill="place", place_target=PlaceTarget(
                    relation=args.relation, reference_id=ref.id if ref else "", region=region))
            res, t_place = n.skill(goal)
            row.update(place=res.reason_code if res else "TIMEOUT", place_s=f"{t_place:.1f}",
                       message=res.message if res else "place timed out")
            time.sleep(1.5)                                    # let it settle
            p = centre(slot)
            h = height(slot)
            succ, err = False, float("nan")
            if res is not None and res.success:
                if args.relation == "in":
                    b = centre(ref_slot)
                    inner = 0.5 * models[slot_model[ref_slot]]["size"][0] - models[slot_model[ref_slot]].get("wall", 0.006)
                    err = float(np.max(np.abs(p[:2] - b[:2])))
                    succ = err < inner and p[2] - h / 2 < b[2] + height(ref_slot) / 2 - 0.01   # bottom below the rim
                elif args.relation == "on":
                    b = centre(ref_slot)
                    err = float(np.linalg.norm(p[:2] - b[:2]))
                    succ = err < half_width(ref_slot) and abs(p[2] - (b[2] + height(ref_slot) / 2 + h / 2)) < 0.02
                elif args.relation == "left_of":
                    b = centre(ref_slot)
                    err = float(p[1] - b[1])
                    succ = err > half_width(ref_slot) + half_width(slot) - 0.005 and abs(p[2] - (table["top_z"] + h / 2)) < 0.02
                else:
                    err = float(math.dist(p[:2], region_xy[region]))
                    succ = err < 0.10 and abs(p[2] - (table["top_z"] + h / 2)) < 0.02
            row.update(success=bool(succ), final_err=f"{err:.3f}")
            moved = [s for s in r.placed if s not in (slot, ref_slot) and np.linalg.norm(centre(s) - gt0[s]) > 0.02]
            row["displaced"] = " ".join(moved)
            if res is not None and res.success and not succ:
                row["message"] = f"skill said OK but ground truth disagrees ({p.round(3).tolist()}); " + row["message"]
        ok_count += row["success"]
        done += 1
        rows.append(row)
        print(f"scene {k:2d}: {'OK  ' if row['success'] else 'FAIL'} {tgt.id}({slot}) -> {args.relation} "
              f"{row['reference'] or region}  pick {row['pick']} {row['pick_s']}s  place {row['place']} {row['place_s']}s"
              f"{'  displaced: ' + row['displaced'] if row['displaced'] else ''}"
              f"{'' if row['success'] else '  | ' + row['message'][:160]}", flush=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=max((list(r_) for r_ in rows), key=len) if rows else ["scene"], restval="")
        w.writeheader()
        w.writerows(rows)
    codes: dict[str, int] = {}
    for row in rows:
        c = row.get("pick") or "SCAN" if row.get("pick") != "OK" else (row.get("place") or "?")
        if not row["success"]:
            c = c if c not in ("OK", "") else ("GT_MISMATCH" if row.get("place") == "OK" else row["message"][:30])
            codes[c] = codes.get(c, 0) + 1
    picks = [float(r_["pick_s"]) for r_ in rows if r_.get("pick") == "OK"]
    places = [float(r_["place_s"]) for r_ in rows if r_.get("place") == "OK"]
    rate = ok_count / max(done, 1)
    print(f"\n{args.relation}: {ok_count}/{done} succeeded ({rate:.0%}); target {args.target:.0%}")
    print(f"  pick OK {len(picks)}/{done}, mean {np.mean(picks) if picks else float('nan'):.1f} s; "
          f"place OK {len(places)}, mean {np.mean(places) if places else float('nan'):.1f} s")
    print(f"  failures by reason: {codes or 'none'}")
    print(f"  scenes with another object displaced > 2 cm: {sum(1 for r_ in rows if r_.get('displaced'))}")
    print(f"  csv: {args.csv}")
    print("PASS" if rate >= args.target else "FAIL")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
