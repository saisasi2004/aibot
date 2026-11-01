#!/usr/bin/env python3
"""Physical AI evaluation in the real simulation: natural-language commands -> planner (local LLM)
-> skills, scored against simulator ground truth. Writes a CSV and prints a summary table.

    ros2 launch aibot_bringup teleop.launch.py mode:=ai llm_backend:=ollama webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false
    python3 scripts/eval_ai.py --csv /tmp/eval_ai.csv               # all commands
    python3 scripts/eval_ai.py --only red_in_bin,clarify_cube

End-to-end demo (Phase 7): mimic (synthetic operator) -> AI -> "pick the red cube and put it in
the bin" -> mimic, checking command continuity at both switches. Launch in mimic mode for it:

    ros2 launch aibot_bringup teleop.launch.py webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false calibrate:=false
    python3 scripts/eval_ai.py --e2e

Each command runs on a fresh seeded scene with chosen object types (so the words refer to
something). The robot only uses its own perception; ground truth (/aibot/object_poses) is used
here only to judge the outcome. Per command: outcome OK, answer kind, LLM time, time to first
motion (request -> first skill sent), total time, replans, the executed skills.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from aibot_ai.geometry.place import regions
from aibot_msgs.srv import ResetScene

CFG = yaml.safe_load((Path(get_package_share_directory("aibot_gazebo")) / "config" / "objects.yaml").read_text())
SLOT_MODEL = {**CFG["slots"]["graspable"], **CFG["slots"]["containers"]}
MODELS, TABLE = CFG["models"], CFG["table"]


def height(slot: str) -> float:
    m = MODELS[SLOT_MODEL[slot]]
    return m["size"][2] if m["shape"] in ("box", "bin", "tray") else m.get("length", 2 * m.get("radius", 0.0))


def half_width(slot: str) -> float:
    m = MODELS[SLOT_MODEL[slot]]
    return 0.5 * max(m["size"][:2]) if m["shape"] in ("box", "bin", "tray") else m["radius"]


class GT:
    """Ground-truth view used to judge outcomes (never given to the robot)."""

    def __init__(self, poses: dict[str, np.ndarray], start: dict[str, np.ndarray], placed: list[str]) -> None:
        self.poses, self.start, self.placed = poses, start, placed

    def slot(self, name: str) -> str:
        """A model name ("cube_small" has two slots) or slot name -> the slot on the table."""
        return next((s for s in self.placed if s == name), None) or next(s for s in self.placed if SLOT_MODEL[s] == name)

    def c(self, slot: str) -> np.ndarray:
        slot = self.slot(slot)
        g = self.poses[slot].copy()
        if MODELS[SLOT_MODEL[slot]]["shape"] in ("bin", "tray"):
            g[2] += height(slot) / 2
        return g

    def on_table(self, slot: str) -> bool:
        slot = self.slot(slot)
        return abs(self.c(slot)[2] - (TABLE["top_z"] + height(slot) / 2)) < 0.02

    def inside(self, slot: str, cont: str) -> bool:
        slot, cont = self.slot(slot), self.slot(cont)
        p, b = self.c(slot), self.c(cont)
        inner = 0.5 * MODELS[SLOT_MODEL[cont]]["size"][0] - MODELS[SLOT_MODEL[cont]].get("wall", 0.006)
        return bool(np.max(np.abs(p[:2] - b[:2])) < inner and p[2] - height(slot) / 2 < b[2] + height(cont) / 2 - 0.01)

    def on(self, slot: str, ref: str) -> bool:
        slot, ref = self.slot(slot), self.slot(ref)
        p, b = self.c(slot), self.c(ref)
        return bool(np.linalg.norm(p[:2] - b[:2]) < half_width(ref) and abs(p[2] - (b[2] + height(ref) / 2 + height(slot) / 2)) < 0.02)

    def left_of(self, slot: str, ref: str) -> bool:
        slot, ref = self.slot(slot), self.slot(ref)
        return bool(self.c(slot)[1] - self.c(ref)[1] > half_width(ref) + half_width(slot) - 0.005 and self.on_table(slot))

    def at_region(self, slot: str, region: str) -> bool:
        slot = self.slot(slot)
        return bool(math.dist(self.c(slot)[:2], regions(TABLE)[region]) < 0.10 and self.on_table(slot))

    def moved(self) -> list[str]:
        return [s for s in self.placed if np.linalg.norm(self.poses[s] - self.start[s]) > 0.02]


@dataclass
class Case:
    name: str
    turns: list[str]
    types: list[str]
    containers: list[str]
    check: Callable[[GT, dict], bool]
    note: str = ""
    seed: int = 0


def unmoved_kind(*kinds: str) -> Callable[[GT, dict], bool]:
    return lambda g, rep: not g.moved() and rep.get("kind") in kinds


CASES = [
    Case("red_in_bin", ["pick the red cube and put it in the bin"], ["cube_small", "cube_blue", "ball"], ["bin"],
         lambda g, r: g.inside("cube_small", "bin")),
    Case("blue_in_bin", ["put the blue cube in the bin"], ["cube_blue", "mug", "cube_yellow"], ["bin"],
         lambda g, r: g.inside("cube_blue", "bin")),
    Case("ball_in_tray", ["put the ball in the tray"], ["ball", "cube_small"], ["tray"],
         lambda g, r: g.inside("ball", "tray")),
    Case("blue_on_yellow", ["put the blue cube on the yellow cube"], ["cube_blue", "cube_yellow", "ball"], ["bin"],
         lambda g, r: g.on("cube_blue", "cube_yellow")),
    Case("ball_left_of_mug", ["put the ball to the left of the mug"], ["ball", "mug", "cube_blue"], ["bin"],
         lambda g, r: g.left_of("ball", "mug")),
    Case("box_front_right", ["move the green box to the front right of the table"], ["box_small", "cube_small"], ["bin"],
         lambda g, r: g.at_region("box_small", "front_right")),
    Case("both_cubes", ["put both cubes in the bin"], ["cube_small", "cube_blue", "ball"], ["bin"],
         lambda g, r: g.inside("cube_small", "bin") and g.inside("cube_blue", "bin")),
    Case("sequence", ["put the ball in the bin, then put the red cube in the bin"], ["ball", "cube_small", "mug"], ["bin"],
         lambda g, r: g.inside("ball", "bin") and g.inside("cube_small", "bin")),
    Case("clarify_cube", ["pick up the cube"], ["cube_small", "cube_blue", "ball"], ["bin"], unmoved_kind("clarify")),
    Case("impossible_banana", ["put the banana in the bin"], ["cube_small", "ball"], ["bin"],
         unmoved_kind("impossible", "error")),
    Case("question", ["what do you see on the table?"], ["cube_small", "mug"], ["bin"], unmoved_kind("answer")),
    Case("can_in_bin", ["put the silver can in the bin"], ["cylinder_can", "cube_blue"], ["bin"],
         lambda g, r: g.inside("cylinder_can", "bin")),
]


class Eval(Node):
    def __init__(self) -> None:
        super().__init__("ai_eval")
        self.gt: dict[str, np.ndarray] = {}
        self.reports: list[dict] = []
        self.robot: list[str] = []
        self.create_subscription(TFMessage, "/aibot/object_poses", self._on_gt, 50)
        self.create_subscription(String, "/aibot/planner/status", self._on_status, 20)
        self.create_subscription(String, "/aibot/chat/robot", lambda m: self.robot.append(m.data), 50)
        self.chat = self.create_publisher(String, "/aibot/chat/user", 10)
        self.reset = self.create_client(ResetScene, "/aibot/scene/reset")

    def _on_gt(self, m: TFMessage) -> None:
        for t in m.transforms:
            p = t.transform.translation
            self.gt[t.child_frame_id.split("::")[-1]] = np.array([p.x, p.y, p.z])

    def _on_status(self, m: String) -> None:
        st = json.loads(m.data)
        if st.get("state") == "idle" and st.get("report"):
            self.reports.append(st["report"])

    def call(self, client, req, timeout=60.0):
        fut = client.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        return fut.result() if fut.done() else None

    def ask(self, text: str, timeout: float) -> dict | None:
        n0 = len(self.reports)
        self.chat.publish(String(data=text))
        t0 = time.monotonic()
        while len(self.reports) == n0 and time.monotonic() - t0 < timeout:
            time.sleep(0.1)
        return self.reports[-1] if len(self.reports) > n0 else None


def run_case(n: Eval, case: Case, timeout: float) -> dict:
    r = n.call(n.reset, ResetScene.Request(seed=case.seed, count=len(case.types), types=case.types,
                                           containers=case.containers))
    if r is None or not r.success:
        return {"case": case.name, "ok": False, "kind": "reset_failed", "reply": r.message if r else "timeout"}
    time.sleep(1.0)
    start = {s: n.gt[s].copy() for s in r.placed}
    t0 = time.monotonic()
    reps = []
    for turn in case.turns:
        rep = n.ask(turn, timeout)
        if rep is None:
            break
        reps.append(rep)
    wall = time.monotonic() - t0
    time.sleep(1.5)
    rep = reps[-1] if reps else {}
    g = GT(dict(n.gt), start, list(r.placed))
    ok = bool(reps) and len(reps) == len(case.turns) and case.check(g, rep)
    return {"case": case.name, "command": " / ".join(case.turns), "ok": ok, "kind": rep.get("kind", "timeout"),
            "claimed_success": rep.get("success"), "llm_calls": sum(x.get("llm_calls", 0) for x in reps),
            "llm_s": round(sum(x.get("llm_s", 0.0) for x in reps), 1), "repairs": sum(x.get("repairs", 0) for x in reps),
            "replans": sum(x.get("replans", 0) for x in reps),
            "first_motion_s": None if rep.get("first_motion_s") is None else round(rep["first_motion_s"], 1),
            "total_s": round(wall, 1), "executed": " ".join(f"{s}:{c}" for x in reps for s, c in x.get("executed", [])),
            "moved": " ".join(g.moved()), "reply": rep.get("reply", "")}


def e2e(timeout: float) -> int:
    """mimic -> AI task -> mimic, with the Phase-1 continuity check at both switches."""
    sys.path.insert(0, str(Path(__file__).parent))
    from check_mode_switch import Checker      # synthetic operator + command-step logger
    from aibot_msgs.srv import ResetScene as RS
    chk = Checker("/tmp/eval_ai_e2e_commands.csv")
    ev = Eval()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(chk)
    ex.add_node(ev)
    threading.Thread(target=ex.spin, daemon=True).start()
    chk.set_mode.wait_for_service(timeout_sec=90)
    ev.reset.wait_for_service(timeout_sec=90)
    t_wait = time.monotonic()
    while (chk.mode != "mimic" or not chk.js) and time.monotonic() - t_wait < 90:
        time.sleep(0.2)
    print(f"mode {chk.mode}; mimic tracking the synthetic operator for 8 s")
    r = ev.call(ev.reset, RS.Request(seed=4242, count=3, types=["cube_small", "cube_blue", "ball"], containers=["bin"]))
    print("scene:", r.message if r else "reset timeout")
    time.sleep(8.0)
    t_a = chk.now()
    ok_a, msg_a, d_a = chk.call_mode("ai")
    print(f"-> ai: {ok_a} in {d_a:.2f} s ({msg_a})")
    start = {s: ev.gt[s].copy() for s in r.placed}
    rep = ev.ask("pick the red cube and put it in the bin", timeout) if ok_a else None
    time.sleep(1.5)
    g = GT(dict(ev.gt), start, list(r.placed))
    task_ok = bool(rep) and g.inside("cube_small", "bin")
    print(f"task: {rep and rep.get('kind')} | {rep and rep.get('reply')} | red cube in bin (GT): {task_ok}")
    t_m = chk.now()
    ok_m, msg_m, d_m = chk.call_mode("mimic")
    time.sleep(1.0)
    held = chk.held
    time.sleep(6.0)
    j_ai, j_mimic = chk.max_jump(t_a, t_a + d_a + 1.0), chk.max_jump(t_m, t_m + d_m + 2.0)
    j_task = chk.max_jump(t_a + d_a + 1.0, t_m)
    print(f"-> mimic: {ok_m} in {d_m:.2f} s; held after: {held}")
    print(f"max consecutive-command step: switch to AI {j_ai[0]:.4f} ({j_ai[1]}), during the AI task {j_task[0]:.4f} "
          f"({j_task[1]}), switch to mimic {j_mimic[0]:.4f} ({j_mimic[1]})  [budget 0.05 rad]")
    if rep:
        print(f"LLM {rep.get('llm_s', 0):.1f} s, first motion {rep.get('first_motion_s')}, total {rep.get('total_s', 0):.1f} s")
    passed = ok_a and ok_m and task_ok and max(j_ai[0], j_mimic[0], j_task[0]) <= 0.05 and not any(held)
    print("E2E PASS" if passed else "E2E FAIL")
    return 0 if passed else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/eval_ai.csv")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--timeout", type=float, default=420.0, help="s per command")
    ap.add_argument("--e2e", action="store_true")
    args = ap.parse_args()
    rclpy.init()
    if args.e2e:
        code = e2e(args.timeout)
        rclpy.try_shutdown()
        sys.exit(code)
    n = Eval()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not n.reset.wait_for_service(timeout_sec=90):
        raise SystemExit("scene reset not available (start the bringup with mode:=ai)")
    time.sleep(2.0)
    cases = [c for c in CASES if not args.only or c.name in args.only.split(",")]
    rows = []
    for i, case in enumerate(cases):
        case.seed = args.seed0 + i
        row = run_case(n, case, args.timeout)
        rows.append(row)
        print(f"{'OK ' if row['ok'] else 'BAD'} {row['case']:18s} {row['kind']:10s} llm {row.get('llm_s', 0):5.1f}s "
              f"first-motion {row.get('first_motion_s')}s total {row.get('total_s', 0):5.1f}s | {row.get('executed', '')[:80]} "
              f"| {row['reply'][:80]}", flush=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=max((list(r) for r in rows), key=len), restval="")
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["ok"] for r in rows)
    fm = [r["first_motion_s"] for r in rows if r.get("first_motion_s") is not None]
    print(f"\n{'case':18s} {'ok':3s} {'kind':10s} {'llm s':>6s} {'1st motion s':>12s} {'total s':>8s}")
    for r in rows:
        print(f"{r['case']:18s} {'yes' if r['ok'] else 'no':3s} {r['kind']:10s} {r.get('llm_s', 0):6.1f} "
              f"{str(r.get('first_motion_s', '-')):>12s} {r.get('total_s', 0):8.1f}")
    print(f"\n{ok}/{len(rows)} commands achieved their goal (ground truth); "
          f"time to first motion mean {np.mean(fm) if fm else float('nan'):.1f} s (n={len(fm)}), "
          f"LLM time mean {np.mean([r.get('llm_s', 0) for r in rows]):.1f} s; csv: {args.csv}")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
