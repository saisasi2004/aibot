#!/usr/bin/env python3
"""Real model benchmark, cloud and local: chat commands and simulated head-camera images.

    python3 scripts/bench_models.py --capture 5 --images DIR      # with a running mode:=ai bringup
    python3 scripts/bench_models.py --images DIR --csv out.csv    # no ROS graph needed

Commands run through the real planner prompt and schema on a fake world (the same scene as the
regression test), so plans can be judged without moving a robot. Images come only from
/head_camera/image, with the objects on the table recorded from the simulator's ground truth for
scoring (this script is evaluation code). Prints time to first byte, total, schema validity and
correctness per model. The key comes from OPENROUTER_API_KEY / .env and is never printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "aibot_ai" / "test"))

from fake_world import FakeSkills, scene, semantic_scene  # noqa: E402

from aibot_ai.backends.base import HEAD_CAMERA, BackendError  # noqa: E402
from aibot_ai.backends.ollama import OllamaBackend  # noqa: E402
from aibot_ai.backends.openrouter import OpenRouterBackend  # noqa: E402
from aibot_ai.backends.schema_check import schema_errors  # noqa: E402
from aibot_ai.backends.secrets import key_status, load_dotenv  # noqa: E402
from aibot_ai.perception.semantics import BOXES_PROMPT, BOXES_SCHEMA, box_agreement  # noqa: E402
from aibot_ai.planning.plan import PLAN_SCHEMA, parse_plan, validate  # noqa: E402
from aibot_ai.planning.prompt import build_messages  # noqa: E402

GESTURES = {"wave", "nod", "shake_head"}


def skills(plan) -> list[str]:
    return [s.skill for s in plan.steps]


def picks(plan) -> list[str]:
    return [s.object for s in plan.steps if s.skill == "pick"]


# (command, expected-result check on the parsed plan)
COMMANDS = [
    ("HI", lambda p: bool(GESTURES & set(skills(p)))),
    ("good morning robot", lambda p: bool(GESTURES & set(skills(p)))),
    ("put the blue cube in the bin", lambda p: picks(p) == ["blue_cube"] and p.steps[-1].target == "silver_bin"),
    ("stack the blue cube on the yellow cube", lambda p: picks(p) == ["blue_cube"] and p.steps[-1].relation == "on"),
    ("put the green box at the back right corner", lambda p: p.steps[-1].region == "back_right"),
    ("put both red cubes in the bin", lambda p: sorted(picks(p)) == ["red_cube", "red_cube_2"]),
    ("pick up the cube", lambda p: p.kind == "clarify"),
    ("put the red cube in the bin", lambda p: p.kind == "clarify"),
    ("put the banana in the bin", lambda p: p.kind == "impossible"),
    ("what objects do you see on the table?", lambda p: p.kind == "answer"),
]
# commands that need the camera model's names: geometry alone says block / box / can
SEMANTIC = [
    ("put the screwdriver in the bin", lambda p: picks(p) == ["purple_block"]),
    ("put the tool in the bin", lambda p: picks(p) == ["purple_block"]),
    ("put the soda in the bin", lambda p: picks(p) == ["silver_can"]),
    ("put the coffee in the bin", lambda p: picks(p) == ["white_mug"]),
    ("move the juice to the back left", lambda p: picks(p) == ["green_box"] and p.steps[-1].region == "back_left"),
    ("put the toy in the bin", lambda p: picks(p) == ["red_cube"]),
    ("put the tennis ball next to the coffee mug", lambda p: picks(p) == ["orange_ball"]),
    ("put the drink carton in the bin", lambda p: picks(p) == ["green_box"]),
    ("put the sports equipment in the bin", lambda p: picks(p) == ["orange_ball"]),
    ("put the hammer in the bin", lambda p: p.kind == "impossible"),
]
VERIFY_SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}}, "required": ["answer"]}
DESCRIBE_SCHEMA = {"type": "object", "properties": {"objects": {"type": "array", "items": {"type": "string"}}},
                   "required": ["objects"]}
COLOR = {"cube_small": "red", "cube_small_2": "red", "cube_blue": "blue", "cube_yellow": "yellow", "box_small": "green",
         "block_flat": "purple", "cylinder_can": "silver", "mug": "white", "ball": "orange"}


def capture(n: int, out: Path) -> None:
    """Head-camera frames of seeded scenes + what is really on the table (needs a running stack)."""
    import threading

    import cv2
    import numpy as np
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image

    from aibot_msgs.action import ExecuteSkill
    from aibot_msgs.msg import DetectedObjectArray
    from aibot_msgs.srv import ResetScene
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("bench_capture")
    latest: dict = {}
    node.create_subscription(Image, "/head_camera/image", lambda m: latest.update(m=m), 2)
    node.create_subscription(DetectedObjectArray, "/aibot/perception/objects", lambda m: latest.update(det=m), 5)
    reset = node.create_client(ResetScene, "/aibot/scene/reset")
    skills_ = ActionClient(node, ExecuteSkill, "/aibot/skills/execute")
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    reset.wait_for_service(timeout_sec=60)
    skills_.wait_for_server(timeout_sec=60)
    fut = skills_.send_goal_async(ExecuteSkill.Goal(skill="home"))
    while not fut.done():
        time.sleep(0.05)
    res = fut.result().get_result_async()
    while not res.done():
        time.sleep(0.05)
    for k in range(n):
        fut = reset.call_async(ResetScene.Request(seed=8000 + k, count=4))
        while not fut.done():
            time.sleep(0.05)
        placed = list(fut.result().placed)
        time.sleep(4.0)
        m = latest["m"]
        boxes = [[int(v) for v in o.box] for o in latest["det"].objects if o.box[2] > o.box[0]] if "det" in latest else []
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        cv2.imwrite(str(out / f"scene{k}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        (out / f"scene{k}.json").write_text(json.dumps({"placed": placed, "boxes": boxes, "size": [m.width, m.height]}))
        print("captured", k, placed, flush=True)
    rclpy.try_shutdown()


def sized(path: Path, px: int) -> bytes:
    import cv2
    img = cv2.imread(str(path))
    s = px / max(img.shape[:2])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def first_byte(b) -> float | None:
    return b.last_stats.get("first_token_s")


def bench_llm(name: str, b, rows: list[dict]) -> None:
    world = FakeSkills(scene()).world()
    for text, check in COMMANDS:
        t0 = time.monotonic()
        row = {"role": "llm", "model": name, "input": text, "first_s": None, "total_s": None, "schema_ok": False, "correct": False, "note": ""}
        try:
            data = b.generate(build_messages(text, world, []), PLAN_SCHEMA)
            row.update(first_s=first_byte(b), schema_ok=not schema_errors(data, PLAN_SCHEMA))
            plan = parse_plan(data)
            errs = validate(plan, world)
            row["correct"] = not errs and bool(check(plan))
            row["note"] = "; ".join(errs)[:90] or f"{plan.kind} {[s.text() for s in plan.steps]}"[:90]
        except (BackendError, ValueError, IndexError) as exc:
            row["note"] = str(exc)[:90]
        row["total_s"] = time.monotonic() - t0
        rows.append(row)
        print(f"  {name:32s} {text[:38]:38s} first {row['first_s'] or 0:5.1f} total {row['total_s']:5.1f} schema {row['schema_ok']!s:5} ok {row['correct']!s:5} {row['note'][:60]}", flush=True)


def bench_vlm(name: str, b, images: Path, px: int, rows: list[dict], verify_only: bool) -> None:
    for jpg in sorted(images.glob("*.jpg")):
        placed = json.loads(jpg.with_suffix(".json").read_text())["placed"]
        colors = {COLOR[s] for s in placed if s in COLOR}
        data = sized(jpg, px)
        tests = []
        present = sorted(colors)[0]
        absent = next(c for c in ("purple", "yellow", "blue", "orange", "green", "red", "white") if c not in colors)
        for color, want in ((present, True), (absent, False)):
            tests.append((f"Is there a {color} object on the table? Answer only with JSON.", VERIFY_SCHEMA,
                          lambda a, w=want: a.get("answer") is w, f"verify {color}={want}"))
        if not verify_only:
            tests.append(("List the objects on the table, as short names with their colours. Answer only with JSON.", DESCRIBE_SCHEMA,
                          lambda a: sum(any(c in o.lower() for o in a.get("objects", [])) for c in colors) >= len(colors) - 1,
                          "describe"))
        for prompt, schema, check, tag in tests:
            t0 = time.monotonic()
            row = {"role": f"vlm{px}", "model": name, "input": f"{jpg.stem} {tag}", "first_s": None, "total_s": None, "schema_ok": False, "correct": False, "note": ""}
            try:
                ans = b.ask(data, prompt, schema, source=HEAD_CAMERA)
                row.update(first_s=first_byte(b), schema_ok=not schema_errors(ans, schema), correct=bool(check(ans)), note=json.dumps(ans)[:70])
            except BackendError as exc:
                row["note"] = str(exc)[:70]
            row["total_s"] = time.monotonic() - t0
            rows.append(row)
            print(f"  {name:32s} {row['input'][:28]:28s} first {row['first_s'] or 0:5.1f} total {row['total_s']:5.1f} schema {row['schema_ok']!s:5} ok {row['correct']!s:5} {row['note']}", flush=True)


def bench_semantic(name: str, b, rows: list[dict]) -> None:
    """The same commands with and without the camera model's names in the world state."""
    for mode in ("with names", "no names"):
        things = semantic_scene()
        if mode == "no names":
            for t in things:
                t.semantic_name, t.description = "", ""
        world = FakeSkills(things).world()
        for text, check in SEMANTIC:
            t0 = time.monotonic()
            row = {"role": f"sem {mode}", "model": name, "input": text, "first_s": None, "total_s": None, "schema_ok": False, "correct": False, "note": ""}
            try:
                data = b.generate(build_messages(text, world, []), PLAN_SCHEMA)
                plan = parse_plan(data)
                errs = validate(plan, world)
                row.update(first_s=first_byte(b), schema_ok=True, correct=not errs and bool(check(plan)),
                           note="; ".join(errs)[:80] or f"{plan.kind} {[s.text() for s in plan.steps]}"[:80])
            except (BackendError, ValueError, IndexError) as exc:
                row["note"] = str(exc)[:80]
            row["total_s"] = time.monotonic() - t0
            rows.append(row)
            print(f"  {name:26s} {mode:10s} {text[:42]:42s} ok {row['correct']!s:5} {row['total_s']:5.1f}s {row['note'][:60]}", flush=True)


def bench_crosscheck(name: str, b, images: Path) -> None:
    """VLM boxes vs the depth detector's, on captured frames. Reported only; never replaces it."""
    scores = []
    for jpg in sorted(images.glob("*.jpg")):
        meta = json.loads(jpg.with_suffix(".json").read_text())
        det = meta.get("boxes")
        if not det:
            continue
        w, h = meta["size"]
        try:
            ans = b.ask(jpg.read_bytes(), BOXES_PROMPT, BOXES_SCHEMA, source=HEAD_CAMERA)
        except BackendError as exc:
            print("  crosscheck failed:", str(exc)[:80])
            continue
        vlm = [(o["box"][0] * w / 1000, o["box"][1] * h / 1000, o["box"][2] * w / 1000, o["box"][3] * h / 1000)
               for o in ans.get("objects", []) if len(o.get("box", [])) == 4]
        scores.append(box_agreement(det, vlm))
        print(f"  {jpg.stem}: detector {len(det)} boxes, VLM {len(vlm)}, agreement {scores[-1]:.0%} ({b.describe_last()})", flush=True)
    if scores:
        print(f"crosscheck {name}: mean agreement {statistics.mean(scores):.0%} over {len(scores)} frames (IoU >= 0.3)")


def summary(rows: list[dict]) -> None:
    print(f"\n{'role':8s} {'model':34s} {'n':>3s} {'first mean':>10s} {'total mean':>10s} {'total p95':>9s} {'schema':>7s} {'correct':>7s}")
    for key in sorted({(r["role"], r["model"]) for r in rows}):
        rs = [r for r in rows if (r["role"], r["model"]) == key]
        tot = sorted(r["total_s"] for r in rs)
        fb = [r["first_s"] for r in rs if r["first_s"]]
        print(f"{key[0]:8s} {key[1]:34s} {len(rs):3d} {statistics.mean(fb) if fb else float('nan'):10.1f} {statistics.mean(tot):10.1f} "
              f"{tot[min(len(tot) - 1, int(0.95 * len(tot)))]:9.1f} {sum(r['schema_ok'] for r in rs):3d}/{len(rs):<3d} {sum(r['correct'] for r in rs):3d}/{len(rs):<3d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("/tmp/aibot_bench_images"))
    ap.add_argument("--capture", type=int, default=0, help="capture N scenes from a running stack first, then exit")
    ap.add_argument("--csv", type=Path, default=Path("/tmp/bench_models.csv"))
    ap.add_argument("--cloud-llm", default="google/gemini-3.5-flash")
    ap.add_argument("--cloud-vlm", default="qwen/qwen3-vl-30b-a3b-instruct")
    ap.add_argument("--local-llm", default="granite4.2:3b")
    ap.add_argument("--local-vlm", nargs="*", default=["moondream"])
    ap.add_argument("--sizes", type=int, nargs="*", default=[448])
    ap.add_argument("--only", choices=("all", "llm", "vlm", "cloud", "local"), default="all")
    ap.add_argument("--semantic", action="store_true", help="only: plans with and without the VLM's names")
    ap.add_argument("--crosscheck", action="store_true", help="only: VLM boxes vs detector boxes on the images")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    print(key_status())
    if args.capture:
        capture(args.capture, args.images)
        return
    rows: list[dict] = []
    if args.crosscheck:
        bench_crosscheck(args.cloud_vlm, OpenRouterBackend(args.cloud_vlm, first_byte_timeout=10, total_timeout=30, max_tokens=500), args.images)
        return
    if args.semantic:
        bench_semantic(args.cloud_llm, OpenRouterBackend(args.cloud_llm, reasoning_effort="minimal"), rows)
        bench_semantic(args.local_llm, OllamaBackend(args.local_llm, num_predict=384, temperature=0.0, think=False), rows)
        summary(rows)
        return
    cloud, local = args.only in ("all", "cloud", "llm", "vlm"), args.only in ("all", "local", "llm", "vlm")
    if args.only in ("all", "llm", "cloud", "local"):
        if cloud and args.only != "local":
            bench_llm(args.cloud_llm, OpenRouterBackend(args.cloud_llm, reasoning_effort="minimal"), rows)
        if local and args.only != "cloud":
            bench_llm(args.local_llm, OllamaBackend(args.local_llm, num_predict=384, temperature=0.0, think=False), rows)
    if args.only in ("all", "vlm", "cloud", "local"):
        for px in args.sizes:
            if cloud and args.only != "local":
                bench_vlm(args.cloud_vlm, OpenRouterBackend(args.cloud_vlm, first_byte_timeout=10, total_timeout=30, max_tokens=300), args.images, px, rows, False)
            if local and args.only != "cloud":
                for model in args.local_vlm:
                    bench_vlm(model, OllamaBackend(model, num_predict=120, temperature=0.0, timeout=180), args.images, px, rows, False)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows)
    print("csv:", args.csv)


if __name__ == "__main__":
    main()
