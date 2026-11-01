"""Planner regression with the REAL local LLM (Ollama) on a fake world with fake skills.

Slow (one or more CPU LLM calls per command), so opt-in:
    AIBOT_LLM_TESTS=1 [AIBOT_LLM_BACKEND=cloud] python -m pytest -s src/aibot_ai/test/test_planner_llm.py
Each command is judged by the final world state (and the kind of answer), not by the exact plan,
so any correct plan passes. Prints a table and writes a JSON report ($AIBOT_LLM_REPORT, default
/tmp/aibot_planner_llm.json) with plan correctness, success rate and time-to-first-motion.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from fake_world import FakeSkills, scene  # noqa: E402

from aibot_ai.backends.base import BackendError  # noqa: E402
from aibot_ai.backends.ollama import OllamaBackend  # noqa: E402
from aibot_ai.backends.openrouter import OpenRouterBackend  # noqa: E402
from aibot_ai.backends.secrets import load_dotenv  # noqa: E402
from aibot_ai.planning.agent import PlannerAgent, Report  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "ai.yaml").read_text())["planner_node"]["ros__parameters"]
LLM = CFG["llm"]

Check = Callable[[FakeSkills, Report], bool]
NO_MOTION_KINDS = ("impossible", "clarify", "answer", "error", "failed")


def nothing_moved(w: FakeSkills, r: Report) -> bool:
    return not w.moved()


def kind(*kinds: str) -> Check:
    return lambda w, r: r.kind in kinds


def all_of(*checks: Check) -> Check:
    return lambda w, r: all(c(w, r) for c in checks)


def is_in(o: str, c: str) -> Check:
    return lambda w, r: w.is_in(o, c)


def is_on(o: str, ref: str) -> Check:
    return lambda w, r: w.is_on(o, ref)


def side(o: str, ref: str, rel: str) -> Check:
    return lambda w, r: w.side_of(o, ref) == rel and o in w.moved()


def region(o: str, reg: str) -> Check:
    return lambda w, r: w.at_region(o, reg)


def near(o: str, ref: str, d: float = 0.16) -> Check:
    return lambda w, r: o in w.moved() and not w.get(o).held_by and np.hypot(w.get(o).x - w.get(ref).x, w.get(o).y - w.get(ref).y) < d


CUBES = ("red_cube_1", "red_cube_2", "blue_cube_1", "yellow_cube_1")
CASES: list[tuple[str, list[str], Check, dict]] = [
    ("simple_in", ["put the blue cube in the bin"], is_in("blue_cube_1", "silver_bin_1"), {}),
    ("pick_and_put", ["pick up the yellow cube and put it in the bin"], is_in("yellow_cube_1", "silver_bin_1"), {}),
    ("in_tray", ["put the ball in the tray"], is_in("orange_ball_1", "black_tray_1"), {}),
    ("on", ["put the blue cube on the yellow cube"], is_on("blue_cube_1", "yellow_cube_1"), {}),
    ("stack", ["stack the yellow cube on top of the blue cube"], is_on("yellow_cube_1", "blue_cube_1"), {}),
    ("left_of", ["place the mug to the left of the blue cube"], side("white_mug_1", "blue_cube_1", "left_of"), {}),
    ("right_of", ["put the ball to the right of the yellow cube"], side("orange_ball_1", "yellow_cube_1", "right_of"), {}),
    ("behind", ["put the mug behind the blue cube"], side("white_mug_1", "blue_cube_1", "behind"), {}),
    ("in_front_of", ["put the green box in front of the yellow cube"], side("green_box_1", "yellow_cube_1", "in_front_of"), {}),
    ("region_front_left", ["move the yellow cube to the front left of the table"], region("yellow_cube_1", "front_left"), {}),
    ("region_center", ["put the blue cube in the middle of the table"], region("blue_cube_1", "center"), {}),
    ("region_back_right", ["put the green box at the back right corner"], region("green_box_1", "back_right"), {}),
    ("both_red", ["put both red cubes in the bin"], all_of(is_in("red_cube_1", "silver_bin_1"), is_in("red_cube_2", "silver_bin_1")), {}),
    ("all_cubes", ["put all the cubes in the bin"], lambda w, r: all(w.is_in(c, "silver_bin_1") for c in CUBES), {}),
    ("sequence", ["put the blue cube in the bin and then the ball in the tray"],
     all_of(is_in("blue_cube_1", "silver_bin_1"), is_in("orange_ball_1", "black_tray_1")), {}),
    ("two_relations", ["put the mug in the bin, then put the yellow cube on the blue cube"],
     all_of(is_in("white_mug_1", "silver_bin_1"), is_on("yellow_cube_1", "blue_cube_1")), {}),
    ("specific_right", ["put the red cube on the right in the bin"], is_in("red_cube_2", "silver_bin_1"), {}),
    ("leftmost", ["put the leftmost red cube in the tray"], is_in("red_cube_1", "black_tray_1"), {}),
    ("box_in_tray", ["put the green box in the tray"], is_in("green_box_1", "black_tray_1"), {}),
    ("next_to", ["put the yellow cube next to the mug"], near("yellow_cube_1", "white_mug_1"), {}),
    ("move_aside", ["move the mug out of the way"], lambda w, r: "white_mug_1" in w.moved() and not w.get("white_mug_1").held_by, {}),
    ("go_home", ["go back to your home position"], all_of(nothing_moved, kind("plan", "answer")), {}),
    ("greeting", ["HI"], lambda w, r: r.success and any(e[0].split("(")[0] in ("wave", "express") for e in r.executed), {}),
    ("clarify_cube", ["pick up the cube"], all_of(nothing_moved, kind("clarify")), {}),
    ("clarify_red", ["put the red cube in the bin"], all_of(nothing_moved, kind("clarify")), {}),
    ("clarify_then_answer", ["put the red cube in the tray", "the one on the left"],
     is_in("red_cube_1", "black_tray_1"), {}),
    ("impossible_banana", ["put the banana in the bin"], all_of(nothing_moved, kind("impossible", "error")), {}),
    ("impossible_container", ["pick up the bin and give it to me"], all_of(nothing_moved, kind(*NO_MOTION_KINDS)), {}),
    ("impossible_on_ball", ["balance the blue cube on top of the ball"],
     lambda w, r: not w.is_on("blue_cube_1", "orange_ball_1") and r.kind in NO_MOTION_KINDS, {}),
    ("question", ["what objects do you see on the table?"], all_of(nothing_moved, kind("answer")), {}),
    ("unreachable_honest", ["put the blue cube in the bin"],
     lambda w, r: not w.is_in("blue_cube_1", "silver_bin_1") and r.kind in ("failed", "impossible", "error") and not r.success,
     {"unreachable": {"blue_cube_1"}}),
]


def _backend():
    """AIBOT_LLM_BACKEND=local (default) or cloud: the real model either way."""
    if os.environ.get("AIBOT_LLM_BACKEND") == "cloud":
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
        b = OpenRouterBackend(LLM["cloud_model"], first_byte_timeout=LLM["first_byte_timeout"], total_timeout=LLM["total_timeout"],
                              max_tokens=LLM["max_tokens"], temperature=LLM["temperature"], reasoning_effort=LLM["reasoning_effort"])
    else:
        b = OllamaBackend(LLM["model"], LLM["url"], LLM["keep_alive"], LLM["num_ctx"], LLM["num_predict"],
                          LLM["temperature"], LLM["timeout"], LLM.get("think"))
    b.check()
    return b


@pytest.mark.skipif(os.environ.get("AIBOT_LLM_TESTS") != "1", reason="set AIBOT_LLM_TESTS=1 (slow, needs Ollama)")
def test_planner_regression_with_real_llm():
    try:
        llm = _backend()
    except BackendError as exc:
        pytest.skip(str(exc))
    if os.environ.get("AIBOT_LLM_BACKEND") != "cloud":
        llm.generate([{"role": "user", "content": "Reply with {}"}], {"type": "object"})   # load once
    rows = []
    for name, turns, check, fake in CASES:
        skills = FakeSkills(scene(), **fake)
        agent = PlannerAgent(llm, skills, lambda m: None)
        t0 = time.monotonic()
        reports = [agent.handle(t) for t in turns]
        rep = reports[-1]
        ok = bool(check(skills, rep))
        rows.append({"case": name, "turns": turns, "ok": ok, "kind": rep.kind, "success": rep.success,
                     "reply": rep.reply, "plans": [p for r in reports for p in r.plans],
                     "executed": [e for r in reports for e in r.executed],
                     "llm_calls": sum(r.llm_calls for r in reports), "llm_s": round(sum(r.llm_s for r in reports), 2),
                     "repairs": sum(r.repairs for r in reports), "replans": sum(r.replans for r in reports),
                     "first_motion_s": None if rep.first_motion_s is None else round(rep.first_motion_s, 2),
                     "wall_s": round(time.monotonic() - t0, 2)})
        print(f"{'OK ' if ok else 'BAD'} {name:22s} {rep.kind:10s} llm {rows[-1]['llm_s']:5.1f}s x{rows[-1]['llm_calls']} "
              f"| {rows[-1]['plans']} | {rep.reply[:90]}", flush=True)
    n = len(rows)
    correct = sum(r["ok"] for r in rows)
    motion = [r["first_motion_s"] for r in rows if r["first_motion_s"] is not None]
    summary = {"model": llm.model, "cases": n, "correct": correct, "plan_correctness": correct / n,
               "requests_succeeded": sum(r["success"] for r in rows) / n,
               "llm_s_mean": float(np.mean([r["llm_s"] for r in rows])),
               "llm_calls_mean": float(np.mean([r["llm_calls"] for r in rows])),
               "repairs_total": sum(r["repairs"] for r in rows),
               "first_motion_s_mean": float(np.mean(motion)) if motion else None,
               "first_motion_s_p95": float(np.percentile(motion, 95)) if motion else None}
    print("\n" + json.dumps(summary, indent=1))
    Path(os.environ.get("AIBOT_LLM_REPORT", "/tmp/aibot_planner_llm.json")).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    assert n >= 25
    assert correct / n >= 0.8, f"plan correctness {correct}/{n}: " + ", ".join(r["case"] for r in rows if not r["ok"])
