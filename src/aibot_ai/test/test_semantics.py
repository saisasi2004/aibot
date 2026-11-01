"""VLM integration, no model: contact sheets, label cache, matching by name, verification, budgets."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_world import FakeSkills, semantic_scene  # noqa: E402
from test_planner import ScriptedLLM, plan  # noqa: E402

from aibot_ai.perception.semantics import SemanticCache, apply_labels, box_agreement, box_iou, contact_sheet  # noqa: E402
from aibot_ai.planning.agent import AgentConfig, PlannerAgent  # noqa: E402
from aibot_ai.planning.plan import Step  # noqa: E402
from aibot_ai.planning.prompt import world_text  # noqa: E402
from aibot_ai.planning.verify import check_for, is_scene_question  # noqa: E402
from aibot_ai.skills.matching import match  # noqa: E402


def test_contact_sheet_tiles_and_is_small():
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[100:150, 100:150] = (200, 30, 30)
    sheet = contact_sheet(rgb, [(100, 100, 150, 150)] * 5)
    assert sheet.shape == (256, 512, 3)                  # 2 rows of 4: much smaller than the frame
    assert sheet[:128, :128].reshape(-1, 3)[:, 0].max() == 255


def test_cache_drops_labels_when_the_object_moves_or_changes():
    c = SemanticCache(0.03)
    c.store("a", [0.4, 0, 1], [0.05, 0.05, 0.05], "mug", "a cup")
    assert c.valid("a", [0.41, 0, 1], [0.05, 0.05, 0.05]) and c.get("a") == ("mug", "a cup")
    assert not c.valid("a", [0.5, 0, 1], [0.05, 0.05, 0.05])
    assert not c.valid("a", [0.4, 0, 1], [0.05, 0.05, 0.12]) and not c.valid("b", [0, 0, 0], [0, 0, 0])


def test_label_answers_ignore_bad_indices():
    got = apply_labels({"labels": [{"index": 0, "name": " Coffee Mug", "description": "cup"}, {"index": 7, "name": "x"},
                                   {"index": 1, "name": ""}, {"index": "1", "name": "y"}]}, ["a", "b"])
    assert got == {"a": ("coffee mug", "cup")}


def test_box_agreement():
    assert box_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)
    assert box_agreement([(0, 0, 10, 10), (20, 20, 30, 30)], [(1, 1, 10, 10)]) == 0.5
    assert box_agreement([(0, 0, 10, 10)], [(0, 0, 10, 10)]) == 1.0 and box_agreement([], []) == 1.0


def test_world_text_shows_names():
    text = world_text(FakeSkills(semantic_scene()).world())
    assert 'purple_block: purple block ("screwdriver": a hand tool)' in text


@pytest.mark.parametrize("text,ids", [
    ("the screwdriver", ["purple_block"]), ("hand me the tool", ["purple_block"]), ("coffee mug", ["white_mug"]),
    ("the soda can", ["silver_can"]), ("tennis ball", ["orange_ball"]), ("the juice", ["green_box"]),
    ("the toy", ["red_cube"]), ("the white cup", ["white_mug"]), ("red cube", ["red_cube"]), ("the hammer", []),
])
def test_match_by_semantic_name(text, ids):
    assert [o.id for o in match(text, FakeSkills(semantic_scene()).world().objects)] == ids


def test_scene_question_and_checks():
    assert is_scene_question("what do you see?") and not is_scene_question("put the mug in the bin")
    w = FakeSkills(semantic_scene()).world()
    assert check_for(Step("pick", "white_mug"), w).expected is False
    c = check_for(Step("place", relation="in", target="silver_bin"), w, "white_mug")
    assert c.expected is True and "coffee mug" in c.question and "metal bin" in c.question
    assert check_for(Step("home"), w) is None and check_for(Step("place", relation="in", target="silver_bin"), w) is None


class Runner(FakeSkills):
    answer = (True, "looks right")
    delay = 0.0

    def verify(self, question):
        time.sleep(self.delay)
        return self.answer

    def describe(self, timeout):
        return "a mug and a bin"


PICK, IN = {"skill": "pick", "object": "white_mug"}, {"skill": "place", "relation": "in", "target": "silver_bin"}


def run(answers, **kw):
    r = Runner(semantic_scene())
    for k, v in kw.items():
        setattr(r, k, v)
    llm = ScriptedLLM(answers)
    cfg = AgentConfig(replan_on_disagree=kw.get("replan", False))
    a = PlannerAgent(llm, r, lambda m: None, cfg)
    a.stale = False
    return a, r, llm


def test_verification_agrees_quietly():
    a, r, _ = run([plan(PICK, IN)], answer=(None, ""))
    rep = a.handle("put the coffee mug in the bin")
    assert rep.success and not rep.disagreements and "doubts" not in rep.reply and len(rep.verifications) == 2


def test_disagreement_is_reported_but_geometry_wins():
    a, r, _ = run([plan(PICK, IN)], answer=(False, "no mug"))
    rep = a.handle("put the coffee mug in the bin")
    assert rep.success and r.is_in("white_mug", "silver_bin") and "doubts" in rep.reply


def test_verification_never_delays_the_next_step():
    a, r, _ = run([plan(PICK, IN)], answer=(True, ""), delay=0.3)
    starts = []
    orig = r.run
    r.run = lambda st, c: (starts.append(time.monotonic()), orig(st, c))[1]
    a.handle("put the coffee mug in the bin")
    assert starts[1] - starts[0] < 0.2


def test_failure_feedback_and_scene_questions_carry_the_camera_words():
    r = Runner(semantic_scene(), unreachable={"white_mug"})
    llm = ScriptedLLM([plan(PICK, IN), plan(kind="impossible", message="cannot")])
    a = PlannerAgent(llm, r, lambda m: None)
    a.stale = False
    a.handle("put the mug in the bin")
    assert "The camera sees: a mug and a bin" in llm.prompts[-1]
    a2, _, llm2 = run([plan(kind="answer", message="a mug")])
    a2.handle("what do you see?")
    assert "CAMERA SAYS: a mug and a bin" in llm2.prompts[-1]
    a3, _, llm3 = run([plan(kind="answer", message="fine")])
    a3.handle("put the mug in the bin")
    assert "CAMERA SAYS" not in llm3.prompts[-1]
