"""Planner logic with a scripted LLM (no Ollama): validation, repair, replanning, clarification,
ambiguity guard, cancel. The real-LLM regression is test_planner_llm.py."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_world import FakeSkills, scene  # noqa: E402

from aibot_ai.backends.base import BackendError, LLMBackend  # noqa: E402
from aibot_ai.planning.agent import PlannerAgent  # noqa: E402
from aibot_ai.planning.plan import LOCAL_PLAN_SCHEMA, PLAN_SCHEMA, Plan, Step, parse_plan, validate  # noqa: E402
from aibot_ai.planning.prompt import build_messages, world_text  # noqa: E402


class ScriptedLLM(LLMBackend):
    """Returns the queued answers in order; records the prompts it was given."""

    def __init__(self, answers: list) -> None:
        self.answers, self.prompts = list(answers), []

    def generate(self, messages, schema=None):
        assert schema is PLAN_SCHEMA or schema is LOCAL_PLAN_SCHEMA
        self.prompts.append(messages[-1]["content"])
        if not self.answers:
            raise BackendError("no more scripted answers")
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def plan(*steps, kind="plan", message="ok"):
    return {"kind": kind, "message": message, "steps": [dict(s) for s in steps]}


PICK_RED1 = {"skill": "pick", "object": "red_cube"}
IN_BIN = {"skill": "place", "relation": "in", "target": "silver_bin"}


def agent(answers, **fake):
    skills = FakeSkills(scene(), **fake)
    said = []
    return PlannerAgent(ScriptedLLM(answers), skills, said.append), skills, said


# ---------------------------------------------------------------- validation
def test_validate_accepts_simple_plan():
    w = FakeSkills(scene()).world()
    assert validate(parse_plan(plan(PICK_RED1, IN_BIN)), w) == []


@pytest.mark.parametrize("steps,needle", [
    ([{"skill": "pick", "object": "ghost_9"}], "unknown object"),
    ([{"skill": "pick", "object": "silver_bin"}], "container"),
    ([IN_BIN], "nothing is held"),
    ([PICK_RED1, {"skill": "place", "relation": "in", "target": "blue_cube"}], "not a container"),
    ([PICK_RED1, {"skill": "place", "relation": "on", "target": "orange_ball"}], "no flat top"),
    ([PICK_RED1, {"skill": "place", "relation": "at_region", "region": "kitchen"}], "region"),
    ([PICK_RED1, {"skill": "place", "relation": "on", "target": "red_cube"}], "itself"),
    ([PICK_RED1, {"skill": "pick", "object": "blue_cube"}, {"skill": "pick", "object": "white_mug"}], "both hands"),
    ([PICK_RED1, PICK_RED1], "already held"),
])
def test_validate_rejects(steps, needle):
    errs = validate(parse_plan(plan(*steps)), FakeSkills(scene()).world())
    assert any(needle in e for e in errs), errs


def test_validate_non_plan_kinds():
    w = FakeSkills(scene()).world()
    assert validate(Plan("clarify", "which one?"), w) == []
    assert validate(Plan("clarify", ""), w)
    assert validate(Plan("impossible", "no banana", [Step("pick", "red_cube")]), w)


def test_prompt_contains_world_and_is_compact():
    w = FakeSkills(scene()).world()
    text = world_text(w)
    assert "red_cube_2: red cube at (50, -11)" in text and "silver_bin: silver bin (container)" in text
    msgs = build_messages("put the ball in the bin", w, ["user: hi -> answer: hello"])
    assert msgs[0]["role"] == "system" and "REQUEST: put the ball in the bin" in msgs[-1]["content"]
    assert sum(len(m["content"]) for m in msgs) < 6000            # ~1.5k tokens: CPU prompt eval budget


# ---------------------------------------------------------------- loop
def test_executes_valid_plan():
    a, skills, said = agent([plan(PICK_RED1, IN_BIN)])
    rep = a.handle("put the red cube on the left in the bin")
    assert rep.success and rep.kind == "plan" and skills.is_in("red_cube", "silver_bin")
    assert rep.first_motion_s is not None and rep.llm_calls == 1 and said[-1].startswith("Done")


def test_repair_round_fixes_invalid_plan():
    a, skills, _ = agent([plan({"skill": "pick", "object": "red_thing"}, IN_BIN), plan(PICK_RED1, IN_BIN)])
    rep = a.handle("put red_cube in the bin")
    assert rep.success and rep.repairs == 1 and rep.llm_calls == 2
    assert "unknown object 'red_thing'" in a.llm.prompts[-1]


def test_gives_up_after_failed_repair_without_moving():
    a, skills, said = agent([plan({"skill": "pick", "object": "x"}), plan({"skill": "pick", "object": "y"})])
    rep = a.handle("put red_cube in the bin")
    assert not rep.success and rep.kind == "error" and skills.motions == [] and "could not make" in said[-1]


def test_replans_after_failed_step():
    """red_cube unreachable -> the LLM is told and chooses red_cube_2."""
    a, skills, _ = agent([plan(PICK_RED1, IN_BIN), plan({"skill": "pick", "object": "red_cube_2"}, IN_BIN)],
                         unreachable={"red_cube"})
    rep = a.handle("put a red cube in the bin")
    assert rep.success and rep.replans == 1 and skills.is_in("red_cube_2", "silver_bin")
    assert "UNREACHABLE" in a.llm.prompts[-1]


def test_reports_failure_after_max_replans():
    a, skills, said = agent([plan(PICK_RED1, IN_BIN)] * 3, unreachable={"red_cube"})
    rep = a.handle("put red_cube in the bin")
    assert not rep.success and rep.kind == "failed" and rep.replans == 2 and "UNREACHABLE" in said[-1]


def test_clarify_then_answer_continues_the_request():
    a, skills, said = agent([plan(kind="clarify", message="Which red cube?"),
                             plan({"skill": "pick", "object": "red_cube_2"}, IN_BIN)])
    r1 = a.handle("put the red cube in the bin")
    assert r1.kind == "clarify" and skills.motions == []
    r2 = a.handle("the one on the right")
    assert r2.success and skills.is_in("red_cube_2", "silver_bin")
    assert "you asked: Which red cube?" in a.llm.prompts[-1]


def test_ambiguity_guard_asks_instead_of_guessing():
    """The LLM guesses red_cube for 'the red cube' although there are two: ask, do not move."""
    a, skills, said = agent([plan(PICK_RED1, IN_BIN)])
    rep = a.handle("put the red cube in the bin")
    assert rep.kind == "clarify" and skills.motions == [] and "red_cube" in said[-1] and "red_cube_2" in said[-1]


@pytest.mark.parametrize("text", ["put the red cube on the left in the bin", "put both red cubes in the bin",
                                  "put red_cube in the bin", "put the blue cube in the bin"])
def test_ambiguity_guard_lets_specific_requests_through(text):
    steps = [{"skill": "pick", "object": "blue_cube" if "blue" in text else "red_cube"}, IN_BIN]
    a, skills, _ = agent([plan(*steps)])
    assert a.handle(text).kind == "plan"


def test_impossible_and_answer_do_not_move():
    a, skills, said = agent([plan(kind="impossible", message="There is no banana.")])
    rep = a.handle("put the banana in the bin")
    assert rep.kind == "impossible" and not rep.success and skills.motions == [] and said == ["There is no banana."]
    assert skills.calls[-1] == "shake_head()"                 # a visible "no"


def test_backend_error_is_reported():
    a, skills, said = agent([BackendError("Ollama is not reachable")])
    rep = a.handle("put red_cube in the bin")
    assert rep.kind == "error" and "not reachable" in said[-1]


def test_cancel_stops_between_steps():
    a, skills, said = agent([plan(PICK_RED1, IN_BIN)])
    original = skills.run

    def run_then_cancel(step, cancelled):
        out = original(step, cancelled)
        a.cancel()                         # STOP pressed while the first step ran
        return out

    skills.run = run_then_cancel
    rep = a.handle("put red_cube in the bin")
    assert rep.kind == "stopped" and skills.motions == ["pick(red_cube)"]


def test_scans_first_when_world_is_empty():
    skills = FakeSkills([])
    a = PlannerAgent(ScriptedLLM([plan(kind="answer", message="I see nothing.")]), skills, lambda m: None)
    a.handle("what do you see?")
    assert skills.calls == ["refresh", "nod()"]                # looked first, then reacted


def test_first_request_in_ai_mode_scans_even_with_a_stale_world():
    a, skills, _ = agent([plan(kind="answer", message="I see cubes.")])
    a.stale = True
    a.handle("what do you see?")
    assert skills.calls[0] == "refresh"
    a2, skills2, _ = agent([plan(kind="answer", message="ok")])
    a2.stale = False
    a2.handle("what do you see?")
    assert "refresh" not in skills2.calls


def test_greeting_plan_waves_without_an_extra_nod():
    a, skills, _ = agent([plan({"skill": "wave"}, message="Hi!")])
    a.stale = False
    rep = a.handle("say hi")
    assert rep.success and skills.calls == ["wave()"]


# ---------------------------------------------------------------- greetings (Phase 1 regression)
@pytest.mark.parametrize("text", ["HI", "Say Hi", "wave hi", "hello robot", "good morning", "bye"])
def test_greeting_answered_with_words_only_still_moves(text):
    """The logged failure: kind=answer, steps=[] -> text, no motion."""
    a, skills, said = agent([plan(kind="answer", message="Hi!")])
    rep = a.handle(text)
    assert rep.kind == "answer" and said == ["Hi!"] and skills.calls == ["wave()"]
    assert rep.executed == [("wave()", "OK")]


def test_greeting_does_not_wait_for_a_scan_and_the_task_after_it_still_scans():
    a, skills, _ = agent([plan({"skill": "wave"}, message="Hi!"), plan(PICK_RED1, IN_BIN)])
    a.stale = True
    a.handle("HI")
    assert "refresh" not in skills.calls and a.stale
    a.handle("put red_cube in the bin")
    assert skills.calls[1] == "refresh" and not a.stale


def test_social_plan_says_one_plain_message_before_moving():
    events = []
    skills = FakeSkills(scene())
    run = skills.run
    skills.run = lambda st, c: (events.append(st.text()), run(st, c))[1]
    a = PlannerAgent(ScriptedLLM([plan({"skill": "wave"}, message="Hi!")]), skills, lambda m: events.append(m))
    assert a.handle("HI").success
    assert events == ["Hi!", "wave()"]


def test_a_task_mentioning_a_greeting_is_not_social():
    from aibot_ai.skills.gestures import is_greeting
    assert is_greeting("HI") and is_greeting("wave hi")
    assert not is_greeting("put the blue cube in the bin and then say hi to me")
    assert not is_greeting("put the cube in the bin")
    assert not is_greeting("this is a highly shiny cube")


def test_head_swing_keeps_its_full_width_at_the_limit():
    import numpy as np
    from aibot_ai.skills.gestures import swing_targets
    lower, upper = np.array([-1.5, -0.15]), np.array([1.5, 0.50])
    rest = np.array([0.0, 0.50])                               # the scan pose, at the pitch limit
    ts = swing_targets(rest, 1, 0.25, lower, upper)
    pitch = [t[1] for t in ts]
    assert max(pitch) - min(pitch) == pytest.approx(0.50)      # was 0.149 in the sim
    assert all(lower[1] <= p <= upper[1] for p in pitch)
    assert np.allclose(ts[-1], rest)                           # ends where it started
    assert all(t[0] == 0.0 for t in ts)                        # only the asked axis moves


@pytest.mark.parametrize("data,kind,n", [
    ({"kind": "answer", "message": "Hi", "steps": [{"skill": "wave"}]}, "plan", 1),      # logged from granite
    ({"kind": "answer", "message": "Hi", "steps": []}, "answer", 0),
    ({"kind": "impossible", "message": "No", "steps": [{"skill": "wave"}]}, "impossible", 0),
    ({"kind": "clarify", "message": "Which?", "steps": [{"skill": "home"}]}, "clarify", 0),
])
def test_kind_and_steps_are_normalised(data, kind, n):
    p = parse_plan(data)
    assert p.kind == kind and len(p.steps) == n


def test_mislabelled_answer_with_steps_runs_instead_of_failing():
    a, skills, said = agent([{"kind": "answer", "message": "I look at it.", "steps": [{"skill": "look_at", "object": "blue_cube"}]}])
    rep = a.handle("look at the blue cube")
    assert rep.success and "look_at(blue_cube)" in skills.calls and "could not" not in said[-1]


def test_small_talk_skips_the_scan_but_tasks_do_not():
    from aibot_ai.skills.gestures import needs_view
    assert not needs_view("How does it feel to have a body??") and not needs_view("tell me a joke")
    assert needs_view("what do you see?") and needs_view("point towards the red cube") and needs_view("put it in the bin")
    a, skills, _ = agent([plan(kind="answer", message="It feels great.")])
    a.stale = True
    a.handle("How does it feel to have a body??")
    assert "refresh" not in skills.calls and a.stale


def test_prompt_speaks_as_the_robot():
    from aibot_ai.planning.prompt import SYSTEM
    assert "You are Aibot" in SYSTEM and "never say you are an AI" in SYSTEM
