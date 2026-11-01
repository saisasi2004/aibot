"""Motion safety layer, gesture library and agent integration, with a fake LLM and real kinematics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_world import FakeSkills, scene  # noqa: E402
from test_motion import _setup  # noqa: E402
from test_planner import ScriptedLLM  # noqa: E402

from aibot_ai.motion.expressive import PRESETS, Limits, MotionError, MotionValidator, hand_poses, seed_motions  # noqa: E402
from aibot_ai.motion.trajectory import Trajectory  # noqa: E402
from aibot_ai.planning.agent import PlannerAgent, StepOutcome  # noqa: E402
from aibot_ai.planning.gestures import GestureLibrary, intent_key  # noqa: E402
from aibot_ai.planning.plan import PLAN_SCHEMA, parse_plan  # noqa: E402
from aibot_ai.planning.prompt import MOTION_EXAMPLES, build_messages  # noqa: E402
from aibot_ai.backends.schema_check import schema_errors  # noqa: E402


@pytest.fixture(scope="module")
def v():
    _, jm, table, m = _setup()
    lo, hi = jm.limits.lower.copy(), jm.limits.upper.copy()
    for side in ("left", "right"):
        lo[m.idx[side]], hi[m.idx[side]] = m.solvers[side].lower, m.solvers[side].upper
    names = [jm.names[i] for i in jm.groups["right_hand"]]

    def plan_point(arm, oid, q):
        if oid != "blue_cube_1":
            raise ValueError(f"unknown object '{oid}'")
        r = m.reach(arm, q, np.array([0.42, 0.0, table["top_z"] + 0.2]))
        wps, why = m.joint_move(arm, q, m.with_arm(arm, q, r.q))
        if wps is None:
            raise ValueError(why)
        return wps

    val = MotionValidator(jm.names, {g: np.array(x) for g, x in jm.groups.items()}, lo, hi,
                          hand_poses(names, np.ones(len(names))), m.path_hit, plan_point, Limits())
    val.q0, val.jm, val.m, val.table = np.zeros(len(jm.names)), jm, m, table
    return val


def kf(t, **joints):
    return {"t": t, "joints": [{"name": k, "angle": a} for k, a in joints.items()]}


def test_valid_motion_returns_to_start(v):
    p = v.prepare({"intent": "x", "keyframes": [kf(1.0, neck_yaw_joint=0.5), kf(2.0, neck_yaw_joint=-0.5)]}, v.q0)
    assert np.allclose(p.waypoints[-1], v.q0) and p.groups == ["head"]
    assert p.waypoints[0][v.jm.index["neck_yaw_joint"]] == 0.5


def test_unknown_joint_rejected_with_a_repairable_message(v):
    with pytest.raises(MotionError, match="unknown joint 'right_elbow'"):
        v.prepare({"keyframes": [kf(1.0, right_elbow=1.0)]}, v.q0)


def test_limits_are_clamped_not_rejected(v):
    p = v.prepare({"keyframes": [kf(1.0, neck_pitch_joint=3.0)]}, v.q0)
    assert p.waypoints[0][v.jm.index["neck_pitch_joint"]] == pytest.approx(v.upper[v.jm.index["neck_pitch_joint"]])
    assert any("clamped" in w for w in p.warnings)


def test_elbow_cannot_hyperextend_and_waist_stays_in_the_skill_range(v):
    p = v.prepare({"keyframes": [kf(1.0, right_elbow_pitch_joint=0.7, waist_pitch_joint=-1.0)]}, v.q0)
    assert p.waypoints[0][v.jm.index["right_elbow_pitch_joint"]] <= 0.0
    assert p.waypoints[0][v.jm.index["waist_pitch_joint"]] >= -0.40


@pytest.mark.parametrize("spec,needle", [
    ({"keyframes": []}, "needs keyframes"), ("nope", "JSON object"),
    ({"keyframes": [kf(1.0, neck_yaw_joint=float("nan"))]}, "not a number"),
    ({"keyframes": [kf(-1.0, neck_yaw_joint=0.1)]}, "time in seconds"),
    ({"keyframes": [kf(1.0, neck_yaw_joint=0.1), kf(1.05, neck_yaw_joint=0.2)]}, "at least"),
    ({"keyframes": [kf(0.5 + i * 0.5, neck_yaw_joint=0.01 * i) for i in range(13)]}, "at most 12"),
    ({"keyframes": [kf(11.0, neck_yaw_joint=0.1)]}, "at most 10"),
    ({"keyframes": [{"t": 1.0, "right_hand": "claw"}]}, "unknown hand preset"),
    ({"keyframes": [{"t": 1.0}]}, "does not move"),
    ({"keyframes": [kf(1.0, neck_yaw_joint=0.1)], "repeat": {"from": 0, "to": 5, "count": 1}}, "outside"),
    ({"keyframes": [kf(1.0, neck_yaw_joint=0.1)], "repeat": {"from": 0, "to": 0, "count": 50}}, "repeat count"),
    ({"point_at": {"object": "ghost"}}, "unknown object"),
])
def test_rejections(v, spec, needle):
    with pytest.raises(MotionError, match=needle):
        v.prepare(spec, v.q0)


def test_repeat_expands_and_keeps_order(v):
    spec = {"keyframes": [kf(1.0, neck_yaw_joint=0.1), kf(1.5, neck_yaw_joint=0.2), kf(2.0, neck_yaw_joint=0.3)],
            "repeat": {"from": 1, "to": 2, "count": 2}}
    p = v.prepare(spec, v.q0)
    yaw = [w[v.jm.index["neck_yaw_joint"]] for w in p.waypoints]
    assert yaw[:7] == pytest.approx([0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3]) and yaw[-1] == 0.0


def test_repeat_past_the_time_limit_is_rejected(v):
    with pytest.raises(MotionError, match="after repeating"):
        v.prepare({"keyframes": [kf(4.0, neck_yaw_joint=0.1), kf(8.0, neck_yaw_joint=0.2)], "repeat": {"from": 0, "to": 1, "count": 2}}, v.q0)


def test_speed_is_limited_by_stretching_time(v):
    p = v.prepare({"keyframes": [kf(0.2, right_shoulder_roll_joint=-1.7)]}, v.q0)
    traj = Trajectory([v.q0] + p.waypoints, np.full(len(v.q0), 1.5), 1.0, 0.15, p.durations)
    assert traj.duration > 2 * 0.2                             # a 1.7 rad move cannot happen in 0.2 s
    q, dq = traj.sample(traj.segments[0].duration / 2)
    assert np.max(np.abs(dq)) <= 1.5 + 1e-6


def test_path_through_the_table_is_rejected(v):
    """Arm brought in over the table's width, then swung forward through its edge."""
    spec = {"keyframes": [kf(1.0, right_shoulder_roll_joint=0.2, right_shoulder_pitch_joint=-0.9, right_elbow_pitch_joint=-0.4)]}
    with pytest.raises(MotionError, match="right arm would hit the table"):
        v.prepare(spec, v.q0)


def test_perceived_objects_and_other_arm_are_checked(v):
    pose = {"right_shoulder_pitch_joint": -0.4, "right_shoulder_roll_joint": -1.6, "right_arm_yaw_joint": -1.0, "right_elbow_pitch_joint": -1.6}
    assert v.prepare({"keyframes": [kf(1.5, **pose)]}, v.q0).groups == ["right_arm"]
    try:
        v.m.set_scene([(0.06, -0.48, 0.05, 2.0)], ids=["pillar"])          # stands where the raised hand goes
        with pytest.raises(MotionError, match="pillar"):
            v.prepare({"keyframes": [kf(1.5, **pose)]}, v.q0)
    finally:
        v.m.set_scene([])


def test_holding_an_object_locks_that_arm_hand_and_the_waist(v):
    held = {"right": "red_cube_1", "left": ""}
    for j in ("right_elbow_pitch_joint", "waist_pitch_joint"):
        with pytest.raises(MotionError, match="locked"):
            v.prepare({"keyframes": [kf(1.0, **{j: -0.2})]}, v.q0, held)
    with pytest.raises(MotionError, match="locked"):
        v.prepare({"keyframes": [{"t": 1.0, "right_hand": "open"}]}, v.q0, held)
    assert v.prepare({"keyframes": [kf(1.0, neck_yaw_joint=0.2)]}, v.q0, held).groups == ["head"]


def test_point_at_uses_ik_and_the_point_hand(v):
    p = v.prepare({"intent": "p", "point_at": {"object": "blue_cube_1", "arm": "right"}}, v.q0)
    top = p.waypoints[len(p.waypoints) // 2 - 1]
    assert np.linalg.norm(v.m.grasp_point("right", top)[0] - [0.42, 0.0, v.table["top_z"] + 0.2]) < 0.02
    assert "right_arm" in p.groups and np.allclose(p.waypoints[-1], v.q0)


def test_hand_presets():
    names = ["h_thumb_a", "h_index_b", "h_mid_c"]
    h = hand_poses(names, np.array([1.0, 2.0, 3.0]))
    assert set(h) == set(PRESETS) and list(h["point"]) == [1.0, 0.0, 3.0] and list(h["thumb_up"]) == [0.0, 2.0, 3.0]
    assert list(h["open"]) == [0, 0, 0] and list(h["half"]) == [0.5, 1.0, 1.5]


@pytest.mark.parametrize("name", ["wave", "nod", "shake_head"])
def test_seed_gestures_pass_the_validator(v, name):
    seeds = seed_motions([-0.4, -1.6, -1.0, -1.6, 0.0, 0.0, 0.0], 0.5, 0.25, 0.3)
    assert v.prepare(seeds[name], v.q0).waypoints


@pytest.mark.parametrize("i", range(len(MOTION_EXAMPLES)))
def test_prompt_examples_are_valid_motions(v, i):
    """What the LLM is shown as good must pass; the worked examples are also schema-valid."""
    answer = MOTION_EXAMPLES[i][1]
    assert not schema_errors(answer, PLAN_SCHEMA)
    assert v.prepare(answer["motion"], v.q0).waypoints


# ---------------------------------------------------------------- library
def test_intent_key():
    assert {intent_key(x) for x in ("HI", "hello!", "hey there", "say hi", "good morning robot")} == {"greet"}
    assert intent_key("bye") == intent_key("goodbye") == "goodbye"
    assert intent_key("wave goodbye") == "wave goodbye" and intent_key("please bow") == "bow"
    assert intent_key("put the red cube in the bin and then the blue cube too") == ""


def test_library_persists_clears_and_skips_object_motions(tmp_path):
    p = tmp_path / "sub" / "g.json"
    lib = GestureLibrary(p)
    lib.store("greet", {"intent": "greet", "keyframes": [1]}, "Hi")
    lib.store("point", {"point_at": {"object": "x"}}, "there")
    assert GestureLibrary(p).get("greet")["message"] == "Hi" and GestureLibrary(p).get("point") is None
    assert lib.clear() == 1 and GestureLibrary(p).get("greet") is None
    p.write_text("{broken")
    assert GestureLibrary(p).entries == {}


# ---------------------------------------------------------------- agent with a fake LLM
class CloudLLM(ScriptedLLM):
    last_backend = "cloud"

    def cloud_ready(self):
        return True


class LocalLLM(ScriptedLLM):
    last_backend = "local"


BOW = {"intent": "bow", "keyframes": [kf(1.0, waist_pitch_joint=-0.3)]}


def ans(message="Hello", motion=BOW):
    return {"kind": "answer", "message": message, "steps": [], "motion": motion}


def agent(llm, tmp_path=None, results=None):
    skills = FakeSkills(scene(), express_results=results or [])
    said = []
    lib = GestureLibrary(tmp_path / "g.json" if tmp_path else "")
    return PlannerAgent(llm, skills, said.append, library=lib), skills, said


def test_generated_motion_plays_with_the_text_and_is_cached(tmp_path):
    a, s, said = agent(CloudLLM([ans("Hi!")]), tmp_path)
    rep = a.handle("HI")
    assert rep.motion_source == "generated" and said == ["Hi!"] and json.loads(s.motions_played[0]) == BOW
    assert s.calls == ["express()"] or s.calls[-1].startswith("express")
    # the repeat: same words or same meaning, no LLM call, same motion
    calls = a.llm.answers, len(a.llm.prompts)
    rep2 = a.handle("hello")
    assert rep2.motion_source == "cache" and rep2.backend == "cache" and rep2.llm_calls == 0 and len(a.llm.prompts) == calls[1]
    assert said == ["Hi!", "Hi!"] and len(s.motions_played) == 2


def test_rejected_motion_gets_one_repair_then_fallback():
    bad = StepOutcome(False, "INVALID", "unknown joint 'x'")
    a, s, _ = agent(CloudLLM([ans(), ans(motion={"intent": "bow", "keyframes": [kf(1.0, neck_yaw_joint=0.2)]})]), results=[bad])
    rep = a.handle("bow please")
    assert rep.motion_source == "generated" and rep.motion_errors == ["unknown joint 'x'"]
    assert "rejected: unknown joint" in a.llm.prompts[-1]
    a2, s2, _ = agent(CloudLLM([ans(), ans()]), results=[bad, bad])
    rep2 = a2.handle("hi")
    assert rep2.motion_source == "seed" and s2.calls[-1] == "wave()" and len(rep2.motion_errors) == 2


def test_local_model_never_supplies_motion_and_uses_the_seed():
    a, s, _ = agent(LocalLLM([ans("Hi")]))
    rep = a.handle("HI")
    assert s.motions_played == [] and s.calls == ["wave()"] and rep.motion_source == "seed"


def test_body_sheet_only_goes_to_the_cloud():
    a, s, _ = agent(CloudLLM([ans()]))
    a.handle("bow")
    w = FakeSkills(scene()).world()
    assert "GESTURES" in build_messages("x", w, [], gestures=True)[0]["content"]
    assert "GESTURES" not in build_messages("x", w, [])[0]["content"]


def test_motion_can_accompany_a_refusal_and_stop_cancels_it():
    a, s, _ = agent(CloudLLM([{"kind": "impossible", "message": "No banana.", "steps": [], "motion": BOW}]))
    assert a.handle("put the banana in the bin").motion_source == "generated"
    a2, s2, _ = agent(CloudLLM([ans()]), results=[StepOutcome(False, "CANCELLED", "stopped")])
    rep = a2.handle("bow")
    assert rep.motion_source == "seed" or s2.motions_played


def test_motion_in_the_schema_and_plan():
    p = parse_plan(ans())
    assert p.motion == BOW and parse_plan({"kind": "answer", "message": "x", "steps": [], "motion": {"intent": "none"}}).motion is None
