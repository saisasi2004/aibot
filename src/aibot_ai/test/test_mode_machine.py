"""Mode state machine: legal transitions, SWITCHING exclusivity, abort into HOLD, STOP."""
from aibot_ai.mode_machine import STEPS_TO_AI, STEPS_TO_MIMIC, Mode, ModeMachine


def run(m: ModeMachine, target: str, fail: str | None = None) -> Mode:
    tr, _ = m.request(target)
    assert tr is not None
    seen = []
    aborted = False
    for step in tr.steps:
        seen.append(m.published_mode)
        if not m.step_result(step, step.name != fail, "boom"):
            aborted = True
            break
    assert all(s == Mode.SWITCHING for s in seen)     # nobody active while steps run
    return m.finish(aborted)


def test_boot_to_mimic_to_ai_and_back():
    m = ModeMachine()
    assert m.published_mode == Mode.BOOT
    assert run(m, "mimic") == Mode.MIMIC
    assert run(m, "ai") == Mode.AI
    assert run(m, "mimic") == Mode.MIMIC
    assert m.published_mode == Mode.MIMIC


def test_rejects_unknown_same_and_busy():
    m = ModeMachine()
    run(m, "mimic")
    assert m.request("mimic")[0] is None
    assert m.request("teleport")[0] is None
    assert m.request("switching")[0] is None
    tr, _ = m.request("ai")
    assert tr is not None
    assert m.request("mimic")[0] is None             # busy while switching
    assert "busy" in m.request("mimic")[1]


def test_required_failure_aborts_into_hold():
    m = ModeMachine()
    run(m, "mimic")
    assert run(m, "ai", fail="pause_retarget") == Mode.HOLD
    assert "pause_retarget" in m.last_error
    assert run(m, "mimic") == Mode.MIMIC             # recover from HOLD


def test_optional_failure_still_completes():
    m = ModeMachine()
    run(m, "mimic")
    assert run(m, "ai", fail="pause_webcam") == Mode.AI
    assert "pause_webcam" in m.last_error
    assert run(m, "mimic", fail="release_objects") == Mode.MIMIC


def test_stop_preempts_and_holds():
    m = ModeMachine()
    run(m, "mimic")
    m.request("ai")
    tr = m.request_hold()
    for step in tr.steps:
        assert m.step_result(step, False)            # nothing in a STOP can abort it
    assert m.finish(False) == Mode.HOLD
    assert run(m, "ai") == Mode.AI


def test_handover_order():
    """Active source stopped and robot settled before the other source is enabled."""
    to_ai = [s.name for s in STEPS_TO_AI]
    assert to_ai.index("pause_retarget") < to_ai.index("settle") < to_ai.index("enable_ai")
    to_mimic = [s.name for s in STEPS_TO_MIMIC]
    assert to_mimic.index("cancel_skills") < to_mimic.index("settle") < to_mimic.index("resume_retarget")
    assert to_mimic.index("release_objects") < to_mimic.index("resume_retarget")


def test_ten_round_trips():
    m = ModeMachine()
    run(m, "mimic")
    for _ in range(10):
        assert run(m, "ai") == Mode.AI
        assert run(m, "mimic") == Mode.MIMIC
