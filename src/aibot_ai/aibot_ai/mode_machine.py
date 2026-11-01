"""Operating-mode state machine: MIMIC <-> AI with an explicit SWITCHING state, plus HOLD.

Pure Python (no ROS) so it can be unit-tested exhaustively; mode_manager_node runs the steps.

Why a SWITCHING state: every node that can command the robot (retarget_node in mimic,
skill_server_node in AI) only publishes when /aibot/mode equals *its* mode. While the steps of a
transition run, the published mode is "switching", so for that window nobody commands the
robot, and there is never a moment where two sources are active.

Why HOLD: STOP, or a required step failing (e.g. retarget_node not answering), must leave the
robot in a state where nothing moves it. HOLD is that state; from HOLD any mode may be requested
again, which re-runs the full sequence for that mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    BOOT = "boot"            # before the first transition: nobody may command
    MIMIC = "mimic"
    AI = "ai"
    SWITCHING = "switching"
    HOLD = "hold"


@dataclass(frozen=True)
class Step:
    name: str
    required: bool = True    # a failing required step aborts the transition into HOLD


# Order matters: first stop whoever is active, let the robot settle, then hand over.
STEPS_TO_AI: tuple[Step, ...] = (
    Step("pause_retarget"),
    Step("pause_webcam", required=False),      # only costs CPU if it fails
    Step("settle"),
    Step("enable_ai", required=False),         # AI nodes may not be launched (mimic-only bringup)
)
STEPS_TO_MIMIC: tuple[Step, ...] = (
    Step("cancel_plan", required=False),
    Step("cancel_skills", required=False),
    Step("disable_ai", required=False),
    Step("release_objects", required=False),
    Step("settle"),
    Step("resume_webcam", required=False),
    Step("resume_retarget"),
)
STEPS_TO_HOLD: tuple[Step, ...] = (
    Step("cancel_plan", required=False),
    Step("cancel_skills", required=False),
    Step("pause_retarget", required=False),
)


@dataclass
class Transition:
    target: Mode
    steps: tuple[Step, ...]
    origin: Mode
    failures: list[str] = field(default_factory=list)


class ModeMachine:
    def __init__(self) -> None:
        self.mode = Mode.BOOT
        self.transition: Transition | None = None
        self.last_error = ""

    @property
    def published_mode(self) -> Mode:
        """The value other nodes see: SWITCHING for the whole duration of a transition."""
        return Mode.SWITCHING if self.transition is not None else self.mode

    def request(self, target: str) -> tuple[Transition | None, str]:
        """Start a transition to `target` ("mimic" | "ai"). Returns (transition, message).

        transition is None when the request is rejected or there is nothing to do.
        """
        try:
            t = Mode(target)
        except ValueError:
            return None, f"unknown mode '{target}' (use mimic or ai)"
        if t not in (Mode.MIMIC, Mode.AI):
            return None, f"'{target}' cannot be requested (use mimic or ai)"
        if self.transition is not None:
            return None, f"busy: already switching to {self.transition.target.value}"
        if t == self.mode:
            return None, f"already in {t.value}"
        steps = STEPS_TO_AI if t == Mode.AI else STEPS_TO_MIMIC
        self.transition = Transition(t, steps, self.mode)
        return self.transition, f"switching {self.mode.value} -> {t.value}"

    def request_hold(self) -> Transition:
        """STOP: always accepted; pre-empts a running transition."""
        self.transition = Transition(Mode.HOLD, STEPS_TO_HOLD, self.mode)
        return self.transition

    def step_result(self, step: Step, ok: bool, detail: str = "") -> bool:
        """Report a step outcome. Returns False if the transition must abort (required step failed)."""
        if self.transition is None:
            return False
        if not ok:
            self.transition.failures.append(f"{step.name}: {detail or 'failed'}")
            if step.required and self.transition.target != Mode.HOLD:
                return False
        return True

    def finish(self, aborted: bool) -> Mode:
        """End the transition: target mode on success, HOLD on abort."""
        tr = self.transition
        if tr is None:
            return self.mode
        self.transition = None
        if aborted:
            self.mode = Mode.HOLD
            self.last_error = "; ".join(tr.failures)
        else:
            self.mode = tr.target
            self.last_error = "; ".join(tr.failures)   # non-required failures are still reported
        return self.mode
