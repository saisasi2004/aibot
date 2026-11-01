"""Planning loop: natural-language request -> validated skill plan -> execution -> replanning.

ROS-free so the whole loop runs in pytest against a fake world and fake skills. The ROS node
(planner_node) plugs in the real skill server and perception through the SkillRunner protocol.

    handle(text):
      world <- runner.world()        (scan first if nothing is known)
      plan  <- LLM(world, request)   validate; on errors one repair round with the error list
      clarify / impossible / answer  -> say the message, done (clarify: next text is the answer)
      ambiguity guard                -> ask instead of guessing (see plan.ambiguous_mentions)
      execute steps one by one; on a failed step: refresh the world, replan with the failure
      as feedback (at most max_replans times), else explain the failure honestly.
Cancel (STOP, mode change) is checked between steps and passed into running skills.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..backends.base import BackendError, LLMBackend
from ..skills.gestures import GESTURE_SKILLS, is_greeting
from .plan import PLAN_SCHEMA, Plan, Step, World, ambiguous_mentions, parse_plan, validate
from .prompt import build_messages

STOP_CODES = ("CANCELLED", "NOT_AI_MODE")


@dataclass
class StepOutcome:
    success: bool
    reason_code: str = "OK"
    message: str = ""


class SkillRunner(Protocol):
    def run(self, step: Step, cancelled: threading.Event) -> StepOutcome: ...
    def world(self) -> World: ...
    def refresh(self) -> None: ...


@dataclass
class AgentConfig:
    max_replans: int = 2
    repair_attempts: int = 1
    history_turns: int = 3


@dataclass
class Report:
    """What happened for one request (for the chat, logs and the evaluation)."""
    request: str
    kind: str = ""
    success: bool = False
    reply: str = ""
    plans: list[list[str]] = field(default_factory=list)
    executed: list[tuple[str, str]] = field(default_factory=list)     # (step, reason code)
    llm_calls: int = 0
    llm_s: float = 0.0
    repairs: int = 0
    replans: int = 0
    first_motion_s: float | None = None
    total_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PlannerAgent:
    def __init__(self, llm: LLMBackend, runner: SkillRunner, say: Callable[[str], None],
                 config: AgentConfig | None = None, log: Callable[[str], None] | None = None) -> None:
        self.llm, self.runner, self.say = llm, runner, say
        self.cfg = config or AgentConfig()
        self.log = log or (lambda _m: None)
        self.cancelled = threading.Event()
        self.history: list[str] = []
        self.pending: tuple[str, str] | None = None        # (request, question) awaiting an answer
        self.stale = True          # look at the table before the next plan (set again on entering AI mode)
        self.gestures = {"answer": "nod", "clarify": "nod", "plan": "nod", "impossible": "shake_head",
                         "failed": "shake_head", "error": "shake_head"}

    def cancel(self) -> None:
        self.cancelled.set()

    # ------------------------------------------------------------------ LLM
    def _plan(self, request: str, world: World, feedback: str, rep: Report) -> tuple[Plan | None, list[str]]:
        msgs = build_messages(request, world, self.history[-self.cfg.history_turns:], feedback)
        plan, errs = None, ["no answer"]
        for attempt in range(1 + self.cfg.repair_attempts):
            t0 = time.monotonic()
            try:
                data = self.llm.generate(msgs, PLAN_SCHEMA)
            except BackendError as exc:
                rep.llm_calls += 1
                rep.llm_s += time.monotonic() - t0
                return None, [str(exc)]
            rep.llm_calls += 1
            rep.llm_s += time.monotonic() - t0
            try:
                plan = parse_plan(data)
                errs = validate(plan, world)
            except (ValueError, TypeError) as exc:
                plan, errs = None, [f"unreadable plan: {exc}"]
            self.log(f"LLM plan (attempt {attempt + 1}): {json.dumps(data)[:400]} -> {errs or 'valid'}")
            if not errs:
                return plan, []
            if attempt < self.cfg.repair_attempts:
                rep.repairs += 1
                msgs = msgs + [{"role": "assistant", "content": json.dumps(data)},
                               {"role": "user", "content": "That plan cannot be executed: " + "; ".join(errs)
                                + ". Reply with a corrected JSON object."}]
        return None, errs

    # ------------------------------------------------------------------ main entry
    def handle(self, text: str) -> Report:
        t0 = time.monotonic()
        self.cancelled.clear()
        rep = Report(text)
        request = text
        if self.pending is not None:                # the user answers our question
            prev, question = self.pending
            request = f"{prev} (you asked: {question} The user answered: {text})"
            self.pending = None
        try:
            self._handle(request, rep, t0)
        finally:
            rep.total_s = time.monotonic() - t0
            self.history.append(f"user: {text} -> {rep.kind or 'error'}: {rep.reply[:120]}")
        return rep

    def _reply(self, rep: Report, kind: str, message: str, success: bool, say: bool = True) -> None:
        rep.kind, rep.reply, rep.success = kind, message, success
        if say:
            self.say(message)
        # every reply gets a visible reaction (nod / head shake), unless it was stopped; a plan that
        # already gestured (e.g. a wave) needs none, and a greeting answered with words only still
        # gets a wave (the model decides the words, not whether the robot reacts)
        gesture = "wave" if kind == "answer" and is_greeting(rep.request) else self.gestures.get(kind)
        gestured = any(s in GESTURE_SKILLS for p in rep.plans[-1:] for s in (x.split("(")[0] for x in p))
        if gesture and not self.cancelled.is_set() and not (kind == "plan" and gestured):
            out = self.runner.run(Step(gesture), self.cancelled)
            rep.executed.append((f"{gesture}()", out.reason_code))

    def _handle(self, request: str, rep: Report, t0: float) -> None:
        world = self.runner.world()
        # a greeting needs no view of the table: react now and keep the scan for the first real task
        # (after a switch from mimic it cost 20-50 s before the wave could start)
        if not is_greeting(rep.request) and (self.stale or not [o for o in world.objects if not o.held_by]):
            self.log("looking at the table first (new in AI mode, or nothing known)")
            self.stale = False
            self.runner.refresh()          # not counted as "first motion": that is the first plan step
            world = self.runner.world()
        feedback = ""
        while True:
            if self.cancelled.is_set():
                return self._reply(rep, "stopped", "Stopped.", False)
            plan, errs = self._plan(request, world, feedback, rep)
            if plan is None:
                rep.errors += errs
                return self._reply(rep, "error", "Sorry, I could not make a workable plan: " + "; ".join(errs[:3]), False)
            rep.plans.append([s.text() for s in plan.steps])
            if plan.kind == "clarify":
                self.pending = (request, plan.message)
                return self._reply(rep, "clarify", plan.message, True)
            if plan.kind in ("impossible", "answer"):
                return self._reply(rep, plan.kind, plan.message, plan.kind == "answer")
            amb = ambiguous_mentions(rep.request, world, plan) if not feedback else []
            if amb:
                options = " or ".join(f"{o.id} at ({o.x * 100:.0f}, {o.y * 100:.0f})" for o in amb[0])
                q = f"Which one do you mean: {options}?"
                self.pending = (request, q)
                return self._reply(rep, "clarify", q, True)
            social = all(s.skill in GESTURE_SKILLS for s in plan.steps)
            # a purely social plan: the words appear as the motion starts, once, without plan jargon
            self.say(plan.message if social else f"{plan.message} Plan: " + ", ".join(s.text() for s in plan.steps))
            failed = None
            for i, step in enumerate(plan.steps):
                if self.cancelled.is_set():
                    return self._reply(rep, "stopped", "Stopped.", False)
                if rep.first_motion_s is None:
                    rep.first_motion_s = time.monotonic() - t0
                out = self.runner.run(step, self.cancelled)
                rep.executed.append((step.text(), out.reason_code))
                self.log(f"{step.text()} -> {out.reason_code} {out.message}")
                if not out.success:
                    failed = (i, step, out)
                    break
            if failed is None:
                return self._reply(rep, "plan", plan.message if social else f"Done: {plan.message}", True, say=not social)
            i, step, out = failed
            if out.reason_code in STOP_CODES or self.cancelled.is_set():
                return self._reply(rep, "stopped", f"Stopped during {step.text()}.", False)
            if rep.replans >= self.cfg.max_replans:
                return self._reply(rep, "failed", f"I could not finish: {step.text()} failed ({out.reason_code}): {out.message}", False)
            rep.replans += 1
            if out.reason_code in ("NOT_FOUND", "GRASP_FAILED", "PLACE_BLOCKED"):
                self.runner.refresh()
            world = self.runner.world()
            done = ", ".join(s.text() for s in plan.steps[:i]) or "nothing"
            feedback = (f"Step {step.text()} failed with {out.reason_code}: {out.message}. Already done: {done}. "
                        "Plan the rest from the WORLD above, choose something else, or say it is impossible.")
