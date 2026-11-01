"""What to ask the VLM after a skill, and how its answer combines with geometry. ROS-free.

Geometry (the skill's own checks) stays authoritative. The VLM only adds a second opinion: a
disagreement is reported, and triggers a replan only if configured, since the benchmark measured
the cloud VLM wrongly saying "yes" to absent objects (README model table).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .plan import Step, World

SCENE_QUESTION = re.compile(r"\b(what|which|how many|describe|is there|are there|do you see|can you see|look)\b", re.I)


def is_scene_question(text: str) -> bool:
    return bool(SCENE_QUESTION.search(text))


def _name(world: World, oid: str) -> str:
    o = world.get(oid)
    if o is None:
        return oid.replace("_", " ")
    return o.semantic_name or f"{o.color} {o.label}".strip()


@dataclass
class Check:
    question: str
    expected: bool


def check_for(step: Step, world: World, picked: str = "") -> Check | None:
    """The yes/no question that should hold after a successful step, or None if there is none."""
    if step.skill == "pick":
        return Check(f"Is there a {_name(world, step.object)} on the table where the robot picked it up?", False) \
            if world.get(step.object) is not None else None
    if step.skill != "place" or not picked:
        return None
    what = _name(world, picked)
    if step.relation == "in":
        return Check(f"Is the {what} inside the {_name(world, step.target)}?", True)
    if step.relation == "on":
        return Check(f"Is the {what} on top of the {_name(world, step.target)}?", True)
    if step.relation == "at_region":
        return Check(f"Is a {what} standing on the table?", True)
    return Check(f"Is a {what} on the table next to the {_name(world, step.target)}?", True)
