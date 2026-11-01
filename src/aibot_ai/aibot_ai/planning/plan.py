"""Plan format for the LLM planner: JSON schema, parsing, and deterministic validation.

The LLM answers with ONE JSON object (Ollama enforces the schema while decoding):
    {"kind": "plan" | "clarify" | "impossible" | "answer",
     "message": "<text for the user>",
     "steps": [{"skill": ..., "object": <id>, "target": <id>, "relation": ..., "region": ...}, ...]}
Skills are named, never joint angles or coordinates. validate() checks a plan against the
current world state by simulating which hand holds what, so a plan that picks a container,
places while holding nothing, or names an id that does not exist is caught before any motion;
its error list goes back to the LLM for one repair attempt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..motion.expressive import MOTION_SCHEMA
from ..geometry.place import CONTAINER_LABELS, FLAT_TOP_LABELS, REGION_NAMES
from ..skills.matching import SHAPES, parse

PLANNER_SKILLS = ("scan", "pick", "place", "move_aside", "look_at", "home", "release", "wave", "nod", "shake_head", "express")
PLACE_RELATIONS = ("in", "on", "left_of", "right_of", "in_front_of", "behind", "at_region")
KINDS = ("plan", "clarify", "impossible", "answer")
MAX_STEPS = 16

PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(KINDS)},
        "message": {"type": "string"},
        "steps": {
            "type": "array",
            "maxItems": MAX_STEPS,
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "enum": list(PLANNER_SKILLS)},
                    "object": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string", "enum": ["", *PLACE_RELATIONS]},
                    "region": {"type": "string", "enum": ["", *REGION_NAMES]},
                },
                "required": ["skill"],
            },
        },
        "motion": MOTION_SCHEMA,
    },
    "required": ["kind", "message", "steps"],
}


# for models that do not design motions (the local fallback): less to decode
LOCAL_PLAN_SCHEMA: dict = {**PLAN_SCHEMA, "properties": {k: v for k, v in PLAN_SCHEMA["properties"].items() if k != "motion"}}


@dataclass
class Step:
    skill: str
    object: str = ""
    target: str = ""
    relation: str = ""
    region: str = ""
    motion: str = ""                  # express: the motion description (JSON)

    def text(self) -> str:
        if self.skill == "place":
            where = self.region if self.relation == "at_region" else self.target
            return f"place({self.relation} {where})"
        args = self.object or self.target or self.region
        return f"{self.skill}({args})" if args else f"{self.skill}()"


@dataclass
class Plan:
    kind: str
    message: str
    steps: list[Step] = field(default_factory=list)
    motion: dict | None = None        # a gesture the LLM designed, validated later by the skill server


@dataclass
class WorldItem:
    """Planner's view of one perceived object."""
    id: str
    label: str
    color: str
    x: float
    y: float
    size: float = 0.0                 # largest footprint extent (m)
    held_by: str = ""
    inside: str = ""
    semantic_name: str = ""           # what the camera model says it is
    description: str = ""

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def is_container(self) -> bool:
        return self.label in CONTAINER_LABELS


@dataclass
class World:
    objects: list[WorldItem]
    holding: dict[str, str] = field(default_factory=lambda: {"left": "", "right": ""})

    def get(self, object_id: str) -> WorldItem | None:
        return next((o for o in self.objects if o.id == object_id), None)


def parse_plan(data: object) -> Plan:
    """Tolerant conversion of the LLM's JSON (already schema-constrained) into a Plan."""
    if not isinstance(data, dict):
        raise ValueError("plan is not a JSON object")
    kind = str(data.get("kind", "")).strip()
    steps = []
    for s in data.get("steps") or []:
        if not isinstance(s, dict):
            raise ValueError(f"step is not an object: {s!r}")
        steps.append(Step(**{k: str(s.get(k) or "").strip() for k in ("skill", "object", "target", "relation", "region")}))
    motion = data.get("motion")
    motion = motion if isinstance(motion, dict) and (motion.get("keyframes") or motion.get("point_at")) else None
    return Plan(kind, str(data.get("message", "")).strip(), steps, motion)


def validate(plan: Plan, world: World) -> list[str]:
    """Problems with the plan in this world ([] = executable as far as we can tell)."""
    errs: list[str] = []
    if plan.kind not in KINDS:
        return [f"kind must be one of {', '.join(KINDS)}"]
    if plan.kind != "plan":
        if plan.steps:
            errs.append(f"a '{plan.kind}' answer must have no steps")
        if not plan.message:
            errs.append(f"a '{plan.kind}' answer needs a message for the user")
        return errs
    if not plan.steps and plan.motion is None:
        return ["a plan needs at least one step (or use kind 'answer'/'clarify'/'impossible')"]
    if len(plan.steps) > MAX_STEPS:
        errs.append(f"at most {MAX_STEPS} steps")
    ids = ", ".join(o.id for o in world.objects) or "none"
    holding = dict(world.holding)
    where_now: dict[str, str] = {}
    for i, s in enumerate(plan.steps, 1):
        tag = f"step {i} {s.text()}"
        if s.skill not in PLANNER_SKILLS:
            errs.append(f"{tag}: unknown skill (use {', '.join(PLANNER_SKILLS)})")
            continue
        if s.skill in ("pick", "move_aside"):
            o = world.get(s.object)
            if o is None:
                errs.append(f"{tag}: unknown object '{s.object}' (known: {ids})")
                continue
            if o.is_container:
                errs.append(f"{tag}: {o.id} is a fixed container and cannot be moved")
                continue
            if s.object in holding.values():
                if s.skill == "pick":
                    errs.append(f"{tag}: {o.id} is already held")
                continue
            free = [h for h, v in holding.items() if not v]
            if not free:
                errs.append(f"{tag}: both hands are full, place something first")
                continue
            if s.skill == "pick":
                holding[free[0]] = s.object
        elif s.skill == "place":
            held = [v for v in holding.values() if v]
            if not held:
                errs.append(f"{tag}: nothing is held here; pick an object before placing it")
                continue
            if s.object and s.object not in held:
                errs.append(f"{tag}: '{s.object}' is not held (held: {', '.join(held)})")
                continue
            if len(held) > 1 and not s.object:
                errs.append(f"{tag}: two objects held, set 'object' to the one to place")
                continue
            obj = s.object or held[0]
            if s.relation not in PLACE_RELATIONS:
                errs.append(f"{tag}: relation must be one of {', '.join(PLACE_RELATIONS)}")
                continue
            if s.relation == "at_region":
                if s.region not in REGION_NAMES:
                    errs.append(f"{tag}: region must be one of {', '.join(REGION_NAMES)}")
                    continue
            else:
                ref = world.get(s.target)
                if ref is None:
                    errs.append(f"{tag}: unknown target '{s.target}' (known: {ids})")
                    continue
                if ref.id == obj:
                    errs.append(f"{tag}: cannot place {obj} relative to itself")
                    continue
                if s.relation == "in" and not ref.is_container:
                    errs.append(f"{tag}: {ref.id} is not a container; use 'on' or put it in a bin/tray")
                    continue
                if s.relation == "on" and not ref.is_container and ref.label not in FLAT_TOP_LABELS:
                    errs.append(f"{tag}: {ref.id} ({ref.label}) has no flat top to stack on")
                    continue
                if ref.id in holding.values():
                    errs.append(f"{tag}: {ref.id} is in a hand, place it first")
                    continue
            for h, v in holding.items():
                if v == obj:
                    holding[h] = ""
            where_now[obj] = s.relation
        elif s.skill == "look_at":
            if not (world.get(s.object) or s.region in REGION_NAMES):
                errs.append(f"{tag}: look_at needs a known object or a region")
        elif s.skill == "release":
            holding = {h: "" for h in holding}
    return errs


# ---------------------------------------------------------------- ambiguity guard
_PLURAL = re.compile(r"\b(all|every|each|both|everything|them|those|these|cubes|boxes|blocks|cans|balls|mugs|cups)\b")
_SPATIAL = re.compile(r"\b(left|right|leftmost|rightmost|nearest|closest|front|back|farthest|furthest|middle|biggest|smallest)\b")


def ambiguous_mentions(text: str, world: World, plan: Plan) -> list[list[WorldItem]]:
    """Deterministic safety net for a small LLM that guesses instead of asking.

    For every "[colour] <shape>" phrase in the user's words that matches several objects, when
    the words carry no plural/"all", no spatial qualifier and no object id, and the plan acts on
    one of those objects, the request was ambiguous and the user is asked instead."""
    t = text.lower()
    if _PLURAL.search(t) or _SPATIAL.search(t) or any(o.id in t for o in world.objects):
        return []
    words = re.findall(r"[a-z]+", t)
    acted = {s.object for s in plan.steps if s.object} | {s.target for s in plan.steps if s.target}
    out = []
    for i, w in enumerate(words):
        if w not in SHAPES:
            continue
        colors, shapes, _ = parse(" ".join(words[max(0, i - 1): i + 1]))
        article = words[i - 1 - bool(colors)] if i - 1 - bool(colors) >= 0 else ""
        if article in ("a", "an", "any", "some", "either", "another"):
            continue                                           # "a red cube": any of them will do
        cands = [o for o in world.objects if not o.held_by and o.label in shapes and (not colors or o.color in colors)]
        if len(cands) > 1 and acted & {o.id for o in cands}:
            out.append(cands)
    return out
