"""Test double for the planner: a tabletop world and skills that act on it (pytest only).

Placement uses the real PlaceResolver, so where a fake place puts an object is where the robot
would aim. Failures can be injected per object (unreachable / grasp fails) to exercise replanning.
"""
from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass, field

import numpy as np

from aibot_ai.geometry.place import PlaceResolver, WorldObject, regions
from aibot_ai.planning.agent import StepOutcome
from aibot_ai.planning.plan import Step, World, WorldItem

TABLE = {"top_z": 0.98, "x": [0.20, 0.60], "y": [-0.18, 0.18], "place_x": [0.32, 0.54], "place_y": [-0.15, 0.15],
         "edge_margin": 0.02, "clearance": 0.025}
SIZES = {"cube": (0.05, 0.05, 0.05), "box": (0.06, 0.04, 0.10), "block": (0.07, 0.05, 0.025), "can": (0.066, 0.066, 0.12),
         "mug": (0.076, 0.076, 0.09), "ball": (0.06, 0.06, 0.06), "bin": (0.13, 0.13, 0.06), "tray": (0.12, 0.12, 0.012)}


@dataclass
class Thing:
    id: str
    label: str
    color: str
    x: float
    y: float
    held_by: str = ""
    inside: str = ""
    on: str = ""
    semantic_name: str = ""
    description: str = ""

    @property
    def size(self) -> tuple[float, float, float]:
        return SIZES[self.label]


@dataclass
class FakeSkills:
    things: list[Thing]
    unreachable: set[str] = field(default_factory=set)
    grasp_fails: set[str] = field(default_factory=set)       # fails once each
    calls: list[str] = field(default_factory=list)
    holding: dict[str, str] = field(default_factory=lambda: {"left": "", "right": ""})
    motions_played: list[str] = field(default_factory=list)
    express_results: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resolver = PlaceResolver(TABLE)
        self.initial = copy.deepcopy(self.things)

    def get(self, oid: str) -> Thing | None:
        return next((t for t in self.things if t.id == oid), None)

    def world(self) -> World:
        items = [WorldItem(t.id, t.label, t.color, t.x, t.y, max(t.size[:2]), t.held_by, t.inside, t.semantic_name, t.description)
                 for t in self.things]
        return World(items, dict(self.holding))

    def refresh(self) -> None:
        self.calls.append("refresh")

    @property
    def motions(self) -> list[str]:
        """Calls that act on objects (not the scan or social gestures)."""
        return [c for c in self.calls if c != "refresh" and c.split("(")[0] not in ("wave", "nod", "shake_head", "express", "scan", "home", "look_at")]

    def _wo(self, t: Thing) -> WorldObject:
        sx, sy, sz = t.size
        z = TABLE["top_z"] + sz / 2
        if t.on:
            ref = self.get(t.on)
            z += ref.size[2] if ref else 0.0
        return WorldObject(t.id, t.label, np.array([t.x, t.y, z]), np.array([max(sx, sy), min(sx, sy), sz]))

    def run(self, step: Step, cancelled: threading.Event) -> StepOutcome:
        self.calls.append(step.text())
        if cancelled.is_set():
            return StepOutcome(False, "CANCELLED", "cancelled")
        s = step.skill
        if s in ("scan", "home", "look_at", "wave", "nod", "shake_head"):
            return StepOutcome(True)
        if s == "express":
            self.motions_played.append(step.motion)
            return self.express_results.pop(0) if self.express_results else StepOutcome(True)
        if s == "release":
            for side, oid in self.holding.items():
                if oid:
                    self.get(oid).held_by = ""
                    self.holding[side] = ""
            return StepOutcome(True)
        if s in ("pick", "move_aside"):
            t = self.get(step.object)
            if t is None:
                return StepOutcome(False, "NOT_FOUND", f"no {step.object}")
            if t.label in ("bin", "tray"):
                return StepOutcome(False, "INVALID", "container")
            if t.id in self.unreachable:
                return StepOutcome(False, "UNREACHABLE", f"cannot reach {t.id}")
            if t.id in self.grasp_fails:
                self.grasp_fails.discard(t.id)
                return StepOutcome(False, "GRASP_FAILED", f"nothing grasped at {t.id}")
            if t.id not in self.holding.values():
                free = [h for h, v in self.holding.items() if not v]
                if not free:
                    return StepOutcome(False, "INVALID", "no free hand")
                side = "left" if (t.y >= 0 and "left" in free) or "right" not in free else "right"
                self.holding[side], t.held_by, t.inside, t.on = t.id, side, "", ""
            if s == "pick":
                return StepOutcome(True, "OK", f"holding {t.id}")
            side = t.held_by
            sol = self.resolver.free_spot(self._wo(t), [self._wo(o) for o in self.things if not o.held_by and not o.inside],
                                          (0.35, 0.12 if side == "left" else -0.12))
            return self._finish(t, sol)
        if s == "place":
            held = [v for v in self.holding.values() if v]
            if not held or (step.object and step.object not in held):
                return StepOutcome(False, "NOT_HOLDING", "not holding it")
            t = self.get(step.object or held[0])
            others = [self._wo(o) for o in self.things if not o.held_by and not o.inside and o.id != t.id]
            sol = self.resolver.resolve(step.relation, step.target, step.region, (0.0, 0.0), self._wo(t), others)
            return self._finish(t, sol, step)
        return StepOutcome(False, "INVALID", f"unknown skill {s}")

    def _finish(self, t: Thing, sol, step: Step | None = None) -> StepOutcome:
        if not sol.ok:
            return StepOutcome(False, sol.reason_code, sol.message)
        t.x, t.y = sol.candidates[0]
        self.holding[t.held_by] = ""
        t.held_by = ""
        t.inside = sol.container_id
        t.on = step.target if step is not None and step.relation == "on" and not sol.container_id else ""
        return StepOutcome(True, "OK", f"placed {t.id} {sol.message}")

    # ---------------------------------------------------------------- outcome checks
    def moved(self) -> set[str]:
        before = {t.id: (t.x, t.y) for t in self.initial}
        return {t.id for t in self.things if math.dist(before[t.id], (t.x, t.y)) > 1e-6 or t.held_by}

    def is_in(self, oid: str, container: str) -> bool:
        return self.get(oid).inside == container

    def is_on(self, oid: str, ref: str) -> bool:
        return self.get(oid).on == ref

    def side_of(self, oid: str, ref: str) -> str:
        a, b = self.get(oid), self.get(ref)
        dx, dy = a.x - b.x, a.y - b.y
        if abs(dy) >= abs(dx):
            return "left_of" if dy > 0 else "right_of"
        return "behind" if dx > 0 else "in_front_of"

    def at_region(self, oid: str, region: str, tol: float = 0.09) -> bool:
        t = self.get(oid)
        return not t.held_by and math.dist((t.x, t.y), regions(TABLE)[region]) < tol


def scene() -> list[Thing]:
    """Default regression scene: 7 objects + bin + tray, with deliberate duplicates (two red cubes)."""
    return [Thing("red_cube", "cube", "red", 0.40, 0.10), Thing("red_cube_2", "cube", "red", 0.50, -0.11),
            Thing("blue_cube", "cube", "blue", 0.36, -0.02), Thing("yellow_cube", "cube", "yellow", 0.47, 0.03),
            Thing("orange_ball", "ball", "orange", 0.34, 0.12), Thing("green_box", "box", "green", 0.52, 0.12),
            Thing("white_mug", "mug", "white", 0.40, -0.12), Thing("silver_bin", "bin", "silver", 0.46, -0.06),
            Thing("black_tray", "tray", "black", 0.35, 0.03)]


def semantic_scene() -> list[Thing]:
    """A scene whose objects only the camera model can name: geometry alone says cube/box/can."""
    return [Thing("red_cube", "cube", "red", 0.40, 0.10, semantic_name="toy block", description="a small wooden toy"),
            Thing("silver_can", "can", "silver", 0.50, -0.11, semantic_name="soda can", description="an aluminium drink"),
            Thing("green_box", "box", "green", 0.52, 0.12, semantic_name="juice carton", description="a drink carton"),
            Thing("white_mug", "mug", "white", 0.40, -0.12, semantic_name="coffee mug", description="a ceramic cup"),
            Thing("purple_block", "block", "purple", 0.36, -0.02, semantic_name="screwdriver", description="a hand tool"),
            Thing("orange_ball", "ball", "orange", 0.34, 0.12, semantic_name="tennis ball", description="a sports ball"),
            Thing("silver_bin", "bin", "silver", 0.46, -0.06, semantic_name="metal bin", description="a storage container")]
