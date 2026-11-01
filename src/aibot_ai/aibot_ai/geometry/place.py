"""Deterministic resolution of a PlaceTarget (relation + reference) to a spot on the table.

The planner (LLM) says *where in words* ("left_of red_cube", "in bin", "at_region front");
this module turns that into coordinates using the perceived world state only. No randomness:
the same scene and target always give the same candidates, in the same order.

Conventions (robot's point of view, world frame: the robot faces +x):
  left = +y, right = -y, in_front_of = -x (between the object and the robot), behind = +x.
  Regions are laid out on the place area: front = near edge (small x), back = far edge.

The result is an ordered candidate list; the skill server takes the first one the IK can reach.
Reason codes follow skills/result.py: NOT_FOUND (unknown reference), INVALID (bad relation or
region, or "on" something without a flat top), TOO_LARGE (does not fit in the container),
PLACE_BLOCKED (no free spot), UNREACHABLE (outside the place area).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CONTAINER_LABELS = ("bin", "tray")
ROUND_LABELS = ("ball", "can", "mug")
FLAT_TOP_LABELS = ("cube", "box", "block", "can", "tray")
RELATIONS = ("on", "in", "left_of", "right_of", "in_front_of", "behind", "at_region", "at_xy")
DIRECTIONS = {"left_of": (0.0, 1.0), "right_of": (0.0, -1.0), "in_front_of": (-1.0, 0.0), "behind": (1.0, 0.0)}
REGION_NAMES = ("center", "left", "right", "front", "back", "front_left", "front_right", "back_left", "back_right")


@dataclass
class WorldObject:
    """What the place logic needs of a perceived object (from DetectedObject)."""
    id: str
    label: str
    position: np.ndarray       # centroid, world (m)
    size: np.ndarray           # [major, minor, height] (m)
    yaw: float = 0.0

    @property
    def radius(self) -> float:
        """Footprint radius: half the diameter for round things, half the diagonal otherwise."""
        major, minor = float(self.size[0]), float(self.size[1])
        return 0.5 * major if self.label in ROUND_LABELS else 0.5 * math.hypot(major, minor)

    def gap_to(self, x: float, y: float) -> float:
        """Distance from a point to this object's footprint: its rotated rectangle, or a circle for
        round things. The circumscribed circle made a 13 cm bin block 15 cm all around it."""
        dx, dy = x - float(self.position[0]), y - float(self.position[1])
        if self.label in ROUND_LABELS:
            return max(0.0, math.hypot(dx, dy) - 0.5 * float(self.size[0]))
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
        return math.hypot(max(0.0, abs(lx) - 0.5 * float(self.size[0])), max(0.0, abs(ly) - 0.5 * float(self.size[1])))

    @property
    def half_extent(self) -> float:
        """Half the footprint's longest side: how far the object reaches along a principal axis."""
        return 0.5 * float(self.size[0])

    @property
    def top_z(self) -> float:
        return float(self.position[2] + 0.5 * self.size[2])

    @property
    def is_container(self) -> bool:
        return self.label in CONTAINER_LABELS


@dataclass
class PlaceSolution:
    ok: bool
    reason_code: str = "OK"
    message: str = ""
    candidates: list[tuple[float, float]] = field(default_factory=list)   # (x, y), best first
    surface_z: float = 0.0           # height the object's bottom rests on (or is dropped from: rim)
    container_id: str = ""


def regions(table: dict, inset: float = 0.03) -> dict[str, tuple[float, float]]:
    (x0, x1), (y0, y1) = table["place_x"], table["place_y"]
    xs = {"front": x0 + inset, "": 0.5 * (x0 + x1), "back": x1 - inset}
    ys = {"left": y1 - inset, "": 0.5 * (y0 + y1), "right": y0 + inset}
    out = {}
    for xn, x in xs.items():
        for yn, y in ys.items():
            out["_".join(n for n in (xn, yn) if n) or "center"] = (x, y)
    return out


class PlaceResolver:
    def __init__(self, table: dict, step: float = 0.02, max_extra: float = 0.12, gap: float = 0.005) -> None:
        """step/max_extra: how far the search slides away from the nominal spot; gap: extra air
        between the placed object and its reference beyond the placing clearance. The arm's own
        collision check still decides whether a free spot can actually be reached."""
        self.table, self.step, self.max_extra, self.gap = table, step, max_extra, gap
        self.clearance = float(table.get("place_clearance", table.get("clearance", 0.025)))
        self.regions = regions(table)

    def in_area(self, x: float, y: float) -> bool:
        (x0, x1), (y0, y1) = self.table.get("reach_x", self.table["place_x"]), self.table.get("reach_y", self.table["place_y"])
        return x0 <= x <= x1 and y0 <= y <= y1

    def is_free(self, x: float, y: float, radius: float, others: list[WorldObject]) -> bool:
        return all(o.gap_to(x, y) >= radius + self.clearance for o in others)

    def _search(self, nominal: tuple[float, float], radius: float, others: list[WorldObject],
                direction: tuple[float, float] | None) -> list[tuple[float, float]]:
        """Free, in-area spots near `nominal`: slide along `direction` (relations) or spiral (regions)."""
        n = int(round(self.max_extra / self.step))
        offsets: list[tuple[float, float]] = []
        if direction is not None:
            dx, dy = direction
            px, py = -dy, dx                                   # perpendicular
            # never closer than the nominal distance along the direction; sideways as far as along,
            # ordered by total offset, straight out first on ties: still "left of", just shifted
            grid = sorted(((k, lat) for k in range(n + 1) for lat in range(-n, n + 1)),
                          key=lambda kl: (kl[0] + abs(kl[1]), abs(kl[1]), kl[1] < 0))
            for k, lat in grid:
                offsets.append((k * self.step * dx + lat * self.step * px, k * self.step * dy + lat * self.step * py))
        else:
            for k in range(n + 1):
                ring = [(0.0, 0.0)] if k == 0 else [
                    (k * self.step * math.cos(a), k * self.step * math.sin(a)) for a in np.arange(0, 2 * math.pi, math.pi / 4)]
                offsets += ring
        out = []
        for ox, oy in offsets:
            x, y = nominal[0] + ox, nominal[1] + oy
            if self.in_area(x, y) and self.is_free(x, y, radius, others):
                out.append((round(x, 4), round(y, 4)))
        return out

    def resolve(self, relation: str, reference_id: str, region: str, xy: tuple[float, float],
                held: WorldObject, objects: list[WorldObject]) -> PlaceSolution:
        table_z = float(self.table["top_z"])
        others = [o for o in objects if o.id != held.id]
        by_id = {o.id: o for o in others}
        r = held.radius
        if relation not in RELATIONS:
            return PlaceSolution(False, "INVALID", f"unknown relation '{relation}'; use one of {', '.join(RELATIONS)}")

        if relation in ("on", "in", *DIRECTIONS):
            ref = by_id.get(reference_id)
            if ref is None:
                return PlaceSolution(False, "NOT_FOUND", f"reference '{reference_id}' is not in the world state")
            rest = [o for o in others if o.id != ref.id]
            if relation == "on" and ref.is_container:
                relation = "in"                                # "on the tray" means into it
            if relation == "in":
                if not ref.is_container:
                    return PlaceSolution(False, "INVALID", f"{ref.id} is a {ref.label}, not a container")
                inner = float(min(ref.size[0], ref.size[1])) - 0.012           # minus the walls
                if 2 * r > inner + 0.01 and float(max(held.size[0], held.size[1])) > inner:
                    return PlaceSolution(False, "TOO_LARGE", f"{held.id} ({held.size[0]*100:.0f} cm) does not fit in {ref.id}")
                return PlaceSolution(True, "OK", f"into {ref.id}", [(float(ref.position[0]), float(ref.position[1]))],
                                     ref.top_z, ref.id)
            if relation == "on":
                if ref.label not in FLAT_TOP_LABELS:
                    return PlaceSolution(False, "INVALID", f"{ref.id} ({ref.label}) has no flat top to stack on")
                if r > ref.radius + 0.03:
                    return PlaceSolution(False, "INVALID", f"{held.id} is too wide to balance on {ref.id}")
                return PlaceSolution(True, "OK", f"on {ref.id}", [(float(ref.position[0]), float(ref.position[1]))], ref.top_z)
            d = DIRECTIONS[relation]
            dist = r + ref.half_extent + self.clearance + self.gap
            nominal = (float(ref.position[0]) + d[0] * dist, float(ref.position[1]) + d[1] * dist)
            cands = self._search(nominal, r, others, d)
            if cands:
                return PlaceSolution(True, "OK", f"{relation} {ref.id}", cands, table_z)
            reason = "PLACE_BLOCKED" if self.in_area(*nominal) else "UNREACHABLE"
            return PlaceSolution(False, reason, f"no free spot {relation.replace('_', ' ')} {ref.id}"
                                 + ("" if reason == "PLACE_BLOCKED" else " inside the reachable place area"))

        if relation == "at_region":
            if region not in self.regions:
                return PlaceSolution(False, "INVALID", f"unknown region '{region}'; use one of {', '.join(REGION_NAMES)}")
            cands = self._search(self.regions[region], r, others, None)
            if cands:
                return PlaceSolution(True, "OK", f"at the {region.replace('_', ' ')} of the table", cands, table_z)
            return PlaceSolution(False, "PLACE_BLOCKED", f"the {region} region is full")

        x, y = xy                                              # at_xy
        if not self.in_area(x, y):
            return PlaceSolution(False, "UNREACHABLE", f"({x:.2f}, {y:.2f}) is outside the place area")
        if not self.is_free(x, y, r, others):
            return PlaceSolution(False, "PLACE_BLOCKED", f"({x:.2f}, {y:.2f}) is occupied")
        return PlaceSolution(True, "OK", f"at ({x:.2f}, {y:.2f})", [(x, y)], table_z)

    def free_spot(self, held: WorldObject, objects: list[WorldObject], prefer: tuple[float, float]) -> PlaceSolution:
        """move_aside: free spots ordered by distance from `prefer` (the side of the holding arm)."""
        others = [o for o in objects if o.id != held.id]
        (x0, x1), (y0, y1) = self.table.get("reach_x", self.table["place_x"]), self.table.get("reach_y", self.table["place_y"])
        grid = [(float(x), float(y)) for x in np.arange(x0, x1 + 1e-9, self.step) for y in np.arange(y0, y1 + 1e-9, self.step)]
        grid = [p for p in grid if self.is_free(p[0], p[1], held.radius, others)
                and math.hypot(p[0] - held.position[0], p[1] - held.position[1]) > 2 * held.radius + self.clearance]
        grid.sort(key=lambda p: (math.hypot(p[0] - prefer[0], p[1] - prefer[1]), p))
        if not grid:
            return PlaceSolution(False, "PLACE_BLOCKED", "no free spot on the table")
        return PlaceSolution(True, "OK", "aside", grid, float(self.table["top_z"]))
