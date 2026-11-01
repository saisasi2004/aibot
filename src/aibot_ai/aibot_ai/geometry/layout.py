"""Random, non-overlapping tabletop layouts for /aibot/scene/reset (pure, unit-tested).

Footprints are approximated by their circumscribed circle (box: half the footprint diagonal,
cylinder/sphere: radius, bin/tray: half diagonal). Conservative but rotation-free, so any random
yaw is safe. Placement is rejection sampling inside the placement region from objects.yaml,
shrunk so the whole footprint stays `edge_margin` inside the table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Placement:
    slot: str
    x: float
    y: float
    z: float            # model origin height for resting on the table
    yaw: float


def footprint_radius(model: dict) -> float:
    shape = model["shape"]
    if shape in ("box", "bin", "tray"):
        sx, sy = model["size"][:2]
        return 0.5 * math.hypot(sx, sy)
    if shape in ("cylinder", "sphere"):
        return float(model["radius"])
    raise ValueError(f"unknown shape {shape}")


def resting_height(model: dict) -> float:
    """Model origin above the table surface when it rests upright."""
    shape = model["shape"]
    if shape == "box":
        return model["size"][2] / 2
    if shape == "cylinder":
        return model["length"] / 2
    if shape == "sphere":
        return model["radius"]
    return 0.0                    # bin / tray origin is on the bottom face


def sample_layout(slots: dict[str, str], models: dict[str, dict], table: dict, rng: np.random.Generator,
                  max_tries: int = 200, restarts: int = 300) -> list[Placement]:
    """Place every slot in `slots` (slot -> model). Largest first; a dead end restarts the layout."""
    for _ in range(restarts):
        try:
            return _sample_once(slots, models, table, rng, max_tries)
        except RuntimeError:
            continue
    raise RuntimeError(f"no non-overlapping layout for {len(slots)} items after {restarts} restarts")


def _sample_once(slots: dict[str, str], models: dict[str, dict], table: dict, rng: np.random.Generator,
                 max_tries: int) -> list[Placement]:
    order = sorted(slots, key=lambda s: -footprint_radius(models[slots[s]]))
    placed: list[tuple[float, float, float]] = []      # x, y, r
    out: list[Placement] = []
    tx, ty, margin, gap = table["x"], table["y"], table["edge_margin"], table["clearance"]
    px, py = table["place_x"], table["place_y"]
    for slot in order:
        m = models[slots[slot]]
        r = footprint_radius(m)
        lo_x, hi_x = max(px[0], tx[0] + margin + r), min(px[1], tx[1] - margin - r)
        lo_y, hi_y = max(py[0], ty[0] + margin + r), min(py[1], ty[1] - margin - r)
        if lo_x > hi_x or lo_y > hi_y:
            raise ValueError(f"{slot} does not fit in the placement region")
        for _ in range(max_tries):
            x, y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            if all(math.hypot(x - ox, y - oy) >= r + orr + gap for ox, oy, orr in placed):
                break
        else:
            raise RuntimeError(f"no free spot for {slot} after {max_tries} tries")
        placed.append((x, y, r))
        yaw = 0.0 if m["shape"] in ("sphere", "cylinder") else float(rng.uniform(-math.pi, math.pi))
        out.append(Placement(slot, x, y, table["top_z"] + resting_height(m) + 0.001, yaw))
    return out


def choose_slots(graspable: dict[str, str], count: int, types: list[str], rng: np.random.Generator) -> list[str]:
    """Pick `count` graspable slots, optionally only of the given slot names or model types.

    Exact slot names are taken first ("cube_small" is both a slot and a model: naming it means
    that slot), the rest is drawn from slots whose model is named, so a request for
    [cube_small, cube_blue, ball] with count 3 gives exactly those three."""
    pool = sorted(graspable)
    if not types:
        if count > len(pool):
            raise ValueError(f"asked for {count} objects but only {len(pool)} slots exist")
        return sorted(rng.choice(pool, size=count, replace=False).tolist())
    exact = [s for s in pool if s in types]
    by_model = [s for s in pool if s not in exact and graspable[s] in types]
    if count > len(exact) + len(by_model):
        raise ValueError(f"asked for {count} objects but only {len(exact) + len(by_model)} slots match {types}")
    if count <= len(exact):
        return sorted(rng.choice(exact, size=count, replace=False).tolist())
    extra = rng.choice(by_model, size=count - len(exact), replace=False).tolist() if count > len(exact) else []
    return sorted(exact + extra)
