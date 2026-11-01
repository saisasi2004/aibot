"""PlaceTarget resolution (geometry/place.py) and description matching (skills/matching.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from aibot_ai.geometry.place import PlaceResolver, WorldObject, regions
from aibot_ai.skills.matching import match

TABLE = {"top_z": 0.98, "x": [0.20, 0.60], "y": [-0.18, 0.18], "place_x": [0.32, 0.54], "place_y": [-0.15, 0.15],
         "edge_margin": 0.02, "clearance": 0.025, "place_clearance": 0.015}


def obj(id_, label, x, y, size=(0.05, 0.05, 0.05)):
    return WorldObject(id_, label, np.array([x, y, TABLE["top_z"] + size[2] / 2]), np.array(size, float))


HELD = obj("red_cube", "cube", 0.0, 0.0)
SCENE = [obj("blue_cube", "cube", 0.42, 0.0, (0.04, 0.04, 0.04)),
         obj("silver_bin", "bin", 0.48, -0.08, (0.13, 0.13, 0.06)),
         obj("orange_ball", "ball", 0.38, 0.10, (0.06, 0.06, 0.06)),
         HELD]
R = PlaceResolver(TABLE)


def _free(x, y, others, r=HELD.radius):
    return all(o.gap_to(x, y) >= r + TABLE["place_clearance"] - 1e-9 for o in others)


def test_on_targets_reference_top():
    s = R.resolve("on", "blue_cube", "", (0, 0), HELD, SCENE)
    assert s.ok and s.candidates == [(0.42, 0.0)]
    assert s.surface_z == pytest.approx(TABLE["top_z"] + 0.04)


def test_on_a_ball_is_invalid_and_on_a_container_means_in():
    assert R.resolve("on", "orange_ball", "", (0, 0), HELD, SCENE).reason_code == "INVALID"
    s = R.resolve("on", "silver_bin", "", (0, 0), HELD, SCENE)
    assert s.ok and s.container_id == "silver_bin"


def test_in_container_and_not_in_a_cube():
    s = R.resolve("in", "silver_bin", "", (0, 0), HELD, SCENE)
    assert s.ok and s.candidates == [(0.48, -0.08)] and s.surface_z == pytest.approx(TABLE["top_z"] + 0.06)
    assert R.resolve("in", "blue_cube", "", (0, 0), HELD, SCENE).reason_code == "INVALID"


def test_too_large_for_container():
    big = obj("box", "box", 0, 0, (0.14, 0.14, 0.05))
    tray = obj("tray", "tray", 0.45, 0.0, (0.12, 0.12, 0.012))
    assert R.resolve("in", "tray", "", (0, 0), big, [tray]).reason_code == "TOO_LARGE"


@pytest.mark.parametrize("rel,axis,sign", [("left_of", 1, 1), ("right_of", 1, -1), ("in_front_of", 0, -1), ("behind", 0, 1)])
def test_directional_relations(rel, axis, sign):
    ref = obj("blue_cube", "cube", 0.43, 0.0, (0.04, 0.04, 0.04))
    s = R.resolve(rel, "blue_cube", "", (0, 0), HELD, [ref, HELD])
    assert s.ok
    x, y = s.candidates[0]
    delta = (np.array([x, y]) - ref.position[:2])[axis]
    assert sign * delta > HELD.radius + ref.radius                          # really on that side
    assert abs((np.array([x, y]) - ref.position[:2])[1 - axis]) < 1e-9      # straight out first
    assert all(_free(cx, cy, [ref]) and R.in_area(cx, cy) for cx, cy in s.candidates)


def test_left_of_slides_past_an_obstacle():
    ref = obj("blue_cube", "cube", 0.43, 0.0, (0.04, 0.04, 0.04))
    blocker = obj("yellow_cube", "cube", 0.40, 0.09, (0.04, 0.04, 0.04))
    s = R.resolve("left_of", "blue_cube", "", (0, 0), HELD, [ref, blocker, HELD])
    assert s.ok
    assert all(_free(x, y, [ref, blocker]) for x, y in s.candidates)
    assert all(y >= ref.position[1] + HELD.radius + ref.radius for _x, y in s.candidates)


def test_left_of_at_the_edge_is_unreachable():
    ref = obj("blue_cube", "cube", 0.43, 0.15, (0.04, 0.04, 0.04))
    s = R.resolve("left_of", "blue_cube", "", (0, 0), HELD, [ref, HELD])
    assert s.ok or s.reason_code == "UNREACHABLE"
    if s.ok:                              # only sideways candidates may remain, all in the area
        assert all(R.in_area(*c) for c in s.candidates)


def test_regions_layout_and_resolution():
    reg = regions(TABLE)
    assert reg["left"][1] > reg["center"][1] > reg["right"][1]
    assert reg["front"][0] < reg["center"][0] < reg["back"][0]
    assert reg["front_left"][0] == reg["front"][0] and reg["front_left"][1] == reg["left"][1]
    s = R.resolve("at_region", "", "front_right", (0, 0), HELD, SCENE)
    assert s.ok and all(_free(x, y, SCENE[:-1]) for x, y in s.candidates)
    assert math.dist(s.candidates[0], reg["front_right"]) < 0.03
    assert R.resolve("at_region", "", "kitchen", (0, 0), HELD, SCENE).reason_code == "INVALID"


def test_at_xy_checks_area_and_occupancy():
    assert R.resolve("at_xy", "", "", (0.50, 0.12), HELD, SCENE).ok
    assert R.resolve("at_xy", "", "", (0.40, -0.10), HELD, SCENE).reason_code == "PLACE_BLOCKED"   # bin rim
    assert R.resolve("at_xy", "", "", (0.70, 0.0), HELD, SCENE).reason_code == "UNREACHABLE"
    assert R.resolve("at_xy", "", "", (0.42, 0.01), HELD, SCENE).reason_code == "PLACE_BLOCKED"


def test_unknown_reference_and_relation():
    assert R.resolve("left_of", "ghost_1", "", (0, 0), HELD, SCENE).reason_code == "NOT_FOUND"
    assert R.resolve("under", "blue_cube", "", (0, 0), HELD, SCENE).reason_code == "INVALID"


def test_full_table_is_blocked():
    grid = [obj(f"c{i}_{j}", "cube", x, y, (0.06, 0.06, 0.06))
            for i, x in enumerate(np.arange(0.32, 0.55, 0.05)) for j, y in enumerate(np.arange(-0.15, 0.16, 0.05))]
    assert R.resolve("at_region", "", "center", (0, 0), HELD, grid + [HELD]).reason_code == "PLACE_BLOCKED"
    assert R.free_spot(HELD, grid + [HELD], (0.4, 0.1)).reason_code == "PLACE_BLOCKED"


def test_deterministic():
    a = R.resolve("at_region", "", "back_left", (0, 0), HELD, SCENE).candidates
    b = R.resolve("at_region", "", "back_left", (0, 0), HELD, list(reversed(SCENE))).candidates
    assert a == b


def test_free_spot_prefers_side_and_moves_away():
    held = obj("red_cube", "cube", 0.42, 0.0)
    s = R.free_spot(held, SCENE[:-1] + [held], (0.36, 0.12))
    assert s.ok
    d = [math.dist(c, (0.36, 0.12)) for c in s.candidates]
    assert d == sorted(d)                                     # nearest free spot to the preferred side first
    others = SCENE[:-1]
    assert all(_free(x, y, others) and math.dist((x, y), (0.42, 0.0)) > 0.07 for x, y in s.candidates)


# ---------------------------------------------------------------- matching
@dataclass
class P:
    id: str
    label: str
    color: str
    position: tuple


OBJS = [P("red_cube", "cube", "red", (0.40, 0.10, 1.0)), P("red_cube_2", "cube", "red", (0.45, -0.10, 1.0)),
        P("blue_cube", "cube", "blue", (0.35, 0.0, 1.0)), P("orange_ball", "ball", "orange", (0.5, 0.05, 1.0)),
        P("silver_bin", "bin", "silver", (0.48, -0.08, 1.0))]


@pytest.mark.parametrize("text,ids", [
    ("red_cube_2", ["red_cube_2"]),
    ("the blue cube", ["blue_cube"]),
    ("red cube", ["red_cube", "red_cube_2"]),            # ambiguous: both returned
    ("the leftmost red cube", ["red_cube"]),
    ("the red cube on the right", ["red_cube_2"]),
    ("closest cube", ["blue_cube"]),
    ("sphere", ["orange_ball"]),
    ("the basket", ["silver_bin"]),
    ("green cube", []),
    ("banana", []),
])
def test_match(text, ids):
    assert [o.id for o in match(text, OBJS)] == ids


def test_footprints_are_rectangles_not_circumscribed_circles():
    b = obj("b", "bin", 0.45, 0.0, (0.13, 0.13, 0.06))
    assert b.gap_to(0.45, 0.10) == pytest.approx(0.035)        # 3.5 cm beyond the bin's side
    assert b.gap_to(0.45, 0.0) == 0.0
    rot = WorldObject("r", "box", np.array([0.4, 0.0, 1.0]), np.array([0.2, 0.02, 0.1]), math.pi / 2)
    assert rot.gap_to(0.4, 0.09) == 0.0 and rot.gap_to(0.42, 0.0) == pytest.approx(0.01)   # long side along y
    assert obj("m", "mug", 0.4, 0.0, (0.076, 0.076, 0.09)).gap_to(0.5, 0.0) == pytest.approx(0.062)


def test_realistic_scene_has_room_for_regions_and_relations():
    """5 objects like a real scene (logged "region is full" and "no free spot" every time)."""
    held = obj("green_box", "box", 0, 0, (0.06, 0.04, 0.10))
    world = [obj("bin", "bin", 0.37, 0.03, (0.13, 0.13, 0.06)), obj("mug", "mug", 0.54, -0.04, (0.076, 0.076, 0.09)),
             obj("cube", "cube", 0.33, -0.13, (0.04, 0.04, 0.04)), obj("ball", "ball", 0.52, 0.07, (0.06, 0.06, 0.06))]
    free = [r for r in regions(TABLE) if R.resolve("at_region", "", r, (0, 0), held, world + [held]).ok]
    assert len(free) >= 7, free
    assert R.resolve("right_of", "mug", "", (0, 0), held, world + [held]).ok
    assert R.resolve("behind", "cube", "", (0, 0), held, world + [held]).ok
