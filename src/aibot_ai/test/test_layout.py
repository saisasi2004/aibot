"""Scene layouts: non-overlap, bounds, determinism, slot list consistent with sim.launch.py."""
import ast
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from aibot_ai.geometry.layout import choose_slots, footprint_radius, sample_layout

GAZEBO = Path(__file__).resolve().parents[2] / "aibot_gazebo"
CFG = yaml.safe_load((GAZEBO / "config" / "objects.yaml").read_text())


def test_slots_match_grasp_objects_in_sim_launch():
    tree = ast.parse((GAZEBO / "launch" / "sim.launch.py").read_text())
    values = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("GRASP_OBJECTS", "CONTAINERS")}
    assert sorted(values["GRASP_OBJECTS"]) == sorted(CFG["slots"]["graspable"])
    assert sorted(values["CONTAINERS"]) == sorted(CFG["slots"]["containers"])


def test_pool_size_and_hand_limits():
    models = CFG["models"]
    graspable = {m for m in CFG["slots"]["graspable"].values()}
    assert len(graspable) >= 8
    lim = CFG["hand_limits"]
    for name in graspable:
        m = models[name]
        width = min(m["size"][:2]) if m["shape"] == "box" else 2 * m["radius"]
        height = m["size"][2] if m["shape"] == "box" else m.get("length", 2 * m.get("radius", 0))
        assert lim["min_width"] <= width <= lim["max_width"], name
        assert height <= lim["max_height"] and m["mass"] <= lim["max_mass"], name


@pytest.mark.parametrize("seed", range(50))
def test_layouts_do_not_overlap_and_stay_on_table(seed):
    rng = np.random.default_rng(seed)
    slots = choose_slots(CFG["slots"]["graspable"], 5, [], rng)
    chosen = {s: CFG["slots"]["graspable"][s] for s in slots} | {"bin": CFG["slots"]["containers"]["bin"]}
    out = sample_layout(chosen, CFG["models"], CFG["table"], rng)
    t = CFG["table"]
    for p in out:
        r = footprint_radius(CFG["models"][chosen[p.slot]])
        assert t["x"][0] + t["edge_margin"] <= p.x - r and p.x + r <= t["x"][1] - t["edge_margin"]
        assert t["y"][0] + t["edge_margin"] <= p.y - r and p.y + r <= t["y"][1] - t["edge_margin"]
    for a, b in itertools.combinations(out, 2):
        ra, rb = footprint_radius(CFG["models"][chosen[a.slot]]), footprint_radius(CFG["models"][chosen[b.slot]])
        assert math.hypot(a.x - b.x, a.y - b.y) >= ra + rb + t["clearance"] - 1e-9


def test_deterministic_and_type_filter():
    a = sample_layout({"cube_small": "cube_small"}, CFG["models"], CFG["table"], np.random.default_rng(3))
    b = sample_layout({"cube_small": "cube_small"}, CFG["models"], CFG["table"], np.random.default_rng(3))
    assert a == b
    cubes = choose_slots(CFG["slots"]["graspable"], 2, ["cube_small"], np.random.default_rng(0))
    assert sorted(cubes) == ["cube_small", "cube_small_2"]
    with pytest.raises(ValueError):
        choose_slots(CFG["slots"]["graspable"], 3, ["cube_small"], np.random.default_rng(0))
    for seed in range(20):                     # exact slot names win over the same-named model
        picked = choose_slots(CFG["slots"]["graspable"], 3, ["cube_small", "cube_blue", "ball"], np.random.default_rng(seed))
        assert picked == ["ball", "cube_blue", "cube_small"]


def test_full_table_always_has_a_layout():
    """Every object at once, the default scene: the sampler must find room for any seed."""
    slots = {**CFG["slots"]["graspable"], **CFG["slots"]["containers"]}
    for seed in range(30):
        assert len(sample_layout(slots, CFG["models"], CFG["table"], np.random.default_rng(seed))) == len(slots)
