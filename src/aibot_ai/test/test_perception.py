"""Depth fusion + tabletop segmentation on ray-cast scenes seen from the robot's head camera pose."""
import math

import numpy as np
import pytest

from aibot_ai.perception.detectors import ColorBlobDetector, dominant_color, split_prompt
from aibot_ai.perception.fusion import Intrinsics, backproject_pixels, fit_table_plane, transform
from aibot_ai.perception.memory import ObjectMemory
from aibot_ai.perception.tabletop import TabletopConfig, segment_tabletop

from raycast import TABLE_Z, bin_prims, camera_pose, render

K = Intrinsics(fx=232.8, fy=232.8, cx=160.0, cy=120.0)     # the 640x480/69 deg camera at half resolution
T = camera_pose()


def run(prims):
    depth = render(prims, T, K)
    pc, uv = backproject_pixels(depth, K, stride=1)
    pw = transform(pc, T)
    plane, fitted = fit_table_plane(pw, TABLE_Z, np.random.default_rng(0))
    rgb = np.full((240, 320, 3), 128, np.uint8)
    segs = segment_tabletop(pw, uv, rgb, plane, TabletopConfig(), [])
    return segs, plane, fitted


def nearest(segs, xy):
    return min(segs, key=lambda s: np.linalg.norm(s.estimate.centroid[:2] - np.array(xy)))


def test_table_plane_is_fitted_at_the_known_height():
    _, plane, fitted = run([])
    assert fitted and abs(plane.z_at(0.4, 0.0) - TABLE_Z) < 0.002


@pytest.mark.parametrize("prim, label, size", [
    ({"type": "box", "center": (0.30, 0.08, TABLE_Z + 0.025), "size": (0.05, 0.05, 0.05), "yaw": 0.4}, "cube", (0.05, 0.05, 0.05)),
    ({"type": "box", "center": (0.32, -0.08, TABLE_Z + 0.05), "size": (0.04, 0.06, 0.10), "yaw": 0.2}, "box", (0.06, 0.04, 0.10)),
    ({"type": "box", "center": (0.38, 0.0, TABLE_Z + 0.0125), "size": (0.07, 0.05, 0.025), "yaw": -0.5}, "block", (0.07, 0.05, 0.025)),
    ({"type": "cylinder", "center": (0.35, 0.05, TABLE_Z + 0.06), "radius": 0.033, "length": 0.12}, "can", (0.066, 0.066, 0.12)),
    ({"type": "cylinder", "center": (0.33, -0.06, TABLE_Z + 0.045), "radius": 0.038, "length": 0.09}, "mug", (0.076, 0.076, 0.09)),
    ({"type": "sphere", "center": (0.30, 0.0, TABLE_Z + 0.03), "radius": 0.03}, "ball", None),
])
def test_single_objects(prim, label, size):
    segs, _, _ = run([prim])
    assert len(segs) == 1
    s = segs[0]
    true_xy = np.array(prim["center"][:2])
    assert np.linalg.norm(s.estimate.centroid[:2] - true_xy) < 0.008
    assert s.label == label
    if size is not None:
        assert np.allclose(s.estimate.size, size, atol=0.012)


def test_rectangle_yaw_and_grasp_across_minor_axis():
    prim = {"type": "box", "center": (0.32, 0.0, TABLE_Z + 0.05), "size": (0.04, 0.06, 0.10), "yaw": 0.3}
    s = run([prim])[0][0]
    major_dir = 0.3 + math.pi / 2              # the 6 cm side lies along the box's local y
    err = abs(math.remainder(s.estimate.yaw - major_dir, math.pi))
    assert err < math.radians(4)
    assert abs(math.remainder(s.estimate.grasp_yaw - 0.3, math.pi)) < math.radians(4)
    assert abs(s.estimate.grasp[2] - (TABLE_Z + 0.10)) < 0.005


def test_bin_and_neighbours_are_separated():
    prims = bin_prims(0.46, 0.0) + [{"type": "box", "center": (0.28, 0.10, TABLE_Z + 0.025), "size": (0.05, 0.05, 0.05)},
                                    {"type": "cylinder", "center": (0.30, -0.10, TABLE_Z + 0.06), "radius": 0.033, "length": 0.12}]
    segs, _, _ = run(prims)
    assert sorted(s.label for s in segs) == ["bin", "can", "cube"]
    assert np.linalg.norm(nearest(segs, (0.46, 0.0)).estimate.centroid[:2] - [0.46, 0.0]) < 0.01


def test_self_filter_removes_points_near_the_hand():
    prim = {"type": "box", "center": (0.30, 0.08, TABLE_Z + 0.025), "size": (0.05, 0.05, 0.05)}
    depth = render([prim], T, K)
    pc, uv = backproject_pixels(depth, K, stride=1)
    pw = transform(pc, T)
    plane, _ = fit_table_plane(pw, TABLE_Z, np.random.default_rng(0))
    segs = segment_tabletop(pw, uv, np.zeros((240, 320, 3), np.uint8), plane, TabletopConfig(), [np.array([0.30, 0.08, TABLE_Z + 0.06])])
    assert segs == []


def test_colour_naming_and_blob_detector():
    img = np.full((60, 80, 3), 120, np.uint8)
    img[10:40, 10:40] = (230, 40, 40)
    img[10:40, 45:75] = (40, 70, 230)
    assert dominant_color(img, (10, 10, 40, 40)) == "red" and dominant_color(img, (45, 10, 75, 40)) == "blue"
    dets = ColorBlobDetector(min_area=50).detect(img, ["red cube"])
    assert len(dets) == 1 and dets[0].color == "red"
    assert split_prompt("small red cube") == ("red", "small cube")


def test_memory_tracks_ids_and_expires():
    seg = run([{"type": "box", "center": (0.30, 0.08, TABLE_Z + 0.025), "size": (0.05, 0.05, 0.05)}])[0][0]
    mem = ObjectMemory(expiry=2.0)
    ids = mem.update([("cube", "red", seg.estimate, 1.0)], now=0.0)
    assert ids == ["red_cube_1"]
    assert mem.update([("cube", "red", seg.estimate, 1.0)], now=1.0) == ["red_cube_1"]
    assert mem.update([("cube", "blue", seg.estimate, 1.0)], now=1.5) == ["blue_cube_1"]   # colour gate
    mem.set_held("red_cube_1", "left")
    mem.expire(10.0)
    assert list(mem.objects) == ["red_cube_1"]                                       # held survives expiry


def test_memory_merges_label_flicker_of_one_object():
    """The same orange object seen as 'ball' twice and 'mug' once keeps one id, labelled ball."""
    from aibot_ai.perception.fusion import ObjectEstimate
    from aibot_ai.perception.memory import ObjectMemory

    def est(x):
        c = np.array([x, 0.07, 1.01])
        return ObjectEstimate(c, np.array([0.06, 0.06, 0.06]), 0.0, c[2] + 0.03, c + [0, 0, 0.03], 0.0, 500, 1.0)

    m = ObjectMemory()
    m.update([("ball", "orange", est(0.517), 1.0)], 0.0)
    m.update([("mug", "orange", est(0.518), 1.0)], 0.5)
    m.update([("ball", "orange", est(0.517), 1.0)], 1.0)
    assert list(m.objects) == ["orange_ball_1"] and m.objects["orange_ball_1"].label == "ball"
    m.update([("cube", "red", est(0.53), 1.0)], 1.5)          # a different colour is a different object
    assert len(m.objects) == 2


def test_memory_merges_overlapping_ghost_of_a_rolling_ball():
    """Seen at two spots 5 cm apart with different labels (beyond the 3 cm flicker gate): one object."""
    from aibot_ai.perception.fusion import ObjectEstimate
    from aibot_ai.perception.memory import ObjectMemory

    def est(x, y=0.07):
        c = np.array([x, y, 1.01])
        return ObjectEstimate(c, np.array([0.06, 0.06, 0.06]), 0.0, c[2] + 0.03, c + [0, 0, 0.03], 0.0, 500, 1.0)

    m = ObjectMemory()
    m.update([("ball", "orange", est(0.50), 1.0)], 0.0)
    m.update([("ball", "orange", est(0.50), 1.0)], 0.5)
    m.update([("box", "orange", est(0.54), 1.0)], 1.0)     # 4 cm on: within 6 cm gate but a new label
    assert len(m.objects) == 1 and "orange_ball_1" in m.objects
    assert m.objects["orange_ball_1"].position[0] >= 0.50
    m.update([("ball", "orange", est(0.70), 1.0)], 1.5)    # a second ball far away stays separate
    assert len(m.objects) == 2
