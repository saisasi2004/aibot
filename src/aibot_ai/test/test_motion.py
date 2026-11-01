"""Skill motion planning against the real URDF: reachability, straight lines, safe joint moves,
head pointing. Needs xacro + the installed aibot_description (skipped otherwise)."""
from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

desc = pytest.importorskip("ament_index_python.packages")


@functools.lru_cache(maxsize=1)
def _setup():
    from ament_index_python.packages import get_package_share_directory
    from aibot_teleop.retarget.joint_map import JointMap
    from aibot_ai.motion.arm_motion import ArmMotion
    if shutil.which("xacro") is None:
        pytest.skip("xacro not found (source /opt/ros/jazzy/setup.bash)")
    d = Path(get_package_share_directory("aibot_description"))
    t = Path(get_package_share_directory("aibot_teleop"))
    g = Path(get_package_share_directory("aibot_gazebo"))
    urdf = subprocess.run(["xacro", str(d / "urdf" / "aibot.urdf.xacro")], capture_output=True, text=True, check=True).stdout
    jm = JointMap(str(t / "config" / "joint_mapping.yaml"), str(d / "config" / "joint_limits.yaml"), 0.02)
    table = yaml.safe_load((g / "config" / "objects.yaml").read_text())["table"]
    ik = yaml.safe_load((Path(get_package_share_directory("aibot_ai")) / "config" / "ai.yaml").read_text())[
        "skill_server_node"]["ros__parameters"]["ik"]
    return urdf, jm, table, ArmMotion(urdf, jm.names, jm.limits.lower, jm.limits.upper, table,
                                      waist_lower=np.array(ik["waist_lower"]), waist_upper=np.array(ik["waist_upper"]))


def test_place_area_reachable_by_some_arm():
    """Every point of the place area (grid, at grasp height) is reachable palm-down by >= 1 arm."""
    _, jm, table, m = _setup()
    q = np.zeros(len(jm.names))
    bad = []
    for x in np.linspace(table["place_x"][0], table["place_x"][1], 5):
        for y in np.linspace(table["place_y"][0], table["place_y"][1], 5):
            p = np.array([x, y, table["top_z"] + 0.025])
            if not any(m.reach(s, q, p).reachable for s in ("left", "right")):
                bad.append((round(x, 2), round(y, 2)))
    assert not bad, f"unreachable: {bad}"


def test_ik_respects_limits_and_no_hyperextension():
    _, jm, table, m = _setup()
    r = m.reach("left", np.zeros(len(jm.names)), np.array([0.44, 0.08, table["top_z"] + 0.03]))
    s = m.solvers["left"]
    assert r.reachable
    assert np.all(r.q >= s.lower - 1e-9) and np.all(r.q <= s.upper + 1e-9)
    assert r.q[6] <= 0.0                                     # elbow_pitch: no hyperextension
    p, _ = s.grasp_point(r.q)
    assert np.linalg.norm(p - [0.44, 0.08, table["top_z"] + 0.03]) < 0.01


def test_unreachable_far_target():
    _, jm, table, m = _setup()
    r = m.reach("left", np.zeros(len(jm.names)), np.array([0.95, 0.0, table["top_z"] + 0.03]))
    assert not r.reachable


def test_target_inside_table_is_a_collision():
    _, jm, table, m = _setup()
    r = m.reach("left", np.zeros(len(jm.names)), np.array([0.40, 0.05, table["top_z"] - 0.03]))
    assert not r.reachable


def test_direct_move_from_home_cuts_table_and_side_waypoint_fixes_it():
    _, jm, table, m = _setup()
    q0 = np.zeros(len(jm.names))
    r = m.reach("left", q0, np.array([0.40, 0.10, table["top_z"] + 0.12]))
    q1 = m.with_arm("left", q0, r.q)
    assert m.path_hit("left", q0, q1) == "table"
    wps, why = m.joint_move("left", q0, q1)
    assert wps is not None and len(wps) == 2, why
    assert m.path_hit("left", q0, wps[0]) == "" and m.path_hit("left", wps[0], wps[1]) == ""


@pytest.mark.parametrize("side,xy", [("left", (0.40, 0.08)), ("right", (0.44, -0.10)), ("left", (0.46, -0.04))])
def test_vertical_lines_are_straight_and_continuous(side, xy):
    _, jm, table, m = _setup()
    q0 = np.zeros(len(jm.names))
    pre = np.array([*xy, table["top_z"] + 0.125])
    grasp = np.array([*xy, table["top_z"] + 0.025])
    q_pre = m.with_arm(side, q0, m.reach(side, q0, pre).q)
    path, why = m.line(side, q_pre, grasp, allow_near=grasp)
    assert path is not None, why
    pts = np.array([m.grasp_point(side, q)[0] for q in path])
    off = pts[:, :2] - np.array(xy)
    assert np.max(np.linalg.norm(off, axis=1)) < 0.01           # stays on the vertical line
    assert np.all(np.diff(pts[:, 2]) < 1e-3)                    # monotonic descent
    assert np.max(np.abs(np.diff(np.array(path), axis=0))) < 0.25


def test_head_pointer_centres_far_targets_and_keeps_near_ones_in_view():
    """Neck pitch tops out at 0.52 rad: the optical axis then meets the table at x ~0.51, so only
    targets beyond that can be centred; nearer ones must still be inside the image."""
    from aibot_ai.motion.ik import HeadPointer
    urdf, jm, table, _ = _setup()
    neck = [jm.index["neck_yaw_joint"], jm.index["neck_pitch_joint"]]
    hp = HeadPointer(urdf, jm.limits.lower[neck], jm.limits.upper[neck])
    for tgt in ([0.56, 0.0, table["top_z"]], [0.58, 0.15, table["top_z"]], [0.55, -0.12, table["top_z"]]):
        n, err = hp.solve(np.array(tgt), np.zeros(3), np.array([0.0, 0.5]))
        assert err < np.radians(2), (tgt, n, np.degrees(err))
    for tgt in ([0.33, 0.0, table["top_z"]], [0.35, 0.15, table["top_z"]], [0.40, -0.15, table["top_z"]]):
        n, err = hp.solve(np.array(tgt), np.zeros(3), np.array([0.0, 0.5]))
        assert err < 0.35, (tgt, n, np.degrees(err))


def test_obstacles_block_reach_and_carried_object_hits_table():
    """A tall object right under the target makes the target a collision; a carried object
    hanging below the hand is checked against the table."""
    _, jm, table, m = _setup()
    q0 = np.zeros(len(jm.names))
    target = np.array([0.42, 0.05, table["top_z"] + 0.06])
    assert m.reach("left", q0, target).reachable
    try:
        m.set_scene([(0.42, 0.05, 0.04, table["top_z"] + 0.10)])          # a 10 cm can right there
        assert m.reach("left", q0, target).collision.startswith("object")
        m.set_scene([], {"left": (np.array([0.0, 0.0, 0.0]), 0.08)})        # carrying 16 cm tall thing
        r = m.reach("left", q0, np.array([0.42, 0.05, table["top_z"] + 0.04]))
        assert not r.reachable and r.collision == "table"
    finally:
        m.set_scene([])


def test_vertical_grasp_avoids_a_tall_neighbour_and_knows_the_limit():
    """A 12 cm can 11 cm (centre to centre) from the target: grasp-first planning finds an
    approach whose hand, fingers and thumb stay clear, and the descent is the reversed climb.
    8 cm is too close for this 20 cm hand with its 16 cm thumb: refused, not attempted."""
    _, jm, table, m = _setup()
    q0 = np.zeros(len(jm.names))
    grasp = np.array([0.42, 0.02, table["top_z"] + 0.025])
    try:
        m.set_scene([(0.42, 0.13, 0.033, table["top_z"] + 0.12)], ids=["silver_can"])
        q, paths = m.vertical("right", q0, grasp, (0.10, 0.12), allow_near=grasp)
        assert q is not None, paths
        s = m.solvers["right"]
        for path in paths:
            for qq in path:
                assert not s.obstacle_hits(s.arm_points(m.arm_q("right", qq)))
        top, _ = m.grasp_point("right", paths[0][-1])
        assert np.allclose(top, grasp + [0, 0, 0.10], atol=0.01)
        m.set_scene([(0.42, 0.10, 0.033, table["top_z"] + 0.12)], ids=["silver_can"])
        for side in ("left", "right"):
            assert m.vertical(side, q0, grasp, (0.10, 0.12), allow_near=grasp)[0] is None
    finally:
        m.set_scene([])


def test_cluttered_grasp_finds_a_clear_finger_direction():
    """Scene of seed 5000 (perceived): grasp the 10 cm box with a mug 12 cm away, a cube and the
    bin around it. Choosing the finger direction by clearance makes it reachable and clear."""
    _, jm, table, m = _setup()
    q0 = np.zeros(len(jm.names))
    top = table["top_z"]
    scene = [(0.539, -0.041, 0.038, 0.038, 0.0, top + 0.09),      # mug
             (0.328, -0.128, 0.025, 0.025, 0.0, top + 0.05),      # blue cube
             (0.372, 0.027, 0.065, 0.065, 0.0, top + 0.06),       # bin
             (0.517, 0.073, 0.03, 0.03, 0.0, top + 0.06)]         # ball
    grasp = np.array([0.438, -0.113, top + 0.075])
    try:
        m.set_scene(scene, ids=["mug", "cube", "bin", "ball"])
        q, paths = m.vertical("right", q0, grasp, (0.10, 0.12), allow_near=grasp)
        assert q is not None, paths
        wps, why = m.joint_move("right", q0, paths[0][-1])
        assert wps is not None, why
    finally:
        m.set_scene([])


def test_target_is_an_obstacle_for_the_transfer_but_not_the_descent():
    """Regression: the hand swept through the object it was approaching (cube pushed 5.7 cm)."""
    _, jm, table, m = _setup()
    q0 = np.zeros(len(jm.names))
    top = table["top_z"]
    box = (0.52, 0.055, 0.03, 0.03, 0.0, top + 0.06)
    grasp = np.array([0.52, 0.055, top + 0.06])
    m.set_scene([])
    legs, q_grasp, paths = m.approach("left", q0, grasp, (0.10, 0.12), allow_near=grasp, target=box, target_id="cube")
    assert legs is not None, paths
    s = m.solvers["left"]
    assert len(s.obstacles) == 0                                # the scene is restored afterwards
    s.obstacles, s.obstacle_ids = np.array([box]), ["cube"]
    try:
        q = q0
        for kind, data in legs:
            for w in data:
                if kind == "joint":
                    assert not m.path_hit("left", q, w)
                q = w
    finally:
        m.set_scene([])
