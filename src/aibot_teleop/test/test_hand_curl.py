"""Finger curl, robot hand mapping and grasp hysteresis on synthetic MediaPipe hands."""
import math

import numpy as np

from aibot_teleop.retarget.hand_curl import FINGERS, GraspDetector, HandMapping, hand_shape

# MCP positions of a right-sized palm (metres), fingers pointing -z when straight.
MCP = {"index": [0.025, 0, -0.085], "middle": [0.003, 0, -0.09], "ring": [-0.017, 0, -0.087],
       "pinky": [-0.035, 0, -0.08]}
BONES = (0.045, 0.027, 0.024)


def synthetic_hand(bend_deg: float, thumb_bend_deg: float = 0.0, thumb_across: bool = False) -> np.ndarray:
    """21 points; each finger joint bent by bend_deg towards the palm (-y here)."""
    p = np.zeros((21, 3))
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        pos = np.array(MCP[name], dtype=float)
        p[mcp] = pos
        heading = 0.0
        for idx, length in zip((pip, dip, tip), BONES):
            heading += math.radians(bend_deg)
            pos = pos + length * np.array([0.0, -math.sin(heading), -math.cos(heading)])
            p[idx] = pos
    base = np.array([0.03, -0.01, -0.03]) if not thumb_across else np.array([0.02, -0.03, -0.04])
    direction = np.array([1.0, 0.0, -0.4]) if not thumb_across else np.array([-0.6, -0.6, -0.5])
    direction /= np.linalg.norm(direction)
    towards_palm = np.array([0, -1.0, 0]) - np.dot([0, -1.0, 0], direction) * direction
    towards_palm /= np.linalg.norm(towards_palm)       # bend axis exactly perpendicular to the bone
    p[1] = base
    pos, heading = base.copy(), 0.0
    for idx, length in zip((2, 3, 4), (0.035, 0.03, 0.025)):
        if idx > 2:
            heading += math.radians(thumb_bend_deg)
        pos = pos + length * (direction * math.cos(heading) + towards_palm * math.sin(heading))
        p[idx] = pos
    return p


def test_open_hand_is_open():
    s = hand_shape(synthetic_hand(3.0))
    assert all(s.curl[f] < 0.05 for f in FINGERS)
    assert s.closure < 0.1


def test_fist_is_closed():
    s = hand_shape(synthetic_hand(85.0, thumb_bend_deg=50.0, thumb_across=True))
    assert all(s.curl[f] > 0.9 for f in FINGERS)
    assert s.curl["thumb"] > 0.8
    assert s.closure > 0.85


def test_curl_is_rotation_invariant():
    pts = synthetic_hand(45.0)
    c, s = math.cos(0.7), math.sin(0.7)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    a, b = hand_shape(pts), hand_shape(pts @ r.T * 1.3)        # rotated and scaled
    assert all(abs(a.curl[f] - b.curl[f]) < 1e-9 for f in FINGERS)


def test_mapping_stays_inside_joint_limits():
    m = HandMapping()
    for bend in (0.0, 45.0, 90.0):
        q = m.targets(hand_shape(synthetic_hand(bend, 60.0, True)))
        assert q.shape == (12,)
        assert np.all(q[[0, 1, 2]] <= [1.83, 1.57, 1.83]) and abs(q[3]) <= 0.175
        assert np.all(q[4:] >= 0.0) and np.all(q[4:] <= 1.92)


def test_grasp_hysteresis_and_confirmation():
    g = GraspDetector(close_threshold=0.6, open_threshold=0.4, confirm_frames=3)
    assert not any(g.update(x) for x in (0.7, 0.7))      # needs 3 frames
    assert g.update(0.7)
    assert g.update(0.5)                                   # in the band: stays closed
    assert g.update(0.3) and g.update(0.3)                 # 2 open frames: still closed
    assert g.update(0.65)                                  # interruption resets the count
    assert g.update(0.3) and g.update(0.3) and not g.update(0.3)
