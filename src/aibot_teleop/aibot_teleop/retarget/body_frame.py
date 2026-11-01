"""Coordinate conventions: MediaPipe camera axes -> robot-style body axes, mirroring, chest frame.

Body axes (same convention as the STAR1 URDF):  x = forward, y = operator's left, z = up.

MediaPipe world landmarks are camera-aligned: x = image right, y = image down, z = away from
the camera. The operator faces the camera, so their forward is towards the camera (-z), their
left appears on the image right (+x) and up is -y:

    body = MP_TO_BODY @ mp,    MP_TO_BODY = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]   (det = +1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MP_TO_BODY = np.array([[0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])

# BlazePose landmark indices (odd = operator's left).
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8
SHOULDER = {"left": 11, "right": 12}
ELBOW = {"left": 13, "right": 14}
WRIST = {"left": 15, "right": 16}
PINKY = {"left": 17, "right": 18}
INDEX = {"left": 19, "right": 20}
THUMB = {"left": 21, "right": 22}
HIP = {"left": 23, "right": 24}

# Pairs swapped when mirroring (left <-> right counterparts).
_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
                 (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MIRROR_INDEX = np.arange(33)
for _a, _b in _MIRROR_PAIRS:
    MIRROR_INDEX[_a], MIRROR_INDEX[_b] = _b, _a

REFLECT_Y = np.diag([1.0, -1.0, 1.0])
SIDES = ("left", "right")


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


@dataclass
class BodySample:
    """One frame of operator pose in body axes (after optional mirroring)."""

    points: np.ndarray        # (33, 3) metres, hip-centred, body axes
    visibility: np.ndarray    # (33,)
    hands: dict[str, np.ndarray | None]   # side -> (21, 3) hand world points in body axes, or None
    stamp: float              # seconds (capture time)

    def visible(self, *indices: int, threshold: float = 0.5) -> bool:
        return bool(np.all(self.visibility[list(indices)] >= threshold))


def to_body_axes(points_mp: np.ndarray) -> np.ndarray:
    return np.asarray(points_mp, dtype=float) @ MP_TO_BODY.T


def make_sample(pose_world_mp: np.ndarray, visibility: np.ndarray,
                hands_world_mp: dict[str, np.ndarray | None], stamp: float, mirror: bool) -> BodySample:
    """Convert MediaPipe outputs to a BodySample.

    mirror=True turns the operator into their own mirror image before retargeting: points are
    reflected across the sagittal plane (y -> -y) and left/right landmarks and hands swap. All
    downstream code is then identical for both modes: the robot's left arm follows whatever is
    the "left" arm of this sample.
    """
    pts = to_body_axes(pose_world_mp)
    vis = np.asarray(visibility, dtype=float)
    hands = {s: (None if h is None else to_body_axes(h)) for s, h in hands_world_mp.items()}
    if mirror:
        pts = pts[MIRROR_INDEX] @ REFLECT_Y
        vis = vis[MIRROR_INDEX]
        hands = {"left": None if hands.get("right") is None else hands["right"] @ REFLECT_Y,
                 "right": None if hands.get("left") is None else hands["left"] @ REFLECT_Y}
    return BodySample(pts, vis, hands, stamp)


def chest_frame(sample: BodySample, use_hips: bool) -> np.ndarray | None:
    """Rotation (columns x fwd, y left, z up) of the operator's chest in body axes.

    y follows the shoulder line. With use_hips, z follows the spine (hip midpoint -> shoulder
    midpoint), so leaning forward/sideways is measured. Sitting at a laptop the hips are out of
    view, and z falls back to the camera's vertical: chest pitch/roll then read 0, while yaw
    and the arms still work. The caller decides use_hips with hysteresis on hip visibility, so
    the frame does not jump between the two definitions.
    """
    if not sample.visible(SHOULDER["left"], SHOULDER["right"], threshold=0.5):
        return None
    p = sample.points
    y = unit(p[SHOULDER["left"]] - p[SHOULDER["right"]])
    if use_hips:
        up = 0.5 * (p[SHOULDER["left"]] + p[SHOULDER["right"]]) - 0.5 * (p[HIP["left"]] + p[HIP["right"]])
    else:
        up = np.array([0.0, 0.0, 1.0])
    z = unit(up - np.dot(up, y) * y)
    if np.linalg.norm(z) < 0.5:
        return None
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (projection of the arithmetic mean onto SO(3))."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # opposite: rotate pi about any axis orthogonal to a
        axis = unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)
