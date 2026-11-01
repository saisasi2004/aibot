"""MediaPipe hand landmarks -> per-finger curl (0..1) -> STAR1's 12 hand joints, plus grasp state.

Curl of a finger = sum of its three bend angles (at MCP, PIP, DIP), each measured between
consecutive bone vectors, normalised between an open-hand and a fist reference:
    curl = clip((sum - OPEN) / (FIST - OPEN), 0, 1)
Sums of angles are invariant to hand rotation and to MediaPipe's per-hand scale, and robust to
its depth noise because each bone contributes one angle.

STAR1 hand joints (controller order) and their signs, verified by FK of the URDF: positive
values curl towards the palm on both hands; positive index_bend adducts the index finger
towards the middle finger.
    thumb_bend (opposition), thumb_rota_joint1, thumb_rota_joint2 (thumb flexion),
    index_bend (spread), {index, mid, ring, pinky}_joint1 (MCP), _joint2 (PIP)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FINGERS: dict[str, tuple[int, int, int, int]] = {   # MCP, PIP, DIP, TIP
    "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
THUMB = (1, 2, 3, 4)                                  # CMC, MCP, IP, TIP
WRIST = 0

OPEN_SUM_DEG, FIST_SUM_DEG = 25.0, 230.0      # finger bend sums: flat hand ~10-20, tight fist ~250
THUMB_OPEN_DEG, THUMB_FLEX_DEG = 10.0, 95.0   # thumb MCP+IP bend
OPPOSE_FAR, OPPOSE_NEAR = 1.45, 0.60          # |thumb tip - pinky MCP| / palm length


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


@dataclass
class HandShape:
    curl: dict[str, float]   # index, middle, ring, pinky, thumb: 0 open .. 1 closed
    opposition: float        # 0 thumb out to the side .. 1 thumb across the palm
    spread: float            # rad, angle between index and middle proximal bones

    @property
    def closure(self) -> float:
        """Overall hand closure used for grasp detection."""
        fingers = np.mean([self.curl[k] for k in FINGERS])
        return float(0.8 * fingers + 0.2 * self.curl["thumb"])


def hand_shape(points: np.ndarray) -> HandShape:
    p = np.asarray(points, dtype=float)
    curl: dict[str, float] = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        bend = (_angle(p[mcp] - p[WRIST], p[pip] - p[mcp]) + _angle(p[pip] - p[mcp], p[dip] - p[pip])
                + _angle(p[dip] - p[pip], p[tip] - p[dip]))
        curl[name] = float(np.clip((math.degrees(bend) - OPEN_SUM_DEG) / (FIST_SUM_DEG - OPEN_SUM_DEG), 0.0, 1.0))
    cmc, mcp, ip, tip = THUMB
    thumb_bend = _angle(p[mcp] - p[cmc], p[ip] - p[mcp]) + _angle(p[ip] - p[mcp], p[tip] - p[ip])
    curl["thumb"] = float(np.clip((math.degrees(thumb_bend) - THUMB_OPEN_DEG) / (THUMB_FLEX_DEG - THUMB_OPEN_DEG), 0.0, 1.0))
    palm = max(float(np.linalg.norm(p[9] - p[WRIST])), 1e-6)
    d = float(np.linalg.norm(p[tip] - p[17])) / palm
    opposition = float(np.clip((OPPOSE_FAR - d) / (OPPOSE_FAR - OPPOSE_NEAR), 0.0, 1.0))
    spread = _angle(p[6] - p[5], p[10] - p[9])
    return HandShape(curl, opposition, spread)


@dataclass
class HandMapping:
    """Scale from normalised hand shape to STAR1 joint angles (rad)."""

    mcp_max: float = 1.45            # {finger}_joint1 at curl 1 (limit 1.92)
    pip_max: float = 1.60            # {finger}_joint2 at curl 1 (limit 1.92)
    thumb_bend_max: float = 1.20     # opposition 1 (limit 1.83)
    thumb_rota1_max: float = 0.90    # thumb flexion 1 (limit 1.57)
    thumb_rota2_max: float = 1.20    # (limit 1.83)
    spread_neutral: float = 0.15     # rad of index/middle angle that maps to index_bend 0
    spread_gain: float = 1.0
    spread_limit: float = 0.17       # index_bend range is +-10 deg

    def targets(self, s: HandShape) -> np.ndarray:
        """12 joint targets in hand-controller order."""
        bend = float(np.clip(-(s.spread - self.spread_neutral) * self.spread_gain, -self.spread_limit, self.spread_limit))
        out = [s.opposition * self.thumb_bend_max,
               s.curl["thumb"] * self.thumb_rota1_max,
               s.curl["thumb"] * self.thumb_rota2_max,
               bend]
        for finger in ("index", "middle", "ring", "pinky"):
            out += [s.curl[finger] * self.mcp_max, s.curl[finger] * self.pip_max]
        return np.array(out)


class GraspDetector:
    """Binary grasp command from hand closure, with hysteresis and a short confirmation delay.

    The state only flips after `confirm_frames` consecutive frames beyond the opposite threshold,
    so a single noisy frame can neither drop an object nor grab one.
    """

    def __init__(self, close_threshold: float = 0.6, open_threshold: float = 0.4, confirm_frames: int = 3) -> None:
        if open_threshold >= close_threshold:
            raise ValueError("open_threshold must be below close_threshold")
        self.close_threshold, self.open_threshold, self.confirm_frames = close_threshold, open_threshold, confirm_frames
        self.closed = False
        self._count = 0

    def update(self, closure: float) -> bool:
        flip = closure < self.open_threshold if self.closed else closure > self.close_threshold
        self._count = self._count + 1 if flip else 0
        if self._count >= self.confirm_frames:
            self.closed = not self.closed
            self._count = 0
        return self.closed
