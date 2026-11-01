"""Torso (waist yaw/roll/pitch) and head (neck yaw/pitch) targets from body landmarks.

Waist. After the pelvis re-rooting in aibot_description, STAR1's torso orientation relative to
the pelvis is
    R_torso = Rz(-q_yaw) Rx(-q_roll) Ry(-q_pitch)       (joint axes are -z, -x, -y)
The operator's chest orientation relative to their calibrated neutral posture is decomposed in
the same ZXY order, R = Rz(a) Rx(b) Ry(c), giving q_yaw = -a, q_roll = -b, q_pitch = -c.
Leaning forward (c > 0) therefore gives q_pitch < 0, matching the joint range -60..+5 deg.

Head. The facing direction is the vector from the ear midpoint to the nose, expressed in the
chest frame. Its yaw and elevation relative to the calibrated neutral drive neck_yaw (+ = look
left) and neck_pitch (+ = look down).
"""
from __future__ import annotations

import math

import numpy as np

from .body_frame import LEFT_EAR, NOSE, RIGHT_EAR, BodySample


def waist_targets(r_chest: np.ndarray, r_neutral: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """[waist_yaw, waist_roll, waist_pitch] joint targets (controller order)."""
    r = r_neutral.T @ r_chest
    b = math.asin(float(np.clip(r[2, 1], -1.0, 1.0)))
    a = math.atan2(-r[0, 1], r[1, 1])
    c = math.atan2(-r[2, 0], r[2, 2])
    return -gain * np.array([a, b, c])


def head_direction(sample: BodySample, r_chest: np.ndarray, threshold: float = 0.5) -> tuple[float, float] | None:
    """(yaw, elevation) in rad of the nose relative to the ear midpoint, in the chest frame."""
    vis = sample.visibility
    if vis[NOSE] < threshold or max(vis[LEFT_EAR], vis[RIGHT_EAR]) < 0.3:
        return None
    p = sample.points
    h = r_chest.T @ (p[NOSE] - 0.5 * (p[LEFT_EAR] + p[RIGHT_EAR]))
    if np.linalg.norm(h) < 1e-3:
        return None
    return math.atan2(h[1], h[0]), math.atan2(h[2], math.hypot(h[0], h[1]))


def head_targets(direction: tuple[float, float], neutral: tuple[float, float], gain: float = 1.0) -> np.ndarray:
    """[neck_yaw, neck_pitch] joint targets (controller order)."""
    yaw = math.remainder(direction[0] - neutral[0], 2.0 * math.pi)
    return gain * np.array([yaw, -(direction[1] - neutral[1])])
