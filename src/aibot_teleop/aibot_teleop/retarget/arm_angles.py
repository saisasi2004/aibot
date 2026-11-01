"""Direct angle mapping: operator arm vectors -> the robot's 7-DOF arm joint angles.

The robot arm, both sides: every joint frame is aligned with the torso at q = 0, where the arm hangs
along -z. The chain is

    shoulder_pitch (y) -> shoulder_roll (x) -> arm_yaw (z) -> elbow_pitch (y)
    -> elbow_yaw (z) -> wrist_pitch (y) -> wrist_roll (x)

so, ignoring the 1-3 cm link offsets,

    upper arm   u = Ry(sp) Rx(sr) (-z)          = (-sin sp cos sr,  sin sr,  -cos sp cos sr)
    forearm     f = Ry(sp) Rx(sr) Rz(ay) Ry(ep) (-z)
    hand frame  H = Ry(sp) Rx(sr) Rz(ay) Ry(ep) Rz(ey) Ry(wp) Rx(wr)

and the angles are read back one stage at a time:
    sr = asin(u_y),  sp = atan2(-u_x, -u_z)
    f' = (Ry(sp) Rx(sr))^T f = (sin(flex) cos ay, sin(flex) sin ay, -cos(flex)),  ep = -flex
    (Ry Rx Rz Ry)^T H = Rz(ey) Ry(wp) Rx(wr)   (ZYX Euler)
The human elbow only flexes one way, which on the robot is negative elbow_pitch (forearm forward).

Two singularities matter in practice:
  * arm straight out sideways (the T-pose!): u_x = u_z = 0 leaves sp undefined;
  * elbow straight: ay is undefined (it is the same rotation as elbow_yaw).
Near them the new value is blended with the previous one. The later stages are always computed
from the angles actually chosen, so the forearm and hand still come out right.
"""
from __future__ import annotations

import math

import numpy as np

from ..utils.filters import unwrap_towards
from .body_frame import ELBOW, INDEX, PINKY, SHOULDER, WRIST, BodySample, unit

ARM_JOINTS = ("shoulder_pitch_joint", "shoulder_roll_joint", "arm_yaw_joint", "elbow_pitch_joint",
              "elbow_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint")


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _blend(prev: float, new: float, weight: float) -> float:
    return prev + weight * (float(unwrap_towards(new, prev)) - prev)


def arm_forward(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction model above: (upper arm dir, forearm dir, hand rotation) for 7 arm angles."""
    sp, sr, ay, ep, ey, wp, wr = q
    r_sh = rot_y(sp) @ rot_x(sr)
    r_fore = r_sh @ rot_z(ay) @ rot_y(ep)
    down = np.array([0.0, 0.0, -1.0])
    return r_sh @ down, r_fore @ down, r_fore @ rot_z(ey) @ rot_y(wp) @ rot_x(wr)


def solve_arm(upper: np.ndarray, fore: np.ndarray, hand_rot: np.ndarray | None,
              prev: np.ndarray) -> np.ndarray:
    """7 arm angles [sp, sr, ay, ep, ey, wp, wr] from directions in the chest/torso frame.

    hand_rot=None keeps the previous wrist angles (hand not tracked).
    """
    q = np.array(prev, dtype=float)
    u, f = unit(np.asarray(upper, float)), unit(np.asarray(fore, float))

    sr = math.asin(float(np.clip(u[1], -1.0, 1.0)))
    horizontal = math.hypot(u[0], u[2])                      # = cos(sr)
    sp = _blend(prev[0], math.atan2(-u[0], -u[2]), float(np.clip((horizontal - 0.05) / 0.20, 0.0, 1.0)))

    r_sh = rot_y(sp) @ rot_x(sr)
    fl = r_sh.T @ f
    flex = math.acos(float(np.clip(-fl[2], -1.0, 1.0)))
    ep = -flex
    # With the elbow straight, arm_yaw is unobservable (same axis as elbow_yaw). Instead of
    # holding a possibly wound-up value, let it relax towards 0; elbow_yaw below absorbs the
    # difference, so the hand orientation stays exact and a later elbow bend causes no twist jump.
    w_ay = float(np.clip((math.sin(flex) - 0.10) / 0.25, 0.0, 1.0))
    ay = _blend(prev[2] * (1.0 - 0.08 * (1.0 - w_ay)), math.atan2(fl[1], fl[0]), w_ay)
    q[:4] = sp, sr, ay, ep

    if hand_rot is not None:
        rw = (r_sh @ rot_z(ay) @ rot_y(ep)).T @ hand_rot
        wp = math.asin(float(np.clip(-rw[2, 0], -1.0, 1.0)))
        ey = float(unwrap_towards(math.atan2(rw[1, 0], rw[0, 0]), prev[4]))
        wr = float(unwrap_towards(math.atan2(rw[2, 1], rw[2, 2]), prev[6]))
        q[4:] = ey, wp, wr
    return q


def hand_frame(points: np.ndarray) -> np.ndarray | None:
    """Rotation of the operator's hand from 21 MediaPipe hand points (body axes).

    Built to match the robot's hand frame at q = 0 (arm hanging, palm facing the thigh):
      -z along the fingers (wrist -> middle MCP), +x towards the thumb side (pinky MCP -> index
      MCP). Both cues are side-independent, so the same code serves both hands and mirror mode.
    """
    d = unit(points[9] - points[0])
    t = points[5] - points[17]
    t = unit(t - np.dot(t, d) * d)
    if np.linalg.norm(d) < 0.5 or np.linalg.norm(t) < 0.5:
        return None
    z = -d
    return np.column_stack([t, np.cross(z, t), z])


def hand_frame_from_pose(sample: BodySample, side: str, threshold: float = 0.6) -> np.ndarray | None:
    """Coarse hand frame from BlazePose's own wrist/index/pinky knuckle points (fallback)."""
    w, i, p = WRIST[side], INDEX[side], PINKY[side]
    if not sample.visible(w, i, p, threshold=threshold):
        return None
    pts = sample.points
    fake = np.zeros((21, 3))
    fake[0], fake[5], fake[17] = pts[w], pts[i], pts[p]
    fake[9] = 0.5 * (pts[i] + pts[p])
    return hand_frame(fake)


def arm_vectors(sample: BodySample, side: str, r_chest: np.ndarray,
                threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray] | None:
    """Upper-arm and forearm vectors (metres) of one side in the chest frame, or None."""
    s, e, w = SHOULDER[side], ELBOW[side], WRIST[side]
    if not sample.visible(s, e, w, threshold=threshold):
        return None
    p = sample.points
    return r_chest.T @ (p[e] - p[s]), r_chest.T @ (p[w] - p[e])
