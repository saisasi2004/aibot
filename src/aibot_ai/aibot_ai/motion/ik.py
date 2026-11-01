"""Skill IK: put a hand's grasp point at a target with the palm down, plus a reachability check.

Chain: world -> waist (yaw, roll, pitch) -> 7 arm joints -> {side}_hand_link (10 DOF). STAR1's
arm is only 0.40 m shoulder-to-wrist and its shoulders are 0.35 m above the table, so the back
of the table is out of reach without leaning; the waist joins in, held back by a posture task
(waist roll is kept near 0, yaw/pitch are used only when the arm alone cannot reach).

Tasks of the damped-least-squares solve (stacked, weighted):
  * grasp point position  (the point the grasp manager measures, `grasp_offset` in hand_link)
  * palm direction        palm normal pointing down (axis constraint only: the fingers may
                          point wherever the reach is easiest, the weld grasp does not need a
                          particular finger direction; a fully fixed hand orientation made most
                          of the table unreachable for this short arm)
  * posture               towards a rest pose (seed), per-joint weights
Joint limits (with margin) are enforced by clipping every iteration; the elbow may not
hyperextend (elbow_pitch <= 0), like a human arm.

Reachability = converged within tolerance AND collision-free against simple primitives:
the table slab, the torso box (base_link frame) and the other arm's segments, sampled every
2 cm along shoulder-elbow-wrist-hand-fingertip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aibot_teleop.kinematics import ArmChain

GRASP_OFFSET = {"left": np.array([0.0, -0.035, -0.10]), "right": np.array([0.0, 0.035, -0.10])}
FINGERTIP = {"left": np.array([0.0, -0.006, -0.20]), "right": np.array([0.0, 0.006, -0.20])}


def _hand_points(sign: float) -> np.ndarray:
    """Collision samples of an open hand in hand_link coordinates (from the URDF finger chains and
    the palm mesh): palm face, the finger plate (x -0.036..0.026, to 20.6 cm) and the thumb,
    which sticks out 15.7 cm sideways along +x near the wrist. A single centre line missed the
    thumb, and the thumb swept a mug off the table. sign: -1 left hand, +1 right (mirror in y)."""
    pts = []
    for z in np.arange(-0.01, -0.12, -0.025):                      # palm face (palm-normal side)
        for x in (-0.045, 0.0, 0.04):
            pts.append((x, sign * 0.024, z))
    for z in np.arange(-0.12, -0.207, -0.025):                     # fingers
        for x in (-0.036, -0.005, 0.026):
            pts.append((x, sign * 0.006, z))
    for u in np.linspace(0.0, 1.0, 6):                             # thumb
        pts.append((0.023 + u * 0.134, sign * (0.010 + u * 0.012), -0.03))
    return np.array(pts)


HAND_POINTS = {"left": _hand_points(-1.0), "right": _hand_points(1.0)}
# Torso box in base_link: mesh bounds minus the shoulder caps (|y| > 0.15 carry the arm mounts).
TORSO_BOX = (np.array([-0.12, -0.15, -0.11]), np.array([0.12, 0.15, 0.33]))
# Palm-down "ready" seed (waist 3 + left arm 7), from an offline solve with the grasp point over
# the front-left of the table. The right arm is its mirror image: roll (x) and yaw (z) joints
# change sign, pitch (y) joints do not.
READY_SEED_LEFT = np.array([0.0, 0.0, 0.0, 0.27, -0.21, -0.53, -1.46, -1.76, 0.54, 0.27])
MIRROR_SIGN = np.array([-1, -1, 1, 1, -1, -1, 1, -1, 1, -1], float)   # waist yaw/roll flip too
ELBOW, WRIST = 6, 8          # indices of elbow_pitch / wrist_pitch among the 10 active joints


PALM_NORMAL = {"left": np.array([0.0, -1.0, 0.0]), "right": np.array([0.0, 1.0, 0.0])}   # hand_link frame
DOWN = np.array([0.0, 0.0, -1.0])


def palm_down_rotation(side: str, finger_yaw: float) -> np.ndarray:
    """Hand-link rotation with the palm facing -z (world) and the fingers along finger_yaw.

    Hand frame (both hands): fingers along -z_hand, thumb side +x_hand; palm normal -y_hand on the
    left hand, +y_hand on the right (verified from the URDF in Phase 1).
    """
    f = np.array([math.cos(finger_yaw), math.sin(finger_yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    z_h = -f
    y_h = up if side == "left" else -up          # palm normal points down
    x_h = np.cross(y_h, z_h)
    return np.column_stack([x_h, y_h, z_h])


@dataclass
class IKResult:
    q: np.ndarray            # 10 joints: waist yaw/roll/pitch + 7 arm
    pos_err: float
    rot_err: float
    collision: str           # "" if free, else what it hits
    reachable: bool


class ReachSolver:
    def __init__(self, urdf: str, side: str, lower: np.ndarray, upper: np.ndarray, table: dict,
                 iterations: int = 120, damping: float = 0.02, w_rot: float = 0.3,
                 posture_weights: np.ndarray | None = None, pos_tol: float = 0.01,
                 palm_tilt_tol: float = math.radians(45)) -> None:
        self.side, self.other = side, ("right" if side == "left" else "left")
        self.chain = ArmChain(urdf, "world", f"{side}_hand_link")
        self.other_chain = ArmChain(urdf, "world", f"{self.other}_hand_link")
        self.torso_chain = ArmChain(urdf, "world", "base_link")
        self.names = self.chain.names
        upper = upper.copy()
        upper[ELBOW] = min(upper[ELBOW], 0.0)     # no elbow hyperextension
        self.lower, self.upper = lower, upper
        self.iterations, self.damping, self.w_rot = iterations, damping, w_rot
        # waist roll held hard, yaw/pitch soft, arm very soft
        self.w_post = posture_weights if posture_weights is not None else np.array([0.02, 1.0, 0.02] + [0.01] * 7)
        self.pos_tol, self.tilt_tol = pos_tol, palm_tilt_tol
        self.give_up_error = 0.05          # m still off at half the iterations: abandon the seed
        self.table = table
        # Perceived objects to keep clear of: oriented boxes, rows (x, y, half_major, half_minor,
        # yaw, top_z), world (perception's footprint rectangle). Set per motion.
        self.obstacles = np.zeros((0, 6))
        self.obstacle_ids: list[str] = []
        self.ignore: set[int] = set()          # obstacle rows not checked (already in contact)
        self.obstacle_margin = 0.015
        # Object in this hand: (centre offset in the hand frame, half height); collision-checked
        # with the arm, so a carried object does not sweep through others.
        self.attached: tuple[np.ndarray, float] | None = None

    # ------------------------------------------------------------------ kinematics
    def grasp_point(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, p = self.chain.tip_pose(q)
        return p + r @ GRASP_OFFSET[self.side], r

    def seeds(self, extra: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """Warm starts: given ones first, then the ready pose and variants with the forearm
        rotated (hand approaching from the side) and the waist leaning."""
        out = []
        # forearm (arm yaw) and elbow-yaw variants turn the palm-down hand's finger direction,
        # so a neighbour in the way of one finger direction is avoided by another
        for d_ay, d_ey, lean in ((0, 0, 0), (0.8, 0, 0), (-0.8, 0, 0), (0.0, 1.0, 0), (1.6, 0, 0), (-1.6, 0, 0),
                                 (0.8, 1.0, 0), (-0.8, 1.0, 0), (0, 0, -0.3), (0.8, 0, -0.3), (-0.8, 0, -0.3)):
            q = READY_SEED_LEFT.copy()
            q[5] += d_ay
            q[7] += d_ey
            q[2] += lean
            out.append(q if self.side == "left" else q * MIRROR_SIGN)
        return [np.asarray(e, float) for e in (extra or [])] + out + [np.zeros(10)]

    def palm_tilt(self, r_tip: np.ndarray) -> float:
        """Angle between the palm normal and straight down."""
        return float(math.acos(float(np.clip(r_tip @ PALM_NORMAL[self.side] @ DOWN, -1.0, 1.0))))

    def solve_best(self, target: np.ndarray, palm_down: bool, seeds: list[np.ndarray],
                   rest: np.ndarray | None = None, other_arm: np.ndarray | None = None) -> IKResult:
        """Multi-start: first reachable solution in seed order, else the one with the least error."""
        best = None
        for sd in seeds:
            r = self.solve(target, palm_down, sd, rest, other_arm)
            if r.reachable:
                return r
            if best is None or r.pos_err + 0.05 * r.rot_err < best.pos_err + 0.05 * best.rot_err:
                best = r
        return best

    def solve(self, target: np.ndarray, palm_down: bool, seed: np.ndarray,
              rest: np.ndarray | None = None, other_arm: np.ndarray | None = None,
              finger_dir: np.ndarray | None = None, w_finger: float = 0.15) -> IKResult:
        """finger_dir: optional horizontal unit vector the fingers should point along (a soft task
        under the palm-down one), used to steer the hand clear of neighbouring objects."""
        q = np.clip(np.array(seed, float), self.lower, self.upper)
        rest = q.copy() if rest is None else np.asarray(rest, float)
        n = len(q)
        for it in range(self.iterations):
            pos, axes, r_tip = self.chain.forward(q)
            _, p_tip = self.chain.tip_pose(q)
            point = p_tip + r_tip @ GRASP_OFFSET[self.side]
            jp = np.cross(axes, point - pos).T             # all joints at once (was the hot spot)
            rows, errs = [jp], [target - point]
            if palm_down:   # align the palm normal with -z; rotation about that axis left free
                n_now = r_tip @ PALM_NORMAL[self.side]
                rows.append(self.w_rot * axes.T)
                errs.append(self.w_rot * np.cross(n_now, DOWN))
                if finger_dir is not None:     # fingers run along -z of the hand frame
                    rows.append(w_finger * axes.T)
                    errs.append(w_finger * np.cross(-r_tip[:, 2], finger_dir))
            rows.append(np.diag(self.w_post))
            errs.append(self.w_post * (rest - q))
            jac, err = np.vstack(rows), np.concatenate(errs)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping ** 2 * np.eye(len(err)), err)
            q = np.clip(q + dq, self.lower, self.upper)
            e = float(np.linalg.norm(errs[0]))
            if e < 2e-4:
                break
            if it == self.iterations // 2 and e > self.give_up_error:
                break                                      # this seed is not converging: save the time
        point, r_tip = self.grasp_point(q)
        pe = float(np.linalg.norm(point - target))
        re = self.palm_tilt(r_tip) if palm_down else 0.0
        hit = ""
        if pe <= self.pos_tol and re <= self.tilt_tol:
            hit = self.collision(q, other_arm, allow_near=target)
        return IKResult(q, pe, re, hit, pe <= self.pos_tol and re <= self.tilt_tol and not hit)

    # ------------------------------------------------------------------ collisions
    def _segment_points(self, chain: ArmChain, q: np.ndarray, side: str) -> np.ndarray:
        """Forearm (elbow -> wrist -> hand, every 2 cm; the upper arm is attached at the shoulder)
        followed by the hand samples (HAND_POINTS) in world coordinates."""
        pos, _, r = chain.forward(q)
        _, p_tip = chain.tip_pose(q)
        knots = [pos[ELBOW], pos[WRIST], p_tip]
        pts = []
        for a, b in zip(knots[:-1], knots[1:]):
            k = max(2, int(np.linalg.norm(b - a) / 0.02) + 1)
            pts.append(a + np.linspace(0, 1, k)[:, None] * (b - a))
        pts.append(p_tip + HAND_POINTS[side] @ r.T)
        return np.vstack(pts)

    def attached_points(self, q: np.ndarray) -> np.ndarray:
        """Centre and bottom of the carried object (world), or none."""
        if self.attached is None:
            return np.zeros((0, 3))
        offset, half_h = self.attached
        point, r = self.grasp_point(q)
        c = point + r @ offset
        return np.array([c, c - [0.0, 0.0, half_h]])

    def obstacle_hits(self, pts: np.ndarray) -> list[int]:
        """Rows of self.obstacles that any point is inside (oriented box + margin)."""
        if not len(self.obstacles) or not len(pts):
            return []
        ob, m = self.obstacles, self.obstacle_margin
        dx = pts[:, None, 0] - ob[None, :, 0]
        dy = pts[:, None, 1] - ob[None, :, 1]
        c, s_ = np.cos(ob[None, :, 4]), np.sin(ob[None, :, 4])
        lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
        inside = (np.abs(lx) < ob[None, :, 2] + m) & (np.abs(ly) < ob[None, :, 3] + m) & (pts[:, None, 2] < ob[None, :, 5] + m)
        return [int(i) for i in np.nonzero(np.any(inside, axis=0))[0] if int(i) not in self.ignore]

    def arm_points(self, q: np.ndarray) -> np.ndarray:
        return np.vstack([self._segment_points(self.chain, q, self.side), self.attached_points(q)])

    def collision(self, q: np.ndarray, other_arm: np.ndarray | None, allow_near: np.ndarray | None = None) -> str:
        """'' if collision-free, else 'table' | 'torso' | 'other_arm' | 'object'.

        q: this arm's 10 joints (waist + arm). other_arm: the other arm's 7 joints (None: hanging
        at zero); it shares the waist of q.
        allow_near: a point (e.g. the object being grasped) around which table contact by the
        fingertips is ignored (they curl down beside it).
        """
        pts = self._segment_points(self.chain, q, self.side)
        t = self.table
        over = (pts[:, 0] > t["x"][0]) & (pts[:, 0] < t["x"][1]) & (pts[:, 1] > t["y"][0]) & (pts[:, 1] < t["y"][1])
        low = pts[:, 2] < t["top_z"] + 0.005
        hit_table = over & low
        if allow_near is not None:
            hit_table &= np.linalg.norm(pts[:, :2] - allow_near[:2], axis=1) > 0.06
        carried = self.attached_points(q)
        if np.any(hit_table) or (len(carried) and np.any(carried[:, 2] < t["top_z"] + 0.003)):
            return "table"
        hits = self.obstacle_hits(np.vstack([pts, carried]))
        if hits:
            return f"object {self.obstacle_ids[hits[0]] if hits[0] < len(self.obstacle_ids) else hits[0]}"
        r_t, p_t = self.torso_chain.tip_pose(q[:3])
        n_upper = max(2, int(np.linalg.norm(pts[1] - pts[0]) / 0.02) + 1)   # skip the upper arm/elbow
        local = (pts[n_upper:] - p_t) @ r_t
        lo, hi = TORSO_BOX
        if np.any(np.all((local > lo) & (local < hi), axis=1)):
            return "torso"
        oq = np.concatenate([q[:3], np.zeros(7) if other_arm is None else other_arm])
        other = self._segment_points(self.other_chain, oq, self.other)
        if np.min(np.linalg.norm(pts[:, None, :] - other[None, :, :], axis=2)) < 0.07:
            return "other_arm"
        return ""


class HeadPointer:
    """Neck yaw/pitch that put a world point on the head camera's optical axis (look_at).

    Chain world -> waist (3, held at their current values) -> neck yaw, pitch -> optical frame;
    optical z is the viewing direction. Two joints, damped least squares on the axis error.
    """

    def __init__(self, urdf: str, lower: np.ndarray, upper: np.ndarray, frame: str = "head_camera_optical_frame",
                 iterations: int = 40) -> None:
        self.chain = ArmChain(urdf, "world", frame)
        self.lower, self.upper, self.iterations = lower, upper, iterations    # neck joints only

    def solve(self, target: np.ndarray, waist: np.ndarray, neck0: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (neck [yaw, pitch], remaining angle between the optical axis and the target)."""
        neck = np.clip(np.array(neck0, float), self.lower, self.upper)
        for _ in range(self.iterations):
            q = np.concatenate([waist, neck])
            pos, axes, r = self.chain.forward(q)
            _, p = self.chain.tip_pose(q)
            d = target - p
            d /= max(np.linalg.norm(d), 1e-9)
            err = np.cross(r[:, 2], d)
            jac = axes[3:].T                      # angular jacobian of the two neck joints
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), err)
            neck = np.clip(neck + dq, self.lower, self.upper)
            if np.linalg.norm(err) < 1e-4:
                break
        q = np.concatenate([waist, neck])
        _, _, r = self.chain.forward(q)
        _, p = self.chain.tip_pose(q)
        d = (target - p) / max(np.linalg.norm(target - p), 1e-9)
        return neck, float(math.acos(float(np.clip(r[:, 2] @ d, -1.0, 1.0))))
