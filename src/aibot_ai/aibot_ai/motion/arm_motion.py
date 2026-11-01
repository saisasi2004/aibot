"""Arm motion planning for skills: palm-down reaches, Cartesian straight lines, safe joint moves.

Pure Python (no ROS) so it is unit-tested offline against the URDF. Everything works on the full
joint vector in JointMap order; each arm's IK owns the 3 waist joints + its 7 arm joints.

  reach(side, q, p)        IK to a grasp-point target, warm-started from the current pose first
  line(side, q, p)         dense IK along the straight line from the current grasp point to p
                           (1 cm steps, each warm-started from the previous: no branch flips)
  joint_move(side, q, q1)  waypoints from q to q1 whose joint-space interpolation is collision
                           free; if the direct path cuts the table (it does from the hanging
                           home pose), go via a waypoint beside the table, outside its y range
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from aibot_teleop.kinematics import ArmChain

from .ik import ReachSolver

SIDES = ("left", "right")


@dataclass
class MotionConfig:
    line_step: float = 0.01            # m between IK points on a straight line
    max_line_joint_step: float = 0.25  # rad: a bigger jump between line points = IK branch flip
    path_samples: int = 15             # minimum collision checks along a joint-space move ...
    path_step: float = 0.03            # ... and at most this much joint motion (rad) between two
    beside_x: float = 0.30             # m, waypoint beside the table (world), y = +-beside_y
    beside_y: float = 0.32             # (0.27 let the 16 cm thumb reach over the table edge on the way up)
    beside_height: float = 0.17        # above the table top
    high_height: float = 0.30          # "over the top" transfers: grasp point this far above the table
    hand_back: float = 0.10            # m of hand behind the grasp point (towards the wrist) ...
    hand_ahead: float = 0.10           # ... and ahead of it (fingertips), for choosing a finger direction
    thumb_back: float = 0.07           # m the thumb sits behind the grasp point (towards the wrist) ...
    thumb_length: float = 0.157        # ... and how far it sticks out sideways (URDF thumb tip)
    clearance_cap: float = 0.06        # m: clearance beyond this does not make a direction better
    natural_weight: float = 0.02       # m-equivalent bonus for fingers pointing away from the shoulder
    refine_below: float = 0.08         # m: a finger-steered IK this close is finished without the finger task
    transfer_height: float = 0.20      # m above the table for Cartesian transfers ...
    transfer_step: float = 0.02        # ... with an IK point every 2 cm
    max_turn_step: float = 0.10        # rad of finger-direction turn per transfer point
    max_candidates: int = 6            # feasible grasp/release poses offered to the transfer planner


def summarize(reasons: dict[str, int], top: int = 3) -> str:
    items = sorted(reasons.items(), key=lambda kv: -kv[1])[:top]
    return "; ".join(f"{w} (x{n})" for w, n in items) or "no IK solution"


class ArmMotion:
    def __init__(self, urdf: str, names: list[str], lower: np.ndarray, upper: np.ndarray, table: dict,
                 cfg: MotionConfig | None = None, solver_kwargs: dict | None = None,
                 waist_lower: np.ndarray | None = None, waist_upper: np.ndarray | None = None) -> None:
        """waist_lower/upper (yaw, roll, pitch): the skills' own, tighter waist range. Without it
        the IK "reaches" 0.95 m by leaning 57 deg forward, which no table-top task should do."""
        self.names, self.table, self.cfg = names, table, cfg or MotionConfig()
        self._beside_cache: dict = {}
        lower, upper = np.array(lower, float), np.array(upper, float)
        waist = [names.index(n) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]
        if waist_lower is not None:
            lower[waist] = np.maximum(lower[waist], waist_lower)
        if waist_upper is not None:
            upper[waist] = np.minimum(upper[waist], waist_upper)
        index = {n: i for i, n in enumerate(names)}
        self.solvers: dict[str, ReachSolver] = {}
        self.idx: dict[str, np.ndarray] = {}
        for side in SIDES:
            idx = np.array([index[n] for n in ArmChain(urdf, "world", f"{side}_hand_link").names])
            self.idx[side] = idx
            self.solvers[side] = ReachSolver(urdf, side, lower[idx], upper[idx], table, **(solver_kwargs or {}))

    def set_scene(self, obstacles: list[tuple[float, ...]],
                  attached: dict[str, tuple[np.ndarray, float] | None] | None = None,
                  ids: list[str] | None = None) -> None:
        """Obstacles for the next plans: (x, y, half_major, half_minor, yaw, top_z) oriented boxes,
        or (x, y, radius, top_z) cylinders (as squares); their ids (for messages); what each hand
        carries."""
        rows = [tuple(o) if len(o) == 6 else (o[0], o[1], o[2], o[2], 0.0, o[3]) for o in obstacles]
        ob = np.array(rows, float).reshape(-1, 6)
        for side, s in self.solvers.items():
            s.obstacles, s.obstacle_ids, s.ignore = ob, list(ids or []), set()
            s.attached = (attached or {}).get(side)

    # ------------------------------------------------------------------ vector helpers
    def arm_q(self, side: str, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, float)[self.idx[side]]

    def other_arm(self, side: str, q: np.ndarray) -> np.ndarray:
        other = "right" if side == "left" else "left"
        return np.asarray(q, float)[self.idx[other][3:]]

    def with_arm(self, side: str, q: np.ndarray, q10: np.ndarray) -> np.ndarray:
        out = np.array(q, float)
        out[self.idx[side]] = q10
        return out

    def grasp_point(self, side: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.solvers[side].grasp_point(self.arm_q(side, q))

    # ------------------------------------------------------------------ planning
    def reach(self, side: str, q: np.ndarray, target: np.ndarray, palm_down: bool = True):
        """IKResult for a grasp-point target; seeds: the current arm pose, then the fixed set."""
        s = self.solvers[side]
        cur = self.arm_q(side, q)
        return s.solve_best(np.asarray(target, float), palm_down, s.seeds([cur]), other_arm=self.other_arm(side, q))

    def line(self, side: str, q: np.ndarray, target: np.ndarray, palm_down: bool = True,
             allow_near: np.ndarray | None = None) -> tuple[list[np.ndarray] | None, str]:
        """Full-vector path along the straight line grasp_point(q) -> target, or (None, why)."""
        s = self.solvers[side]
        start, _ = self.grasp_point(side, q)
        target = np.asarray(target, float)
        n = max(1, int(np.ceil(np.linalg.norm(target - start) / self.cfg.line_step)))
        other = self.other_arm(side, q)
        prev = self.arm_q(side, q)
        path = [np.array(q, float)]
        for k in range(1, n + 1):
            p = start + (target - start) * k / n
            r = s.solve(p, palm_down, prev, prev, other)
            if not (r.pos_err <= s.pos_tol and r.rot_err <= s.tilt_tol):
                return None, f"line point {k}/{n} unreachable ({r.pos_err * 100:.1f} cm off)"
            hit = s.collision(r.q, other, allow_near if allow_near is not None else None)
            if hit:
                return None, f"line point {k}/{n} collides with the {hit.replace('other_arm', 'other arm')}"
            if np.max(np.abs(r.q - prev)) > self.cfg.max_line_joint_step:
                return None, f"line point {k}/{n}: IK jumped {np.max(np.abs(r.q - prev)):.2f} rad"
            prev = r.q
            path.append(self.with_arm(side, q, r.q))
        return path, ""

    def path_hit(self, side: str, qa: np.ndarray, qb: np.ndarray) -> str:
        """First collision along the joint-space interpolation qa -> qb ('' if none).

        Objects the arm already touches at qa are not checked: moving away from them must stay
        possible (e.g. lifting a hand that rests next to a tall object)."""
        s = self.solvers[side]
        a, b = self.arm_q(side, qa), self.arm_q(side, qb)
        other = self.other_arm(side, qa)
        saved = s.ignore
        s.ignore = saved | set(s.obstacle_hits(s.arm_points(a)))
        try:
            # dense enough that no joint moves more than path_step between checks: with 15 fixed
            # samples a 2-3 rad move let the fingertips (0.6 m out) skip over a 7 cm mug
            n = max(self.cfg.path_samples, int(np.ceil(np.max(np.abs(b - a)) / self.cfg.path_step)) + 1)
            for u in np.linspace(0.0, 1.0, n):
                hit = s.collision(a + u * (b - a), other)
                if hit:
                    return hit
            return ""
        finally:
            s.ignore = saved

    def verticals(self, side: str, q: np.ndarray, point: np.ndarray, heights: tuple[float, ...],
                  allow_near: np.ndarray | None = None, reasons: dict[str, int] | None = None):
        """Grasp-first planning of a vertical approach, as a generator of candidates
        (q_point, [path up to each height]) best first: IK at `point` per finger direction
        (ranked by clearance) and seed, obstacle-aware, then straight lines up. Reversed, the
        first path is the descent onto `point`. Failure reasons are counted into `reasons`."""
        s = self.solvers[side]
        point = np.asarray(point, float)
        other = self.other_arm(side, q)
        reasons = {} if reasons is None else reasons
        seeds = s.seeds([self.arm_q(side, q)])
        attempts = [(d, sd) for d in self.finger_directions(side, q, point) for sd in seeds[:3]]
        attempts += [(None, sd) for sd in seeds]
        found = []
        for fdir, seed in attempts:
            r = s.solve(point, True, seed, None, other, finger_dir=fdir)
            if fdir is not None and not r.reachable and not r.collision and r.pos_err < self.cfg.refine_below:
                # the soft finger task can leave a few cm near the workspace edge: finish the
                # position from there without it (the finger direction stays close)
                r = s.solve(point, True, r.q, r.q, other)
            if not r.reachable:
                why = (f"collides with the {r.collision.replace('other_arm', 'other arm')}" if r.collision
                       else "out of reach")
                reasons[why] = reasons.get(why, 0) + 1
                continue
            if any(np.max(np.abs(r.q - f)) < 0.05 for f in found):
                continue                                   # same solution from another seed
            found.append(r.q)
            q_pt = self.with_arm(side, q, r.q)
            paths = []
            for h in heights:
                path, w = self.line(side, q_pt, point + [0.0, 0.0, h], allow_near=allow_near)
                if path is None:
                    why = f"line up {h * 100:.0f} cm: {w.split(' (')[0]}"
                    reasons[why] = reasons.get(why, 0) + 1
                    break
                paths.append(path)
            else:
                yield q_pt, paths
                if len(found) >= self.cfg.max_candidates:
                    return

    def vertical(self, side: str, q: np.ndarray, point: np.ndarray, heights: tuple[float, ...],
                 allow_near: np.ndarray | None = None):
        """First candidate of verticals(), or (None, the most common failure reasons)."""
        reasons: dict[str, int] = {}
        for cand in self.verticals(side, q, point, heights, allow_near, reasons):
            return cand
        return None, summarize(reasons)

    def _transfer_avoiding(self, side: str, q: np.ndarray, q1: np.ndarray, target: tuple | None, target_id: str):
        """transfer() with the object being approached counted as an obstacle: only the final
        vertical move may come close to it. Leaving it out of the whole approach let the hand
        sweep through it on the way (measured: a cube pushed 5.7 cm before the descent began)."""
        s = self.solvers[side]
        if target is None:
            return self.transfer(side, q, q1)
        saved, saved_ids = s.obstacles, s.obstacle_ids
        s.obstacles = np.vstack([saved, np.array(target, float).reshape(1, 6)])
        s.obstacle_ids = list(saved_ids) + [target_id]
        try:
            return self.transfer(side, q, q1)
        finally:
            s.obstacles, s.obstacle_ids = saved, saved_ids

    def approach(self, side: str, q: np.ndarray, point: np.ndarray, heights: tuple[float, ...],
                 allow_near: np.ndarray | None = None, deadline: float | None = None,
                 target: tuple | None = None, target_id: str = ""):
        """Best vertical candidate that also has a collision-free transfer from q to its top:
        (legs, q_point, paths) or (None, None, why). Stops trying at `deadline` (time.monotonic).
        target: oriented box of the object approached, an obstacle for the transfer only."""
        reasons: dict[str, int] = {}
        for q_pt, paths in self.verticals(side, q, point, heights, allow_near, reasons):
            if deadline is not None and time.monotonic() > deadline:
                reasons["planning time budget used up"] = 1
                break
            legs, why = self._transfer_avoiding(side, q, paths[0][-1], target, target_id)
            if legs is not None:
                return legs, q_pt, paths
            key = "transfer: " + why.split(";")[-1].strip()[:80]
            reasons[key] = reasons.get(key, 0) + 1
        return None, None, summarize(reasons)

    def finger_directions(self, side: str, q: np.ndarray, point: np.ndarray, n: int = 8, keep: int = 8) -> list[np.ndarray]:
        """Horizontal finger directions to try at `point`, best first: the hand spans ~10 cm behind
        (wrist) to ~10 cm ahead (fingertips) of the grasp point, so score each direction by its
        clearance from the obstacles along that span, plus a bonus for pointing away from the
        shoulder (a natural, well-conditioned reach)."""
        s = self.solvers[side]
        pos, _, _ = s.chain.forward(self.arm_q(side, q))
        shoulder = pos[3]                                  # first arm joint (after the 3 waist joints)
        natural = (point - shoulder)[:2]
        natural = natural / max(np.linalg.norm(natural), 1e-9)
        ob = s.obstacles
        scored = []
        thumb_sign = -1.0 if side == "left" else 1.0       # palm down: left thumb to the fingers' right
        for a in np.arange(n) * 2 * np.pi / n:
            d = np.array([np.cos(a), np.sin(a)])
            t = thumb_sign * np.array([-d[1], d[0]])
            span = point[:2] + np.linspace(-self.cfg.hand_back, self.cfg.hand_ahead, 9)[:, None] * d
            thumb_base = point[:2] - self.cfg.thumb_back * d
            span = np.vstack([span, thumb_base + np.linspace(0.0, self.cfg.thumb_length, 6)[:, None] * t])
            if len(ob):
                dist = np.linalg.norm(span[:, None, :] - ob[None, :, :2], axis=2) - np.maximum(ob[:, 2], ob[:, 3])[None, :]
                clear = float(min(np.min(dist), self.cfg.clearance_cap))
            else:
                clear = self.cfg.clearance_cap
            scored.append((clear + self.cfg.natural_weight * float(d @ natural), a, d))
        scored.sort(key=lambda t: -t[0])
        return [np.array([d[0], d[1], 0.0]) for _sc, _a, d in scored[:keep]]

    def beside(self, side: str, q: np.ndarray) -> np.ndarray | None:
        """Palm-down pose beside the table (outside its y range), for this start pose."""
        key = (side, tuple(np.round(self.arm_q(side, q), 3)), tuple(np.round(self.other_arm(side, q), 2)),
               self.solvers[side].obstacles.tobytes())
        if key not in self._beside_cache:
            sign = 1.0 if side == "left" else -1.0
            target = np.array([self.cfg.beside_x, sign * self.cfg.beside_y, self.table["top_z"] + self.cfg.beside_height])
            r = self.reach(side, q, target, palm_down=True)
            if len(self._beside_cache) > 64:
                self._beside_cache.clear()
            self._beside_cache[key] = r.q if r.reachable else None
        q10 = self._beside_cache[key]
        return None if q10 is None else self.with_arm(side, q, q10)

    def lifted(self, side: str, q: np.ndarray, height: float) -> np.ndarray | None:
        """The same arm configuration family raised straight up to `height` above the table
        (end of a Cartesian line, so the hand keeps its orientation), or None."""
        p, _ = self.grasp_point(side, q)
        z = self.table["top_z"] + height
        if p[2] >= z - 0.01:
            return np.array(q, float)
        path, _why = self.line(side, q, np.array([p[0], p[1], z]))
        return path[-1] if path is not None else None

    def finger_dir(self, side: str, q: np.ndarray) -> np.ndarray:
        """Horizontal direction the fingers point at pose q (unit, z = 0)."""
        _, r = self.grasp_point(side, q)
        f = -r[:, 2]
        f = np.array([f[0], f[1], 0.0])
        return f / max(np.linalg.norm(f), 1e-9)

    def palm_down(self, side: str, q: np.ndarray, tol: float = 0.35) -> bool:
        _, r = self.grasp_point(side, q)
        return self.solvers[side].palm_tilt(r) < tol

    def cartesian_transfer(self, side: str, q: np.ndarray, q1: np.ndarray) -> tuple[list[np.ndarray] | None, str]:
        """Dense path from q to q1 (both palm down): rise to a safe height, cross over while the
        finger direction turns gradually, come straight down; every point is IK'd from the
        previous one (no branch flips) and collision-checked. Joint-space interpolation between
        two different palm-down postures can swing the fingers through the table instead."""
        s = self.solvers[side]
        other = self.other_arm(side, q)
        p0, _ = self.grasp_point(side, q)
        p1, _ = self.grasp_point(side, q1)
        d0, d1 = self.finger_dir(side, q), self.finger_dir(side, q1)
        h = max(p0[2], p1[2], self.table["top_z"] + self.cfg.transfer_height)
        corners = [p0, np.array([p0[0], p0[1], h]), np.array([p1[0], p1[1], h]), p1]
        a0, a1 = np.arctan2(d0[1], d0[0]), np.arctan2(d1[1], d1[0])
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        pts, dirs = [], []
        for i, (a, b) in enumerate(zip(corners[:-1], corners[1:])):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / self.cfg.transfer_step)))
            if i == 1:                                     # crossing: also <= max_turn_step per point
                n = max(n, int(np.ceil(abs(da) / self.cfg.max_turn_step)))
            for k in range(1, n + 1):
                pts.append(a + (b - a) * k / n)
                u = (k / n) if i == 1 else (0.0 if i == 0 else 1.0)          # turn while crossing
                ang = a0 + u * da
                dirs.append(np.array([np.cos(ang), np.sin(ang), 0.0]))
        prev, path = self.arm_q(side, q), [np.array(q, float)]
        for k, (p, d) in enumerate(zip(pts, dirs), 1):
            r = s.solve(p, True, prev, prev, other, finger_dir=d)
            if not (r.pos_err <= s.pos_tol and r.rot_err <= s.tilt_tol):
                r = s.solve(p, True, r.q, r.q, other)
            if not (r.pos_err <= s.pos_tol and r.rot_err <= s.tilt_tol):
                return None, f"transfer point {k}/{len(pts)} unreachable"
            hit = s.collision(r.q, other)
            if hit:
                return None, f"transfer point {k}/{len(pts)} hits the {hit}"
            if np.max(np.abs(r.q - prev)) > self.cfg.max_line_joint_step:
                return None, f"transfer point {k}/{len(pts)}: IK jumped"
            prev = r.q
            path.append(self.with_arm(side, q, r.q))
        if np.max(np.abs(prev - self.arm_q(side, q1))) > self.cfg.max_line_joint_step:
            return None, "transfer ends in a different arm posture than the goal"
        path.append(np.array(q1, float))
        return path, ""

    def transfer(self, side: str, q: np.ndarray, q1: np.ndarray) -> tuple[list[tuple[str, list[np.ndarray]]] | None, str]:
        """Collision-free way from q to q1 as legs ("joint", waypoints) / ("path", dense path).

        Direct joint move if clear; else a Cartesian transfer when the hand is palm down;
        else the side waypoint (joint) followed by a Cartesian transfer; else joint_move's
        routes. The skill server runs joint legs as min-jerk waypoint trajectories and paths
        with one time law (PathTrajectory)."""
        hit = self.path_hit(side, q, q1)
        if not hit:
            return [("joint", [np.array(q1, float)])], ""
        tried = [f"direct: {hit}"]
        if self.palm_down(side, q):
            path, why = self.cartesian_transfer(side, q, q1)
            if path is not None:
                return [("path", path)], ""
            tried.append(f"cartesian: {why}")
        via = self.beside(side, q)
        if via is not None and not self.path_hit(side, q, via):
            path, why = self.cartesian_transfer(side, via, q1)
            if path is not None:
                return [("joint", [via]), ("path", path)], ""
            tried.append(f"side + cartesian: {why}")
        wps, why = self.joint_move(side, q, q1)
        if wps is not None:
            return [("joint", wps)], ""
        tried.append(why)
        return None, "; ".join(tried)

    def joint_move(self, side: str, q: np.ndarray, q1: np.ndarray) -> tuple[list[np.ndarray] | None, str]:
        """Waypoints (excluding q) from q to q1 whose joint-space legs are collision free, or (None, why).

        Routes, in order: direct; via the side of the table (from the hanging home pose the
        direct move cuts the table edge); over the top, i.e. the goal's own configuration
        raised straight up (and the start's, if it is over the table), joined high above
        everything, then straight down."""
        q1 = np.array(q1, float)
        hit = self.path_hit(side, q, q1)
        if not hit:
            return [q1], ""
        tried = [f"direct: {hit}"]
        via = self.beside(side, q)
        high = self.cfg.high_height
        up_goal = self.lifted(side, q1, high)
        up_start = self.lifted(side, q, high) if self._over_table(side, q) else None
        routes = []
        if via is not None:
            routes.append(("via the side", [via, q1]))
            if up_goal is not None:
                routes.append(("side then over the top", [via, up_goal, q1]))
        if up_goal is not None:
            routes.append(("over the top", ([up_start] if up_start is not None else []) + [up_goal, q1]))
        for name, wps in routes:
            legs = list(zip([np.array(q, float)] + wps[:-1], wps))
            leg_hit = next((h for a, b in legs if (h := self.path_hit(side, a, b))), "")
            if not leg_hit:
                return wps, ""
            tried.append(f"{name}: {leg_hit}")
        return None, "no collision-free path (" + ", ".join(tried) + ")"

    def _over_table(self, side: str, q: np.ndarray, margin: float = 0.05) -> bool:
        p, _ = self.grasp_point(side, q)
        t = self.table
        return t["x"][0] - margin < p[0] < t["x"][1] + margin and t["y"][0] - margin < p[1] < t["y"][1] + margin
