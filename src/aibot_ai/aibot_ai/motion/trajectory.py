"""Minimum-jerk joint-space trajectories for the skill server.

Why minimum jerk: the gz position servo is a first-order lag (tau ~25 ms) that turns any step
in the command into an almost instantaneous velocity jump; fingers and the waist then visibly
flop (see README pitfalls). A minimum-jerk profile starts and ends with zero velocity and
acceleration, so the servo tracks it closely and nothing is shaken.

    q(t) = q0 + (q1 - q0) * (10 s^3 - 15 s^4 + 6 s^5),   s = t / T
    peak velocity = 1.875 * |q1 - q0| / T

The duration is chosen per segment so that no joint exceeds `speed * v_max` (the URDF limits).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PEAK_VELOCITY_FACTOR = 1.875   # max of d/ds (10 s^3 - 15 s^4 + 6 s^5), at s = 0.5


def min_jerk_scalar(s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Position and d/ds of the normalised minimum-jerk profile for s in [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


def segment_duration(q0: np.ndarray, q1: np.ndarray, v_max: np.ndarray, speed: float = 1.0,
                     min_duration: float = 0.2) -> float:
    """Shortest duration keeping every joint's peak velocity within speed * v_max."""
    dq = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    if not np.any(dq > 1e-9):
        return min_duration
    return max(min_duration, float(np.max(PEAK_VELOCITY_FACTOR * dq / (speed * np.asarray(v_max, float)))))


@dataclass
class Segment:
    q0: np.ndarray
    q1: np.ndarray
    duration: float

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        s = t / self.duration if self.duration > 0 else 1.0
        p, dp = min_jerk_scalar(s)
        dq = self.q1 - self.q0
        return self.q0 + p * dq, dp * dq / max(self.duration, 1e-9)


class Trajectory:
    """Chain of minimum-jerk segments through joint-space waypoints (rest at each waypoint)."""

    def __init__(self, waypoints: list[np.ndarray], v_max: np.ndarray, speed: float = 1.0,
                 min_duration: float = 0.2, durations: list[float] | None = None) -> None:
        if len(waypoints) < 2:
            raise ValueError("a trajectory needs at least a start and a goal")
        self.segments: list[Segment] = []
        for i in range(len(waypoints) - 1):
            a, b = np.asarray(waypoints[i], float), np.asarray(waypoints[i + 1], float)
            d = segment_duration(a, b, v_max, speed, min_duration)
            if durations is not None:
                d = max(d, durations[i])        # a requested duration may only slow it down
            self.segments.append(Segment(a, b, d))
        self.duration = sum(s.duration for s in self.segments)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity at time t (clamped to [0, duration])."""
        for seg in self.segments:
            if t <= seg.duration:
                return seg.sample(max(t, 0.0))
            t -= seg.duration
        return self.segments[-1].q1.copy(), np.zeros_like(self.segments[-1].q1)

    @property
    def goal(self) -> np.ndarray:
        return self.segments[-1].q1.copy()


class PathTrajectory:
    """One minimum-jerk time law along a dense joint path (e.g. IK of a Cartesian straight line).

    Why not a Trajectory through the same points: that would stop at every 1 cm waypoint. Here
    the path parameter s (the joint-space "time at unit speed": cumulative max_j |dq_j| / v_max_j)
    follows a single minimum-jerk profile, so the hand moves smoothly along the line, starts and
    stops at rest, and no joint exceeds speed * v_max (peak ds/dt = 1.875 * S / T <= speed).
    """

    def __init__(self, path: list[np.ndarray], v_max: np.ndarray, speed: float = 1.0,
                 min_duration: float = 0.2) -> None:
        if len(path) < 2:
            raise ValueError("a path needs at least a start and a goal")
        self.path = np.array(path, float)
        steps = np.max(np.abs(np.diff(self.path, axis=0)) / (speed * np.asarray(v_max, float)), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(steps)])
        self.duration = max(min_duration, PEAK_VELOCITY_FACTOR * float(self.s[-1]))

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        total = float(self.s[-1])
        if total <= 1e-12:
            return self.path[-1].copy(), np.zeros(self.path.shape[1])
        p, dp = min_jerk_scalar(min(max(t, 0.0), self.duration) / self.duration)
        s = float(p) * total
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2))
        ds = self.s[i + 1] - self.s[i]
        u = (s - self.s[i]) / ds if ds > 0 else 1.0
        q = self.path[i] + u * (self.path[i + 1] - self.path[i])
        v = (self.path[i + 1] - self.path[i]) / ds * float(dp) * total / self.duration if ds > 0 else np.zeros_like(q)
        return q, v

    @property
    def goal(self) -> np.ndarray:
        return self.path[-1].copy()
