"""T-pose calibration: neutral offsets and arm lengths of the current operator.

Hold a T-pose (both arms straight out to the sides, palms down) for `duration` seconds. While
the pose is held, each frame is checked and stored; if the pose breaks, the countdown restarts.
The result:
  * neutral chest orientation. The robot's waist angles are measured against it, which also
    absorbs a tilted laptop camera and an operator not squarely facing it;
  * neutral head direction (ear midpoint -> nose), so looking straight ahead means neck 0;
  * upper-arm and forearm lengths (median of metric 3D landmarks), used to scale IK targets;
  * a small per-arm bias rotation taking the measured T-pose arm direction onto the ideal
    horizontal one, compensating MediaPipe's systematic depth bias (capped at max_bias_deg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .retarget.body_frame import ELBOW, SHOULDER, WRIST, BodySample, average_rotation, rotation_between, unit
from .retarget.torso_head import head_direction


@dataclass
class Calibration:
    neutral_chest: np.ndarray = field(default_factory=lambda: np.eye(3))
    head_neutral: tuple[float, float] = (0.0, -0.20)   # typical: nose ~11 deg below the ear line
    upper_arm: dict[str, float] = field(default_factory=lambda: {"left": 0.28, "right": 0.28})
    forearm: dict[str, float] = field(default_factory=lambda: {"left": 0.26, "right": 0.26})
    arm_bias: dict[str, np.ndarray] = field(default_factory=lambda: {"left": np.eye(3), "right": np.eye(3)})
    calibrated: bool = False

    def summary(self) -> str:
        if not self.calibrated:
            return "defaults (not calibrated)"
        bias = {s: math.degrees(math.acos(float(np.clip((np.trace(r) - 1) / 2, -1, 1)))) for s, r in self.arm_bias.items()}
        return (f"arms L {self.upper_arm['left']:.2f}+{self.forearm['left']:.2f} m, "
                f"R {self.upper_arm['right']:.2f}+{self.forearm['right']:.2f} m; "
                f"head neutral yaw {math.degrees(self.head_neutral[0]):+.0f} elev {math.degrees(self.head_neutral[1]):+.0f} deg; "
                f"arm bias L {bias['left']:.0f} R {bias['right']:.0f} deg")


class TPoseCalibrator:
    def __init__(self, duration: float = 3.0, timeout: float = 30.0, arm_tolerance_deg: float = 30.0,
                 elbow_tolerance_deg: float = 35.0, max_bias_deg: float = 20.0, min_samples: int = 15) -> None:
        self.duration, self.timeout = duration, timeout
        self.cos_arm = math.cos(math.radians(arm_tolerance_deg))
        self.cos_elbow = math.cos(math.radians(elbow_tolerance_deg))
        self.max_bias = math.radians(max_bias_deg)
        self.min_samples = min_samples
        self.active = False
        self._t_start = self._t_hold = 0.0
        self._samples: list[tuple[BodySample, np.ndarray]] = []

    def start(self, now: float) -> None:
        self.active, self._t_start, self._t_hold = True, now, None
        self._samples = []

    def cancel(self) -> None:
        self.active = False

    def is_t_pose(self, sample: BodySample, r_chest: np.ndarray) -> bool:
        p = sample.points
        for side, sign in (("left", 1.0), ("right", -1.0)):
            s, e, w = SHOULDER[side], ELBOW[side], WRIST[side]
            if not sample.visible(s, e, w, threshold=0.6):
                return False
            u = unit(r_chest.T @ (p[e] - p[s]))
            f = unit(r_chest.T @ (p[w] - p[e]))
            if sign * u[1] < self.cos_arm or sign * f[1] < self.cos_arm or np.dot(u, f) < self.cos_elbow:
                return False
        return True

    def update(self, sample: BodySample, r_chest: np.ndarray, now: float) -> tuple[str, Calibration | None]:
        """Feed one frame. Returns (status text, result once finished else None)."""
        if not self.active:
            return "idle", None
        if now - self._t_start > self.timeout:
            self.active = False
            return f"no T-pose within {self.timeout:.0f} s: using defaults", Calibration()
        if not self.is_t_pose(sample, r_chest):
            self._t_hold, self._samples = None, []
            return "hold a T-pose: both arms straight out to the sides", None
        if self._t_hold is None:
            self._t_hold = now
        self._samples.append((sample, r_chest))
        held = now - self._t_hold
        if held < self.duration or len(self._samples) < self.min_samples:
            return f"hold the T-pose: {held:.1f} / {self.duration:.1f} s", None
        self.active = False
        return "calibrated", self._compute()

    def _compute(self) -> Calibration:
        cal = Calibration(calibrated=True)
        cal.neutral_chest = average_rotation([r for _, r in self._samples])
        heads = [h for s, r in self._samples if (h := head_direction(s, r)) is not None]
        if heads:
            cal.head_neutral = (float(np.median([h[0] for h in heads])), float(np.median([h[1] for h in heads])))
        for side, sign in (("left", 1.0), ("right", -1.0)):
            s, e, w = SHOULDER[side], ELBOW[side], WRIST[side]
            cal.upper_arm[side] = float(np.median([np.linalg.norm(x.points[e] - x.points[s]) for x, _ in self._samples]))
            cal.forearm[side] = float(np.median([np.linalg.norm(x.points[w] - x.points[e]) for x, _ in self._samples]))
            measured = unit(np.mean([unit(r.T @ (x.points[w] - x.points[s])) for x, r in self._samples], axis=0))
            bias = rotation_between(measured, np.array([0.0, sign, 0.0]))
            angle = math.acos(float(np.clip((np.trace(bias) - 1) / 2, -1, 1)))
            cal.arm_bias[side] = bias if angle <= self.max_bias else np.eye(3)
        return cal
