"""Joint-limit handling: clamping, velocity limiting and dead-band for streamed joint targets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class JointLimits:
    """Position/velocity limits for an ordered list of joints."""

    names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray

    @classmethod
    def from_yaml(cls, path: str | Path, names: list[str], margin: float = 0.0) -> "JointLimits":
        """Load ros2_control joint_limits YAML (as generated into aibot_description/config).

        `margin` (rad) shrinks every range so commands never press against the hard stop.
        """
        data = yaml.safe_load(Path(path).read_text())["joint_limits"]
        missing = [n for n in names if n not in data]
        if missing:
            raise KeyError(f"joints missing from {path}: {missing}")
        lo = np.array([data[n]["min_position"] for n in names], dtype=float) + margin
        hi = np.array([data[n]["max_position"] for n in names], dtype=float) - margin
        vel = np.array([data[n]["max_velocity"] for n in names], dtype=float)
        return cls(list(names), np.minimum(lo, hi), np.maximum(lo, hi), vel)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)


class RateLimiter:
    """Limit how far each element may move per update: |dq| <= v_max * dt."""

    def __init__(self, max_velocity: np.ndarray | float) -> None:
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        self._q: np.ndarray | None = None

    @property
    def value(self) -> np.ndarray | None:
        return None if self._q is None else self._q.copy()

    def reset(self, q: np.ndarray) -> None:
        self._q = np.array(q, dtype=float)

    def __call__(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if self._q is None:
            self._q = target.copy()
            return target.copy()
        step = np.clip(target - self._q, -self.max_velocity * dt, self.max_velocity * dt)
        self._q = self._q + step
        return self._q.copy()


def deadband(target: np.ndarray, last: np.ndarray, width: np.ndarray | float) -> np.ndarray:
    """Keep `last` wherever the new target differs from it by less than `width`.

    Removes the residual sub-degree shimmer of landmark noise when the operator holds still,
    so the robot does not tremble; any real motion exceeds the band within a frame or two.
    """
    target = np.asarray(target, dtype=float)
    return np.where(np.abs(target - last) < width, last, target)
