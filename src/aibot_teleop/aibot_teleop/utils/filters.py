"""Low-latency smoothing filters for teleoperation signals, vectorised over joints."""
from __future__ import annotations

import math

import numpy as np


def smoothing_alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray | float:
    """Exponential smoothing factor of a first-order low-pass with the given cutoff."""
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff_hz, dtype=float))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter (Casiez, Roussel & Vogel, CHI 2012) applied element-wise to a vector.

    A low-pass whose cutoff rises with speed:  fc = min_cutoff + beta * |dx/dt|.
    Holding still -> low cutoff -> MediaPipe jitter is removed.
    Moving fast   -> high cutoff -> little lag, which is what teleoperation needs.
    min_cutoff and beta may be scalars or per-element arrays.
    """

    def __init__(self, min_cutoff: float | np.ndarray = 1.0, beta: float | np.ndarray = 0.5,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = np.asarray(min_cutoff, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None

    def reset(self, x: np.ndarray | None = None) -> None:
        self._x = None if x is None else np.array(x, dtype=float)
        self._dx = None if x is None else np.zeros_like(self._x)

    @property
    def value(self) -> np.ndarray | None:
        return None if self._x is None else self._x.copy()

    def __call__(self, x: np.ndarray, dt: float, mask: np.ndarray | None = None) -> np.ndarray:
        """Filter one sample. Elements where mask is False keep their previous state (held)."""
        x = np.asarray(x, dtype=float)
        if self._x is None:
            self.reset(x)
            return x.copy()
        if dt <= 0.0:
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = smoothing_alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        a = smoothing_alpha(self.min_cutoff + self.beta * np.abs(dx_hat), dt)
        x_hat = a * x + (1.0 - a) * self._x
        if mask is not None:
            x_hat = np.where(mask, x_hat, self._x)
            dx_hat = np.where(mask, dx_hat, 0.0)
        self._x, self._dx = x_hat, dx_hat
        return x_hat.copy()


class EMAFilter:
    """Plain exponential moving average: y += alpha * (x - y)."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._y: np.ndarray | None = None

    def reset(self) -> None:
        self._y = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self._y = x.copy() if self._y is None else self._y + self.alpha * (x - self._y)
        return self._y.copy()


def unwrap_towards(angle: np.ndarray | float, reference: np.ndarray | float) -> np.ndarray | float:
    """Shift `angle` by multiples of 2*pi so it is the representative closest to `reference`.

    atan2 jumps from +pi to -pi; filtering or rate-limiting across that jump would spin the
    joint the long way round.
    """
    return angle + 2.0 * np.pi * np.round((np.asarray(reference) - angle) / (2.0 * np.pi))
