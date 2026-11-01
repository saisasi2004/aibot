"""Trajectory executor: streams minimum-jerk trajectories to the six position controllers.

Kept free of skill logic so it can be reused by every skill and tested in isolation.
Guarantees:
  * every motion starts from the robot's ACTUAL joint positions (passed in by the caller from
    /joint_states), never from a remembered command;
  * commands are clamped to the URDF limits (with margin) before they leave;
  * groups the trajectory does not move are not published at all, so they keep holding;
  * stop() takes effect on the next tick (<= 1/rate s): the last commanded pose is held because
    the controllers keep their last command;
  * trajectory time advances by at most `max_step_factor` nominal ticks per tick. If the Python
    process stalls (GIL, executor load) the motion slows down for a moment instead of skipping
    ahead, so consecutive commands never jump further than the profile allows.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from .trajectory import Trajectory


class TrajectoryExecutor:
    def __init__(self, lower: np.ndarray, upper: np.ndarray, publish: Callable[[np.ndarray, list[str]], None],
                 rate_hz: float = 50.0, max_step_factor: float = 1.5) -> None:
        self.lower, self.upper = lower, upper
        self.max_dt = max_step_factor / rate_hz
        # Trajectory time per wall second: the simulation's real-time factor (1.0 on a robot).
        # Streaming in wall time while Gazebo runs at 0.4x let the servos lag behind and take a
        # different, unchecked path (measured: fingers hooked under the table edge).
        self.time_scale = 1.0
        self._publish = publish
        self._lock = threading.Lock()
        self._traj: Trajectory | None = None
        self._groups: list[str] = []
        self._t = 0.0
        self._last_tick = 0.0
        self._done = threading.Event()
        self._done.set()
        self.last_command: np.ndarray | None = None

    def start(self, traj: Trajectory, groups: list[str]) -> None:
        with self._lock:
            self._traj, self._groups = traj, groups
            self._t, self._last_tick = 0.0, time.monotonic()
            self._done.clear()

    def stop(self) -> None:
        with self._lock:
            self._traj = None
            self._done.set()

    @property
    def busy(self) -> bool:
        return not self._done.is_set()

    def wait(self, timeout: float, should_abort: Callable[[], bool], poll: float = 0.01) -> str:
        """Block until the trajectory finishes. Returns "done", "aborted" or "timeout"."""
        deadline = time.monotonic() + timeout
        while not self._done.wait(poll):
            if should_abort():
                self.stop()
                return "aborted"
            if time.monotonic() > deadline:
                self.stop()
                return "timeout"
        return "done"

    def tick(self, allowed: bool) -> None:
        """Called at the command rate. allowed=False (wrong mode / disabled) stops the motion."""
        with self._lock:
            traj = self._traj
            if traj is None:
                return
            if not allowed:
                self._traj = None
                self._done.set()
                return
            now = time.monotonic()
            self._t += min(now - self._last_tick, self.max_dt) * self.time_scale
            self._last_tick = now
            t = self._t
            q, _ = traj.sample(t)
            finished = t >= traj.duration
            groups = self._groups
            if finished:
                self._traj = None
        q = np.clip(q, self.lower, self.upper)
        self.last_command = q
        self._publish(q, groups)
        if finished:
            self._done.set()
