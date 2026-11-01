"""Pure helpers for the social gestures: what counts as a greeting, and head-swing targets."""
from __future__ import annotations

import re

import numpy as np

GESTURE_SKILLS = ("wave", "nod", "shake_head")
_GREETING = re.compile(r"\b(hi+|hello+|hey+|hiya|namaste|greetings|good (morning|afternoon|evening|night)|"
                       r"wave|bye+|goodbye|see you)\b", re.IGNORECASE)
MAX_SOCIAL_WORDS = 5


def is_greeting(text: str) -> bool:
    """A short social message ("HI", "wave hi", "good morning robot"), not a task that mentions one."""
    return bool(_GREETING.search(text)) and len(text.split()) <= MAX_SOCIAL_WORDS


def swing_targets(rest: np.ndarray, axis: int, amplitude: float, lower: np.ndarray, upper: np.ndarray,
                  swings: int = 2) -> list[np.ndarray]:
    """Head targets for a nod/shake, ending at `rest`.

    The swing is centred so its full width fits inside the limits: the scan pose sits at the
    neck pitch limit, where a swing around it lost its lower half (measured: a nod of 0.149 rad
    in total, easy to miss in Gazebo)."""
    centre = float(np.clip(rest[axis], lower[axis] + amplitude, upper[axis] - amplitude))
    out = []
    for k in range(2 * swings):
        t = np.array(rest, float)
        t[axis] = centre + (amplitude if k % 2 == 0 else -amplitude)
        out.append(np.clip(t, lower, upper))
    out.append(np.array(rest, float))
    return out
