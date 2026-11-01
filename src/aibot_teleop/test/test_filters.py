"""Filters and limiters used on the retargeted joint stream."""
import math

import numpy as np

from aibot_teleop.utils.filters import EMAFilter, OneEuroFilter, unwrap_towards
from aibot_teleop.utils.limits import RateLimiter, deadband

DT = 1.0 / 30.0


def test_one_euro_removes_jitter_at_rest():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    raw = 0.3 + 0.01 * rng.standard_normal(300)
    out = np.array([f(np.array([x]), DT)[0] for x in raw])
    assert np.std(out[50:]) < np.std(raw[50:]) / 3
    assert abs(np.mean(out[50:]) - 0.3) < 0.005


def test_one_euro_tracks_fast_motion_with_little_lag():
    """A 1 Hz, 1 rad sine (peak 6 rad/s): the speed-adaptive cutoff keeps the lag small."""
    f = OneEuroFilter(min_cutoff=1.2, beta=0.6)
    t = np.arange(0, 3, DT)
    sig = np.sin(2 * math.pi * t)
    out = np.array([f(np.array([x]), DT)[0] for x in sig])
    lags = range(0, 10)
    best = min(lags, key=lambda k: np.mean((out[30 + k:] - sig[30:len(sig) - k]) ** 2))
    assert best * DT <= 0.1          # <= 3 frames
    assert np.max(np.abs(out[60:] - sig[60:])) < 0.35


def test_one_euro_mask_holds_value():
    f = OneEuroFilter()
    f(np.array([0.0, 0.0]), DT)
    for _ in range(30):
        out = f(np.array([1.0, 1.0]), DT, mask=np.array([True, False]))
    assert out[0] > 0.9 and out[1] == 0.0


def test_ema():
    f = EMAFilter(alpha=0.5)
    assert f(np.array([1.0]))[0] == 1.0
    assert f(np.array([0.0]))[0] == 0.5


def test_rate_limiter_caps_speed():
    r = RateLimiter(max_velocity=np.array([3.0]))
    r.reset(np.array([0.0]))
    q = r(np.array([1.0]), DT)
    assert math.isclose(q[0], 3.0 * DT)
    for _ in range(20):
        q = r(np.array([1.0]), DT)
    assert math.isclose(q[0], 1.0)


def test_deadband_keeps_small_changes_out():
    last = np.array([0.0, 0.0])
    out = deadband(np.array([0.005, 0.05]), last, 0.01)
    assert out[0] == 0.0 and out[1] == 0.05


def test_unwrap_towards():
    assert math.isclose(unwrap_towards(-math.pi + 0.1, math.pi - 0.1), math.pi + 0.1)
    assert math.isclose(unwrap_towards(0.2, 0.0), 0.2)
