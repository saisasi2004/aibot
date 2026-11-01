"""Minimum-jerk trajectories: boundary conditions, velocity limits, waypoint chaining."""
import numpy as np

from aibot_ai.motion.trajectory import Trajectory, min_jerk_scalar, segment_duration


def test_profile_boundaries():
    p, dp = min_jerk_scalar(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(p, [0, 0.5, 1]) and np.allclose(dp[[0, 2]], 0) and np.isclose(dp[1], 1.875)


def test_respects_velocity_limits():
    q0, q1 = np.zeros(3), np.array([1.0, -2.0, 0.1])
    vmax = np.array([1.0, 4.0, 0.05])
    traj = Trajectory([q0, q1], vmax, speed=0.5)
    ts = np.linspace(0, traj.duration, 2001)
    v = np.array([traj.sample(t)[1] for t in ts])
    assert np.all(np.abs(v) <= 0.5 * vmax * (1 + 1e-6))
    assert np.allclose(traj.sample(traj.duration)[0], q1) and np.allclose(traj.sample(0)[0], q0)


def test_consecutive_samples_at_50hz_are_small():
    """At 50 Hz no command step exceeds v_max / 50 (the hand-over jump budget is 0.05 rad)."""
    traj = Trajectory([np.zeros(2), np.array([1.5, -0.8])], np.array([2.0, 2.0]))
    qs = np.array([traj.sample(t)[0] for t in np.arange(0, traj.duration + 0.02, 0.02)])
    assert np.max(np.abs(np.diff(qs, axis=0))) <= 2.0 / 50 + 1e-9


def test_waypoints_rest_and_chain():
    wps = [np.zeros(1), np.ones(1), np.zeros(1)]
    traj = Trajectory(wps, np.array([1.0]))
    t_mid = traj.segments[0].duration
    q, v = traj.sample(t_mid)
    assert np.allclose(q, 1.0) and np.allclose(v, 0.0)
    assert np.allclose(traj.goal, 0.0)


def test_zero_motion_and_requested_duration():
    assert segment_duration(np.ones(2), np.ones(2), np.ones(2)) == 0.2
    traj = Trajectory([np.zeros(1), np.full(1, 0.01)], np.array([5.0]), durations=[1.0])
    assert np.isclose(traj.duration, 1.0)


def test_executor_slows_down_instead_of_skipping_after_a_stall(monkeypatch):
    from aibot_ai.motion import executor as ex_mod
    clock = {"t": 100.0}
    monkeypatch.setattr(ex_mod.time, "monotonic", lambda: clock["t"])
    sent = []
    ex = ex_mod.TrajectoryExecutor(np.full(1, -10.0), np.full(1, 10.0), lambda q, g: sent.append(q[0]), rate_hz=50)
    traj = Trajectory([np.zeros(1), np.ones(1)], np.array([1.0]))
    ex.start(traj, ["g"])
    for dt in (0.02, 0.02, 0.30, 0.02):          # a 300 ms stall in the middle
        clock["t"] += dt
        ex.tick(True)
    steps = np.diff([0.0] + sent)
    q_ref = [traj.sample(t)[0][0] for t in (0.02, 0.04, 0.07, 0.09)]
    assert np.allclose(sent, q_ref)               # time advanced 0.03 s, not 0.30 s, during the stall
    assert np.max(np.abs(steps)) < 0.05
