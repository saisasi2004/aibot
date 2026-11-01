"""Arm retargeting: analytic round trips, agreement with the real URDF, IK, mirroring, waist."""
import math
import shutil
import subprocess

import numpy as np
import pytest

from aibot_teleop.retarget.arm_angles import arm_forward, hand_frame, rot_x, rot_y, rot_z, solve_arm
from aibot_teleop.retarget.body_frame import MP_TO_BODY, chest_frame, make_sample
from aibot_teleop.retarget.torso_head import waist_targets

RNG = np.random.default_rng(42)


def random_arm(side: str) -> np.ndarray:
    """Random arm pose away from the two singularities (sr = +-90 deg, straight elbow)."""
    roll = RNG.uniform(-0.2, 1.2) * (1 if side == "left" else -1)
    return np.array([RNG.uniform(-2.5, 1.0), roll, RNG.uniform(-1.5, 1.5), -RNG.uniform(0.4, 2.0),
                     RNG.uniform(-1.5, 1.5), RNG.uniform(-1.2, 1.2), RNG.uniform(-1.2, 1.2)])


@pytest.mark.parametrize("side", ["left", "right"])
def test_round_trip_recovers_joint_angles(side):
    for _ in range(200):
        q = random_arm(side)
        u, f, h = arm_forward(q)
        prev = q + RNG.normal(0, 0.05, 7)          # previous frame close by, as in streaming
        est = solve_arm(0.3 * u, 0.25 * f, h, prev)
        err = np.abs(np.remainder(est - q + math.pi, 2 * math.pi) - math.pi)
        assert err.max() < 1e-6, (q, est)


def test_canonical_poses():
    zero = np.zeros(7)
    # arm straight forward, horizontal: shoulder pitch -90 deg, no roll
    q = solve_arm(np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), None, zero)
    assert math.isclose(q[0], -math.pi / 2, abs_tol=1e-9) and abs(q[1]) < 1e-9
    # left T-pose: roll +90 deg; elbow straight
    q = solve_arm(np.array([0, 1.0, 0]), np.array([0, 1.0, 0]), None, zero)
    assert math.isclose(q[1], math.pi / 2, abs_tol=1e-9) and abs(q[3]) < 1e-6
    # hanging arm, forearm forward: elbow -90 deg (the robot flexes forward with negative elbow_pitch)
    q = solve_arm(np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), None, zero)
    assert math.isclose(q[3], -math.pi / 2, abs_tol=1e-9) and abs(q[0]) < 1e-9


def test_singularity_holds_previous_pitch():
    prev = np.array([-0.7, 1.5, 0.2, -0.5, 0, 0, 0])
    q = solve_arm(np.array([0, 1.0, 0]), np.array([0, 1.0, 0]), None, prev)
    assert math.isclose(q[0], prev[0])          # exact T-pose: pitch undefined, kept


def test_hand_frame_matches_robot_convention():
    """Arm hanging, palm towards the thigh, thumb forward -> identity, like the robot at q = 0."""
    pts = np.zeros((21, 3))
    pts[9] = [0, 0, -0.09]          # middle MCP: fingers point down
    pts[5] = [0.02, 0, -0.085]      # index MCP: forward
    pts[17] = [-0.03, 0, -0.08]     # pinky MCP: back
    assert np.allclose(hand_frame(pts), np.eye(3), atol=1e-6)
    r = rot_z(0.4) @ rot_y(-0.3) @ rot_x(0.2)
    assert np.allclose(hand_frame(pts @ r.T), r, atol=1e-6)


def test_waist_decomposition_inverts_robot_convention():
    a, b, c = 0.3, -0.1, 0.4                      # chest turned left, tilted, leaning forward
    r = rot_z(a) @ rot_x(b) @ rot_y(c)
    yaw, roll, pitch = waist_targets(r, np.eye(3))
    assert np.allclose([yaw, roll, pitch], [-a, -b, -c])
    # Robot: R_torso = Rz(-yaw) Rx(-roll) Ry(-pitch) must reproduce the chest rotation
    assert np.allclose(rot_z(-yaw) @ rot_x(-roll) @ rot_y(-pitch), r)


def _mp_pose_with_left_arm_forward() -> np.ndarray:
    """Operator facing the camera, left arm straight forward, right arm hanging (MP world axes)."""
    body = np.zeros((33, 3))
    body[11], body[12] = [0, 0.18, 0.45], [0, -0.18, 0.45]          # shoulders
    body[13], body[15] = [0.28, 0.18, 0.45], [0.54, 0.18, 0.45]     # left elbow, wrist (forward)
    body[14], body[16] = [0, -0.18, 0.17], [0, -0.18, -0.09]        # right elbow, wrist (down)
    body[23], body[24] = [0, 0.1, 0], [0, -0.1, 0]
    return body @ MP_TO_BODY      # body -> MP (inverse of the orthogonal MP_TO_BODY)


@pytest.mark.parametrize("mirror", [False, True])
def test_camera_axes_and_mirroring(mirror):
    mp_pts = _mp_pose_with_left_arm_forward()
    s = make_sample(mp_pts, np.ones(33), {"left": None, "right": None}, 0.0, mirror)
    r = chest_frame(s, use_hips=True)
    assert np.allclose(r, np.eye(3), atol=1e-9)
    forward_side, hanging_side = ("right", "left") if mirror else ("left", "right")
    from aibot_teleop.retarget.arm_angles import arm_vectors
    u, f = arm_vectors(s, forward_side, r)
    q = solve_arm(u, f, None, np.zeros(7))
    assert math.isclose(q[0], -math.pi / 2, abs_tol=1e-6)     # that robot arm points forward
    u, f = arm_vectors(s, hanging_side, r)
    assert np.allclose(solve_arm(u, f, None, np.zeros(7))[:4], 0, atol=1e-6)


# ---------------------------------------------------------------- against the real URDF
@pytest.fixture(scope="module")
def urdf() -> str:
    if shutil.which("xacro") is None:
        pytest.skip("xacro not on PATH (source ROS 2 and the workspace)")
    try:
        from ament_index_python.packages import get_package_share_directory
        path = get_package_share_directory("aibot_description") + "/urdf/aibot.urdf.xacro"
    except Exception:
        pytest.skip("aibot_description not built/sourced")
    return subprocess.run(["xacro", path], check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize("side", ["left", "right"])
def test_direction_model_matches_urdf(urdf, side):
    """The analytic model's forearm/upper-arm directions agree with URDF FK (link offsets are cm)."""
    from aibot_teleop.retarget.arm_ik import ArmChain
    chain = ArmChain(urdf, "base_link", f"{side}_hand_link")
    assert chain.names == [f"{side}_{j}" for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "arm_yaw_joint",
                                                    "elbow_pitch_joint", "elbow_yaw_joint", "wrist_pitch_joint",
                                                    "wrist_roll_joint")]
    for _ in range(100):
        q = random_arm(side)
        pos, _, r_tip = chain.forward(q)
        u, f, h = arm_forward(q)
        u_urdf = pos[3] - pos[1]
        f_urdf = pos[5] - pos[3]
        assert np.dot(u, u_urdf / np.linalg.norm(u_urdf)) > math.cos(math.radians(6))
        assert np.dot(f, f_urdf / np.linalg.norm(f_urdf)) > math.cos(math.radians(8))
        assert np.allclose(h, r_tip, atol=1e-6)      # hand frame convention identical


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_reaches_fk_targets(urdf, side):
    from aibot_teleop.retarget.arm_ik import ArmChain, ArmIK
    chain = ArmChain(urdf, "base_link", f"{side}_hand_link")
    lo, hi = np.full(7, -2.7), np.full(7, 2.7)
    # Raw solver convergence from a deliberately poor seed: no posture pull towards that seed.
    ik = ArmIK(chain, lo, hi, iterations=60, w_posture=0.0)
    for _ in range(20):
        q_true = random_arm(side)
        pos, _, r = chain.forward(q_true)
        q = ik.solve(pos[5], pos[3], r, q_true + RNG.normal(0, 0.15, 7))
        p, _, _ = chain.forward(q)
        assert np.linalg.norm(p[5] - pos[5]) < 0.01
