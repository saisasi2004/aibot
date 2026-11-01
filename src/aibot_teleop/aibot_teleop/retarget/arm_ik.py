"""Alternative arm mode (use_ik:=true): wrist-position IK with an elbow hint.

Trade-offs against direct angle mapping (arm_angles.py, the default):
  + IK puts the robot's wrist where the operator's wrist is (scaled by arm length), which
    matters for reaching objects when human and robot proportions differ.
  + Infeasible targets degrade gracefully: you get the closest reachable point.
  - MediaPipe's monocular depth noise goes straight into the reach direction.
  - It needs the T-pose arm-length calibration.
  - An iterative solver can drift between elbow solutions for the same wrist position. The
    elbow hint and warm start counteract that.
  - It costs ~1 ms per arm per frame instead of microseconds.

Solver: damped least squares on the URDF chain base_link -> {side}_hand_link with stacked
weighted tasks: wrist-centre position (primary), elbow position (hint: keeps the robot's elbow
plane like the operator's) and hand orientation. Warm-started from the previous solution, a few
iterations per frame suffice because targets move little between frames.
"""
from __future__ import annotations

import numpy as np

from ..kinematics import ArmChain, rot_log  # noqa: F401  (ArmChain re-exported for existing imports)

WRIST_JOINT, ELBOW_JOINT, SHOULDER_JOINT = 5, 3, 1   # indices in the 7 active joints


class ArmIK:
    def __init__(self, chain: ArmChain, lower: np.ndarray, upper: np.ndarray, iterations: int = 8,
                 damping: float = 0.03, w_wrist: float = 1.0, w_elbow: float = 0.35, w_rot: float = 0.15,
                 w_posture: float = 0.05) -> None:
        self.chain, self.lower, self.upper = chain, lower, upper
        self.iterations, self.damping = iterations, damping
        self.w = (w_wrist, w_elbow, w_rot)
        self.w_posture = w_posture
        pos, axes, _ = chain.forward(np.zeros(len(chain.names)))
        # Shoulder reference: the point of the shoulder-pitch axis level with the roll joint.
        # (The roll joint itself sits 2.5 cm below that axis and swings with pitch; using it
        # would make the targets disagree with the robot by ~4 cm once the arm is raised.)
        pitch_axis = axes[0] / np.linalg.norm(axes[0])
        self.shoulder = pos[0] + pitch_axis * float(np.dot(pos[SHOULDER_JOINT] - pos[0], pitch_axis))
        self.upper_len = float(np.linalg.norm(pos[ELBOW_JOINT] - self.shoulder))
        self.fore_len = float(np.linalg.norm(pos[WRIST_JOINT] - pos[ELBOW_JOINT]))

    def targets_from_human(self, upper: np.ndarray, fore: np.ndarray, human_upper: float,
                           human_fore: float) -> tuple[np.ndarray, np.ndarray]:
        """Scale operator arm vectors (chest frame) onto the robot: (wrist target, elbow target)."""
        elbow = self.shoulder + upper * (self.upper_len / max(human_upper, 0.1))
        wrist = elbow + fore * (self.fore_len / max(human_fore, 0.1))
        return wrist, elbow

    def solve(self, wrist_target: np.ndarray, elbow_target: np.ndarray | None,
              rot_target: np.ndarray | None, q0: np.ndarray) -> np.ndarray:
        """Iterate from seed q0. A weak posture task pulls towards q0, so directions the tasks do
        not constrain (e.g. spinning the arm about itself in the T-pose singularity) stay put."""
        q_rest = np.clip(np.array(q0, dtype=float), self.lower, self.upper)
        q = q_rest.copy()
        n = len(q)
        w_wrist, w_elbow, w_rot = self.w
        for _ in range(self.iterations):
            pos, axes, r_tip = self.chain.forward(q)
            rows, errs = [], []

            jw = np.zeros((3, n))
            for i in range(WRIST_JOINT):
                jw[:, i] = np.cross(axes[i], pos[WRIST_JOINT] - pos[i])
            rows.append(w_wrist * jw)
            errs.append(w_wrist * (wrist_target - pos[WRIST_JOINT]))

            if elbow_target is not None:
                je = np.zeros((3, n))
                for i in range(ELBOW_JOINT):
                    je[:, i] = np.cross(axes[i], pos[ELBOW_JOINT] - pos[i])
                rows.append(w_elbow * je)
                errs.append(w_elbow * (elbow_target - pos[ELBOW_JOINT]))

            if rot_target is not None:
                rows.append(w_rot * axes.T)
                errs.append(w_rot * rot_log(rot_target @ r_tip.T))

            rows.append(self.w_posture * np.eye(n))
            errs.append(self.w_posture * (q_rest - q))

            jac, err = np.vstack(rows), np.concatenate(errs)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping ** 2 * np.eye(len(err)), err)
            q = np.clip(q + dq, self.lower, self.upper)
            if np.linalg.norm(err) < 1e-4:
                break
        return q
