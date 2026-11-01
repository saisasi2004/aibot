"""Controller groups, joint order and per-joint corrections (config/joint_mapping.yaml)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..utils.limits import JointLimits

GROUPS = ("waist", "head", "left_arm", "right_arm", "left_hand", "right_hand")


class JointMap:
    """Fixed ordering of all 43 controlled joints and the controller each belongs to.

    Retargeting writes into one 43-vector; `split` slices it back into the Float64MultiArray
    payloads of the six JointGroupPositionControllers. Per-joint corrections
        q_cmd = sign * scale * q + offset
    let you fix a sign flip or trim an offset from YAML without touching code.
    """

    def __init__(self, mapping_file: str | Path, limits_file: str | Path, limit_margin: float = 0.0) -> None:
        cfg = yaml.safe_load(Path(mapping_file).read_text())
        self.controllers: dict[str, list[str]] = {f"{g}_controller": list(cfg["controllers"][f"{g}_controller"])
                                                  for g in GROUPS}
        self.names: list[str] = [n for joints in self.controllers.values() for n in joints]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.groups: dict[str, np.ndarray] = {}
        start = 0
        for g in GROUPS:
            n = len(self.controllers[f"{g}_controller"])
            self.groups[g] = np.arange(start, start + n)
            start += n
        corr = cfg.get("corrections") or {}
        unknown = set(corr) - set(self.names)
        if unknown:
            raise KeyError(f"corrections for unknown joints: {sorted(unknown)}")
        self.sign = np.array([float(corr.get(n, {}).get("sign", 1.0)) for n in self.names])
        self.scale = np.array([float(corr.get(n, {}).get("scale", 1.0)) for n in self.names])
        self.offset = np.array([float(corr.get(n, {}).get("offset", 0.0)) for n in self.names])
        self.limits = JointLimits.from_yaml(limits_file, self.names, limit_margin)

    def correct(self, q: np.ndarray) -> np.ndarray:
        return self.sign * self.scale * q + self.offset

    def split(self, q: np.ndarray) -> dict[str, list[float]]:
        return {f"{g}_controller": [float(v) for v in q[idx]] for g, idx in self.groups.items()}

    def per_group(self, values: dict[str, float]) -> np.ndarray:
        """Expand {group: value} into a 43-vector (e.g. per-group filter or speed settings)."""
        out = np.zeros(len(self.names))
        for g, idx in self.groups.items():
            out[idx] = values[g.split("_", 1)[-1] if g not in values else g]
        return out
