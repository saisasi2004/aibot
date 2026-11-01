"""Shared kinematics: URDF joint chains, forward kinematics and Jacobian ingredients.

Used by mimic mode (retarget/arm_ik.py) and by the Physical AI skill server (aibot_ai motion/ik.py),
so both drive the robot with exactly the same model of it. Pure numpy, parsed from the URDF
string (robot_description), no ROS dependency.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def rot_log(r: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    angle = math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-6:
        return np.zeros(3)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w * (angle / (2.0 * math.sin(angle))) if angle < math.pi - 1e-3 else w * 0.5 * angle


@dataclass
class _Joint:
    name: str
    origin_r: np.ndarray
    origin_p: np.ndarray
    axis: np.ndarray | None  # None for fixed joints


class ArmChain:
    """Kinematic chain between two links of a URDF, revolute joints active."""

    def __init__(self, urdf_xml: str, base: str, tip: str) -> None:
        joints = {j.find("child").get("link"): j for j in ET.fromstring(urdf_xml).findall("joint")}
        chain: list[_Joint] = []
        link = tip
        while link != base:
            if link not in joints:
                raise ValueError(f"no path from {base} to {tip} in robot_description")
            j = joints[link]
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            axis = None
            if j.get("type") in ("revolute", "continuous"):
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                axis /= np.linalg.norm(axis)
            chain.append(_Joint(j.get("name"), _rpy(*rpy), np.array(xyz), axis))
            link = j.find("parent").get("link")
        self.joints = chain[::-1]
        self.names = [j.name for j in self.joints if j.axis is not None]

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Joint positions (n,3), world joint axes (n,3), tip rotation; all in the base frame."""
        r, p = np.eye(3), np.zeros(3)
        pos, axes, k = [], [], 0
        for j in self.joints:
            p = p + r @ j.origin_p
            r = r @ j.origin_r
            if j.axis is not None:
                pos.append(p.copy())
                axes.append(r @ j.axis)
                r = r @ _axis_rot(j.axis, float(q[k]))
                k += 1
        return np.array(pos), np.array(axes), r

    def tip_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rotation, position) of the tip link in the base frame."""
        r, p, k = np.eye(3), np.zeros(3), 0
        for j in self.joints:
            p = p + r @ j.origin_p
            r = r @ j.origin_r
            if j.axis is not None:
                r = r @ _axis_rot(j.axis, float(q[k]))
                k += 1
        return r, p
