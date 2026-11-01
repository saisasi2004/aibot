"""Tiny numpy ray caster for perception tests: depth images of boxes/cylinders/spheres on a table."""
from __future__ import annotations

import math

import numpy as np

from aibot_ai.perception.fusion import Intrinsics

TABLE_Z, TABLE_X, TABLE_Y = 0.98, (0.20, 0.60), (-0.18, 0.18)


def camera_pose(pos=(0.10, 0.0475, 1.635), down_deg: float = 60.0) -> np.ndarray:
    """Optical frame (x right, y down, z forward) looking forward and down_deg below horizontal."""
    a = math.radians(down_deg)
    z = np.array([math.cos(a), 0.0, -math.sin(a)])
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = pos
    return T


def render(prims: list[dict], T: np.ndarray, K: Intrinsics, w: int = 320, h: int = 240) -> np.ndarray:
    vs, us = np.mgrid[0:h, 0:w]
    d_cam = np.stack([(us - K.cx) / K.fx, (vs - K.cy) / K.fy, np.ones_like(us, float)], axis=-1).reshape(-1, 3)
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    o = T[:3, 3]
    d = d_cam @ T[:3, :3].T
    t_best = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (TABLE_Z - o[2]) / d[:, 2]
        p = o + t[:, None] * d
        on = (t > 0) & (p[:, 0] >= TABLE_X[0]) & (p[:, 0] <= TABLE_X[1]) & (p[:, 1] >= TABLE_Y[0]) & (p[:, 1] <= TABLE_Y[1])
        t_best = np.where(on, t, t_best)
        tf = (0.0 - o[2]) / d[:, 2]
        t_best = np.where((tf > 0) & (tf < t_best), tf, t_best)
    for pr in prims:
        t_best = np.minimum(t_best, _hit(pr, o, d))
    depth_along_z = t_best * d_cam[:, 2]          # depth image stores z in the optical frame
    return depth_along_z.reshape(h, w).astype(np.float32)


def _hit(pr: dict, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    c = np.array(pr["center"], float)
    if pr["type"] == "box":
        yaw = pr.get("yaw", 0.0)
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        ol, dl = (o - c) @ R, d @ R
        half = np.array(pr["size"]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - ol) / dl, (half - ol) / dl
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)
    if pr["type"] == "cylinder":
        r, hh = pr["radius"], pr["length"] / 2
        oc = o - c
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (oc[0] * d[:, 0] + oc[1] * d[:, 1])
        cc = oc[0] ** 2 + oc[1] ** 2 - r * r
        disc = b * b - 4 * a * cc
        with np.errstate(invalid="ignore", divide="ignore"):
            ts = (-b - np.sqrt(disc)) / (2 * a)
        z = oc[2] + ts * d[:, 2]
        side = np.where((disc >= 0) & (ts > 0) & (np.abs(z) <= hh), ts, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = (hh - oc[2]) / d[:, 2]
        pc = oc + tc[:, None] * d
        cap = np.where((tc > 0) & (pc[:, 0] ** 2 + pc[:, 1] ** 2 <= r * r), tc, np.inf)
        return np.minimum(side, cap)
    r = pr["radius"]
    oc = o - c
    b = 2 * d @ oc
    disc = b * b - 4 * (oc @ oc - r * r)
    with np.errstate(invalid="ignore"):
        ts = (-b - np.sqrt(disc)) / 2
    return np.where((disc >= 0) & (ts > 0), ts, np.inf)


def bin_prims(cx: float, cy: float, size=0.13, height=0.06, wall=0.006) -> list[dict]:
    z0 = TABLE_Z
    return [{"type": "box", "center": (cx, cy, z0 + wall / 2), "size": (size, size, wall)},
            {"type": "box", "center": (cx + size / 2 - wall / 2, cy, z0 + height / 2), "size": (wall, size, height)},
            {"type": "box", "center": (cx - size / 2 + wall / 2, cy, z0 + height / 2), "size": (wall, size, height)},
            {"type": "box", "center": (cx, cy + size / 2 - wall / 2, z0 + height / 2), "size": (size - 2 * wall, wall, height)},
            {"type": "box", "center": (cx, cy - size / 2 + wall / 2, z0 + height / 2), "size": (size - 2 * wall, wall, height)}]
