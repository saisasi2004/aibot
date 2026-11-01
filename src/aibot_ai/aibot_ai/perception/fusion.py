"""Depth fusion: 2D detection box + depth image -> 3D object estimate in the world frame.

No segmentation model (SAM etc.): on a tabletop the table plane separates objects for free.
  1. Back-project only the pixels inside the box (stride `stride`), transform to the world frame.
  2. Table plane: RANSAC on a sparse full-frame cloud, accepted only if it is horizontal and
     within `plane_tolerance` of the known table height (TF/config), else that known plane.
  3. Keep box points 5 mm .. 30 cm above the plane; voxel-connected components (1 cm voxels),
     largest component = the object (drops table edges, floor and neighbours peeking into the box).
  4. Top face = points within 12 mm of the highest; its minimum-area rectangle gives the
     footprint centre, extents and yaw (PCA would be ambiguous for squares, whose covariance is
     isotropic). Height = top above the plane. The camera looks ~60 deg down, so the top
     face is seen whole while the back faces are hidden; using the top face keeps the footprint
     unbiased.
  5. Grasp pose for a top-down grasp: centre of the top face, yaw along the minor axis (the
     fingers close across the narrow side).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Plane:
    normal: np.ndarray    # unit, pointing up
    d: float              # normal . p + d = 0

    def height(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d

    def z_at(self, x: float, y: float) -> float:
        n = self.normal
        return float(-(self.d + n[0] * x + n[1] * y) / n[2])

    @classmethod
    def horizontal(cls, z: float) -> "Plane":
        return cls(np.array([0.0, 0.0, 1.0]), -z)


@dataclass
class ObjectEstimate:
    centroid: np.ndarray          # world, metres
    size: np.ndarray              # major, minor (footprint, along yaw / yaw+90 deg), height
    yaw: float                    # rad, direction of the major footprint axis
    top_z: float
    grasp: np.ndarray             # x, y, z (centre of the top face)
    grasp_yaw: float              # direction across which the hand closes (minor axis)
    n_points: int
    valid_fraction: float         # finite depth pixels in the box


def backproject(depth: np.ndarray, K: Intrinsics, box: tuple[int, int, int, int] | None = None,
                stride: int = 2, near: float = 0.15, far: float = 3.0) -> tuple[np.ndarray, float]:
    """Camera-optical-frame points of the (boxed) depth image, and the valid-pixel fraction."""
    h, w = depth.shape
    x0, y0, x1, y1 = box if box is not None else (0, 0, w, h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    vs, us = np.mgrid[y0:y1:stride, x0:x1:stride]
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > near) & (z < far)
    frac = float(np.mean(ok)) if ok.size else 0.0
    z, u, v = z[ok], us[ok], vs[ok]
    return np.stack([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z], axis=1), frac


def backproject_pixels(depth: np.ndarray, K: Intrinsics, stride: int = 2, near: float = 0.15,
                      far: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """All valid pixels (stride) as camera-frame points, with their (u, v) pixel coordinates."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > near) & (z < far)
    z, u, v = z[ok], us[ok], vs[ok]
    pts = np.stack([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z], axis=1)
    return pts, np.stack([u, v], axis=1)


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def fit_table_plane(points: np.ndarray, expected_z: float, rng: np.random.Generator, iters: int = 80,
                    inlier: float = 0.006, plane_tolerance: float = 0.02) -> tuple[Plane, bool]:
    """RANSAC table plane near expected_z. Returns (plane, fitted?)."""
    cand = points[np.abs(points[:, 2] - expected_z) < 0.05]
    if len(cand) < 50:
        return Plane.horizontal(expected_z), False
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if n[2] < 0:
            n = -n
        if n[2] < 0.98:                       # table must be horizontal (within ~11 deg)
            continue
        d = -float(n @ a)
        count = int(np.sum(np.abs(cand @ n + d) < inlier))
        if count > best_n:
            best, best_n = Plane(n, d), count
    if best is None or abs(best.z_at(0.4, 0.0) - expected_z) > plane_tolerance:
        return Plane.horizontal(expected_z), False
    return best, True


def cluster_indices(points: np.ndarray, voxel: float = 0.01, min_points: int = 1) -> list[np.ndarray]:
    """Index arrays of every 26-connected voxel component (largest first) with >= min_points points."""
    if len(points) == 0:
        return []
    keys = np.floor(points / voxel).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    seen: set[tuple[int, int, int]] = set()
    comps: list[list[int]] = []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    for start in cells:
        if start in seen:
            continue
        comp, queue = [], deque([start])
        seen.add(start)
        while queue:
            c = queue.popleft()
            comp.extend(cells[c])
            for o in offsets:
                nb = (c[0] + o[0], c[1] + o[1], c[2] + o[2])
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        if len(comp) >= min_points:
            comps.append(comp)
    return [np.array(c) for c in sorted(comps, key=len, reverse=True)]


def clusters(points: np.ndarray, voxel: float = 0.01, min_points: int = 1) -> list[np.ndarray]:
    """Points of every 26-connected voxel component (largest first)."""
    return [points[i] for i in cluster_indices(points, voxel, min_points)]


def largest_cluster(points: np.ndarray, voxel: float = 0.01) -> np.ndarray:
    """Points of the largest 26-connected voxel component."""
    found = clusters(points, voxel)
    return found[0] if found else points


def estimate_object(points_world: np.ndarray, plane: Plane, valid_fraction: float = 1.0,
                    min_height: float = 0.005, max_height: float = 0.30, top_band: float = 0.012,
                    min_points: int = 20) -> ObjectEstimate | None:
    h = plane.height(points_world)
    pts = points_world[(h > min_height) & (h < max_height)]
    pts = largest_cluster(pts)
    if len(pts) < min_points:
        return None
    hh = plane.height(pts)
    top = float(np.percentile(hh, 97))
    face = pts[hh > top - top_band]
    if len(face) < 8:
        face = pts
    (cx, cy), (w, l), angle = cv2.minAreaRect(face[:, :2].astype(np.float32))
    yaw = math.radians(angle)
    if l > w:                                 # make (ext_a, yaw) the long side
        w, l, yaw = l, w, yaw + math.pi / 2
    ext_a, ext_b = float(w), float(l)
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    center = np.array([cx, cy], dtype=float)
    minor = np.array([-math.sin(yaw), math.cos(yaw)])
    z_table = plane.z_at(*center)
    centroid = np.array([center[0], center[1], z_table + top / 2])
    grasp = np.array([center[0], center[1], z_table + top])
    return ObjectEstimate(centroid, np.array([ext_a, ext_b, top]), yaw, z_table + top, grasp,
                          math.atan2(minor[1], minor[0]), len(pts), valid_fraction)
