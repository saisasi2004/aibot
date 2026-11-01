"""Class-agnostic tabletop segmentation and shape/colour labelling from the head RGB-D camera.

The default localiser of the Physical AI mode (detector:=tabletop). Measured on the sim scenes,
YOLO-World-s found ~2 of 4 synthetic, flat-shaded objects at <0.3 confidence, while depth
cleanly separates anything standing on the table. Camera only: no simulator ground truth.

  1. All depth pixels (stride) -> world points; table plane by RANSAC (fusion.fit_table_plane).
  2. Keep points 6 mm..30 cm above the plane, inside the table footprint, and not within
     `self_radius` of the robot's own hands/wrists (TF), then voxel-connected clusters.
  3. Each cluster -> fusion.estimate_object (centroid, footprint, height, grasp pose), then a
     label from geometry and a colour from the pixels it came from:
       hollow top ring + large           -> bin       flat + large            -> tray
       round footprint + rounded top     -> ball      round, tall (h/w > 1.5)  -> can
       round, short                      -> mug       square-ish, h ~ w         -> cube
       flat (h < 0.6 x narrow side)      -> block     otherwise                 -> box
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import dominant_color
from .fusion import ObjectEstimate, Plane, cluster_indices, estimate_object


@dataclass
class TabletopConfig:
    x_range: tuple[float, float] = (0.20, 0.60)
    y_range: tuple[float, float] = (-0.18, 0.18)
    margin: float = 0.01             # m outside the table edge still counted (objects overhanging)
    min_height: float = 0.006
    max_height: float = 0.18         # m: tallest table object is 12 cm; above that it is the robot's arm
    voxel: float = 0.01
    min_points: int = 30
    self_radius: float = 0.13        # m around each hand/wrist frame removed from the cloud


@dataclass
class Segment:
    label: str
    color: str
    estimate: ObjectEstimate
    box: tuple[int, int, int, int]   # image box of the cluster's pixels
    score: float


def shape_label(points: np.ndarray, plane: Plane, est: ObjectEstimate) -> str:
    major, minor, height = (float(v) for v in est.size)
    width = 0.5 * (major + minor)
    hh = plane.height(points)
    top = est.top_z - plane.z_at(*est.centroid[:2])
    face = points[hh > top - 0.012]
    r = np.linalg.norm(face[:, :2] - est.centroid[:2], axis=1) if len(face) else np.zeros(1)
    if width > 0.10:
        hollow = float(np.mean(r < 0.25 * width)) < 0.05
        if hollow and height > 0.02:
            return "bin"
        if height < 0.025:
            return "tray"
    circular = float(np.max(r)) / max(width / 2, 1e-6) < 1.2
    # Rounded top: the top 5 mm of a sphere is a small cap (radius sqrt(2 R 5mm), ~57% of R for
    # R = 3 cm), while a flat top (cylinder, cube) spans its whole footprint at any band height.
    full = float(np.max(np.linalg.norm(points[:, :2] - est.centroid[:2], axis=1)))
    cap = points[hh > top - 0.005]
    cap_r = float(np.max(np.linalg.norm(cap[:, :2] - est.centroid[:2], axis=1))) if len(cap) else full
    rounded_top = cap_r < 0.75 * full
    if circular:
        if rounded_top and 0.7 < height / width < 1.35:
            return "ball"
        return "can" if height / width > 1.5 else "mug"
    if height < 0.6 * minor:
        return "block"
    if abs(major - minor) < 0.2 * major and 0.7 < height / width < 1.35:
        return "cube"
    return "box"


def segment_tabletop(points_world: np.ndarray, pixels: np.ndarray, rgb: np.ndarray, plane: Plane,
                     cfg: TabletopConfig, exclude_centers: list) -> list[Segment]:
    """points_world (N,3) with their pixel coordinates `pixels` (N,2 as u,v)."""
    h = plane.height(points_world)
    x, y = points_world[:, 0], points_world[:, 1]
    keep = ((h > cfg.min_height) & (h < cfg.max_height)
            & (x > cfg.x_range[0] - cfg.margin) & (x < cfg.x_range[1] + cfg.margin)
            & (y > cfg.y_range[0] - cfg.margin) & (y < cfg.y_range[1] + cfg.margin))
    for c in exclude_centers:                # a centre, or (centre, radius) for thinner parts (forearm)
        centre, radius = (c if isinstance(c, tuple) else (c, cfg.self_radius))
        keep &= np.linalg.norm(points_world - centre, axis=1) > radius
    idx = np.nonzero(keep)[0]
    pts = points_world[idx]
    out = []
    for members in cluster_indices(pts, cfg.voxel, cfg.min_points):
        comp = pts[members]
        est = estimate_object(comp, plane, min_height=cfg.min_height, max_height=cfg.max_height,
                              min_points=cfg.min_points)
        if est is None:
            continue
        uv = pixels[idx[members]]
        box = (int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()) + 1, int(uv[:, 1].max()) + 1)
        out.append(Segment(shape_label(comp, plane, est), _cluster_color(rgb, uv), est, box,
                           float(min(1.0, len(comp) / 200.0))))
    return _drop_container_parts(out)


def _drop_container_parts(segs: list[Segment]) -> list[Segment]:
    """A wall or floor patch of a bin/tray that clustered on its own (occlusion, shading) shows up
    as a same-coloured "block" inside the container. Drop segments that are inside a container's
    footprint, have its colour and are no taller than it; real objects put in a container differ
    in colour (the pool has no grey objects) or stick out."""
    conts = [c for c in segs if c.label in ("bin", "tray")]
    keep = []
    for sg in segs:
        part = False
        for c in conts:
            if sg is c or sg.color != c.color:
                continue
            d = np.abs(sg.estimate.centroid[:2] - c.estimate.centroid[:2])
            half = 0.5 * float(c.estimate.size[0])
            if np.all(d < half) and sg.estimate.top_z <= c.estimate.top_z + 0.01:
                part = True
                break
        if not part:
            keep.append(sg)
    return keep


def _cluster_color(rgb: np.ndarray, uv: np.ndarray) -> str:
    """Colour name from the cluster's own pixels (not the whole box: neighbours may intrude)."""
    patch = rgb[uv[:, 1], uv[:, 0]][None, :, :]
    return dominant_color(np.ascontiguousarray(patch), (0, 0, patch.shape[1], 1)) if patch.size else ""
