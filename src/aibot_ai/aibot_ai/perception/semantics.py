"""VLM helpers for perception, ROS-free: contact sheets, label cache, verification, box agreement.

The depth detector stays the source of geometry. The VLM only adds meaning: names and descriptions
for objects, a second opinion after a skill, words for failures and scene questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

LABEL_SCHEMA = {"type": "object", "properties": {"labels": {"type": "array", "items": {
    "type": "object", "properties": {"index": {"type": "integer"}, "name": {"type": "string"},
                                     "description": {"type": "string"}},
    "required": ["index", "name", "description"]}}}, "required": ["labels"]}
BOXES_SCHEMA = {"type": "object", "properties": {"objects": {"type": "array", "items": {
    "type": "object", "properties": {"name": {"type": "string"}, "box": {"type": "array", "items": {"type": "number"}}},
    "required": ["name", "box"]}}}, "required": ["objects"]}
BOXES_PROMPT = ("This is a robot's head camera looking down at a table. Give every object on the table a short name "
                "and its bounding box [x0, y0, x1, y1] with coordinates from 0 to 1000 relative to the image. JSON only.")


def label_prompt(n: int) -> str:
    return (f"Each of the {n} numbered tiles shows one object on a table, seen by a robot's camera. For each tile say "
            "what the object most likely is (name: one to three everyday words, like 'coffee mug' or 'toy block') and "
            "give a short description. Answer in JSON with one entry per tile, using the tile's number as index.")


def contact_sheet(rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], tile: int = 128, margin: int = 6,
                  columns: int = 4) -> np.ndarray:
    """Numbered crops of the objects in one image (RGB), small enough to send cheaply."""
    rows = (len(boxes) + columns - 1) // columns
    sheet = np.full((rows * tile, min(len(boxes), columns) * tile, 3), 255, np.uint8)
    h, w = rgb.shape[:2]
    for i, (u0, v0, u1, v1) in enumerate(boxes):
        crop = rgb[max(0, v0 - margin):min(h, v1 + margin), max(0, u0 - margin):min(w, u1 + margin)]
        if crop.size == 0:
            continue
        s = (tile - 8) / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        y, x = (i // columns) * tile + 4, (i % columns) * tile + 4
        sheet[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        cv2.putText(sheet, str(i), (x + 2, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    return sheet


@dataclass
class SemanticCache:
    """Labels by object id; dropped when the object moves or changes size."""
    move_tolerance: float = 0.03
    labels: dict[str, dict] = field(default_factory=dict)

    @staticmethod
    def _signature(position, size) -> np.ndarray:
        return np.concatenate([np.asarray(position, float), np.asarray(size, float)])

    def valid(self, oid: str, position, size) -> bool:
        e = self.labels.get(oid)
        return e is not None and float(np.max(np.abs(e["sig"] - self._signature(position, size)))) <= self.move_tolerance

    def store(self, oid: str, position, size, name: str, description: str) -> None:
        self.labels[oid] = {"sig": self._signature(position, size), "name": name[:40], "description": description[:120]}

    def get(self, oid: str) -> tuple[str, str]:
        e = self.labels.get(oid)
        return (e["name"], e["description"]) if e else ("", "")


def apply_labels(answer: dict, ids: list[str]) -> dict[str, tuple[str, str]]:
    """Valid entries of a label answer: {object id: (name, description)}; bad indices are ignored."""
    out = {}
    for e in answer.get("labels", []):
        i = e.get("index")
        if isinstance(i, int) and 0 <= i < len(ids) and str(e.get("name", "")).strip():
            out[ids[i]] = (str(e["name"]).strip().lower(), str(e.get("description", "")).strip())
    return out


def box_iou(a, b) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_agreement(detector: list, vlm: list, iou: float = 0.3) -> float:
    """Fraction of detector boxes with an unused VLM box overlapping by at least `iou`."""
    used, hits = set(), 0
    for d in detector:
        best = max(((box_iou(d, v), i) for i, v in enumerate(vlm) if i not in used), default=(0.0, -1))
        if best[0] >= iou:
            used.add(best[1])
            hits += 1
    return hits / len(detector) if detector else 1.0
