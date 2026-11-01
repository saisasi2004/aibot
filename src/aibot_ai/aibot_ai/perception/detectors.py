"""2D detectors behind one interface: where in the image is "X"?

The detector only localises (boxes + labels); 3D comes from depth fusion (fusion.py) and
understanding from the VLM, on demand. Two implementations:
  * ColorBlobDetector: HSV thresholds, no dependencies. Baseline, test double, and a fallback
    when the open-vocabulary model is not installed.
  * YoloWorldDetector: ultralytics YOLO-World (open vocabulary, CPU). Text prompts can change at
    runtime; the CLIP text embedding is only recomputed when they do.
OWL-ViT was considered and cut: it needs `transformers` plus a ~600 MB model and is several
times slower on this CPU; the interface makes it easy to add later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

# Hue ranges in OpenCV units (0-179). Saturation/value gates separate colours from gray/white/black.
COLOR_HUES: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 8), (170, 179)], "orange": [(9, 20)], "yellow": [(21, 35)], "green": [(36, 85)],
    "blue": [(95, 125)], "purple": [(126, 160)],
}
COLOR_WORDS = set(COLOR_HUES) | {"white", "black", "gray", "grey", "silver"}


@dataclass
class Detection:
    label: str                         # noun the detector matched, e.g. "cube" ("" for blobs)
    box: tuple[int, int, int, int]     # x0, y0, x1, y1 (pixels, x1/y1 exclusive)
    score: float
    color: str = ""                    # dominant colour name inside the box


def dominant_color(rgb: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """Colour name of the most saturated pixels in the central part of a box."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    crop = rgb[y0 + h // 4:y1 - h // 4 or y1, x0 + w // 4:x1 - w // 4 or x1]
    if crop.size == 0:
        return ""
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    sat = hsv[:, 1] > 80
    if np.mean(sat) < 0.3:                               # mostly unsaturated: white / gray / black
        v = float(np.median(hsv[:, 2]))
        return "white" if v > 200 else "black" if v < 60 else "silver"
    hues = hsv[sat, 0]
    counts = {name: sum(int(np.sum((hues >= a) & (hues <= b))) for a, b in ranges) for name, ranges in COLOR_HUES.items()}
    return max(counts, key=counts.get)


def split_prompt(prompt: str) -> tuple[str, str]:
    """'red cube' -> ('red', 'cube'); 'cube' -> ('', 'cube')."""
    words = prompt.lower().split()
    colors = [w for w in words if w in COLOR_WORDS]
    nouns = [w for w in words if w not in COLOR_WORDS]
    return (colors[0] if colors else ""), " ".join(nouns)


class Detector(ABC):
    name = "detector"

    @abstractmethod
    def detect(self, rgb: np.ndarray, prompts: list[str]) -> list[Detection]:
        """Boxes for the prompted things in an RGB (HxWx3 uint8) image."""


class ColorBlobDetector(Detector):
    """Saturated colour blobs (and, optionally, bright white blobs) above a minimum area.

    Ignores the noun of a prompt: anything of a prompted colour is returned; with no colour in
    the prompts every colour is returned. Gray/silver/black are not segmentable by colour alone
    (table, floor, shadows) and are only found by the open-vocabulary detector.
    """

    name = "color_blob"

    def __init__(self, min_area: int = 120, min_saturation: int = 90, min_value: int = 60) -> None:
        self.min_area, self.min_s, self.min_v = min_area, min_saturation, min_value

    def detect(self, rgb: np.ndarray, prompts: list[str]) -> list[Detection]:
        wanted = {split_prompt(p)[0] for p in prompts} - {""}
        colors = [c for c in COLOR_HUES if not wanted or c in wanted]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        out = []
        for c in colors:
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for a, b in COLOR_HUES[c]:
                mask |= cv2.inRange(hsv, (a, self.min_s, self.min_v), (b, 255, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area >= self.min_area:
                    out.append(Detection("", (int(x), int(y), int(x + w), int(y + h)), min(1.0, area / 2000.0), c))
        return out


class YoloWorldDetector(Detector):
    """Open-vocabulary YOLO-World on CPU (ultralytics)."""

    name = "yolo_world"

    def __init__(self, weights: str = "yolov8s-worldv2.pt", conf: float = 0.05, imgsz: int = 640,
                 max_det: int = 30) -> None:
        from ultralytics import YOLOWorld   # heavy import only when this detector is chosen
        self.model = YOLOWorld(weights)
        self.conf, self.imgsz, self.max_det = conf, imgsz, max_det
        self._classes: list[str] = []

    def detect(self, rgb: np.ndarray, prompts: list[str]) -> list[Detection]:
        nouns = sorted({split_prompt(p)[1] or p for p in prompts})
        if nouns != self._classes:
            self.model.set_classes(nouns)          # CLIP text encoding, only when prompts change
            self._classes = nouns
        res = self.model.predict(rgb[:, :, ::-1], conf=self.conf, imgsz=self.imgsz, max_det=self.max_det,
                                 device="cpu", verbose=False)[0]
        out = []
        for box, cls, score in zip(res.boxes.xyxy.numpy(), res.boxes.cls.numpy(), res.boxes.conf.numpy()):
            b = tuple(int(v) for v in box)
            out.append(Detection(self._classes[int(cls)], b, float(score), dominant_color(rgb, b)))
        return out


def nms(dets: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Class-agnostic non-maximum suppression (one object should give one box)."""
    def iou(a, b) -> float:
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(iou(d.box, k.box) < iou_threshold for k in keep):
            keep.append(d)
    return keep
