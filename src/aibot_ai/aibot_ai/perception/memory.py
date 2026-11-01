"""Object memory: stable ids for detected objects across frames.

Data association is greedy nearest-neighbour on world position, gated by `match_distance` and
by compatible label/colour, so a red cube is never merged into a blue one. Positions and sizes are
exponentially smoothed (`alpha` weight of the new observation). Objects not seen for `expiry`
seconds are forgotten, except while held (in the hand they are occluded or moving, and the
skill server marks them held once the grasp is verified).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fusion import ObjectEstimate


@dataclass
class TrackedObject:
    id: str
    label: str
    color: str
    position: np.ndarray
    size: np.ndarray
    yaw: float
    grasp: np.ndarray
    grasp_yaw: float
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    held_by: str = ""
    votes: dict[str, int] = field(default_factory=dict)   # label -> frames; label = the majority
    box: tuple[int, int, int, int] = (0, 0, 0, 0)          # image box of the last observation (for display)
    inside: str = ""          # container id it was put into (not separable in depth from it)

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "color": self.color,
                "position": [round(float(v), 3) for v in self.position],
                "size": [round(float(v), 3) for v in self.size], "yaw": round(self.yaw, 3),
                "confidence": round(self.confidence, 2), "held_by": self.held_by, "inside": self.inside}


@dataclass
class ObjectMemory:
    match_distance: float = 0.06
    alpha: float = 0.5
    expiry: float = 5.0
    tentative_hits: int = 3          # seen fewer times than this = tentative ...
    tentative_expiry: float = 3.0    # ... and forgotten sooner (one-frame artifacts)
    objects: dict[str, TrackedObject] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _compatible(a_label: str, a_color: str, b_label: str, b_color: str) -> bool:
        return (not a_label or not b_label or a_label == b_label) and (not a_color or not b_color or a_color == b_color)

    def _new_id(self, label: str, color: str) -> str:
        base = "_".join(x for x in (color, label or "object") if x)
        self._counters[base] = self._counters.get(base, 0) + 1
        return f"{base}_{self._counters[base]}"

    def update(self, observations: list[tuple], now: float) -> list[str]:
        """observations: (label, colour, estimate, detector score[, image box]). Returns ids seen this frame."""
        boxes = [tuple(o[4]) if len(o) > 4 else (0, 0, 0, 0) for o in observations]
        observations = [tuple(o[:4]) for o in observations]
        free = {k: o for k, o in self.objects.items() if not o.held_by}
        pairs = []
        for i, (label, color, est, _s) in enumerate(observations):
            for k, o in free.items():
                d = float(np.linalg.norm(o.position[:2] - est.centroid[:2]))
                # same place, same colour, different label = shape flicker of one object
                flicker = d <= 0.5 * self.match_distance and (not color or not o.color or color == o.color)
                if d <= self.match_distance and (self._compatible(label, color, o.label, o.color) or flicker):
                    pairs.append((d, i, k))
        used_obs, used_obj, seen = set(), set(), []
        for _d, i, k in sorted(pairs):
            if i in used_obs or k in used_obj:
                continue
            used_obs.add(i)
            used_obj.add(k)
            label, color, est, score = observations[i]
            o = self.objects[k]
            a = self.alpha
            o.position = (1 - a) * o.position + a * est.centroid
            o.size = (1 - a) * o.size + a * est.size
            o.grasp = (1 - a) * o.grasp + a * est.grasp
            o.yaw = math.atan2((1 - a) * math.sin(o.yaw) + a * math.sin(est.yaw), (1 - a) * math.cos(o.yaw) + a * math.cos(est.yaw))
            o.grasp_yaw = est.grasp_yaw
            o.confidence = (1 - a) * o.confidence + a * score * est.valid_fraction
            if label:
                o.votes[label] = o.votes.get(label, 0) + 1
                o.label = max(o.votes, key=o.votes.get)
            o.color = o.color or color
            o.last_seen, o.hits, o.box = now, o.hits + 1, boxes[i]
            seen.append(k)
        for i, (label, color, est, score) in enumerate(observations):
            if i in used_obs:
                continue
            k = self._new_id(label, color)
            self.objects[k] = TrackedObject(k, label, color, est.centroid.copy(), est.size.copy(), est.yaw,
                                            est.grasp.copy(), est.grasp_yaw, score * est.valid_fraction, now, now,
                                            votes={label: 1} if label else {}, box=boxes[i])
            seen.append(k)
        self.expire(now)
        return seen

    def expire(self, now: float) -> None:
        for k in [k for k, o in self.objects.items() if not o.held_by and not (o.inside and o.inside in self.objects)
                  and now - o.last_seen > (self.expiry if o.hits >= self.tentative_hits else self.tentative_expiry)]:
            del self.objects[k]

    def set_held(self, object_id: str, side: str, position: np.ndarray | None = None, inside: str = "",
                 now: float | None = None) -> bool:
        """Held by `side`, or released ("") at `position`, optionally into container `inside`."""
        o = self.objects.get(object_id)
        if o is None:
            return False
        o.held_by, o.inside = side, ("" if side else inside)
        if position is not None:
            self.move(object_id, position)
        if now is not None:
            o.last_seen = now          # released: the expiry clock restarts from the release
        return True

    def move(self, object_id: str, position: np.ndarray) -> None:
        """Update a held/placed object's believed position (e.g. after place) until seen again."""
        o = self.objects.get(object_id)
        if o is not None:
            dz = position - o.position
            o.position, o.grasp = position.copy(), o.grasp + dz
