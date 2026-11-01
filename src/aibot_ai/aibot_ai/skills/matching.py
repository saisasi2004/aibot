"""Match a short description ("the red cube", "leftmost ball", "red_cube_1") to perceived objects.

Shared by the locate skill and the planner, so both resolve words to ids the same way. Colour and
shape words filter; spatial superlatives (leftmost, nearest, ...) pick one of several candidates;
anything still ambiguous is returned as several candidates and the caller asks the user.
"""
from __future__ import annotations

import re
from typing import Protocol

COLORS = ("red", "blue", "yellow", "green", "purple", "silver", "white", "orange", "black", "gray")
COLOR_SYNONYMS = {"grey": "gray", "violet": "purple", "metal": "silver", "metallic": "silver"}
SHAPES = {
    "cube": "cube", "cubes": "cube", "box": "box", "boxes": "box", "block": "block", "blocks": "block",
    "brick": "block", "can": "can", "cans": "can", "cylinder": "can", "tin": "can", "mug": "mug", "cup": "mug",
    "ball": "ball", "balls": "ball", "sphere": "ball", "bin": "bin", "basket": "bin", "bucket": "bin",
    "container": "bin", "tray": "tray", "plate": "tray",
}
# superlative -> (axis, sign): pick the object with the largest sign * coordinate
SUPERLATIVES = {
    "leftmost": (1, 1), "left": (1, 1), "rightmost": (1, -1), "right": (1, -1),
    "nearest": (0, -1), "closest": (0, -1), "front": (0, -1), "farthest": (0, 1), "furthest": (0, 1), "back": (0, 1),
}


class Perceived(Protocol):
    id: str
    label: str
    color: str
    position: object      # sequence (x, y, z)


def parse(text: str) -> tuple[set[str], set[str], tuple[int, int] | None]:
    words = re.findall(r"[a-z]+", text.lower())
    colors = {COLOR_SYNONYMS.get(w, w) for w in words if COLOR_SYNONYMS.get(w, w) in COLORS}
    shapes = {SHAPES[w] for w in words if w in SHAPES}
    sup = next((SUPERLATIVES[w] for w in words if w in SUPERLATIVES), None)
    return colors, shapes, sup


def match(text: str, objects: list[Perceived]) -> list[Perceived]:
    """Candidates for `text`, best first. Exact id wins; otherwise filter by colour and shape
    (a label of "" or an object's unknown colour never matches a stated word), then apply a
    superlative if one is given."""
    t = text.strip()
    for o in objects:
        if o.id == t:
            return [o]
    colors, shapes, sup = parse(t)
    if not colors and not shapes and sup is None:
        return []
    cands = [o for o in objects if (not colors or o.color in colors) and (not shapes or o.label in shapes)]
    if sup is not None and len(cands) > 1:
        axis, sign = sup
        cands = [max(cands, key=lambda o: sign * float(o.position[axis]))]
    return cands
