"""Gesture library: generated motions that worked, on disk, so a repeat is instant and costs nothing.

Keyed by normalised intent ("hi", "hello" and "hey" are all "greet"). Motions aimed at a specific
object are not cached (the object moves). ROS-free.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

_GREET = re.compile(r"^(say )?(hi+|hello+|hey+|hiya|namaste|greetings|good (morning|afternoon|evening))( there| robot)?$")
_BYE = re.compile(r"^(say )?(bye+|goodbye|see you)( robot)?$")
_FILLER = re.compile(r"\b(please|the|a|an|robot|can you|could you)\b")
MAX_WORDS = 6


def intent_key(text: str) -> str:
    """Normalised request: same meaning, same key. Empty for anything too long to be a gesture."""
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if _GREET.match(t):
        return "greet"
    if _BYE.match(t):
        return "goodbye"
    t = re.sub(r"\s+", " ", _FILLER.sub(" ", t)).strip()
    return t if t and len(t.split()) <= MAX_WORDS else ""


class GestureLibrary:
    def __init__(self, path: str | Path = "", seeds: dict[str, dict] | None = None) -> None:
        self.path = Path(os.path.expanduser(str(path))) if path else None
        self.seeds = seeds or {}
        self.entries: dict[str, dict] = {}
        if self.path is not None and self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                self.entries = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self.entries = {}                     # a damaged cache only costs regenerating

    def get(self, key: str) -> dict | None:
        """A generated entry {motion, message}, or None. Seeds are the fallback, not a cache hit."""
        return self.entries.get(key) if key else None

    def seed(self, name: str) -> dict | None:
        return self.seeds.get(name)

    def store(self, key: str, motion: dict, message: str) -> None:
        if not key or motion.get("point_at"):
            return
        self.entries[key] = {"motion": motion, "message": message}
        self._save()

    def clear(self) -> int:
        n = len(self.entries)
        self.entries = {}
        self._save()
        return n

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, delete=False) as f:
            json.dump(self.entries, f, indent=1)
        os.replace(f.name, self.path)
