"""Which video the camera window shows: decided by the mode alone, never by which frame came last.

Physical AI shows the robot's head camera, mimic the webcam. While switching or stopped the current
source stays (no flash of the other one). Frames from the other source are dropped. Pure logic.
"""
from __future__ import annotations

WEBCAM, HEAD = "webcam", "head"
SOURCE_FOR_MODE = {"ai": HEAD, "mimic": WEBCAM}


class VideoSource:
    def __init__(self) -> None:
        self.current = WEBCAM
        self.swaps = 0

    def set_mode(self, mode: str) -> str:
        """The source for this mode; other modes (boot, switching, hold) keep the current one."""
        wanted = SOURCE_FOR_MODE.get(mode, self.current)
        if wanted != self.current:
            self.current = wanted
            self.swaps += 1
        return self.current

    def accepts(self, source: str) -> bool:
        return source == self.current
