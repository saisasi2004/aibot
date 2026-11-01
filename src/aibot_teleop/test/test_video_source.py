"""The window's video source depends on the mode only; inactive-source frames never get drawn."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np

from aibot_teleop.utils.video_source import HEAD, WEBCAM, VideoSource


def test_source_follows_the_mode_and_swaps_once():
    v = VideoSource()
    assert v.current == WEBCAM and v.accepts(WEBCAM) and not v.accepts(HEAD)
    assert v.set_mode("boot") == WEBCAM and v.swaps == 0
    assert v.set_mode("mimic") == WEBCAM and v.swaps == 0
    assert v.set_mode("switching") == WEBCAM and v.swaps == 0     # no flash while switching
    assert v.set_mode("ai") == HEAD and v.swaps == 1
    assert v.set_mode("ai") == HEAD and v.swaps == 1              # repeated message: no second swap
    assert v.set_mode("hold") == HEAD and v.set_mode("switching") == HEAD and v.swaps == 1
    assert v.set_mode("mimic") == WEBCAM and v.swaps == 2


def test_inactive_frames_never_reach_the_display_over_10_switches():
    """Real window (offscreen): frames from both sources arrive interleaved through 10 mode switches."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5 import QtWidgets

    from aibot_teleop import camera_gui

    class FakeNode:
        mirror_debug = True
        draw_skeleton = True
        tracking = {"body": False, "left": False, "right": False}

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
    win = camera_gui.CameraWindow(FakeNode(), camera_gui._Bridge())
    drawn = []
    show = win._show
    win._show = lambda img, stats=None: (drawn.append(int(img[0, 0, 0])), show(img, stats))[1]
    web, head = np.full((48, 64, 3), 10, np.uint8), np.full((48, 64, 3), 200, np.uint8)
    win.on_mode("mimic")
    bad = 0
    for cycle in range(10):
        target = "ai" if cycle % 2 == 0 else "mimic"
        win.on_mode("switching")
        drawn.clear()
        for _ in range(5):
            win.on_frame(web, {})
            win.on_head(head)
        before = win.source.current
        assert set(drawn) == ({10} if before == WEBCAM else {200})   # the previous source, no alternation
        win.on_mode(target)
        drawn.clear()
        for _ in range(20):
            win.on_frame(web, {})
            win.on_head(head)
            win.on_frame(web, {})
        app.processEvents()
        bad += set(drawn) != ({200} if target == "ai" else {10})
    assert bad == 0 and win.source.swaps == 10
    win.close()


def test_the_webcam_node_releases_and_never_keeps_a_frame_path_open():
    """_srv_enable(false) must hand the capture to close(); guards a return to mere pausing."""
    src = (Path(__file__).resolve().parents[1] / "aibot_teleop" / "webcam_node.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_srv_enable")
    text = ast.get_source_segment(src, fn)
    assert "self._close_camera(cam)" in text and "cam, self.camera = self.camera, None" in text
    assert "self._cap.release()" in src
