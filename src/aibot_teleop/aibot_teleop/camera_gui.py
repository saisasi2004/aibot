"""PyQt5 control window for webcam_node: live annotated video, teleop status and controls.

Runs inside the webcam_node process. Each annotated frame arrives straight from the inference
thread through a queued Qt signal, with no ROS image transport and no polling, so the picture is
about one frame behind the camera. The buttons call retarget_node's services.

Mode toggle (Mimic | Physical AI) and STOP call mode_manager_node asynchronously: a switch takes
1-2 s (the robot is allowed to settle) and the event loop never blocks on it.

In Physical AI mode the video shows the robot's head camera (/head_camera/image, subscribed only
in that mode so Gazebo does not render it otherwise) with perception's boxes and ids, and a chat
panel under it talks to planner_node (/aibot/chat/user and /aibot/chat/robot) and shows what the
planner is doing (/aibot/planner/status).

Keys:  Tab toggle mode | Esc STOP | C calibrate (T-pose) | Space pause / resume | M mirror view
       S skeleton | Q quit   (while typing in the chat box, letters go to the text, Esc still stops)
"""
from __future__ import annotations

import html
import json
import os
import signal
import time

# pip's opencv-contrib-python points Qt at its own bundled plugins when cv2 is imported. They do
# not match the system PyQt5, and the window would fail with "could not load the Qt platform
# plugin xcb". Drop those variables before Qt starts.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    os.environ.pop(_var, None)

import numpy as np  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import SetBool, Trigger  # noqa: E402

from sensor_msgs.msg import Image  # noqa: E402

from aibot_msgs.msg import DetectedObjectArray, GraspState  # noqa: E402
from aibot_msgs.srv import SetMode  # noqa: E402

from .utils.mode_gate import MODE_TOPIC, mode_qos  # noqa: E402
from .utils.video_source import HEAD, WEBCAM, VideoSource  # noqa: E402

SIDES = ("left", "right")
OK, BAD, WARN = "#3ccf5a", "#e0463c", "#f0b429"
BANNERS = {"CALIBRATING": ("#f0b429", "#000000"), "LOST": ("#e0463c", "#ffffff"), "PAUSED": ("#5f6b7a", "#ffffff"),
           "RESUMING": ("#1f6feb", "#ffffff")}
MODE_BANNERS = {"ai": ("PHYSICAL AI: robot head camera (webcam paused)", "#7c3aed"), "switching": ("SWITCHING MODE...", "#5f6b7a"),
                "hold": ("STOPPED: holding pose. Choose a mode to continue.", "#e0463c"),
                "boot": ("STARTING...", "#5f6b7a")}

STYLE = """
QWidget { background: #1e2227; color: #e6e6e6; font-size: 13px; }
QGroupBox { border: 1px solid #3a4049; border-radius: 6px; margin-top: 10px; padding: 8px 6px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #9aa4b2; }
QPushButton { background: #2d333b; border: 1px solid #444c56; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { background: #373e47; }
QPushButton#primary { background: #1f6feb; border-color: #1f6feb; color: white; font-weight: bold; }
QPushButton#mode { padding: 9px 6px; font-weight: bold; }
QPushButton#mode:checked { background: #7c3aed; border-color: #7c3aed; color: white; }
QPushButton#stop { background: #d0302a; border: 2px solid #ff7b72; color: white; font-size: 16px; font-weight: bold; padding: 10px; }
QPushButton#stop:hover { background: #e5534b; }
QProgressBar { border: 1px solid #444c56; border-radius: 4px; background: #2d333b; height: 14px; text-align: center; }
QProgressBar::chunk { background: #1f6feb; border-radius: 3px; }
QLabel#video { background: #0d0f12; border-radius: 6px; }
QLabel#hint { color: #9aa4b2; font-size: 12px; }
QTextBrowser { background: #0d0f12; border: 1px solid #3a4049; border-radius: 4px; }
QLineEdit { background: #0d0f12; border: 1px solid #444c56; border-radius: 4px; padding: 6px; }
"""


class _Bridge(QtCore.QObject):
    """Carries data from ROS / inference threads into the Qt thread (queued signals)."""

    frame = QtCore.pyqtSignal(object, object)   # annotated BGR frame, stats dict
    teleop_status = QtCore.pyqtSignal(str)
    grasp_status = QtCore.pyqtSignal(str)
    grasp = QtCore.pyqtSignal(object)           # GraspState
    reply = QtCore.pyqtSignal(str, bool, str)   # service, success, message
    mode = QtCore.pyqtSignal(str)               # /aibot/mode
    mode_status = QtCore.pyqtSignal(str)        # /aibot/mode_status
    chat = QtCore.pyqtSignal(str)               # /aibot/chat/robot
    head = QtCore.pyqtSignal(object)            # head-camera frame (BGR), AI mode only
    planner = QtCore.pyqtSignal(str)            # /aibot/planner/status (JSON)


class CameraWindow(QtWidgets.QWidget):
    def __init__(self, node, bridge: _Bridge) -> None:
        super().__init__()
        self.node, self.bridge = node, bridge
        self.paused = False
        self._status = ""
        self._status_time = 0.0
        self.setWindowTitle("Aibot teleop - camera")
        self.setStyleSheet(STYLE)

        self.video = QtWidgets.QLabel("waiting for camera...")
        self.video.setObjectName("video")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(480, 360)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        title = QtWidgets.QLabel("<b style='font-size:16px'>Aibot teleop</b>")
        self.mode = "boot"
        self.source = VideoSource()
        mode_row = QtWidgets.QHBoxLayout()
        self.mode_btns = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for key, text in (("mimic", "Mimic"), ("ai", "Physical AI")):
            b = QtWidgets.QPushButton(text)
            b.setObjectName("mode")
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=key: self.request_mode(k))
            group.addButton(b)
            mode_row.addWidget(b)
            self.mode_btns[key] = b
        self.mode_widget = QtWidgets.QWidget()
        self.mode_widget.setLayout(mode_row)
        self.mode_label = QtWidgets.QLabel("mode: starting...")
        self.mode_label.setWordWrap(True)
        self.stop_btn = QtWidgets.QPushButton("STOP  (Esc)")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.clicked.connect(self.stop)
        self.tracking = QtWidgets.QLabel()
        self.perf = QtWidgets.QLabel("-")

        teleop_box = QtWidgets.QGroupBox("Teleop")
        self.status = QtWidgets.QLabel("waiting for retarget_node...")
        self.status.setWordWrap(True)
        self.status.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.status.setMinimumHeight(5 * self.status.fontMetrics().lineSpacing())  # headline + 4 detail lines
        tl = QtWidgets.QVBoxLayout(teleop_box)
        tl.addWidget(self.status)

        hands_box = QtWidgets.QGroupBox("Robot hands")
        hl = QtWidgets.QGridLayout(hands_box)
        self.closure_bar, self.grip_label = {}, {}
        for row, side in enumerate(SIDES):
            hl.addWidget(QtWidgets.QLabel(side.capitalize()), row, 0)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("closure %p%")
            self.closure_bar[side] = bar
            hl.addWidget(bar, row, 1)
            self.grip_label[side] = QtWidgets.QLabel("-")
            self.grip_label[side].setMinimumWidth(70)
            hl.addWidget(self.grip_label[side], row, 2)
        self.holding = QtWidgets.QLabel("holding: -")
        hl.addWidget(self.holding, 2, 0, 1, 3)

        self.calib_btn = QtWidgets.QPushButton("Calibrate T-pose  (C)")
        self.calib_btn.setObjectName("primary")
        self.pause_btn = QtWidgets.QPushButton("Pause teleop  (Space)")
        self.mirror_cb = QtWidgets.QCheckBox("Mirror view  (M)")
        self.mirror_cb.setChecked(node.mirror_debug)
        self.skeleton_cb = QtWidgets.QCheckBox("Show skeleton  (S)")
        self.skeleton_cb.setChecked(node.draw_skeleton)
        hint = QtWidgets.QLabel(
            "Stand 1-2 m from the camera with your upper body in view. Calibrate by holding a "
            "T-pose for 3 s. To pick something up, lift your hand above the table, move it "
            "over the object, lower it and close your fist. Open your hand to release.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)

        panel = QtWidgets.QVBoxLayout()
        for w in (title, self.mode_widget, self.mode_label, self.stop_btn, self.tracking, self.perf, teleop_box,
                  hands_box, self.calib_btn, self.pause_btn,
                  self.mirror_cb, self.skeleton_cb, hint):
            panel.addWidget(w)
        panel.addStretch(1)
        side_panel = QtWidgets.QWidget()
        side_panel.setLayout(panel)
        side_panel.setFixedWidth(330)
        self.chat_box = QtWidgets.QGroupBox("Physical AI: tell the robot what to do")
        cl = QtWidgets.QVBoxLayout(self.chat_box)
        self.chat_log = QtWidgets.QTextBrowser()
        self.chat_log.setFixedHeight(7 * self.chat_log.fontMetrics().lineSpacing())
        self.planner_state = QtWidgets.QLabel("planner: -")
        self.planner_state.setObjectName("hint")
        entry_row = QtWidgets.QHBoxLayout()
        self.chat_entry = QtWidgets.QLineEdit()
        self.chat_entry.setPlaceholderText("e.g. pick the red cube and put it in the bin   (Enter to send)")
        self.chat_entry.returnPressed.connect(self.send_chat)
        send = QtWidgets.QPushButton("Send")
        send.clicked.connect(self.send_chat)
        entry_row.addWidget(self.chat_entry, 1)
        entry_row.addWidget(send)
        for w in (self.chat_log, self.planner_state):
            cl.addWidget(w)
        cl.addLayout(entry_row)
        self.chat_box.setVisible(False)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.video, 1)
        left.addWidget(self.chat_box)
        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(left, 1)
        root.addWidget(side_panel)

        self.calib_btn.clicked.connect(self.calibrate)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.mirror_cb.toggled.connect(lambda on: setattr(node, "mirror_debug", on))
        self.skeleton_cb.toggled.connect(lambda on: setattr(node, "draw_skeleton", on))
        for key, slot in (("C", self.calibrate), ("Space", self.toggle_pause), ("M", self.mirror_cb.toggle),
                          ("S", self.skeleton_cb.toggle), ("Q", self.close), ("Esc", self.stop),
                          ("Tab", lambda: self.request_mode("mimic" if self.mode == "ai" else "ai"))):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, slot)

        bridge.frame.connect(self.on_frame)
        bridge.teleop_status.connect(self.on_status)
        bridge.grasp_status.connect(lambda s: self.holding.setText(f"holding: {s}"))
        bridge.grasp.connect(self.on_grasp)
        bridge.reply.connect(self.on_reply)
        bridge.mode.connect(self.on_mode)
        bridge.mode_status.connect(lambda t: self.mode_label.setText(f"mode: {self.mode} - {t}"))
        bridge.chat.connect(lambda t: self._chat_line("robot", t))
        bridge.head.connect(self.on_head)
        bridge.planner.connect(self.on_planner)
        self._set_tracking(node.tracking)

    # ------------------------------------------------------------------ slots
    def on_head(self, img: np.ndarray) -> None:
        if self.source.accepts(HEAD):
            self._show(img)

    def on_frame(self, img: np.ndarray, stats: dict) -> None:
        if self.source.accepts(WEBCAM):
            self._show(img, stats)

    def _show(self, img: np.ndarray, stats: dict | None = None) -> None:
        stats = stats or {}
        h, w = img.shape[:2]
        qimg = QtGui.QImage(img.data, w, h, img.strides[0], QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video.size(), QtCore.Qt.KeepAspectRatio,
                                                   QtCore.Qt.SmoothTransformation)
        banner = next((v for k, v in BANNERS.items() if self._status.startswith(k)), None)
        text = self._status.split("|")[0].strip()
        if self.mode in MODE_BANNERS:
            text, color = MODE_BANNERS[self.mode]
            banner = (color, "#ffffff")
        if banner is not None:
            painter = QtGui.QPainter(pix)
            bg, fg = banner
            rect = QtCore.QRect(0, pix.height() - 44, pix.width(), 44)
            painter.fillRect(rect, QtGui.QColor(bg))
            painter.setPen(QtGui.QColor(fg))
            font = painter.font()
            font.setPointSize(13)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, QtCore.Qt.AlignCenter, text)
            painter.end()
        self.video.setPixmap(pix)
        if stats.get("fps"):
            self.perf.setText(f"{stats['fps']:.0f} fps  |  inference {stats['infer_ms']:.0f} ms  |  "
                              f"capture->ROS {stats['latency_ms']:.0f} ms")
        self._set_tracking(self.node.tracking)
        if self._status_time and time.monotonic() - self._status_time > 2.0:
            self.status.setText("<span style='color:%s'>no status from retarget_node (is it running?)</span>" % WARN)

    def on_status(self, text: str) -> None:
        self._status, self._status_time = text, time.monotonic()
        color = BAD if text.startswith("LOST") else WARN if text.startswith(("CALIBRATING", "PAUSED")) else OK
        self.status.setText(f"<span style='color:{color}'><b>{text.split(' |')[0]}</b></span>"
                            + "".join(f"<br>{part.strip()}" for part in text.split("|")[1:]))

    def on_grasp(self, msg: GraspState) -> None:
        self.closure_bar[msg.side].setValue(int(100 * msg.closure))
        if not msg.tracking:
            self.grip_label[msg.side].setText(f"<span style='color:{WARN}'>no hand</span>")
        else:
            self.grip_label[msg.side].setText(f"<span style='color:{OK if msg.closed else '#9aa4b2'}'>"
                                              f"{'CLOSED' if msg.closed else 'open'}</span>")

    def _chat_line(self, who: str, text: str) -> None:
        color = "#9ecbff" if who == "you" else "#e6e6e6"
        self.chat_log.append(f"<span style='color:{color}'><b>{who}:</b> {html.escape(text)}</span>")

    def send_chat(self) -> None:
        text = self.chat_entry.text().strip()
        if not text:
            return
        self.chat_entry.clear()
        self._chat_line("you", text)
        self.node.chat_pub.publish(String(data=text))

    def on_planner(self, raw: str) -> None:
        try:
            st = json.loads(raw)
        except ValueError:
            return
        state = st.get("state", "?")
        detail = st.get("step") or st.get("request") or ""
        rep = st.get("report") or {}
        if state == "idle" and rep:
            fm = rep.get("first_motion_s")
            detail = (f"last: {rep.get('kind')} in {rep.get('total_s', 0):.0f} s, model: {rep.get('backend') or '-'}"
                      + (f", first motion after {fm:.0f} s" if fm is not None else ""))
        self.planner_state.setText(f"planner: {state}" + (f" - {detail}" if detail else ""))

    def on_mode(self, mode: str) -> None:
        self.chat_box.setVisible(mode == "ai")
        source = self.source.set_mode(mode)
        if hasattr(self.node, "head_camera"):
            self.node.head_camera(source == HEAD)    # stays on while switching or stopped from AI
        self.mode = mode
        for key, b in self.mode_btns.items():
            b.setChecked(key == mode)
            b.setEnabled(mode != "switching")
        self.mode_label.setText(f"mode: {mode}")

    def request_mode(self, mode: str) -> None:
        self.mode_btns[mode].setChecked(self.mode == mode)     # stays as is until /aibot/mode confirms
        if mode != self.mode:
            self.mode_label.setText(f"mode: {self.mode} - requesting {mode}...")
            self._call(self.node.set_mode_client, SetMode.Request(mode=mode), "set_mode")

    def stop(self) -> None:
        self._call(self.node.stop_client, Trigger.Request(), "stop")

    def on_reply(self, service: str, success: bool, message: str) -> None:
        if service in ("set_mode", "stop"):
            color = OK if success else BAD
            self.mode_label.setText(f"mode: {self.mode} - <span style='color:{color}'>{message}</span>")
            return
        if service == "enable" and success:
            self.paused = message == "paused"
            self.pause_btn.setText("Resume teleop  (Space)" if self.paused else "Pause teleop  (Space)")
        elif not success:
            self.status.setText(f"<span style='color:{BAD}'>{service}: {message}</span>")

    def _set_tracking(self, t: dict) -> None:
        def dot(name: str, on: bool) -> str:
            return f"<span style='color:{OK if on else BAD}'>&#9679;</span> {name}"
        self.tracking.setText("&nbsp;&nbsp;".join((dot("body", t["body"]), dot("left hand", t["left"]),
                                                   dot("right hand", t["right"]))))

    # ------------------------------------------------------------------ actions
    def calibrate(self) -> None:
        self._call(self.node.calibrate_client, Trigger.Request(), "calibrate")

    def toggle_pause(self) -> None:
        self._call(self.node.enable_client, SetBool.Request(data=self.paused), "enable")

    def _call(self, client, request, name: str) -> None:
        if not client.service_is_ready():
            owner = "mode_manager_node" if name in ("set_mode", "stop") else "retarget_node"
            self.bridge.reply.emit(name, False, f"{owner} is not running")
            return
        future = client.call_async(request)
        future.add_done_callback(lambda f: self.bridge.reply.emit(
            name, bool(f.result() and f.result().success), f.result().message if f.result() else "no reply"))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.node.frame_callback = None
        QtWidgets.QApplication.quit()
        event.accept()


class CameraGui:
    """Create before the ROS executor starts spinning (subscriptions/clients are added here)."""

    def __init__(self, node) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["aibot_camera"])
        self.bridge = _Bridge()
        node.calibrate_client = node.create_client(Trigger, "/teleop/calibrate")
        node.enable_client = node.create_client(SetBool, "/teleop/enable")
        node.set_mode_client = node.create_client(SetMode, "/aibot/set_mode")
        node.stop_client = node.create_client(Trigger, "/aibot/stop")
        node.create_subscription(String, MODE_TOPIC, lambda m: self.bridge.mode.emit(m.data), mode_qos())
        node.create_subscription(String, "/aibot/mode_status", lambda m: self.bridge.mode_status.emit(m.data), 5)
        node.chat_pub = node.create_publisher(String, "/aibot/chat/user", 10)
        node.create_subscription(String, "/aibot/chat/robot", lambda m: self.bridge.chat.emit(m.data), 20)
        node.create_subscription(String, "/aibot/planner/status", lambda m: self.bridge.planner.emit(m.data), 5)
        self._objects = None
        self._head_sub = None
        node.create_subscription(DetectedObjectArray, "/aibot/perception/objects",
                                 lambda m: setattr(self, "_objects", m), 5)
        node.head_camera = self._head_camera
        node.create_subscription(String, "/teleop/status", lambda m: self.bridge.teleop_status.emit(m.data), 5)
        node.create_subscription(String, "/teleop/grasp_status", lambda m: self.bridge.grasp_status.emit(m.data), 5)
        for side in SIDES:
            # rclpy inspects callback signatures, which fails on PyQt built-ins: wrap emit in a lambda
            node.create_subscription(GraspState, f"/teleop/grasp_state/{side}", lambda m: self.bridge.grasp.emit(m), 5)
        self.node = node
        self.window = CameraWindow(node, self.bridge)
        node.frame_callback = lambda img, stats: self.bridge.frame.emit(img, stats)

    def _head_camera(self, on: bool) -> None:
        if on and self._head_sub is None:
            self._head_sub = self.node.create_subscription(Image, "/head_camera/image", self._on_head_image, 1)
        elif not on and self._head_sub is not None:
            self.node.destroy_subscription(self._head_sub)
            self._head_sub = None

    def _on_head_image(self, msg: Image) -> None:
        """ROS thread: rgb8 -> BGR with perception's boxes and ids drawn on, then to the Qt thread."""
        import cv2
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
        arr = self._objects
        for o in (arr.objects if arr is not None else []):
            u0, v0, u1, v1 = (int(v) for v in o.box)
            if u1 <= u0 or v1 <= v0 or o.held_by:
                continue
            cv2.rectangle(img, (u0, v0), (u1, v1), (255, 200, 40), 2)
            cv2.putText(img, o.id, (u0, max(12, v0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, o.id, (u0, max(12, v0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        self.bridge.head.emit(img)

    def exec(self) -> int:
        self.window.resize(1180, 740)
        self.window.show()
        signal.signal(signal.SIGINT, lambda *_: self.app.quit())   # Ctrl-C / ros2 launch shutdown
        keepalive = QtCore.QTimer()       # gives Python a chance to run the signal handler
        keepalive.timeout.connect(lambda: None)
        keepalive.start(200)
        return self.app.exec_()
