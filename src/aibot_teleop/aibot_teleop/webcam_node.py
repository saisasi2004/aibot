"""webcam_node: laptop webcam -> MediaPipe Pose + Hand landmarkers -> ROS 2 landmark topics.

Publishes
  /teleop/body_landmarks        aibot_msgs/BodyLandmarks   33 BlazePose points of the operator
  /teleop/hand_landmarks/left   aibot_msgs/HandLandmarks   operator's left hand, 21 points
  /teleop/hand_landmarks/right  aibot_msgs/HandLandmarks   operator's right hand, 21 points
  /teleop/debug_image           sensor_msgs/Image (bgr8)   overlay, only while someone subscribes
Service
  /teleop/webcam_enable         std_srvs/SetBool           false: stop MediaPipe and release the camera
                                                           device (light off); true: reopen, with retries

With gui:=true (default in the bringup) a PyQt5 window shows the annotated video plus teleop
status and controls (camera_gui.py).

Pausing (Physical AI mode): MediaPipe work stops completely and nothing is published, but the
camera stays open and the GUI keeps showing live video, so resuming costs no device re-open and
the first landmarks arrive one frame later.

Latency design (capture -> landmarks published ~30-40 ms on a laptop CPU):
  * A capture thread keeps only the newest frame (V4L2 buffer of 1, MJPG). Processing always
    takes the latest image, so there is never a queue of stale frames.
  * Pose and hand landmarkers run in parallel: MediaPipe releases the GIL during inference,
    and the full pose model + 2 hands drop from ~45 ms sequential to ~30 ms.
  * The GUI receives frames in-process; no ROS image transport sits in the display path. The
    ROS debug image is down-scaled (debug_image_scale) and only built when subscribed.

Sides: MediaPipe runs on the RAW (un-mirrored) image and "left"/"right" are the operator's
anatomical sides as BlazePose reports them. The hand landmarker's handedness label assumes a
mirrored selfie image, so each hand is matched to the pose wrist (15/16) nearest its own wrist;
the label is only a fallback when no pose wrist is visible. Mirroring the display flips only the
picture, never the published coordinates.
"""
from __future__ import annotations

import array
import contextlib
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import SetBool

from aibot_msgs.msg import BodyLandmarks, HandLandmarks

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError as exc:  # environment problem: tell the user how to fix it
    raise SystemExit(
        f"mediapipe is not importable ({exc}).\n"
        "Rebuild with the workspace venv's Python so the node scripts use it:\n"
        "  source ~/aibot/.venv/bin/activate && cd ~/aibot && python3 -m colcon build --symlink-install\n"
        "(create the venv once with ~/aibot/scripts/setup_env.sh)") from exc

# BlazePose topology (MediaPipe indices). Odd indices are the operator's left side.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
POSE_WRIST = {"left": 15, "right": 16}
SIDES = ("left", "right")
COLOR_BGR = {"left": (60, 200, 60), "right": (30, 140, 255), "centre": (230, 230, 230)}


class LatestFrameCamera:
    """OpenCV capture on a background thread that only ever keeps the most recent frame."""

    def __init__(self, camera_id: int, width: int, height: int, fps: int, clock) -> None:
        self._cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open /dev/video{camera_id} (is another program using the camera?)")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # raw YUYV is ~10 fps at 640x480
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._clock = clock
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._stamp: Time | None = None
        self._seq = 0
        self.frames_captured = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            stamp = self._clock.now()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._stamp = frame, stamp
                self._seq += 1
                self.frames_captured += 1
                self._cond.notify_all()

    def wait_newer(self, seq: int, timeout: float) -> tuple[int, np.ndarray, Time] | None:
        """Block until a frame newer than `seq` exists; returns (seq, frame, capture stamp)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > seq or not self._running, timeout):
                return None
            if not self._running or self._frame is None:
                return None
            return self._seq, self._frame, self._stamp

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)
        self._cap.release()


def _points(landmarks) -> list[Point]:
    return [Point(x=float(lm.x), y=float(lm.y), z=float(lm.z)) for lm in landmarks]


def _image_msg(header: Header, img: np.ndarray) -> Image:
    msg = Image(header=header, height=img.shape[0], width=img.shape[1], encoding="bgr8",
                is_bigendian=0, step=img.shape[1] * 3)
    data = array.array("B")
    data.frombytes(img.tobytes())  # array('B') takes the fast path; plain bytes is checked per element
    msg.data = data
    return msg


class WebcamNode(Node):
    def __init__(self) -> None:
        super().__init__("webcam_node")
        p = self.declare_parameter
        self.camera_id = p("camera_id", 0).value
        width = p("width", 640).value
        height = p("height", 480).value
        fps = p("fps", 30).value
        pose_model = p("pose_model", "full").value          # full | lite | /abs/path.task
        hand_model = p("hand_model", "").value              # "" -> bundled hand_landmarker.task
        self.num_hands = p("num_hands", 2).value
        det_conf = p("min_detection_confidence", 0.5).value
        presence_conf = p("min_presence_confidence", 0.5).value
        track_conf = p("min_tracking_confidence", 0.5).value
        self.max_assoc_dist = p("max_hand_association_distance", 0.12).value  # normalised image units
        self.handedness_input_mirrored = p("handedness_input_mirrored", False).value
        self.gui_enabled = p("gui", False).value
        self.mirror_debug = p("mirror_debug_image", True).value
        self.draw_skeleton = p("draw_skeleton", True).value
        self.debug_scale = p("debug_image_scale", 0.5).value
        self.frame_id = p("frame_id", "webcam").value

        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path(pose_model, "pose_landmarker_{}.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=det_conf, min_pose_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf, output_segmentation_masks=False))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path(hand_model or "hand_landmarker.task", "{}")),
            running_mode=vision.RunningMode.VIDEO, num_hands=self.num_hands,
            min_hand_detection_confidence=det_conf, min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf))
        self._hand_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hands")

        self.body_pub = self.create_publisher(BodyLandmarks, "/teleop/body_landmarks", 5)
        self.hand_pubs = {s: self.create_publisher(HandLandmarks, f"/teleop/hand_landmarks/{s}", 5) for s in SIDES}
        self.debug_pub = self.create_publisher(Image, "/teleop/debug_image", 1)
        self.inference_enabled = bool(p("start_enabled", True).value)
        self._cam_args = (self.camera_id, width, height, fps, self.get_clock())
        self._cam_lock = threading.Lock()
        self._cam_generation = 0
        self.reopen_delay: float = p("reopen_delay", 0.5).value
        self.camera: LatestFrameCamera | None = None
        self.create_service(SetBool, "/teleop/webcam_enable", self._srv_enable)

        # GUI hook: called from the processing thread with (annotated BGR frame, stats dict).
        self.frame_callback: Callable[[np.ndarray, dict], None] | None = None
        self.stats: dict[str, float] = {"fps": 0.0, "infer_ms": 0.0, "latency_ms": 0.0,
                                        "body": 0.0, "left": 0.0, "right": 0.0}
        self.tracking = {"body": False, "left": False, "right": False}

        busy = False
        if self.inference_enabled:          # started for Physical AI: the camera stays closed until mimic
            try:
                self.camera = LatestFrameCamera(*self._cam_args)
            except RuntimeError as exc:     # the GUI and the AI mode must not die with the camera
                busy = True
                self.get_logger().error(f"{exc}; the window stays up and keeps trying")
        self.get_logger().info(
            f"camera /dev/video{self.camera_id} {'open' if self.camera else 'closed until mimic'} @ {fps} fps requested; "
            f"pose model '{pose_model}', {self.num_hands} hands, GUI {'on' if self.gui_enabled else 'off'}")

        self._last_ts_ms = 0
        self._acc = {"frames": 0, "body": 0, "left": 0, "right": 0, "infer_ms": 0.0, "latency_ms": 0.0}
        self._acc_t0 = time.monotonic()
        self._acc_cam0 = 0
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._process_loop, name="mediapipe", daemon=True)
        self._worker.start()
        if busy:
            threading.Thread(target=self._reopen, args=(self._cam_generation,), name="camera_open", daemon=True).start()

    # ------------------------------------------------------------------ setup helpers
    def _model_path(self, value: str, pattern: str) -> str:
        """Resolve a model short name ('full', 'lite') or file name to a path in share/models."""
        path = Path(value)
        if not path.is_absolute():
            path = Path(get_package_share_directory("aibot_teleop")) / "models" / pattern.format(value)
        if not path.is_file():
            raise SystemExit(f"MediaPipe model not found: {path}\n"
                             "Run ~/aibot/scripts/setup_env.sh (downloads the .task files), then rebuild.")
        return str(path)

    def _next_timestamp_ms(self) -> int:
        """VIDEO mode needs strictly increasing timestamps."""
        ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        return ts

    # ------------------------------------------------------------------ main loop
    def _process_loop(self) -> None:
        seq, used = 0, None
        while not self._stop.is_set():
            cam = self.camera
            if cam is None:                     # released in Physical AI mode
                self._stop.wait(0.05)
                continue
            if cam is not used:                 # reopened: its frame counter starts again
                seq, used = 0, cam
            got = cam.wait_newer(seq, timeout=1.0)
            if got is None:
                if self.camera is not cam:
                    continue
                if not self._stop.is_set():
                    self.get_logger().warn("no camera frames for 1 s", throttle_duration_sec=5.0)
                continue
            seq, frame, stamp = got
            try:
                self._process(frame, stamp)
            except Exception as exc:  # keep the node alive; a single bad frame must not kill teleop
                if self._stop.is_set() or not self.context.ok():  # shutting down mid-frame
                    return
                self.get_logger().error(f"frame processing failed: {exc!r}", throttle_duration_sec=2.0)

    def _srv_enable(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        """False releases the V4L2 device, so its light goes off and no frame exists at all; true
        reopens it in the background (opening takes up to a second, the caller must not wait)."""
        on = bool(req.data)
        self.inference_enabled = on
        with self._cam_lock:
            self._cam_generation += 1
            generation = self._cam_generation
            cam = None
            if not on:
                cam, self.camera = self.camera, None
                self.tracking = {"body": False, "left": False, "right": False}
            need_open = on and self.camera is None
        self._close_camera(cam)
        if need_open:
            threading.Thread(target=self._reopen, args=(generation,), name="camera_open", daemon=True).start()
        self.get_logger().info("camera on, inference resumed" if on else "camera released, inference stopped")
        res.success, res.message = True, "camera reopening" if need_open else ("camera on" if on else "camera released")
        return res

    def _close_camera(self, cam: "LatestFrameCamera | None") -> None:
        if cam is not None:
            cam.close()

    def _reopen(self, generation: int) -> None:
        """Open the camera in the background, until it works or the mode no longer wants it: another
        program may hold it for a long time, and video should come back as soon as that one lets go."""
        for attempt in itertools.count(1):
            if self._stop.is_set() or generation != self._cam_generation:
                return
            try:
                t0 = time.monotonic()
                cam = LatestFrameCamera(*self._cam_args)
            except RuntimeError as exc:
                self.get_logger().warn(f"camera reopen attempt {attempt}: {exc}", throttle_duration_sec=10.0)
                self._stop.wait(self.reopen_delay)
                continue
            with self._cam_lock:
                if generation == self._cam_generation:
                    self.camera = cam
                    cam = None
            if cam is not None:                 # disabled again while opening
                cam.close()
                return
            self.get_logger().info(f"camera reopened in {time.monotonic() - t0:.2f} s (attempt {attempt})")
            return

    def _process(self, frame: np.ndarray, stamp: Time) -> None:
        if not self.inference_enabled:
            self.tracking = {"body": False, "left": False, "right": False}
            if self.frame_callback is not None:
                self.frame_callback(cv2.flip(frame, 1) if self.mirror_debug else frame, dict(self.stats))
            return
        t0 = time.perf_counter()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts_ms = self._next_timestamp_ms()
        hands_future = self._hand_pool.submit(self.hands.detect_for_video, mp_image, ts_ms)
        pose = self.pose.detect_for_video(mp_image, ts_ms)
        hands = hands_future.result()
        infer_ms = (time.perf_counter() - t0) * 1000

        header = Header(stamp=stamp.to_msg(), frame_id=self.frame_id)
        h, w = frame.shape[:2]
        body = BodyLandmarks(header=header, detected=bool(pose.pose_landmarks), image_width=w, image_height=h)
        pose_img = None
        if pose.pose_landmarks:
            pose_img = pose.pose_landmarks[0]
            body.image_points = _points(pose_img)
            body.world_points = _points(pose.pose_world_landmarks[0])
            body.visibility = [float(lm.visibility) for lm in pose_img]
            body.presence = [float(lm.presence) for lm in pose_img]
        self.body_pub.publish(body)

        assignment = self._assign_hands(hands, pose_img)
        for side in SIDES:
            msg = HandLandmarks(header=header, side=side, detected=False)
            i = assignment[side]
            if i is not None:
                msg.detected = True
                msg.score = float(hands.handedness[i][0].score)
                msg.image_points = _points(hands.hand_landmarks[i])
                msg.world_points = _points(hands.hand_world_landmarks[i])
            self.hand_pubs[side].publish(msg)

        latency_ms = (self.get_clock().now() - stamp).nanoseconds * 1e-6
        self.tracking = {"body": body.detected, "left": assignment["left"] is not None,
                         "right": assignment["right"] is not None}
        self._update_stats(infer_ms, latency_ms)

        want_ros = self.debug_pub.get_subscription_count() > 0
        if self.frame_callback is not None or want_ros:
            img = self._draw(frame, pose_img, hands, assignment) if self.draw_skeleton else frame
            if self.mirror_debug:
                img = cv2.flip(img, 1)
            self._draw_labels(img, hands, assignment)
            if self.frame_callback is not None:
                self.frame_callback(img, dict(self.stats))
            if want_ros:
                small = img if self.debug_scale == 1.0 else cv2.resize(
                    img, None, fx=self.debug_scale, fy=self.debug_scale, interpolation=cv2.INTER_AREA)
                self._draw_status(small, infer_ms)
                self.debug_pub.publish(_image_msg(header, small))

    # ------------------------------------------------------------------ hand association
    def _assign_hands(self, hands, pose_lms) -> dict[str, int | None]:
        """Map detected hands to the operator's left/right hand.

        Primary cue: distance between the hand's wrist and the pose wrist of each side, solved as
        a 2x2 assignment. Fallback (no usable pose wrist): MediaPipe's handedness label, which is
        swapped on a raw (non-mirrored) image.
        """
        out: dict[str, int | None] = {"left": None, "right": None}
        n = len(hands.hand_landmarks)
        if n == 0:
            return out
        wrists = [np.array([hl[0].x, hl[0].y]) for hl in hands.hand_landmarks]
        pose_wrist: dict[str, np.ndarray] = {}
        if pose_lms is not None:
            for side, idx in POSE_WRIST.items():
                lm = pose_lms[idx]
                if lm.visibility > 0.3:
                    pose_wrist[side] = np.array([lm.x, lm.y])

        def dist(i: int, side: str) -> float:
            return float(np.linalg.norm(wrists[i] - pose_wrist[side])) if side in pose_wrist else np.inf

        if n >= 2 and len(pose_wrist) == 2:
            straight = dist(0, "left") + dist(1, "right")
            crossed = dist(0, "right") + dist(1, "left")
            pairs = [(0, "left"), (1, "right")] if straight <= crossed else [(0, "right"), (1, "left")]
        else:
            pairs = [(i, min(pose_wrist, key=lambda s: dist(i, s))) for i in range(n)] if pose_wrist else []
        for i, side in pairs:
            if dist(i, side) <= self.max_assoc_dist and out[side] is None:
                out[side] = i

        for i in range(n):  # handedness fallback for hands the pose could not place
            if i in out.values():
                continue
            label = hands.handedness[i][0].category_name.lower()  # "left" / "right"
            if not self.handedness_input_mirrored:
                label = "right" if label == "left" else "left"
            if label in out and out[label] is None:
                out[label] = i
        return out

    # ------------------------------------------------------------------ drawing
    def _draw(self, frame: np.ndarray, pose_lms, hands, assignment) -> np.ndarray:
        img = frame  # cap.read() returns a fresh array per frame, safe to draw on
        h, w = img.shape[:2]

        def px(lm) -> tuple[int, int]:
            return int(lm.x * w), int(lm.y * h)

        if pose_lms is not None:
            for a, b in POSE_CONNECTIONS:
                if min(pose_lms[a].visibility, pose_lms[b].visibility) < 0.5:
                    continue
                side = "left" if a % 2 and b % 2 else "right" if a > 0 and not a % 2 and not b % 2 else "centre"
                cv2.line(img, px(pose_lms[a]), px(pose_lms[b]), COLOR_BGR[side], 2, cv2.LINE_AA)
            for lm in pose_lms[11:25]:
                if lm.visibility >= 0.5:
                    cv2.circle(img, px(lm), 4, COLOR_BGR["centre"], -1, cv2.LINE_AA)
        for side, i in assignment.items():
            if i is None:
                continue
            hl = hands.hand_landmarks[i]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, px(hl[a]), px(hl[b]), COLOR_BGR[side], 2, cv2.LINE_AA)
            for lm in hl:
                cv2.circle(img, px(lm), 3, (255, 255, 255), -1, cv2.LINE_AA)
        return img

    def _draw_labels(self, img: np.ndarray, hands, assignment) -> None:
        """L/R tags at the wrists, drawn after any flip so the text stays readable."""
        h, w = img.shape[:2]
        for side, i in assignment.items():
            if i is None:
                continue
            x, y = hands.hand_landmarks[i][0].x * w, hands.hand_landmarks[i][0].y * h
            x = w - x if self.mirror_debug else x
            cv2.putText(img, side[0].upper(), (int(x) - 8, int(y) + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        COLOR_BGR[side], 2, cv2.LINE_AA)

    def _draw_status(self, img: np.ndarray, infer_ms: float) -> None:
        t = self.tracking
        text = (f"infer {infer_ms:3.0f} ms  body {'OK' if t['body'] else '--'}  "
                f"L {'OK' if t['left'] else '--'}  R {'OK' if t['right'] else '--'}")
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ stats / shutdown
    def _update_stats(self, infer_ms: float, latency_ms: float) -> None:
        a = self._acc
        a["frames"] += 1
        a["body"] += self.tracking["body"]
        a["left"] += self.tracking["left"]
        a["right"] += self.tracking["right"]
        a["infer_ms"] += infer_ms
        a["latency_ms"] += latency_ms
        dt = time.monotonic() - self._acc_t0
        if dt < 2.0:
            return
        n = a["frames"]
        cam = self.camera
        captured = cam.frames_captured if cam is not None else self._acc_cam0
        self.stats = {"fps": n / dt, "infer_ms": a["infer_ms"] / n, "latency_ms": a["latency_ms"] / n,
                      "body": a["body"] / n, "left": a["left"] / n, "right": a["right"] / n,
                      "camera_fps": max(0, captured - self._acc_cam0) / dt}
        if not hasattr(self, "_last_log") or time.monotonic() - self._last_log > 10.0:
            self._last_log = time.monotonic()
            s = self.stats
            self.get_logger().info(
                f"camera {s['camera_fps']:4.1f} fps | processed {s['fps']:4.1f} fps | inference {s['infer_ms']:4.0f} ms | "
                f"capture->publish {s['latency_ms']:4.0f} ms | tracked: body {100 * s['body']:3.0f}% "
                f"L-hand {100 * s['left']:3.0f}% R-hand {100 * s['right']:3.0f}%")
        self._acc = dict.fromkeys(a, 0)
        self._acc_t0 = time.monotonic()
        self._acc_cam0 = captured

    def shutdown(self) -> None:
        self._stop.set()
        with self._cam_lock:
            self._cam_generation += 1
            cam, self.camera = self.camera, None
        self._close_camera(cam)
        self._worker.join(timeout=2.0)
        self._hand_pool.shutdown(wait=True)
        self.pose.close()
        self.hands.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WebcamNode()
    if not node.gui_enabled:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            with contextlib.suppress(KeyboardInterrupt):   # second Ctrl-C during cleanup
                node.shutdown()
                node.destroy_node()
                rclpy.try_shutdown()
        return

    # GUI: Qt owns the main thread and ROS spins in the background. CameraGui.exec() replaces
    # rclpy's SIGINT handler with one that closes the window, so Ctrl-C / ros2 launch shut down
    # cleanly. The GUI adds its subscriptions/clients before the executor starts spinning.
    from .camera_gui import CameraGui  # imported lazily: PyQt5 is only needed for the GUI
    gui = CameraGui(node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="ros_spin", daemon=True)
    spin_thread.start()
    try:
        gui.exec()
    finally:
        with contextlib.suppress(KeyboardInterrupt):
            node.shutdown()
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
