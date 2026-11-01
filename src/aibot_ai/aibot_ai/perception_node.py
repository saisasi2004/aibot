"""perception_node: head RGB-D camera -> tracked tabletop objects; VLM questions on demand.

Subscribes /head_camera/{image, depth_image, camera_info}  only while /aibot/mode == "ai" (the
           bridge is lazy, so Gazebo renders the camera only then), TF, /aibot/mode
Publishes  /aibot/perception/objects   aibot_msgs/DetectedObjectArray (world frame)
           /aibot/perception/markers   visualization_msgs/MarkerArray (RViz)
Services   /aibot/perception/detect_now   std_srvs/Trigger   run one detection on the next frame
           /aibot/perception/clear        std_srvs/Trigger   forget all unheld objects (fresh scan)
           /aibot/perception/set_prompts  aibot_msgs/SetPrompts
           /aibot/perception/set_held     aibot_msgs/SetHeld (skill server, after a verified grasp)
           /aibot/vlm/describe, /aibot/vlm/verify  aibot_msgs/VlmQuery (Ollama VLM, on demand)

Detectors (parameter `detector`): "tabletop" (default: depth segmentation + shape/colour labels),
"yolo_world" (open vocabulary 2D boxes + depth fusion), "color_blob" (HSV baseline). Simulator
ground truth is never used here. Detection runs at detect_rate_hz (default 2) or on request.
The VLM is never in a control loop: it is asked only through the two services.
"""
from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from aibot_msgs.msg import DetectedObject, DetectedObjectArray
from aibot_msgs.srv import SetHeld, SetPrompts, VlmQuery
from aibot_teleop.utils.mode_gate import MODE_TOPIC, mode_qos

from .backends.base import BackendError
from .backends.base import HEAD_CAMERA, BackendError
from .backends.factory import build
from .perception.semantics import LABEL_SCHEMA, SemanticCache, apply_labels, contact_sheet, label_prompt
from .perception.detectors import ColorBlobDetector, Detector, YoloWorldDetector, nms
from .perception.fusion import (Intrinsics, backproject, backproject_pixels, estimate_object,
                                fit_table_plane, transform)
from .perception.memory import ObjectMemory
from .perception.tabletop import TabletopConfig, segment_tabletop

DEFAULT_PROMPTS = ["cube", "ball", "can", "box", "bin", "tray", "cylinder", "block", "mug"]
DESCRIBE_SCHEMA = {"type": "object", "properties": {
    "objects": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "color": {"type": "string"}, "position": {"type": "string"}},
        "required": ["name", "color", "position"]}},
    "relations": {"type": "array", "items": {"type": "string"}}}, "required": ["objects", "relations"]}
VERIFY_SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}, "reason": {"type": "string"}},
                 "required": ["answer", "reason"]}


def _quat_z(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        p = self.declare_parameter
        objects_file = Path(get_package_share_directory("aibot_gazebo")) / "config" / "objects.yaml"
        table = yaml.safe_load(Path(p("objects_file", str(objects_file)).value).read_text())["table"]
        self.table_z = float(table["top_z"])
        self.tcfg = TabletopConfig(tuple(table["x"]), tuple(table["y"]))
        self.world_frame: str = p("world_frame", "world").value
        self.rate: float = p("detect_rate_hz", 2.0).value
        self.stride: int = p("depth_stride", 2).value
        self.ai_only: bool = p("ai_only", True).value
        self.self_frames: list[str] = list(p("self_filter_frames", ["left_hand_link", "right_hand_link",
                                                                     "left_wrist_pitch_link", "right_wrist_pitch_link"]).value)
        self.forearm_radius: float = p("self_filter.forearm_radius", 0.07).value
        self.forearm_step: float = p("self_filter.forearm_step", 0.04).value
        self.memory = ObjectMemory(p("memory.match_distance", 0.06).value, p("memory.alpha", 0.5).value,
                                   p("memory.expiry", 5.0).value, p("memory.tentative_hits", 3).value,
                                   p("memory.tentative_expiry", 3.0).value)
        self.prompts = list(DEFAULT_PROMPTS)
        self.detector_name: str = p("detector", "tabletop").value
        self.detector: Detector | None = None
        if self.detector_name == "yolo_world":
            models_dir = Path(get_package_share_directory("aibot_ai")) / "models"
            weights = p("yolo_world.weights", str(models_dir / "yolov8s-worldv2.pt")).value
            self.detector = YoloWorldDetector(weights, p("yolo_world.conf", 0.05).value, p("yolo_world.imgsz", 640).value)
        elif self.detector_name == "color_blob":
            self.detector = ColorBlobDetector()
        elif self.detector_name != "tabletop":
            raise SystemExit(f"unknown detector '{self.detector_name}' (tabletop | yolo_world | color_blob)")
        self.vlm = build(p("vlm_backend", "hybrid").value,
                         cloud={"model": p("vlm.cloud_model", "qwen/qwen3-vl-30b-a3b-instruct").value,
                                "url": p("vlm.cloud_url", "https://openrouter.ai/api/v1").value,
                                "first_byte_timeout": p("vlm.first_byte_timeout", 8.0).value,
                                "total_timeout": p("vlm.total_timeout", 20.0).value,
                                "max_tokens": p("vlm.max_tokens", 500).value},
                         local={"model": p("vlm.model", "moondream").value, "url": p("vlm.url", "http://localhost:11434").value,
                                "keep_alive": p("vlm.keep_alive", "10m").value, "num_ctx": p("vlm.num_ctx", 4096).value,
                                "num_predict": p("vlm.num_predict", 384).value},
                         breaker={"failures": p("breaker.failures", 3).value, "cooldown": p("breaker.cooldown", 60.0).value,
                                  "retry_backoff": p("breaker.retry_backoff", 0.5).value},
                         log=lambda m: self.get_logger().info(m))
        self.vlm_size: int = p("vlm.image_size", 448).value
        self.labels = SemanticCache(p("semantic.move_tolerance", 0.03).value)
        self.label_enabled: bool = p("semantic.enabled", True).value
        self.label_min_hits: int = p("semantic.min_hits", 2).value
        self.label_max_objects: int = p("semantic.max_objects", 8).value
        self.label_retry: float = p("semantic.retry_s", 10.0).value
        self._labelling = False
        self._last_label_try = 0.0

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self._lock = threading.Lock()
        self._rgb: Image | None = None
        self._depth: Image | None = None
        self._K: Intrinsics | None = None
        self._subs: list = []
        self._detect_requests: list[threading.Event] = []
        self._rng = np.random.default_rng(0)
        self.mode: str | None = None
        self.last_ms = (0.0, 0.0)
        self.cb_io = ReentrantCallbackGroup()
        self.cb_work = MutuallyExclusiveCallbackGroup()

        self.pub = self.create_publisher(DetectedObjectArray, "/aibot/perception/objects", 5)
        self.marker_pub = self.create_publisher(MarkerArray, "/aibot/perception/markers", 5)
        self.create_subscription(String, MODE_TOPIC, self._on_mode, mode_qos(), callback_group=self.cb_io)
        self.create_service(Trigger, "/aibot/perception/detect_now", self._srv_detect_now, callback_group=self.cb_io)
        self.create_service(Trigger, "/aibot/perception/clear", self._srv_clear, callback_group=self.cb_io)
        self.create_service(SetPrompts, "/aibot/perception/set_prompts", self._srv_prompts, callback_group=self.cb_io)
        self.create_service(SetHeld, "/aibot/perception/set_held", self._srv_held, callback_group=self.cb_io)
        self.create_service(VlmQuery, "/aibot/vlm/describe", self._srv_describe, callback_group=self.cb_io)
        self.create_service(VlmQuery, "/aibot/vlm/verify", self._srv_verify, callback_group=self.cb_io)
        self.create_timer(1.0 / self.rate, self._on_timer, callback_group=self.cb_work)
        if not self.ai_only:
            self._subscribe(True)
        self.get_logger().info(f"perception: detector '{self.detector_name}', {self.rate:.1f} Hz in AI mode")

    # ------------------------------------------------------------------ camera plumbing
    def _subscribe(self, on: bool) -> None:
        if on and not self._subs:
            self._subs = [
                self.create_subscription(Image, "/head_camera/image", lambda m: setattr(self, "_rgb", m), 2, callback_group=self.cb_io),
                self.create_subscription(Image, "/head_camera/depth_image", lambda m: setattr(self, "_depth", m), 2, callback_group=self.cb_io),
                self.create_subscription(CameraInfo, "/head_camera/camera_info", self._on_info, 2, callback_group=self.cb_io)]
        elif not on and self._subs:
            for s in self._subs:
                self.destroy_subscription(s)
            self._subs, self._rgb, self._depth = [], None, None

    def _on_info(self, m: CameraInfo) -> None:
        self._K = Intrinsics(m.k[0], m.k[4], m.k[2], m.k[5])

    def _on_mode(self, m: String) -> None:
        self.mode = m.data
        if self.ai_only:
            self._subscribe(m.data == "ai")

    def _frame(self) -> tuple[np.ndarray, np.ndarray, Time] | None:
        rgb, depth = self._rgb, self._depth
        if rgb is None or depth is None or self._K is None:
            return None
        if (rgb.header.stamp.sec, rgb.header.stamp.nanosec) != (depth.header.stamp.sec, depth.header.stamp.nanosec):
            return None                                   # rgbd frames share a stamp; wait for a pair
        img = np.frombuffer(bytes(rgb.data), np.uint8).reshape(rgb.height, rgb.width, 3)
        d = np.frombuffer(bytes(depth.data), np.float32).reshape(depth.height, depth.width)
        return img, d, Time.from_msg(rgb.header.stamp)

    def _T(self, target: str, source: str, stamp: Time) -> np.ndarray | None:
        try:
            tf = self.tf.lookup_transform(target, source, stamp, timeout=rclpy.duration.Duration(seconds=0.2))
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _arm_over_table(self) -> bool:
        for side in ("left", "right"):
            T = self._T(self.world_frame, f"{side}_hand_link", Time())
            if T is None:
                continue
            x, y, _ = T[:3, 3]
            if (self.tcfg.x_range[0] - 0.05 < x < self.tcfg.x_range[1] + 0.05
                    and self.tcfg.y_range[0] - 0.05 < y < self.tcfg.y_range[1] + 0.05):
                return True
        return False

    def _forearm_samples(self, stamp: Time) -> list[tuple[np.ndarray, float]]:
        """(centre, radius) every few cm along shoulder -> elbow -> wrist -> hand of both arms: the
        robot's own grey arm reaching over the table edge would otherwise be segmented as objects
        (measured: the upper arm at the carry pose blocked every place as "silver_box")."""
        out = []
        for side in ("left", "right"):
            pts = [T[:3, 3] for f in (f"{side}_shoulder_roll_link", f"{side}_elbow_pitch_link", f"{side}_wrist_pitch_link",
                                      f"{side}_hand_link")
                   if (T := self._T(self.world_frame, f, stamp)) is not None]
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(2, int(np.linalg.norm(b - a) / self.forearm_step) + 1)
                out += [(a + u * (b - a), self.forearm_radius) for u in np.linspace(0.0, 1.0, n)]
        return out

    # ------------------------------------------------------------------ detection
    def _on_timer(self) -> None:
        requested = bool(self._detect_requests)
        if not requested and (self.ai_only and self.mode != "ai"):
            return
        if requested and not self._subs:
            self._subscribe(True)
        frame = self._frame()
        if frame is None:
            return
        rgb, depth, stamp = frame
        if not requested and self._arm_over_table():
            return                                # occluded, moving view: wait for the arm to leave
        T = self._T(self.world_frame, self._rgb.header.frame_id, stamp)
        if T is None:
            self.get_logger().warn("no TF world <- camera at image time", throttle_duration_sec=5.0)
            return
        self.detect(rgb, depth, T, stamp)
        with self._lock:
            reqs, self._detect_requests = self._detect_requests, []
        for ev in reqs:
            ev.set()

    def detect(self, rgb: np.ndarray, depth: np.ndarray, T: np.ndarray, stamp: Time) -> None:
        K = self._K
        t0 = time.perf_counter()
        pts_cam, uv = backproject_pixels(depth, K, stride=self.stride)
        pts = transform(pts_cam, T)
        plane, _ = fit_table_plane(pts[:: max(1, len(pts) // 4000)], self.table_z, self._rng)
        # the arm where it was when the image was taken (not "latest": a moving arm would leave a
        # trail of arm points in the cloud that become phantom objects)
        exclude = [T_h[:3, 3] for f in self.self_frames if (T_h := self._T(self.world_frame, f, stamp)) is not None]
        exclude += self._forearm_samples(stamp)
        observations = []
        if self.detector is None:                              # tabletop segmentation
            t1 = time.perf_counter()
            for seg in segment_tabletop(pts, uv, rgb, plane, self.tcfg, exclude):
                observations.append((seg.label, seg.color, seg.estimate, seg.score, seg.box))
            t_det, t_fus = (time.perf_counter() - t1) * 1000, (t1 - t0) * 1000
        else:                                                  # 2D detector + per-box depth fusion
            t1 = time.perf_counter()
            dets = nms(self.detector.detect(rgb, self.prompts))
            t2 = time.perf_counter()
            for det in dets:
                pc, frac = backproject(depth, K, det.box, stride=self.stride)
                est = estimate_object(transform(pc, T), plane, frac)
                if est is not None and not any(np.linalg.norm(est.centroid - (c[0] if isinstance(c, tuple) else c)) < 0.08
                                               for c in exclude):
                    observations.append((det.label, det.color, est, det.score, tuple(int(v) for v in det.box)))
            t_det, t_fus = (t2 - t1) * 1000, ((t1 - t0) + (time.perf_counter() - t2)) * 1000
        now = stamp.nanoseconds * 1e-9
        self.memory.update(observations, now)
        self.last_ms = (t_det, t_fus)
        self._maybe_label(rgb)
        self._publish(stamp)

    def _publish(self, stamp: Time) -> None:
        arr = DetectedObjectArray(detector=self.detector_name, detect_ms=float(self.last_ms[0]), fusion_ms=float(self.last_ms[1]))
        arr.header.stamp, arr.header.frame_id = stamp.to_msg(), self.world_frame
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, o in enumerate(self.memory.objects.values()):
            m = DetectedObject(id=o.id, label=o.label, color=o.color, confidence=float(o.confidence), held_by=o.held_by,
                               inside=o.inside, box=[int(v) for v in o.box], semantic_name=o.semantic_name,
                               description=o.description)
            m.pose.position = Point(x=float(o.position[0]), y=float(o.position[1]), z=float(o.position[2]))
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = _quat_z(o.yaw)
            m.size.x, m.size.y, m.size.z = (float(v) for v in o.size)
            m.grasp_pose.position = Point(x=float(o.grasp[0]), y=float(o.grasp[1]), z=float(o.grasp[2]))
            gq = _quat_z(o.grasp_yaw)
            m.grasp_pose.orientation.x, m.grasp_pose.orientation.y, m.grasp_pose.orientation.z, m.grasp_pose.orientation.w = gq
            m.last_seen = Time(seconds=o.last_seen).to_msg()
            arr.objects.append(m)
            box = Marker(type=Marker.CUBE, ns="objects", id=2 * i, pose=m.pose)
            box.header = arr.header
            box.scale.x, box.scale.y, box.scale.z = max(m.size.x, 0.01), max(m.size.y, 0.01), max(m.size.z, 0.01)
            box.color = ColorRGBA(r=0.2, g=0.8, b=1.0, a=0.35)
            txt = Marker(type=Marker.TEXT_VIEW_FACING, ns="labels", id=2 * i + 1, text=o.id)
            txt.header, txt.pose.position = arr.header, Point(x=m.pose.position.x, y=m.pose.position.y, z=m.grasp_pose.position.z + 0.04)
            txt.scale.z, txt.color = 0.025, ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            markers.markers += [box, txt]
        self.pub.publish(arr)
        self.marker_pub.publish(markers)

    # ------------------------------------------------------------------ services
    def _srv_detect_now(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        ev = threading.Event()
        with self._lock:
            self._detect_requests.append(ev)
        ok = ev.wait(5.0)
        res.success = ok
        res.message = (f"{len(self.memory.objects)} objects: {', '.join(self.memory.objects)}" if ok
                       else "no camera frame within 5 s (is the head camera bridged and the sim running?)")
        return res

    def _srv_clear(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        for k in [k for k, o in self.memory.objects.items() if not o.held_by]:
            del self.memory.objects[k]
        res.success, res.message = True, "forgot all unheld objects"
        return res

    def _srv_prompts(self, req: SetPrompts.Request, res: SetPrompts.Response) -> SetPrompts.Response:
        self.prompts = list(req.prompts) or list(DEFAULT_PROMPTS)
        res.success, res.message = True, f"prompts: {self.prompts}"
        return res

    def _srv_held(self, req: SetHeld.Request, res: SetHeld.Response) -> SetHeld.Response:
        pos = np.array([req.position.x, req.position.y, req.position.z]) if req.has_position else None
        res.success = self.memory.set_held(req.object_id, req.side, pos, req.container_id,
                                           self.get_clock().now().nanoseconds * 1e-9)
        res.message = "ok" if res.success else f"unknown object {req.object_id}"
        return res

    # ------------------------------------------------------------------ semantic labels
    def _maybe_label(self, rgb: np.ndarray) -> None:
        """Called after a detection, never waits: objects seen a few times without a valid label go
        to one background VLM request (a numbered contact sheet); detection and control go on."""
        for o in self.memory.objects.values():
            if self.labels.valid(o.id, o.position, o.size):
                o.semantic_name, o.description = self.labels.get(o.id)
            else:
                o.semantic_name, o.description = "", ""
        if not self.label_enabled or self._labelling or self._arm_over_table():
            return
        todo = [o for o in self.memory.objects.values() if not o.held_by and not o.inside and o.hits >= self.label_min_hits
                and not self.labels.valid(o.id, o.position, o.size) and o.box[2] > o.box[0]]
        if not todo or time.monotonic() - self._last_label_try < self.label_retry:
            return
        self._labelling, self._last_label_try = True, time.monotonic()
        snapshot = [(o.id, o.position.copy(), o.size.copy(), o.box) for o in todo[:self.label_max_objects]]
        threading.Thread(target=self._label, args=(rgb.copy(), snapshot), daemon=True).start()

    def _label(self, rgb: np.ndarray, snapshot: list) -> None:
        t0 = time.monotonic()
        try:
            ids = [e[0] for e in snapshot]
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(contact_sheet(rgb, [e[3] for e in snapshot]), cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return
            answer = self.vlm.ask(buf.tobytes(), label_prompt(len(ids)), LABEL_SCHEMA, source=HEAD_CAMERA)
            got = apply_labels(answer, ids)
            for oid, pos, size, _box in snapshot:
                if oid in got:
                    self.labels.store(oid, pos, size, *got[oid])
            self.get_logger().info(f"semantic labels: {len(got)}/{len(ids)} objects in {time.monotonic() - t0:.1f} s "
                                   f"[{self.vlm.describe_last()}]: {[v[0] for v in got.values()]}")
        except BackendError as exc:
            self.get_logger().warn(f"semantic labelling failed: {exc}", throttle_duration_sec=30.0)
        finally:
            self._labelling = False

    def _jpeg(self) -> bytes | None:
        frame = self._frame()
        if frame is None:
            return None
        img = frame[0]
        s = self.vlm_size / max(img.shape[:2])
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def _vlm(self, prompt: str, schema: dict, res: VlmQuery.Response) -> VlmQuery.Response:
        if not self._subs:
            self._subscribe(True)
            time.sleep(0.5)
        jpg = self._jpeg()
        if jpg is None:
            res.success, res.message = False, "no head-camera image"
            return res
        t0 = time.monotonic()
        try:
            answer = self.vlm.ask(jpg, prompt, schema, source=HEAD_CAMERA)
            res.success, res.answer_json, res.message = True, json.dumps(answer), "ok"
        except BackendError as exc:
            res.success, res.message = False, str(exc)
        res.latency_s = float(time.monotonic() - t0)
        return res

    def _srv_describe(self, _req: VlmQuery.Request, res: VlmQuery.Response) -> VlmQuery.Response:
        return self._vlm("This is a robot's head camera looking down at a small table. List every object on the "
                         "table (name, colour, position on the table such as front-left or back-right) and the "
                         "spatial relations between them. Answer in JSON.", DESCRIBE_SCHEMA, res)

    def _srv_verify(self, req: VlmQuery.Request, res: VlmQuery.Response) -> VlmQuery.Response:
        return self._vlm(f"This is a robot's head camera looking down at a table. Is the following true? "
                         f"'{req.question}'. Answer in JSON with a boolean and a one-sentence reason.", VERIFY_SCHEMA, res)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(KeyboardInterrupt):
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
