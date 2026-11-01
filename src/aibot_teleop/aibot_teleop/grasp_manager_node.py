"""grasp_manager_node: weld an object to a closed robot hand (Gazebo DetachableJoint), release on open.

Physics-only finger grasps are unreliable in Gazebo (light, force-limited fingers on convex
hulls slip). Instead, while a hand is closed near an object, the object is welded to the palm;
the fingers still close visually.

Subscribes  /teleop/grasp_state/{left,right}       aibot_msgs/GraspState from retarget_node (mimic)
            /aibot/grasp_cmd/{left,right}          aibot_msgs/GraspState from skill_server_node (AI)
            /aibot/mode                            std_msgs/String, selects which of the two is obeyed
            /aibot/object_poses                    tf2_msgs/TFMessage, named object poses from gz
            /aibot/grasp/<side>/<obj>/state        std_msgs/String "attached" | "detached"
            TF world -> {side}_hand_link           from robot_state_publisher
Publishes   /aibot/grasp/<side>/<obj>/attach       std_msgs/Empty (bridged to the gz plugin)
            /aibot/grasp/<side>/<obj>/detach       std_msgs/Empty
            /teleop/grasp_status                   std_msgs/String, e.g. "L: cube_small | R: -"
            /teleop/grasp_markers                  visualization_msgs/MarkerArray
            /aibot/grasp/status                    aibot_msgs/GraspStatus: held objects as REPORTED by
                                                   the plugin ("attached"), for skill verification
Services    /aibot/grasp/release_all               std_srvs/Trigger: detach everything (mode switch, STOP)

Command sources: each mode has its own input topic, so two nodes never publish on the same one.
The mode picks the source; after a switch the new source is ignored until it sends its first
message (until then the hands keep whatever they hold), and outside mimic/ai nothing changes.

Rules, checked at `rate` Hz:
  * Startup: the gz DetachableJoint plugin attaches EVERY pair automatically when the robot
    spawns (gz-sim 8 behaviour). All pairs are detached until each reports "detached" or
    `startup_timeout` passes. No grasping happens before that.
  * Attach: the hand is closed and an unattached object's centre is within `grasp_radius` of
    the hand's grasp point (between palm and fingers). The nearest such object is attached.
  * Release: the hand opens. Losing hand tracking keeps the current state.
  * An "attached" report that was not requested (e.g. robot respawned) is detached immediately.
"""
from __future__ import annotations

import contextlib
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import ColorRGBA, Empty, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from aibot_msgs.msg import GraspState, GraspStatus

from .utils.mode_gate import ModeGate

SIDES = ("left", "right")
SOURCE_FOR_MODE = {None: "teleop", "mimic": "teleop", "ai": "ai"}   # other modes: no source (hold)
COLORS = {"open": (0.2, 0.85, 0.3), "closed": (1.0, 0.6, 0.1), "holding": (0.2, 0.5, 1.0)}


def _quat_to_rot(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class GraspManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("grasp_manager_node")
        p = self.declare_parameter
        self.objects: list[str] = list(p("objects", ["cube_small", "cylinder_can", "box_small", "cube_small_2"]).value)
        self.radius: float = p("grasp_radius", 0.08).value
        self.world_frame: str = p("world_frame", "world").value
        self.offset = {"left": np.array(p("grasp_offset.left", [0.0, -0.035, -0.10]).value),
                       "right": np.array(p("grasp_offset.right", [0.0, 0.035, -0.10]).value)}
        self.startup_timeout: float = p("startup_timeout", 15.0).value
        prefix: str = p("topic_prefix", "/aibot/grasp").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.grasp_in: dict[str, dict[str, GraspState | None]] = {src: {s: None for s in SIDES} for src in ("teleop", "ai")}
        self._fresh_after = self.get_clock().now()      # source messages older than the last switch are ignored
        self._arrival: dict[tuple[str, str], Time] = {}
        self.object_pos: dict[str, np.ndarray] = {}
        self.pair_state: dict[tuple[str, str], str] = {(s, o): "unknown" for s in SIDES for o in self.objects}
        self.held: dict[str, str | None] = {s: None for s in SIDES}
        self.attach_pub: dict[tuple[str, str], object] = {}
        self.detach_pub: dict[tuple[str, str], object] = {}
        for s in SIDES:
            self.create_subscription(GraspState, f"/teleop/grasp_state/{s}", lambda m: self._on_grasp("teleop", m), 10)
            self.create_subscription(GraspState, f"/aibot/grasp_cmd/{s}", lambda m: self._on_grasp("ai", m), 10)
            for o in self.objects:
                base = f"{prefix}/{s}/{o}"
                self.attach_pub[(s, o)] = self.create_publisher(Empty, f"{base}/attach", 10)
                self.detach_pub[(s, o)] = self.create_publisher(Empty, f"{base}/detach", 10)
                self.create_subscription(String, f"{base}/state",
                                         lambda m, k=(s, o): self._on_pair_state(k, m.data), 10)
        self.create_subscription(TFMessage, "/aibot/object_poses", self._on_object_poses, 10)
        self.status_pub = self.create_publisher(String, "/teleop/grasp_status", 5)
        self.marker_pub = self.create_publisher(MarkerArray, "/teleop/grasp_markers", 5)
        self.grasp_status_pub = self.create_publisher(GraspStatus, "/aibot/grasp/status", 5)
        self.create_service(Trigger, "/aibot/grasp/release_all", self._srv_release_all)
        self.gate = ModeGate(self, "mimic", required=False, on_change=self._on_mode_change)

        self._startup_done = False
        self._t0 = self.get_clock().now()
        self._last_detach_burst = None
        self._armed = {s: True for s in SIDES}     # False after release_all until the hand opens
        self.create_timer(1.0 / p("rate", 30.0).value, self._on_timer)
        self.get_logger().info(f"managing {len(self.objects)} objects x 2 hands, grasp radius {self.radius:.2f} m")

    # ------------------------------------------------------------------ inputs
    def _on_grasp(self, source: str, msg: GraspState) -> None:
        if msg.side in SIDES:
            self.grasp_in[source][msg.side] = msg
            self._arrival[(source, msg.side)] = self.get_clock().now()

    def _on_mode_change(self, _old: str | None, _new: str) -> None:
        self._fresh_after = self.get_clock().now()

    def _active_command(self, side: str) -> GraspState | None:
        """Command of the source selected by the mode, if it spoke since the last mode switch."""
        source = SOURCE_FOR_MODE.get(self.gate.mode)
        if source is None:
            return None
        arrived = self._arrival.get((source, side))
        if arrived is None or arrived < self._fresh_after:
            return None
        return self.grasp_in[source][side]

    def confirmed(self, side: str) -> str:
        """Object the plugin reports attached to this hand ("" if none)."""
        return next((o for (s, o), st in self.pair_state.items() if s == side and st == "attached"), "")

    def _srv_release_all(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        released = []
        for (side, obj), st in self.pair_state.items():
            if st == "attached" or self.held[side] == obj:
                self.detach_pub[(side, obj)].publish(Empty())
                released.append(f"{side}:{obj}")
        self.held = {s: None for s in SIDES}
        # A hand that is still commanded closed must not grab again right away (the scene reset
        # that calls this teleports objects next to it): re-arm only after it has been opened.
        self._armed = {s: False for s in SIDES}
        res.success, res.message = True, ("released " + ", ".join(released)) if released else "nothing held"
        self.get_logger().info(res.message)
        return res

    def _on_object_poses(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            name = next((o for o in self.objects if tr.child_frame_id.split("::")[-1] == o), None)
            if name is not None:
                t = tr.transform.translation
                self.object_pos[name] = np.array([t.x, t.y, t.z])

    def _on_pair_state(self, key: tuple[str, str], state: str) -> None:
        self.pair_state[key] = state
        side, obj = key
        if state == "attached" and self.held[side] != obj:
            self.get_logger().info(f"unrequested attach {side}/{obj} (plugin startup behaviour): detaching")
            self.detach_pub[key].publish(Empty())

    # ------------------------------------------------------------------ logic
    def _grasp_point(self, side: str) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, f"{side}_hand_link", Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]) + _quat_to_rot(q) @ self.offset[side]

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        if not self._startup_done:
            self._startup_detach(now)
        points = {s: self._grasp_point(s) for s in SIDES}
        if self._startup_done:
            for side in SIDES:
                self._update_side(side, points[side])
        self._publish_feedback(points)

    def _startup_detach(self, now: Time) -> None:
        pending = [k for k, v in self.pair_state.items() if v != "detached"]
        timed_out = (now - self._t0) > Duration(seconds=self.startup_timeout)
        if not pending or timed_out:
            self._startup_done = True
            msg = "all pairs detached" if not pending else f"startup timeout, {len(pending)} pairs unconfirmed"
            self.get_logger().info(f"grasping enabled ({msg})")
            return
        if self._last_detach_burst is None or (now - self._last_detach_burst) > Duration(seconds=0.5):
            for k in pending:
                self.detach_pub[k].publish(Empty())
            self._last_detach_burst = now

    def _update_side(self, side: str, point: np.ndarray | None) -> None:
        g = self._active_command(side)
        if g is None:
            return
        held = self.held[side]
        if not g.closed:
            self._armed[side] = True
        if held is not None and not g.closed:
            self.detach_pub[(side, held)].publish(Empty())
            self.held[side] = None
            self.get_logger().info(f"{side} hand opened: released {held}")
            return
        if held is None and g.closed and point is not None and self._armed[side]:
            taken = {o for o in self.held.values() if o}
            best, best_d = None, self.radius
            for obj, pos in self.object_pos.items():
                d = float(np.linalg.norm(pos - point))
                if obj not in taken and d <= best_d:
                    best, best_d = obj, d
            if best is not None:
                self.held[side] = best
                self.attach_pub[(side, best)].publish(Empty())
                self.get_logger().info(f"{side} hand closed {best_d * 100:.1f} cm from {best}: attached")

    # ------------------------------------------------------------------ feedback
    def _publish_feedback(self, points: dict[str, np.ndarray | None]) -> None:
        self.status_pub.publish(String(data=" | ".join(f"{s[0].upper()}: {self.held[s] or '-'}" for s in SIDES)
                                       + ("" if self._startup_done else " (starting: detaching objects)")))
        st = GraspStatus(left_object=self.confirmed("left"), right_object=self.confirmed("right"), ready=self._startup_done)
        st.header.stamp = self.get_clock().now().to_msg()
        self.grasp_status_pub.publish(st)
        markers = MarkerArray()
        stamp = Time().to_msg()           # 0 = "latest": valid whichever clock RViz runs on
        for i, side in enumerate(SIDES):
            pt = points[side]
            g = self._active_command(side)
            if pt is None:
                continue
            state = "holding" if self.held[side] else ("closed" if g is not None and g.closed else "open")
            r, gr, b = COLORS[state]
            sphere = Marker(type=Marker.SPHERE, action=Marker.ADD, ns="grasp_zone", id=i)
            sphere.header.frame_id, sphere.header.stamp = self.world_frame, stamp
            sphere.pose.position = Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 2 * self.radius
            sphere.color = ColorRGBA(r=r, g=gr, b=b, a=0.25)
            text = Marker(type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, ns="grasp_label", id=i)
            text.header = sphere.header
            text.pose.position = Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]) + self.radius + 0.04)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.035
            text.color = ColorRGBA(r=r, g=gr, b=b, a=1.0)
            closure = f"{g.closure:.2f}" if g is not None else "?"
            text.text = f"{side}: {self.held[side] or state} ({closure})"
            markers.markers += [sphere, text]
        self.marker_pub.publish(markers)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GraspManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(KeyboardInterrupt):   # second Ctrl-C during cleanup
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
