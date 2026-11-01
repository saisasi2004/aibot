"""retarget_node: operator landmarks -> STAR1 joint commands for the six position controllers.

Subscribes  /teleop/body_landmarks, /teleop/hand_landmarks/{left,right}   (aibot_msgs)
            /joint_states   (until the first command, to start from the robot's actual pose)
Publishes   /{waist,head,left_arm,right_arm,left_hand,right_hand}_controller/commands
            /teleop/grasp_state/{left,right}   aibot_msgs/GraspState of the ROBOT hands
            /teleop/joint_targets              sensor_msgs/JointState (final commands, for plots)
            /teleop/status                     std_msgs/String (shown by the camera GUI)
Services    /teleop/calibrate (std_srvs/Trigger)   restart the T-pose calibration
            /teleop/enable    (std_srvs/SetBool)   pause / resume streaming to the robot

Mode hand-over (Physical AI mode): commands are only published while /aibot/mode is "mimic"
(ModeGate) and /teleop/enable is true. Whenever streaming resumes, after a pause or after the AI
mode moved the arms, the filter, rate limiter, last command and direct-mapping history are reset
to the robot's ACTUAL /joint_states, and for `resume.duration` seconds each command step is
capped at resume.max_velocity * (1/30 s).

Bumpless transfer: per joint, the restart point is the controllers' current setpoint, i.e. the
last command sent by EITHER mode (falling back to the actual position if none was ever sent).
Free joints have settled onto that setpoint (mode_manager waits for it), so it equals the actual
pose. A joint blocked by contact (fingers around an object, an arm pressing on the table) sits
away from its setpoint; restarting at its actual position would make the command jump, while
gliding from the setpoint keeps both the command and the robot continuous. Joints further than
resume.tolerance from their setpoint are reported in the log. The robot therefore glides from wherever the AI left it
to the operator's pose instead of jumping.

Latency: each camera frame is processed the moment its last message arrives. There is no
fixed-rate timer resampling the stream (that would add up to one period of delay). Commands
therefore go out at the camera rate (~30 Hz) after ~1 ms of processing. Per frame:
  landmarks -> body axes (+ mirror) -> chest frame -> waist / head / arm / hand targets
  -> per-joint corrections -> One-Euro filter (frozen where tracking is lost) -> URDF limits
  -> velocity limit -> dead-band -> controllers
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger

from aibot_msgs.msg import BodyLandmarks, GraspState, HandLandmarks

from .calibration import Calibration, TPoseCalibrator
from .retarget.arm_angles import arm_vectors, hand_frame, hand_frame_from_pose, solve_arm
from .retarget.arm_ik import ArmChain, ArmIK
from .retarget.body_frame import HIP, SIDES, chest_frame, make_sample
from .retarget.hand_curl import GraspDetector, HandMapping, hand_shape
from .retarget.joint_map import JointMap
from .retarget.torso_head import head_direction, head_targets, waist_targets
from .utils.filters import OneEuroFilter
from .utils.limits import RateLimiter, deadband
from .utils.mode_gate import ModeGate

NOMINAL_DT = 1.0 / 30.0   # camera frame period: bounds the per-step jump while resuming


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _xyz(points) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in points], dtype=float)


class RetargetNode(Node):
    def __init__(self) -> None:
        super().__init__("retarget_node")
        p = self.declare_parameter
        teleop_share = Path(get_package_share_directory("aibot_teleop"))
        desc_share = Path(get_package_share_directory("aibot_description"))

        self.mirror: bool = p("mirror", False).value
        self.use_ik: bool = p("use_ik", False).value
        self.jm = JointMap(p("joint_mapping_file", str(teleop_share / "config" / "joint_mapping.yaml")).value,
                           p("joint_limits_file", str(desc_share / "config" / "joint_limits.yaml")).value,
                           p("limit_margin", 0.02).value)
        self.vis_threshold: float = p("visibility_threshold", 0.5).value
        self.hand_score_threshold: float = p("hand_score_threshold", 0.5).value
        self.lost_timeout: float = p("lost_timeout", 0.5).value
        self.enabled_groups = {g: p(f"enable.{g}", True).value for g in ("waist", "head", "arms", "hands")}
        self.gain_waist: float = p("gain.waist", 1.0).value
        self.gain_head: float = p("gain.head", 1.0).value
        self.hip_on, self.hip_off = p("hips.visible_on", 0.7).value, p("hips.visible_off", 0.4).value
        self.resume_vmax: float = p("resume.max_velocity", 1.0).value
        self.resume_duration: float = p("resume.duration", 2.0).value       # minimum glide time
        self.resume_max_duration: float = p("resume.max_duration", 6.0).value
        self.resume_converged: float = p("resume.converged", 0.05).value     # rad: glide ends below this
        self.resume_tolerance: float = p("resume.tolerance", 0.2).value      # rad: last command vs actual
        self.recalibrate_after: float = p("recalibrate_hint_after", 120.0).value
        mode_required: bool = p("mode_required", False).value

        def per_group(name: str, defaults: dict[str, float]) -> np.ndarray:
            return self.jm.per_group({k: p(f"{name}.{k}", v).value for k, v in defaults.items()})

        self.filter = OneEuroFilter(per_group("filter.min_cutoff", {"waist": 0.8, "head": 1.0, "arm": 1.2, "hand": 1.5}),
                                    per_group("filter.beta", {"waist": 0.3, "head": 0.4, "arm": 0.6, "hand": 0.4}),
                                    p("filter.d_cutoff", 1.0).value)
        vmax = per_group("max_velocity", {"waist": 1.0, "head": 2.5, "arm": 3.0, "hand": 8.0})
        self.rate = RateLimiter(np.minimum(vmax, self.jm.limits.velocity))
        self.band = per_group("deadband", {"waist": 0.01, "head": 0.01, "arm": 0.008, "hand": 0.02})
        self.hand_map = HandMapping()
        close_t, open_t = p("grasp.close_threshold", 0.6).value, p("grasp.open_threshold", 0.4).value
        confirm = p("grasp.confirm_frames", 3).value
        self.grasp = {s: GraspDetector(close_t, open_t, confirm) for s in SIDES}
        self.closure = {s: 0.0 for s in SIDES}

        self.calib = Calibration()
        self.calibrator = TPoseCalibrator(p("calibration.duration", 3.0).value, p("calibration.timeout", 30.0).value)
        self._calibrate_pending: bool = p("calibrate_on_startup", True).value

        self.ik: dict[str, ArmIK] = {}
        if self.use_ik:
            urdf = p("robot_description", "").value
            if not urdf:
                raise SystemExit("use_ik:=true needs the robot_description parameter (the URDF string)")
            for side in SIDES:
                chain = ArmChain(urdf, "base_link", f"{side}_hand_link")
                idx = self.jm.groups[f"{side}_arm"]
                expected = self.jm.controllers[f"{side}_arm_controller"]
                if chain.names != expected:
                    raise SystemExit(f"IK chain joints {chain.names} do not match controller joints {expected}")
                upper = self.jm.limits.upper[idx].copy()
                upper[3] = min(upper[3], 0.0)   # no elbow hyperextension: humans cannot, IK must not
                self.ik[side] = ArmIK(chain, self.jm.limits.lower[idx], upper, iterations=12)

        n = len(self.jm.names)
        self.q_target = np.zeros(n)              # raw targets (held where tracking is lost)
        self.q_out: np.ndarray | None = None     # last command sent
        self._direct_prev = {s: np.zeros(7) for s in SIDES}   # direct-mapping history per arm
        self.joint_state: dict[str, float] = {}
        self.enabled = True
        self._hips_visible = False
        self._last_frame_stamp: float | None = None
        self._last_rx = time.monotonic()
        self._processed_key: tuple[int, int] | None = None
        self._body: BodyLandmarks | None = None
        self._hands: dict[str, HandLandmarks | None] = {"left": None, "right": None}
        self._status = "waiting for landmarks"
        self._latency_ms: float | None = None
        self._resync_pending = False
        self._resume_until = 0.0
        self._resume_deadline = 0.0
        self._paused_since: float | None = None
        self._hint_until = 0.0
        self.gate = ModeGate(self, "mimic", mode_required, self._on_mode_change)

        self.cmd_pubs = {c: self.create_publisher(Float64MultiArray, f"/{c}/commands", 5) for c in self.jm.controllers}
        self.grasp_pubs = {s: self.create_publisher(GraspState, f"/teleop/grasp_state/{s}", 5) for s in SIDES}
        self.targets_pub = self.create_publisher(JointState, "/teleop/joint_targets", 5)
        self.status_pub = self.create_publisher(String, "/teleop/status", 5)
        self.create_subscription(BodyLandmarks, "/teleop/body_landmarks", self._on_body, 5)
        for s in SIDES:
            self.create_subscription(HandLandmarks, f"/teleop/hand_landmarks/{s}", self._on_hand, 5)
        # kept for the node's lifetime: every resume re-reads the robot's actual pose ...
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 5)
        # ... and the last command of ANY source (this node or the Physical AI skill server)
        self.last_cmd: dict[str, float] = {}
        for c, joints in self.jm.controllers.items():
            self.create_subscription(Float64MultiArray, f"/{c}/commands",
                                     lambda m, js=joints: self.last_cmd.update(zip(js, m.data)), 5)
        self.create_service(Trigger, "/teleop/calibrate", self._srv_calibrate)
        self.create_service(SetBool, "/teleop/enable", self._srv_enable)
        self.create_timer(0.2, self._on_timer)
        self.get_logger().info(
            f"retargeting {n} joints | mirror={self.mirror} | arms: {'IK' if self.use_ik else 'direct angles'} | "
            f"calibrate on startup: {self._calibrate_pending}")

    # ------------------------------------------------------------------ inputs
    def _on_joint_states(self, msg: JointState) -> None:
        self.joint_state.update(zip(msg.name, msg.position))

    def _on_body(self, msg: BodyLandmarks) -> None:
        prev = self._body
        if prev is not None and self._key(prev) != self._processed_key:   # its hands never came: run it anyway
            self._process(prev, self._matching_hands(prev))
        self._body = msg
        self._try_process()

    def _on_hand(self, msg: HandLandmarks) -> None:
        self._hands[msg.side] = msg
        self._try_process()

    @staticmethod
    def _key(msg) -> tuple[int, int]:
        return msg.header.stamp.sec, msg.header.stamp.nanosec

    def _matching_hands(self, body: BodyLandmarks) -> dict[str, HandLandmarks]:
        k = self._key(body)
        return {s: m for s, m in self._hands.items() if m is not None and self._key(m) == k}

    def _try_process(self) -> None:
        body = self._body
        if body is None or self._key(body) == self._processed_key:
            return
        hands = self._matching_hands(body)
        if len(hands) == len(SIDES):     # body + both hands of the same frame are in
            self._process(body, hands)

    # ------------------------------------------------------------------ pipeline
    def _process(self, body: BodyLandmarks, hand_msgs: dict[str, HandLandmarks]) -> None:
        self._processed_key = self._key(body)
        self._last_rx = time.monotonic()
        now = _stamp(body)
        dt = 1.0 / 30.0 if self._last_frame_stamp is None else float(np.clip(now - self._last_frame_stamp, 1 / 120, 0.2))
        self._last_frame_stamp = now

        if not body.detected:
            return self._hold("LOST: nobody in view, holding pose")
        vis = np.array(body.visibility, dtype=float)
        hands_mp = {s: (_xyz(m.world_points) if m.detected and m.score >= self.hand_score_threshold else None)
                    for s, m in hand_msgs.items()}
        sample = make_sample(_xyz(body.world_points), vis, {s: hands_mp.get(s) for s in SIDES}, now, self.mirror)

        hip_vis = min(sample.visibility[HIP["left"]], sample.visibility[HIP["right"]])
        if hip_vis > self.hip_on:
            self._hips_visible = True
        elif hip_vis < self.hip_off:
            self._hips_visible = False
        r_chest = chest_frame(sample, self._hips_visible)
        if r_chest is None:
            return self._hold("LOST: shoulders not visible, holding pose")

        if self._calibrate_pending:
            self.calibrator.start(now)
            self._calibrate_pending = False
        if self.calibrator.active:
            text, result = self.calibrator.update(sample, r_chest, now)
            if result is None:
                return self._hold(f"CALIBRATING: {text}")
            self.calib = result
            self.get_logger().info(f"calibration: {text} - {result.summary()}")

        active = self.enabled and self.gate.active
        if self.q_out is None or (self._resync_pending and active):
            if not all(n in self.joint_state for n in self.jm.names):
                return self._hold("waiting for /joint_states from the robot (is Gazebo running?)")
            self._initialise_from_robot()
            if self._resync_pending:
                self._resync_pending = False
                self._resume_until = time.monotonic() + self.resume_duration
                self._resume_deadline = time.monotonic() + self.resume_max_duration
                dt = NOMINAL_DT
        targets = self.q_target.copy()
        valid = np.zeros(len(targets), dtype=bool)
        g = self.jm.groups

        if self.enabled_groups["waist"]:
            targets[g["waist"]] = waist_targets(r_chest, self.calib.neutral_chest, self.gain_waist)
            valid[g["waist"]] = True
        if self.enabled_groups["head"] and (d := head_direction(sample, r_chest, self.vis_threshold)) is not None:
            targets[g["head"]] = head_targets(d, self.calib.head_neutral, self.gain_head)
            valid[g["head"]] = True

        tracked = []
        for side in SIDES:
            idx = g[f"{side}_arm"]
            vecs = arm_vectors(sample, side, r_chest, self.vis_threshold) if self.enabled_groups["arms"] else None
            if vecs is not None:
                bias = self.calib.arm_bias[side]
                upper, fore = bias @ vecs[0], bias @ vecs[1]
                pts = sample.hands[side]
                h_body = hand_frame(pts) if pts is not None else hand_frame_from_pose(sample, side)
                hand_rot = None if h_body is None else bias @ r_chest.T @ h_body
                # The direct mapping keeps its own history: feeding IK output back as its "previous"
                # pose would let IK's free directions accumulate at the singularities.
                q_direct = solve_arm(upper, fore, hand_rot, self._direct_prev[side])
                self._direct_prev[side] = q_direct
                if self.use_ik:
                    # Seed IK with the direct mapping: STAR1's shoulder is singular in the T-pose
                    # (pitch axis along the upper arm), where a solver warm-started from the
                    # previous pose cannot move the elbow forward. The analytic seed already has
                    # the operator's arm configuration; IK only corrects the wrist position for
                    # the human/robot proportion difference.
                    wrist_t, elbow_t = self.ik[side].targets_from_human(
                        upper, fore, self.calib.upper_arm[side], self.calib.forearm[side])
                    targets[idx] = self.ik[side].solve(wrist_t, elbow_t, hand_rot, q_direct)
                else:
                    targets[idx] = q_direct
                valid[idx] = True
                tracked.append(f"{side[0].upper()}-arm")

            hidx = g[f"{side}_hand"]
            pts = sample.hands[side]
            if pts is not None and self.enabled_groups["hands"]:
                shape = hand_shape(pts)
                targets[hidx] = self.hand_map.targets(shape)
                valid[hidx] = True
                self.closure[side] = shape.closure
                self.grasp[side].update(shape.closure)
                tracked.append(f"{side[0].upper()}-hand")
            self._publish_grasp(side, tracking=pts is not None)

        self.q_target = targets
        q = self.jm.limits.clamp(self.filter(self.jm.correct(targets), dt, valid))
        if not active:
            self._status = ("PAUSED (press Resume in the camera GUI)" if self.gate.active or not self.enabled
                            else f"IDLE: mode is '{self.gate.mode}', mimic inactive")
            return
        q_goal = q
        q = self.rate(q, dt)
        now_m = time.monotonic()
        gliding = now_m < self._resume_until or (
            now_m < self._resume_deadline and float(np.max(np.abs(q_goal - self.q_out))) > self.resume_converged)
        if gliding:                                    # gentle glide back to the operator's pose
            step = self.resume_vmax * min(dt, NOMINAL_DT)
            q = self.q_out + np.clip(q - self.q_out, -step, step)
        else:
            self._resume_deadline = 0.0
        q = deadband(q, self.q_out, self.band)
        self.rate.reset(q)
        self.q_out = q
        self._publish_commands(q)
        # camera capture -> command published (webcam_node and this node share the wall clock)
        latency = (self.get_clock().now().nanoseconds * 1e-9 - now) * 1000.0
        self._latency_ms = latency if self._latency_ms is None else 0.9 * self._latency_ms + 0.1 * latency
        self._status = (f"TRACKING {' '.join(tracked) or 'torso only'} | mirror {'on' if self.mirror else 'off'} | "
                        f"arms {'IK' if self.use_ik else 'direct'} | grip L {'closed' if self.grasp['left'].closed else 'open'} "
                        f"R {'closed' if self.grasp['right'].closed else 'open'} | camera->command {self._latency_ms:.0f} ms")
        if gliding:
            self._status = "RESUMING (gliding to your pose) | " + self._status.split(" | ", 1)[1]
        if time.monotonic() < self._hint_until:
            self._status += " | long pause: press C to recalibrate if tracking feels off"

    def _initialise_from_robot(self) -> None:
        """(Re)start filter, rate limiter, last command and arm history at the robot's pose.

        The controllers' setpoint (last command of any source) where known, else the actual position
        (see module docstring: bumpless transfer, joints blocked by contact).
        """
        actual = np.array([self.joint_state[n] for n in self.jm.names])
        last = np.array([self.last_cmd.get(n, np.nan) for n in self.jm.names])
        known = np.isfinite(last)
        start = np.where(known, last, actual)
        blocked = [f"{n} {d:+.2f}" for n, d, k in zip(self.jm.names, last - actual, known) if k and abs(d) > self.resume_tolerance]
        if blocked:
            self.get_logger().warn(f"resuming with joints away from their setpoint (blocked?): {', '.join(blocked)}; "
                                   "gliding from the setpoint")
        self.q_target = start.copy()
        self._direct_prev = {s: start[self.jm.groups[f"{s}_arm"]].copy() for s in SIDES}
        self.filter.reset(start)
        self.rate.reset(start)
        self.q_out = start
        self._last_frame_stamp = None

    def _mark_paused(self) -> None:
        if self._paused_since is None:
            self._paused_since = time.monotonic()

    def _mark_resumed(self) -> None:
        self._resync_pending = True
        if self._paused_since is not None and time.monotonic() - self._paused_since > self.recalibrate_after:
            self._hint_until = time.monotonic() + 20.0
        self._paused_since = None

    def _on_mode_change(self, old: str | None, new: str) -> None:
        if new == "mimic":
            self._mark_resumed()
        else:
            self._mark_paused()

    def _hold(self, status: str) -> None:
        self._status = status
        for side in SIDES:
            self._publish_grasp(side, tracking=False)

    # ------------------------------------------------------------------ outputs
    def _publish_commands(self, q: np.ndarray) -> None:
        for controller, values in self.jm.split(q).items():
            self.cmd_pubs[controller].publish(Float64MultiArray(data=values))
        js = JointState(name=self.jm.names, position=[float(v) for v in q])
        js.header.stamp = self.get_clock().now().to_msg()
        self.targets_pub.publish(js)

    def _publish_grasp(self, side: str, tracking: bool) -> None:
        msg = GraspState(side=side, closure=float(self.closure[side]), closed=self.grasp[side].closed, tracking=tracking)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.grasp_pubs[side].publish(msg)

    def _on_timer(self) -> None:
        if not self.gate.active:
            self._status = f"IDLE: mode is '{self.gate.mode}' (mimic inactive)"
        elif time.monotonic() - self._last_rx > self.lost_timeout and not self.calibrator.active:
            self._status = "LOST: no landmarks from webcam_node, holding pose"
        self.status_pub.publish(String(data=self._status))

    # ------------------------------------------------------------------ services
    def _srv_calibrate(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self._calibrate_pending = True
        res.success, res.message = True, f"hold a T-pose for {self.calibrator.duration:.0f} s"
        return res

    def _srv_enable(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        if req.data and not self.enabled:
            self._mark_resumed()              # resync to the robot's actual pose on the next frame
        elif not req.data and self.enabled:
            self._mark_paused()
        self.enabled = bool(req.data)
        res.success, res.message = True, "streaming" if self.enabled else "paused"
        return res


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RetargetNode()
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
