"""skill_server_node: executes robot skills for the Physical AI mode (ExecuteSkill action).

The planner (LLM) only ever names skills and object ids; all geometry and motion happens here,
in deterministic code, from what perception reports (never simulator ground truth).

Action    /aibot/skills/execute   aibot_msgs/action/ExecuteSkill (feedback, cancel)
          scan | look_at | home | locate | pick | place | place_in | move_aside | open_hand | release
Services  /aibot/skills/stop      std_srvs/Trigger: cancel the running skill, hold the pose
          /aibot/skills/enable    std_srvs/SetBool: armed by mode_manager_node on entering AI mode
Publishes /{group}_controller/commands  (only the groups a motion moves, only while allowed)
          /aibot/grasp_cmd/{side}       aibot_msgs/GraspState (AI-side input of grasp_manager_node)
Subscribes /joint_states, /aibot/mode, /aibot/perception/objects, /aibot/grasp/status
Calls     /aibot/perception/{detect_now, clear, set_held}

Motion: every move is a minimum-jerk trajectory streamed at 50 Hz from the controllers' current
setpoint (bumpless, see retarget_node for the same rule in mimic mode). Approach, descent and
lift are Cartesian straight lines (dense IK, one time law); transfers are joint-space moves that
are collision-checked (table, torso, other arm) and routed beside the table when needed.

Grasping: the hand closes and the grasp manager welds the object (sim stand-in for friction).
A pick succeeds only if (1) the grasp manager reports the weld from the gz plugin and (2) after
lifting, the head camera no longer sees the object at its old spot. A place succeeds only if the
grasp manager reports the release.

Safety: a skill runs, and the executor publishes, only while enabled AND /aibot/mode == "ai".
Every skill has a timeout. STOP, a goal cancel, or a mode change abort on the next executor
tick (1/rate_hz = 20 ms) and the controllers hold the last command.
"""
from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from aibot_msgs.action import ExecuteSkill
from aibot_msgs.msg import DetectedObject, DetectedObjectArray, GraspState, GraspStatus, PlaceTarget
from aibot_msgs.srv import SetHeld
from aibot_teleop.retarget.hand_curl import HandMapping, HandShape
from aibot_teleop.retarget.joint_map import JointMap
from aibot_teleop.utils.mode_gate import ModeGate

from .geometry.place import CONTAINER_LABELS, PlaceResolver, PlaceSolution, WorldObject, regions
from .motion.arm_motion import ArmMotion, MotionConfig
from .motion.executor import TrajectoryExecutor
from .motion.ik import HeadPointer
from .motion.trajectory import PathTrajectory, Trajectory
from .skills.matching import match
from .skills.result import SkillResult

SIDES = ("left", "right")
Abort = Callable[[], bool]
Feedback = Callable[[str, float], None]


def hand_presets(mapping: HandMapping, grip_fraction: float = 0.5) -> dict[str, np.ndarray]:
    """Open, power-grasp (fist) and grip poses of one hand, from the same mapping mimic mode uses.

    grip: the fist scaled down, the fingers wrap the object instead of closing through it. A
    full fist around a welded object drives the fingers into it (the weld pins the object, the
    position servo keeps pushing); after the release they stay jammed and carry the object away.
    """
    fist = HandShape(curl={k: 1.0 for k in ("index", "middle", "ring", "pinky", "thumb")}, opposition=1.0,
                     spread=mapping.spread_neutral)
    close = mapping.targets(fist)
    return {"open": np.zeros(12), "close": close, "grip": grip_fraction * close}


@dataclass
class Held:
    """What the skill server knows about an object in a hand (from perception at pick time)."""
    id: str
    label: str
    color: str
    size: np.ndarray            # [major, minor, height]
    offset: np.ndarray          # object centre relative to the grasp point, in the hand frame


def _wo(d: DetectedObject) -> WorldObject:
    return WorldObject(d.id, d.label, np.array([d.pose.position.x, d.pose.position.y, d.pose.position.z]),
                       np.array([d.size.x, d.size.y, d.size.z]))


def _xyz(d: DetectedObject) -> np.ndarray:
    return np.array([d.pose.position.x, d.pose.position.y, d.pose.position.z])


class SkillServerNode(Node):
    def __init__(self) -> None:
        super().__init__("skill_server_node")
        p = self.declare_parameter
        teleop_share = Path(get_package_share_directory("aibot_teleop"))
        desc_share = Path(get_package_share_directory("aibot_description"))
        gz_share = Path(get_package_share_directory("aibot_gazebo"))
        self.jm = JointMap(p("joint_mapping_file", str(teleop_share / "config" / "joint_mapping.yaml")).value,
                           p("joint_limits_file", str(desc_share / "config" / "joint_limits.yaml")).value,
                           p("limit_margin", 0.02).value)
        objects_cfg = yaml.safe_load(Path(p("objects_file", str(gz_share / "config" / "objects.yaml")).value).read_text())
        self.table: dict = objects_cfg["table"]
        self.hand_limits: dict = objects_cfg["hand_limits"]
        self.speed: float = p("speed", 0.5).value
        self.max_joint_speed: float = p("max_joint_speed", 2.0).value
        self.hand_close_time: float = p("hand_close_time", 0.5).value
        self.default_timeout: float = p("default_timeout", 20.0).value
        self.timeouts = {"pick": p("pick_timeout", 60.0).value, "place": p("place_timeout", 60.0).value}
        self.timeouts.update(place_in=self.timeouts["place"], move_aside=self.timeouts["pick"] + self.timeouts["place"])
        self.line_speed: float = p("line_speed", 0.10).value
        self.plan_budget: float = p("plan_budget", 25.0).value
        self.arrival_tolerance: float = p("arrival_tolerance", 0.03).value
        self.arrival_timeout: float = p("arrival_timeout", 4.0).value
        self.g = {k: p(f"grasp.{k}", v).value for k, v in {
            "approach_height": 0.10, "lift_height": 0.12, "min_grasp_height": 0.025, "top_offset": 0.0,
            "verify_timeout": 1.5, "still_there_radius": 0.03}.items()}
        self.pl = {k: p(f"place.{k}", v).value for k, v in {
            "drop_height": 0.01, "rim_clearance": 0.02, "max_candidates": 12, "confirm_radius": 0.04,
            "drop_wait": 0.5, "aim_tolerance": 0.015}.items()}
        self.scan_repeats: int = p("scan_repeats", 3).value
        self.wave_pose: list[float] = list(p("gestures.wave_pose", [-0.4, -1.6, -1.0, -1.6, 0.0, 0.0, 0.0]).value)
        self.wave_amplitude: float = p("gestures.wave_amplitude", 0.5).value
        self.wave_cycles: int = p("gestures.wave_cycles", 2).value
        self.nod_amplitude: float = p("gestures.nod_amplitude", 0.15).value
        self.shake_amplitude: float = p("gestures.shake_amplitude", 0.3).value
        self.place_approach_heights: list[float] = list(p("place.approach_heights", [0.10, 0.06, 0.03]).value)
        self.look_at_max_error: float = p("look_at_max_error", 0.35).value
        rate = p("rate_hz", 50.0).value
        self.home = {g: np.array(p(f"home.{g}", [0.0] * len(self.jm.groups[g])).value, dtype=float)
                     for g in ("waist", "head", "left_arm", "right_arm")}
        self.presets = hand_presets(HandMapping(), p("grasp.grip_fraction", 0.5).value)

        urdf = p("robot_description", "").value
        if not urdf:
            raise SystemExit("skill_server_node needs the robot_description parameter (the URDF string)")
        self.motion = ArmMotion(urdf, self.jm.names, self.jm.limits.lower, self.jm.limits.upper, self.table,
                                MotionConfig(), waist_lower=np.array(p("ik.waist_lower", [-0.6, -0.05, -0.40]).value),
                                waist_upper=np.array(p("ik.waist_upper", [0.6, 0.05, 0.10]).value))
        self.neck_idx = np.array([self.jm.index[n] for n in ("neck_yaw_joint", "neck_pitch_joint")])
        self.waist_idx = np.array([self.jm.index[n] for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")])
        self.head = HeadPointer(urdf, self.jm.limits.lower[self.neck_idx], self.jm.limits.upper[self.neck_idx])
        self.resolver = PlaceResolver(self.table)

        self._js_msg: JointState | None = None
        self._rtf_ref: tuple[float, float] | None = None
        self.follow_sim_time: bool = p("follow_sim_time", True).value
        self.rtf_window: float = p("rtf_window", 1.0).value
        self.min_time_scale: float = p("min_time_scale", 0.05).value
        self._objects: DetectedObjectArray | None = None
        self._grasp_status: GraspStatus | None = None
        self.enabled = False
        self._stop = threading.Event()
        self.grasp_closed = {s: False for s in SIDES}
        self.holding: dict[str, Held | None] = {s: None for s in SIDES}
        self.cb = ReentrantCallbackGroup()
        self.gate = ModeGate(self, "ai", p("mode_required", True).value, self._on_mode_change)

        self.cmd_pubs = {c: self.create_publisher(Float64MultiArray, f"/{c}/commands", 5) for c in self.jm.controllers}
        self.grasp_pubs = {s: self.create_publisher(GraspState, f"/aibot/grasp_cmd/{s}", 5) for s in SIDES}
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10, callback_group=self.cb)
        self.create_subscription(DetectedObjectArray, "/aibot/perception/objects", lambda m: setattr(self, "_objects", m),
                                 5, callback_group=self.cb)
        self.create_subscription(GraspStatus, "/aibot/grasp/status", self._on_grasp_status, 5, callback_group=self.cb)
        self.setpoint = np.full(len(self.jm.names), np.nan)     # last command of any source, per joint
        for c, joints in self.jm.controllers.items():
            idx = np.array([self.jm.index[j] for j in joints])
            self.create_subscription(Float64MultiArray, f"/{c}/commands",
                                     lambda m, idx=idx: self.setpoint.__setitem__(idx, m.data), 10, callback_group=self.cb)
        self.cli_detect = self.create_client(Trigger, "/aibot/perception/detect_now", callback_group=self.cb)
        self.cli_clear = self.create_client(Trigger, "/aibot/perception/clear", callback_group=self.cb)
        self.cli_held = self.create_client(SetHeld, "/aibot/perception/set_held", callback_group=self.cb)
        self.executor_ = TrajectoryExecutor(self.jm.limits.lower, self.jm.limits.upper, self._publish_q, rate,
                                            p("max_step_factor", 1.25).value)
        # A plain thread, not an rclpy timer: inside the MultiThreadedExecutor (200 Hz joint states,
        # action threads) a 50 Hz timer measured ~29 Hz with stalls up to 0.5 s.
        self._period = 1.0 / rate
        threading.Thread(target=self._tick_loop, name="executor_tick", daemon=True).start()
        self.create_timer(1.0 / p("grasp_keepalive_hz", 5.0).value, self._publish_grasp, callback_group=self.cb)
        self.create_service(Trigger, "/aibot/skills/stop", self._srv_stop, callback_group=self.cb)
        self.create_service(SetBool, "/aibot/skills/enable", self._srv_enable, callback_group=self.cb)

        self.skills: dict[str, Callable[[ExecuteSkill.Goal, Abort, Feedback], SkillResult]] = {
            "scan": self.skill_scan, "look_at": self.skill_look_at, "home": self.skill_home,
            "locate": self.skill_locate, "pick": self.skill_pick, "place": self.skill_place,
            "place_in": self.skill_place_in, "move_aside": self.skill_move_aside,
            "open_hand": self.skill_open_hand, "close_hand": self.skill_close_hand, "release": self.skill_release,
            "wave": self.skill_wave, "nod": self.skill_nod, "shake_head": self.skill_shake_head,
        }
        self._busy = threading.Lock()
        self.action = ActionServer(self, ExecuteSkill, "/aibot/skills/execute", self._execute,
                                   goal_callback=self._on_goal, cancel_callback=lambda _g: CancelResponse.ACCEPT,
                                   callback_group=self.cb)
        self.get_logger().info(f"skill server ready: {sorted(self.skills)} (speed {self.speed:.0%} of URDF limits)")

    # ------------------------------------------------------------------ state
    @property
    def can_command(self) -> bool:
        return self.enabled and self.gate.active

    def _tick_loop(self) -> None:
        next_t = time.monotonic()
        while rclpy.ok():
            self.executor_.tick(self.can_command)
            next_t += self._period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()      # fell behind: do not try to catch up with a burst

    def _on_joint_states(self, msg: JointState) -> None:
        self._js_msg = msg               # 200 Hz: just keep it; parse only when a motion starts
        # real-time factor from the stamps (sim time) against arrival (wall time), no /clock needed
        now, st = time.monotonic(), msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self._rtf_ref is None or now - self._rtf_ref[0] > self.rtf_window:
            if self._rtf_ref is not None and st > self._rtf_ref[1]:
                rtf = (st - self._rtf_ref[1]) / (now - self._rtf_ref[0])
                self.executor_.time_scale = float(np.clip(rtf, self.min_time_scale, 1.0)) if self.follow_sim_time else 1.0
            self._rtf_ref = (now, st)

    @property
    def q_actual(self) -> np.ndarray | None:
        msg = self._js_msg
        if msg is None:
            return None
        pos = dict(zip(msg.name, msg.position))
        if not all(n in pos for n in self.jm.names):
            return None
        return np.array([pos[n] for n in self.jm.names])

    def _on_mode_change(self, _old: str | None, new: str) -> None:
        if new != "ai":
            self.executor_.stop()

    def _publish_q(self, q: np.ndarray, groups: list[str]) -> None:
        for g in groups:
            idx = self.jm.groups[g]
            self.setpoint[idx] = q[idx]          # our own command is the setpoint now (no topic round trip)
            self.cmd_pubs[f"{g}_controller"].publish(Float64MultiArray(data=[float(v) for v in q[idx]]))

    def _publish_grasp(self) -> None:
        if not self.can_command:
            return
        for side in SIDES:
            msg = GraspState(side=side, closure=1.0 if self.grasp_closed[side] else 0.0,
                             closed=self.grasp_closed[side], tracking=True)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.grasp_pubs[side].publish(msg)

    def world(self) -> list[DetectedObject]:
        return list(self._objects.objects) if self._objects is not None else []

    def find(self, object_id: str) -> DetectedObject | None:
        return next((o for o in self.world() if o.id == object_id), None)

    def weld(self, side: str) -> str:
        """Object the grasp manager reports welded to this hand ("" if none / no report yet)."""
        st = self._grasp_status
        return "" if st is None else (st.left_object if side == "left" else st.right_object)

    # ------------------------------------------------------------------ services
    def _srv_stop(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self._stop.set()
        self.executor_.stop()
        res.success, res.message = True, "skills stopped, holding pose"
        return res

    def _srv_enable(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        self.enabled = bool(req.data)
        if not self.enabled:
            self._stop.set()
            self.executor_.stop()
        else:
            self._stop.clear()
            self.grasp_closed = {s: False for s in SIDES}
            self.holding = {s: None for s in SIDES}     # the switch to mimic released everything
            threading.Thread(target=self._look_at_table, daemon=True).start()
        res.success, res.message = True, "armed" if self.enabled else "disarmed"
        return res

    def _look_at_table(self) -> None:
        """On entering AI mode the head goes to the scan pose (mimic left it wherever the operator
        looked), so the camera view and perception show the table right away. Yields to skills."""
        time.sleep(0.3)                                   # the mode is published right after arming
        if not self._busy.acquire(blocking=False):
            return
        try:
            self.move_groups({"head": self.home["head"]}, lambda: not self.can_command, 0.8)
        finally:
            self._busy.release()

    def _call(self, client, request, timeout: float = 7.0):
        """Blocking service call from a skill thread (the reentrant group serves the response)."""
        if not client.wait_for_service(timeout_sec=1.0):
            return None
        done = threading.Event()
        fut = client.call_async(request)
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            client.remove_pending_request(fut)
            return None
        return fut.result()

    def detect_now(self) -> bool:
        r = self._call(self.cli_detect, Trigger.Request())
        if r is not None and r.success:
            time.sleep(0.05)                     # let the published array arrive
            return True
        return False

    def set_held(self, object_id: str, side: str, position: np.ndarray | None = None, container: str = "") -> None:
        req = SetHeld.Request(object_id=object_id, side=side, has_position=position is not None, container_id=container)
        if position is not None:
            req.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        if self._call(self.cli_held, req) is None:
            self.get_logger().warn(f"perception set_held({object_id}, '{side}') failed")

    def _on_grasp_status(self, msg: GraspStatus) -> None:
        self._grasp_status = msg
        # a weld that vanished while no skill runs (scene reset, release_all, mode switch): forget it
        # now, so the planner's world state does not keep showing the object in the hand
        if any(self.holding[s_] and not self.weld(s_) for s_ in SIDES) and self._busy.acquire(blocking=False):
            try:
                self._reconcile_holding()
            finally:
                self._busy.release()

    def _reconcile_holding(self) -> None:
        """Forget objects the hand no longer reports (scene reset / release_all / mode switch)."""
        if self._grasp_status is None:
            return
        for side in SIDES:
            h = self.holding[side]
            if h is not None and not self.weld(side):
                self.get_logger().info(f"{h.id} is no longer in the {side} hand")
                self.holding[side] = None
                self.set_held(h.id, "")
            if self.holding[side] is None and self.grasp_closed[side]:
                self.grasp_closed[side] = False          # stop asking the grasp manager to hold
                self._publish_grasp()

    # ------------------------------------------------------------------ action
    def _on_goal(self, goal: ExecuteSkill.Goal) -> GoalResponse:
        if goal.skill not in self.skills:
            self.get_logger().warn(f"unknown skill '{goal.skill}'")
        return GoalResponse.ACCEPT     # problems are reported in the result, with a reason code

    def _execute(self, goal_handle) -> ExecuteSkill.Result:
        goal: ExecuteSkill.Goal = goal_handle.request
        t0 = time.monotonic()
        if not self.can_command:
            res = SkillResult.fail("NOT_AI_MODE", f"mode is '{self.gate.mode}', skills need 'ai' and an armed server")
        elif goal.skill not in self.skills:
            res = SkillResult.fail("INVALID", f"unknown skill '{goal.skill}'; known: {sorted(self.skills)}")
        elif not self._busy.acquire(blocking=False):
            res = SkillResult.fail("BUSY", "another skill is running")
        else:
            try:
                self._stop.clear()
                self._reconcile_holding()
                limit = goal.timeout or self.timeouts.get(goal.skill, self.default_timeout)
                # timeouts count robot time: in a simulation running at 0.4x a skill takes 2.5x the wall time
                clock = {"t": 0.0, "last": time.monotonic()}

                def elapsed() -> float:
                    now = time.monotonic()
                    clock["t"] += (now - clock["last"]) * self.executor_.time_scale
                    clock["last"] = now
                    return clock["t"]

                def should_abort() -> bool:
                    return (goal_handle.is_cancel_requested or self._stop.is_set() or not self.can_command
                            or elapsed() > limit)

                def feedback(stage: str, progress: float) -> None:
                    goal_handle.publish_feedback(ExecuteSkill.Feedback(stage=stage, progress=float(progress)))

                try:
                    res = self.skills[goal.skill](goal, should_abort, feedback)
                except Exception as exc:          # a bug must not leave the action hanging
                    self.get_logger().error(f"skill {goal.skill} raised: {exc!r}")
                    self.executor_.stop()
                    res = SkillResult.fail("INVALID", f"internal error: {exc}")
                if not res.success and res.reason_code == "ABORTED":
                    res = SkillResult.fail("TIMEOUT" if elapsed() > limit else "CANCELLED", res.message)
            finally:
                self._busy.release()
        out = res.to_msg(ExecuteSkill.Result())
        if res.reason_code == "CANCELLED" and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif res.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f"skill {goal.skill}({goal.object_id or goal.text}) -> {res.reason_code} "
                               f"in {time.monotonic() - t0:.1f} s: {res.message}")
        return out

    # ------------------------------------------------------------------ motion primitives
    def start_pose(self) -> np.ndarray | None:
        """Bumpless start: controllers' setpoint where known, else the actual joint positions."""
        actual = self.q_actual
        if actual is None:
            return None
        sp = self.setpoint.copy()
        return np.where(np.isfinite(sp), sp, actual)

    @property
    def v_max(self) -> np.ndarray:
        return np.minimum(self.speed * self.jm.limits.velocity, self.max_joint_speed)

    def _run(self, traj, groups: list[str], should_abort: Abort, what: str) -> SkillResult:
        self.executor_.start(traj, groups)
        # trajectory time runs at the sim's real-time factor: allow for it (and for it dropping)
        outcome = self.executor_.wait(2.0 * traj.duration / max(self.executor_.time_scale, self.min_time_scale) + 2.0,
                                      should_abort)
        if outcome != "done":
            return SkillResult.fail("ABORTED" if outcome == "aborted" else "TIMEOUT", f"{what}: motion {outcome}")
        return self._wait_arrival(traj.goal, groups, should_abort, what)

    def _wait_arrival(self, goal: np.ndarray, groups: list[str], should_abort: Abort, what: str) -> SkillResult:
        """The trajectory is streamed in wall time; the robot (or a simulation running slower than
        real time) arrives later. Grasp checks, closing the hand and camera views need the arm
        where it was sent, so wait until the measured joints are within tolerance of the goal.
        Hands are not waited for: fingers stop on the object by design."""
        idx = np.concatenate([self.jm.groups[g] for g in groups if not g.endswith("_hand")] or [np.array([], int)])
        if not len(idx):
            return SkillResult.ok()
        t_end = time.monotonic() + self.arrival_timeout
        err = np.inf
        while time.monotonic() < t_end:
            if should_abort():
                return SkillResult.fail("ABORTED", f"{what}: aborted while settling")
            q = self.q_actual
            if q is not None:
                err = float(np.max(np.abs(q[idx] - goal[idx])))
                if err < self.arrival_tolerance:
                    return SkillResult.ok()
            time.sleep(0.02)
        self.get_logger().warn(f"{what}: still {err:.3f} rad from the goal after {self.arrival_timeout:.1f} s "
                               "(blocked by contact?), continuing")
        return SkillResult.ok()

    def move_groups(self, targets: dict[str, np.ndarray], should_abort: Abort, min_duration: float = 0.3) -> SkillResult:
        """Minimum-jerk move of the given groups from the current setpoint to `targets`."""
        q0 = self.start_pose()
        if q0 is None:
            return SkillResult.fail("INVALID", "no /joint_states yet")
        q1 = q0.copy()
        for g, values in targets.items():
            q1[self.jm.groups[g]] = values
        q1 = np.clip(q1, self.jm.limits.lower, self.jm.limits.upper)
        return self._run(Trajectory([q0, q1], self.v_max, 1.0, min_duration), list(targets), should_abort, "move")

    def move_waypoints(self, side: str, waypoints: list[np.ndarray], should_abort: Abort) -> SkillResult:
        q0 = self.start_pose()
        traj = Trajectory([q0] + [np.asarray(w, float) for w in waypoints], self.v_max, 1.0, 0.4)
        return self._run(traj, ["waist", f"{side}_arm"], should_abort, "transfer")

    def move_line(self, side: str, path: list[np.ndarray], should_abort: Abort) -> SkillResult:
        q0 = self.start_pose()
        path = [q0] + [np.asarray(q, float) for q in path[1:]]
        a, _ = self.motion.grasp_point(side, path[0])
        b, _ = self.motion.grasp_point(side, path[-1])
        min_t = 1.875 * float(np.linalg.norm(b - a)) / self.line_speed       # Cartesian speed cap
        return self._run(PathTrajectory(path, self.v_max, 1.0, max(min_t, 0.3)), ["waist", f"{side}_arm"],
                         should_abort, "line")

    def move_legs(self, side: str, legs: list[tuple[str, list[np.ndarray]]], should_abort: Abort) -> SkillResult:
        """Run a planned transfer: joint legs as waypoint trajectories, paths with one time law."""
        for kind, data in legs:
            res = self.move_waypoints(side, data, should_abort) if kind == "joint" else self.move_line(side, data, should_abort)
            if not res.success:
                return res
        return SkillResult.ok()

    def set_hand(self, side: str, closed: bool, should_abort: Abort, preset: str | None = None) -> SkillResult:
        """The weld holds the object whenever the fingers move. Close: weld first, then the fingers
        (closing on a free object pushed it away before the weld caught it). Open: fingers first,
        then release the weld (fingers that dug into the welded object would otherwise carry it
        off once it is free; measured: a ball thrown 0.6 m when the arm went home). The grip is
        a partial fist so the fingers rest on the object instead of closing through it."""
        if closed:
            self.grasp_closed[side] = True
            self._publish_grasp()
        res = self.move_groups({f"{side}_hand": self.presets[preset or ("grip" if closed else "open")]}, should_abort,
                               self.hand_close_time)
        if not closed:
            self.grasp_closed[side] = False       # also on an aborted opening: never keep a weld
            self._publish_grasp()
        return res

    def update_scene(self, exclude: set[str] = frozenset()) -> None:
        """Perceived objects become obstacles for the next plans (minus `exclude`: the object to
        grasp, the support to place on); carried objects are checked along with the arm."""
        ob, ids = [], []
        for o in self.world():
            if o.held_by or o.id in exclude or o.inside:
                continue
            w = _wo(o)
            q = o.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            ob.append((float(w.position[0]), float(w.position[1]), 0.5 * float(o.size.x), 0.5 * float(o.size.y),
                       yaw, w.top_z))
            ids.append(o.id)
        attached = {s: (h.offset, 0.5 * float(h.size[2])) if h else None for s, h in self.holding.items()}
        self.motion.set_scene(ob, attached, ids)

    def hand_over_table(self, side: str, q: np.ndarray, margin: float = 0.05) -> bool:
        p, _ = self.motion.grasp_point(side, q)
        t = self.table
        return (t["x"][0] - margin < p[0] < t["x"][1] + margin and t["y"][0] - margin < p[1] < t["y"][1] + margin
                and p[2] < t["top_z"] + 0.35)

    # ------------------------------------------------------------------ skills: looking
    def skill_home(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Arms down, waist straight, head at the scan pose. An arm over the table first rises
        vertically, then leaves via the side of the table (never swept through the table edge)."""
        q = self.start_pose()
        if q is None:
            return SkillResult.fail("INVALID", "no /joint_states yet")
        note = ""
        home_full = q.copy()
        for g, v in self.home.items():
            home_full[self.jm.groups[g]] = v
        self.update_scene()
        for side in SIDES:
            q = self.start_pose()
            if not self.hand_over_table(side, q):
                continue
            feedback(f"{side} arm leaving the table", 0.1)
            p, _ = self.motion.grasp_point(side, q)
            safe_z = self.table["top_z"] + self.motion.cfg.beside_height
            if p[2] < safe_z - 0.01:
                path, why = self.motion.line(side, q, np.array([p[0], p[1], safe_z]))
                if path is not None:
                    res = self.move_line(side, path, should_abort)
                    if not res.success:
                        return res
                    q = self.start_pose()
            q_goal = self.motion.with_arm(side, q, self.motion.arm_q(side, home_full))
            via = self.motion.beside(side, q)
            if via is not None:                     # obstacle-aware way off the table, then down
                legs, why = self.motion.transfer(side, q, via)
                if legs is not None and not self.motion.path_hit(side, via, q_goal):
                    res = self.move_legs(side, legs + [("joint", [q_goal])], should_abort)
                    if not res.success:
                        return res
                    continue
            wps, why = self.motion.joint_move(side, q, q_goal)
            if wps is None and "object" in why:
                # last resort for the recovery move (perception may be confused by the arm in view):
                # leave via the side ignoring perceived objects; this can push them
                self.get_logger().warn(f"home: {why}; retrying without perceived obstacles")
                note = " (ignored perceived obstacles)"
                self.motion.set_scene([], {s_: None for s_ in SIDES})
                wps, why = self.motion.joint_move(side, q, q_goal)
                self.update_scene()
            if wps is None:
                return SkillResult.fail("UNREACHABLE", f"cannot bring the {side} arm home: {why}")
            res = self.move_waypoints(side, wps, should_abort)
            if not res.success:
                return res
        feedback("moving home", 0.8)
        targets = dict(self.home)
        for side in SIDES:                          # an empty hand left half-closed opens too
            if self.holding[side] is None:
                targets[f"{side}_hand"] = self.presets["open"]
                self.grasp_closed[side] = False
        res = self.move_groups(targets, should_abort, min_duration=1.0)
        return SkillResult.ok("at home" + note) if res.success else res

    def skill_scan(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Home pose (whole table in view), forget old detections, detect a few times."""
        res = self.skill_home(goal, should_abort, feedback)
        if not res.success:
            return res
        held_ids = {h.id for h in self.holding.values() if h}
        self._call(self.cli_clear, Trigger.Request())
        for o in self.world():                   # stale "held" marks the robot no longer has
            if o.held_by and o.id not in held_ids:
                self.set_held(o.id, "")
        for i in range(self.scan_repeats):
            if should_abort():
                return SkillResult.fail("ABORTED", "scan aborted")
            feedback("detecting", 0.5 + 0.5 * i / self.scan_repeats)
            if not self.detect_now():
                return SkillResult.fail("NOT_FOUND", "no head-camera frame (is the simulation running?)")
        objs = [o for o in self.world() if not o.held_by]
        summary = [{"id": o.id, "label": o.label, "color": o.color,
                    "xy": [round(o.pose.position.x, 3), round(o.pose.position.y, 3)]} for o in objs]
        return SkillResult.ok(json.dumps(summary), object_id=",".join(o.id for o in objs))

    def skill_look_at(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Point the head camera at an object (object_id) or a table region (text)."""
        reg = regions(self.table)
        if goal.object_id:
            o = self.find(goal.object_id)
            if o is None:
                return SkillResult.fail("NOT_FOUND", f"no object '{goal.object_id}' in the world state")
            target = _xyz(o)
        elif goal.text in reg:
            target = np.array([*reg[goal.text], self.table["top_z"]])
        else:
            return SkillResult.fail("INVALID", f"look_at needs object_id or a region ({', '.join(reg)})")
        q = self.start_pose()
        neck, err = self.head.solve(target, q[self.waist_idx], q[self.neck_idx])
        res = self.move_groups({"head": neck}, should_abort, 0.5)
        if not res.success:
            return res
        if err > self.look_at_max_error:
            return SkillResult.fail("UNREACHABLE", f"target is {math.degrees(err):.0f} deg off the camera axis: out of view")
        return SkillResult.ok(f"looking at {goal.object_id or goal.text} ({math.degrees(err):.0f} deg off centre)",
                              object_id=goal.object_id, pose=tuple(float(v) for v in target))

    def skill_locate(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Resolve a description to one object id. Several matches -> AMBIGUOUS (ask the user)."""
        text = goal.text or goal.object_id
        if not text:
            return SkillResult.fail("INVALID", "locate needs text (e.g. 'red cube') or an object_id")
        cands = match(text, [o for o in self.world() if not o.held_by])
        if not cands:
            feedback("looking again", 0.5)
            self.detect_now()
            cands = match(text, [o for o in self.world() if not o.held_by])
        if not cands:
            seen = ", ".join(o.id for o in self.world()) or "nothing"
            return SkillResult.fail("NOT_FOUND", f"no '{text}' on the table (I see: {seen})")
        if len(cands) > 1:
            return SkillResult.fail("AMBIGUOUS", f"'{text}' matches {len(cands)} objects: {', '.join(o.id for o in cands)}",
                                    object_id=",".join(o.id for o in cands))
        o = cands[0]
        return SkillResult.ok(f"{o.id} at ({o.pose.position.x:.2f}, {o.pose.position.y:.2f})", object_id=o.id,
                              pose=tuple(float(v) for v in _xyz(o)))

    # ------------------------------------------------------------------ skills: gestures
    def skill_wave(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Raise a free hand beside the head (clear of the table on the way up), wave, lower it."""
        free = [s_ for s_ in (["right", "left"] if goal.side not in SIDES else [goal.side]) if self.holding[s_] is None]
        if not free:
            return SkillResult.fail("INVALID", "both hands are holding something")
        side = free[0]
        q = self.start_pose()
        if self.hand_over_table(side, q):
            r = self.skill_home(goal, should_abort, feedback)
            if not r.success:
                return r
        g = f"{side}_arm"
        up = np.array(self.wave_pose)                      # right arm; the left one is the mirror image
        if side == "left":
            up = up * np.array([1, -1, -1, 1, -1, 1, -1])
        feedback("waving", 0.2)
        r = self.move_groups({g: up}, should_abort, 1.0)
        for k in range(self.wave_cycles * 2):
            if not r.success:
                return r
            wave = up.copy()
            wave[4] += (1 if k % 2 == 0 else -1) * self.wave_amplitude      # elbow yaw swings the forearm
            r = self.move_groups({g: wave}, should_abort, 0.35)
        if not r.success:
            return r
        r = self.move_groups({g: self.home[g]}, should_abort, 1.0)
        return SkillResult.ok(f"waved with the {side} hand") if r.success else r

    def _head_gesture(self, axis: int, amplitude: float, should_abort: Abort, what: str) -> SkillResult:
        rest = self.start_pose()[self.jm.groups["head"]]
        r = SkillResult.ok()
        for k in range(4):
            target = rest.copy()
            target[axis] += amplitude * (1 if k % 2 == 0 else -1)
            target = np.clip(target, self.jm.limits.lower[self.neck_idx], self.jm.limits.upper[self.neck_idx])
            r = self.move_groups({"head": target}, should_abort, 0.3)
            if not r.success:
                return r
        r = self.move_groups({"head": rest}, should_abort, 0.3)
        return SkillResult.ok(what) if r.success else r

    def skill_nod(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        return self._head_gesture(1, self.nod_amplitude, should_abort, "nodded")

    def skill_shake_head(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        return self._head_gesture(0, self.shake_amplitude, should_abort, "shook the head")

    # ------------------------------------------------------------------ skills: hands
    def skill_open_hand(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        sides = [goal.side] if goal.side in SIDES else list(SIDES)
        res = self.move_groups({f"{s}_hand": self.presets["open"] for s in sides}, should_abort, self.hand_close_time)
        for side in sides:                        # fingers first, then the weld (see set_hand)
            self.grasp_closed[side] = False
        self._publish_grasp()
        return SkillResult.ok(f"opened {', '.join(sides)}") if res.success else res

    def skill_close_hand(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Power grasp on one hand; the grasp manager welds an object if one is at the grasp point."""
        if goal.side not in SIDES:
            return SkillResult.fail("INVALID", "close_hand needs side=left|right")
        res = self.set_hand(goal.side, True, should_abort)
        return SkillResult.ok(f"closed {goal.side}") if res.success else res

    def skill_release(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Open the hand(s) where they are; a held object drops and perception is told where."""
        sides = [goal.side] if goal.side in SIDES else [s for s in SIDES if self.holding[s]] or list(SIDES)
        q = self.start_pose()
        res = self.skill_open_hand(goal if goal.side in SIDES else ExecuteSkill.Goal(), should_abort, feedback)
        for side in sides:
            h = self.holding[side]
            if h is not None:
                gp, r = self.motion.grasp_point(side, q)
                self.set_held(h.id, "", gp + r @ h.offset)
                self.holding[side] = None
        return res

    # ------------------------------------------------------------------ skills: pick
    def _grasp_targets(self, o: DetectedObject) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = float(o.size.z)
        top = float(o.pose.position.z) + 0.5 * h
        z = max(self.table["top_z"] + self.g["min_grasp_height"], top - min(self.g["top_offset"], 0.5 * h))
        grasp = np.array([o.pose.position.x, o.pose.position.y, z])
        return (grasp + [0.0, 0.0, self.g["approach_height"]], grasp, grasp + [0.0, 0.0, self.g["lift_height"]])

    def _plan_pick(self, side: str, grasp: np.ndarray):
        """Grasp-first: IK at the grasp point (obstacle-aware, finger directions ranked by
        clearance), vertical lines up to the pre-grasp and lift heights, and a collision-free
        transfer from the current pose to the pre-grasp (the first candidate that has one)."""
        q = self.start_pose()
        legs, _q_grasp, paths = self.motion.approach(side, q, grasp, (self.g["approach_height"], self.g["lift_height"]),
                                                     allow_near=grasp, deadline=time.monotonic() + self.plan_budget)
        if legs is None:
            return None, paths
        to_pre, up = paths
        return (legs, to_pre[::-1], up), ""

    def skill_pick(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        if self.find(goal.object_id) is not None and not any(self.hand_over_table(s, self.start_pose()) for s in SIDES):
            self.detect_now()                    # fresh estimate (the arms are out of the view)
        o = self.find(goal.object_id)
        if o is None:
            return SkillResult.fail("NOT_FOUND", f"no object '{goal.object_id}' in the world state (scan or locate first)")
        if o.held_by:
            return SkillResult.fail("INVALID", f"{o.id} is already in the {o.held_by} hand")
        if o.label in CONTAINER_LABELS:
            return SkillResult.fail("INVALID", f"{o.id} is a fixed container, it cannot be picked")
        width, height, lim = min(o.size.x, o.size.y), o.size.z, self.hand_limits
        if width > lim["max_width"] or height > lim["max_height"]:
            return SkillResult.fail("TOO_LARGE", f"{o.id} is {width * 100:.0f} cm wide and {height * 100:.0f} cm tall; "
                                    f"the hand holds up to {lim['max_width'] * 100:.0f} x {lim['max_height'] * 100:.0f} cm")
        if width < lim["min_width"]:
            return SkillResult.fail("INVALID", f"{o.id} is only {width * 100:.1f} cm wide: too thin for the hand")
        free = [s for s in SIDES if self.holding[s] is None]
        if goal.side in SIDES:
            free = [s for s in free if s == goal.side]
        if not free:
            return SkillResult.fail("INVALID", f"no free hand ({goal.side or 'both'} holding something)")
        free.sort(key=lambda s: (s == "left") != (o.pose.position.y >= 0))        # nearer arm first
        _pre, grasp, _lift = self._grasp_targets(o)
        feedback("planning", 0.05)
        self.update_scene(exclude={o.id})
        plan, side, reasons = None, "", []
        for s in free:
            plan, why = self._plan_pick(s, grasp)
            if plan is not None:
                side = s
                break
            reasons.append(f"{s} arm: {why}")
        if plan is None:
            return SkillResult.fail("UNREACHABLE", f"cannot reach {o.id}: " + "; ".join(reasons))
        legs, down, up = plan
        origin = _xyz(o)

        feedback(f"{side} hand opening", 0.1)
        for step, res in (("open", lambda: self.set_hand(side, False, should_abort)),
                          ("approach", lambda: self.move_legs(side, legs, should_abort)),
                          ("descend", lambda: self.move_line(side, down, should_abort))):
            feedback(step, {"open": 0.1, "approach": 0.2, "descend": 0.5}[step])
            r = res()
            if not r.success:
                return r
        gp, rot = self.motion.grasp_point(side, self.start_pose())
        offset = rot.T @ (origin - gp)
        feedback("closing", 0.6)
        r = self.set_hand(side, True, should_abort)
        if not r.success:
            return r
        t_end = time.monotonic() + self.g["verify_timeout"] / max(self.executor_.time_scale, self.min_time_scale)
        while not self.weld(side) and time.monotonic() < t_end and not should_abort():
            time.sleep(0.05)
        if not self.weld(side):
            self.set_hand(side, False, should_abort)
            self.move_line(side, up, should_abort)
            return SkillResult.fail("GRASP_FAILED", f"closed the {side} hand at {o.id} but nothing was grasped")
        feedback("lifting", 0.8)
        r = self.move_line(side, up, should_abort)
        if not r.success:
            return r
        self.holding[side] = Held(o.id, o.label, o.color, np.array([o.size.x, o.size.y, o.size.z]), offset)
        self.set_held(o.id, side)
        # carry it beside the table: out of the camera's view, so the check below (and the world
        # state the next place is planned in) is not made of half-occluded fragments
        self.update_scene(exclude={o.id})
        q = self.start_pose()
        carry = self.motion.beside(side, q)
        wps = self.motion.joint_move(side, q, carry)[0] if carry is not None else None
        if wps is not None:
            feedback("carrying", 0.85)
            r = self.move_waypoints(side, wps, should_abort)
            if not r.success:
                return r
        feedback("verifying", 0.9)
        if self.detect_now():
            left_behind = [w for w in self.world() if not w.held_by and w.id != o.id and w.label == o.label
                           and w.color == o.color and np.linalg.norm(_xyz(w)[:2] - origin[:2]) < self.g["still_there_radius"]]
            if left_behind:
                self.set_hand(side, False, should_abort)
                self.set_held(o.id, "")
                self.holding[side] = None
                return SkillResult.fail("GRASP_FAILED", f"the camera still sees {o.id} on the table after lifting")
            seen = "camera confirms it left the table"
        else:
            seen = "no camera frame to confirm"
        gp, _ = self.motion.grasp_point(side, self.start_pose())
        return SkillResult.ok(f"holding {o.id} in the {side} hand (grasp reported by the hand; {seen})",
                              object_id=o.id, pose=tuple(float(v) for v in gp))

    # ------------------------------------------------------------------ skills: place
    def _holding_side(self, object_id: str) -> tuple[str | None, str]:
        sides = [s for s in SIDES if self.holding[s] is not None and (not object_id or self.holding[s].id == object_id)]
        if not sides:
            return None, f"not holding {object_id or 'anything'}"
        if len(sides) > 1:
            return None, "holding two objects: say which one to place"
        side = sides[0]
        if not self.weld(side):
            held = self.holding[side].id
            self.holding[side] = None
            self.set_held(held, "")
            return None, f"{held} is no longer in the {side} hand (the hand reports no grasp)"
        return side, ""

    def _plan_place(self, side: str, held: Held, sol: PlaceSolution):
        h = float(held.size[2])
        bottom = sol.surface_z + (self.pl["rim_clearance"] if sol.container_id else self.pl["drop_height"])
        reasons = []
        deadline = time.monotonic() + self.plan_budget
        for x, y in sol.candidates[: int(self.pl["max_candidates"])]:
            if time.monotonic() > deadline:
                reasons.append("planning time budget used up")
                break
            centre = np.array([x, y, bottom + 0.5 * h])
            q = self.start_pose()
            _, rot = self.motion.grasp_point(side, q)
            g = centre - rot @ held.offset
            # the object hangs at `offset` in the hand frame, which turns with the hand: aim with
            # the current hand yaw, then re-aim once with the chosen release pose's yaw
            # a tall object released high (a can into the bin) leaves less room above: lower approaches
            legs = None
            for app in self.place_approach_heights:
                q_rel, paths = self.motion.vertical(side, q, g, (app,), allow_near=g)
                if q_rel is not None:
                    _, rot = self.motion.grasp_point(side, q_rel)
                    g = centre - rot @ held.offset
                legs, q_rel, paths = self.motion.approach(side, q, g, (app,), allow_near=g, deadline=deadline)
                if legs is not None:
                    break
            if legs is None:
                reasons.append(f"({x:.2f},{y:.2f}): {paths}")
                continue
            gp, rot = self.motion.grasp_point(side, q_rel)
            if np.linalg.norm(gp + rot @ held.offset - centre) > self.pl["aim_tolerance"]:
                g2 = centre - rot @ held.offset           # the chosen pose's yaw differs: re-aim once
                again = self.motion.approach(side, q, g2, (app,), allow_near=g2)
                if again[0] is not None:
                    legs, q_rel, paths = again
            up = paths[0]
            down = up[::-1]
            return (legs, down, up, centre), ""
        return None, "; ".join(reasons[:3])

    def _place(self, side: str, sol: PlaceSolution, should_abort: Abort, feedback: Feedback,
               support: str = "") -> SkillResult:
        held = self.holding[side]
        feedback("planning", 0.05)
        self.update_scene(exclude={held.id, support})
        plan, why = self._plan_place(side, held, sol)
        if plan is None:
            return SkillResult.fail("UNREACHABLE", f"no reachable spot {sol.message}: {why}")
        legs, down, up, centre = plan
        for step, fn, prog in (("transfer", lambda: self.move_legs(side, legs, should_abort), 0.2),
                               ("lowering", lambda: self.move_line(side, down, should_abort), 0.5),
                               ("releasing", lambda: self.set_hand(side, False, should_abort), 0.7)):
            feedback(step, prog)
            r = fn()
            if not r.success:
                return r
        t_end = time.monotonic() + self.g["verify_timeout"] / max(self.executor_.time_scale, self.min_time_scale)
        while self.weld(side) and time.monotonic() < t_end:
            time.sleep(0.05)
        if self.weld(side):
            return SkillResult.fail("GRASP_FAILED", f"opened the {side} hand but {held.id} is still attached")
        time.sleep(self.pl["drop_wait"])                       # let it fall/settle before the hand moves
        if sol.container_id:
            ref = self.find(sol.container_id)
            floor = (float(ref.pose.position.z) - 0.5 * float(ref.size.z) + 0.006) if ref else self.table["top_z"]
            rest = np.array([centre[0], centre[1], floor + 0.5 * held.size[2]])
        else:
            rest = np.array([centre[0], centre[1], sol.surface_z + 0.5 * held.size[2]])
        self.holding[side] = None
        self.set_held(held.id, "", rest, sol.container_id)
        feedback("retreating", 0.8)
        r = self.move_line(side, up, should_abort)
        if not r.success:
            return r
        r = self.skill_home(ExecuteSkill.Goal(), should_abort, feedback)
        if not r.success:
            return r
        seen = ""
        if self.detect_now() and not sol.container_id:
            near = [w for w in self.world() if not w.held_by and w.label == held.label and w.color == held.color
                    and np.linalg.norm(_xyz(w)[:2] - rest[:2]) < self.pl["confirm_radius"]]
            seen = "; camera sees it there" if near else "; NOT re-detected at the target"
        return SkillResult.ok(f"placed {held.id} {sol.message}{seen}", object_id=held.id,
                              pose=tuple(float(v) for v in rest))

    def _world_for_place(self, held: Held) -> list[WorldObject]:
        return [_wo(o) for o in self.world() if not o.held_by and o.id != held.id and not o.inside]

    def skill_place(self, goal, should_abort: Abort, feedback: Feedback, target: PlaceTarget | None = None) -> SkillResult:
        side, why = self._holding_side(goal.object_id)
        if side is None:
            return SkillResult.fail("NOT_HOLDING", why)
        held = self.holding[side]
        t = target or goal.place_target
        held_wo = WorldObject(held.id, held.label, np.zeros(3), held.size)
        sol = self.resolver.resolve(t.relation, t.reference_id, t.region, (t.x, t.y), held_wo, self._world_for_place(held))
        if not sol.ok:
            return SkillResult.fail(sol.reason_code, sol.message)
        # the support (on) / container (in) is approached from 10 cm above and released onto: not
        # an obstacle (its perceived top is only cm-accurate, the release is 1-2 cm above it)
        return self._place(side, sol, should_abort, feedback, t.reference_id if t.relation in ("on", "in") else "")

    def skill_place_in(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """place_in(container): object_id names the container; the held object is implicit."""
        return self.skill_place(ExecuteSkill.Goal(side=goal.side), should_abort, feedback,
                                PlaceTarget(relation="in", reference_id=goal.object_id))

    def skill_move_aside(self, goal, should_abort: Abort, feedback: Feedback) -> SkillResult:
        """Pick an object (unless already held) and put it on the nearest free spot on its arm's side."""
        if not any(h and h.id == goal.object_id for h in self.holding.values()):
            r = self.skill_pick(goal, should_abort, feedback)
            if not r.success:
                return r
        side, why = self._holding_side(goal.object_id)
        if side is None:
            return SkillResult.fail("NOT_HOLDING", why)
        held = self.holding[side]
        prefer = (self.table["place_x"][0] + 0.03, (1 if side == "left" else -1) * (self.table["place_y"][1] - 0.03))
        o = self.find(held.id)
        held_wo = WorldObject(held.id, held.label, _xyz(o) if o else np.zeros(3), held.size)
        sol = self.resolver.free_spot(held_wo, self._world_for_place(held), prefer)
        if not sol.ok:
            return SkillResult.fail(sol.reason_code, sol.message)
        return self._place(side, sol, should_abort, feedback)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SkillServerNode()
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
