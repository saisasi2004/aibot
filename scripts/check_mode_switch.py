#!/usr/bin/env python3
"""Phase-1 acceptance check: repeated mimic <-> Physical AI switches without jumps or stuck welds.

Run against a headless bringup WITHOUT the real webcam (this script plays the operator):

    ros2 launch aibot_bringup teleop.launch.py webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false calibrate:=false
    python3 scripts/check_mode_switch.py --cycles 10 --log /tmp/mode_switch.csv

Per cycle: switch to AI -> skill home -> move the left hand onto cube_small and close it (the
grasp manager welds it) -> lift -> switch to mimic -> expect the weld released and the robot to
glide back to the synthetic operator pose.

Measured: the largest change between consecutive commands of any controller group in a window
around every switch (budget: 0.05 rad), switch durations, and held objects after each return to
mimic. Every command message is logged to CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from aibot_msgs.action import ExecuteSkill
from aibot_msgs.msg import BodyLandmarks, GraspStatus, HandLandmarks
from aibot_msgs.srv import SetMode
from aibot_teleop.utils.mode_gate import MODE_TOPIC, mode_qos

CONTROLLERS = ["waist_controller", "head_controller", "left_arm_controller", "right_arm_controller",
               "left_hand_controller", "right_hand_controller"]
MP_TO_BODY = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# Left-arm poses that bring the grasp point onto cube_small at (0.28, 0.10, 1.005), solved with the
# URDF kinematics; "side" lifts the hand beside the table first so it does not sweep the cube.
LEFT_ARM = {
    "side": [1.096, -0.221, 0.754, -2.34, -1.087, -0.855, 0.327],
    "pre": [0.323, -0.221, -0.628, -1.898, -2.031, 0.491, -0.193],
    "grasp": [0.273, -0.213, -0.531, -1.465, -1.756, 0.541, 0.269],
    "lift": [0.172, -0.119, -0.821, -2.037, -2.14, 0.638, -0.538],
}
LEFT_ARM_JOINTS = [f"left_{j}" for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "arm_yaw_joint",
                                         "elbow_pitch_joint", "elbow_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint")]


def operator_pose(t: float) -> np.ndarray:
    """33 BlazePose points (body axes): left arm swinging slowly forward, right arm hanging."""
    p = np.zeros((33, 3))
    p[23], p[24] = [0, 0.10, 0], [0, -0.10, 0]
    sh_l, sh_r = np.array([0, 0.18, 0.45]), np.array([0, -0.18, 0.45])
    p[11], p[12] = sh_l, sh_r
    p[0], p[7], p[8] = [0.10, 0, 0.62], [0.0, 0.07, 0.64], [0.0, -0.07, 0.64]
    a = 0.6 + 0.3 * math.sin(2 * math.pi * 0.15 * t)          # shoulder flexion 0.3..0.9 rad
    d = np.array([math.sin(a), 0.0, -math.cos(a)])
    p[13], p[15] = sh_l + 0.28 * d, sh_l + 0.54 * d
    p[14], p[16] = sh_r + [0, 0, -0.28], sh_r + [0, 0, -0.54]
    return p


class Checker(Node):
    def __init__(self, log_path: str) -> None:
        super().__init__("mode_switch_checker")
        self.lock = threading.Lock()
        self.last_cmd: dict[str, np.ndarray] = {}
        self.events: list[tuple[float, str, float]] = []          # (t, controller, max |delta|)
        self.mode = None
        self.mode_log: list[tuple[float, str]] = []
        self.held = ("", "")
        self.js: dict[str, float] = {}
        self.csv = csv.writer(open(log_path, "w", newline=""))
        self.csv.writerow(["t", "controller", "max_abs_delta", "mode"])
        for c in CONTROLLERS:
            self.create_subscription(Float64MultiArray, f"/{c}/commands", lambda m, c=c: self._on_cmd(c, m), 50)
        self.create_subscription(String, MODE_TOPIC, self._on_mode, mode_qos())
        self.create_subscription(GraspStatus, "/aibot/grasp/status", lambda m: setattr(self, "held", (m.left_object, m.right_object)), 10)
        self.create_subscription(JointState, "/joint_states", lambda m: self.js.update(zip(m.name, m.position)), 10)
        self.body_pub = self.create_publisher(BodyLandmarks, "/teleop/body_landmarks", 5)
        self.hand_pubs = {s: self.create_publisher(HandLandmarks, f"/teleop/hand_landmarks/{s}", 5) for s in ("left", "right")}
        self.arm_pub = self.create_publisher(Float64MultiArray, "/left_arm_controller/commands", 5)
        self.set_mode = self.create_client(SetMode, "/aibot/set_mode")
        self.skills = ActionClient(self, ExecuteSkill, "/aibot/skills/execute")
        self.t0 = time.monotonic()
        self.create_timer(1 / 30, self._publish_operator)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def _on_cmd(self, c: str, m: Float64MultiArray) -> None:
        q = np.array(m.data)
        with self.lock:
            prev = self.last_cmd.get(c)
            self.last_cmd[c] = q
            if prev is not None and prev.shape == q.shape:
                jump = float(np.max(np.abs(q - prev)))
                self.events.append((self.now(), c, jump))
                self.csv.writerow([f"{self.now():.3f}", c, f"{jump:.5f}", self.mode])

    def _on_mode(self, m: String) -> None:
        self.mode = m.data
        self.mode_log.append((self.now(), m.data))

    def _publish_operator(self) -> None:
        stamp = self.get_clock().now().to_msg()
        body = BodyLandmarks(detected=True, image_width=640, image_height=480)
        body.header.stamp = stamp
        mp = operator_pose(self.now()) @ MP_TO_BODY
        body.world_points = [Point(x=float(a), y=float(b), z=float(c)) for a, b, c in mp]
        body.image_points = [Point() for _ in range(33)]
        body.visibility = body.presence = [1.0] * 33
        self.body_pub.publish(body)
        for s, pub in self.hand_pubs.items():
            h = HandLandmarks(side=s, detected=False)
            h.header.stamp = stamp
            pub.publish(h)

    # ------------------------------------------------------------------ actions
    def call_mode(self, mode: str) -> tuple[bool, str, float]:
        t = time.monotonic()
        fut = self.set_mode.call_async(SetMode.Request(mode=mode))
        while not fut.done():
            time.sleep(0.01)
        r = fut.result()
        return r.success, r.message, time.monotonic() - t

    def skill(self, name: str, side: str = "") -> ExecuteSkill.Result:
        goal = ExecuteSkill.Goal(skill=name, side=side)
        gh_f = self.skills.send_goal_async(goal)
        while not gh_f.done():
            time.sleep(0.01)
        res_f = gh_f.result().get_result_async()
        while not res_f.done():
            time.sleep(0.01)
        return res_f.result().result

    def ramp_left_arm(self, target: list[float], seconds: float) -> None:
        """Stand-in for a not-yet-written arm skill: smooth ramp from the ACTUAL pose, 30 Hz."""
        q0 = np.array([self.js[n] for n in LEFT_ARM_JOINTS])
        q1 = np.array(target)
        t0 = time.monotonic()
        while (s := (time.monotonic() - t0) / seconds) < 1.0:
            w = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.arm_pub.publish(Float64MultiArray(data=list(q0 + w * (q1 - q0))))
            time.sleep(1 / 30)
        self.arm_pub.publish(Float64MultiArray(data=list(q1)))
        time.sleep(0.5)

    def max_jump(self, t_from: float, t_to: float) -> tuple[float, str]:
        with self.lock:
            ev = [(j, c) for t, c, j in self.events if t_from <= t <= t_to]
        return max(ev) if ev else (0.0, "-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--log", default="/tmp/mode_switch.csv")
    args = ap.parse_args()
    rclpy.init()
    n = Checker(args.log)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    print("waiting for mode manager, skill server and joint states ...")
    n.set_mode.wait_for_service(timeout_sec=60)
    n.skills.wait_for_server(timeout_sec=60)
    t_wait = time.monotonic()
    while (n.mode != "mimic" or not n.js) and time.monotonic() - t_wait < 60:
        time.sleep(0.2)
    time.sleep(8.0)            # let mimic track the synthetic operator first
    rows, welds, stuck = [], 0, 0
    for i in range(1, args.cycles + 1):
        # switch window = from the request until 1 s (to AI) / 2 s (to mimic, covers the glide)
        # after the new mode became active; skills/glides inside it are included on purpose.
        t_a = n.now()
        ok_a, msg_a, d_a = n.call_mode("ai")
        home = n.skill("home") if ok_a else None
        jump_to_ai = n.max_jump(t_a, t_a + d_a + 1.0)
        held_ai, close = "", None
        if ok_a:
            n.ramp_left_arm(LEFT_ARM["side"], 2.0)
            n.ramp_left_arm(LEFT_ARM["pre"], 1.5)
            n.ramp_left_arm(LEFT_ARM["grasp"], 1.5)
            close = n.skill("close_hand", "left")
            time.sleep(1.0)
            held_ai = n.held[0]
            welds += bool(held_ai)
            n.ramp_left_arm(LEFT_ARM["lift"], 1.5)
        t_m = n.now()
        ok_m, msg_m, d_m = n.call_mode("mimic")
        time.sleep(1.0)
        held_after = n.held
        stuck += bool(held_after[0] or held_after[1])
        time.sleep(6.0)                       # glide back and track
        jump_to_mimic = n.max_jump(t_m, t_m + d_m + 2.0)
        home_code = home.reason_code if home else "skipped"
        close_code = close.reason_code if close else "skipped"
        rows.append((i, ok_a, d_a, home_code, close_code, held_ai or "-", ok_m, d_m,
                     held_after, jump_to_ai, jump_to_mimic))
        print(f"cycle {i:2d}: ->ai {('ok' if ok_a else 'FAIL'):4s} {d_a:4.2f}s home {home_code:11s} "
              f"close {close_code:4s} held '{held_ai or '-'}' | ->mimic {('ok' if ok_m else 'FAIL'):4s} {d_m:4.2f}s "
              f"held after {held_after} | max step around switch: to-AI {jump_to_ai[0]:.3f} ({jump_to_ai[1]}), "
              f"to-mimic {jump_to_mimic[0]:.3f} ({jump_to_mimic[1]})")
        if not (ok_a and ok_m):
            print("   messages:", msg_a, "|", msg_m)
    worst = max(max(r[9][0], r[10][0]) for r in rows)
    print(f"\nSUMMARY: {len(rows)} cycles, switches ok {sum(r[1] and r[6] for r in rows)}/{len(rows)}, "
          f"welds made in AI {welds}, objects still welded after return to mimic {stuck}, "
          f"worst consecutive-command step {worst:.4f} rad (budget 0.05)")
    print("PASS" if worst <= 0.05 and stuck == 0 and all(r[1] and r[6] for r in rows) else "FAIL")
    print(f"log: {args.log}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
