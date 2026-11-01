#!/usr/bin/env python3
"""Sim check: a greeting produces a reply AND visible motion, within a time bound, and ends at rest.

    ros2 launch aibot_bringup teleop.launch.py mode:=ai webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false
    python3 scripts/check_greeting.py "HI" "Say Hi" "wave hi" "how are you?"

Per message: reply text, planner kind and skills, time to the reply and to the first visible motion
(any arm/head joint more than --min-motion rad from where it started), the largest joint change,
and the distance from the start pose at the end. Pass: every message moved visibly and came back.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("messages", nargs="*", default=["HI"])
    ap.add_argument("--min-motion", type=float, default=0.2, help="rad that counts as visible")
    ap.add_argument("--rest-tolerance", type=float, default=0.05, help="rad from the start pose at the end")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("greeting_check")
    js: dict[str, float] = {}
    replies: list[tuple[float, str]] = []
    reports: list[dict] = []

    def on_status(m: String) -> None:
        st = json.loads(m.data)
        if st.get("state") == "idle" and st.get("report"):
            reports.append(st["report"])

    n.create_subscription(JointState, "/joint_states", lambda m: js.update(zip(m.name, m.position)), 20)
    n.create_subscription(String, "/aibot/chat/robot", lambda m: replies.append((time.monotonic(), m.data)), 20)
    n.create_subscription(String, "/aibot/planner/status", on_status, 20)
    pub = n.create_publisher(String, "/aibot/chat/user", 10)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    t0 = time.monotonic()
    while (not js or pub.get_subscription_count() == 0) and time.monotonic() - t0 < 60:
        time.sleep(0.2)
    time.sleep(2.0)
    passed = 0
    for text in args.messages:
        start = {k: v for k, v in js.items() if "hand" not in k}
        k, r0, ts = len(reports), len(replies), time.monotonic()
        first, peak = None, 0.0
        pub.publish(String(data=text))
        while len(reports) == k and time.monotonic() - ts < args.timeout:
            d = max(abs(js[j] - start[j]) for j in start)
            peak = max(peak, d)
            if first is None and d > args.min_motion:
                first = time.monotonic() - ts
            time.sleep(0.02)
        time.sleep(1.0)
        rep = reports[-1] if len(reports) > k else {}
        end = max(abs(js[j] - start[j]) for j in start)
        said = replies[r0:]
        ok = first is not None and end <= args.rest_tolerance and bool(said)
        passed += ok
        print(f"{'OK ' if ok else 'BAD'} {text!r}: kind={rep.get('kind')} skills={[e[0] for e in rep.get('executed', [])]} "
              f"results={[e[1] for e in rep.get('executed', [])]}\n    reply {said[0][0] - ts if said else float('nan'):.1f} s: "
              f"{[t for _, t in said]}\n    first visible motion {first and round(first, 1)} s, peak {peak:.2f} rad, "
              f"back to within {end:.3f} rad, total {time.monotonic() - ts:.1f} s", flush=True)
    print(f"{passed}/{len(args.messages)} passed")
    print("PASS" if passed == len(args.messages) else "FAIL")
    os._exit(0 if passed == len(args.messages) else 1)


if __name__ == "__main__":
    main()
