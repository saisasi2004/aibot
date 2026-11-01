#!/usr/bin/env python3
"""Phase 4 acceptance, in the sim with the real LLM: expressive requests produce valid, visibly
different motions; the repeat of one is a cache hit. Logs joint trajectories to CSV.

    ros2 launch aibot_bringup teleop.launch.py mode:=ai webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false
    python3 scripts/check_gestures.py --csv /tmp/gestures.csv

The library is cleared first, so generation is measured. Per request: reply, where the motion came
from, validator rejections, joints that moved most, and whether the robot came back to rest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

REQUESTS = ["hi", "wave goodbye", "bow", "point at the blue cube", "thumbs up", "show me you're happy", "hi"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("requests", nargs="*", default=REQUESTS)
    ap.add_argument("--csv", default="/tmp/gestures.csv")
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("gesture_check")
    js: dict[str, float] = {}
    reports: list[dict] = []
    replies: list[str] = []

    def on_status(m: String) -> None:
        st = json.loads(m.data)
        if st.get("state") == "idle" and st.get("report"):
            reports.append(st["report"])

    n.create_subscription(JointState, "/joint_states", lambda m: js.update(zip(m.name, m.position)), 20)
    n.create_subscription(String, "/aibot/planner/status", on_status, 20)
    n.create_subscription(String, "/aibot/chat/robot", lambda m: replies.append(m.data), 20)
    pub = n.create_publisher(String, "/aibot/chat/user", 10)
    clear = n.create_client(Trigger, "/aibot/planner/clear_gestures")
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    t0 = time.monotonic()
    while (not js or pub.get_subscription_count() == 0) and time.monotonic() - t0 < 60:
        time.sleep(0.2)
    if not args.keep_cache and clear.wait_for_service(timeout_sec=10):
        fut = clear.call_async(Trigger.Request())
        while not fut.done():
            time.sleep(0.05)
        print(fut.result().message)
    time.sleep(2.0)
    rows, f = [], open(args.csv, "w", newline="")
    log = csv.writer(f)
    log.writerow(["request", "t", *sorted(js)])
    played = rejected = 0
    signatures: dict[str, tuple] = {}
    for text in args.requests:
        start = dict(js)
        k, r0, ts = len(reports), len(replies), time.monotonic()
        peak = dict.fromkeys(start, 0.0)
        pub.publish(String(data=text))
        while len(reports) == k and time.monotonic() - ts < args.timeout:
            now = dict(js)
            for j in start:
                peak[j] = max(peak[j], abs(now[j] - start[j]))
            log.writerow([text, f"{time.monotonic() - ts:.2f}", *[f"{now[j]:.4f}" for j in sorted(start)]])
            time.sleep(0.05)
        time.sleep(1.0)
        rep = reports[-1] if len(reports) > k else {}
        moved = sorted(((v, j) for j, v in peak.items() if "hand" not in j and v > 0.05), reverse=True)[:4]
        end = max(abs(js[j] - start[j]) for j in start if "hand" not in j)
        rejected += len(rep.get("motion_errors", []))
        played += sum(1 for e in rep.get("executed", []) if e[0].startswith("express"))
        signatures[text] = tuple(j for _v, j in moved)
        rows.append((text, rep.get("motion_source"), rep.get("backend"), moved, end, rep))
        print(f"{text!r}: source={rep.get('motion_source')} model=[{rep.get('backend')}] executed={rep.get('executed')} "
              f"{time.monotonic() - ts:.1f} s\n   reply {replies[r0:]}\n   moved {[(j.replace('_joint', ''), round(v, 2)) for v, j in moved]}; "
              f"back to within {end:.3f} rad; rejections {rep.get('motion_errors', [])}", flush=True)
    f.close()
    gen = [r for r in rows if r[1] == "generated"]
    distinct = len({r[0] and signatures[r[0]] for r in rows})
    print(f"\ngenerated {len(gen)}, cached {sum(r[1] == 'cache' for r in rows)}, seed fallback {sum(r[1] == 'seed' for r in rows)}, "
          f"distinct joint patterns {distinct}; express attempts {played}, validator rejections {rejected} "
          f"({rejected / max(played, 1):.0%}); csv {args.csv}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
