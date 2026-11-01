"""Terminal chat with the planner (ros2 run aibot_ai chat): the GUI chat panel without a webcam.

Type a task and press Enter; replies from /aibot/chat/robot are printed as they arrive. 'stop'
calls /aibot/stop (same as the GUI STOP button), 'ai' / 'mimic' switch the mode, Ctrl-D quits.
"""
from __future__ import annotations

import sys
import threading

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aibot_msgs.srv import SetMode


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Node("aibot_chat")
    pub = node.create_publisher(String, "/aibot/chat/user", 10)
    node.create_subscription(String, "/aibot/chat/robot", lambda m: print(f"\rrobot> {m.data}\nyou> ", end="", flush=True), 20)
    stop = node.create_client(Trigger, "/aibot/stop")
    set_mode = node.create_client(SetMode, "/aibot/set_mode")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    print("Chat with the robot (Physical AI mode). Commands: stop | ai | mimic | Ctrl-D to quit.")
    try:
        while True:
            line = input("you> ").strip()
            if line == "stop":
                stop.call_async(Trigger.Request())
            elif line in ("ai", "mimic"):
                set_mode.call_async(SetMode.Request(mode=line))
            elif line:
                pub.publish(String(data=line))
    except (EOFError, KeyboardInterrupt, ExternalShutdownException):
        print()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
