"""Defence in depth for the two operating modes: never command the robot outside your own mode.

mode_manager_node already pauses the inactive source through its services. Every commanding
node additionally watches /aibot/mode (transient-local, so a late joiner gets the current value)
and refuses to publish unless the mode is exactly its own. During "switching", "hold" or
"boot" nobody publishes, so a lost service call can never leave two sources active.

`required=False` keeps standalone use working (e.g. retarget_node without the mode manager):
until the first /aibot/mode message arrives, the gate is open. The bringup sets required=True.
"""
from __future__ import annotations

from typing import Callable

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

MODE_TOPIC = "/aibot/mode"


def mode_qos() -> QoSProfile:
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class ModeGate:
    def __init__(self, node: Node, own_mode: str, required: bool,
                 on_change: Callable[[str | None, str], None] | None = None) -> None:
        self.own_mode, self.required = own_mode, required
        self.mode: str | None = None
        self._on_change = on_change
        node.create_subscription(String, MODE_TOPIC, self._on_mode, mode_qos())

    def _on_mode(self, msg: String) -> None:
        old, self.mode = self.mode, msg.data
        if old != msg.data and self._on_change is not None:
            self._on_change(old, msg.data)

    @property
    def active(self) -> bool:
        if self.mode is None:
            return not self.required
        return self.mode == self.own_mode
