"""Structured skill outcome shared by every skill, the action result and the planner."""
from __future__ import annotations

from dataclasses import dataclass, field

REASON_CODES = ("OK", "NOT_FOUND", "AMBIGUOUS", "UNREACHABLE", "TOO_LARGE", "GRASP_FAILED", "PLACE_BLOCKED",
                "NOT_HOLDING", "TIMEOUT", "CANCELLED", "NOT_AI_MODE", "INVALID", "BUSY", "ABORTED")


@dataclass
class SkillResult:
    success: bool
    reason_code: str = "OK"
    message: str = ""
    object_id: str = ""
    pose: tuple[float, ...] = field(default_factory=tuple)   # x y z [qx qy qz qw]

    def __post_init__(self) -> None:
        if self.reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason code {self.reason_code}")

    @classmethod
    def ok(cls, message: str = "", **kw) -> "SkillResult":
        return cls(True, "OK", message, **kw)

    @classmethod
    def fail(cls, reason_code: str, message: str, **kw) -> "SkillResult":
        return cls(False, reason_code, message, **kw)

    def to_msg(self, msg):
        msg.success, msg.reason_code, msg.message, msg.object_id = self.success, self.reason_code, self.message, self.object_id
        if self.pose:
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.pose[:3]
            if len(self.pose) == 7:
                o = msg.pose.orientation
                o.x, o.y, o.z, o.w = self.pose[3:]
        return msg

    def as_dict(self) -> dict:
        return {"success": self.success, "reason_code": self.reason_code, "message": self.message,
                "object_id": self.object_id}
