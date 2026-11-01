"""Expressive motion: a declarative description (from the LLM or the library) -> validated waypoints.

No LLM-written code ever runs. A motion is data:

    {"intent": "greet",
     "keyframes": [{"t": 1.2, "joints": [{"name": "right_elbow_pitch_joint", "angle": -1.6}],
                    "right_hand": "open"}, ...],
     "repeat": {"from": 1, "to": 2, "count": 2},          # optional: replay a keyframe range
     "point_at": {"object": "blue_cube_1", "arm": "right"}}  # optional: reach above an object (IK)

This module is the safety layer, ROS-free and deterministic: joints are whitelisted, angles clamped
to the limits, keyframes and time bounded, speed limited by stretching time (the trajectory does
that), objects in a hand lock that arm and the waist, the path is dry-run through forward kinematics
and the collision checks, and every motion returns to where it started. A motion that fails comes
back as a list of errors the LLM can repair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

HAND_GROUPS = {"left_hand": "left", "right_hand": "right"}
PRESETS = ("open", "half", "fist", "point", "thumb_up")

MOTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "keyframes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "t": {"type": "number"},
                "joints": {"type": "array", "items": {
                    "type": "object", "properties": {"name": {"type": "string"}, "angle": {"type": "number"}},
                    "required": ["name", "angle"]}},
                "left_hand": {"type": "string", "enum": list(PRESETS)},
                "right_hand": {"type": "string", "enum": list(PRESETS)},
            },
            "required": ["t"]}},
        "repeat": {"type": "object", "properties": {"from": {"type": "integer"}, "to": {"type": "integer"},
                                                    "count": {"type": "integer"}},
                   "required": ["from", "to", "count"]},
        "point_at": {"type": "object", "properties": {"object": {"type": "string"},
                                                      "arm": {"type": "string", "enum": ["left", "right"]}},
                     "required": ["object"]},
    },
    "required": ["intent"],
}


def hand_poses(names: list[str], close: np.ndarray) -> dict[str, np.ndarray]:
    """Named hand poses (order `names`, one hand's 12 joints) from the fist of the hand mapping."""
    close = np.asarray(close, float)
    index = np.array(["index" in n for n in names])
    thumb = np.array(["thumb" in n for n in names])
    return {"open": np.zeros(len(names)), "half": 0.5 * close, "fist": close.copy(),
            "point": np.where(index, 0.0, close), "thumb_up": np.where(thumb, 0.0, close)}


class MotionError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Limits:
    max_keyframes: int = 12
    max_expanded: int = 30         # after repeats
    max_time: float = 10.0         # s of described motion
    max_repeat: int = 6
    min_step: float = 0.15         # s, shortest gap between keyframes
    hold_s: float = 1.0            # s to hold a point_at


@dataclass
class Prepared:
    """Ready to play: waypoints (full joint vectors, ending at the start pose) and minimum durations."""
    waypoints: list[np.ndarray]
    durations: list[float]
    groups: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class MotionValidator:
    names: list[str]                                # all controller joints
    groups: dict[str, np.ndarray]                   # group -> indices
    lower: np.ndarray
    upper: np.ndarray
    hands: dict[str, np.ndarray]                    # preset -> 12 joint angles (per hand; signs from the mapping)
    path_hit: Callable[[str, np.ndarray, np.ndarray], str]   # (side, qa, qb) -> what the arm hits, "" if clear
    plan_point: Callable[[str, str, np.ndarray], list[np.ndarray]] | None = None   # (arm, object, q) -> waypoints
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        self.index = {n: i for i, n in enumerate(self.names)}
        self.group_of = {i: g for g, idx in self.groups.items() for i in idx}

    def prepare(self, spec: dict, q_start: np.ndarray, holding: dict[str, str] | None = None) -> Prepared:
        """Validated plan from the current joints; raises MotionError with every problem found."""
        errors: list[str] = []
        warnings: list[str] = []
        holding = holding or {}
        if not isinstance(spec, dict):
            raise MotionError(["motion must be a JSON object"])
        frames = spec.get("keyframes") or []
        point = spec.get("point_at")
        if not isinstance(frames, list) or (not frames and not point):
            raise MotionError(["a motion needs keyframes or point_at"])
        if len(frames) > self.limits.max_keyframes:
            errors.append(f"{len(frames)} keyframes: at most {self.limits.max_keyframes}")
        locked = {g for side, oid in holding.items() if oid for g in (f"{side}_arm", f"{side}_hand", "waist")}

        q = np.array(q_start, float)
        waypoints: list[np.ndarray] = []
        times: list[float] = []
        touched: set[str] = set()
        last_t = 0.0
        for k, fr in enumerate(frames):
            tag = f"keyframe {k}"
            if not isinstance(fr, dict):
                errors.append(f"{tag}: not an object")
                continue
            t = fr.get("t")
            if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t < 0:
                errors.append(f"{tag}: t must be a time in seconds")
                continue
            if t < last_t + self.limits.min_step:
                errors.append(f"{tag}: t={t} must be at least {self.limits.min_step} s after the previous keyframe")
            last_t = max(last_t, t)
            for j in _joint_items(fr.get("joints")):
                name, angle = j
                i = self.index.get(name)
                if i is None:
                    errors.append(f"{tag}: unknown joint '{name}'")
                    continue
                if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not math.isfinite(angle):
                    errors.append(f"{tag}: {name} angle is not a number")
                    continue
                g = self.group_of.get(i, "")
                if g in locked:
                    errors.append(f"{tag}: {name}: {g} is locked, the robot is holding an object")
                    continue
                touched.add(g)
                clamped = float(np.clip(angle, self.lower[i], self.upper[i]))
                if abs(clamped - angle) > 1e-9:
                    warnings.append(f"{name} clamped {angle:.2f} -> {clamped:.2f}")
                q[i] = clamped
            for hand, side in HAND_GROUPS.items():
                preset = fr.get(f"{side}_hand")
                if preset is None:
                    continue
                if preset not in self.hands:
                    errors.append(f"{tag}: unknown hand preset '{preset}' (use {', '.join(self.hands)})")
                elif hand in locked:
                    errors.append(f"{tag}: {hand} is locked, the robot is holding an object")
                else:
                    q[self.groups[hand]] = self.hands[preset]
                    touched.add(hand)
            waypoints.append(q.copy())
            times.append(float(t))
        if last_t > self.limits.max_time:
            errors.append(f"motion lasts {last_t:.1f} s: at most {self.limits.max_time:.0f} s")

        rep = spec.get("repeat")
        if rep and not errors:
            waypoints, times, err = _expand(waypoints, times, rep, self.limits)
            errors += err
        if point:
            if not isinstance(point, dict) or not point.get("object"):
                errors.append("point_at needs an object id")
            elif self.plan_point is None:
                errors.append("point_at is not available")
            else:
                arm = point.get("arm") or "right"
                if arm not in ("left", "right"):
                    errors.append(f"point_at: unknown arm '{arm}'")
                elif f"{arm}_arm" in locked:
                    errors.append(f"point_at: the {arm} arm is holding an object")
                else:
                    base = waypoints[-1] if waypoints else np.array(q_start, float)
                    try:
                        way = self.plan_point(arm, str(point["object"]), base)
                    except ValueError as exc:
                        errors.append(f"point_at: {exc}")
                        way = []
                    if way:
                        t0 = times[-1] if times else 0.0
                        for n, w in enumerate(way):
                            w = np.array(w, float)
                            w[self.groups[f"{arm}_hand"]] = self.hands["point"]
                            waypoints.append(w)
                            times.append(t0 + n + 1.0)
                        waypoints.append(waypoints[-1].copy())          # hold it, so it reads as pointing
                        times.append(times[-1] + self.limits.hold_s)
                        touched |= {"waist", f"{arm}_arm", f"{arm}_hand"}
        if errors:
            raise MotionError(errors)
        if not touched:
            raise MotionError(["the motion does not move anything"])

        # back to where it started, the same way (a direct return can cut a corner through the table)
        back = [w.copy() for w in reversed(waypoints[:-1])] if len(waypoints) > 1 else []
        all_way = waypoints + _thin(back, np.array(q_start, float))
        q_prev = np.array(q_start, float)
        for w in all_way:
            for side in ("left", "right"):
                a = self.groups[f"{side}_arm"]
                moves = np.any(np.abs(w[a] - q_prev[a]) > 1e-6) or np.any(np.abs(w[self.groups["waist"]] - q_prev[self.groups["waist"]]) > 1e-6)
                if moves and (hit := self.path_hit(side, q_prev, w)):
                    errors.append(f"the {side} arm would hit the {hit.replace('_', ' ')}")
            if errors:
                raise MotionError(sorted(set(errors)))
            q_prev = w
        durations, prev = [], 0.0
        for t in times:
            durations.append(max(t - prev, self.limits.min_step))
            prev = t
        durations += [0.0] * (len(all_way) - len(durations))     # the return: as fast as the speed limit allows
        return Prepared(all_way, durations, sorted(touched), warnings)


def _joint_items(joints: object) -> list[tuple[str, object]]:
    """[{name, angle}] (what the schema asks for) or a {name: angle} map."""
    if isinstance(joints, dict):
        return list(joints.items())
    out = []
    for j in joints or []:
        if isinstance(j, dict):
            out.append((str(j.get("name", "")), j.get("angle")))
    return out


def _expand(waypoints: list, times: list, rep: object, lim: Limits) -> tuple[list, list, list[str]]:
    if not isinstance(rep, dict):
        return waypoints, times, ["repeat must be an object"]
    a, b, n = rep.get("from"), rep.get("to"), rep.get("count")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (a, b, n)) or not 0 <= a <= b < len(waypoints):
        return waypoints, times, [f"repeat range {a}..{b} is outside the {len(waypoints)} keyframes"]
    if not 1 <= n <= lim.max_repeat:
        return waypoints, times, [f"repeat count {n}: use 1 to {lim.max_repeat}"]
    if len(waypoints) + n * (b - a + 1) > lim.max_expanded:
        return waypoints, times, [f"more than {lim.max_expanded} keyframes after repeating"]
    gap = max((times[a] - times[a - 1]) if a > 0 else times[a], lim.min_step)
    out_w, out_t = list(waypoints[:b + 1]), list(times[:b + 1])
    for _ in range(n):
        offset = out_t[-1] + gap - times[a]
        for i in range(a, b + 1):
            out_w.append(waypoints[i].copy())
            out_t.append(times[i] + offset)
    offset = out_t[-1] - times[b]
    for i in range(b + 1, len(waypoints)):
        out_w.append(waypoints[i].copy())
        out_t.append(times[i] + offset)
    if out_t[-1] > lim.max_time:
        return waypoints, times, [f"motion lasts {out_t[-1]:.1f} s after repeating: at most {lim.max_time:.0f} s"]
    return out_w, out_t, []


def _thin(back: list, start: np.ndarray) -> list:
    """The reversed path without consecutive duplicates, ending at the start pose."""
    out: list = []
    for w in back + [start]:
        if not out or np.any(np.abs(w - out[-1]) > 1e-6):
            out.append(w)
    return out


def seed_motions(wave_pose: list[float], wave_amplitude: float, nod_amplitude: float, shake_amplitude: float) -> dict[str, dict]:
    """The hand-written gestures as library entries: the offline fallback (right arm; neck from 0.25)."""
    arm = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint", "right_elbow_pitch_joint",
           "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]
    up = [{"name": n, "angle": a} for n, a in zip(arm, wave_pose)]
    elbow_yaw = wave_pose[4]
    return {
        "wave": {"intent": "wave", "keyframes": [
            {"t": 2.0, "joints": up, "right_hand": "open"},
            {"t": 2.8, "joints": [{"name": "right_elbow_yaw_joint", "angle": elbow_yaw + wave_amplitude}]},
            {"t": 3.6, "joints": [{"name": "right_elbow_yaw_joint", "angle": elbow_yaw - wave_amplitude}]}],
            "repeat": {"from": 1, "to": 2, "count": 1}},
        "nod": {"intent": "nod", "keyframes": [
            {"t": 0.5, "joints": [{"name": "neck_pitch_joint", "angle": 0.5}]},
            {"t": 1.0, "joints": [{"name": "neck_pitch_joint", "angle": 0.5 - 2 * nod_amplitude}]}],
            "repeat": {"from": 0, "to": 1, "count": 1}},
        "shake_head": {"intent": "shake_head", "keyframes": [
            {"t": 0.5, "joints": [{"name": "neck_yaw_joint", "angle": shake_amplitude}]},
            {"t": 1.0, "joints": [{"name": "neck_yaw_joint", "angle": -shake_amplitude}]}],
            "repeat": {"from": 0, "to": 1, "count": 1}},
    }
