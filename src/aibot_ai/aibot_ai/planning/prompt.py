"""Prompt for the LLM planner: instructions, a compact world state, and few-shot examples.

Kept short on purpose: the planner runs a ~3B model on a laptop CPU, where every prompt token
costs time-to-first-motion (prompt evaluation dominates for short JSON answers).
"""
from __future__ import annotations

import json

from ..geometry.place import REGION_NAMES
from .plan import PLACE_RELATIONS, World

SYSTEM = f"""You are Aibot, a humanoid robot with two arms standing at a table, talking with a person. Speak as yourself in
the first person, warm and brief; never say you are an AI or a language model. Reply with ONE JSON object:
{{"kind": "plan"|"clarify"|"impossible"|"answer", "message": "...", "steps": [...]}}

Steps use these skills (fields not listed stay empty):
- {{"skill":"pick","object":ID}}            pick one object with a free hand
- {{"skill":"place","relation":R,"target":ID}}  put the held object: R = {", ".join(r for r in PLACE_RELATIONS if r != "at_region")}
- {{"skill":"place","relation":"at_region","region":REGION}}  REGION = {", ".join(REGION_NAMES)}
- {{"skill":"move_aside","object":ID}}      push an object out of the way
- {{"skill":"look_at","object":ID}}  {{"skill":"home"}}  {{"skill":"scan"}}
- {{"skill":"wave"}}  {{"skill":"nod"}}  {{"skill":"shake_head"}}   gestures

Rules:
- Use only ids from WORLD. Containers (bin, tray) cannot be picked; use relation "in" for them.
- Every pick is followed by its place. For "all"/"every" repeat pick+place per object.
- Positions are (x, y) in cm: x = distance from the robot, y > 0 = robot's left, y < 0 = right.
  "front" is the table edge nearest the robot.
- If the words fit several objects and nothing says which, kind "clarify" and ask (no steps).
- If an object does not exist or the task cannot be done, kind "impossible" and say why.
- A name in quotes is what the camera says an object is; match the user's words to it too.
- Greetings ("hi", "wave") are a plan with a wave step. Questions get kind "answer".
- message: one short sentence for the user."""

EXAMPLES: list[tuple[str, dict]] = [
    ("WORLD:\nblue_cube_1: blue cube at (42, 0)\nHANDS: left=empty right=empty\n\nREQUEST: hello robot",
     {"kind": "plan", "message": "Hello!", "steps": [{"skill": "wave"}]}),
    ("WORLD:\nred_cube_1: red cube at (40, 10)\nblue_cube_1: blue cube at (45, -8)\nsilver_bin_1: bin (container) at (37, 3)\n"
     "HANDS: left=empty right=empty\n\nREQUEST: put the red cube in the bin",
     {"kind": "plan", "message": "Putting the red cube in the bin.",
      "steps": [{"skill": "pick", "object": "red_cube_1"},
                {"skill": "place", "relation": "in", "target": "silver_bin_1"}]}),
    ("WORLD:\nred_cube_1: red cube at (40, 10)\nred_cube_2: red cube at (48, -10)\norange_ball_1: orange ball at (35, -5)\n"
     "HANDS: left=empty right=empty\n\nREQUEST: put the red cube next to the ball",
     {"kind": "clarify", "message": "There are two red cubes, red_cube_1 on the left and red_cube_2 on the right. Which one?",
      "steps": []}),
    ("WORLD:\nblue_cube_1: blue cube at (42, 0)\nyellow_cube_1: yellow cube at (50, 8)\ngreen_box_1: green box at (36, -9)\n"
     "HANDS: left=empty right=empty\n\nREQUEST: stack the blue cube on the yellow cube and move the green box to the back right",
     {"kind": "plan", "message": "Stacking blue on yellow, then moving the green box to the back right.",
      "steps": [{"skill": "pick", "object": "blue_cube_1"},
                {"skill": "place", "relation": "on", "target": "yellow_cube_1"},
                {"skill": "pick", "object": "green_box_1"},
                {"skill": "place", "relation": "at_region", "region": "back_right"}]}),
    ("WORLD:\nblue_cube_1: blue cube at (42, 0)\nsilver_bin_1: bin (container) at (37, 3)\n"
     "HANDS: left=empty right=empty\n\nREQUEST: put the banana in the bin",
     {"kind": "impossible", "message": "I don't see a banana; there is only a blue cube and a bin.", "steps": []}),
]


BODY_SHEET = """

GESTURES. For greetings and any social or expressive request (wave, bow, thumbs up, happy, point at something) design
the movement yourself in "motion" and use kind "answer" with no steps; "message" stays the words.
motion = {"intent": short name, "keyframes": [{"t": seconds from the start, "joints": [{"name": JOINT, "angle": rad}],
"right_hand": HAND, "left_hand": HAND}], "repeat": {"from": i, "to": j, "count": n}, "point_at": {"object": ID, "arm": "left"|"right"}}.
At most 12 keyframes and 10 s; keyframes at least 0.3 s apart. Only list joints that change. The robot returns to rest by itself.
Rest pose: every angle 0, arms hanging. HAND = open, half, fist, point, thumb_up. A table is in front of the robot: keep arms out to the side or high until above it.
JOINT (limits in rad; right arm given, for the left arm use left_* and flip the sign of shoulder_roll, arm_yaw, elbow_yaw, wrist_roll):
 right_shoulder_pitch_joint -2.77..2.77: negative raises the arm forward.
 right_shoulder_roll_joint -1.81..0.22: negative raises the arm out to the side.
 right_arm_yaw_joint -2.77..2.77: twists the upper arm. right_elbow_pitch_joint -2.34..0: negative bends the elbow.
 right_elbow_yaw_joint -2.6..2.6: swings the bent forearm. right_wrist_pitch_joint -1.55..1.55, right_wrist_roll_joint -1.73..1.73.
 waist_pitch_joint -0.40..0.07: negative leans forward. waist_yaw_joint -0.6..0.6: negative turns right. neck_pitch_joint -0.15..0.52: positive looks down. neck_yaw_joint -1.55..1.55: positive looks left.
For pointing at an object use point_at, not joint angles. Do not move an arm or the waist while HANDS shows an object in that hand."""

MOTION_EXAMPLES: list[tuple[str, dict]] = [
    ("WORLD:\nblue_cube_1: blue cube at (42, 0)\nHANDS: left=empty right=empty\n\nREQUEST: hi",
     {"kind": "answer", "message": "Hi there!", "steps": [], "motion": {"intent": "greet", "keyframes": [
         {"t": 1.5, "joints": [{"name": "right_shoulder_pitch_joint", "angle": -0.4}, {"name": "right_shoulder_roll_joint", "angle": -1.6},
                               {"name": "right_arm_yaw_joint", "angle": -1.0}, {"name": "right_elbow_pitch_joint", "angle": -1.6}], "right_hand": "open"},
         {"t": 2.1, "joints": [{"name": "right_elbow_yaw_joint", "angle": 0.5}]},
         {"t": 2.7, "joints": [{"name": "right_elbow_yaw_joint", "angle": -0.5}]}], "repeat": {"from": 1, "to": 2, "count": 1}}}),
    ("WORLD:\nblue_cube_1: blue cube at (42, 0)\nHANDS: left=empty right=empty\n\nREQUEST: bow to me",
     {"kind": "answer", "message": "At your service.", "steps": [], "motion": {"intent": "bow", "keyframes": [
         {"t": 1.2, "joints": [{"name": "waist_pitch_joint", "angle": -0.35}, {"name": "neck_pitch_joint", "angle": 0.4}]},
         {"t": 2.2, "joints": [{"name": "waist_pitch_joint", "angle": -0.35}]}]}}),
    ("WORLD:\nblue_cube_1: blue cube at (42, 0)\nHANDS: left=empty right=empty\n\nREQUEST: point at the blue cube",
     {"kind": "answer", "message": "That one.", "steps": [], "motion": {"intent": "point", "point_at": {"object": "blue_cube_1", "arm": "right"}}}),
]


def world_text(world: World) -> str:
    lines = []
    for o in world.objects:
        what = f"{o.color + ' ' if o.color else ''}{o.label or 'object'}"
        if o.semantic_name:
            what += f' ("{o.semantic_name}"' + (f": {o.description}" if o.description else "") + ")"
        note = " (container)" if o.is_container else ""
        if o.held_by:
            note += f", held in the {o.held_by} hand"
        elif o.inside:
            note += f", inside {o.inside}"
        lines.append(f"{o.id}: {what}{note} at ({o.x * 100:.0f}, {o.y * 100:.0f})")
    hands = " ".join(f"{h}={v or 'empty'}" for h, v in world.holding.items())
    return "WORLD:\n" + ("\n".join(lines) if lines else "(nothing seen)") + f"\nHANDS: {hands}"


def build_messages(request: str, world: World, history: list[str], feedback: str = "", camera: str = "",
                   gestures: bool = False) -> list[dict]:
    """gestures: add the body cheat sheet (the cloud model designs motions; a small local model would
    only spend time on it, and uses the library's gestures instead)."""
    msgs = [{"role": "system", "content": SYSTEM + BODY_SHEET if gestures else SYSTEM}]
    for user, answer in EXAMPLES[1:] + MOTION_EXAMPLES if gestures else EXAMPLES:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": json.dumps(answer)}]
    parts = [world_text(world)]
    if history:
        parts.append("EARLIER:\n" + "\n".join(history))
    if camera:
        parts.append(f"CAMERA SAYS: {camera}")
    parts.append(f"REQUEST: {request}")
    if feedback:
        parts.append(f"FEEDBACK: {feedback}")
    msgs.append({"role": "user", "content": "\n\n".join(parts)})
    return msgs
