"""Prompt for the LLM planner: instructions, a compact world state, and few-shot examples.

Kept short on purpose: the planner runs a ~3B model on a laptop CPU, where every prompt token
costs time-to-first-motion (prompt evaluation dominates for short JSON answers).
"""
from __future__ import annotations

import json

from ..geometry.place import REGION_NAMES
from .plan import PLACE_RELATIONS, World

SYSTEM = f"""You plan actions for a robot with two arms at a table. Reply with ONE JSON object:
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


def build_messages(request: str, world: World, history: list[str], feedback: str = "", camera: str = "") -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for user, answer in EXAMPLES:
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
