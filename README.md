# aibot: webcam teleoperation and Physical AI for the RobotEra STAR1 humanoid

Two modes, switchable at runtime from the camera window (or `mode:=mimic|ai`):

- **Mimic** (default): move your arms, hands, torso and head in front of a laptop webcam and the
  STAR1 upper body copies you in Gazebo. Close your hand near an object to pick it up.
- **Physical AI:** type a task ("pick the red cube and put it in the bin"). The robot looks at
  the table with its head RGB-D camera, a local LLM (Ollama) plans it as a sequence of skills,
  and deterministic skill code does the perception-based picking and placing.

Stack: Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic (gz-sim 8) · ros2_control + gz_ros2_control ·
MediaPipe 0.10.21 (Pose + Hand landmarkers) · Python 3.12 · PyQt5.

```
webcam ─► webcam_node ──landmarks──► retarget_node ──Float64MultiArray──► 6 position controllers ─► Gazebo
           (MediaPipe,               (body frame, arm angles / IK,              (gz_ros2_control)
            camera GUI)               hands, torso/head, filters)
                                            │ GraspState
                                            ▼
                                     grasp_manager_node ──attach/detach──► DetachableJoint (gz)
```

## Quick start (mimic)

```bash
cd ~/aibot
./scripts/fetch_star1.sh          # STAR1 model -> src/aibot_description (temp clone, deleted afterwards)
./scripts/setup_env.sh            # apt + rosdep + .venv with MediaPipe + model download (asks for sudo once)
source /opt/ros/jazzy/setup.bash && source .venv/bin/activate
python3 -m colcon build --symlink-install
source install/setup.bash
ros2 launch aibot_bringup teleop.launch.py
```

Stand 1-2 m from the camera with your upper body in view and hold a **T-pose for 3 s** (arms
straight out to the sides). When the camera window says TRACKING, the robot follows you.

## Quick start (Physical AI)

```bash
curl -fsSL https://ollama.com/install.sh | sh     # once: local LLM server (CPU is enough)
./scripts/setup_ai.sh                             # checks Ollama, pulls granite4.2:3b + gemma3:4b
ros2 launch aibot_bringup teleop.launch.py mode:=ai llm_backend:=ollama
```

Type into the chat box under the video (visible in Physical AI mode), or without a webcam:
`ros2 run aibot_ai chat`. Try "put the blue cube in the bin", "stack the blue cube on the yellow
cube", "put both cubes in the bin", "what do you see?". New random scene:
`ros2 service call /aibot/scene/reset aibot_msgs/srv/ResetScene "{seed: 3, count: 4}"`.
**STOP** (button or Esc) halts everything within one 20 ms control tick and holds the pose;
switching back to Mimic releases held objects and glides to your pose without a jump.

> **Build rule:** always build with `python3 -m colcon build` and the venv active. With
> `--symlink-install`, colcon writes the interpreter that ran it into every Python node's
> shebang. Plain `colcon` writes `/usr/bin/python3`, which cannot import MediaPipe.

### Launch arguments

| Argument | Default | Meaning |
|---|---|---|
| `mirror` | `false` | `true`: the robot moves like your mirror image (your right arm drives its left) |
| `use_ik` | `false` | `true`: wrist-position IK with elbow hint instead of direct angle mapping |
| `camera_id` | `0` | `/dev/video<N>` |
| `pose_model` | `full` | `full` or `lite` (lite: about 4 ms faster, less accurate depth) |
| `calibrate` | `true` | wait for the T-pose before streaming |
| `gazebo_gui` / `rviz` / `camera_gui` | `true` | the three windows |
| `mode` | `mimic` | start in `mimic` or `ai` (switch at runtime in the camera window, Tab) |
| `detector` | `tabletop` | AI perception: `tabletop` (depth segmentation), `yolo_world`, `color_blob` |
| `llm_backend` | `ollama` | planner LLM (model, URL, timeouts in `aibot_ai/config/ai.yaml`) |
| `webcam` | `true` | `false`: no webcam node (AI-only use, or tests that play the operator) |

### Camera window (PyQt5)

Live video with your skeleton (left side green, right side orange), tracking lights, fps,
inference and capture→command latency, teleop state, robot-hand closure bars and the object
being held, the **Mimic | Physical AI** toggle and a **STOP** button. In Physical AI mode a chat
panel appears under the video, with the planner's state and timing. Keys: **Tab** toggle mode ·
**Esc** STOP · **C** calibrate · **Space** pause/resume · **M** mirror view · **S** skeleton · **Q** quit
(letters typed into the chat box go to the text).

## Workspace layout

```
aibot/
├── requirements.txt          mediapipe==0.10.21, numpy<2, opencv-contrib-python==4.11.0.86
├── requirements-ai.txt       optional YOLO-World detector (torch CPU, ultralytics deps)
├── scripts/
│   ├── setup_env.sh          apt, rosdep, .venv (--system-site-packages), MediaPipe models
│   ├── setup_ai.sh           Ollama check + model pulls, optional YOLO-World, import check
│   ├── fetch_star1.sh        pinned clone of roboterax/models into a temp dir, copy, generate, delete
│   ├── generate_description.py   (extra) upstream URDF -> aibot_body.xacro, collision hulls, joint_limits.yaml
│   ├── generate_objects.py   (extra) objects.yaml -> object model SDFs with exact inertias
│   ├── check_mode_switch.py  P1 acceptance: 10 mimic<->AI switches, command continuity, welds
│   ├── check_scene_reset.py  P3 acceptance: 20 seeded resets, drift
│   ├── eval_perception.py    P4: detector vs ground truth over 30 scenes
│   ├── check_pick_place.py   P5: pick + place over seeded scenes, scored with ground truth
│   └── eval_ai.py            P6/P7: natural-language commands in the sim (CSV + table), --e2e demo
└── src/
    ├── aibot_description/    STAR1 model: original URDF + meshes (BSD-3, see LICENSE), xacro,
    │                         ros2_control, aibot_head_camera.xacro (head RGB-D camera)
    ├── aibot_gazebo/         pick_table world, objects.yaml (9-object pool + bin/tray), controllers, bridge
    ├── aibot_msgs/           landmarks, GraspState/Status, DetectedObject(Array), PlaceTarget,
    │                         ExecuteSkill action, SetMode, ResetScene, SetPrompts, SetHeld, VlmQuery
    ├── aibot_teleop/         webcam_node (+ camera_gui.py), retarget_node, grasp_manager_node,
    │                         kinematics.py (shared URDF chains), utils/mode_gate.py, tests
    ├── aibot_ai/             Physical AI: mode_manager, perception, scene_manager, skill_server,
    │                         planner nodes; backends/ (Ollama), perception/, motion/ (IK, trajectories,
    │                         executor, arm planning), geometry/ (layouts, place targets), skills/,
    │                         planning/ (plan schema, prompt, agent loop); chat CLI; tests
    └── aibot_bringup/        teleop.launch.py, teleop_params.yaml, teleop.rviz
```

Beyond the planned layout: `scripts/generate_description.py` (the body xacro is ~2,000 generated
lines, so nothing in it is typed by hand), `aibot_description/LICENSE` (the BSD-3 notice must
travel with RobotEra's meshes) and `aibot_teleop/camera_gui.py`.

## Install details

`scripts/setup_env.sh` runs exactly this:

```bash
sudo apt-get install -y python3-venv python3-pip v4l-utils python3-rosdep
rosdep install --from-paths src --ignore-src -y --rosdistro jazzy
python3 -m venv --system-site-packages .venv && touch .venv/COLCON_IGNORE
.venv/bin/python -m pip install -r requirements.txt
# + pose_landmarker_{full,lite}.task and hand_landmarker.task from storage.googleapis.com/mediapipe-models
```

Why a venv **with** `--system-site-packages`: it sees ROS's rclpy and messages, but pip installs
never touch the system Python. Version pins:
- **numpy < 2.** Jazzy's compiled Python extensions are built against numpy 1.26, and numpy 2
  would break them with ABI errors.
- **opencv-contrib-python instead of opencv-python.** MediaPipe depends on the contrib
  package, and installing both puts two conflicting `cv2` packages in one site-packages.
- **4.11.0.86** is the last OpenCV that accepts numpy 1.x.

## How it works: Physical AI mode

```
            ┌──────────── /aibot/mode (mode_manager_node: MIMIC | SWITCHING | AI | HOLD) ────────────┐
            │  every node below is gated by it; STOP = HOLD (cancel plan + skills, hold the pose)    │
chat ──► planner_node ──skills──► skill_server_node ──50 Hz min-jerk──► 6 position controllers ─► Gazebo
 ▲       (Ollama LLM:          (IK, Cartesian lines,      │ /aibot/grasp_cmd        ▲ head RGB-D camera
 │        JSON-schema plan,     collision checks,         ▼                         │
 │        validate, repair,     place resolution)    grasp_manager_node      perception_node
 └─ replies, questions          ▲ world state         (weld shim)            (table plane, clusters,
                                └──────────────────── /aibot/perception/objects ◄─ ids, memory, VLM)
```

Design rules: the LLM plans in **skills**, never joint angles or coordinates; the detector
localises and the VLM only answers on demand; perception never reads simulator ground truth
(only the weld shim, evaluation scripts and tests do); everything is gated by the mode.

**Mode hand-over (`mode_manager_node`).** A small state machine owns `/aibot/mode`. To AI: pause
retargeting and MediaPipe, wait until the robot is still (no joint moved > 0.02 rad over 0.3 s;
velocity is useless because fingers chatter in contact), arm the skill server. To mimic: cancel
plan and skills, release welds, settle, resume; retargeting then glides from the *controllers'
setpoint* (not the measured pose, which jumps where a joint is blocked by contact) to the
operator at ≤ 1 rad/s. Every node also checks the mode itself (defence in depth).

**Perception (`perception_node`, `detector:=tabletop`).** Each head-camera RGB-D frame →
RANSAC table plane → points above it, minus the robot's own hands and forearms (TF at the
image's timestamp) → voxel clusters → footprint (`minAreaRect`), height, shape label
(cube/box/block/can/mug/ball/bin/tray) and dominant colour → object memory with stable ids
(`red_cube_1`), smoothing, label voting and expiry. Runs at 2 Hz in AI mode only, skipped while
an arm is over the table (occlusion artefacts); skills ask for `detect_now` when they need a
fresh view. The VLM (`/aibot/vlm/describe`, `/aibot/vlm/verify`, gemma3:4b) is on demand only.

**Skills (`skill_server_node`, action `/aibot/skills/execute`).** `scan`, `look_at`, `home`,
`locate`, `pick`, `place`, `place_in`, `move_aside`, `open_hand`, `release`; each returns a
reason code (`OK`, `NOT_FOUND`, `AMBIGUOUS`, `UNREACHABLE`, `TOO_LARGE`, `GRASP_FAILED`,
`PLACE_BLOCKED`, `NOT_HOLDING`, `TIMEOUT`, `CANCELLED`, `NOT_AI_MODE`, `INVALID`, `BUSY`).
- **IK** (`motion/ik.py`): damped least squares over waist (yaw, pitch; ±0.6 / -0.40..+0.10 rad)
  + 7 arm joints, palm normal down (finger direction free), 2 cm limit margin, no elbow
  hyperextension, multi-start seeds. Reachable = converged *and* collision-free against the
  table, the torso box, the other arm, and perceived objects (oriented boxes), sampled every
  2 cm along elbow-wrist-hand-fingertips, plus the object in the hand.
- **Grasp-first approach:** IK at the grasp point over 8 finger directions ranked by clearance
  from neighbours, then straight vertical lines up to the pre-grasp/lift heights (dense IK, 1 cm),
  then a collision-checked transfer (direct, via the side of the table, or over the top).
- **Motion:** one minimum-jerk time law per segment (Cartesian lines: along the whole dense path),
  ≤ 50 % of URDF joint speed, ≤ 1.5 rad/s, ≤ 0.10 m/s in Cartesian moves, streamed at 50 Hz from
  the controllers' setpoint; every command step ≤ 0.0375 rad.
- **Grasp:** the weld shim (gz DetachableJoint via `grasp_manager_node`), commanded first, then a
  half fist (a full fist around a welded object drives the fingers into it and jams them).
  Verified by the weld report *and* by the head camera no longer seeing the object on the table
  once it is carried beside the table. Place: release command first, then open the fingers.
- **Place targets** (`geometry/place.py`): `on`, `in`, `left_of`, `right_of`, `in_front_of`,
  `behind`, `at_region` (front/back/left/right/centre and corners of the place area), `at_xy`,
  resolved deterministically from the perceived world state into an ordered list of free spots.
- **Safety:** STOP, a cancel or a mode change stops on the next 20 ms tick; every skill has a timeout.

**Planner (`planner_node`).** The request plus a compact world state (ids, colours, labels,
positions in cm, what each hand holds) go to the local LLM with a JSON schema (Ollama structured
output). The answer is a `plan`, `clarify`, `impossible` or `answer`. A plan is validated against
the world by simulating the hands (unknown ids, picking containers, placing while empty, `in`
something that is not a container...); errors go back for one repair. Steps run one at a time;
a failed step is fed back for a replan (at most 2), otherwise the robot explains the failure.
A deterministic guard asks "which one?" when the words match several objects and the model
guessed. Chat: `/aibot/chat/user` → `/aibot/chat/robot`; status `/aibot/planner/status`.

## How it works: mimic mode

### Model preparation (`aibot_description`)
- **Re-rooted at the pelvis.** The upstream root `base_link` is the *torso*; the waist chain hangs
  below it and the legs hang from `waist_yaw_link`. Fixing `base_link` would make the waist swing
  the legs, so the chain is inverted. Axes are negated so each joint value means the same pose as
  on the real robot. This was verified against upstream forward kinematics on 200 random poses,
  with a maximum error of 1e-15.
- **Legs locked** as fixed joints at the README upright pose (hip 30°, knee -60°, ankle 30°;
  the zero pose is a crouch). The pelvis is welded to the world with the soles 5 mm above the floor.
- **Collisions:** convex hulls of the visual meshes (426k → 39.6k triangles).
- **Fixed for simulation:**
  - The upstream finger friction (1 Nm) exceeds the finger effort limits (0.4–1.1 Nm), so the
    fingers could never move; friction and damping are overridden.
  - One inertia that violated the triangle inequality (`left_hip_pitch_link`) is balanced.
- **ros2_control:** position command interfaces on 43 joints (waist 3, head 2, arms 14, hands 24).

### Controllers and gains (`aibot_gazebo`)
`JointGroupPositionController` × 6 rather than `JointTrajectoryController`. Teleop streams a new
absolute target every 33 ms; a forward controller applies it on the next 5 ms tick. JTC would
re-plan every message and need `time_from_start` ≥ one period (≥ 33 ms added delay).

gz_ros2_control's position interface is a velocity servo: `v = gain · error · update_rate`, so
the joint follows a first-order lag with τ = 1/(gain·rate). Gain 0.2 at 200 Hz gives τ = 25 ms:
no overshoot, and it holds against gravity. Measured with 30 Hz streamed sinusoids up to
3.1 rad/s: 40 ms lag, RMS error ≤ 0.009 rad, held fingers disturbed ≤ 3°, no oscillation.

### Retargeting (`retarget_node`)
1. **Body frame.** MediaPipe world landmarks (x right, y down, z away from the camera) become
   robot-style body axes: x forward, y your left, z up.
   - Mirror mode reflects y and swaps left/right before anything else.
   - The chest frame comes from the shoulder line plus the spine when the hips are visible
     (with hysteresis), otherwise the camera's vertical.
2. **Arms, direct mode (default).** STAR1's arm is pitch Y → roll X → yaw Z → elbow Y →
   elbow-yaw Z → wrist Y → wrist X, so the angles are read back stage by stage from the
   upper-arm, forearm and hand directions:
   - `sr = asin(u_y)`, `sp = atan2(-u_x, -u_z)`; then flex/yaw from the forearm; then ZYX Euler of the hand.
   - Singularities (T-pose: pitch undefined; straight elbow: arm yaw undefined) are blended
     with the previous value or relaxed toward neutral.
3. **Arms, IK mode.**
   - Damped least squares on the URDF chain with three tasks: wrist position (scaled by your
     calibrated arm lengths), an elbow-position hint and hand orientation.
   - It is seeded every frame by the direct solution. STAR1's shoulder is singular in the
     T-pose (pitch axis along the arm), where a free solver cannot move the elbow forward.
   - Trade-off: IK reaches where your scaled wrist is (better for grasping with different
     proportions), but it passes MediaPipe's depth noise straight into the reach and needs
     calibration. Direct mapping copies your arm *shape* with no reach correction.
4. **Hands.** Per-finger curl = sum of the MCP+PIP+DIP bend angles, normalised from open (25°)
   to fist (230°); thumb flexion; thumb opposition (tip–pinky-MCP distance); index spread.
   These map to 12 joints, with signs verified on the URDF. The grasp state uses hysteresis
   (close > 0.6, open < 0.4, 3 frames).
5. **Torso and head.** The waist is the chest rotation relative to the calibrated neutral,
   decomposed ZXY to match STAR1's re-rooted waist. Head yaw/pitch come from the
   ear-midpoint → nose direction.
6. **Filtering.** One-Euro filter (holds joints whose tracking is lost) → URDF limits (2 cm
   margin) → velocity limits → dead-band. Processing runs on landmark arrival, with no
   resampling timer. The robot starts from its actual `/joint_states`.
7. **T-pose calibration (3 s).** Neutral chest and head, arm lengths, and a small per-arm
   depth-bias correction.

### Grasping (`grasp_manager_node`)
Physics-only finger grasps slip in Gazebo, so a gz **DetachableJoint** welds the object to the
palm while the hand is closed and the object's centre is within 8 cm of the grasp point (between
palm and fingers), and releases it when the hand opens. The fingers still close visually.

The plugin attaches every object to both hands when the robot spawns (gz-sim 8 behaviour). The
grasp manager detaches all pairs at startup and undoes any attach it didn't request. That's
why the plugins are only loaded with `grasping:=true`, which `teleop.launch.py` sets.

### Latency (measured on this laptop's CPU)
| Stage | Time |
|---|---|
| camera → frame in memory | camera exposure + USB (inherent) |
| pose + hands inference, **in parallel** (was 45 ms sequential) | 30–32 ms |
| capture → landmarks published | ~41 ms |
| retargeting | ~1–5 ms |
| controller + servo (τ 25 ms) | ~40 ms lag at 30 Hz streaming |
| camera GUI display | same process, no ROS image transport |
| **camera capture → controller command, end to end** | **mean 62–67 ms, p95 88–96 ms** (n ≈ 300) |

The end-to-end row is measured from the frame's capture stamp to the published controller
command: 62.3 / p95 95.5 ms before the Physical AI work, 66.7 / p95 88 ms after it (the
retargeting share stayed 4.3 ms; the difference is run-to-run load). An earlier README version
quoted ~40 ms here; that was the servo lag alone, not the whole chain.

The camera GUI shows frames straight from the inference thread. Earlier the overlay was viewed
through `rqt_image_view`, which moved about 0.9 MB per frame through DDS; that was the visible lag.

## Measured results (Physical AI)

Laptop: i7-1255U (10 cores), 15 GB RAM, MX550 2 GB (treated as CPU-only). Headless Gazebo.
Each line is the acceptance script's output, not an estimate; ground truth only in the scripts.

| Phase | What | Result |
|---|---|---|
| P1 mode switch | 10 mimic↔AI cycles with a weld (`check_mode_switch.py`) | 10/10 switches; largest command step 0.037 rad (budget 0.05); 0 welds left after return to mimic; switch 0.5–2.5 s |
| P2 head camera | 640×480 RGB-D | 10.1 Hz, whole table in view at neck pitch 0.52; real-time factor 0.94 → 0.87 with the camera |
| P3 scenes | 20 seeded resets (`check_scene_reset.py`) | 20/20 stable; drift ≤ 1.0 mm; ≥ 8 graspable models + bin + tray |
| P4 perception | 30 scenes, 174 objects (`eval_perception.py`) | tabletop: recall 100 %, precision 98.3 %, position error mean 0.09 cm / p95 0.16 cm (≤ 0.6 m: p95 0.15 cm), label + colour 100 %, detect 142 ms + fusion 60 ms. YOLO-World: recall 43 %, 504 ms (not the default) |
| P4 VLM | gemma3:4b describe/verify, 448 px | correct answers, but 59–73 s per query on this CPU (image encoding): on demand only |
| P5 pick → place_in(bin) | 30 seeded scenes, 4 objects + bin (`check_pick_place.py`) | ⟦P5_IN⟧ |
| P5 other relations | on / left_of / at_region, 10 scenes each | ⟦P5_REL⟧ |
| P5 STOP / mode change | interrupted picks (`check_pick_place.py --stop-test 5`) | ⟦P5_STOP⟧ |
| P6 planner regression | 30 commands, real LLM, fake world (`test_planner_llm.py`) | ⟦P6_LLM⟧ |
| P6/P7 in the sim | 12 commands, real LLM + sim (`eval_ai.py`) | ⟦P6_SIM⟧ |
| P7 end to end | mimic → AI "pick the red cube and put it in the bin" → mimic (`eval_ai.py --e2e`) | ⟦P7_E2E⟧ |

Memory: ⟦MEM⟧

## Testing checklist

1. **URDF.** Expect `root Link: world` and `43`:
   ```bash
   xacro src/aibot_description/urdf/aibot.urdf.xacro > /tmp/a.urdf && check_urdf /tmp/a.urdf | head -4 && grep -c 'type="revolute"' /tmp/a.urdf
   ```
   ```bash
   ros2 launch aibot_description display.launch.py
   ```
   In RViz: the robot stands upright, with 43 sliders and none for the legs.
2. **Controllers.**
   ```bash
   ros2 launch aibot_gazebo sim.launch.py
   ```
   ```bash
   ros2 control list_controllers
   ```
   Expect 7 active controllers. Then send a command:
   ```bash
   ros2 topic pub --once /left_arm_controller/commands std_msgs/msg/Float64MultiArray "{data: [-1.2, 0.4, 0.0, -1.0, 0.0, 0.3, 0.0]}"
   ```
   The left arm rises forward, beside the table.
3. **Webcam landmarks.**
   ```bash
   ros2 run aibot_teleop webcam_node --ros-args -p gui:=true
   ```
   You should see the skeleton, an L/R label on each hand, about 30 fps and about 30 ms inference.
4. **Retargeting math.** Expect `24 passed`. The tests cover round trips, agreement with the
   URDF, IK, mirroring, the waist, hand curl, hysteresis and the filters:
   ```bash
   python3 -m pytest -q src/aibot_teleop/test
   ```
5. **Full teleop.**
   ```bash
   ros2 launch aibot_bringup teleop.launch.py
   ```
   Hold the T-pose until the status reads TRACKING. Then check each group: arms (raise each one
   forward and sideways), elbows, wrists, waist (turn and lean), head (turn and nod), fingers.
   Also try it with `mirror:=true`.
6. **Pick test.**
   1. Lift your right hand above table height, beside the table.
   2. Move it over the green box and lower it, palm down.
   3. Make a fist. The RViz grasp sphere turns blue and the GUI shows `holding: R: box_small`.
   4. Lift: the box comes with the hand.
   5. Open your hand: the box drops.

Physical AI (headless commands; the GUI works the same with the windows on):

7. **Unit tests** (geometry, IK/reachability, trajectories, perception, planner logic):
   ```bash
   python3 -m pytest -q src/aibot_ai/test src/aibot_teleop/test
   ```
   Expect `⟦N_TESTS⟧ passed` (the real-LLM regression is skipped unless `AIBOT_LLM_TESTS=1`).
8. **Planner with the real LLM** (needs Ollama, ~⟦LLM_MIN⟧ min):
   ```bash
   AIBOT_LLM_TESTS=1 python3 -m pytest -s src/aibot_ai/test/test_planner_llm.py
   ```
9. **Mode switching, scenes, perception, pick/place** (each against a headless bringup):
   ```bash
   ros2 launch aibot_bringup teleop.launch.py webcam:=false gazebo_gui:=false rviz:=false camera_gui:=false calibrate:=false
   ```
   ```bash
   python3 scripts/check_mode_switch.py --cycles 10
   ```
   For the others start it with `mode:=ai` added:
   ```bash
   python3 scripts/check_scene_reset.py --seeds 20
   ```
   ```bash
   python3 scripts/eval_perception.py --scenes 30
   ```
   ```bash
   python3 scripts/check_pick_place.py --relation in --scenes 30
   ```
   ```bash
   python3 scripts/check_pick_place.py --stop-test 5
   ```
10. **Natural-language tasks in the sim** (`mode:=ai`), then the end-to-end demo (mimic bringup):
    ```bash
    python3 scripts/eval_ai.py --csv /tmp/eval_ai.csv
    ```
    ```bash
    python3 scripts/eval_ai.py --e2e
    ```
11. **By hand:** `ros2 launch aibot_bringup teleop.launch.py mode:=ai`, type "put the blue cube
    in the bin" in the chat panel, press STOP while it moves (it stops at once and reports
    "Stopped"), Tab back to Mimic (the arm glides to your pose).

## Known limits (Physical AI)

- **Not verified here:** the GUI windows on a real display (rendered offscreen only), live
  webcam teleop with a person in front of the camera during this work, real camera noise (the
  simulated depth is noise-free, so perception numbers are optimistic), and a real robot.
- **Grasping is a weld shim**: the hand closes to a half fist and a DetachableJoint holds the
  object. Real grasp physics (friction, slip, finger contact) are not simulated or tested; the
  weld also lets things be held that a real hand would drop.
- **Slow LLM/VLM on CPU:** ⟦LLM_LIMIT⟧ The VLM (gemma3:4b) needs about a minute per image, so it
  is never in the control loop; verification is geometric (weld report + camera).
- **The 20 cm hand needs room:** with the palm down the fingers stick out 10 cm beyond the grasp
  point. Objects closer than ~5 cm to a neighbour on all sides may be unreachable
  (`UNREACHABLE: collides with the object ...`), and closing the fingers can nudge a neighbour.
- **Workspace:** object centres x 0.32–0.54 m, |y| ≤ 0.15 m in front of the robot (the 0.40 m arm
  with waist lean ≤ 23°). Closer than x = 0.32 the forearm runs into the torso.
- **Spheres** may roll after a place (`on`/`at_region`), since nothing holds them in place.
- **Perception** labels shapes from depth geometry only: no object categories beyond
  cube/box/block/can/mug/ball/bin/tray, and colours from a fixed palette. Objects inside a
  container merge with it in depth; the memory keeps them as "inside" instead.
- **Cut, with notice:** voice input, the Anthropic backend, pinch grasps and OWL-ViT. Each was
  optional, and none was needed for the acceptance tasks.

## Known pitfalls and fixes

## Known pitfalls and fixes

| Symptom | Cause | Fix (applied here) |
|---|---|---|
| White/missing meshes in Gazebo | `package://` not resolvable by gz | `sim.launch.py` appends `install/aibot_description/share` to `GZ_SIM_RESOURCE_PATH` |
| Robot huge or tiny | mesh units | STAR1 meshes are metres; no scale needed |
| Waist moves the legs | upstream root is the torso | re-rooted at `waist_yaw_link` |
| Fingers never move | upstream joint friction > effort limit | sim damping 0.05, friction 0 |
| A joint moves the wrong way | convention mismatch | `aibot_teleop/config/joint_mapping.yaml` → `corrections: {joint: {sign: -1}}` |
| Controller fails to load | joint name / type typos | names come from the URDF; `ros2 control list_hardware_interfaces` |
| Arms frozen after spawn | DetachableJoint welds all objects at startup | grasp manager detaches; plugins only with `grasping:=true` |
| Hands stuck under the table | fingertips hang below the table top | narrow table (\|y\| ≤ 0.18), arms swing past it |
| Cannot reach objects | 0.40 m arm, 1.33 m shoulders | table top at 0.98 m (robot elbow height) |
| Fingers/waist flop on big jumps | velocity servo jumps to max speed | teleop rate-limits; never send large steps by hand |
| `mediapipe is not importable` | nodes built with system Python | `source .venv/bin/activate && python3 -m colcon build` |
| `numpy.core.multiarray failed to import` | numpy 2 in the venv | `pip install "numpy<2"` (pinned in requirements.txt) |
| Camera GUI: "could not load Qt platform plugin xcb" | pip OpenCV redirects Qt plugins | `camera_gui.py` clears `QT_QPA_PLATFORM_PLUGIN_PATH` before Qt starts |
| Laggy video | ROS image transport | use the camera GUI; RViz image is down-scaled; `pose_model:=lite` |
| Waist pitch never changes | hips out of view | stand back so the hips are visible (yaw works without them) |
| Hands labelled the wrong side | MediaPipe handedness assumes a mirrored image | hands are matched to the pose wrists instead |

## License

Code: BSD-3-Clause. STAR1 URDF and meshes © 2024 RobotEra, BSD 3-Clause
(`src/aibot_description/LICENSE`), from https://github.com/roboterax/models.
