#!/usr/bin/env bash
# One-time environment setup for the aibot workspace.
#
#   1. apt:   python3-venv / pip / v4l-utils, rosdep keys declared in src/*/package.xml
#   2. venv:  ~/aibot/.venv created with --system-site-packages, so it sees ROS 2 Jazzy's
#             rclpy etc.; MediaPipe & co. from requirements.txt are installed into it
#   3. models: MediaPipe .task files into src/aibot_teleop/models/
#
# Usage:  ./scripts/setup_env.sh              (asks for your sudo password once)
#         SKIP_APT=1 ./scripts/setup_env.sh    (no sudo: only venv + models)
#
# Afterwards build with the venv's Python, so the node scripts use it as their interpreter:
#   source /opt/ros/jazzy/setup.bash && source ~/aibot/.venv/bin/activate
#   cd ~/aibot && python3 -m colcon build --symlink-install
# (plain `colcon build` would bake /usr/bin/python3 into the scripts, which cannot see MediaPipe)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
MODELS="$ROOT/src/aibot_teleop/models"
source_ros() { set +u; source /opt/ros/jazzy/setup.bash; set -u; }

if [[ "${SKIP_APT:-0}" != 1 ]]; then
  echo "[setup_env] apt packages"
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip v4l-utils python3-rosdep
  echo "[setup_env] rosdep"
  source_ros
  [[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]] || sudo rosdep init
  rosdep update --rosdistro jazzy
  rosdep install --from-paths "$ROOT/src" --ignore-src -y --rosdistro jazzy
fi

echo "[setup_env] python venv at $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
touch "$VENV/COLCON_IGNORE"   # keep colcon from crawling site-packages
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

echo "[setup_env] MediaPipe models into $MODELS"
mkdir -p "$MODELS"
BASE="https://storage.googleapis.com/mediapipe-models"
fetch() {  # url file
  if [[ -s "$MODELS/$2" ]]; then echo "  have $2"; return; fi
  curl -fL --retry 3 -o "$MODELS/$2.part" "$1" && mv "$MODELS/$2.part" "$MODELS/$2"
  echo "  got  $2 ($(du -h "$MODELS/$2" | cut -f1))"
}
fetch "$BASE/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" pose_landmarker_full.task
fetch "$BASE/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" pose_landmarker_lite.task
fetch "$BASE/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" hand_landmarker.task

echo "[setup_env] checking imports inside the venv (with ROS sourced)"
source_ros
"$VENV/bin/python" - <<'EOF'
import os, cv2, mediapipe, numpy, rclpy
from mediapipe.tasks.python import vision  # noqa: F401
print(f"  numpy {numpy.__version__} from {os.path.dirname(numpy.__file__)}")
print(f"  cv2 {cv2.__version__} | mediapipe {mediapipe.__version__} | rclpy OK")
assert numpy.__version__.startswith("1."), "numpy 2 in the venv would break ROS 2 Jazzy"
EOF
echo "[setup_env] done. Build with the venv's Python:"
echo "  source /opt/ros/jazzy/setup.bash && source $VENV/bin/activate && cd $ROOT && python3 -m colcon build --symlink-install"
