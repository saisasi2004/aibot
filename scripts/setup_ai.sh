#!/usr/bin/env bash
# Setup for the Physical AI mode (run after scripts/setup_env.sh).
#
#   1. Ollama: checks the server is installed and reachable, pulls the planner and VLM models
#      named in src/aibot_ai/config/ai.yaml (granite4.2:3b ~2.2 GB, gemma3:4b ~3.4 GB)
#   2. optional (WITH_YOLO=1): YOLO-World detector packages from requirements-ai.txt into the venv
#      and its weights into src/aibot_ai/models/
#   3. import check with ROS sourced (numpy must still be < 2)
#
# Usage:  ./scripts/setup_ai.sh                 (Ollama + models)
#         WITH_YOLO=1 ./scripts/setup_ai.sh     (also the optional YOLO-World detector, ~1 GB)
#         SKIP_MODELS=1 ./scripts/setup_ai.sh   (do not pull models)
# Install Ollama first if needed:  curl -fsSL https://ollama.com/install.sh | sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
CFG="$ROOT/src/aibot_ai/config/ai.yaml"
URL="${OLLAMA_URL:-http://localhost:11434}"
source_ros() { set +u; source /opt/ros/jazzy/setup.bash; set -u; }
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV: run scripts/setup_env.sh first"; exit 1; }

model_of() {  # section -> model name from ai.yaml
  "$VENV/bin/python" -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(c['$1']['ros__parameters']['$2']['model'])"
}
PLANNER="$(model_of planner_node llm)"
VLM="$(model_of perception_node vlm)"

echo "[setup_ai] Ollama at $URL"
if ! command -v ollama >/dev/null; then
  echo "  ollama not installed: curl -fsSL https://ollama.com/install.sh | sh"; exit 1
fi
if ! curl -fsS "$URL/api/tags" >/dev/null; then
  echo "  server not reachable: start it with 'sudo systemctl start ollama' (or 'ollama serve')"; exit 1
fi
if [[ "${SKIP_MODELS:-0}" != 1 ]]; then
  for m in "$PLANNER" "$VLM"; do
    echo "  pulling $m"
    ollama pull "$m"
  done
fi
echo "  installed models:"; ollama list | sed 's/^/    /'

if [[ "${WITH_YOLO:-0}" == 1 ]]; then
  echo "[setup_ai] YOLO-World detector (optional)"
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements-ai.txt"
  "$VENV/bin/python" -m pip install --no-deps ultralytics==8.4.188
  mkdir -p "$ROOT/src/aibot_ai/models"
  (cd "$ROOT/src/aibot_ai/models" && "$VENV/bin/python" -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8s-worldv2.pt')")
fi

echo "[setup_ai] import check (ROS sourced)"
source_ros
"$VENV/bin/python" - <<PY
import numpy, cv2, rclpy, json, urllib.request
assert int(numpy.__version__.split(".")[0]) < 2, f"numpy {numpy.__version__}: ROS needs numpy<2"
print(f"  numpy {numpy.__version__} | cv2 {cv2.__version__} | rclpy OK")
tags = [m["name"] for m in json.load(urllib.request.urlopen("$URL/api/tags"))["models"]]
for m in ("$PLANNER", "$VLM"):
    print(f"  {m}: {'installed' if m in tags or m + ':latest' in tags else 'MISSING (ollama pull ' + m + ')'}")
if "${WITH_YOLO:-0}" == "1":
    import torch, ultralytics
    print(f"  torch {torch.__version__} | ultralytics {ultralytics.__version__}")
PY
echo "[setup_ai] done. Build: source /opt/ros/jazzy/setup.bash && source .venv/bin/activate && python3 -m colcon build --symlink-install"
