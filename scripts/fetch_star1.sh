#!/usr/bin/env bash
# Fetch the RobotEra STAR1 model into src/aibot_description and regenerate the derived files.
#
#   urdf/star1_original.urdf   untouched upstream URDF
#   meshes/visual/*.STL        untouched upstream meshes (only the ones the URDF references)
#   LICENSE                    upstream BSD-3 license (must travel with the model files)
#
# The upstream clone lives in a temporary directory that is always deleted on exit.
# Override the pinned commit with STAR1_COMMIT=<sha> ./scripts/fetch_star1.sh
set -euo pipefail

REPO_URL="https://github.com/roboterax/models.git"
COMMIT="${STAR1_COMMIT:-e8660e664e39e5b80ebb1f252e1b77b0e6929cff}"   # 2025-01-16, inspected
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/src/aibot_description"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch_star1] cloning $REPO_URL @ ${COMMIT:0:12} into a temp dir"
git -C "$TMP" init --quiet models
git -C "$TMP/models" remote add origin "$REPO_URL"
git -C "$TMP/models" fetch --quiet --depth 1 origin "$COMMIT"
git -C "$TMP/models" checkout --quiet FETCH_HEAD
SRC="$TMP/models/star1"

mkdir -p "$PKG/urdf" "$PKG/meshes/visual"
cp "$SRC/urdf/l3_with_hand_fixedpin_xml.urdf" "$PKG/urdf/star1_original.urdf"
cp "$TMP/models/LICENSE" "$PKG/LICENSE"

# Copy only meshes the URDF references (upstream also ships unused map/foot_ee meshes).
rm -f "$PKG"/meshes/visual/*.STL
count=0
while read -r mesh; do
  cp "$SRC/meshes/$mesh" "$PKG/meshes/visual/$mesh"
  count=$((count + 1))
done < <(grep -o 'meshes/[A-Za-z0-9_]*\.STL' "$PKG/urdf/star1_original.urdf" | sed 's#meshes/##' | sort -u)
echo "[fetch_star1] copied URDF, LICENSE and $count meshes"

rm -f "$PKG"/meshes/collision/*.STL
python3 "$ROOT/scripts/generate_description.py" --package-dir "$PKG"
echo "[fetch_star1] done; temp clone removed"
