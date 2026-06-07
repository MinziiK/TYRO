#!/usr/bin/env bash
# Open the Robot-B nut-fastening GUI preview on the headless noVNC display.
#
# Browser: http://<server>:6082/vnc.html  (display :2)
#
# Usage:
#   bash scripts/gui_nut_preview.sh              # static setup
#   bash scripts/gui_nut_preview.sh oracle       # IK demo (all 10 bolts)
#   bash scripts/gui_nut_preview.sh zero         # untrained zero-policy
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export DISPLAY="${DISPLAY:-:2}"

MODE="${1:-setup}"
EXTRA=()
if [[ -f data/nut_mount_endpose.npz ]]; then
  EXTRA+=(--endpose data/nut_mount_endpose.npz)
fi

echo "[gui] DISPLAY=$DISPLAY  mode=$MODE"
python -u -m scripts.preview_nut_fastening --mode "$MODE" "${EXTRA[@]}"
