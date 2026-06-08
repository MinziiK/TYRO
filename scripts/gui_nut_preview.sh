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
if [[ "$MODE" == "replay" ]]; then
  TRAJ="${2:-/tmp/nut_traj.npz}"
  EXTRA+=(--traj "$TRAJ" --step-sleep 0.03)
fi
if [[ "$MODE" == "policy" ]]; then
  MODEL="${2:-runs/nut_fastening_v5/final.zip}"
  EXTRA+=(--model "$MODEL" --alpha 0.0 --max-steps 600 --step-sleep 0.03)
  export OMP_NUM_THREADS=1
fi

echo "[gui] DISPLAY=$DISPLAY  mode=$MODE  extra=${EXTRA[*]}"
exec python -u -m scripts.preview_nut_fastening --mode "$MODE" "${EXTRA[@]}"
