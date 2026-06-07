#!/usr/bin/env bash
# Robot-B nut-fastening retrain (v2) — reverse-curriculum hot-start fix.
#
# v1 failed: the flat exp-reach landscape gave no gradient from the 1.7 m
# HOME→bolt standoff, so the policy never reached a bolt (0/10 fastened).
# v2 adds: reach_decay 0.15→0.50, nut_pos_scale 0.05, and a reverse
# curriculum that starts B at bolt 0's approach point (alpha=1) then ramps
# the start pose back to full HOME (alpha=0) over training. Eval env stays
# at alpha=0 (deployment HOME) for an honest success metric.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ENDPOSE="data/nut_mount_endpose.npz"
if [[ ! -f "$ENDPOSE" ]]; then
  echo "[nut] ERROR: missing $ENDPOSE (run endpose extraction first)" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') reusing endposes: $ENDPOSE"

echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v2 (curriculum) ==="
python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 12 \
  --nut-hotstart-curriculum \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.0 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 2000000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --num-envs 72 --n-steps 341 --batch-size 1024 \
  --device cpu \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --terminate-on never --max-steps 600 \
  --total-steps 3500000 --run-name nut_fastening_v2 \
  2>&1 | tee runs/nut_fastening_v2.log

echo "[nut] $(date '+%F %T') DONE"
