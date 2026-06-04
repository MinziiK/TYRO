#!/usr/bin/env bash
# Phase-1 mount (A) then full-cycle (B) with monitoring-friendly logging.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=0

COMMON=(
  --stage 3 --phase 1 --scene-layout fanuc_spacious
  --num-envs 72 --n-steps 341 --batch-size 1024
  --device cuda
  --eval-freq 250000 --eval-episodes 5
)

echo "[pipeline] === Phase A: mount-only ==="
python -u -m src.train "${COMMON[@]}" \
  --total-steps 2000000 --run-name phase1_mount \
  2>&1 | tee runs/phase1_mount.log

A_CKPT="runs/phase1_mount/best/best_model.zip"
if [[ ! -f "$A_CKPT" ]]; then
  A_CKPT="runs/phase1_mount/final.zip"
fi
if [[ ! -f "$A_CKPT" ]]; then
  echo "[pipeline] ERROR: no A checkpoint at $A_CKPT" >&2
  exit 1
fi
echo "[pipeline] A checkpoint: $A_CKPT"

echo "[pipeline] === Phase B: 6-stage full cycle (resume A) ==="
python -u -m src.train "${COMMON[@]}" \
  --remount-cycle --terminate-on never --max-steps 1000 \
  --total-steps 5000000 --run-name phase1_fullcycle \
  --resume "$A_CKPT" \
  2>&1 | tee runs/phase1_fullcycle.log

echo "[pipeline] DONE"
