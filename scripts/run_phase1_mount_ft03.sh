#!/usr/bin/env bash
# Phase-1 mount finetune: resume best @ planner_pos_offset_scale=0.03 (fanuc_spacious).
# New run dir — does not overwrite runs/phase1_mount_v2/.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

A_CKPT="runs/phase1_mount_v2/best/best_model.zip"
if [[ ! -f "$A_CKPT" ]]; then
  echo "[ft03] ERROR: missing checkpoint $A_CKPT" >&2
  exit 1
fi

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --num-envs 72 --n-steps 341 --batch-size 1024 \
  --device cpu \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --start-pos-easy-prob-schedule-mid 0.85 \
  --start-pos-easy-prob-schedule-end 0.8 \
  --mount-radius-soft 0.55 --mount-angle-soft-deg 45 \
  --mount-tol-ramp-steps 2500000 \
  --total-steps 3000000 \
  --run-name phase1_mount_v2_ft03 \
  --resume "$A_CKPT" \
  --resume-mode full \
  2>&1 | tee runs/phase1_mount_v2_ft03.log

echo "[ft03] DONE"
