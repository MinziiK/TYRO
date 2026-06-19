#!/usr/bin/env bash
# GUI E2E demo: one hub scenario, A mount → B nut (same offset).
# A finishes (smooth seat glide), then B takes over on the seated tire.
# Requires DISPLAY (e.g. DISPLAY=:2).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=.

A_CKPT="${A_CKPT:-runs/phase1_mount_v3_dr/final.zip}"
B_CKPT="${B_CKPT:-runs/nut_fastening_v24_dr_stageB3/ckpts/ppo_1749440_steps.zip}"
DR_CM="${DR_CM:-5.0}"
SEED="${SEED:-42}"

python -u scripts/e2e_eval.py \
  --v24 \
  --model-a "$A_CKPT" \
  --model-b "$B_CKPT" \
  --scenarios 1 \
  --seed "$SEED" \
  --dr-range-cm "$DR_CM" \
  --b-max-steps 2500 \
  --render
