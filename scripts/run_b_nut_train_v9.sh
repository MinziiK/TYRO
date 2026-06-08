#!/usr/bin/env bash
# Robot-B nut-fastening training v9 — COLLISION-AVOIDANCE via exploration.
#
# Hypothesis (user): drop the A↔B distance *shaping* reward and let raised
# exploration find collision-free joint angles on its own, keeping only the
# real-contact hard penalty as the avoidance teacher.
#
# Changes vs v8 (all isolated):
#   * w_nut_ba_clear 0.4 → 0.0  (remove mesh-blind joint-center shaping; it
#     floored at ~0.3 m and gave no gradient in the ~6 cm bottom-bolt corridor)
#   * w_nut_collision 40.0 (KEPT) — real getContactPoints A↔B penalty is the
#     only avoidance signal now.
#   * ent_coef 0.0 → 0.008      (exploration: was literally zero in v7/v8)
#   * nut_b_hotstart_random_bolt True → False — always cold-start at bolt 0 so
#     n_fastened_policy / eval success are honest (no premark inflation) and the
#     policy must learn the full traverse + collision-free approach.
#
# WATCH: rollout/n_fastened_policy and eval success (NOT n_fastened).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ENDPOSE="data/nut_mount_endpose.npz"
if [[ ! -f "$ENDPOSE" ]]; then
  echo "[nut] ERROR: missing $ENDPOSE (run endpose extraction first)" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') reusing endposes: $ENDPOSE"
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v9 (collision-avoid via exploration) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-hotstart-curriculum \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.0 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 2000000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.008 \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --terminate-on never --max-steps 600 \
  --total-steps 3500000 --run-name nut_fastening_v9 \
  2>&1 | tee runs/nut_fastening_v9.log

echo "[nut] $(date '+%F %T') DONE"
