#!/usr/bin/env bash
# (v7 script — superseded by run_b_nut_train_v8.sh; kept for reference)
# Robot-B nut-fastening training v7 — reward-based A-avoidance + custom bolt order.
#
# Changes vs v6 (all in src/config.py defaults, no CLI flags needed):
#   * B↔A clearance bonus on joint-center separation (w_nut_ba_clear=2.0,
#     floor=0.30, cap=0.60) — teaches B to fasten with its arm held clear of A.
#   * Stronger dedicated collision penalty (w_nut_collision=40, was shared 10).
#   * Custom balanced bolt order: 0,5,7,2,3,8,9,4,6,1.
#   * Sequential bolt learning: nut_b_hotstart_random_bolt=False.
#   * Hot-start (alpha=1) seeds B at bolt-0 approach point via iterative IK.
#
# Geometry unchanged vs v6 (raised B base 0.90,-0.75,0.0 + 10 cm nut-runner).
# Fresh training (reward/curriculum changed; do not resume v6 checkpoint).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
# One math thread per worker so the 88 SubprocVecEnv processes don't oversubscribe
# the 96 logical cores (each worker stays on ~1 core; no BLAS/OMP contention).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ENDPOSE="data/nut_mount_endpose.npz"
if [[ ! -f "$ENDPOSE" ]]; then
  echo "[nut] ERROR: missing $ENDPOSE (run endpose extraction first)" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') reusing endposes: $ENDPOSE"
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v7 (reward A-avoid + bolt order) ==="

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
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --terminate-on never --max-steps 600 \
  --total-steps 3500000 --run-name nut_fastening_v7 \
  2>&1 | tee runs/nut_fastening_v7.log

echo "[nut] $(date '+%F %T') DONE"
