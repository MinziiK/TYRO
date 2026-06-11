#!/usr/bin/env bash
# Robot-B nut-fastening v22 STAGE 1 — fresh per-leg bootstrap.
#
# v22 fix over v21 (forearm-vs-tire collision on edge bolts 1/7/8/9):
#   * collision-aware clean-branch INSERT — at APPROACH→INSERT handoff the env
#     solves a tire-free coaxial seat branch (scipy least_squares with a tire-
#     penetration cost) and switches B into the staging config of that branch.
#     The INSERT plunge is a joint-space lerp staging→seat in the same branch,
#     so the forearm never clips the mounted tire and nut_collision_fail no
#     longer kills the leg ~6 cm short of the seat.
#   * Validated on the v21 checkpoint WITHOUT retraining: 6/10 → 10/10 per-bolt
#     seat at alpha=1.0. Retrain stage1 fresh so the policy learns approach on
#     the clean branch for every bolt.
#   * Enabled via --nut-v20 (train.py sets nut_b_clean_branch_insert=True).
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
echo "[nut] $(date '+%F %T') === Robot-B nut v22 STAGE 1 (fresh, per-leg) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl --nut-v20 \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.3 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 3500000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 5 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 12 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 400000 --nut-arrive-pos-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.003 \
  --eval-freq 250000 --eval-episodes 30 \
  --log-std-init -1.0 \
  --terminate-on never --max-steps 800 \
  --total-steps 4000000 --run-name nut_fastening_v22_stage1 \
  2>&1 | tee runs/nut_fastening_v22_stage1.log

echo "[nut] $(date '+%F %T') DONE"
