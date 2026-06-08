#!/usr/bin/env bash
# Robot-B nut-fastening training v14 — planner + residual (oracle path).
#
# v1–v13 root cause: Robot B had no nominal trajectory — policy did 1.7 m
# magnetic exploration from HOME. v14 mirrors Robot A: min-jerk nominal path
# (HOME→hub center→staging, XZ hop bolt→bolt) + PPO XYZ residual ±5 cm.
#
#   Nominal: oracle answer path (_generate_nut_approach_traj)
#   Residual: action[6:9] × nut_planner_pos_residual_scale (0.05 m)
#   Orientation: nut_b_lock_coaxial (env-controlled)
#   Insert/retract: scripted macro (not learned)
#   Hot-start curriculum: OFF (planner owns HOME→bolt)
#   Reward: v13 farm-proof PB + corridor (standing kernels = 0)
#   Bolt order: (0,5,7,2,3,8,9,4,6,1) — unchanged
#
# WATCH: n_fastened_policy + eval success at alpha-free training.
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
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v14 (planner+residual) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-b-planner-residual \
  --no-nut-hotstart-curriculum \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.008 \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --terminate-on never --max-steps 600 \
  --total-steps 3500000 --run-name nut_fastening_v14 \
  2>&1 | tee runs/nut_fastening_v14.log

echo "[nut] $(date '+%F %T') DONE"
