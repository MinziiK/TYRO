#!/usr/bin/env bash
# Robot-B nut-fastening v21 STAGE 2 — multi-bolt CHAIN fine-tune.
#
# Resumes (policy-only) from the v21 STAGE 1 checkpoint and trains the full
# 10-bolt chain per episode (per-leg OFF), hot-start alpha 0.3 -> 0.0.
#
# v21 effect on the chain: with branch-aware INSERT the plunge no longer
# freezes on edge-of-workspace bolts, so the chain advances past them and the
# policy finally earns APPROACH/transit reward for the FULL ring (v20 stage2
# stalled at ~4-5 bolts because bolt#3 froze in INSERT and killed all
# downstream reward signal). Gates stay at the v20 tight end.
#
# Eval measures the REAL task: HOME start (alpha end 0.0), 10/10 bolts,
# deterministic, 30 episodes. WATCH: n_fastened_policy must climb toward 10.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT="$(ls runs/nut_fastening_v21_stage1/ckpts/ppo_*_steps.zip 2>/dev/null \
        | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2)"
if [[ -z "${CKPT}" ]]; then
  echo "[nut] ERROR: no stage-1 checkpoint found" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') === Robot-B nut v21 STAGE 2 (chain fine-tune) from ${CKPT} ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl --nut-v20 \
  --nut-per-leg false \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 0.3 --nut-hotstart-alpha-end 0.0 \
  --nut-hotstart-hold-steps 200000 --nut-hotstart-ramp-steps 1500000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 5 --nut-arrive-ang-end-deg 5 \
  --nut-arrive-ang-hold-steps 0 --nut-arrive-ang-ramp-steps 1 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 8 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 0 --nut-arrive-pos-ramp-steps 1 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --lr 1e-4 --ent-coef 0.001 \
  --eval-freq 250000 --eval-episodes 30 \
  --log-std-init -1.0 \
  --terminate-on never --max-steps 2000 \
  --resume "${CKPT}" --resume-mode policy-only --reset-timesteps \
  --total-steps 3000000 --run-name nut_fastening_v21_stage2 \
  2>&1 | tee runs/nut_fastening_v21_stage2.log

echo "[nut] $(date '+%F %T') DONE"
