#!/usr/bin/env bash
# Robot-B nut-fastening v20 STAGE 2 — multi-bolt CHAIN fine-tune.
#
# Stage 1 (v20) taught approach+insert+retract per-leg with the v20 fixes:
#   * INSERT axial servo drives the socket to the hub-face base (full bolt
#     envelopment), seat depth tol 0.7 cm, joint-movement penalty 0.06,
#     and the bugfix that gates the servo window out of the action/jerk/
#     joint-vel/axial-PB shaping (no reward paid for env-driven motion).
#
# What stage 1 could NOT teach (per-leg distribution gaps):
#   * retract-point -> next-staging TRANSIT across the lug ring.
#   * full approach from HOME (alpha floor was 0.3).
#   * stringing 10 bolts inside one horizon.
# Stage 2 closes them: policy-only resume from stage-1, per-leg OFF (whole
# 10-bolt chain per episode in nut_bolt_order), horizon 2000, hot-start
# alpha 0.3 -> 0.0 (wean to full HOME). Gates stay at the v20 tight end
# (5 deg / 8 cm arrive, coaxial seat, 0.7 cm depth) — no re-loosening.
#
# Eval measures the REAL task: HOME start (alpha end 0.0), 10/10 bolts,
# deterministic, 30 episodes.
#
# WATCH: n_fastened_policy must climb toward 10; eval success = full chains.
#
# Throughput note: num_envs stays 88. Measured stage1 at 96 cores showed the
# box ~93% IDLE (load avg ~3.7, only ~7 of 96 cores busy) — the bottleneck is
# the per-step SubprocVecEnv sync barrier + reset cost, NOT core count. Going
# past ~the core count just adds IPC overhead / oversubscription, so more envs
# does not help here.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT="$(ls runs/nut_fastening_v20_stage1/ckpts/ppo_*_steps.zip 2>/dev/null \
        | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2)"
if [[ -z "${CKPT}" ]]; then
  echo "[nut] ERROR: no stage-1 checkpoint found" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') === Robot-B nut v20 STAGE 2 (chain fine-tune) from ${CKPT} ==="

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
  --total-steps 3000000 --run-name nut_fastening_v20_stage2 \
  2>&1 | tee runs/nut_fastening_v20_stage2.log

echo "[nut] $(date '+%F %T') DONE"
