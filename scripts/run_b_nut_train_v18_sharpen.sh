#!/usr/bin/env bash
# Robot-B nut-fastening v18 — SHARPENING fine-tune of the v17 pure-RL policy.
#
# Why: v17 attempt-4 finished 4M steps with train success ~0.85 (stochastic,
# alpha=0.3, all 10 bolts incl. the bottom arc) but DETERMINISTIC success is 0%
# — the policy mean is laterally 3-6 cm off and only the exploration noise
# (log_std ~ -0.7) closes the gap (measured: scripts/_tmp_det_vs_stoch.py,
# stoch 18/20 vs det 0/20). v18 polishes the MEAN:
#   * policy-only resume from the v17 final ckpt (fresh optimizer/schedules)
#   * --force-log-std -1.5 → exploration restarts tight around the learned mean
#   * ent_coef 0 → no entropy pressure to stay noisy
#   * lr 1e-4 → gentle polish, don't wreck the learned skill
#   * curricula PINNED at final values (start=end: alpha .3, ang 12°, pos 8 cm)
#
# WATCH: eval/success_rate (deterministic) must lift off 0 — that's the metric
# this run exists for. Train success should stay >=0.8.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Sort by the numeric step count in the FILENAME (the path itself contains
# underscores, so a naive `sort -t_ -k2` keys on the wrong field).
CKPT="$(ls runs/nut_fastening_v17_purerl/ckpts/ppo_*_steps.zip \
        | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2)"
echo "[nut] $(date '+%F %T') === v18 SHARPEN fine-tune from ${CKPT} ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 0.3 --nut-hotstart-alpha-end 0.3 \
  --nut-hotstart-hold-steps 0 --nut-hotstart-ramp-steps 1 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 12 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 0 --nut-arrive-ang-ramp-steps 1 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 8 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 0 --nut-arrive-pos-ramp-steps 1 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --lr 1e-4 --ent-coef 0.0 \
  --eval-freq 250000 --eval-episodes 10 \
  --log-std-init -1.5 --force-log-std \
  --terminate-on never --max-steps 800 \
  --resume "${CKPT}" --resume-mode policy-only --reset-timesteps \
  --total-steps 1000000 --run-name nut_fastening_v18_sharpen \
  2>&1 | tee runs/nut_fastening_v18_sharpen.log

echo "[nut] $(date '+%F %T') DONE"
