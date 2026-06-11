#!/usr/bin/env bash
# Resume v17 pure-RL from the latest checkpoint with the EVAL FIX applied.
#
# Why resume (not restart): attempt-3 reached ~1M steps healthy (train
# success_rate ~0.6, no collapse). The only change is a measurement fix — the
# eval env now pins nut_b_hotstart_alpha to the curriculum end (0.3) so eval
# measures the real deployment target instead of full-HOME (which read 0%).
# Full resume restores num_timesteps + optimizer/schedules, so the curriculum
# continues exactly where it left off.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT="$(ls -t runs/nut_fastening_v17_purerl/ckpts/ppo_*_steps.zip | head -1)"
echo "[nut] $(date '+%F %T') === v17 RESUME from ${CKPT} (eval fix) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.3 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 3500000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 12 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 400000 --nut-arrive-pos-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.003 \
  --eval-freq 250000 --eval-episodes 10 \
  --log-std-init -1.0 \
  --terminate-on never --max-steps 800 \
  --resume "${CKPT}" --resume-mode full \
  --total-steps 4000000 --run-name nut_fastening_v17_purerl \
  2>&1 | tee runs/nut_fastening_v17_purerl.log

echo "[nut] $(date '+%F %T') DONE"
