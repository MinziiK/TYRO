#!/usr/bin/env bash
# Resume v22 STAGE 2 from its LATEST stage-2 checkpoint with the v22 HOLD/RETRACT
# clean-branch fix in effect. Full resume (optimizer + num_timesteps restored,
# no --reset-timesteps) so the chain fine-tune continues from where it was.
#
# The bolt-7 fix is env-side (HOLD/RETRACT now lerp inside the collision-free
# branch instead of handing back to the Cartesian servo, which snapped the
# isolated clean branch back into the tire). The learned APPROACH policy is
# unaffected, so resuming (not restarting) is correct.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT="$(ls runs/nut_fastening_v22_stage2/ckpts/ppo_*_steps.zip 2>/dev/null \
        | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2)"
if [[ -z "${CKPT}" ]]; then
  echo "[nut] ERROR: no stage-2 checkpoint to resume from" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') === v22 STAGE 2 RESUME (bolt-7 fix) from ${CKPT} ==="

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
  --eval-freq 250000 --eval-episodes 30 \
  --terminate-on never --max-steps 2000 \
  --resume "${CKPT}" --resume-mode full \
  --total-steps 3000000 --run-name nut_fastening_v22_stage2 \
  2>&1 | tee -a runs/nut_fastening_v22_stage2.log

echo "[nut] $(date '+%F %T') DONE"
