#!/usr/bin/env bash
# Robot-B nut-fastening v24 STAGE B2 — gentle DR finish from best mid-run ckpt.
#
# Stage B (0->5cm, 3M) collapsed after ~1.5M: final.zip is unusable (nom 3/10).
# Checkpoint scan (2026-06-15) found the sweet spot at 1.25M:
#   ppo_1249600_steps.zip — nom 10/10 (5/5 seeds), 2cm mean 4.0, 5cm mean 4.0
# vs Stage A: nom 10/10, 2cm mean 0.2, 5cm mean 0.8.
#
# B2 resumes that ckpt and ONLY ramps DR 3.5->5 cm slowly (hold 500k, ramp 2M)
# with a lower LR so the chain is not sacrificed for DR.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RESUME="${STAGE_RESUME_CKPT:-runs/nut_fastening_v24_dr_stageB/ckpts/ppo_1249600_steps.zip}"
if [[ ! -f "${RESUME}" ]]; then
  echo "[nut] ERROR: best mid-run ckpt missing: ${RESUME}" >&2
  exit 1
fi

TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
if [[ -n "${STAGE_RESUME_CKPT:-}" ]] && [[ "${STAGE_RESUME_CKPT}" != "runs/nut_fastening_v24_dr_stageB/ckpts/ppo_1249600_steps.zip" ]]; then
  echo "[nut] continue from ${RESUME} (full resume, total=${TOTAL_STEPS})"
  RESUME_ARGS=(--resume "$RESUME" --resume-mode full)
else
  echo "[nut] $(date '+%F %T') === v24 STAGE B2 DR finish (3.5->5cm) from ${RESUME} ==="
  RESUME_ARGS=(--resume "$RESUME" --resume-mode policy-only --reset-timesteps)
fi

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl --nut-v24 \
  --nut-per-leg false \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 0.05 --nut-hotstart-alpha-end 0.0 \
  --nut-hotstart-hold-steps 150000 --nut-hotstart-ramp-steps 800000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 5 --nut-arrive-ang-end-deg 5 \
  --nut-arrive-ang-hold-steps 0 --nut-arrive-ang-ramp-steps 1 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 8 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 0 --nut-arrive-pos-ramp-steps 1 \
  --dr-hub-offset --no-dr-cargo \
  --dr-range-curriculum --dr-range-start-cm 3.5 --dr-range-end-cm 5 \
  --dr-range-hold-steps 500000 --dr-range-ramp-steps 2000000 \
  --nut-a-hold-jitter-deg 6.0 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --lr 5e-5 --ent-coef 0.0005 \
  --eval-freq 250000 --eval-episodes 20 \
  --log-std-init -1.0 \
  --terminate-on never --max-steps 4000 \
  "${RESUME_ARGS[@]}" \
  --total-steps "${TOTAL_STEPS}" --run-name nut_fastening_v24_dr_stageB2 \
  2>&1 | tee -a runs/nut_fastening_v24_dr_stageB2.log

echo "[nut] $(date '+%F %T') DONE"
