#!/usr/bin/env bash
# Robot-B nut-fastening v24 STAGE B — DR robustness on the spin-free chain.
#
# Prereq: STAGE A done -> runs/nut_fastening_v24_chain/final.zip already has
# the spin-free clean-branch macro (nut_clean_shortest_macro) AND the
# continuous 10/10 chain at the NOMINAL hub.
#
# Why a separate stage (decided 2026-06-14): the deprecated single-shot
# finetune resumed from v22 (OLD macro) and tried to learn "new macro + DR"
# at once -> chain collapsed. Measured: the Stage-A model scores 10/10 at
# offset 0 but 0/10 at a 2 cm hub offset, so it CANNOT do E2E (+-5 cm) as-is.
# This stage perturbs ONLY one thing: add the hub-XY offset curriculum 0->5 cm
# on top of the already-good spin-free chain. Macro dynamics are unchanged
# from the resume distribution, so the chain should adapt rather than collapse.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RESUME="${STAGE_RESUME_CKPT:-}"
if [[ -z "${RESUME}" ]]; then
  if [[ -f runs/nut_fastening_v24_chain/final.zip ]]; then
    RESUME="runs/nut_fastening_v24_chain/final.zip"
  else
    RESUME="$(ls runs/nut_fastening_v24_chain/ckpts/ppo_*_steps.zip 2>/dev/null \
              | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
  fi
fi
if [[ -z "${RESUME}" ]]; then
  echo "[nut] ERROR: no v24 STAGE-A checkpoint — run STAGE A first." >&2
  exit 1
fi

TOTAL_STEPS="${TOTAL_STEPS:-3000000}"
if [[ -n "${STAGE_RESUME_CKPT:-}" ]]; then
  echo "[nut] retry/continue from ${RESUME} (full resume, total=${TOTAL_STEPS})"
  RESUME_ARGS=(--resume "$RESUME" --resume-mode full)
else
  echo "[nut] $(date '+%F %T') === v24 STAGE B DR finetune (hub 0->5cm) from ${RESUME} ==="
  RESUME_ARGS=(--resume "$RESUME" --resume-mode policy-only --reset-timesteps)
fi

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl --nut-v24 \
  --nut-per-leg false \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 0.15 --nut-hotstart-alpha-end 0.0 \
  --nut-hotstart-hold-steps 100000 --nut-hotstart-ramp-steps 1000000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 5 --nut-arrive-ang-end-deg 5 \
  --nut-arrive-ang-hold-steps 0 --nut-arrive-ang-ramp-steps 1 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 8 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 0 --nut-arrive-pos-ramp-steps 1 \
  --dr-hub-offset --no-dr-cargo \
  --dr-range-curriculum --dr-range-start-cm 0 --dr-range-end-cm 5 \
  --dr-range-hold-steps 200000 --dr-range-ramp-steps 1500000 \
  --nut-a-hold-jitter-deg 6.0 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --lr 1e-4 --ent-coef 0.001 \
  --eval-freq 250000 --eval-episodes 20 \
  --log-std-init -1.0 \
  --terminate-on never --max-steps 4000 \
  "${RESUME_ARGS[@]}" \
  --total-steps "${TOTAL_STEPS}" --run-name nut_fastening_v24_dr_stageB \
  2>&1 | tee -a runs/nut_fastening_v24_dr_stageB.log

echo "[nut] $(date '+%F %T') DONE"
