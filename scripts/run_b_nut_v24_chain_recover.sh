#!/usr/bin/env bash
# Robot-B nut-fastening v24 STAGE A — spin-free chain RECOVERY (no DR).
#
# Goal (decided 2026-06-14): the user requires the wrist 360° spin GONE. That
# mandates retraining (proven: changing the macro winding moves the policy's
# resume obs out of distribution). v23 (approach-seed IK) and v24-DR (spin fix
# + DR simultaneously) both regressed the continuous chain (1/10, 3/10).
#
# This stage isolates ONE hard problem: re-learn the continuous 10-bolt chain
# under the spin-free clean-branch macro (nut_clean_shortest_macro), at the
# NOMINAL hub only (DR OFF). DR robustness is deferred to STAGE B once the
# spin-free chain is back to ~10/10.
#
# Resume from v22 STAGE-2 final (continuous 10/10 prior) so we perturb an
# already-good chain policy with the new macro and let it adapt, rather than
# rebuilding the chain from scratch.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RESUME="${STAGE_RESUME_CKPT:-}"
if [[ -z "${RESUME}" ]]; then
  if [[ -f runs/nut_fastening_v22_stage2/final.zip ]]; then
    RESUME="runs/nut_fastening_v22_stage2/final.zip"
  else
    RESUME="$(ls runs/nut_fastening_v22_stage2/ckpts/ppo_*_steps.zip 2>/dev/null \
              | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
  fi
fi
if [[ -z "${RESUME}" ]]; then
  echo "[nut] ERROR: no v22 stage-2 checkpoint — need chain 10/10 base." >&2
  exit 1
fi

TOTAL_STEPS="${TOTAL_STEPS:-4000000}"
if [[ -n "${STAGE_RESUME_CKPT:-}" ]]; then
  echo "[nut] retry/continue from ${RESUME} (full resume, total=${TOTAL_STEPS})"
  RESUME_ARGS=(--resume "$RESUME" --resume-mode full)
else
  echo "[nut] $(date '+%F %T') === v24 STAGE A chain recovery (spin-free, no DR) from ${RESUME} ==="
  RESUME_ARGS=(--resume "$RESUME" --resume-mode policy-only --reset-timesteps)
fi

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl --nut-v24 \
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
  "${RESUME_ARGS[@]}" \
  --total-steps "${TOTAL_STEPS}" --run-name nut_fastening_v24_chain \
  2>&1 | tee -a runs/nut_fastening_v24_chain.log

echo "[nut] $(date '+%F %T') DONE"
