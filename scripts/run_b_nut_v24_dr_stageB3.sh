#!/usr/bin/env bash
# Robot-B nut-fastening v24 STAGE B3 — 5cm corner hardening.
#
# B2 (3.5->5cm, 2M) fixed the 2cm regime (B2_1.5M: 0cm 9.6, 2cm 10.0) but the
# 5cm corner stayed weak (mean ~4.6, continuous 10/10 only 1/5 seeds). The
# offsets at the very edge put some bolts near Robot B's workspace limit.
#
# B3 resumes the best B2 ckpt (1.5M) and concentrates DR at the 4.5->5cm edge
# (hold 4.5 for 300k, ramp to 5 over 1.5M) so the policy spends its whole
# budget on the hard corner instead of re-learning the easy regimes. Low LR
# keeps the already-good 0/2cm chain intact.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DEFAULT_BASE="runs/nut_fastening_v24_dr_stageB2/ckpts/ppo_1499520_steps.zip"
RESUME="${STAGE_RESUME_CKPT:-$DEFAULT_BASE}"
if [[ ! -f "${RESUME}" ]]; then
  echo "[nut] ERROR: base ckpt missing: ${RESUME}" >&2
  exit 1
fi

TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
if [[ -n "${STAGE_RESUME_CKPT:-}" ]] && [[ "${STAGE_RESUME_CKPT}" != "${DEFAULT_BASE}" ]]; then
  echo "[nut] continue from ${RESUME} (full resume, total=${TOTAL_STEPS})"
  RESUME_ARGS=(--resume "$RESUME" --resume-mode full)
else
  echo "[nut] $(date '+%F %T') === v24 STAGE B3 5cm corner harden (4.5->5cm) from ${RESUME} ==="
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
  --dr-range-curriculum --dr-range-start-cm 4.5 --dr-range-end-cm 5 \
  --dr-range-hold-steps 300000 --dr-range-ramp-steps 1500000 \
  --nut-a-hold-jitter-deg 6.0 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --lr 3e-5 --ent-coef 0.0005 \
  --eval-freq 250000 --eval-episodes 20 \
  --log-std-init -1.2 \
  --terminate-on never --max-steps 4000 \
  "${RESUME_ARGS[@]}" \
  --total-steps "${TOTAL_STEPS}" --run-name nut_fastening_v24_dr_stageB3 \
  2>&1 | tee -a runs/nut_fastening_v24_dr_stageB3.log

echo "[nut] $(date '+%F %T') DONE"
