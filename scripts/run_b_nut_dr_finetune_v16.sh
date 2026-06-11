#!/usr/bin/env bash
# Robot-B nut-fastening v16 — DR fine-tune (hub placement robustness).
#
# Goal: take the converged v15 nominal policy (planner + residual, no DR) and
# fine-tune it under HUB placement error so the PPO residual learns to correct
# for the hub being off-nominal. This is where the residual earns its keep:
# at 0 cm the nominal planner+macro already solves the task (smoke = 10/10), so
# the policy's value is only measurable once the hub moves.
#
# Design (decided 2026-06-09):
#   * Resume v15 weights, RESET the step counter (policy-only) so the DR
#     curriculum schedule starts at t=0. Full resume would restore t=3.5M and
#     skip the ramp entirely.
#   * DR hub offset CURRICULUM 0 -> 5 cm (hold 0 for 200k to re-confirm the
#     nominal, then ramp to 5 cm over 1.0M). Straight 5 cm risks OOD collapse.
#   * HUB ONLY (--no-dr-cargo): the nut task doesn't depend on the cargo spawn;
#     an independent cargo offset would just add noise to the measurement.
#   * Arrive-angle gate stays FIXED at the hard spec (12 deg, no re-ramp): the
#     v15 policy already aligns to 12 deg, so --no-nut-arrive-ang-curriculum
#     keeps the config default 12 deg instead of re-loosening to 35 deg.
#   * Everything else mirrors v15 (planner+residual, scripted macro, horizon
#     2000, ent 0, log_std -1.2).
#   * A–B robustness (2026-06-09):
#       - Hub DR resets IK A to the *current* seated-tire 6-o'clock (bank is
#         warm-start only) so A tracks the offset hub instead of a stale pose.
#       - nut_a_hold_jitter 3 -> 6 deg (wider support-pose spread).
#       - w_nut_ba_clear 0.3 restored (engagement-gated clearance shaping).
#
# WATCH: curriculum/dr_range_cm climbing 0->5, success_rate vs dr_range.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RESUME="runs/nut_fastening_v15/final.zip"
if [[ ! -f "$RESUME" ]]; then
  echo "[nut] ERROR: missing $RESUME — let v15 finish first (saves final.zip on exit)." >&2
  echo "[nut] (fallback: use runs/nut_fastening_v15/best/best_model.zip)" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') === Robot-B nut DR fine-tune v16 (hub 0->5cm) from $RESUME ==="

# Resume policy: normally policy-only from v15 with a reset step counter (so the
# DR curriculum schedule starts at t=0). A crash-retry from the chain sets
# ``STAGE_RESUME_CKPT`` to this run's latest checkpoint; in that case continue
# the SAME run (full resume, keep the step counter so the curriculum picks up
# where it left off) instead of restarting the DR ramp.
if [[ -n "${STAGE_RESUME_CKPT:-}" ]]; then
  echo "[nut] retry: continuing from $STAGE_RESUME_CKPT (full resume, keep steps)"
  RESUME_ARGS=(--resume "$STAGE_RESUME_CKPT" --resume-mode full)
else
  RESUME_ARGS=(--resume "$RESUME" --resume-mode policy-only --reset-timesteps)
fi

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-b-planner-residual \
  --no-nut-hotstart-curriculum \
  --no-nut-arrive-ang-curriculum \
  --dr-hub-offset --no-dr-cargo \
  --dr-range-curriculum --dr-range-start-cm 0 --dr-range-end-cm 5 \
  --dr-range-hold-steps 200000 --dr-range-ramp-steps 1000000 \
  --nut-a-hold-jitter-deg 6.0 --w-nut-ba-clear 0.3 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.0 \
  --eval-freq 250000 --eval-episodes 10 \
  --log-std-init -1.2 \
  --terminate-on never --max-steps 2000 \
  "${RESUME_ARGS[@]}" \
  --total-steps 1500000 --run-name nut_fastening_v16_dr \
  2>&1 | tee -a runs/nut_fastening_v16_dr.log

echo "[nut] $(date '+%F %T') DONE"
