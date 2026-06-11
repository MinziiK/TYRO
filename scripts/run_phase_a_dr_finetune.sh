#!/usr/bin/env bash
# Phase-A mount DR fine-tune — hub placement robustness for Robot A.
#
# Motivation: the converged Phase-A policy (phase1_mount_v2_ft03) was trained at
# --phase 1, i.e. a FIXED hub (phase_ranges_cm[0] = 0 cm). It has never seen a
# displaced hub, so its robustness to hub placement error is untested/weak. This
# run fine-tunes it under a HUB DR curriculum 0 -> 5 cm so the planner residual
# learns to correct for an offset hub — the Robot-A analogue of the Robot-B v16
# nut DR fine-tune, making the end-to-end (A mount + B fasten) DR eval coherent.
#
# Design (mirrors v16):
#   * Resume ft03 best weights, policy-only + reset step counter so the DR
#     curriculum schedule starts at t=0 (full resume would skip the ramp).
#   * Hub DR CURRICULUM 0 -> 5 cm (hold 0 for 200k, ramp over 1.0M). Straight
#     5 cm risks collapsing the mount gate before the policy adapts.
#   * HUB ONLY (--no-dr-cargo): cargo + back wall already translate WITH the hub
#     (scene drift), so the relative mount geometry is preserved; an independent
#     cargo offset would only add unrelated noise.
#   * Residual authority 0.03 -> 0.06 m: a +-5 cm hub offset is un-correctable
#     by a +-3 cm residual, so the budget must exceed the offset.
#   * Mount tolerance ramp kept long so the gate tightens slowly under DR.
#
# WATCH: curriculum/dr_range_cm 0->5, rollout/success_rate vs dr_range.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

A_CKPT="runs/phase1_mount_v2_ft03/best/best_model.zip"
if [[ ! -f "$A_CKPT" ]]; then
  echo "[A-dr] ERROR: missing checkpoint $A_CKPT" >&2
  echo "[A-dr] (fallback: runs/phase1_mount_v2/best/best_model.zip)" >&2
  exit 1
fi
echo "[A-dr] $(date '+%F %T') === Phase-A mount DR fine-tune (hub 0->5cm) from $A_CKPT ==="

# Resume policy: normally policy-only from ft03 with a reset step counter (DR
# curriculum starts at t=0). A crash-retry from the chain sets
# ``STAGE_RESUME_CKPT`` to this run's latest checkpoint; continue the SAME run
# (full resume, keep steps) so the curriculum resumes where it left off.
if [[ -n "${STAGE_RESUME_CKPT:-}" ]]; then
  echo "[A-dr] retry: continuing from $STAGE_RESUME_CKPT (full resume, keep steps)"
  RESUME_ARGS=(--resume "$STAGE_RESUME_CKPT" --resume-mode full)
else
  RESUME_ARGS=(--resume "$A_CKPT" --resume-mode policy-only --reset-timesteps)
fi

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --dr-hub-offset --no-dr-cargo \
  --dr-range-curriculum --dr-range-start-cm 0 --dr-range-end-cm 5 \
  --dr-range-hold-steps 200000 --dr-range-ramp-steps 1000000 \
  --planner-pos-offset-scale 0.06 \
  --mount-radius-soft 0.55 --mount-angle-soft-deg 45 \
  --mount-tol-ramp-steps 2500000 \
  --start-pos-easy-prob-schedule-mid 0.85 --start-pos-easy-prob-schedule-end 0.8 \
  --num-envs 72 --n-steps 341 --batch-size 1024 \
  --device cpu \
  --eval-freq 250000 --eval-episodes 10 \
  --log-std-init -0.5 \
  --total-steps 2000000 \
  "${RESUME_ARGS[@]}" \
  --run-name phase1_mount_v3_dr \
  2>&1 | tee -a runs/phase1_mount_v3_dr.log

echo "[A-dr] $(date '+%F %T') DONE"
