#!/usr/bin/env bash
# Robot-B nut-fastening v19 STAGE 1 — pure-RL + precision rework (per-leg).
#
# Why v19 (over v17/v18): v17 attempt-4 learned the task stochastically
# (train ~0.85) but deterministic success stayed 0% — the policy mean was
# laterally 3-6 cm off and only exploration noise closed the gap. v18's
# log_std sharpening lifted det to only 10-20%. The v19 fix is STRUCTURAL
# (user-spec'd process rules):
#   * --nut-v19 bundle:
#       - align servo: INSERT/RETRACT laterally servoed onto the bolt axis by
#         the env (rate-limited) -> every plunge is an exact on-axis +-Y slide.
#         The policy keeps only the axial DOF. Kills the noise-dependence.
#       - arrive gate tightened: lateral < 1.5 cm (was 3) — "the nut runner
#         must be exactly above the bolt before it may operate".
#       - seat gate coaxial: 1x nut_lateral_tol (was 2x).
#       - Robot A kinematically frozen: a rigid fixture, B contact can't move it.
#       - B collision = INSTANT FAILURE (process rule), not a soft penalty.
#       - solo 3-d action space (B Delta-pos only; no dead A/rot/grip channels).
#       - minimal-path waste cost: straight-line transit is the optimum.
#       - 250-step stall truncation: parked episodes stop burning horizon.
#   * arrive ANGLE curriculum end 5 deg (was 12) — exact Y alignment at trigger.
#   * eval 30 episodes (10 was +-10%p noise).
#
# STAGE 1 trains per-leg (one bolt per episode, random-bolt premark,
# hot-start alpha 1.0 -> 0.3). STAGE 2 (run_b_nut_train_v19_stage2.sh) then
# fine-tunes the CHAIN: per-leg OFF, full 10-bolt episodes, alpha -> 0 (HOME).
#
# WATCH: train success >= v17's profile; eval/success_rate (deterministic)
# must NOT be ~0 once alpha reaches 0.3 — the align servo should carry the
# deterministic mean through the seat gate.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ENDPOSE="data/nut_mount_endpose.npz"
if [[ ! -f "$ENDPOSE" ]]; then
  echo "[nut] ERROR: missing $ENDPOSE (run endpose extraction first)" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') === Robot-B nut v19 STAGE 1 (pure RL + precision rework, per-leg) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl --nut-v19 \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.3 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 3500000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 5 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 12 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 400000 --nut-arrive-pos-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.003 \
  --eval-freq 250000 --eval-episodes 30 \
  --log-std-init -1.0 \
  --terminate-on never --max-steps 800 \
  --total-steps 4000000 --run-name nut_fastening_v19_stage1 \
  2>&1 | tee runs/nut_fastening_v19_stage1.log

echo "[nut] $(date '+%F %T') DONE"
