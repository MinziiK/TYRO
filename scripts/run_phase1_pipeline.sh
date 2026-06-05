#!/usr/bin/env bash
# Phase-1 mount (A) then full-cycle (B) with monitoring-friendly logging.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=0

COMMON=(
  --stage 3 --phase 1 --scene-layout fanuc_spacious
  --num-envs 72 --n-steps 341 --batch-size 1024
  --device cuda
  --eval-freq 250000 --eval-episodes 5
  # Keep more attached/easy (carry+mount) samples so the mount policy is
  # mastered before hard pickup spawns dominate (Phase-A goal).
  --start-pos-easy-prob-schedule-mid 0.7
  --start-pos-easy-prob-schedule-end 0.6
  # Mount-tol curriculum: the callback re-broadcasts these every rollout
  # from EnvConfig() defaults (which DON'T see the fanuc_spacious layout
  # overrides), so they MUST be passed on the CLI or the soft gate reverts
  # to an unreachable 0.30 m. 0.55 m / 45deg: wider than hard gate but
  # narrower than the 0.70 m standoff so the +Y insertion leg runs first.
  --mount-radius-soft 0.55 --mount-angle-soft-deg 45
  --mount-tol-ramp-steps 1500000
)

echo "[pipeline] === Phase A: mount-only ==="
python -u -m src.train "${COMMON[@]}" \
  --total-steps 2000000 --run-name phase1_mount \
  2>&1 | tee runs/phase1_mount.log

A_CKPT="runs/phase1_mount/best/best_model.zip"
if [[ ! -f "$A_CKPT" ]]; then
  A_CKPT="runs/phase1_mount/final.zip"
fi
if [[ ! -f "$A_CKPT" ]]; then
  echo "[pipeline] ERROR: no A checkpoint at $A_CKPT" >&2
  exit 1
fi
echo "[pipeline] A checkpoint: $A_CKPT"

echo "[pipeline] === Phase B: 6-stage full cycle (resume A) ==="
# Speed: env collection is ~98.6% of wall time (scripts/profile_train.py),
# gradient update only ~1.4%. So Phase B targets collection throughput:
#   * --physics-num-sub-steps 8 : heavy-tire physics cost ~1.46x cheaper than
#     the layout default 12 (measured 23.3->33.9 env-steps/s). Verified stable
#     (tire seats, no force spikes/NaN) at 8 and 6 via
#     scripts/check_substep_stability.py; 8 keeps a solver-accuracy margin for
#     exploration-time contacts over the 5M-step run.
#   * --num-envs 128 : ~1.3x more parallel collection (scripts/bench_vecenv_fps
#     .py); near-free since the update phase is negligible. These flags come
#     AFTER ${COMMON[@]} so argparse's last-value-wins overrides --num-envs 72.
# Combined ~1.9x (Phase B ~11h -> ~6h). For max speed use 6 (~2x physics) at a
# small stability margin cost.
python -u -m src.train "${COMMON[@]}" \
  --num-envs 128 --physics-num-sub-steps 8 \
  --remount-cycle --terminate-on never --max-steps 1000 \
  --total-steps 5000000 --run-name phase1_fullcycle \
  --resume "$A_CKPT" \
  2>&1 | tee runs/phase1_fullcycle.log

echo "[pipeline] DONE"
