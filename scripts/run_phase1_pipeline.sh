#!/usr/bin/env bash
# Phase-1 mount (A) then full-cycle (B) with monitoring-friendly logging.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
# CPU policy (SB3 recommends CPU for MLP; profiling shows collection is ~98.6%
# of wall time so device barely matters). Pin BLAS/OMP to 1 thread so the many
# SubprocVecEnv workers do not oversubscribe the cores.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

COMMON=(
  --stage 3 --phase 1 --scene-layout fanuc_spacious
  --num-envs 72 --n-steps 341 --batch-size 1024
  --device cpu
  --eval-freq 250000 --eval-episodes 5
  # 2026-06-06 — tighten exploration so the stochastic residual stays near the
  # already-good baked nominal (which mounts at d=0). std 1.0 -> ~0.61; paired
  # with planner_pos_offset_scale 0.20->0.12 (config) this cuts the per-step
  # residual noise the 0.04 m mount gate sees from ~0.16 m to ~0.07 m. Moderate
  # (-0.5, not -1.0) so Phase B still has exploration for the new FSM stages.
  --log-std-init -0.5
  # Keep more attached/easy (carry+mount) samples so the mount policy is
  # mastered before hard pickup spawns dominate (Phase-A goal).
  --start-pos-easy-prob-schedule-mid 0.7
  --start-pos-easy-prob-schedule-end 0.6
  # Mount-tol curriculum: the callback re-broadcasts these every rollout
  # from EnvConfig() scalar defaults, so they MUST be passed on the CLI or
  # the soft gate reverts
  # to an unreachable 0.30 m. 0.55 m / 45deg: wider than hard gate but
  # narrower than the 0.70 m standoff so the +Y insertion leg runs first.
  --mount-radius-soft 0.55 --mount-angle-soft-deg 45
  # 2026-06-06 — ramp stretched 1.5M -> 2.5M. With Phase-A total = 2M, the
  # gate now reaches only ~80% tightened (≈0.14 m / 13°) by the end instead of
  # snapping to the hard 0.04 m / 5° gate at 1.5M. That kept rollout
  # success_rate high (the std≈0.78 exploration noise no longer over-shoots a
  # razor-thin gate) while deterministic eval already mounts at r=0. Phase B
  # overrides this to 3M below (scaled to its 5M horizon).
  --mount-tol-ramp-steps 2500000
)

echo "[pipeline] === Phase A: mount-only ==="
python -u -m src.train "${COMMON[@]}" \
  --total-steps 2000000 --run-name phase1_mount_v2 \
  2>&1 | tee runs/phase1_mount_v2.log

A_CKPT="runs/phase1_mount_v2/best/best_model.zip"
if [[ ! -f "$A_CKPT" ]]; then
  A_CKPT="runs/phase1_mount_v2/final.zip"
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
  --mount-tol-ramp-steps 3000000 \
  --remount-cycle --terminate-on never --max-steps 1000 \
  --total-steps 5000000 --run-name phase1_fullcycle_v3 \
  --resume "$A_CKPT" \
  2>&1 | tee runs/phase1_fullcycle_v2.log

echo "[pipeline] DONE"
