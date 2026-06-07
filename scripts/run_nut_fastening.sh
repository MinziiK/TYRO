#!/usr/bin/env bash
# Robot-B sequential nut-fastening (Phase-A condition).
#
# The tire is held mounted on the hub (Robot A frozen as a static fixture,
# tire bonded to the hub flange) and Robot B (UR10e + nut-runner tool) learns
# to seat its tool on each of the 10 hub bolts in turn. "Fastening" is purely
# geometric (no nut bodies / torque in sim): the tool_tip must hold within
# nut_reach_tol / nut_align_tol_rad of a bolt for nut_hold_steps consecutive
# steps, then the target advances to the next bolt. Success = all bolts done.
#
# This is a standalone single-arm (Robot-B-only) policy — it does NOT resume
# the Robot-A mount checkpoint (different action manifold: A frozen, B live).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
# CPU policy (collection-bound; see run_phase1_pipeline.sh). Pin BLAS/OMP so
# the SubprocVecEnv workers don't oversubscribe the cores.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[nut] === Robot B sequential nut-fastening ==="
python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 12 \
  --num-envs 72 --n-steps 341 --batch-size 1024 \
  --device cpu \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --terminate-on never --max-steps 600 \
  --total-steps 3000000 --run-name nut_fastening_v1 \
  2>&1 | tee runs/nut_fastening_v1.log

echo "[nut] DONE"
