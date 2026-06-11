#!/usr/bin/env bash
# Robot-B nut-fastening training v15 — planner + residual, horizon + entropy fix.
#
# v14 diagnosis (runs/nut_fastening_v14.log @2.6M):
#   * n_fastened_policy stuck ~1, ep_rew flat ~390, success_rate 0.0.
#   * ROOT CAUSE 1 (horizon): each bolt needs ~159 env steps (zero-residual
#     smoke = 10/10 in 1592 steps), but --max-steps was 600 → even a perfect
#     policy can only fasten ~3-4 bolts/episode, and eval (also 600) can NEVER
#     reach all-10, so success_rate=0.0 was guaranteed by the horizon.
#   * ROOT CAUSE 2 (entropy): action std GREW 0.62→0.85 over training because
#     ent_coef 0.008 inflated exploration. The nominal path is optimal at
#     residual≈0, so the ~4 cm/step residual noise knocked the socket off the
#     approach and made arrivals unreliable.
#
# v15 fixes:
#   * --max-steps 600 → 2000   (10 bolts × ~159 + retry margin; eval can solve)
#   * --ent-coef 0.008 → 0.0   (let the residual collapse toward 0)
#   * --log-std-init -0.5 → -1.2 (start tight: residual ~1.5 cm, not ~3 cm)
# Everything else mirrors v14 (planner+residual, scripted macro, arrive-angle
# curriculum, bolt order, farm-proof PB reward).
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
echo "[nut] $(date '+%F %T') reusing endposes: $ENDPOSE"
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v15 (horizon+entropy fix) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-b-planner-residual \
  --no-nut-hotstart-curriculum \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.0 \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -1.2 \
  --terminate-on never --max-steps 2000 \
  --total-steps 3500000 --run-name nut_fastening_v15 \
  2>&1 | tee runs/nut_fastening_v15.log

echo "[nut] $(date '+%F %T') DONE"
