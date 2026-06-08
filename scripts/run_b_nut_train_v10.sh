#!/usr/bin/env bash
# Robot-B nut-fastening training v10 — HUB-CENTER hot-start + softer ramp.
#
# Direction (user): follow the answer path *approximately* (HOME → hub center →
# bolt0 → bolt5 → … → HOME), exploring joint angles for minimal joint change +
# collision-free, with the existing +Y capture gate doing the fasten trigger.
#
# Changes vs v9:
#   * Hot-start target = bolt-ring CENTER at staging depth (0,-0.21,0), not the
#     per-bolt approach over bolt 0 (src/config.py nut_b_hotstart_hub_center=True).
#     Equidistant 0.21 m to every bolt at fixed Y → symmetric pure-XZ radial
#     reach; removes the bolt-0 bias that floored only bolt 0 as "free".
#   * Hot-start curriculum kept (alpha 1→end), but the start pose is BACKED OFF
#     to HOME more gently and never all the way to 0:
#       - alpha_end 0.0 → 0.25   (retain partial scaffolding; v9 collapsed to
#         n_fastened_policy=0 once alpha hit ~0 because B never learned the full
#         HOME→bolt approach unscaffolded)
#       - ramp 2.0M → 2.5M       (recede toward HOME more slowly)
#   * random bolt start OFF (src/config.py nut_b_hotstart_random_bolt=False) —
#     honest sequential start at bolt 0; no premark inflation.
#   * ent_coef 0.008 kept (exploration for joint-angle search).
#   * w_nut_ba_clear 0.0 kept; w_nut_collision 40 kept (only real-contact penalty).
#
# WATCH: rollout/n_fastened_policy and eval success (NOT n_fastened).
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
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v10 (hub-center hot-start) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-hotstart-curriculum \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.25 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 2500000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.008 \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --terminate-on never --max-steps 600 \
  --total-steps 3500000 --run-name nut_fastening_v10 \
  2>&1 | tee runs/nut_fastening_v10.log

echo "[nut] $(date '+%F %T') DONE"
