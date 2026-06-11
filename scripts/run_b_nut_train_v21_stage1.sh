#!/usr/bin/env bash
# Robot-B nut-fastening v21 STAGE 1 — fresh per-leg bootstrap.
#
# v21 fix over v20 (user: "전부 체결할 수 있어야 의미가 있지"):
#   * branch-aware INSERT — when the env-driven plunge saturates IK ~1-2 cm
#     short of the seat (edge-of-workspace bolts), it searches for a reachable
#     elbow/wrist branch that DOES seat and switches into it, instead of the
#     leg hanging forever (v20: bolt froze in INSERT → nut_stall, no reward).
#     The bolt is reachable; only the carried-over approach branch was short.
#     Env-side control only (the policy never owned the axial DOF), so the v20
#     policy weights stay valid — but we retrain stage1 fresh so every bolt now
#     earns its fasten credit cleanly (v20 stage1 froze edge bolts too).
#   * Enabled via --nut-v20 (sets nut_b_insert_branch_search=True in train.py).
#
# All v20 reward/gate/servo settings are unchanged.
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
echo "[nut] $(date '+%F %T') === Robot-B nut v21 STAGE 1 (fresh, per-leg) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl --nut-v20 \
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
  --total-steps 4000000 --run-name nut_fastening_v21_stage1 \
  2>&1 | tee runs/nut_fastening_v21_stage1.log

echo "[nut] $(date '+%F %T') DONE"
