#!/usr/bin/env bash
# Robot-B nut-fastening v20 STAGE 1 — fresh per-leg bootstrap.
#
# v20 fixes over v19 (user feedback: "socket must fully envelop the bolt"):
#   * INSERT axial servo — env drives the socket to hub-face base (-L/2)
#     so the nut runner fully wraps the stud (v19 stopped at ~-3 cm).
#   * seat depth tol 2 cm -> 0.7 cm — seated only when truly at base.
#   * joint-movement penalty 0.02 -> 0.06, all phases (approach + insert).
#
# Fresh training (no resume): v19 policy learned the loose seat gate; fine-
# tuning would fight that habit. Stage 2 chain script follows after stage1.
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
echo "[nut] $(date '+%F %T') === Robot-B nut v20 STAGE 1 (fresh, per-leg) ==="

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
  --total-steps 4000000 --run-name nut_fastening_v20_stage1 \
  2>&1 | tee runs/nut_fastening_v20_stage1.log

echo "[nut] $(date '+%F %T') DONE"
