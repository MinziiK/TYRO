#!/usr/bin/env bash
# Quick Robot-A mount verification under hub DR (±5 cm default).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

A_CKPT="${A_CKPT:-runs/phase1_mount_v3_dr/final.zip}"
SCENARIOS="${SCENARIOS:-20}"
DR_CM="${DR_CM:-5.0}"
SEED="${SEED:-42}"

if [[ ! -f "$A_CKPT" ]]; then
  echo "[verify-a] ERROR: missing $A_CKPT" >&2
  exit 1
fi

echo "[verify-a] $(date '+%F %T') === Mount A DR check: ${SCENARIOS}× ±${DR_CM}cm ==="

python -u scripts/e2e_eval.py \
  --only a \
  --model-a "$A_CKPT" \
  --scenarios "$SCENARIOS" \
  --dr-range-cm "$DR_CM" \
  --seed "$SEED" \
  2>&1 | tee "runs/e2e_eval/verify_mount_a_dr_${SCENARIOS}sc.log"
