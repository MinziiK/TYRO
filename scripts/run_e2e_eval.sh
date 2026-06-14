#!/usr/bin/env bash
# Headless E2E eval: Robot A (mount DR) + Robot B (nut DR), 100 hub scenarios.
# For v23 pure-RL B stack use scripts/run_e2e_eval_v23.sh instead.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

A_CKPT="${A_CKPT:-runs/phase1_mount_v3_dr/final.zip}"
B_CKPT="${B_CKPT:-runs/nut_fastening_v16_dr/final.zip}"
SCENARIOS="${SCENARIOS:-100}"
DR_CM="${DR_CM:-5.0}"
SEED="${SEED:-42}"

for ck in "$A_CKPT" "$B_CKPT"; do
  if [[ ! -f "$ck" ]]; then
    echo "[e2e] ERROR: missing checkpoint $ck" >&2
    exit 1
  fi
done

LOG="runs/e2e_eval/e2e_${SCENARIOS}sc_${DR_CM}cm.log"
mkdir -p runs/e2e_eval
echo "[e2e] $(date '+%F %T') === E2E eval ${SCENARIOS} scenarios, DR ±${DR_CM} cm ==="

python -u scripts/e2e_eval.py \
  --model-a "$A_CKPT" \
  --model-b "$B_CKPT" \
  --scenarios "$SCENARIOS" \
  --dr-range-cm "$DR_CM" \
  --seed "$SEED" \
  --out-dir runs/e2e_eval \
  2>&1 | tee -a "$LOG"

echo "[e2e] $(date '+%F %T') DONE (log: $LOG)"
