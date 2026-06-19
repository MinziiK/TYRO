#!/usr/bin/env bash
# Headless E2E eval: Robot A (mount DR) + Robot B (nut DR), 100 hub scenarios.
# v24 stack: A=v3_dr, B=B3_1.75M shortest-macro pure-RL.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

A_CKPT="${A_CKPT:-runs/phase1_mount_v3_dr/final.zip}"
B_CKPT="${B_CKPT:-runs/nut_fastening_v24_dr_stageB3/ckpts/ppo_1749440_steps.zip}"
SCENARIOS="${SCENARIOS:-100}"
DR_CM="${DR_CM:-5.0}"
SEED="${SEED:-42}"
V24="${V24:-1}"

for ck in "$A_CKPT" "$B_CKPT"; do
  if [[ ! -f "$ck" ]]; then
    echo "[e2e] ERROR: missing checkpoint $ck" >&2
    exit 1
  fi
done

LOG="runs/e2e_eval/e2e_v24_${SCENARIOS}sc_${DR_CM}cm.log"
mkdir -p runs/e2e_eval
echo "[e2e] $(date '+%F %T') === E2E v24 eval ${SCENARIOS} scenarios, DR ±${DR_CM} cm ==="
echo "[e2e] A=${A_CKPT}  B=${B_CKPT}"

V24_ARGS=()
if [[ "${V24}" != "0" ]]; then
  V24_ARGS=(--v24)
fi

python -u scripts/e2e_eval.py \
  --model-a "$A_CKPT" \
  --model-b "$B_CKPT" \
  --scenarios "$SCENARIOS" \
  --dr-range-cm "$DR_CM" \
  --seed "$SEED" \
  --b-max-steps 2500 \
  --out-dir runs/e2e_eval \
  "${V24_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

echo "[e2e] $(date '+%F %T') DONE (log: $LOG)"
