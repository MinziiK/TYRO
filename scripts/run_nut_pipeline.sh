#!/usr/bin/env bash
# End-to-end orchestrator (run AFTER the A mount finetune has been launched):
#   1. Wait for the phase1_mount_v2_ft03 finetune to finish.
#   2. Compare mount precision: old best (0.12-trained) vs ft03 best (0.03-adapted).
#   3. Extract Robot-A mount-completion endposes from the ft03 best model.
#   4. Launch Robot-B sequential nut-fastening training (uses the fresh endposes).
#
# Idempotent-ish: each stage checks for its own completion marker / output.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

FT_LOG="runs/phase1_mount_v2_ft03.log"
FT_BEST="runs/phase1_mount_v2_ft03/best/best_model.zip"
FT_FINAL="runs/phase1_mount_v2_ft03/final.zip"
OLD_BEST="runs/phase1_mount_v2/best/best_model.zip"
ENDPOSE="data/nut_mount_endpose.npz"

echo "[pipe] $(date '+%F %T') === STAGE 1: wait for finetune ==="
# Wait until the finetune saves final.zip (the train.py finally-block marker).
while true; do
  if [[ -f "$FT_FINAL" ]] && grep -q "saved final to" "$FT_LOG" 2>/dev/null; then
    echo "[pipe] finetune finished (final.zip present)."
    break
  fi
  if ! pgrep -f "src\.train.*phase1_mount_v2_ft03" >/dev/null 2>&1; then
    # Training process gone — give the finally-block a moment to flush.
    sleep 10
    if [[ -f "$FT_FINAL" ]]; then
      echo "[pipe] finetune process exited; final.zip present."
      break
    fi
    echo "[pipe] WARNING: finetune process gone but no final.zip; aborting." >&2
    exit 1
  fi
  sleep 60
done

# Prefer the best checkpoint; fall back to final.zip.
A_CKPT="$FT_BEST"
[[ -f "$A_CKPT" ]] || A_CKPT="$FT_FINAL"
echo "[pipe] A checkpoint for downstream: $A_CKPT"

echo "[pipe] $(date '+%F %T') === STAGE 2: mount precision (before vs after) ==="
echo "[pipe] --- OLD best (0.12-trained, run @0.03) ---"
python -u -m scripts.eval_mount_precision "$OLD_BEST" \
  --episodes 40 --tag "OLD(phase1_mount_v2)" \
  2>&1 | tee runs/precision_old.log || echo "[pipe] WARN: old precision eval failed"
echo "[pipe] --- NEW best (0.03-adapted finetune) ---"
python -u -m scripts.eval_mount_precision "$A_CKPT" \
  --episodes 40 --tag "NEW(ft03)" \
  2>&1 | tee runs/precision_new.log || echo "[pipe] WARN: new precision eval failed"

echo "[pipe] $(date '+%F %T') === STAGE 3: extract mount endposes ==="
python -u -m scripts.extract_mount_endpose "$A_CKPT" \
  --episodes 30 --out "$ENDPOSE" \
  2>&1 | tee runs/extract_endpose.log
if [[ ! -f "$ENDPOSE" ]]; then
  echo "[pipe] ERROR: endpose extraction produced no $ENDPOSE; aborting B training." >&2
  exit 1
fi
echo "[pipe] endpose ready: $ENDPOSE"

echo "[pipe] $(date '+%F %T') === STAGE 4: Robot-B nut-fastening training ==="
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

echo "[pipe] $(date '+%F %T') === PIPELINE DONE ==="
