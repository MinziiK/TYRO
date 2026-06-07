#!/usr/bin/env bash
# Downstream pipeline starting from an ALREADY-TRAINED A mount checkpoint
# (the ft03 finetune was stopped at convergence; best_model.zip is final).
#   1. Mount precision: old best (0.12-trained) vs ft03 best (0.03-adapted).
#   2. Extract Robot-A mount-completion endposes from the ft03 best model.
#   3. Launch Robot-B insertion-retract nut-fastening training.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

A_CKPT="runs/phase1_mount_v2_ft03/best/best_model.zip"
OLD_BEST="runs/phase1_mount_v2/best/best_model.zip"
ENDPOSE="data/nut_mount_endpose.npz"

if [[ ! -f "$A_CKPT" ]]; then
  echo "[pipe] ERROR: missing $A_CKPT" >&2
  exit 1
fi

echo "[pipe] $(date '+%F %T') === STAGE 1: mount precision (before vs after) ==="
echo "[pipe] --- OLD best (0.12-trained, run @0.03) ---"
python -u -m scripts.eval_mount_precision "$OLD_BEST" \
  --episodes 40 --tag "OLD(phase1_mount_v2)" \
  2>&1 | tee runs/precision_old.log || echo "[pipe] WARN: old precision eval failed"
echo "[pipe] --- NEW best (0.03-adapted finetune) ---"
python -u -m scripts.eval_mount_precision "$A_CKPT" \
  --episodes 40 --tag "NEW(ft03)" \
  2>&1 | tee runs/precision_new.log || echo "[pipe] WARN: new precision eval failed"

echo "[pipe] $(date '+%F %T') === STAGE 2: extract mount endposes ==="
python -u -m scripts.extract_mount_endpose "$A_CKPT" \
  --episodes 30 --out "$ENDPOSE" \
  2>&1 | tee runs/extract_endpose.log
if [[ ! -f "$ENDPOSE" ]]; then
  echo "[pipe] ERROR: endpose extraction produced no $ENDPOSE; aborting." >&2
  exit 1
fi
echo "[pipe] endpose ready: $ENDPOSE"

echo "[pipe] $(date '+%F %T') === STAGE 3: Robot-B nut-fastening training ==="
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
