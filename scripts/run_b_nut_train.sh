#!/usr/bin/env bash
# Extract ft03 mount endposes then launch Robot-B nut-fastening training.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

A_CKPT="runs/phase1_mount_v2_ft03/best/best_model.zip"
ENDPOSE="data/nut_mount_endpose.npz"

if [[ ! -f "$A_CKPT" ]]; then
  echo "[nut] ERROR: missing $A_CKPT" >&2
  exit 1
fi

echo "[nut] $(date '+%F %T') === extract endposes (15 ep) ==="
python -u -m scripts.extract_mount_endpose "$A_CKPT" \
  --episodes 15 --out "$ENDPOSE" \
  2>&1 | tee runs/extract_endpose.log

echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training ==="
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

echo "[nut] $(date '+%F %T') DONE"
