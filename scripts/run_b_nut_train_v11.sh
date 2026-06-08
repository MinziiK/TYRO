#!/usr/bin/env bash
# Robot-B nut-fastening training v11 — COAXIAL ORIENTATION LOCK + widened kernels.
#
# Direction (user): "start → bolt 0 만 이동하면 1개는 체결되는데 왜 그걸 학습 못하나."
#
# TWO root causes fixed (src/config.py + tyro_env.py — curriculum/flags as v10):
#
# (1) ORIENTATION DRIFT (the dominant blocker). In the delta path B's tool
#     orientation accumulated the policy's rotation residuals with NO nominal
#     anchor, so it drifted off-axis (v10: theta_B → 46°, far above the 12–35°
#     arrive gate) — the macro could never trigger no matter how good the
#     position got (n_fastened_policy = 0 the whole run). All 10 bolts share the
#     SAME axis (world −Y), so we now HARD-LOCK B's tool orientation coaxial
#     during APPROACH (nut_b_lock_coaxial=True): policy controls XYZ only,
#     theta ≈ 0 every step (smoke: max 5° under adversarial rot actions), and
#     the dead rotation channels are masked from the action/jerk penalty.
#
# (2) DEAD ZONE. The v10 hub-CENTER hot-start places B ~0.21 m off every bolt's
#     axis, but the dense kernels were too sharp to give a gradient there:
#       * nut_coax_gate 0.05 → exp(-0.21/0.05)=0.015 killed axial reach + the
#         (farm-proof) axial PB leg across the whole start→bolt gap.
#       * nut_lateral_decay 0.08 → 1.5·exp(-0.21/0.08)=0.11/step, too weak.
#     Widened (weights UNCHANGED, anti-farm — only decays widen so the gradient
#     reaches farther without raising the standing-kernel peaks):
#       * nut_coax_gate     0.05 → 0.16   (axial reach + axial PB alive at 0.21 m)
#       * nut_lateral_decay 0.08 → 0.16   (lateral pull ~0.40/step at the start)
#       * nut_axial_decay   0.05 → 0.12   (seating gradient extends across gap)
#     The potential-based pb_nut (w=14) remains the farm-proof primary driver.
#
# WATCH: rollout/n_fastened_policy and eval success (NOT n_fastened). If a farm
# plateau re-emerges (ep_len pinned at 600, ep_rew climbing while
# n_fastened_policy stays 0), trim w_nut_path / w_nut_align next.
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
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v11 (widened approach kernels) ==="

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
  --total-steps 3500000 --run-name nut_fastening_v11 \
  2>&1 | tee runs/nut_fastening_v11.log

echo "[nut] $(date '+%F %T') DONE"
