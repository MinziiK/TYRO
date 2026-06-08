#!/usr/bin/env bash
# Robot-B nut-fastening training v12 — FARM SUPPRESSION (on top of v11).
#
# v11 result: the coaxial orientation lock worked (theta_B ~4-6°, n_fastened_policy
# reached 0.5 at alpha≈1.0 — best ever), but as the hot-start alpha decayed the
# policy COLLAPSED into dense farming: ep_rew climbed to +200 while
# n_fastened_policy fell to 0, then diverged (eval -6900). Two standing kernels
# fed that farm:
#
#   (1) nut_align — with the orientation HARD-LOCKED coaxial, theta_B ≈ 0 every
#       step BY CONSTRUCTION, so ``w_nut_align·exp(-theta/decay)`` paid a constant
#       ~0.4/step (~240/episode) of position-INDEPENDENT income with zero useful
#       gradient (the alignment it rewards is already guaranteed by the lock).
#       → ZEROED: w_nut_align 0.4 → 0.0 (redundant under the lock).
#   (2) nut_path — the constant-Y in-plane bonus pays up to w/step whenever the
#       tool Y is near the staging plane REGARDLESS of XZ, i.e. a parkable
#       plateau the policy farmed by hovering in-plane far from any bolt.
#       → CUT: w_nut_path 0.6 → 0.2 (keeps a gentle corridor bias, no longer
#         out-earns fastening).
#
# Everything else is IDENTICAL to v11 (orientation lock, widened approach
# kernels, alpha/arrive-angle curricula). Only the two anti-farm weights change
# (src/config.py), so the farm-proof pb_nut (w=14) + sparse R_arrive/R_fasten now
# dominate the return.
#
# WATCH: n_fastened_policy must KEEP RISING as alpha decays (it must NOT collapse
# to 0 with ep_rew climbing — that is the farm signature). eval success > 0.
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
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v12 (farm suppression) ==="

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
  --total-steps 3500000 --run-name nut_fastening_v12 \
  2>&1 | tee runs/nut_fastening_v12.log

echo "[nut] $(date '+%F %T') DONE"
