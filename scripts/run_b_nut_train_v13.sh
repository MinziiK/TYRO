#!/usr/bin/env bash
# Robot-B nut-fastening training v13 — FARM-PROOF reward redesign.
#
# v1–v12 root cause: standing positive exp kernels (align/path/lateral/reach/
# axial) were farmable — policy parked/retreated and collected per-step income
# without fastening. Whack-a-mole gates could not fix this structurally.
#
# v13 design (user: "no collision + follow the path = correct"):
#   + pb_nut ONLY positive dense: w·(d_prev − d_now) toward staging (w=25)
#   − w_nut_corridor · hub-side Y excursion past staging plane (linear penalty)
#   − collision / workspace / action / jerk / joint_vel / step_alive
#   + sparse R_arrive / R_fasten / R_all_fastened
#   Orientation: nut_b_lock_coaxial (control, not reward)
#   Insert/retract: scripted macro (not learned)
#   Bolt order: (0,5,7,2,3,8,9,4,6,1) — unchanged
#
# Standing kernels ZEROED: w_nut_align/reach/lateral/axial/path = 0
# mix_dense = 0.5 for nut task (config make_env_config)
#
# WATCH: n_fastened_policy + eval success. ep_rew must NOT climb while
# n_fastened_policy stays 0 (farm signature — should be impossible now).
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
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v13 (farm-proof PB) ==="

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
  --total-steps 3500000 --run-name nut_fastening_v13 \
  2>&1 | tee runs/nut_fastening_v13.log

echo "[nut] $(date '+%F %T') DONE"
