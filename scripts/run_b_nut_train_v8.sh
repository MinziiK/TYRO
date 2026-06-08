#!/usr/bin/env bash
# Robot-B nut-fastening training v8 — REWARD REBALANCE (anti dense-farm).
#
# Root cause found in v7 (3 M steps): eval success = 0 across the ENTIRE run
# (ep_len pinned at 600 — never completed the task). The high n_fastened (~5)
# was an artifact of the random-bolt curriculum pre-marking earlier bolts as
# fastened, NOT the policy fastening. The policy had learned to PARK ~12 cm off
# a bolt and FARM the dense reward (clearance + approach kernels ≈ 1.8/step →
# ~1080/episode), which exceeded the value of fastening all 10 bolts (~945).
#
# Fix (all in src/config.py defaults — single variable changed vs v7, the
# curriculum is identical so the reward rebalance is isolated):
#   * Standing exp kernels shrunk (anti-farm):
#       w_nut_reach 3.0→0.5, w_nut_lateral 4.0→1.5, w_nut_align 1.5→0.4,
#       w_nut_axial 2.0→0.5, w_nut_ba_clear 2.0→0.4
#   * Potential-based approach driver promoted (farm-proof): w_pb_nut 8→14
#   * Sparse fasten bonuses boosted so completion dominates:
#       R_arrive 25→40, R_insert 30→60, R_fasten 50→120, R_all_fastened 300→500
#   New balance (per ~600-step episode): parking-farm ~84, fasten-1 ~154,
#   fasten-all ~1890 — fastening even one bolt now beats parking forever.
#   * New honest metric logged: rollout/n_fastened_policy (= fastened by the
#     policy, premark excluded). WATCH THIS, not n_fastened.
#
# Fresh training (reward landscape changed — old value fn learned parking=high).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES=""
# One math thread per worker so the 88 SubprocVecEnv processes don't oversubscribe
# the 96 logical cores. The learner (main process) re-enables 16 torch threads
# internally (see src/train.py) so the PPO update isn't single-threaded.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ENDPOSE="data/nut_mount_endpose.npz"
if [[ ! -f "$ENDPOSE" ]]; then
  echo "[nut] ERROR: missing $ENDPOSE (run endpose extraction first)" >&2
  exit 1
fi
echo "[nut] $(date '+%F %T') reusing endposes: $ENDPOSE"
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening training v8 (reward rebalance) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-hotstart-curriculum \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.0 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 2000000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --eval-freq 250000 --eval-episodes 5 \
  --log-std-init -0.5 \
  --terminate-on never --max-steps 600 \
  --total-steps 3500000 --run-name nut_fastening_v8 \
  2>&1 | tee runs/nut_fastening_v8.log

echo "[nut] $(date '+%F %T') DONE"
