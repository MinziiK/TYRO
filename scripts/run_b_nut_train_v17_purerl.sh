#!/usr/bin/env bash
# Robot-B nut-fastening training v17 — PURE RL (no planner, no scripted macro).
#
# Why: v14-v16 used a min-jerk planner nominal + PPO residual for APPROACH and a
# scripted insert->hold->retract macro. That solved the task (v15 ~80%) but the
# motion looks robotic: fixed-waypoint transit + near-teleport insert (the user's
# complaint). v17 hands the WHOLE cycle to the policy so the motion is RL-natural.
#
# IMPORTANT context (why this is a fresh, careful design, NOT a v13 rerun):
#   * v13 was the last pure-RL attempt and got 0% success. Root causes:
#       1) --max-steps 600  → 10 bolts need ~1600+ steps; success was impossible
#          by horizon (same artifact that pinned v14 at 0%). v15 fixed it at 2000.
#       2) NO dense reward/obs for the insert — RL never even had a gradient to
#          learn the in/out (insert was always the scripted macro from v1).
#   v17 fixes BOTH: horizon 2000 + a per-leg axial potential reward + extended
#   obs (axial depth / lateral / subphase / stage / axial-error). See
#   --nut-pure-rl wiring in config.make_env_config / tyro_env.
#
# v17 design:
#   * --nut-pure-rl  → macro OFF, planner OFF, coaxial lock ON (3-DOF XYZ),
#                      obs 99→104 (12-d task block), per-leg watchdog disabled.
#   * HYBRID motion: APPROACH/transit = free 3-DOF RL; INSERT/RETRACT (subphase
#     1) = bolt-axis only (±Y plunge/retract). Bolt order enforced by FSM.
#     Collision = soft penalty (w_nut_collision=40), episode survives contact.
#   * Hot-start curriculum ON (alpha 1->0): start B already near a bolt's
#     staging so it samples arrive+insert reward early, then wean off.
#   * Random-bolt premark ON: mark earlier bolts fastened and hot-start at a
#     random position in nut_bolt_order so every transition gets training mass
#     (avoids the frontier where only bolt-0 approach is ever practiced).
#   * Arrive-angle curriculum 35->12 deg (same as v15).
#   * Arrive-position curriculum 12->8 cm: insert triggers from a generous
#     staging region early, then the axial capture sphere tightens. (Lateral
#     coaxiality is fixed by seat physics, so only the axial capture is ramped.)
#   * Per-leg episodes (auto with --nut-pure-rl): terminate after ONE bolt so
#     each reset = random-bolt hot-start + approach+insert only — no failed
#     transit farming. v17 attempt-1 collapsed at alpha~0.80 because insert-only
#     skill (n_fastened_policy~0.5) couldn't survive longer approach spawns.
#   * Horizon 800 (one approach+insert cycle; was 2000 which wasted ~1800 idle
#     steps after the spawn-bolt insert and masked the transit gap).
#   * Alpha floor 0.3 (not 0.0): don't spawn at full HOME until approach is
#     solid across bolts; ramp slowed to 3.5M steps.
#   * ent_coef small (0.003), log_std_init -1.0.
#   * NO DR yet — get the nominal pure-RL policy solid first (a v18 DR fine-tune
#     can follow, mirroring the v15->v16 step).
#
# WATCH: n_fastened_policy must climb WITH ep_rew (no farm); success_rate at 2000.
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
echo "[nut] $(date '+%F %T') === Robot-B nut-fastening v17 (PURE RL, no planner/macro) ==="

python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --nut-fastening --nut-hold-steps 6 \
  --nut-pure-rl \
  --nut-hotstart-curriculum \
  --nut-hotstart-random-bolt \
  --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.3 \
  --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 3500000 \
  --nut-arrive-ang-curriculum \
  --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 12 \
  --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
  --nut-arrive-pos-curriculum \
  --nut-arrive-pos-start-cm 12 --nut-arrive-pos-end-cm 8 \
  --nut-arrive-pos-hold-steps 400000 --nut-arrive-pos-ramp-steps 2000000 \
  --num-envs 88 --n-steps 279 --batch-size 1024 \
  --device cpu \
  --ent-coef 0.003 \
  --eval-freq 250000 --eval-episodes 10 \
  --log-std-init -1.0 \
  --terminate-on never --max-steps 800 \
  --total-steps 4000000 --run-name nut_fastening_v17_purerl \
  2>&1 | tee runs/nut_fastening_v17_purerl.log

echo "[nut] $(date '+%F %T') DONE"
