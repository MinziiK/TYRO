#!/usr/bin/env bash
# Periodic health monitor for v22 nut training pipeline.
# - Ensures stage1 (or stage2) train process is alive; restarts from latest ckpt if dead.
# - Ensures chain watcher is alive while stage1 runs.
# - Logs anomalies (fps drop, errors) to runs/monitor_v22.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="runs/monitor_v22.log"
INTERVAL=300  # 5 min

log() { echo "[monitor] $(date '+%F %T') $*" | tee -a "$LOG"; }

latest_ckpt() {
  local dir="$1"
  ls "${dir}"/ckpts/ppo_*_steps.zip 2>/dev/null \
    | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2-
}

ensure_chain_watcher() {
  if pgrep -f "chain_v22_stage2_after_stage1" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -f "src.train.*nut_fastening_v22_stage1" >/dev/null 2>&1; then
    log "chain watcher missing — relaunching"
    setsid bash scripts/chain_v22_stage2_after_stage1.sh >> runs/chain_v22_stage2.log 2>&1 &
    disown
  fi
}

restart_stage1() {
  local ckpt
  ckpt="$(latest_ckpt runs/nut_fastening_v22_stage1)"
  if [[ -n "${ckpt}" ]]; then
    log "RESTART stage1 from ckpt ${ckpt}"
    # Patch resume into a one-off relaunch (stage1 script is fresh-only).
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate tyro
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    setsid bash -c "
      cd '$REPO' && source ~/anaconda3/etc/profile.d/conda.sh && conda activate tyro &&
      python -u -m src.train \
        --stage 3 --phase 1 --scene-layout fanuc_spacious \
        --nut-fastening --nut-hold-steps 6 --nut-pure-rl --nut-v20 \
        --nut-hotstart-curriculum --nut-hotstart-random-bolt \
        --nut-hotstart-alpha-start 1.0 --nut-hotstart-alpha-end 0.3 \
        --nut-hotstart-hold-steps 400000 --nut-hotstart-ramp-steps 3500000 \
        --nut-arrive-ang-curriculum \
        --nut-arrive-ang-start-deg 35 --nut-arrive-ang-end-deg 5 \
        --nut-arrive-ang-hold-steps 400000 --nut-arrive-ang-ramp-steps 2000000 \
        --nut-arrive-pos-curriculum \
        --nut-arrive-pos-start-cm 12 --nut-arrive-pos-end-cm 8 \
        --nut-arrive-pos-hold-steps 400000 --nut-arrive-pos-ramp-steps 2000000 \
        --num-envs 88 --n-steps 279 --batch-size 1024 --device cpu \
        --ent-coef 0.003 --eval-freq 250000 --eval-episodes 30 \
        --log-std-init -1.0 --terminate-on never --max-steps 800 \
        --resume '${ckpt}' --resume-mode full \
        --total-steps 4000000 --run-name nut_fastening_v22_stage1 \
        >> runs/nut_fastening_v22_stage1.log 2>&1
    " &
    disown
  else
    log "RESTART stage1 fresh (no ckpt)"
    setsid bash scripts/run_b_nut_train_v22_stage1.sh >> runs/nut_fastening_v22_stage1.log 2>&1 &
    disown
  fi
}

log "monitor started (interval=${INTERVAL}s)"

while true; do
  # --- stage1 running? ---
  if pgrep -f "src.train.*nut_fastening_v22_stage1" >/dev/null 2>&1; then
    ensure_chain_watcher
    steps="$(grep -a 'total_timesteps' runs/nut_fastening_v22_stage1.log 2>/dev/null | tail -1 \
             | grep -oE '[0-9]+' | tail -1)"
    fps="$(grep -a '|    fps' runs/nut_fastening_v22_stage1.log 2>/dev/null | tail -1 \
           | grep -oE '[0-9.]+' | tail -1)"
  elif pgrep -f "src.train.*nut_fastening_v22_stage2" >/dev/null 2>&1; then
    steps="$(grep -a 'total_timesteps' runs/nut_fastening_v22_stage2.log 2>/dev/null | tail -1 \
             | grep -oE '[0-9]+' | tail -1)"
    fps="$(grep -a '|    fps' runs/nut_fastening_v22_stage2.log 2>/dev/null | tail -1 \
           | grep -oE '[0-9.]+' | tail -1)"
    log "stage2 active  steps=${steps:-?}  fps=${fps:-?}"
  else
  # Neither stage running — check if we're done or need restart
    if [[ -f runs/nut_fastening_v22_stage2/final.zip ]]; then
      log "ALL DONE — stage2 final exists"
      break
    elif [[ -f runs/nut_fastening_v22_stage1/final.zip ]] \
         && ! pgrep -f "chain_v22_stage2_after_stage1" >/dev/null 2>&1; then
      log "stage1 done but stage2 not started — launching chain/stage2"
      setsid bash scripts/run_b_nut_train_v22_stage2.sh >> runs/nut_fastening_v22_stage2.log 2>&1 &
      disown
    else
      log "WARNING: no train process — attempting stage1 restart"
      restart_stage1
      ensure_chain_watcher
    fi
  fi

  # --- error scan ---
  if grep -aqE "Traceback|MemoryError|Killed" runs/nut_fastening_v22_stage1.log 2>/dev/null; then
    log "ERROR detected in stage1 log (see tail)"
  fi

  sleep "$INTERVAL"
done

log "monitor exiting"
