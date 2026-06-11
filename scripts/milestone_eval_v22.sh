#!/usr/bin/env bash
# Milestone-based POLICY evaluation for the v22 run (not process babysitting).
# Waits until training crosses each milestone step, then loads the latest
# checkpoint and runs scripts/eval_v22_milestone.py, appending a verdict to
# runs/eval_v22_report.log so we can judge "is it learning + OK to continue".
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
REPORT="runs/eval_v22_report.log"

# Stage1 milestones (per-leg). Stage2 milestones tagged separately.
S1_MILES=(500000 1000000 2000000 3000000 3990000)
S2_MILES=(500000 1000000 2000000 2990000)

latest_ckpt() {
  ls "$1"/ckpts/ppo_*_steps.zip 2>/dev/null \
    | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1
}

ckpt_steps() { grep -a 'total_timesteps' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | tail -1; }

log() { echo "[mile-eval] $(date '+%F %T') $*" | tee -a "$REPORT"; }

run_eval() {
  local rundir="$1" tag="$2" skipchain="$3"
  local row ck
  row="$(latest_ckpt "$rundir")"
  ck="$(echo "$row" | cut -d' ' -f2-)"
  if [[ -z "$ck" ]]; then log "$tag: no ckpt yet"; return; fi
  log "=== $tag eval on $(basename "$ck") ==="
  python -u scripts/eval_v22_milestone.py --model "$ck" $skipchain 2>/dev/null \
    | grep -aE "RESULT|per_bolt_seat|branch_search|chain_from_home" | tee -a "$REPORT"
}

log "milestone evaluator started"

# ---- STAGE 1 ----
for M in "${S1_MILES[@]}"; do
  while pgrep -f "src.train.*nut_fastening_v22_stage1" >/dev/null 2>&1; do
    S="$(ckpt_steps runs/nut_fastening_v22_stage1.log)"
    [[ -n "$S" && "$S" -ge "$M" ]] && break
    sleep 120
  done
  # stage1 may have finished; eval whatever the latest ckpt is for this milestone
  run_eval runs/nut_fastening_v22_stage1 "S1@${M}" "--skip-chain"
  pgrep -f "src.train.*nut_fastening_v22_stage1" >/dev/null 2>&1 || break
done

log "stage1 milestones done; waiting for stage2 to start..."
for _ in $(seq 1 120); do
  pgrep -f "src.train.*nut_fastening_v22_stage2" >/dev/null 2>&1 && break
  [[ -f runs/nut_fastening_v22_stage2/final.zip ]] && break
  sleep 60
done

# ---- STAGE 2 ----
for M in "${S2_MILES[@]}"; do
  while pgrep -f "src.train.*nut_fastening_v22_stage2" >/dev/null 2>&1; do
    S="$(ckpt_steps runs/nut_fastening_v22_stage2.log)"
    [[ -n "$S" && "$S" -ge "$M" ]] && break
    sleep 120
  done
  run_eval runs/nut_fastening_v22_stage2 "S2@${M}" ""
  pgrep -f "src.train.*nut_fastening_v22_stage2" >/dev/null 2>&1 \
    || { [[ -f runs/nut_fastening_v22_stage2/final.zip ]] && break; }
done

# Final eval on stage2 final.
if [[ -f runs/nut_fastening_v22_stage2/final.zip ]]; then
  log "=== S2 FINAL eval ==="
  python -u scripts/eval_v22_milestone.py \
    --model runs/nut_fastening_v22_stage2/final.zip 2>/dev/null \
    | grep -aE "RESULT|per_bolt_seat|branch_search|chain_from_home" | tee -a "$REPORT"
fi
log "milestone evaluator finished"
