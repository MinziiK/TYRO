#!/usr/bin/env bash
# Watcher: wait until v22 STAGE 1 finishes, then auto-launch STAGE 2.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STAGE1_PAT="src.train.*nut_fastening_v22_stage1"
LOG="runs/chain_v22_stage2.log"
echo "[chain] $(date '+%F %T') watching for v22 stage1 completion..." | tee -a "$LOG"

while pgrep -f "$STAGE1_PAT" >/dev/null 2>&1; do
  sleep 120
done

echo "[chain] $(date '+%F %T') stage1 process gone." | tee -a "$LOG"

CKPT="$(ls runs/nut_fastening_v22_stage1/ckpts/ppo_*_steps.zip 2>/dev/null \
        | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2)"
if [[ -z "${CKPT}" ]]; then
  echo "[chain] $(date '+%F %T') ABORT: no stage-1 checkpoint." | tee -a "$LOG"
  exit 1
fi

STEPS="$(echo "$CKPT" | awk -F'ppo_' '{print $2+0}')"
if (( STEPS < 3000000 )); then
  echo "[chain] $(date '+%F %T') WARNING: stage1 last ckpt only ${STEPS} steps." \
    | tee -a "$LOG"
fi

echo "[chain] $(date '+%F %T') launching stage2 from ${CKPT}" | tee -a "$LOG"
exec bash scripts/run_b_nut_train_v22_stage2.sh
