#!/usr/bin/env bash
# Watcher: wait until v21 STAGE 1 finishes, then auto-launch STAGE 2.
#
# "Finished" = the stage-1 training python process (matched by run-name) is no
# longer running AND a stage-1 checkpoint exists. Guards against launching
# stage 2 if stage 1 died before producing any checkpoint.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STAGE1_PAT="src.train.*nut_fastening_v21_stage1"
LOG="runs/chain_v21_stage2.log"
echo "[chain] $(date '+%F %T') watching for v21 stage1 completion..." | tee -a "$LOG"

# Poll until no stage-1 training process remains.
while pgrep -f "$STAGE1_PAT" >/dev/null 2>&1; do
  sleep 120
done

echo "[chain] $(date '+%F %T') stage1 process gone." | tee -a "$LOG"

CKPT="$(ls runs/nut_fastening_v21_stage1/ckpts/ppo_*_steps.zip 2>/dev/null \
        | awk -F'ppo_' '{print $2+0, $0}' | sort -n | tail -1 | cut -d' ' -f2)"
if [[ -z "${CKPT}" ]]; then
  echo "[chain] $(date '+%F %T') ABORT: no stage-1 checkpoint — stage1 likely failed." \
    | tee -a "$LOG"
  exit 1
fi

# Sanity: require stage1 to have reached a reasonable step count before chaining.
STEPS="$(echo "$CKPT" | awk -F'ppo_' '{print $2+0}')"
if (( STEPS < 3000000 )); then
  echo "[chain] $(date '+%F %T') WARNING: stage1 last ckpt only ${STEPS} steps " \
       "(<3M). Launching stage2 anyway from ${CKPT}." | tee -a "$LOG"
fi

echo "[chain] $(date '+%F %T') launching stage2 from ${CKPT}" | tee -a "$LOG"
exec bash scripts/run_b_nut_train_v21_stage2.sh
