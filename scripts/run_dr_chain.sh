#!/usr/bin/env bash
# DR fine-tune chain: wait for v15 -> run B DR (v16) -> run A DR (v3_dr),
# with automatic crash-recovery (resume from the latest checkpoint).
#
# Sequencing:
#   * v15 (Robot-B nominal) must finish first: v16 resumes its final.zip.
#   * B then A, one heavy CPU job at a time (both --device cpu, 72-88 envs).
#
# AUTO-RECOVERY (transient failures only):
#   Each stage is wrapped in run_stage(), which retries up to MAX_RETRIES. On a
#   retry it points STAGE_RESUME_CKPT at the run's newest ckpts/ppo_*_steps.zip,
#   so the stage script continues the SAME run (full resume, keep step counter)
#   instead of restarting from zero. This recovers from segfaults / transient
#   OOM / physics hiccups WITHOUT losing training progress.
#
#   What this CANNOT fix: genuine code/config bugs (same crash every retry).
#   When a stage exhausts its retries the chain writes runs/dr_chain.FAILED
#   with the stage name + last log tail and stops, so an operator/agent can
#   diagnose, patch, and re-launch (the patched stage will itself resume from
#   the latest checkpoint via the same STAGE_RESUME_CKPT path).
#
# Completion signal for v15 = runs/nut_fastening_v15/final.zip (written by
# src.train in a `finally` block).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

V15_FINAL="runs/nut_fastening_v15/final.zip"
LOG="runs/dr_chain.log"
FAIL_MARKER="runs/dr_chain.FAILED"
COMPLETE_MARKER="runs/dr_chain.COMPLETE"
MAX_RETRIES=3
RETRY_BACKOFF_S=30

log() { echo "[chain] $(date '+%F %T') $*" | tee -a "$LOG"; }

fail_stop() {
  # $1 = stage name, $2 = run dir (for log tail), $3 = exit code
  local name="$1" run_dir="$2" rc="$3"
  {
    echo "FAILED stage=$name exit=$rc at $(date '+%F %T')"
    echo "--- last 40 log lines ($run_dir.log) ---"
    tail -n 40 "${run_dir}.log" 2>/dev/null || echo "(no log)"
  } > "$FAIL_MARKER"
  log "STAGE '$name' exhausted $MAX_RETRIES retries (exit $rc). Wrote $FAIL_MARKER. Stopping chain."
}

# run_stage <name> <script> <run_dir>
# run_dir is the run directory (so <run_dir>/ckpts/ holds checkpoints and
# <run_dir>.log is the stage log).
run_stage() {
  local name="$1" script="$2" run_dir="$3"
  local attempt=1 rc=0
  while (( attempt <= MAX_RETRIES )); do
    if (( attempt == 1 )); then
      log "stage '$name' attempt $attempt/$MAX_RETRIES (fresh)."
      unset STAGE_RESUME_CKPT
    else
      local latest
      latest="$(ls -t "${run_dir}/ckpts/"ppo_*_steps.zip 2>/dev/null | head -1 || true)"
      if [[ -n "$latest" ]]; then
        export STAGE_RESUME_CKPT="$latest"
        log "stage '$name' attempt $attempt/$MAX_RETRIES (resume $latest)."
      else
        unset STAGE_RESUME_CKPT
        log "stage '$name' attempt $attempt/$MAX_RETRIES (no ckpt found; fresh)."
      fi
    fi

    if bash "$script"; then
      unset STAGE_RESUME_CKPT
      log "stage '$name' DONE on attempt $attempt."
      return 0
    fi
    rc=$?
    unset STAGE_RESUME_CKPT
    log "stage '$name' attempt $attempt FAILED (exit $rc)."
    (( attempt++ ))
    (( attempt <= MAX_RETRIES )) && sleep "$RETRY_BACKOFF_S"
  done
  fail_stop "$name" "$run_dir" "$rc"
  return "$rc"
}

# Clear any stale failure/completion markers from a previous run.
rm -f "$FAIL_MARKER" "$COMPLETE_MARKER"
log "=== DR fine-tune chain armed (auto-recovery on, max ${MAX_RETRIES} retries/stage). Waiting for $V15_FINAL ==="

# 1. Wait for v15 to finish.
while [[ ! -f "$V15_FINAL" ]]; do
  sleep 60
done
log "v15 final.zip detected."

# 2. Robot-B DR fine-tune (v16).
run_stage "v16_B_DR" "scripts/run_b_nut_dr_finetune_v16.sh" "runs/nut_fastening_v16_dr" || exit $?

# 3. Robot-A mount DR fine-tune (v3_dr).
run_stage "v3_A_DR" "scripts/run_phase_a_dr_finetune.sh" "runs/phase1_mount_v3_dr" || exit $?

log "=== DR fine-tune chain COMPLETE (B v16 + A v3_dr) ==="
echo "COMPLETE at $(date '+%F %T')" > "$COMPLETE_MARKER"
