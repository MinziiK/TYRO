#!/usr/bin/env bash
# Watchdog for the DR fine-tune chain.
#
# Monitors:
#   * runs/dr_chain.FAILED        → autofix + resume failed stage
#   * chain process death         → restart resume if work remains
#   * runs/dr_chain.NEEDS_AGENT   → written when autofix cannot recover
#
# Does NOT interfere while v15 is training or the chain is healthy.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

POLL_S=90
LOG="runs/dr_chain_watchdog.log"
FAIL_MARKER="runs/dr_chain.FAILED"
NEEDS_AGENT="runs/dr_chain.NEEDS_AGENT"
COMPLETE="runs/dr_chain.COMPLETE"
V15_FINAL="runs/nut_fastening_v15/final.zip"
V16_FINAL="runs/nut_fastening_v16_dr/final.zip"
A_FINAL="runs/phase1_mount_v3_dr/final.zip"
MAX_AGENT_ROUNDS=8
AGENT_ROUNDS=0

log() { echo "[watchdog] $(date '+%F %T') $*" | tee -a "$LOG"; }

chain_running() {
  pgrep -f "scripts/run_dr_chain.sh" >/dev/null 2>&1 \
    || pgrep -f "scripts/dr_chain_resume.sh" >/dev/null 2>&1
}

train_running() {
  pgrep -f "run-name nut_fastening_v15" >/dev/null 2>&1 \
    || pgrep -f "run-name nut_fastening_v16_dr" >/dev/null 2>&1 \
    || pgrep -f "run-name phase1_mount_v3_dr" >/dev/null 2>&1
}

work_remaining() {
  [[ -f "$COMPLETE" ]] && return 1
  [[ ! -f "$V15_FINAL" ]] && return 0   # waiting for v15 counts as in-progress
  [[ ! -f "$V16_FINAL" ]] && return 0
  [[ ! -f "$A_FINAL" ]] && return 0
  return 1
}

write_needs_agent() {
  local reason="$1"
  {
    echo "NEEDS_AGENT at $(date '+%F %T')"
    echo "reason: $reason"
    echo "agent_round: $AGENT_ROUNDS / $MAX_AGENT_ROUNDS"
    if [[ -f "$FAIL_MARKER" ]]; then
      echo "--- dr_chain.FAILED ---"
      cat "$FAIL_MARKER"
    fi
    echo "--- dr_chain.log (last 30) ---"
    tail -n 30 runs/dr_chain.log 2>/dev/null || true
    echo "--- autofix.log (last 30) ---"
    tail -n 30 runs/dr_chain_autofix.log 2>/dev/null || true
    echo ""
    echo "To recover after patching: bash scripts/dr_chain_resume.sh"
  } > "$NEEDS_AGENT"
  log "Wrote $NEEDS_AGENT ($reason)"
}

handle_failure() {
  local verdict
  verdict="$(bash scripts/dr_chain_autofix.sh | tail -1)"
  log "autofix verdict: $verdict"

  case "$verdict" in
    WAIT)
      log "upstream not ready; will recheck."
      return 0
      ;;
    FIXED|RETRY)
      rm -f "$NEEDS_AGENT"
      log "launching dr_chain_resume.sh in background"
      nohup bash scripts/dr_chain_resume.sh >> runs/dr_chain_resume.nohup.log 2>&1 &
      return 0
      ;;
    NEEDS_AGENT|*)
      AGENT_ROUNDS=$((AGENT_ROUNDS + 1))
      if (( AGENT_ROUNDS >= MAX_AGENT_ROUNDS )); then
        write_needs_agent "autofix exhausted ($MAX_AGENT_ROUNDS rounds): $verdict"
        return 1
      fi
      write_needs_agent "autofix returned $verdict (round $AGENT_ROUNDS)"
      # Still try one resume in case it was a flake
      log "attempting resume despite NEEDS_AGENT (round $AGENT_ROUNDS)"
      nohup bash scripts/dr_chain_resume.sh >> runs/dr_chain_resume.nohup.log 2>&1 &
      return 0
      ;;
  esac
}

log "=== DR chain watchdog started (poll=${POLL_S}s) ==="

while true; do
  if [[ -f "$COMPLETE" ]]; then
    log "chain COMPLETE — watchdog exiting."
    exit 0
  fi

  if [[ -f "$FAIL_MARKER" ]]; then
    if chain_running || train_running; then
      log "FAILED marker present but chain/train active — waiting."
    else
      log "FAILED marker detected — invoking autofix + resume."
      handle_failure || true
    fi
    sleep "$POLL_S"
    continue
  fi

  # Chain died unexpectedly while work remains
  if work_remaining && ! chain_running; then
    if [[ ! -f "$V15_FINAL" ]]; then
      if pgrep -f "run-name nut_fastening_v15" >/dev/null 2>&1; then
        log "v15 training; chain dead — restarting run_dr_chain.sh (wait mode)."
        nohup bash scripts/run_dr_chain.sh >> runs/dr_chain.nohup.log 2>&1 &
      else
        log "v15 not running and no final.zip — NEEDS_AGENT"
        write_needs_agent "v15 training stopped before final.zip"
      fi
    elif ! train_running; then
      log "chain not running, work remains, no active train — restarting resume."
      rm -f "$NEEDS_AGENT"
      nohup bash scripts/dr_chain_resume.sh >> runs/dr_chain_resume.nohup.log 2>&1 &
    fi
  fi

  sleep "$POLL_S"
done
