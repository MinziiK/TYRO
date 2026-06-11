#!/usr/bin/env bash
# Resume the DR chain from a failed or interrupted stage.
# Reads runs/dr_chain.FAILED if present; otherwise infers from artifacts.
#
# Usage:
#   bash scripts/dr_chain_resume.sh              # auto-detect
#   bash scripts/dr_chain_resume.sh v16_B_DR   # force stage
#   bash scripts/dr_chain_resume.sh v3_A_DR
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

V15_FINAL="runs/nut_fastening_v15/final.zip"
V16_FINAL="runs/nut_fastening_v16_dr/final.zip"
A_FINAL="runs/phase1_mount_v3_dr/final.zip"
FAIL_MARKER="runs/dr_chain.FAILED"
LOG="runs/dr_chain.log"
MAX_RETRIES=3
RETRY_BACKOFF_S=30

log() { echo "[chain] $(date '+%F %T') $*" | tee -a "$LOG"; }

run_stage() {
  local name="$1" script="$2" run_dir="$3"
  local attempt=1 rc=0
  while (( attempt <= MAX_RETRIES )); do
    local latest
    latest="$(ls -t "${run_dir}/ckpts/"ppo_*_steps.zip 2>/dev/null | head -1 || true)"
    if [[ -n "$latest" ]]; then
      export STAGE_RESUME_CKPT="$latest"
      log "resume stage '$name' attempt $attempt/$MAX_RETRIES (ckpt $latest)."
    else
      unset STAGE_RESUME_CKPT
      log "resume stage '$name' attempt $attempt/$MAX_RETRIES (fresh)."
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
  {
    echo "FAILED stage=$name exit=$rc at $(date '+%F %T')"
    echo "--- last 40 log lines (${run_dir}.log) ---"
    tail -n 40 "${run_dir}.log" 2>/dev/null || echo "(no log)"
  } > "$FAIL_MARKER"
  log "STAGE '$name' exhausted retries on resume. Wrote $FAIL_MARKER."
  return "$rc"
}

FORCE_STAGE="${1:-}"

if [[ -z "$FORCE_STAGE" && -f "$FAIL_MARKER" ]]; then
  FORCE_STAGE="$(grep -oE 'stage=[^ ]+' "$FAIL_MARKER" | head -1 | cut -d= -f2 || true)"
fi

if [[ -z "$FORCE_STAGE" ]]; then
  if [[ ! -f "$V15_FINAL" ]]; then
    log "resume: v15 not done — nothing to resume yet."
    exit 0
  fi
  if [[ ! -f "$V16_FINAL" ]]; then
    FORCE_STAGE="v16_B_DR"
  elif [[ ! -f "$A_FINAL" ]]; then
    FORCE_STAGE="v3_A_DR"
  else
    log "resume: all stages complete."
    exit 0
  fi
fi

log "=== DR chain RESUME from stage=$FORCE_STAGE ==="
rm -f "$FAIL_MARKER"

if [[ ! -f "$V15_FINAL" ]]; then
  log "resume: waiting for $V15_FINAL"
  exit 1
fi

case "$FORCE_STAGE" in
  v16_B_DR)
    run_stage "v16_B_DR" "scripts/run_b_nut_dr_finetune_v16.sh" "runs/nut_fastening_v16_dr" || exit $?
    run_stage "v3_A_DR" "scripts/run_phase_a_dr_finetune.sh" "runs/phase1_mount_v3_dr" || exit $?
    ;;
  v3_A_DR)
    run_stage "v3_A_DR" "scripts/run_phase_a_dr_finetune.sh" "runs/phase1_mount_v3_dr" || exit $?
    ;;
  *)
    log "resume: unknown stage '$FORCE_STAGE'"
    exit 1
    ;;
esac

log "=== DR chain RESUME COMPLETE ==="
echo "COMPLETE at $(date '+%F %T')" > runs/dr_chain.COMPLETE
