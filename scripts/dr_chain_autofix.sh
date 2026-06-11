#!/usr/bin/env bash
# Heuristic auto-fix for DR chain failures. Prints one line to stdout:
#   FIXED     — applied a repair; caller should retry the failed stage
#   RETRY     — no code change needed; transient failure, retry as-is
#   WAIT      — upstream not ready (e.g. v15 still running); do nothing yet
#   NEEDS_AGENT — unrecoverable without human/agent code changes
#
# Writes details to runs/dr_chain_autofix.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

FAIL_MARKER="runs/dr_chain.FAILED"
LOG="runs/dr_chain_autofix.log"
V15_FINAL="runs/nut_fastening_v15/final.zip"
V16_FINAL="runs/nut_fastening_v16_dr/final.zip"

log() { echo "[autofix] $(date '+%F %T') $*" | tee -a "$LOG"; }

if [[ ! -f "$FAIL_MARKER" ]]; then
  echo "RETRY"
  exit 0
fi

STAGE="$(grep -oE 'stage=[^ ]+' "$FAIL_MARKER" | head -1 | cut -d= -f2 || true)"
log "=== autofix invoked for stage=${STAGE:-unknown} ==="
cat "$FAIL_MARKER" >> "$LOG"

# Resolve stage log path
case "$STAGE" in
  v16_B_DR) STAGE_LOG="runs/nut_fastening_v16_dr.log" RUN_DIR="runs/nut_fastening_v16_dr" ;;
  v3_A_DR)  STAGE_LOG="runs/phase1_mount_v3_dr.log"  RUN_DIR="runs/phase1_mount_v3_dr" ;;
  *)        STAGE_LOG="" RUN_DIR="" ;;
esac

TAIL=""
if [[ -n "$STAGE_LOG" && -f "$STAGE_LOG" ]]; then
  TAIL="$(tail -n 80 "$STAGE_LOG")"
fi
FULL_TAIL="${TAIL}$(tail -n 40 "$FAIL_MARKER" 2>/dev/null)"

# --- upstream readiness ---------------------------------------------------
if [[ "$STAGE" == "v16_B_DR" && ! -f "$V15_FINAL" ]]; then
  if pgrep -f "run-name nut_fastening_v15" >/dev/null 2>&1; then
    log "v15 still training; WAIT"
    echo "WAIT"
    exit 0
  fi
  log "v15 final.zip missing and v15 not running → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

# --- syntax / import sanity (catches bad patches) -------------------------
if ! python3 -m py_compile src/train.py src/env/tyro_env.py src/config.py 2>>"$LOG"; then
  log "Python syntax error in core modules → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

source ~/anaconda3/etc/profile.d/conda.sh 2>>"$LOG" || true
if ! conda activate tyro 2>>"$LOG"; then
  log "conda env 'tyro' unavailable → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

if ! python -c "import src.train; import src.env.tyro_env" 2>>"$LOG"; then
  log "import check failed → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

# --- log-pattern fixes ----------------------------------------------------
if grep -qiE "ModuleNotFoundError|ImportError" <<< "$FULL_TAIL"; then
  log "ImportError in stage log — env should be OK after conda check; NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

if grep -qiE "argparse.*error|unrecognized arguments" <<< "$FULL_TAIL"; then
  log "CLI flag mismatch → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

if grep -qiE "SyntaxError|IndentationError|NameError|AttributeError|TypeError" <<< "$FULL_TAIL"; then
  log "Python runtime bug in traceback → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

if grep -qiE "missing checkpoint|ERROR: missing" <<< "$FULL_TAIL"; then
  if [[ "$STAGE" == "v16_B_DR" && ! -f "$V15_FINAL" ]]; then
    log "v16 needs v15 final; WAIT"
    echo "WAIT"
    exit 0
  fi
  log "missing checkpoint → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

# OOM / killed — reduce parallel envs in the stage script (one step down)
if grep -qiE "Killed|MemoryError|Cannot allocate memory|std::bad_alloc" <<< "$FULL_TAIL"; then
  case "$STAGE" in
    v16_B_DR)
      SCRIPT="scripts/run_b_nut_dr_finetune_v16.sh"
      OLD="--num-envs 88"
      NEW="--num-envs 64"
      ;;
    v3_A_DR)
      SCRIPT="scripts/run_phase_a_dr_finetune.sh"
      OLD="--num-envs 72"
      NEW="--num-envs 52"
      ;;
    *)
      SCRIPT=""
      ;;
  esac
  if [[ -n "$SCRIPT" && -f "$SCRIPT" ]] && grep -q "$OLD" "$SCRIPT" && ! grep -q "$NEW" "$SCRIPT"; then
    sed -i "s/${OLD}/${NEW}/" "$SCRIPT"
    log "OOM detected: patched $SCRIPT ${OLD} → ${NEW}"
    echo "FIXED"
    exit 0
  fi
  log "OOM but env count already reduced or unknown stage → NEEDS_AGENT"
  echo "NEEDS_AGENT"
  exit 0
fi

# Stale / zombie training for same run-name — kill so resume can start clean
case "$STAGE" in
  v16_B_DR) RUN_NAME="nut_fastening_v16_dr" ;;
  v3_A_DR)  RUN_NAME="phase1_mount_v3_dr" ;;
  *)        RUN_NAME="" ;;
esac
if [[ -n "$RUN_NAME" ]] && pgrep -f "run-name ${RUN_NAME}" >/dev/null 2>&1; then
  log "stale train process for $RUN_NAME still running — sending SIGTERM"
  pkill -TERM -f "run-name ${RUN_NAME}" 2>/dev/null || true
  sleep 10
  pkill -KILL -f "run-name ${RUN_NAME}" 2>/dev/null || true
  log "cleared stale process → FIXED (retry resume)"
  echo "FIXED"
  exit 0
fi

# Segfault / transient — retry from latest checkpoint
if grep -qiE "Segmentation fault|Aborted|core dumped|BrokenPipe|EOFError" <<< "$FULL_TAIL"; then
  log "transient crash signature → RETRY"
  echo "RETRY"
  exit 0
fi

# Default: assume transient, retry once more from checkpoint
if [[ -n "$RUN_DIR" && -d "${RUN_DIR}/ckpts" ]]; then
  latest="$(ls -t "${RUN_DIR}/ckpts/"ppo_*_steps.zip 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" ]]; then
    log "checkpoint exists ($latest); default RETRY"
    echo "RETRY"
    exit 0
  fi
fi

log "no heuristic matched → NEEDS_AGENT"
echo "NEEDS_AGENT"
exit 0
