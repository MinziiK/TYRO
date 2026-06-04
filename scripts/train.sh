#!/usr/bin/env bash
# Phase 1 PPO training launcher (Linux / server).
#
# Usage (defaults — 48 envs, 2M steps, auto-timestamped run name):
#   ./scripts/train.sh
#
# Override any flag inline — anything after train.sh is forwarded
# verbatim to `python -m src.train` (last occurrence wins in argparse):
#   ./scripts/train.sh --num-envs 24 --total-steps 500000
#   ./scripts/train.sh --run-name quicktest
#
# Assumes the `tyro` conda env is active (or set TYRO_PY to a python path).
set -euo pipefail

# Python interpreter: honour an explicit TYRO_PY, else use the active python.
TYRO_PY="${TYRO_PY:-python}"

cd "$(dirname "$0")/.."

# Default run name = phase1_<yyyymmdd-HHMM>; override with --run-name.
DEFAULT_RUN="phase1_$(date +%Y%m%d-%H%M)"

# 2026-06-04 — full-cycle run: the 7-state FSM runs
#   pick -> carry -> mount -> tighten-hold -> retract-to-HOME ->
#   re-approach+re-grasp -> loosen-hold -> return-to-rack.
# --terminate-on never keeps the episode alive through the whole loop and
# the long --max-steps horizon gives every leg time to finish.
# --num-envs 48 matches the 48 physical cores on the Xeon Gold 5220R box
# (PyBullet sim is the CPU-bound bottleneck; the MLP policy stays on CPU).
"$TYRO_PY" -m src.train \
    --stage 3 --phase 1 \
    --num-envs 48 \
    --total-steps 2000000 \
    --full-cycle \
    --terminate-on never \
    --max-steps 1200 \
    --mount-hold-steps 40 \
    --loosen-hold-steps 40 \
    --batch-size 512 \
    --run-name "$DEFAULT_RUN" \
    --contact-force-done 0 \
    "$@"
