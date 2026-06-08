#!/usr/bin/env bash
# Launch the nut-fastening GUI preview as a fully detached daemon.
# setsid + background + immediate parent exit reparents python to init (PID 1),
# escaping the agent harness's process-tree cleanup. Logs to $LOG.
#
# Usage: bash scripts/_launch_gui_daemon.sh <mode> [traj_or_model]
MODE="${1:-replay}"
ARG2="${2:-/tmp/nut_traj.npz}"
LOG="/tmp/preview_${MODE}.log"

cd /home/red/nhkweon/TYRO || exit 1
# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tyro
export DISPLAY="${DISPLAY:-:2}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2

EXTRA=(--endpose data/nut_mount_endpose.npz --step-sleep 0.03)
if [[ "$MODE" == "replay" ]]; then
  EXTRA+=(--traj "$ARG2")
elif [[ "$MODE" == "policy" ]]; then
  EXTRA+=(--model "$ARG2" --alpha 0.0 --max-steps 600)
  export OMP_NUM_THREADS=1
fi

rm -f "$LOG"
setsid python -u -m scripts.preview_nut_fastening --mode "$MODE" "${EXTRA[@]}" \
  > "$LOG" 2>&1 < /dev/null &
echo "daemon pid=$! log=$LOG"
# Parent exits immediately; the setsid child reparents to init and persists.
exit 0
