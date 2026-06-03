# Phase 1 — smooth carry / mount (2026-06-04 control law)
#
# Uses the new defaults in src/config.py:
#   - baked Min-Jerk joint trajectory (smooth carry; the 0-51 step carry was
#     always smooth — only the hub insertion snapped)
#   - ur10_motor_max_velocity_rad_s = 1.0  ← KEY: caps the stiff-PD whip
#     through the near-singular hub insertion (worst EE jump 70 -> 25 cm,
#     mean 6.3 -> 3.2 cm). Mounts at the curriculum-start gate (mount@112);
#     the policy residual closes the final ~2 cm as the gate tightens to 4 cm.
#   - DLS Cartesian servo present but OFF (use_dls_cartesian_servo=False):
#     it smooths the motion but cannot seat the tire at 88% reach.
#   - planner_stage1_approach_standoff = 0 (shipping arch geometry kept; the
#     hub-reposition layout was measured then reverted — see README).
#
# NOTE: ur10_motor_max_velocity_rad_s changes the env dynamics, so any
# pre-2026-06-04 checkpoint is out-of-distribution. Train fresh; 500k steps
# is enough for a first usable policy when easy_prob stays at 1.0.
#
# Usage:
#   .\scripts\train_v8_smooth.ps1
#   .\scripts\train_v8_smooth.ps1 --total-steps 300000

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$TYRO_PY = "C:\Users\nhdkweon\miniconda3\envs\tyro\python.exe"
if (-not (Test-Path $TYRO_PY)) {
    Write-Error "tyro python not found at $TYRO_PY"
    exit 1
}

$defaultRun = "phase1_smooth_" + (Get-Date -Format "yyyyMMdd-HHmm")

& $TYRO_PY -m src.train `
    --stage 3 --phase 1 `
    --num-envs 12 `
    --total-steps 500000 `
    --run-name $defaultRun `
    --contact-force-done 0 `
    --start-pos-easy-prob 1.0 `
    --no-start-pos-easy-prob-curriculum `
    @args
