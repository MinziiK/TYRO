# Phase 1 PPO training launcher (PowerShell).
#
# Usage (defaults — 12 envs, 2M steps, auto-timestamped run name):
#   .\scripts\train.ps1
#
# Override any flag inline — anything after ``train.ps1`` is forwarded
# verbatim to ``python -m src.train``:
#   .\scripts\train.ps1 --total-steps 500000
#   .\scripts\train.ps1 --num-envs 8 --run-name quicktest
#
# Defaults can be replaced wholesale by passing the same flag — argparse
# treats the *last* occurrence as authoritative, so the user's extra
# flags always win over the baked-in defaults.

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$TYRO_PY = "C:\Users\nhdkweon\miniconda3\envs\tyro\python.exe"
if (-not (Test-Path $TYRO_PY)) {
    Write-Error "tyro python not found at $TYRO_PY"
    exit 1
}

# Default run name = phase1_<yyyymmdd-HHmm>; override with --run-name.
$defaultRun = "phase1_" + (Get-Date -Format "yyyyMMdd-HHmm")

& $TYRO_PY -m src.train `
    --stage 3 --phase 1 `
    --num-envs 12 `
    --total-steps 2000000 `
    --run-name $defaultRun `
    --contact-force-done 0 `
    @args
