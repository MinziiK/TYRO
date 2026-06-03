# Phase 1 training with FANUC R-2000iC as Robot A (experimental).
# Prerequisites: poc_fanuc_urdf.py --fetch --convert
# Default UR10 training remains scripts/train_v8_smooth.ps1

$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..").Path
$run = "phase1_fanuc_poc_v1"

python -m src.train `
  --run-name $run `
  --stage 1 --phase 1 `
  --robot-a-kind fanuc_r2000ic `
  --total-timesteps 500000
