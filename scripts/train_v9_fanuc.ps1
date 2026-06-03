# Phase 1 training: FANUC(A) + wheel gripper + UR10e(B) spacious layout + 100 kg tire.
# Prerequisites:
#   python scripts/poc_fanuc_urdf.py --fetch --convert
#   python scripts/fetch_ur10e.py
#   python scripts/generate_wheel_gripper_urdf.py
#   python scripts/merge_fanuc_wheeltool.py
#   python scripts/measure_fanuc_home.py --layout fanuc_spacious
#   python scripts/replay_fanuc_scene.py --render --home-start
#
# Default UR10 shipping training remains scripts/train_v8_smooth.ps1

$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..").Path
$run = "phase1_fanuc_wheel_100kg_v1"

python -m src.train `
  --run-name $run `
  --stage 1 --phase 1 `
  --robot-a-kind fanuc_r2000ic `
  --robot-b-kind ur10e `
  --scene-layout fanuc_spacious `
  --tire-mass 100 `
  --total-timesteps 500000
