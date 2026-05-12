# Tyro Team Project

Dual-arm tyre / bolt alignment environment (UR10 + Franka Panda) with PPO
training, evaluation, and hyperparameter sweep scripts. All commands below
assume the **`tyro` conda environment is activated** (`conda activate tyro`).

- Observation: 89-d   Action: 13-d (Δpose A, Δpose B, gripper bit)
- Reward: align + reach + cooperation + sync_joint_A + success + penalties (curriculum-gated)
- Simulator: PyBullet, 240 Hz physics, 20 Hz control

## Installation (`tyro` env)

```bash
git clone https://github.com/MinziiK/TYRO.git
cd TYRO

conda create -n tyro python=3.10 -y
conda activate tyro

# On Windows etc., if pip-built pybullet fails, prefer conda-forge:
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
```

## Project Layout

```
src/
  config.py             # Reward / action / curriculum / env config (single source of truth)
  train.py              # PPO training entry
  eval.py               # Load a checkpoint and run N episodes (GUI optional)
  sweep.py              # Optuna hyperparameter sweep (PPO + reward weights)
  test.py               # Quick env / GUI sanity run
  env/
    tyro_env.py         # Gymnasium env (reset / step / reward / termination)
    scene.py            # Wheel station (URDF) + tyre primitives
    models.py           # Mesh-free tyre + wheel disk builders
    robots.py           # UR10 / Panda wrappers (IK, EE pose, joint state)
    rewards.py          # All reward terms + aggregate()
    utils.py            # Quaternion / axis helpers
scripts/
  check_shaping_and_bolts.py        # Verifies shaping path + bolt randomization
  check_alignment.py                # Gripper-bolt static alignment check
  check_cargo_reachability.py       # Per-bolt reachability sweep
  generate_truck_wheel_station_urdf.py
data/urdf/
  truck_assembly/truck_wheel_station.urdf   # Hub + 10 bolts + brake assembly
  ur10_robot/...                            # UR10 description + Robotiq gripper
```

---

## Step 1 — Environment and Hardware Assets

- **UR10**
  - `data/urdf/ur10_robot/ur10_robot.urdf`
  - Meshes reference `ur_description/meshes/ur10/visual/*.dae` etc. via **paths relative to the URDF directory** (PyBullet search path: `data/urdf/ur10_robot`). The ROS `package://` prefix is not used in this repo.

- **Truck wheel station URDF** (ISO 335 mm PCD · **10 bolts** · hub ≈ ø420 mm; bolts on local +Z, world placement via `hub_base_rpy`) — regenerate with:
  ```bash
  python scripts/generate_truck_wheel_station_urdf.py \
      --bolt-collision-radius-factor 0.95 \
      --bolt-pattern-phase-deg 0 \
      --output data/urdf/truck_assembly/truck_wheel_station.urdf
  ```
  The visual radius is preserved; only the **collision cylinder is shrunk (~5% gap)** for interference margin. The same script can include a **hub pilot boss** (wheel center bore slides over it) and **simple brake rotor / caliper collision proxies** on the **−Z** side of the flange by default (disable with `--no-hub-pilot`, `--no-brake-proxy`, `--hub-pilot-radius`, etc.).

- **Cargo (wheel-well proxy)**
  - `EnvConfig.spawn_vehicle_primitive_box=True` (alias: `spawn_cargo_box`)
  - Service layout: world center `vehicle_center_world`, half-extents `vehicle_half_extents` (+X depth from hub / fender height; see `EnvConfig` comments). The cargo follows the curriculum-induced hub offset.

  Scene construction picks URDF vs. legacy primitives via `use_truck_hub_urdf` in `src/env/scene.py`.

- **Tyre + wheel (single-body approximation)**
  `src/env/models.py:create_tire_wheel_multibody` builds the assembly **without meshes**, using primitives only. The **black tyre tread** is a toroidal ring of boxes (`tire_annulus_collision_segments`, `tire_inner_radius` defaults to 0.282 m). The **silver wheel disk**, when `tire_wheel_disk_style="three_piece"` (default), approximates each lug hole with **three silver boxes per hole** (2 sidewalls + 1 inner sill). Because Bullet caps a body's visual array at **~16 shapes**, the wheel disk is split into **fixed child links**. The legacy `inter_lug_wedge` style packs 10 radial wedges into a single compound. PCD / phase / hole radius are aligned with the URDF generator via `bolt_circle_radius`, `wheel_disk_bolt_phase_rad`, `wheel_disk_bolt_hole_radius`.

- **Cargo wheel-well cutout**
  When enabled, the cargo AABB is subdivided into cells, and cells inside a `+world X`-aligned cylinder through the hub center are removed from both collision and visuals to make room for the tyre (`cargo_use_wheel_well_cutout`, `cargo_wheel_well_radius_yz`, `cargo_wheel_well_x_range_from_hub`, `cargo_collision_subdiv`).

---

## Step 2 — Frames and Reward

- **Hub root pose**: studs are along **local +Z** in URDF. The default `hub_base_rpy` (**pitch = −π/2**) rotates this axis to **world −X** (the side where Robot A/B bases sit). If bolts point up/sideways in sim, flip the sign to **`+π/2`** — the scene rotation lives in **`src/config.py`'s `hub_base_rpy` / `hub_axis_world`**, not in the URDF generator.
- Panda EE and bolt target use **`getLinkState`'s world link frame (index 4 — position, 5 — orientation)** to match the robot EE convention.

Reward terms (see `src/env/rewards.py` + `TyroEnv`):

| Term | Description | Active at stage |
|---|---|---|
| `align_A` / `reach_B` | Tyre-hub / gripper-bolt distance + axis angle (`angle_between`) | ≥ 1 |
| `coop` | `w_c · exp(-α d_A) · exp(-β d_B)` (multiplicative coupling) | ≥ 2 |
| `sync_joint_A` | Robot A joint-velocity-magnitude penalty → frees B for precise work | ≥ 2 |
| `success` | Sparse bonus (`R_success`) when all four thresholds are met | ≥ 3 |
| `collision` / `workspace` / `action` / `jerk` | Collision · out-of-workspace · action-norm · jerk penalties | ≥ 3 |
| `shape_A` / `shape_B` | Potential shaping: `w · (d_{t-1} − d_t)` | 4 (replaces dense) |

- The **89-d observation** ends with 3 alignment-error scalars: hub-normal separation, lateral offset, and a normalized lug angular error (`wheel_disk` vs. `bolt_0` reference ray; tune alongside `RewardConfig.success_*`, `lug_spin_tolerance_rad`).
- **Dense / sparse mix**: defaults **dense 30% / sparse-success 70%** (`RewardConfig.mix_dense`, `mix_sparse_success`).
- **Stage 1** (Phase-1 training): `R_success=0`, so sparse success termination is off and the policy is driven purely by dense `align_reward` / `reach_reward`. For sparse hub-contact termination, use **stage ≥ 3** or edit `make_reward_config`.

**Excessive-contact-force termination**
- `contact_force_done`: terminates the episode when the max normal force across all contacts exceeds `contact_force_terminate_above` (`termination=contact_force`).
- Disable by setting it `<= 0`.
- Override at training time: `--contact-force-done 2500`.

Physics stability settings live in `EnvConfig` (overridable via train CLI) and `TyroEnv.reset()`'s `setPhysicsEngineParameter` call.

---

## Step 3 — Static Alignment Checks

Quick sanity checks that don't need a trained policy:

```bash
# Gripper Z-axis vs bolt-axis alignment (draws debug lines in GUI)
python scripts/check_alignment.py --render
python scripts/check_alignment.py --render --stage 1
# Options: --stage, --phase, --no-truck-hub-urdf

# With cargo proxy on: IK residual near the hub, Robot A self-collision, cargo contacts
python scripts/check_cargo_reachability.py --render

# Shaping reward path + 10-bolt randomization (run as module)
python -m scripts.check_shaping_and_bolts

# Regenerate the wheel station URDF from parameters
python scripts/generate_truck_wheel_station_urdf.py --output data/urdf/truck_assembly/truck_wheel_station.urdf
```

---

## Step 4 — Physics and PPO Defaults

`EnvConfig` defaults: `physics_num_sub_steps=6` (recommended range 4–8), `contact_erp≈0.15`, `contact_cfm≈1e-5` (forwarded to PyBullet's `globalCFM`).

Training defaults: `batch_size=128`, `gamma=0.995`, `n_steps=2048`.

Override at training time:
```bash
python -m src.train --physics-num-sub-steps 6 --contact-force-done 2500 ...
```

TensorBoard logs `reward/*` means and `env/contact_force_mean`.

---

## GUI Smoke

```bash
python -m src.test --render                       # 200-step live rollout, holds window when done
python -m src.test --render --action-scale 0      # static scene (no policy actions)
python -m src.test --render --no-hold --steps 500 # auto-exit, longer rollout
```

---

## Train / Evaluate

Stage (reward curriculum) and phase (domain-randomization strength) compose independently:

| Stage | Active reward terms | Goal |
|---|---|---|
| 1 | `align_A`, `reach_B` | Each robot solves its sub-task |
| 2 | + `coop`, `sync_joint_A` | Joint behavior emerges |
| 3 | + `success` + collision/action/jerk penalties | Full dense reward |
| 4 | dense replaced by `shape_A/B` potential shaping | Sim2Real-ready dense gradient |

```bash
# Alias mode — task=phase1 ↔ stage 1 + curriculum phase 1
python -m src.train --task phase1 --num-envs 8 --total-steps 1_000_000

# Stage 1 warmup (~30-60 min / 1M steps)
python -m src.train --stage 1 --phase 1 --total-steps 1_000_000 --num-envs 8

# Stage 3 main training, resumed from stage 1 (~3 h / 3M steps)
python -m src.train --stage 3 --phase 1 --total-steps 3_000_000 --num-envs 8 \
    --resume runs/stage1_phase1_*/final.zip

# Phase 3 — full domain randomization (±5cm hub offset)
python -m src.train --stage 3 --phase 3 --total-steps 2_000_000 --num-envs 8 \
    --resume runs/stage3_phase1_*/final.zip

# TensorBoard
tensorboard --logdir runs/

# Evaluation
python -m src.eval runs/stage3_phase3_*/best/best_model.zip --render --episodes 5
python -m src.eval runs/stage3_phase3_*/best/best_model.zip --episodes 20  # headless stats
```

Extra training CLI knobs:
- `--mix-dense FLOAT --mix-sparse-success FLOAT` — dense vs. sparse-success weight mix
- `--contact-force-done FLOAT` — contact-force termination threshold
- `--physics-num-sub-steps / --contact-erp / --contact-cfm` — solver tuning

---

## Hyperparameter Sweep (Phase 2)

Optuna sweep over PPO core hyperparameters and cooperation reward weights
(`w_c`, `alpha`, `beta`). Uses TPE sampling + median pruning; SQLite storage so
runs can be paused and resumed.

```bash
# Verify the full sweep loop end-to-end (1 trial × 5k steps, ~1-2 min)
python -m src.sweep --study smoke --smoke-test

# Real sweep
python -m src.sweep --study tyro1 --n-trials 30 --steps-per-trial 200000

# Resume the same study later (storage: runs/sweep/<study>/<study>.db)
python -m src.sweep --study tyro1 --n-trials 20
```

Search space lives in `src/sweep.py:suggest_hparams` — edit it to change the
PPO / reward variables Optuna explores.

---

## Roadmap (Progress Report §3)

| Phase | Goal | Status |
|---|---|---|
| 0 | Build PPO loop with 8-env parallelism | Done |
| 1 | Stage 1→3 reward curriculum, tune coop weights | In progress |
| 2 | Optuna sweep over PPO + reward weights | Tooling done |
| 3 | Full penalty tuning, prevent reward hacking | Planned |
| 4 | Potential shaping, ≥20-episode Sim2Real eval | Planned |
