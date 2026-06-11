#!/usr/bin/env python3
"""Interactive single-world E2E viewer: Robot A mount → Robot B nut-fastening.

ONE PyBullet GUI window, ONE client, used for both phases. PyBullet's camera
lives on the GUI client (not the simulation), so reusing the same env across the
A→B hand-off keeps the camera continuous and lets you rotate freely with the
mouse the whole time (drag = orbit, scroll = zoom, ctrl/cmd+drag = pan).

Both phases use the SAME hub offset (reproducible via --seed / --scenario-idx),
so it reads as one continuous take. The default scenario is a known full E2E
success (B fastens 10/10).

Run (needs a display; on this box use DISPLAY=:2):
    DISPLAY=:2 python scripts/e2e_view.py --scenario-idx 2 --seed 42
    DISPLAY=:2 python scripts/e2e_view.py --scenario-idx 2 --loop
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _resolve_model_path(path: str) -> str:
    pp = Path(path)
    if pp.suffix.lower() == ".zip" and pp.exists():
        return str(pp.with_suffix(""))
    if not pp.exists() and pp.suffix.lower() == ".zip":
        bare = pp.with_suffix("")
        if bare.with_suffix(".zip").exists():
            return str(bare)
    return path


def _scenario_offset(seed: int, idx0: int, dr_range_m: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dr_range_m <= 0.0:
        return np.zeros(2, dtype=np.float64)
    offs = rng.uniform(-dr_range_m, dr_range_m, size=(idx0 + 1, 2))
    return offs[idx0]


def _play_phase(env, model, *, seed, hub_offset, speed, banner):
    """Run one policy to episode end in real time. Returns final info."""
    # The obs/action masks are cached for the active layout; clear them so a
    # cfg/layout switch (A 85-d/6-act ↔ B 99-d/13-act) rebuilds them correctly.
    env._obs_mask_cache = None
    env._action_mask_cache = None
    env.set_dr_hub_xy_offset(hub_offset)
    obs, _ = env.reset(seed=seed)
    period = 1.0 / (float(env.cfg.control_freq_hz) * max(speed, 1e-6))
    terminated = truncated = False
    info: dict = {}
    steps = 0
    while not (terminated or truncated):
        if not p.isConnected(env.client):
            return None
        t0 = time.time()
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        steps += 1
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)
    print(f"  [{banner}] steps={steps}  success={info.get('is_success')}  "
          f"term={info.get('termination')}")
    return info


def _hold(env, seconds: float) -> bool:
    """Keep the GUI responsive (rotatable) for a pause. False if window closed."""
    end = time.time() + seconds
    while time.time() < end:
        if not p.isConnected(env.client):
            return False
        time.sleep(0.03)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive single-world E2E viewer.")
    ap.add_argument("--model-a", default="runs/phase1_mount_v3_dr/final.zip")
    ap.add_argument("--model-b", default="runs/nut_fastening_v16_dr/final.zip")
    ap.add_argument("--scenario-idx", type=int, default=2,
                    help="1-based scenario index (default 2: passed headless "
                         "E2E once; B policy ~55%% at 5cm DR, not guaranteed).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dr-range-cm", type=float, default=5.0)
    ap.add_argument("--mix-easy-prob", type=float, default=0.8)
    ap.add_argument("--a-max-steps", type=int, default=2000)
    ap.add_argument("--mount-radius-tol", type=float, default=0.55)
    ap.add_argument("--speed", type=float, default=1.5,
                    help="Playback speed multiplier (1.0 = real time).")
    ap.add_argument("--loop", action="store_true",
                    help="Replay the scenario continuously.")
    ap.add_argument("--cam-dist", type=float, default=2.6)
    ap.add_argument("--cam-yaw", type=float, default=55.0)
    ap.add_argument("--cam-pitch", type=float, default=-28.0)
    args = ap.parse_args()

    dr_range_m = float(args.dr_range_cm) / 100.0
    idx0 = int(args.scenario_idx) - 1
    if idx0 < 0:
        ap.error("--scenario-idx is 1-based and must be >= 1")
    off = _scenario_offset(args.seed, idx0, dr_range_m)
    norm_cm = float(np.linalg.norm(off)) * 100.0
    print(f"[view] scenario {args.scenario_idx}  "
          f"hub=({off[0]*100:+.2f}, {off[1]*100:+.2f}) cm  |hub|={norm_cm:.2f} cm")

    model_a = PPO.load(_resolve_model_path(args.model_a), device="cpu")
    model_b = PPO.load(_resolve_model_path(args.model_b), device="cpu")

    mount_overrides = dict(
        render=True, scene_layout="fanuc_spacious", terminate_on="mount",
        max_steps=int(args.a_max_steps), USE_DOMAIN_RANDOMIZATION=True,
        RANDOM_POSITION_RANGE=dr_range_m, DR_CARGO_ENABLE=False,
        planner_pos_offset_scale=0.06, mount_radius_tol=float(args.mount_radius_tol),
        contact_force_terminate_above=0.0, start_pos_curriculum_enable=True,
        include_hub_guide_obs=True,
    )
    nut_overrides = dict(
        render=True, scene_layout="fanuc_spacious", nut_fastening_task=True,
        nut_b_planner_residual=True, terminate_on="never", max_steps=2000,
        USE_DOMAIN_RANDOMIZATION=True, RANDOM_POSITION_RANGE=dr_range_m,
        DR_CARGO_ENABLE=False, nut_a_hold_jitter_rad=float(np.deg2rad(6.0)),
        contact_force_terminate_above=0.0, collision_terminates=False,
    )

    cfg_a = make_env_config(stage=3, phase=1, **mount_overrides)
    cfg_b = make_env_config(stage=3, phase=1, **nut_overrides)

    # ONE env / ONE GUI client, reused across both phases (camera persists).
    env = TyroEnv(cfg=cfg_a, render=True, seed=args.seed)
    env.set_start_pos_easy_prob(float(args.mix_easy_prob))

    # Initial camera; the user can freely orbit/zoom/pan with the mouse and it
    # stays put across the A→B reset (camera is a GUI-client property).
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client)
    except p.error:
        pass
    p.resetDebugVisualizerCamera(
        cameraDistance=float(args.cam_dist),
        cameraYaw=float(args.cam_yaw),
        cameraPitch=float(args.cam_pitch),
        cameraTargetPosition=[
            env.cfg.tire_mount_pos[0] + float(off[0]),
            env.cfg.tire_mount_pos[1] + float(off[1]),
            env.cfg.tire_mount_pos[2],
        ],
        physicsClientId=env.client,
    )
    print("[view] GUI: drag = orbit, scroll = zoom, ctrl/cmd+drag = pan. "
          "Close window or Ctrl-C to quit.")

    seed_a = args.seed + idx0 * 2
    seed_b = args.seed + idx0 * 2 + 1

    try:
        while True:
            print("[view] === Phase A: Robot A mount ===")
            env.cfg = cfg_a
            env.cfg.render = True
            info_a = _play_phase(env, model_a, seed=seed_a, hub_offset=off,
                                 speed=args.speed, banner="A")
            if info_a is None:
                break
            if not _hold(env, 1.0):
                break

            print("[view] === Phase B: Robot B nut-fastening ===")
            env.cfg = cfg_b
            env.cfg.render = True
            # Match Gym spaces to B's layout (obs 99 / act 13) for cleanliness.
            from gymnasium import spaces  # local import keeps top tidy
            env.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(cfg_b.action.dim,), dtype=np.float32)
            env.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(cfg_b.obs.dim,), dtype=np.float32)
            info_b = _play_phase(env, model_b, seed=seed_b, hub_offset=off,
                                 speed=args.speed, banner="B")
            if info_b is None:
                break

            ok = bool(info_a.get("is_success") and info_b.get("is_success"))
            print(f"[view] E2E: {'PASS' if ok else 'FAIL'}")
            # Restore A's spaces for a potential next loop.
            env.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(cfg_a.action.dim,), dtype=np.float32)
            env.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(cfg_a.obs.dim,), dtype=np.float32)

            if not args.loop:
                print("[view] done. Holding window open — close it or Ctrl-C to exit.")
                while p.isConnected(env.client):
                    time.sleep(0.05)
                break
            if not _hold(env, 1.5):
                break
    except KeyboardInterrupt:
        print("\n[view] interrupted.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
