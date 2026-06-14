#!/usr/bin/env python3
"""GUI viewer for the trained Robot-B nut-fastening policy (only Robot B).

Loads a nut-fastening checkpoint and plays it in a single PyBullet GUI window,
episode after episode, so you can watch B drive the socket around the lug ring
and fasten the bolts. Robot A is a static support fixture (tire already mounted).

This is NOT the E2E eval — it shows only the Robot-B task in isolation.

Run (needs a display; on this box use DISPLAY=:2):
    DISPLAY=:2 python scripts/view_nut.py                       # v16 DR, 5cm
    DISPLAY=:2 python scripts/view_nut.py --dr-range-cm 0       # nominal hub
    DISPLAY=:2 python scripts/view_nut.py --model runs/nut_fastening_v15/final.zip
    DISPLAY=:2 python scripts/view_nut.py --episodes 5 --speed 2.0

Controls: drag = orbit, scroll = zoom, ctrl/cmd+drag = pan.
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


def main() -> int:
    ap = argparse.ArgumentParser(description="GUI viewer: Robot-B nut-fastening policy.")
    ap.add_argument("--model", default="runs/nut_fastening_v16_dr/final.zip",
                    help="Nut-fastening checkpoint (.zip).")
    ap.add_argument(
        "--nut-pure-rl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="v17 pure-RL wiring (no planner/macro, coaxial lock, 12-d task obs).",
    )
    ap.add_argument(
        "--hotstart-alpha", type=float, default=1.0,
        help="Hot-start alpha for Robot B (1=at bolt staging, 0=HOME). "
             "Match training curriculum for honest replay.",
    )
    ap.add_argument(
        "--hotstart-random-bolt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Random-bolt premark + hot-start (matches v17 training).",
    )
    ap.add_argument(
        "--per-leg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Terminate after one bolt (default: on for --nut-pure-rl).",
    )
    ap.add_argument(
        "--v19",
        action="store_true",
        help="v19 wiring: align servo, rigid A, collision=fail, solo 3-d "
             "action (match v19 checkpoints; implies --nut-pure-rl).",
    )
    ap.add_argument(
        "--v20",
        action="store_true",
        help="v20 wiring: v19 + INSERT axial servo + seat depth 0.7 cm "
             "(match v20 checkpoints; implies --nut-pure-rl).",
    )
    ap.add_argument(
        "--v23",
        action="store_true",
        help="v23 wiring: v20/v22 + approach-seeded clean IK (no wrist_1 spin).",
    )
    ap.add_argument(
        "--v24",
        action="store_true",
        help="v24 wiring: v20/v22 + lightweight shortest-macro winding cleanup "
             "(no wrist spin, no IK re-solve).",
    )
    ap.add_argument(
        "--smooth-macro",
        action="store_true",
        help="Smooth clean-branch PREP (waypoint routing, no 1-frame snap on "
             "most bolts). No retrain needed.",
    )
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dr-range-cm", type=float, default=5.0,
                    help="Hub offset half-width (0 = nominal hub).")
    ap.add_argument(
        "--scenario", type=int, default=2,
        help=(
            "Replay a specific E2E scenario (1-based) with the EXACT same hub "
            "offset + seed as scripts/e2e_eval.py, so a known-good case is "
            "reproducible. Default 2 = a B-success case. Set 0 to instead "
            "resample a fresh random hub each episode."
        ),
    )
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the policy mean (deployment mode). --no-deterministic "
             "samples actions like training rollouts (v17/v18 policies "
             "currently succeed far more often stochastically).",
    )
    ap.add_argument("--speed", type=float, default=1.5,
                    help="Playback speed multiplier (1.0 = real time).")
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="Episode horizon (v17 training uses 800).")
    ap.add_argument("--loop", action="store_true",
                    help="Replay forever (wrap after the last episode).")
    ap.add_argument("--cam-dist", type=float, default=2.4)
    ap.add_argument("--cam-yaw", type=float, default=55.0)
    ap.add_argument("--cam-pitch", type=float, default=-28.0)
    args = ap.parse_args()

    dr_range_m = float(args.dr_range_cm) / 100.0

    nut_overrides = dict(
        render=True, scene_layout="fanuc_spacious", nut_fastening_task=True,
        nut_b_planner_residual=not bool(args.nut_pure_rl),
        terminate_on="never",
        max_steps=int(args.max_steps),
        USE_DOMAIN_RANDOMIZATION=(dr_range_m > 0.0),
        RANDOM_POSITION_RANGE=dr_range_m, DR_CARGO_ENABLE=False,
        nut_a_hold_jitter_rad=float(np.deg2rad(6.0)),
        contact_force_terminate_above=0.0, collision_terminates=False,
    )
    if bool(args.v19) or bool(args.v20) or bool(args.v23) or bool(args.v24):
        args.nut_pure_rl = True
    if bool(args.nut_pure_rl):
        nut_overrides.update(
            nut_pure_rl=True,
            nut_b_planner_residual=False,
            nut_b_hotstart_enable=True,
            nut_b_hotstart_alpha=float(args.hotstart_alpha),
            nut_b_hotstart_random_bolt=bool(args.hotstart_random_bolt),
        )
        if args.per_leg is not None:
            nut_overrides["nut_per_leg_episode"] = bool(args.per_leg)
        if bool(args.v19) or bool(args.v20) or bool(args.v23) or bool(args.v24):
            nut_overrides.update(
                nut_b_align_servo=True,
                nut_a_kinematic_freeze=True,
                nut_collision_fail=True,
                nut_b_solo_action=True,
                nut_arrive_lat_tol=0.015,
                nut_seat_lat_mult=1.0,
                nut_a_hold_jitter_rad=0.0,
            )
        if bool(args.v20) or bool(args.v23) or bool(args.v24):
            nut_overrides.update(
                nut_b_axial_insert_servo=True,
                nut_insert_depth_tol=0.007,
                nut_b_insert_branch_search=True,
                nut_b_clean_branch_insert=True,
            )
        if bool(args.v23):
            nut_overrides["nut_clean_approach_seed"] = True
            nut_overrides["nut_clean_seat_cache"] = ""
        if bool(args.v24):
            nut_overrides["nut_clean_shortest_macro"] = True
        if bool(args.smooth_macro):
            nut_overrides["nut_clean_macro_smooth"] = True
            nut_overrides["nut_clean_prep_len"] = 72
            nut_overrides["nut_clean_plunge_len"] = 45
    cfg = make_env_config(stage=3, phase=1, **nut_overrides)
    if bool(args.nut_pure_rl) and args.per_leg is None:
        cfg.nut_per_leg_episode = True

    model = PPO.load(_resolve_model_path(args.model), device="cpu")
    print(f"[nut-view] model: {args.model}")
    print(f"[nut-view] layout obs={model.observation_space.shape[0]} "
          f"act={model.action_space.shape[0]}  DR=±{args.dr_range_cm:.1f} cm")

    env = TyroEnv(cfg=cfg, render=True, seed=args.seed)
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client)
    except p.error:
        pass
    p.resetDebugVisualizerCamera(
        cameraDistance=float(args.cam_dist),
        cameraYaw=float(args.cam_yaw),
        cameraPitch=float(args.cam_pitch),
        cameraTargetPosition=[
            env.cfg.tire_mount_pos[0], env.cfg.tire_mount_pos[1],
            env.cfg.tire_mount_pos[2],
        ],
        physicsClientId=env.client,
    )
    print("[nut-view] GUI: drag = orbit, scroll = zoom, ctrl/cmd+drag = pan. "
          "Close window or Ctrl-C to quit.")

    # Scenario mode: lock the hub offset + seed to a specific E2E scenario so a
    # known-good (B-success) case is reproducible. The offset is sampled exactly
    # like scripts/e2e_eval.py (np.random.default_rng(seed) uniform sweep) and
    # the per-scenario B seed is seed + (idx*2 + 1).
    scenario_off = None
    scenario_seed = None
    if args.scenario and args.scenario >= 1 and dr_range_m > 0.0:
        idx0 = args.scenario - 1
        sweep = np.random.default_rng(args.seed).uniform(
            -dr_range_m, dr_range_m, size=(idx0 + 1, 2))
        scenario_off = sweep[idx0]
        scenario_seed = args.seed + idx0 * 2 + 1
        print(f"[nut-view] scenario {args.scenario}: "
              f"hub=({scenario_off[0]*100:+.2f}, {scenario_off[1]*100:+.2f}) cm  "
              f"seed={scenario_seed}")

    period = 1.0 / (float(env.cfg.control_freq_hz) * max(args.speed, 1e-6))
    n_success = 0
    ep = 0
    try:
        while True:
            if scenario_off is not None:
                env.set_dr_hub_xy_offset(scenario_off)
                ep_seed = scenario_seed
            else:
                env.set_dr_hub_xy_offset(None)  # resample each reset
                ep_seed = args.seed + ep
            obs, _ = env.reset(seed=ep_seed)
            off = np.asarray(getattr(env, "_dr_hub_xy_offset", np.zeros(2)))
            norm_cm = float(np.linalg.norm(off)) * 100.0
            print(f"\n[nut-view] episode {ep + 1}  "
                  f"hub=({off[0]*100:+.2f}, {off[1]*100:+.2f}) cm  "
                  f"|hub|={norm_cm:.2f} cm")
            terminated = truncated = False
            info: dict = {}
            steps = 0
            while not (terminated or truncated):
                if not p.isConnected(env.client):
                    print("\n[nut-view] window closed — stopping.")
                    env.close()
                    return 0
                t0 = time.time()
                action, _ = model.predict(
                    obs, deterministic=bool(args.deterministic))
                obs, _r, terminated, truncated, info = env.step(action)
                steps += 1
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
            ok = bool(info.get("is_success"))
            n_success += int(ok)
            print(f"  steps={steps}  success={ok}  "
                  f"n_fastened={info.get('n_fastened', 0)}/10  "
                  f"term={info.get('termination')}")
            ep += 1
            if ep >= args.episodes and not args.loop:
                break
            if args.loop and ep >= args.episodes:
                ep = 0
    except KeyboardInterrupt:
        print("\n[nut-view] interrupted.")
    finally:
        print(f"\n[nut-view] success {n_success}/{ep if ep else '?'}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
