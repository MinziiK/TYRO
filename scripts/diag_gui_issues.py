#!/usr/bin/env python3
"""Diagnose the two GUI-observed issues on the Phase A mount policy.

Issue 1: tire teleports away and back during the carry-to-hub leg.
Issue 2: after seating on the hub the episode never terminates.

Replays the checkpoint headless with the EXACT src.eval config (4-stage,
terminate_on=mount, easy-start) and logs per step:
  * tire-position jump vs previous step (teleport detector)
  * task_stage
  * mount gate: d_mount, theta_deg, current radius/angle tol, fired?
Then prints the termination + the max tire jump and where it happened.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pybullet as p  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def tire_obstacle_min_dist(env):
    """Closest distance tire<->(vehicle, cargo back wall). Negative=overlap."""
    uid = env.handles.tire
    mind = 9.0
    who = "-"
    obstacles = []
    if env.handles.vehicle is not None:
        obstacles.append(("vehicle", int(env.handles.vehicle)))
    bw = getattr(env.handles, "cargo_back_wall", None)
    if bw is not None:
        obstacles.append(("backwall", int(bw)))
    for name, ob in obstacles:
        cps = p.getClosestPoints(bodyA=uid, bodyB=ob, distance=0.2,
                                 physicsClientId=env.client)
        for cp in cps:
            if len(cp) > 8 and float(cp[8]) < mind:
                mind = float(cp[8]); who = name
    return mind, who


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase1_mount_v2/best/best_model.zip")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--terminate-on", default="mount")
    ap.add_argument("--jump-thresh", type=float, default=0.08,
                    help="tire jump (m) above this is flagged as a teleport")
    args = ap.parse_args()

    # Mirror src.eval overrides exactly.
    overrides = dict(
        render=False,
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        contact_force_terminate_above=0.0,
        scene_layout="fanuc_spacious",
        terminate_on=str(args.terminate_on),
    )
    cfg = make_env_config(stage=3, phase=1, **overrides)
    env = TyroEnv(cfg=cfg, render=False, seed=42)
    env.set_start_pos_easy_prob(0.999)
    model = PPO.load(args.ckpt, device="cpu")

    r_tol, a_tol = env.get_mount_tol()
    print(f"remount_cycle_enable = {cfg.remount_cycle_enable}")
    print(f"mount gate tol: radius={r_tol:.3f} m  angle={np.rad2deg(a_tol):.1f}°  "
          f"(config hard: {cfg.mount_radius_tol} m / {np.rad2deg(cfg.reward.delta_A):.1f}°)")
    print(f"terminate_on={cfg.terminate_on}  max_steps={cfg.max_steps}  "
          f"pin_tire_on_mount={cfg.pin_tire_on_mount}  mount_hold_steps={cfg.mount_hold_steps}")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=42 + ep)
        prev_tire = np.asarray(env.scene.tire_pose()[0], dtype=np.float64)
        max_jump = 0.0
        max_jump_t = -1
        max_jump_stage = -1
        n_teleports = 0
        mount_fired_step = None
        last_stage = int(env.task_stage)
        min_d_mount = 1e9
        min_theta = 1e9
        d_mount_series = []

        for t in range(int(cfg.max_steps)):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            tire = np.asarray(env.scene.tire_pose()[0], dtype=np.float64)
            jump = float(np.linalg.norm(tire - prev_tire))
            stage = int(info.get("task_stage", -1))
            if jump > max_jump:
                max_jump, max_jump_t, max_jump_stage = jump, t, last_stage
            if jump > args.jump_thresh:
                n_teleports += 1
                md, who = tire_obstacle_min_dist(env)
                if n_teleports <= 12:
                    print(f"  ep{ep} t={t:4d} TELEPORT jump={jump*100:5.1f}cm "
                          f"stage={stage}  tire={np.round(tire,3)} prev={np.round(prev_tire,3)} "
                          f"| nearest_obstacle={who} dist={md*100:+.1f}cm")
            prev_tire = tire

            # Mount-gate telemetry while in carry stage.
            if stage == 1:
                hub = np.asarray(env.scene.hub_pose()[0], dtype=np.float64)
                mount_target = np.asarray(cfg.tire_mount_pos, dtype=np.float64)
                d_mount = float(np.linalg.norm(tire - mount_target))
                ta = env.scene.tire_axis(); ha = env.scene.hub_axis()
                theta = float(np.rad2deg(np.arccos(np.clip(np.dot(ta, ha), -1, 1))))
                min_d_mount = min(min_d_mount, d_mount)
                min_theta = min(min_theta, theta)
                d_mount_series.append(d_mount)
            if stage != last_stage:
                print(f"  ep{ep} t={t:4d} stage {last_stage}->{stage}")
                if last_stage == 1 and stage == 2:
                    mount_fired_step = t
                last_stage = stage
            if term or trunc:
                tag = info.get("termination", "?")
                print(f"  ep{ep} END t={t} term={tag} success={info.get('is_success')} stage={stage}")
                break
        else:
            print(f"  ep{ep} END t={cfg.max_steps} term=NO-TERMINATION (ran full horizon) stage={last_stage}")

        print(f"    -> closest mount approach: d_mount_min={min_d_mount:.3f} m "
              f"(gate {r_tol:.3f})  theta_min={min_theta:.1f}° (gate {np.rad2deg(a_tol):.1f}°)")
        # Oscillation analysis: after first reaching within 12 cm, how much
        # does the tire bounce back out (the visible "teleport away & back")?
        dm = np.asarray(d_mount_series)
        if dm.size > 10:
            i_min = int(np.argmin(dm))
            after = dm[i_min:]
            max_rebound = float(after.max() - after.min()) if after.size else 0.0
            # count direction reversals once near the hub (< 0.15 m)
            near = dm < 0.15
            reversals = 0
            if near.any():
                seg = dm[np.argmax(near):]
                dsg = np.diff(seg)
                reversals = int(np.sum(np.diff(np.sign(dsg)) != 0))
            print(f"    -> d_mount: first_min at step {i_min}, value {dm[i_min]:.3f}; "
                  f"max rebound AFTER min = {max_rebound*100:.1f} cm; "
                  f"near-hub(<15cm) direction reversals = {reversals}")
            print(f"       d_mount last 20 steps: {np.round(dm[-20:],3)}")
        print(f"    -> max tire jump = {max_jump*100:.1f} cm at t={max_jump_t} (stage {max_jump_stage}); "
              f"teleports>{args.jump_thresh*100:.0f}cm = {n_teleports}")
        print()

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
