#!/usr/bin/env python3
"""Diagnose workspace terminations in Phase B remount cycle.

For each episode, log the task_stage and body positions when the workspace
gate fires, and compare against the zero-residual nominal path envelope.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_baselines3 import PPO  # noqa: E402
from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def ws_violation_detail(env: TyroEnv) -> tuple[bool, str, dict]:
    ws = float(env.cfg.obs.workspace_radius)
    z_lo = float(env.cfg.floor_z) - 0.05
    xy_lim = ws * 1.5
    details = {}
    for name, getter in (
        ("robotA_ee", env.robot_A.ee_pose),
        ("robotB_ee", env.robot_B.ee_pose),
        ("tire", env.scene.tire_pose),
    ):
        pos, _ = getter()
        pos = np.asarray(pos, dtype=np.float64)
        xy = float(np.linalg.norm(pos[:2]))
        details[name] = {
            "pos": pos,
            "xy": xy,
            "z": float(pos[2]),
            "xy_bad": xy > xy_lim,
            "z_hi": float(pos[2]) > 2.5,
            "z_lo": float(pos[2]) < z_lo,
        }
    for name, d in details.items():
        if d["xy_bad"] or d["z_hi"] or d["z_lo"]:
            reasons = []
            if d["xy_bad"]:
                reasons.append(f"xy={d['xy']:.3f}>{xy_lim:.1f}")
            if d["z_hi"]:
                reasons.append(f"z={d['z']:.3f}>2.5")
            if d["z_lo"]:
                reasons.append(f"z={d['z']:.3f}<{z_lo:.3f}")
            return True, f"{name}({','.join(reasons)})", details
    return False, "", details


def run_rollouts(ckpt: str | None, episodes: int, deterministic: bool, seed: int) -> None:
    cfg = make_env_config(
        stage=3, phase=1, scene_layout="fanuc_spacious",
        remount_cycle_enable=True, terminate_on="never",
        reverse_curriculum_enable=False,
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        start_pos_easy_prob=0.7,
        max_steps=1000,
    )
    env = TyroEnv(cfg=cfg, seed=seed)
    env.set_start_pos_easy_prob(0.7)

    ws = float(cfg.obs.workspace_radius)
    xy_lim = ws * 1.5
    print(f"workspace_radius={ws}  xy_limit={xy_lim:.2f}  z_lo={cfg.floor_z - 0.05:.3f}")

    # Nominal envelope (zero residual).
    print("\n=== Zero-residual nominal envelope ===")
    env2 = TyroEnv(cfg=cfg, seed=seed)
    env2.set_start_pos_easy_prob(0.999)
    env2.reset(seed=seed)
    env2.set_mount_tol(0.25, np.deg2rad(40.0))
    zero = np.zeros(env2.action_space.shape, dtype=np.float32)
    max_xy = {"robotA_ee": 0.0, "robotB_ee": 0.0, "tire": 0.0}
    stage_at_max = {k: -1 for k in max_xy}
    for t in range(1500):
        _, _, term, trunc, info = env2.step(zero)
        st = int(info.get("task_stage", -1))
        for name, getter in (
            ("robotA_ee", env2.robot_A.ee_pose),
            ("robotB_ee", env2.robot_B.ee_pose),
            ("tire", env2.scene.tire_pose),
        ):
            pos, _ = getter()
            xy = float(np.linalg.norm(np.asarray(pos)[:2]))
            if xy > max_xy[name]:
                max_xy[name] = xy
                stage_at_max[name] = st
        if term or trunc:
            print(f"  nominal ended t={t} term={info.get('termination')} success={info.get('is_success')}")
            break
    for name in max_xy:
        flag = "OVER" if max_xy[name] > xy_lim else "ok"
        print(f"  {name:12s} max_xy={max_xy[name]:.3f} at stage {stage_at_max[name]}  [{flag}]")
    env2.close()

    if ckpt:
        model = PPO.load(ckpt, device="cpu")
        print(f"\n=== Policy rollouts (deterministic={deterministic}, n={episodes}) ===")
        print(f"  ckpt: {Path(ckpt).name}")
    else:
        model = None
        print(f"\n=== Random-action rollouts (n={episodes}) ===")

    term_counts: Counter = Counter()
    stage_counts: Counter = Counter()
    body_counts: Counter = Counter()
    ep_lens = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(int(cfg.max_steps)):
            if model is not None:
                act, _ = model.predict(obs, deterministic=deterministic)
            else:
                act = env.action_space.sample()
            obs, _, term, trunc, info = env.step(act)
            bad, reason, _ = ws_violation_detail(env)
            if bad and (term or trunc):
                st = int(info.get("task_stage", -1))
                tag = info.get("termination", "?")
                term_counts[tag] += 1
                if tag == "workspace":
                    stage_counts[st] += 1
                    body_counts[reason.split("(")[0]] += 1
                    if ep < 5:
                        print(f"  ep{ep} t={t:4d} stage={st}  {reason}")
                ep_lens.append(t + 1)
                break
            if term or trunc:
                tag = info.get("termination", "?")
                term_counts[tag] += 1
                ep_lens.append(t + 1)
                break

    print(f"\n  termination: {dict(term_counts)}")
    print(f"  workspace by stage: {dict(sorted(stage_counts.items()))}")
    print(f"  workspace by body: {dict(body_counts)}")
    if ep_lens:
        print(f"  ep_len mean={np.mean(ep_lens):.0f}  p50={np.median(ep_lens):.0f}")
    env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase1_fullcycle_v3/best/best_model.zip")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ckpt = args.ckpt if Path(args.ckpt).exists() else None
    if ckpt is None:
        print(f"[warn] ckpt not found: {args.ckpt}")
    run_rollouts(ckpt, args.episodes, not args.stochastic, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
