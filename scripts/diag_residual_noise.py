#!/usr/bin/env python3
"""Quantify how policy exploration noise vs planner_pos_offset_scale affects
the mount landing precision.

Diagnosis context: Phase A success collapses as the mount gate tightens
(0.55m -> 0.04m) even though the zero-action nominal seats the tire at
d=0.000m, and 67% of spawns are easy (carry+mount only). The trained policy
std is stuck ~0.82 (ent_coef=0). Hypothesis: per-step residual noise
(std 0.82 -> tanh -> x pos_scale) exceeds the gate.

This injects Gaussian residual noise matching the policy (pre-squash std)
at several pos_offset_scale values and records the min tire<->hub distance
over the carry+mount, so we can pick a pos_scale whose noise floor fits the
hard 0.04m gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def run(pos_scale: float, std: float, episodes: int, seed: int) -> dict:
    cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious")
    cfg.planner_pos_offset_scale = float(pos_scale)
    env = TyroEnv(cfg=cfg, render=False, seed=seed)
    env.set_start_pos_easy_prob(0.999)  # easy (attached) spawns, like 67% case
    rng = np.random.default_rng(seed)
    mount_target = np.asarray(cfg.tire_mount_pos, float)
    adim = env.action_space.shape[0]
    min_ds = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        env.set_mount_tol(0.55, np.deg2rad(45.0))
        best = 1e9
        for _ in range(int(cfg.max_steps)):
            # emulate the stochastic policy: pre-squash Gaussian -> tanh
            a = np.tanh(rng.normal(0.0, std, size=adim)).astype(np.float32)
            _, _, term, trunc, _ = env.step(a)
            tp = np.asarray(env.scene.tire_pose()[0], float)
            best = min(best, float(np.linalg.norm(tp - mount_target)))
            if term or trunc:
                break
        min_ds.append(best)
    env.close()
    arr = np.asarray(min_ds)
    return {
        "pos_scale": pos_scale,
        "mean": arr.mean(),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "frac_le_004": float((arr <= 0.04).mean()),
        "frac_le_010": float((arr <= 0.10).mean()),
        "frac_le_015": float((arr <= 0.15).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+", default=[0.20, 0.10, 0.08, 0.05])
    ap.add_argument("--std", type=float, default=0.82,
                    help="pre-squash policy action std (from training log).")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("=" * 78)
    print(f"RESIDUAL NOISE vs pos_offset_scale  (policy std={args.std}, "
          f"easy spawns, {args.episodes} eps)")
    print("  min tire<->hub distance reached; gate hard=0.04m, mid=0.10-0.15m")
    print("=" * 78)
    for sc in args.scales:
        r = run(sc, args.std, args.episodes, args.seed)
        print(f"  pos_scale={sc:.2f}: mean={r['mean']:.3f}m p50={r['p50']:.3f} "
              f"p90={r['p90']:.3f} | <=0.04m:{r['frac_le_004']*100:3.0f}% "
              f"<=0.10m:{r['frac_le_010']*100:3.0f}% <=0.15m:{r['frac_le_015']*100:3.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
