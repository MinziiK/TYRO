#!/usr/bin/env python3
"""Deterministically evaluate a checkpoint's mount precision.

Decisive diagnostic for the Phase-A success collapse: the zero-action nominal
seats the tire at d=0.000m and is robust to white residual noise, yet success
fell 1.0 -> 0.17 as the gate tightened. Remaining hypothesis: the policy's
LEARNED MEAN drifted to harmful residuals. Compare an early checkpoint
(success ~1.0) vs a late one (success ~0.17): run the deterministic policy
(no exploration) on easy spawns and record the best tire<->hub radius AND
bore-vs-hub angle reached, plus whether a fixed gate would fire.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_baselines3 import PPO  # noqa: E402
from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def angle_deg(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    d = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))
    return float(np.degrees(np.arccos(d)))


def eval_ckpt(path: str, episodes: int, seed: int, device: str) -> dict:
    cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious")
    env = TyroEnv(cfg=cfg, seed=seed)
    env.set_start_pos_easy_prob(0.999)
    model = PPO.load(path, device=device)
    hub_axis = np.asarray(cfg.hub_axis_world, float)
    mount_target = np.asarray(cfg.tire_mount_pos, float)
    best_r = []
    ang_at_best = []
    fired_004 = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        env.set_mount_tol(0.55, np.deg2rad(45.0))
        br = 1e9
        ba = 180.0
        for _ in range(int(cfg.max_steps)):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            tp = np.asarray(env.scene.tire_pose()[0], float)
            r = float(np.linalg.norm(tp - mount_target))
            if r < br:
                br = r
                ba = angle_deg(env.scene.tire_axis(), hub_axis)
            if term or trunc:
                break
        best_r.append(br)
        ang_at_best.append(ba)
        if br <= 0.04 and ba <= 5.0:
            fired_004 += 1
    env.close()
    r = np.asarray(best_r)
    a = np.asarray(ang_at_best)
    return {
        "ckpt": Path(path).name,
        "r_mean": r.mean(), "r_p50": np.percentile(r, 50), "r_p90": np.percentile(r, 90),
        "a_mean": a.mean(), "a_p90": np.percentile(a, 90),
        "hard_gate_fire": fired_004 / episodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    print("=" * 80)
    print(f"DETERMINISTIC mount precision per checkpoint ({args.episodes} easy eps)")
    print("  radius=tire<->hub(m), angle=bore<->hubaxis(deg) at best approach")
    print("=" * 80)
    for c in args.ckpts:
        r = eval_ckpt(c, args.episodes, args.seed, args.device)
        print(f"  {r['ckpt']:<26} r:mean={r['r_mean']:.3f} p50={r['r_p50']:.3f} "
              f"p90={r['r_p90']:.3f} | ang:mean={r['a_mean']:4.1f} p90={r['a_p90']:4.1f} "
              f"| hard(0.04m/5d) fire={r['hard_gate_fire']*100:3.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
