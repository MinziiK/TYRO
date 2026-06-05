#!/usr/bin/env python3
"""Benchmark SubprocVecEnv throughput (env-steps/sec) vs num_envs.

Faithfully mirrors src/train.py: same cfg_factory (fanuc_spacious, phase 1,
stage 3), SubprocVecEnv(start_method="spawn"), VecMonitor. Runs a random
policy for a fixed number of vec-steps and reports wall-clock fps so we can
decide whether raising --num-envs (the machine has 96 cores, load ~15) buys
real speed before paying a training restart.

Usage:
    python scripts/bench_vecenv_fps.py --envs 72 128 --steps 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def make_env(rank: int, seed: int):
    def _init():
        cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious")
        return TyroEnv(cfg=cfg, seed=seed + rank)
    return _init


def bench(n_envs: int, n_steps: int, warmup: int, seed: int) -> dict:
    vec = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)],
                        start_method="spawn")
    vec = VecMonitor(vec)
    try:
        vec.reset()
        act = np.stack([vec.action_space.sample() for _ in range(n_envs)])
        # warmup (let workers JIT/settle, exclude from timing)
        for _ in range(warmup):
            vec.step(act)
        t0 = time.time()
        for _ in range(n_steps):
            vec.step(act)
        dt = time.time() - t0
    finally:
        vec.close()
    env_steps = n_steps * n_envs
    return {
        "n_envs": n_envs,
        "wall_s": dt,
        "iter_per_s": n_steps / dt,
        "fps": env_steps / dt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[72, 128])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 64)
    print(f"VecEnv FPS benchmark  steps={args.steps} warmup={args.warmup}")
    print("(env-steps/sec = throughput that maps to SB3 'fps')")
    print("=" * 64)
    rows = []
    for n in args.envs:
        r = bench(n, args.steps, args.warmup, args.seed)
        rows.append(r)
        print(f"  num_envs={r['n_envs']:4d}  fps={r['fps']:7.1f}  "
              f"vec_iter/s={r['iter_per_s']:6.2f}  wall={r['wall_s']:5.1f}s")
    if len(rows) >= 2:
        base = rows[0]
        print("-" * 64)
        for r in rows[1:]:
            print(f"  {r['n_envs']} vs {base['n_envs']}: "
                  f"{r['fps']/base['fps']:.2f}x fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
