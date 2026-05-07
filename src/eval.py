"""Evaluate a trained PPO checkpoint on TyroEnv.

Examples
--------
    # Run 20 deterministic episodes, print success rate / reward stats:
    python -m src.eval runs/stage3_phase1_*/best/best_model.zip --episodes 20

    # Watch the policy in GUI, hold window after each episode:
    python -m src.eval <ckpt.zip> --render --episodes 3

    # Evaluate at a different DR phase than training to check transfer:
    python -m src.eval <ckpt.zip> --phase 3
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from src.config import make_env_config
from src.env import TyroEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="Path to .zip checkpoint.")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--phase", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of using deterministic mean.")
    ap.add_argument("--no-hold", action="store_true",
                    help="Don't hold the GUI window after the last episode.")
    args = ap.parse_args()

    cfg = make_env_config(stage=args.stage, phase=args.phase, render=args.render)
    env = TyroEnv(cfg=cfg, render=args.render, seed=args.seed)
    print(f"[eval] loading {args.model}")
    model = PPO.load(args.model, device="cpu")  # CPU is plenty for inference

    successes: list[bool] = []
    rewards: list[float] = []
    lengths: list[int] = []
    term_counts: dict[str, int] = {}
    step_period = 1.0 / cfg.control_freq_hz

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_r = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            t_start = time.time()
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            steps += 1
            if args.render:
                dt = time.time() - t_start
                if dt < step_period:
                    time.sleep(step_period - dt)
        is_success = bool(info.get("is_success", False))
        term = info.get("termination", "unknown")
        term_counts[term] = term_counts.get(term, 0) + 1
        successes.append(is_success)
        rewards.append(total_r)
        lengths.append(steps)
        print(f"  ep {ep:3d}  r={total_r:+8.2f}  len={steps:4d}  "
              f"success={is_success}  termination={term}")

    n = len(successes)
    sr = sum(successes) / n * 100
    print("\n=== Eval summary ===")
    print(f"  episodes:     {n}")
    print(f"  success rate: {sr:.1f}%  ({sum(successes)}/{n})")
    print(f"  reward:       mean={statistics.mean(rewards):+.2f}  "
          f"std={statistics.pstdev(rewards):.2f}  "
          f"min={min(rewards):+.2f}  max={max(rewards):+.2f}")
    print(f"  length:       mean={statistics.mean(lengths):.1f}  "
          f"min={min(lengths)}  max={max(lengths)}")
    print(f"  terminations: {term_counts}")

    if args.render and not args.no_hold:
        print("[eval] GUI held open. Close window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
