#!/usr/bin/env python3
"""Zero-residual smoke: nominal nut planner + scripted macro → 10/10."""
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


def run_smoke(seed: int = 0, max_steps: int = 4000, verbose: bool = True) -> dict:
    cfg = make_env_config(
        stage=3, phase=1, nut_fastening_task=True,
        nut_b_planner_residual=True,
        scene_layout="fanuc_spacious", terminate_on="never",
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
        contact_force_terminate_above=0.0,
        max_steps=max_steps,
    )
    cfg.nut_b_hotstart_random_bolt = False
    cfg.nut_planner_traj_steps = 120

    env = TyroEnv(cfg=cfg, render=False, seed=seed)
    env.reset(seed=seed)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)

    n_policy = 0
    for step in range(max_steps):
        _, _, term, trunc, info = env.step(zero)
        nf = int(info.get("n_fastened_policy", 0))
        if nf > n_policy and verbose:
            print(f"  step {step:4d}: bolt fastened (policy) → {nf}/10")
            n_policy = nf
        if term or trunc:
            break
        if info.get("all_fastened"):
            break

    order = env._nut_order()
    ok = len(env._nut_fastened) >= len(order)
    result = {
        "ok": ok,
        "n_fastened": len(env._nut_fastened),
        "n_fastened_policy": len(env._nut_fastened) - int(env._nut_premark),
        "steps": step + 1,
        "order": list(order),
        "fastened": list(env._nut_fastened),
    }
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"[smoke v14] {status}: {result['n_fastened_policy']}/10 policy "
              f"({result['n_fastened']}/10 total) in {result['steps']} steps")
        if not ok:
            print(f"  fastened: {result['fastened']}")
    env.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()
    r = run_smoke(seed=args.seed, max_steps=args.max_steps)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
