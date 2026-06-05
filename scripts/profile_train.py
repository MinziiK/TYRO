#!/usr/bin/env python3
"""Precisely split where training wall-time goes, to target speedups.

Two modes (run both):

  --mode physics : time a single TyroEnv.step() for several
                   ``physics_num_sub_steps`` values (isolates the per-step
                   collection cost — the dominant rollout lever). Fast.

  --mode ppo     : build the SAME PPO + SubprocVecEnv as src/train.py and
                   measure, per iteration, COLLECTION time (rollout) vs
                   UPDATE time (gradient epochs) via a timing callback.
                   Reports the split + device. Use --device cpu/cuda to A/B.

Does not touch the live training run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def mode_physics(substeps_list, n_steps: int) -> None:
    print("=" * 66)
    print(f"PHYSICS per-step cost  (single env, {n_steps} steps each)")
    print("=" * 66)
    base = None
    for ss in substeps_list:
        cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious")
        cfg.physics_num_sub_steps = int(ss)
        env = TyroEnv(cfg=cfg, render=False, seed=0)
        env.set_start_pos_easy_prob(0.999)
        env.reset()
        act = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(10):  # warmup
            env.step(act)
        t0 = time.time()
        for _ in range(n_steps):
            env.step(act)
        dt = time.time() - t0
        sps = n_steps / dt
        if base is None:
            base = sps
        print(f"  num_sub_steps={ss:3d}  {sps:7.1f} env-steps/s  "
              f"({dt*1000/n_steps:5.2f} ms/step)  speedup_vs_first={sps/base:.2f}x")
        env.close()


def mode_ppo(device: str, num_envs: int, n_steps: int, batch_size: int,
             n_epochs: int, net_arch: str, iters: int) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback

    def make_env(rank: int):
        def _init():
            cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious")
            return TyroEnv(cfg=cfg, seed=rank)
        return _init

    class SplitTimer(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t_rollout_start = None
            self.t_rollout_end = None
            self.collect = []
            self.update = []

        def _on_rollout_start(self) -> None:
            now = time.time()
            if self.t_rollout_end is not None:
                self.update.append(now - self.t_rollout_end)
            self.t_rollout_start = now

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self.t_rollout_end = time.time()
            self.collect.append(self.t_rollout_end - self.t_rollout_start)

    arch = [int(w) for w in net_arch.split(",") if w]
    vec = SubprocVecEnv([make_env(i) for i in range(num_envs)],
                        start_method="spawn")
    vec = VecMonitor(vec)
    model = PPO("MlpPolicy", vec, n_steps=n_steps, batch_size=batch_size,
                n_epochs=n_epochs, device=device, verbose=0,
                policy_kwargs=dict(net_arch=arch))
    timer = SplitTimer()
    total = iters * n_steps * num_envs
    t0 = time.time()
    model.learn(total_timesteps=total, callback=timer, progress_bar=False)
    wall = time.time() - t0
    vec.close()

    print("=" * 66)
    print(f"PPO SPLIT  device={model.device}  num_envs={num_envs} "
          f"n_steps={n_steps} batch={batch_size} epochs={n_epochs} arch={arch}")
    print("=" * 66)
    # First collect has cold-start; report mean of later iterations.
    col = np.asarray(timer.collect[1:]) if len(timer.collect) > 1 else np.asarray(timer.collect)
    upd = np.asarray(timer.update) if timer.update else np.asarray([0.0])
    rollout_steps = n_steps * num_envs
    print(f"  iterations measured: collect={len(timer.collect)} update={len(timer.update)}")
    print(f"  COLLECT: {col.mean():6.2f}s/iter  -> {rollout_steps/col.mean():7.1f} fps")
    print(f"  UPDATE : {upd.mean():6.2f}s/iter")
    tot = col.mean() + upd.mean()
    print(f"  RATIO  : collect {col.mean()/tot*100:4.1f}%  |  update {upd.mean()/tot*100:4.1f}%")
    print(f"  EFFECTIVE fps = {rollout_steps/tot:7.1f}  (wall total {wall:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["physics", "ppo"], required=True)
    ap.add_argument("--substeps", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-envs", type=int, default=72)
    ap.add_argument("--n-steps", type=int, default=341)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--net-arch", type=str, default="256,256")
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "physics":
        mode_physics(args.substeps, args.steps)
    else:
        mode_ppo(args.device, args.num_envs, args.n_steps, args.batch_size,
                 args.n_epochs, args.net_arch, args.iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
