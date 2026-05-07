"""PPO training script for TyroEnv (centralized policy, spec §1, §8).

Examples
--------
    # Stage-1 warmup, 4 parallel envs, 1 M steps:
    python -m src.train --stage 1 --total-steps 1_000_000

    # Continue from stage-1 checkpoint, enable cooperation + sparse bonus:
    python -m src.train --stage 3 --resume runs/stage1_*/final.zip --total-steps 3_000_000

    # Full curriculum (stage 3 + DR phase 3, GPU, longer horizon):
    python -m src.train --stage 3 --phase 3 --total-steps 5_000_000 --device cuda

    # With W&B (set WANDB_API_KEY env var first):
    python -m src.train --wandb tyro --tags stage3,phase1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecMonitor,
)

from src.config import EnvConfig, make_env_config
from src.env import TyroEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_env(rank: int, cfg_factory: Callable[[], EnvConfig], seed: int):
    """Returns a thunk that builds one TyroEnv. Used by SubprocVecEnv."""
    def _init():
        env = TyroEnv(cfg=cfg_factory(), seed=seed + rank)
        return env
    return _init


class RewardBreakdownCallback(BaseCallback):
    """Logs per-term reward means + success rate to TB every rollout.

    Reads `info["reward_terms"]` (set by TyroEnv) and `info["is_success"]`.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._term_sums: dict = {}
        self._term_count: int = 0
        self._success_episodes: int = 0
        self._total_episodes: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            terms = info.get("reward_terms")
            if terms:
                for k, v in terms.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        self._term_sums[k] = self._term_sums.get(k, 0.0) + float(v)
                self._term_count += 1
            # SB3 exposes the final info on episode end at info["episode"]
            if "episode" in info:
                self._total_episodes += 1
                if info.get("is_success", False):
                    self._success_episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._term_count > 0:
            for k, s in self._term_sums.items():
                self.logger.record(f"reward/{k}", s / self._term_count)
        if self._total_episodes > 0:
            self.logger.record("rollout/success_rate",
                               self._success_episodes / self._total_episodes)
        self._term_sums = {}
        self._term_count = 0
        self._success_episodes = 0
        self._total_episodes = 0


def build_callbacks(args, eval_env, out_dir: Path) -> CallbackList:
    cbs: list[BaseCallback] = []
    cbs.append(RewardBreakdownCallback())
    cbs.append(CheckpointCallback(
        save_freq=max(args.save_freq // max(args.num_envs, 1), 1),
        save_path=str(out_dir / "ckpts"),
        name_prefix="ppo",
        save_replay_buffer=False,
    ))
    cbs.append(EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.num_envs, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    ))
    if args.wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(
                project=args.wandb,
                name=out_dir.name,
                config=vars(args),
                sync_tensorboard=True,
                tags=args.tags.split(",") if args.tags else None,
                dir=str(out_dir),
            )
            cbs.append(WandbCallback(
                model_save_path=str(out_dir / "wandb_models"),
                model_save_freq=args.save_freq,
                verbose=1,
            ))
        except ImportError:
            print("[train] wandb not installed; skipping W&B logging.")
    return CallbackList(cbs)


def main() -> int:
    ap = argparse.ArgumentParser()
    # Curriculum
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4],
                    help="Reward stage (spec §4.3): 1=align+reach, 2=+coop, 3=full dense, 4=potential shaping.")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                    help="Domain-randomization phase (spec §6): 1=fixed, 2=±2cm, 3=±5cm.")

    # Compute / parallelism
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto",
                    help='"cpu", "cuda", or "auto".')
    ap.add_argument("--seed", type=int, default=0)

    # Schedule
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--save-freq", type=int, default=50_000,
                    help="Checkpoint every N global env steps.")
    ap.add_argument("--eval-freq", type=int, default=25_000,
                    help="Run eval every N global env steps.")
    ap.add_argument("--eval-episodes", type=int, default=10)

    # PPO hyperparams (sb3 defaults that work well for continuous control)
    ap.add_argument("--n-steps", type=int, default=2048,
                    help="Rollout length per env before each PPO update.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--net-arch", type=str, default="256,256",
                    help="Comma-separated hidden layer widths for MlpPolicy.")

    # IO
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a .zip checkpoint to load weights from.")

    # Logging
    ap.add_argument("--wandb", type=str, default=None,
                    help="W&B project name. Omit to disable.")
    ap.add_argument("--tags", type=str, default=None,
                    help="Comma-separated W&B tags.")

    args = ap.parse_args()
    set_random_seed(args.seed)

    # ------------------------------------------------------------------
    # Run dir
    # ------------------------------------------------------------------
    if args.run_name is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.run_name = f"stage{args.stage}_phase{args.phase}_{ts}"
    out_dir = Path(args.out) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir: {out_dir}")

    # ------------------------------------------------------------------
    # Vec env
    # ------------------------------------------------------------------
    cfg_factory = lambda: make_env_config(stage=args.stage, phase=args.phase)

    if args.num_envs > 1:
        vec = SubprocVecEnv(
            [make_env(i, cfg_factory, args.seed) for i in range(args.num_envs)],
            start_method="spawn",
        )
    else:
        vec = DummyVecEnv([make_env(0, cfg_factory, args.seed)])
    vec = VecMonitor(vec, filename=str(out_dir / "monitor.csv"),
                     info_keywords=("is_success", "termination"))

    eval_env = DummyVecEnv([make_env(0, cfg_factory, args.seed + 10_000)])
    eval_env = VecMonitor(eval_env, info_keywords=("is_success", "termination"))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    net_arch = [int(w) for w in args.net_arch.split(",") if w]
    policy_kwargs = dict(net_arch=net_arch)

    if args.resume:
        print(f"[train] loading weights from {args.resume}")
        model = PPO.load(args.resume, env=vec, device=args.device)
        # Override hyperparams in case stage changes between resumes.
        model.learning_rate = args.lr
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.n_epochs = args.n_epochs
        model.gamma = args.gamma
        model.gae_lambda = args.gae_lambda
        model.clip_range = args.clip_range
        model.ent_coef = args.ent_coef
        model.vf_coef = args.vf_coef
        model.max_grad_norm = args.max_grad_norm
        model.tensorboard_log = str(out_dir / "tb")
    else:
        model = PPO(
            "MlpPolicy", vec,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            device=args.device,
            seed=args.seed,
            tensorboard_log=str(out_dir / "tb"),
            policy_kwargs=policy_kwargs,
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    callback = build_callbacks(args, eval_env, out_dir)
    print(f"[train] device={model.device}  policy net_arch={net_arch}")
    print(f"[train] total {args.total_steps:,} steps over {args.num_envs} envs")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            tb_log_name="ppo",
            reset_num_timesteps=(args.resume is None),
            progress_bar=False,
        )
    finally:
        final_path = out_dir / "final.zip"
        model.save(final_path)
        print(f"[train] saved final to {final_path}  ({(time.time() - t0)/60:.1f} min)")
        try:
            vec.close()
            eval_env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
