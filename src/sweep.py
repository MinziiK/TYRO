"""Optuna sweep over PPO + reward hyperparameters on TyroEnv.

Phase-2 lookahead (per progress report §3): explore the PPO and cooperation-
reward weight space to push success rate beyond what manual tuning achieves.

Examples
--------
    # Smoke test: 1 trial × 5k steps to verify the full loop.
    python -m src.sweep --study tyro1 --smoke-test

    # Real sweep: 30 trials × 200k steps, 4 envs, stage 3.
    python -m src.sweep --study tyro1 --n-trials 30 --steps-per-trial 200000

    # Resume the same study (storage is SQLite — runs/sweep/<study>.db).
    python -m src.sweep --study tyro1 --n-trials 20

Notes
-----
* Score = best mean eval reward seen during the trial (EvalCallback).
* MedianPruner cancels under-performing trials after the warmup window.
* TPESampler is Optuna's Bayesian default; good for ≤ a few dozen trials.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecMonitor,
)

from src.config import EnvConfig, make_env_config
from src.env import TyroEnv
from src.train import make_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Pruning bridge: report each EvalCallback evaluation back to Optuna.
# ----------------------------------------------------------------------
class OptunaPruningCallback(BaseCallback):
    """Forwards EvalCallback's latest mean reward to ``trial.report()``.

    Optuna's MedianPruner uses these reports to decide if a trial is
    falling below the cohort median and should be killed early.
    """
    def __init__(self, trial: optuna.Trial, eval_cb: EvalCallback,
                 verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_cb = eval_cb
        self._last_seen: float = -np.inf

    def _on_step(self) -> bool:
        last = float(self.eval_cb.last_mean_reward)
        # last_mean_reward is updated in-place by EvalCallback after each eval.
        # Detect "new evaluation completed" by checking for a change.
        if last != self._last_seen and np.isfinite(last):
            self._last_seen = last
            self.trial.report(last, step=int(self.num_timesteps))
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True


# ----------------------------------------------------------------------
# Search space (edit here to tune what Optuna explores)
# ----------------------------------------------------------------------
def suggest_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Returns a dict of hyperparameters drawn from the trial."""
    p = dict(
        # PPO core
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        n_steps=trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        n_epochs=trial.suggest_int("n_epochs", 4, 15),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        ent_coef=trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
        # Cooperation reward (spec §4.1.3): r_coop = w_c · exp(-α d_A) · exp(-β d_B)
        w_c=trial.suggest_float("w_c", 0.5, 5.0),
        alpha=trial.suggest_float("alpha", 5.0, 20.0),
        beta=trial.suggest_float("beta", 5.0, 20.0),
    )
    return p


def _build_cfg(stage: int, phase: int, params: Dict[str, Any]) -> EnvConfig:
    cfg = make_env_config(stage=stage, phase=phase)
    cfg.reward.w_c = float(params["w_c"])
    cfg.reward.alpha = float(params["alpha"])
    cfg.reward.beta = float(params["beta"])
    return cfg


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------
def make_objective(args: argparse.Namespace) -> Callable[[optuna.Trial], float]:
    sweep_root = Path(args.out) / args.study

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hparams(trial)
        # PPO requires (n_steps × num_envs) to be divisible by batch_size.
        if (params["n_steps"] * args.num_envs) % params["batch_size"] != 0:
            raise optuna.TrialPruned()

        trial_dir = sweep_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        cfg_factory = lambda: _build_cfg(args.stage, args.phase, params)
        trial_seed = args.seed + 1000 * trial.number

        if args.num_envs > 1:
            vec = SubprocVecEnv(
                [make_env(i, cfg_factory, trial_seed) for i in range(args.num_envs)],
                start_method="spawn",
            )
        else:
            vec = DummyVecEnv([make_env(0, cfg_factory, trial_seed)])
        vec = VecMonitor(vec, filename=str(trial_dir / "monitor.csv"),
                         info_keywords=("is_success", "termination"))

        eval_env = DummyVecEnv([make_env(0, cfg_factory, trial_seed + 99_999)])
        eval_env = VecMonitor(eval_env, info_keywords=("is_success", "termination"))

        model = PPO(
            "MlpPolicy", vec,
            learning_rate=params["lr"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=0.99,
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            verbose=0,
            device=args.device,
            seed=trial_seed,
            tensorboard_log=str(trial_dir / "tb"),
            policy_kwargs=dict(net_arch=[256, 256]),
        )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(trial_dir / "best"),
            log_path=str(trial_dir / "eval"),
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            verbose=0,
        )
        prune_cb = OptunaPruningCallback(trial, eval_cb)

        try:
            model.learn(
                total_timesteps=args.steps_per_trial,
                callback=CallbackList([eval_cb, prune_cb]),
                progress_bar=False,
            )
        finally:
            try:
                vec.close()
                eval_env.close()
            except Exception:
                pass

        score = float(eval_cb.best_mean_reward)
        dur = (time.time() - t0) / 60.0
        print(f"[trial {trial.number:>3d}] score={score:+.3f}  "
              f"({dur:.1f} min)  params={ {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()} }")
        return score

    return objective


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=str, required=True,
                    help="Study name. Used for SQLite filename and run dir.")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Trials to run in this invocation (resumes prior trials).")
    ap.add_argument("--steps-per-trial", type=int, default=200_000)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--eval-freq", type=int, default=25_000,
                    help="Global env steps between eval evaluations (per env).")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "runs" / "sweep"))
    ap.add_argument("--timeout", type=int, default=None,
                    help="Wall-clock seconds to spend on this invocation.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="1 trial × 5k steps, single env. Verifies full loop.")
    args = ap.parse_args()

    set_random_seed(args.seed)
    if args.smoke_test:
        args.n_trials = 1
        args.steps_per_trial = 5_000
        args.num_envs = 1
        args.eval_freq = 2_500
        args.eval_episodes = 2

    sweep_root = Path(args.out) / args.study
    sweep_root.mkdir(parents=True, exist_ok=True)
    storage_path = sweep_root / f"{args.study}.db"
    storage = f"sqlite:///{storage_path}"
    print(f"[sweep] storage = {storage}")
    print(f"[sweep] run dir = {sweep_root}")
    print(f"[sweep] {args.n_trials} trial(s) × {args.steps_per_trial:,} steps, "
          f"num_envs={args.num_envs}, stage={args.stage}, phase={args.phase}")

    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        sampler=TPESampler(seed=args.seed),
        # Don't prune until we have a few completed trials and some warmup steps.
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50_000),
        direction="maximize",
        load_if_exists=True,
    )

    t0 = time.time()
    study.optimize(make_objective(args), n_trials=args.n_trials,
                   timeout=args.timeout, gc_after_trial=True)
    dur = (time.time() - t0) / 60.0
    print(f"\n[sweep] done in {dur:.1f} min")

    # Summary
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  completed: {len(completed)}   pruned: {len(pruned)}   "
          f"total: {len(study.trials)}")
    if completed:
        best = study.best_trial
        print(f"  best trial: #{best.number}  score={best.value:+.3f}")
        for k, v in best.params.items():
            vs = f"{v:.6g}" if isinstance(v, float) else str(v)
            print(f"    {k}: {vs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
