"""PPO training script for TyroEnv (centralized policy, spec §1, §8).

Examples
--------
    # Phase 1 (reach & align warmup, stage 1 + curriculum phase 1):
    python -m src.train --task phase1 --num-envs 8 --total-steps 500_000

    # Equivalent explicit flags:
    python -m src.train --stage 1 --phase 1 --num-envs 8 --total-steps 500_000

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
from typing import Callable, Optional, Tuple

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
    """Logs per-term reward means + success rate + physics metrics each rollout.

    Note: ``reward/{term}`` values are **per-step means** across the rollout
    (sum of term values / number of steps observed), not per-episode means.
    Cross-reference with ``rollout/success_rate`` (per-episode) and
    ``env/contact_force_mean`` (per-step) accordingly.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._term_sums: dict = {}
        self._term_count: int = 0
        self._success_episodes: int = 0
        self._total_episodes: int = 0
        self._cf_sum: float = 0.0
        self._cf_n: int = 0
        self._ik_a_sum: float = 0.0
        self._ik_b_sum: float = 0.0
        self._ik_n: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "contact_force_max" in info:
                self._cf_sum += float(info["contact_force_max"])
                self._cf_n += 1
            if "ik_residual_A" in info:
                self._ik_a_sum += float(info["ik_residual_A"])
                self._ik_b_sum += float(info.get("ik_residual_B", 0.0))
                self._ik_n += 1
            terms = info.get("reward_terms")
            if terms:
                for k, v in terms.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        self._term_sums[k] = self._term_sums.get(k, 0.0) + float(v)
                self._term_count += 1
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
        if self._cf_n > 0:
            self.logger.record("env/contact_force_mean", self._cf_sum / self._cf_n)
        if self._ik_n > 0:
            # Each episode contributes one inflated step-1 sample (~0.5 m) from
            # the tire-grasp constraint impulse — see ``TyroEnv.reset()``. The
            # rollout-mean partially absorbs it; what matters is the trend over
            # training. Sustained mean > 0.01–0.02 m on a long rollout signals
            # IK saturation (UR10 near reach limit, joint clamp); inspect the
            # layout / DR ranges.
            self.logger.record("env/ik_residual_A_mean", self._ik_a_sum / self._ik_n)
            self.logger.record("env/ik_residual_B_mean", self._ik_b_sum / self._ik_n)
        self._term_sums = {}
        self._term_count = 0
        self._success_episodes = 0
        self._total_episodes = 0
        self._cf_sum = 0.0
        self._cf_n = 0
        self._ik_a_sum = 0.0
        self._ik_b_sum = 0.0
        self._ik_n = 0


class ApproachTolCurriculumCallback(BaseCallback):
    """Schedules the Stage 0 → 1 pickup gate ``approach_tol`` over training.

    Layout — three regimes keyed on the PPO ``num_timesteps`` counter:

      ``t ≤ soft_steps``                       → tol = ``soft``
      ``soft_steps < t ≤ soft + ramp``         → tol = linear(soft → hard)
      ``t  > soft + ramp``                      → tol = ``hard``

    The schedule re-broadcasts the current tol to **all** sub-envs at
    every rollout boundary via ``env_method``. Using ``num_timesteps``
    (the global PPO step counter, summed across ``num_envs`` workers)
    keeps the schedule independent of vec-env parallelism and exactly
    matches the CLI / config view of "total timesteps".

    Logged scalars (TensorBoard / monitor):
      - ``curriculum/approach_tol``       — current scheduled tol (m)
      - ``curriculum/approach_tol_frac``  — interpolation progress 0..1
    """

    def __init__(
        self,
        soft: float,
        hard: float,
        soft_steps: int,
        ramp_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._soft = float(soft)
        self._hard = float(hard)
        self._soft_steps = int(max(0, soft_steps))
        self._ramp_steps = int(max(1, ramp_steps))
        self._last_pushed: Optional[float] = None

    def _scheduled_tol(self, t: int) -> Tuple[float, float]:
        if t <= self._soft_steps:
            return self._soft, 0.0
        frac = (t - self._soft_steps) / float(self._ramp_steps)
        if frac >= 1.0:
            return self._hard, 1.0
        return self._soft * (1.0 - frac) + self._hard * frac, frac

    def _broadcast(self, tol: float) -> None:
        try:
            self.training_env.env_method("set_approach_tol", tol)
        except AttributeError:
            # DummyVecEnv path (num_envs == 1) supports env_method too;
            # fall back silently if a wrapper hides it.
            pass

    def _on_training_start(self) -> None:
        tol, _ = self._scheduled_tol(int(self.model.num_timesteps))
        self._last_pushed = tol
        self._broadcast(tol)
        if self.verbose:
            print(f"[curriculum] approach_tol init = {tol:.3f} m "
                  f"(t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        # Cheap to call every rollout step — env_method is a no-op when
        # the value hasn't changed (we guard with ``_last_pushed``).
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        tol, frac = self._scheduled_tol(t)
        if self._last_pushed is None or abs(tol - self._last_pushed) > 1e-6:
            self._broadcast(tol)
            self._last_pushed = tol
            if self.verbose:
                print(f"[curriculum] approach_tol = {tol:.3f} m "
                      f"(t={t}, frac={frac:.3f})")
        self.logger.record("curriculum/approach_tol", float(tol))
        self.logger.record("curriculum/approach_tol_frac", float(frac))


def build_callbacks(args, eval_env, out_dir: Path) -> CallbackList:
    cbs: list[BaseCallback] = []
    cbs.append(RewardBreakdownCallback())
    cbs.append(ApproachTolCurriculumCallback(
        soft=args.approach_tol_soft,
        hard=args.approach_tol_hard,
        soft_steps=args.approach_tol_curriculum_steps,
        ramp_steps=args.approach_tol_ramp_steps,
        verbose=1,
    ))
    cbs.append(CheckpointCallback(
        save_freq=max(args.save_freq // max(args.num_envs, 1), 1),
        save_path=str(out_dir / "ckpts"),
        name_prefix="ppo",
        save_replay_buffer=False,
    ))
    if not args.no_eval_callback:
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
    ap.add_argument(
        "--task",
        type=str,
        default=None,
        choices=("phase1", "phase2", "phase3"),
        help='Optional shorthand: phase1=("stage", "phase")=(1,1) reach & align warmup; '
             "phase2=(2,1); phase3=(3,1) full SB3 rollout defaults.",
    )

    # Compute / parallelism
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto",
                    help='"cpu", "cuda", or "auto".')
    ap.add_argument("--seed", type=int, default=0)

    # Schedule (``--total-timesteps`` is an alias used in docs / shell snippets)
    ap.add_argument("--total-steps", "--total-timesteps", type=int, default=1_000_000,
                    dest="total_steps")

    # Scene overrides (omit to keep ``EnvConfig`` defaults)
    ap.add_argument(
        "--use-truck-hub-urdf",
        type=str,
        default="",
        metavar="BOOL",
        help="true/false: load ``truck_wheel_station`` URDF hub. Omit = default from ``EnvConfig``.",
    )
    ap.add_argument(
        "--spawn-cargo-box",
        type=str,
        default="",
        metavar="BOOL",
        help="true/false: vehicle / wheel-well primitive box scene. Omit = EnvConfig.",
    )

    # Reward mix overrides (applied after ``make_reward_config(stage)``; Phase‑1 preset is already dense-only)
    ap.add_argument("--mix-dense", type=float, default=None,
                    help="Weight on dense process reward branch (override ``RewardConfig.mix_dense``).")
    ap.add_argument("--mix-sparse-success", type=float, default=None,
                    help="Weight on sparse success (override ``RewardConfig.mix_sparse_success``).")
    ap.add_argument(
        "--dense-baseline-with-shaping",
        type=str,
        default="",
        metavar="BOOL",
        help=(
            "true/false: when shaping is on (stage 4), keep the absolute "
            "distance penalty (align_A+reach_B) as a baseline so the policy is "
            "not driven by per-step state diffs alone. Empty = RewardConfig "
            "default (True)."
        ),
    )
    ap.add_argument(
        "--dense-baseline-scale",
        type=float,
        default=None,
        help=(
            "Scale on the absolute-distance baseline when blended with shaping "
            "(override ``RewardConfig.w_dense_baseline_scale``). 1.0 keeps the "
            "same magnitude as stages 1–3; lower it to let shaping dominate."
        ),
    )
    ap.add_argument("--save-freq", type=int, default=50_000,
                    help="Checkpoint every N global env steps.")
    ap.add_argument("--eval-freq", type=int, default=25_000,
                    help="Run eval every N global env steps.")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument(
        "--no-eval-callback",
        action="store_true",
        help="Disable EvalCallback (faster; useful for Phase‑1 smoke or when eval resets are heavy).",
    )

    # PPO hyperparams (sb3 defaults that work well for continuous control)
    ap.add_argument("--n-steps", type=int, default=2048,
                    help="Rollout length per env before each PPO update.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--net-arch", type=str, default="256,256",
                    help="Comma-separated hidden layer widths for MlpPolicy.")

    # Pickup-gate curriculum (Stage 0 → 1 trigger) — defaults follow EnvConfig.
    _cfg_defaults = EnvConfig()
    ap.add_argument(
        "--approach-tol-soft",
        type=float,
        default=_cfg_defaults.approach_tol_soft,
        help=(
            "Soft / lenient pickup-gate radius (m) used until "
            "``--approach-tol-curriculum-steps`` global steps elapse. "
            "Lets a pre-trained hover policy still pay R_pickup early "
            "in the new raised-rack layout."
        ),
    )
    ap.add_argument(
        "--approach-tol-hard",
        type=float,
        default=_cfg_defaults.approach_radius_tol,
        help=(
            "Final / strict pickup-gate radius (m) the curriculum "
            "asymptotes to. Defaults to ``EnvConfig.approach_radius_tol``."
        ),
    )
    ap.add_argument(
        "--approach-tol-curriculum-steps",
        type=int,
        default=_cfg_defaults.approach_tol_curriculum_steps,
        help="Hold ``--approach-tol-soft`` for this many global PPO steps.",
    )
    ap.add_argument(
        "--approach-tol-ramp-steps",
        type=int,
        default=_cfg_defaults.approach_tol_ramp_steps,
        help=(
            "Linear ramp length (global PPO steps) from soft to hard "
            "after the hold phase."
        ),
    )

    # Physics overrides (Bulleted defaults live in EnvConfig)
    ap.add_argument("--physics-num-sub-steps", type=int, default=None)
    ap.add_argument("--contact-erp", type=float, default=None)
    ap.add_argument("--contact-cfm", type=float, default=None)
    ap.add_argument("--contact-force-done", type=float, default=None,
                    help="Terminate if normal force exceeds this (N); ≤0 disables.")
    ap.add_argument("--tire-mass", type=float, default=None,
                    help="Tire base mass in kg (default: EnvConfig.tire_mass = 1.0).")

    # IO
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a .zip checkpoint to load weights from.")
    ap.add_argument(
        "--resume-mode",
        type=str,
        default="policy-only",
        choices=("policy-only", "full"),
        help=(
            "policy-only: load weights only (Adam/LR schedule/value head reset; "
            "use this for stage transitions, e.g. stage1→stage3). "
            "full: restore the entire PPO state (optimizer, schedules, "
            "num_timesteps, RMS stats); CLI PPO hparams are then ignored."
        ),
    )

    # Logging
    ap.add_argument("--wandb", type=str, default=None,
                    help="W&B project name. Omit to disable.")
    ap.add_argument("--tags", type=str, default=None,
                    help="Comma-separated W&B tags.")

    args = ap.parse_args()
    if args.task == "phase1":
        args.stage, args.phase = 1, 1
    elif args.task == "phase2":
        args.stage, args.phase = 2, 1
    elif args.task == "phase3":
        args.stage, args.phase = 3, 1
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
    def cfg_factory():
        overrides = {}
        if args.physics_num_sub_steps is not None:
            overrides["physics_num_sub_steps"] = args.physics_num_sub_steps
        if args.contact_erp is not None:
            overrides["contact_erp"] = args.contact_erp
        if args.contact_cfm is not None:
            overrides["contact_cfm"] = args.contact_cfm
        if args.contact_force_done is not None:
            overrides["contact_force_terminate_above"] = args.contact_force_done
        if args.tire_mass is not None:
            overrides["tire_mass"] = args.tire_mass
        if args.use_truck_hub_urdf.strip():
            s = args.use_truck_hub_urdf.strip().lower()
            overrides["use_truck_hub_urdf"] = s in ("1", "true", "t", "yes", "y")
        if args.spawn_cargo_box.strip():
            s = args.spawn_cargo_box.strip().lower()
            overrides["spawn_vehicle_primitive_box"] = s in ("1", "true", "t", "yes", "y")
        cfg = make_env_config(stage=args.stage, phase=args.phase, **overrides)
        if args.mix_dense is not None:
            cfg.reward.mix_dense = float(args.mix_dense)
        if args.mix_sparse_success is not None:
            cfg.reward.mix_sparse_success = float(args.mix_sparse_success)
        if args.dense_baseline_with_shaping.strip():
            s = args.dense_baseline_with_shaping.strip().lower()
            cfg.reward.use_dense_baseline_with_shaping = s in (
                "1", "true", "t", "yes", "y"
            )
        if args.dense_baseline_scale is not None:
            cfg.reward.w_dense_baseline_scale = float(args.dense_baseline_scale)
        return cfg

    if args.num_envs > 1:
        vec = SubprocVecEnv(
            [make_env(i, cfg_factory, args.seed) for i in range(args.num_envs)],
            start_method="spawn",
        )
    else:
        vec = DummyVecEnv([make_env(0, cfg_factory, args.seed)])
    vec = VecMonitor(vec, filename=str(out_dir / "monitor.csv"),
                     info_keywords=("is_success", "termination"))

    # Skip the eval env entirely when EvalCallback is disabled — PyBullet
    # connect + URDF load is non-trivial and otherwise wasted.
    if args.no_eval_callback:
        eval_env = None
    else:
        eval_env = DummyVecEnv([make_env(0, cfg_factory, args.seed + 10_000)])
        eval_env = VecMonitor(eval_env, info_keywords=("is_success", "termination"))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    net_arch = [int(w) for w in args.net_arch.split(",") if w]
    policy_kwargs = dict(net_arch=net_arch)

    if args.resume and args.resume_mode == "full":
        # Full resume: restore optimizer, LR/clip schedules, num_timesteps, and
        # rollout buffer config from the checkpoint. CLI PPO hparam args are
        # IGNORED in this mode — the checkpoint is the source of truth.
        print(f"[train] full resume from {args.resume} "
              f"(CLI PPO hyperparameters will be ignored)")
        model = PPO.load(
            args.resume,
            env=vec,
            device=args.device,
            tensorboard_log=str(out_dir / "tb"),
        )
    else:
        # Build a fresh PPO with the requested hyperparameters so SB3's internal
        # state (rollout buffer size, LR/clip schedules, optimizer) stays
        # consistent with CLI args. policy-only resume then transfers only the
        # weights — mutating a loaded model's hyperparameters in place corrupts
        # those schedules (clip_range/learning_rate are stored as callables, a
        # bare-float assign either crashes the next update or is silently ignored).
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

        if args.resume:
            print(f"[train] policy-only resume from {args.resume}")
            ckpt = PPO.load(args.resume, device=args.device)
            model.policy.load_state_dict(ckpt.policy.state_dict())
            model.num_timesteps = ckpt.num_timesteps
            del ckpt

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
            if eval_env is not None:
                eval_env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
