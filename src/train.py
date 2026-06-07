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

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _scheduled_tol(self, t: int) -> Tuple[float, float]:
        if t <= self._soft_steps:
            return self._soft, 0.0
        frac = (t - self._soft_steps) / float(self._ramp_steps)
        if frac >= 1.0:
            return self._hard, 1.0
        s = self._smoothstep(frac)
        return self._soft * (1.0 - s) + self._hard * s, frac

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


class StartPosCurriculumCallback(BaseCallback):
    """Schedules the Stage 0 UR10 starting-pose blend ``alpha`` over training.

    Layout — three regimes keyed on the PPO ``num_timesteps`` counter:

      ``t ≤ easy_steps``                       → alpha = 0  (full easy)
      ``easy_steps < t ≤ easy + ramp``         → alpha = smoothstep(0 → 1)
      ``t  > easy + ramp``                      → alpha = 1  (full HOME)

    The scheduled alpha is broadcast to all sub-envs via ``env_method`` at
    every rollout boundary. Each env's next ``reset()`` calls
    ``_apply_start_pos_curriculum(alpha)`` to teleport the UR10 EE to
    ``lerp(easy_start, HOME_EE, alpha)``.

    Logged scalars:
      - ``curriculum/start_pos_alpha``       — current blend 0..1
      - ``curriculum/start_pos_alpha_frac``  — raw ramp progress 0..1
    """

    def __init__(
        self,
        easy_steps: int,
        ramp_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._easy_steps = int(max(0, easy_steps))
        self._ramp_steps = int(max(1, ramp_steps))
        self._last_pushed: Optional[float] = None

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _scheduled_alpha(self, t: int) -> Tuple[float, float]:
        if t <= self._easy_steps:
            return 0.0, 0.0
        frac = (t - self._easy_steps) / float(self._ramp_steps)
        if frac >= 1.0:
            return 1.0, 1.0
        return self._smoothstep(frac), frac

    def _broadcast(self, alpha: float) -> None:
        try:
            self.training_env.env_method("set_start_pos_alpha", alpha)
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        alpha, _ = self._scheduled_alpha(int(self.model.num_timesteps))
        self._last_pushed = alpha
        self._broadcast(alpha)
        if self.verbose:
            print(f"[curriculum] start_pos_alpha init = {alpha:.3f} "
                  f"(t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        alpha, frac = self._scheduled_alpha(t)
        if self._last_pushed is None or abs(alpha - self._last_pushed) > 1e-6:
            self._broadcast(alpha)
            self._last_pushed = alpha
            if self.verbose:
                print(f"[curriculum] start_pos_alpha = {alpha:.3f} "
                      f"(t={t}, frac={frac:.3f})")
        self.logger.record("curriculum/start_pos_alpha", float(alpha))
        self.logger.record("curriculum/start_pos_alpha_frac", float(frac))


class MountTolCurriculumCallback(BaseCallback):
    """Drives the Stage 1 → 2 mount gate (radius, angle) over training.

    Both tolerances fade from soft → hard along a single progress scalar
    ``frac ∈ [0, 1]`` via one smoothstep, so the easy/hard regime stays
    consistent across the radius and axis checks. How ``frac`` moves is
    selected by ``mode``:

    ``"adaptive"`` (default) — a **success-gated, advance-and-rollback**
        loop. After a soft-step warm-up, each rollout reads the recent
        episode success rate from ``model.ep_info_buffer`` and:
          - advances (``frac += step_up``)  when sr ≥ ``advance_sr``
          - rolls back (``frac -= step_down``) when sr ≤ ``rollback_sr``
          - holds otherwise (hysteresis band between the two thresholds).
        A dwell period after every change lets the policy and the success
        estimate settle before the next decision. This closes the loop the
        legacy schedule left open: difficulty only tightens while the
        policy can keep up, and a collapse is actively unwound instead of
        being baked in for the rest of training.

    ``"schedule"`` — the legacy open-loop ``num_timesteps`` smoothstep
        (``soft_steps`` hold, then ``ramp_steps`` ramp). Kept for
        reproducing pre-adaptive runs; not recommended for new training.
    """

    def __init__(
        self,
        radius_soft: float, radius_hard: float,
        angle_soft_rad: float, angle_hard_rad: float,
        soft_steps: int, ramp_steps: int, verbose: int = 0,
        eval_env=None,
        mode: str = "adaptive",
        advance_sr: float = 0.80,
        rollback_sr: float = 0.55,
        step_up: float = 0.05,
        step_down: float = 0.10,
        min_episodes: int = 40,
        dwell_rollouts: int = 3,
    ):
        super().__init__(verbose)
        self._r_soft = float(radius_soft)
        self._r_hard = float(radius_hard)
        self._a_soft = float(angle_soft_rad)
        self._a_hard = float(angle_hard_rad)
        self._soft_steps = int(max(0, soft_steps))
        self._ramp_steps = int(max(1, ramp_steps))
        self._last_pushed: Optional[Tuple[float, float]] = None
        # The EvalCallback's env is a *separate* vec-env not reachable via
        # ``self.training_env``; broadcast the schedule to it too so eval
        # measures success at the current curriculum difficulty (otherwise
        # eval stays pinned at the hard config tol and reads 0% until the
        # ramp finishes). ``None`` when --no-eval-callback.
        self._eval_env = eval_env
        # --- adaptive-mode state / knobs ---
        self._mode = str(mode)
        self._advance_sr = float(advance_sr)
        self._rollback_sr = float(rollback_sr)
        self._step_up = float(step_up)
        self._step_down = float(step_down)
        self._min_episodes = int(max(1, min_episodes))
        self._dwell = int(max(0, dwell_rollouts))
        # Adaptive progress along soft→hard, advanced/rolled-back per rollout.
        self._frac = 0.0
        # Rollouts remaining to hold ``frac`` fixed after a change.
        self._dwell_left = 0

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _tol_for_frac(self, frac: float) -> Tuple[float, float]:
        """Map progress ``frac`` ∈ [0, 1] → (radius, angle) via smoothstep."""
        s = self._smoothstep(frac)
        r = self._r_soft * (1.0 - s) + self._r_hard * s
        a = self._a_soft * (1.0 - s) + self._a_hard * s
        return r, a

    def _scheduled(self, t: int) -> Tuple[float, float, float]:
        """Legacy open-loop schedule: frac keyed on global PPO step ``t``."""
        if t <= self._soft_steps:
            return self._r_soft, self._a_soft, 0.0
        frac = (t - self._soft_steps) / float(self._ramp_steps)
        frac = float(np.clip(frac, 0.0, 1.0))
        r, a = self._tol_for_frac(frac)
        return r, a, frac

    def _recent_success_rate(self) -> Tuple[float, int]:
        """(success_rate, n_episodes) over the model's episode-info buffer.

        ``ep_info_buffer`` carries ``is_success`` because the training
        ``VecMonitor`` is built with ``info_keywords=("is_success", ...)``.
        Older episodes without the key count as failures (False).
        """
        buf = getattr(self.model, "ep_info_buffer", None)
        if not buf:
            return 0.0, 0
        n = len(buf)
        succ = sum(1 for ep in buf if ep.get("is_success", False))
        return succ / float(n), n

    def _broadcast(self, r: float, a: float) -> None:
        try:
            self.training_env.env_method("set_mount_tol", r, a)
        except AttributeError:
            pass
        if self._eval_env is not None:
            try:
                self._eval_env.env_method("set_mount_tol", r, a)
            except (AttributeError, Exception):  # noqa: BLE001
                pass

    def _on_training_start(self) -> None:
        if self._mode == "adaptive":
            self._frac = 0.0
            r, a = self._tol_for_frac(self._frac)
        else:
            r, a, _ = self._scheduled(int(self.model.num_timesteps))
        self._last_pushed = (r, a)
        self._broadcast(r, a)
        if self.verbose:
            print(f"[curriculum] mount_tol init = ({r:.3f} m, "
                  f"{np.degrees(a):.1f}°) mode={self._mode} "
                  f"(t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _adaptive_update(self, t: int) -> Tuple[float, float, float]:
        """Advance / hold / roll back ``self._frac`` from recent success."""
        sr, n = self._recent_success_rate()
        self.logger.record("curriculum/mount_recent_success", float(sr))
        self.logger.record("curriculum/mount_recent_episodes", int(n))
        # Warm-up: keep the gate soft until the policy has had some steps
        # *and* enough episodes have accrued for a trustworthy estimate.
        warming = t < self._soft_steps or n < self._min_episodes
        if warming or self._dwell_left > 0:
            if self._dwell_left > 0:
                self._dwell_left -= 1
            r, a = self._tol_for_frac(self._frac)
            return r, a, self._frac
        prev = self._frac
        if sr >= self._advance_sr and self._frac < 1.0:
            self._frac = min(1.0, self._frac + self._step_up)
        elif sr <= self._rollback_sr and self._frac > 0.0:
            self._frac = max(0.0, self._frac - self._step_down)
        if abs(self._frac - prev) > 1e-9:
            self._dwell_left = self._dwell
            if self.verbose:
                direction = "advance" if self._frac > prev else "ROLLBACK"
                print(f"[curriculum] mount {direction}: frac {prev:.2f}"
                      f"→{self._frac:.2f} (sr={sr:.2f} over {n} eps, t={t})")
        r, a = self._tol_for_frac(self._frac)
        return r, a, self._frac

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        if self._mode == "adaptive":
            r, a, frac = self._adaptive_update(t)
        else:
            r, a, frac = self._scheduled(t)
        changed = (
            self._last_pushed is None
            or abs(r - self._last_pushed[0]) > 1e-6
            or abs(a - self._last_pushed[1]) > 1e-6
        )
        if changed:
            self._broadcast(r, a)
            self._last_pushed = (r, a)
            if self.verbose:
                print(f"[curriculum] mount_tol = ({r:.3f} m, "
                      f"{np.degrees(a):.1f}°) (t={t}, frac={frac:.3f})")
        self.logger.record("curriculum/mount_radius_tol", float(r))
        self.logger.record("curriculum/mount_angle_tol_deg", float(np.degrees(a)))
        self.logger.record("curriculum/mount_tol_frac", float(frac))

def build_callbacks(args, eval_env, out_dir: Path) -> CallbackList:
    cbs: list[BaseCallback] = []
    cbs.append(RewardBreakdownCallback())
    if bool(getattr(args, "approach_curriculum", False)):
        cbs.append(ApproachTolCurriculumCallback(
            soft=args.approach_tol_soft,
            hard=args.approach_tol_hard,
            soft_steps=args.approach_tol_curriculum_steps,
            ramp_steps=args.approach_tol_ramp_steps,
            verbose=1,
        ))
    if args.mount_curriculum:
        cbs.append(MountTolCurriculumCallback(
            radius_soft=args.mount_radius_soft,
            radius_hard=args.mount_radius_hard,
            angle_soft_rad=float(np.deg2rad(args.mount_angle_soft_deg)),
            angle_hard_rad=float(np.deg2rad(args.mount_angle_hard_deg)),
            soft_steps=args.mount_tol_curriculum_steps,
            ramp_steps=args.mount_tol_ramp_steps,
            verbose=1,
            eval_env=eval_env,
            mode=args.mount_curriculum_mode,
            advance_sr=args.mount_adapt_advance_sr,
            rollback_sr=args.mount_adapt_rollback_sr,
            step_up=args.mount_adapt_step_up,
            step_down=args.mount_adapt_step_down,
            min_episodes=args.mount_adapt_min_episodes,
            dwell_rollouts=args.mount_adapt_dwell_rollouts,
        ))
    if args.start_pos_curriculum and str(args.start_pos_mode) == "lerp":
        # ``StartPosCurriculumCallback`` is meaningful only in lerp mode,
        # where it smoothsteps a single alpha 0→1. In v8 "mix" mode the
        # env reads ``cfg.start_pos_easy_prob`` directly per reset, so
        # adding the callback would unintentionally drive that knob to
        # 1.0 (= always hard) as training progresses, killing the easy
        # tap that mix mode is supposed to preserve.
        cbs.append(StartPosCurriculumCallback(
            easy_steps=args.start_pos_curriculum_steps,
            ramp_steps=args.start_pos_ramp_steps,
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

    # Pickup-gate curriculum (Stage 0 → 1 trigger) — opt-in; mount-only skips Stage 0.
    _cfg_defaults = EnvConfig()
    ap.add_argument(
        "--approach-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Stage 0 pickup-gate tol schedule "
            "(ApproachTolCurriculumCallback). Default OFF for "
            "terminate_on=mount / attached hot-start; turn ON for "
            "full cycle or --terminate-on pickup."
        ),
    )
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
            "Smoothstep ramp length (global PPO steps) from soft to hard "
            "after the hold phase."
        ),
    )

    # Stage 0 starting-pose curriculum — defaults follow EnvConfig.
    ap.add_argument(
        "--start-pos-curriculum",
        action=argparse.BooleanOptionalAction,
        default=_cfg_defaults.start_pos_curriculum_enable,
        help=(
            "Enable UR10 EE starting-pose curriculum (easy → HOME smoothstep). "
            "Disable with --no-start-pos-curriculum for eval-style HOME starts."
        ),
    )
    ap.add_argument(
        "--start-pos-easy-lift",
        type=float,
        default=_cfg_defaults.start_pos_easy_lift,
        help="Easy start Z offset below grasp anchor (m). Default from EnvConfig (v9: 0.10).",
    )
    ap.add_argument(
        "--start-pos-curriculum-steps",
        type=int,
        default=_cfg_defaults.start_pos_curriculum_steps,
        help="Hold full-easy start pose for this many global PPO steps.",
    )
    ap.add_argument(
        "--start-pos-ramp-steps",
        type=int,
        default=_cfg_defaults.start_pos_ramp_steps,
        help="Smoothstep ramp length (global PPO steps) from easy to HOME.",
    )
    ap.add_argument(
        "--start-pos-mode",
        type=str,
        choices=["lerp", "mix"],
        default=_cfg_defaults.start_pos_curriculum_mode,
        help=(
            "Starting-pose curriculum mode. 'lerp' (legacy) smoothsteps "
            "a single alpha 0→1 across all resets. 'mix' (v8 default) "
            "samples Bernoulli per reset: full-easy or full-hard."
        ),
    )
    ap.add_argument(
        "--start-pos-easy-prob",
        type=float,
        default=_cfg_defaults.start_pos_easy_prob,
        help=(
            "Bernoulli probability of an easy spawn each reset when "
            "--start-pos-mode=mix. Default from EnvConfig (v9: 0.75 easy / 0.25 HOME)."
        ),
    )
    ap.add_argument(
        "--approach-a-gate",
        type=float,
        default=_cfg_defaults.approach_A_gate,
        help=(
            "v11c: distance gate (m) on the Stage 0 dense approach_A "
            "kernel. Zeros far/close terms when d_approach > gate so the "
            "policy can't farm dense reward outside the grasp anchor. "
            "Set > 5.0 to disable."
        ),
    )
    ap.add_argument(
        "--terminate-on-pickup",
        action=argparse.BooleanOptionalAction,
        default=_cfg_defaults.terminate_on_pickup,
        help=(
            "[LEGACY] End episode with success immediately on Stage 0 pickup. "
            "Superseded by --terminate-on; only consulted when "
            "--terminate-on=never."
        ),
    )
    # Mount-gate curriculum (Stage 1 → 2 trigger). Defaults follow EnvConfig.
    ap.add_argument(
        "--mount-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Stage 1 → 2 mount-gate curriculum (radius/angle smoothstep). "
            "Disable with --no-mount-curriculum to use the hard EnvConfig gate."
        ),
    )
    ap.add_argument(
        "--mount-radius-soft",
        type=float,
        default=_cfg_defaults.mount_radius_tol_soft,
        help="Initial (loose) mount radius tolerance (m).",
    )
    ap.add_argument(
        "--mount-radius-hard",
        type=float,
        default=_cfg_defaults.mount_radius_tol,
        help="Final (strict) mount radius tolerance (m).",
    )
    ap.add_argument(
        "--mount-angle-soft-deg",
        type=float,
        default=float(np.degrees(_cfg_defaults.mount_angle_tol_soft_rad)),
        help="Initial (loose) tire-vs-hub axis angle tolerance (deg).",
    )
    ap.add_argument(
        "--mount-angle-hard-deg",
        type=float,
        default=float(np.degrees(_cfg_defaults.reward.delta_A)),
        help="Final (strict) tire-vs-hub axis angle tolerance (deg).",
    )
    ap.add_argument(
        "--mount-tol-curriculum-steps",
        type=int,
        default=_cfg_defaults.mount_tol_curriculum_steps,
        help="Hold soft mount gate for this many global PPO steps.",
    )
    ap.add_argument(
        "--mount-tol-ramp-steps",
        type=int,
        default=_cfg_defaults.mount_tol_ramp_steps,
        help="Smoothstep ramp length (global PPO steps) from soft to hard.",
    )
    ap.add_argument(
        "--mount-curriculum-mode",
        type=str,
        default=_cfg_defaults.mount_curriculum_mode,
        choices=("adaptive", "schedule"),
        help=(
            "Mount-gate driver. 'adaptive' (default) advances soft→hard "
            "only while recent success ≥ advance-sr and rolls back when it "
            "drops ≤ rollback-sr. 'schedule' = legacy open-loop "
            "num_timesteps smoothstep (reproduces pre-adaptive runs)."
        ),
    )
    ap.add_argument(
        "--mount-adapt-advance-sr",
        type=float,
        default=_cfg_defaults.mount_adapt_advance_sr,
        help="Adaptive mode: advance difficulty when recent success ≥ this.",
    )
    ap.add_argument(
        "--mount-adapt-rollback-sr",
        type=float,
        default=_cfg_defaults.mount_adapt_rollback_sr,
        help="Adaptive mode: roll difficulty back when recent success ≤ this.",
    )
    ap.add_argument(
        "--mount-adapt-step-up",
        type=float,
        default=_cfg_defaults.mount_adapt_step_up,
        help="Adaptive mode: difficulty (frac) increment per advance.",
    )
    ap.add_argument(
        "--mount-adapt-step-down",
        type=float,
        default=_cfg_defaults.mount_adapt_step_down,
        help="Adaptive mode: difficulty (frac) decrement per rollback.",
    )
    ap.add_argument(
        "--mount-adapt-min-episodes",
        type=int,
        default=_cfg_defaults.mount_adapt_min_episodes,
        help="Adaptive mode: min episodes in window before a decision.",
    )
    ap.add_argument(
        "--mount-adapt-dwell-rollouts",
        type=int,
        default=_cfg_defaults.mount_adapt_dwell_rollouts,
        help="Adaptive mode: rollouts to hold frac fixed after a change.",
    )

    ap.add_argument(
        "--terminate-on",
        type=str,
        default=_cfg_defaults.terminate_on,
        choices=("never", "pickup", "mount", "demount"),
        help=(
            "v6 curriculum brake-lock: stop the episode (success=True) at "
            "the named FSM event. 'never' = full Stage 0→1→2→3 cycle, "
            "'pickup'/'mount'/'demount' = early shortcut for stage-by-"
            "stage curriculum. Stage 2/3 env code stays active in all "
            "modes — only the termination point moves."
        ),
    )
    ap.add_argument(
        "--use-planner-residual",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "use_planner_residual", True)),
        help=(
            "Min-Jerk nominal trajectory + PPO residual (2026-06-01 default). "
            "Disable with --no-use-planner-residual for v11c-era checkpoints."
        ),
    )
    ap.add_argument(
        "--attached-spawn-when-easy",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "attached_spawn_when_easy", True)),
        help=(
            "On easy spawn: tire already grasped at cradle, task_stage=1. "
            "Disable with --no-attached-spawn-when-easy for pickup-from-scratch."
        ),
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=_cfg_defaults.max_steps,
        help=(
            "Episode horizon in env steps (overrides EnvConfig.max_steps). "
            "Use ~200 for pickup-only runs, ~400 for full 4-stage FSM."
        ),
    )
    ap.add_argument(
        "--dual-arm-coop",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "dual_arm_coop", False)),
        help=(
            "Enable dual-arm cooperative carry (UR10 + Panda jointly carry "
            "the UR10-feasible URDF tire onto the hub). Bundles "
            "use_tire_urdf + the small-tire dims + drops the cradle rack. "
            "PPO still drives only the UR10 residual; Panda is planner-driven."
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
    ap.add_argument(
        "--reset-timesteps",
        action="store_true",
        help=(
            "After policy-only resume, zero the PPO global step counter so "
            "curriculum schedules (mount/approach/start-pos) begin at t=0. "
            "Use when loading a pickup-only checkpoint for a new task phase."
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
        overrides["start_pos_curriculum_enable"] = bool(args.start_pos_curriculum)
        overrides["start_pos_easy_lift"] = float(args.start_pos_easy_lift)
        overrides["start_pos_curriculum_steps"] = int(args.start_pos_curriculum_steps)
        overrides["start_pos_ramp_steps"] = int(args.start_pos_ramp_steps)
        overrides["start_pos_curriculum_mode"] = str(args.start_pos_mode)
        overrides["start_pos_easy_prob"] = float(args.start_pos_easy_prob)
        overrides["terminate_on_pickup"] = bool(args.terminate_on_pickup)
        overrides["terminate_on"] = str(args.terminate_on)
        overrides["use_planner_residual"] = bool(args.use_planner_residual)
        overrides["attached_spawn_when_easy"] = bool(args.attached_spawn_when_easy)
        overrides["max_steps"] = int(args.max_steps)
        # Dual-arm cooperative carry bundles the UR10-feasible URDF tire +
        # its dims and drops the (single-arm) cradle rack. The UR10 base is
        # already at the relayout pose in EnvConfig.
        if bool(getattr(args, "dual_arm_coop", False)):
            overrides["dual_arm_coop"] = True
            overrides["use_tire_urdf"] = True
            overrides["tire_outer_radius"] = 0.30
            overrides["tire_inner_radius"] = 0.23
            overrides["tire_thickness"] = 0.16
            overrides["tire_mass"] = 1.5
            overrides["spawn_tire_rack"] = False
        overrides["approach_A_gate"] = float(args.approach_a_gate)
        overrides["approach_tol_soft"] = float(args.approach_tol_soft)
        overrides["approach_radius_tol"] = float(args.approach_tol_hard)
        overrides["approach_tol_curriculum_steps"] = int(
            args.approach_tol_curriculum_steps
        )
        overrides["approach_tol_ramp_steps"] = int(args.approach_tol_ramp_steps)
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
            if args.reset_timesteps:
                print("[train] reset_timesteps: curriculum counters start at t=0 "
                      f"(checkpoint had {ckpt.num_timesteps:,} steps)")
                model.num_timesteps = 0
                model._episode_num = 0
            else:
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
        reset_steps = (args.resume is None) or bool(args.reset_timesteps)
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            tb_log_name="ppo",
            reset_num_timesteps=reset_steps,
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
