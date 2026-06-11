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
import torch
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


class StartPosEasyProbCurriculumCallback(BaseCallback):
    """Schedules Bernoulli easy-spawn probability in ``mix`` mode (v5 / Q3).

    Piecewise smoothstep schedule on global PPO ``num_timesteps``:

      ``t ≤ mid_steps``     → prob = smoothstep(start → mid)
      ``mid < t ≤ end``     → prob = smoothstep(mid → end)
      ``t > end_steps``     → prob = end (held)

    Broadcasts via ``env_method("set_start_pos_easy_prob", p)`` so every
    sub-env's next ``reset()`` uses the updated mix weight.
    """

    def __init__(
        self,
        prob_start: float,
        prob_mid: float,
        prob_end: float,
        mid_steps: int,
        end_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._p0 = float(prob_start)
        self._p1 = float(prob_mid)
        self._p2 = float(prob_end)
        self._mid_steps = int(max(1, mid_steps))
        self._end_steps = int(max(self._mid_steps + 1, end_steps))
        self._last_pushed: Optional[float] = None

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _scheduled_prob(self, t: int) -> Tuple[float, float]:
        t = int(max(0, t))
        if t <= self._mid_steps:
            frac = t / float(self._mid_steps)
            s = self._smoothstep(frac)
            p = self._p0 * (1.0 - s) + self._p1 * s
            return p, frac * 0.5
        if t <= self._end_steps:
            span = float(self._end_steps - self._mid_steps)
            frac = (t - self._mid_steps) / span
            s = self._smoothstep(frac)
            p = self._p1 * (1.0 - s) + self._p2 * s
            return p, 0.5 + frac * 0.5
        return self._p2, 1.0

    def _broadcast(self, prob: float) -> None:
        try:
            self.training_env.env_method(
                "set_start_pos_easy_prob", float(prob),
            )
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        p, _ = self._scheduled_prob(int(self.model.num_timesteps))
        self._last_pushed = p
        self._broadcast(p)
        if self.verbose:
            print(f"[curriculum] start_pos_easy_prob init = {p:.3f} "
                  f"(t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        p, frac = self._scheduled_prob(t)
        if self._last_pushed is None or abs(p - self._last_pushed) > 1e-5:
            self._broadcast(p)
            self._last_pushed = p
            if self.verbose:
                print(f"[curriculum] start_pos_easy_prob = {p:.3f} "
                      f"(t={t}, frac={frac:.3f})")
        self.logger.record("curriculum/start_pos_easy_prob", float(p))
        self.logger.record("curriculum/start_pos_easy_prob_frac", float(frac))


class NutHotStartCurriculumCallback(BaseCallback):
    """Reverse-curriculum for the Robot-B nut hot-start (insertion task).

    Linearly ramps ``alpha`` from ``hold`` (held at the start value for
    ``hold_steps``) down to ``end_alpha`` over ``ramp_steps`` global PPO
    timesteps. ``alpha = 1`` starts B at bolt 0's approach point (easy,
    rich reach gradient); ``alpha = 0`` starts B at full HOME distance.
    Broadcast via ``env_method("set_nut_b_hotstart_alpha", a)``.
    """

    def __init__(
        self,
        start_alpha: float,
        end_alpha: float,
        hold_steps: int,
        ramp_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._a0 = float(start_alpha)
        self._a1 = float(end_alpha)
        self._hold = int(max(0, hold_steps))
        self._ramp = int(max(1, ramp_steps))
        self._last_pushed: Optional[float] = None

    def _scheduled_alpha(self, t: int) -> float:
        t = int(max(0, t))
        if t <= self._hold:
            return self._a0
        frac = min(1.0, (t - self._hold) / float(self._ramp))
        return self._a0 * (1.0 - frac) + self._a1 * frac

    def _broadcast(self, alpha: float) -> None:
        try:
            self.training_env.env_method("set_nut_b_hotstart_alpha", float(alpha))
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        a = self._scheduled_alpha(int(self.model.num_timesteps))
        self._last_pushed = a
        self._broadcast(a)
        if self.verbose:
            print(f"[curriculum] nut_b_hotstart_alpha init = {a:.3f} "
                  f"(t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        a = self._scheduled_alpha(t)
        if self._last_pushed is None or abs(a - self._last_pushed) > 1e-5:
            self._broadcast(a)
            self._last_pushed = a
            if self.verbose:
                print(f"[curriculum] nut_b_hotstart_alpha = {a:.3f} (t={t})")
        self.logger.record("curriculum/nut_b_hotstart_alpha", float(a))


class NutArriveAngCurriculumCallback(BaseCallback):
    """Ramp the nut arrive-alignment gate from loose → tight during training.

    Holds ``start_deg`` for ``hold_steps`` (so the value function learns the
    macro is valuable while the gate is easy to hit), then linearly ramps to
    ``end_deg`` over ``ramp_steps``. Broadcast via
    ``env_method("set_nut_arrive_ang_tol", rad)``.
    """

    def __init__(
        self,
        start_deg: float,
        end_deg: float,
        hold_steps: int,
        ramp_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._a0 = float(np.deg2rad(start_deg))
        self._a1 = float(np.deg2rad(end_deg))
        self._hold = int(max(0, hold_steps))
        self._ramp = int(max(1, ramp_steps))
        self._last_pushed: Optional[float] = None

    def _scheduled(self, t: int) -> float:
        t = int(max(0, t))
        if t <= self._hold:
            return self._a0
        frac = min(1.0, (t - self._hold) / float(self._ramp))
        return self._a0 * (1.0 - frac) + self._a1 * frac

    def _broadcast(self, rad: float) -> None:
        try:
            self.training_env.env_method("set_nut_arrive_ang_tol", float(rad))
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        a = self._scheduled(int(self.model.num_timesteps))
        self._last_pushed = a
        self._broadcast(a)
        if self.verbose:
            print(f"[curriculum] nut_arrive_ang_tol init = "
                  f"{np.degrees(a):.1f}deg (t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        a = self._scheduled(t)
        if self._last_pushed is None or abs(a - self._last_pushed) > 1e-5:
            self._broadcast(a)
            self._last_pushed = a
            if self.verbose:
                print(f"[curriculum] nut_arrive_ang_tol = "
                      f"{np.degrees(a):.1f}deg (t={t})")
        self.logger.record("curriculum/nut_arrive_ang_deg", float(np.degrees(a)))


class NutArrivePosCurriculumCallback(BaseCallback):
    """Ramp the nut arrive-position capture radius loose → tight during training.

    Mirrors ``NutArriveAngCurriculumCallback`` but for the staging capture
    sphere (``d_stage < nut_arrive_pos_tol``). Holds ``start_m`` for
    ``hold_steps`` (so insert is reachable from a generous staging region while
    the policy bootstraps the in/out), then linearly ramps to ``end_m`` over
    ``ramp_steps`` so the final policy must arrive precisely. Broadcast via
    ``env_method("set_nut_arrive_pos_tol", m)``.
    """

    def __init__(
        self,
        start_m: float,
        end_m: float,
        hold_steps: int,
        ramp_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._m0 = float(start_m)
        self._m1 = float(end_m)
        self._hold = int(max(0, hold_steps))
        self._ramp = int(max(1, ramp_steps))
        self._last_pushed: Optional[float] = None

    def _scheduled(self, t: int) -> float:
        t = int(max(0, t))
        if t <= self._hold:
            return self._m0
        frac = min(1.0, (t - self._hold) / float(self._ramp))
        return self._m0 * (1.0 - frac) + self._m1 * frac

    def _broadcast(self, m: float) -> None:
        try:
            self.training_env.env_method("set_nut_arrive_pos_tol", float(m))
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        m = self._scheduled(int(self.model.num_timesteps))
        self._last_pushed = m
        self._broadcast(m)
        if self.verbose:
            print(f"[curriculum] nut_arrive_pos_tol init = "
                  f"{m*100:.1f}cm (t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        m = self._scheduled(t)
        if self._last_pushed is None or abs(m - self._last_pushed) > 1e-5:
            self._broadcast(m)
            self._last_pushed = m
            if self.verbose:
                print(f"[curriculum] nut_arrive_pos_tol = {m*100:.1f}cm (t={t})")
        self.logger.record("curriculum/nut_arrive_pos_cm", float(m * 100.0))


class DRRangeCurriculumCallback(BaseCallback):
    """Ramp the domain-randomization hub-offset half-range during training.

    Holds ``start_m`` (typically 0) for ``hold_steps`` so the policy first
    re-confirms the nominal task, then linearly ramps to ``end_m`` (e.g.
    0.05 m) over ``ramp_steps``. Broadcast via
    ``env_method("set_random_position_range", m)`` each rollout boundary.
    Used by the Robot-B nut DR fine-tune to grow hub placement error 0 → 5 cm.
    """

    def __init__(
        self,
        start_m: float,
        end_m: float,
        hold_steps: int,
        ramp_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._m0 = float(max(0.0, start_m))
        self._m1 = float(max(0.0, end_m))
        self._hold = int(max(0, hold_steps))
        self._ramp = int(max(1, ramp_steps))
        self._last_pushed: Optional[float] = None

    def _scheduled(self, t: int) -> float:
        t = int(max(0, t))
        if t <= self._hold:
            return self._m0
        frac = min(1.0, (t - self._hold) / float(self._ramp))
        return self._m0 * (1.0 - frac) + self._m1 * frac

    def _broadcast(self, m: float) -> None:
        try:
            self.training_env.env_method("set_random_position_range", float(m))
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        m = self._scheduled(int(self.model.num_timesteps))
        self._last_pushed = m
        self._broadcast(m)
        if self.verbose:
            print(f"[curriculum] dr_range init = {m*100:.1f}cm "
                  f"(t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
        m = self._scheduled(t)
        if self._last_pushed is None or abs(m - self._last_pushed) > 1e-6:
            self._broadcast(m)
            self._last_pushed = m
            if self.verbose:
                print(f"[curriculum] dr_range = {m*100:.1f}cm (t={t})")
        self.logger.record("curriculum/dr_range_cm", float(m * 100.0))


class MountTolCurriculumCallback(BaseCallback):
    """Schedules the Stage 1 → 2 mount gate (radius, angle) over training.

    Mirrors ``ApproachTolCurriculumCallback`` but ramps two scalars in
    lockstep — Euclidean radius (m) and axis tolerance (rad). Both fade
    from soft → hard via a single smoothstep so the easy/hard regime is
    consistent across the two checks.
    """

    def __init__(
        self,
        radius_soft: float, radius_hard: float,
        angle_soft_rad: float, angle_hard_rad: float,
        soft_steps: int, ramp_steps: int, verbose: int = 0,
    ):
        super().__init__(verbose)
        self._r_soft = float(radius_soft)
        self._r_hard = float(radius_hard)
        self._a_soft = float(angle_soft_rad)
        self._a_hard = float(angle_hard_rad)
        self._soft_steps = int(max(0, soft_steps))
        self._ramp_steps = int(max(1, ramp_steps))
        self._last_pushed: Optional[Tuple[float, float]] = None

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _scheduled(self, t: int) -> Tuple[float, float, float]:
        if t <= self._soft_steps:
            return self._r_soft, self._a_soft, 0.0
        frac = (t - self._soft_steps) / float(self._ramp_steps)
        if frac >= 1.0:
            return self._r_hard, self._a_hard, 1.0
        s = self._smoothstep(frac)
        r = self._r_soft * (1.0 - s) + self._r_hard * s
        a = self._a_soft * (1.0 - s) + self._a_hard * s
        return r, a, frac

    def _broadcast(self, r: float, a: float) -> None:
        try:
            self.training_env.env_method("set_mount_tol", r, a)
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        r, a, _ = self._scheduled(int(self.model.num_timesteps))
        self._last_pushed = (r, a)
        self._broadcast(r, a)
        if self.verbose:
            print(f"[curriculum] mount_tol init = ({r:.3f} m, "
                  f"{np.degrees(a):.1f}°) (t={self.model.num_timesteps})")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.model.num_timesteps)
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


class ReverseCurriculumCallback(BaseCallback):
    """**v11 (2026-05-31)** — broadcast Phase A / B / C label to all envs.

    The env reads ``self._rev_curriculum_phase`` (set via
    ``set_reverse_curriculum_phase``) on every ``reset()`` and routes
    the spawn accordingly. Phases:

    * **A** — pure hub-aligned hot-start. ``0 .. phase_a_steps``.
    * **A/B blend** — Bernoulli mix between A and B. ``phase_a_steps
      .. phase_a_steps + a_to_b_overlap``. ``phase_a_mix_prob`` of
      resets stay in A, the rest fall through to Phase B.
    * **B** — legacy easy-mix start. ``phase_a_end .. phase_b_steps``.
    * **C** — pure HOME. ``phase_b_steps ..``.
    """

    def __init__(
        self,
        phase_a_steps: int,
        a_to_b_overlap: int,
        phase_a_mix_prob: float,
        phase_b_steps: int,
        verbose: int = 0,
        phase_a_terminate_on: str = "mount",
        phase_bc_terminate_on: str = "never",
        phase_a_mount_tol_lock: bool = True,
        phase_b_mount_tol_lock: bool = True,
        mount_tol_soft_radius: float = 0.30,
        mount_tol_soft_angle_rad: float = float(np.deg2rad(35.0)),
        phase_a_contact_force_term: float = 1.0e9,
        phase_bc_contact_force_term: Optional[float] = None,
    ) -> None:
        super().__init__(verbose)
        self.phase_a_steps = int(phase_a_steps)
        self.a_to_b_overlap = int(a_to_b_overlap)
        self.phase_a_mix_prob = float(phase_a_mix_prob)
        self.phase_b_steps = int(phase_b_steps)
        self._last_logged: Optional[str] = None
        # v11c — terminate_on toggle. Phase A flips to ``"mount"`` so
        # the first-step mount fire collects R_mount + is_success and
        # the episode ends — gives the policy a clean sparse signal
        # for "tire-aligned-at-hub" → success. Phase B/C restore the
        # CLI default (typically ``"never"`` for full cycle).
        self.phase_a_terminate_on = str(phase_a_terminate_on).lower()
        self.phase_bc_terminate_on = str(phase_bc_terminate_on).lower()
        # v11c — Phase A mount-tol lock. While Phase A is active we
        # broadcast the soft tol (e.g. 0.30 m / 35°) and freeze the
        # MountTolCurriculumCallback ramp until Phase A ends — this
        # is what guarantees the first-step mount fire even with the
        # small jitter set in ``reverse_phase_a_*_jitter``.
        self.phase_a_mount_tol_lock = bool(phase_a_mount_tol_lock)
        # v11c5 (2026-05-31) — Phase B soft mount-tol lock. v11c4 showed
        # fsm_bonus collapse once MountTolCurriculumCallback reached hard
        # tol (0.04 m) at ~375k global steps while d_A stayed at 2.07 m.
        # Keep soft (0.30 m / 35°) through all of Phase B so the policy
        # can learn carry→mount before tol tightens in Phase C.
        self.phase_b_mount_tol_lock = bool(phase_b_mount_tol_lock)
        self._mount_tol_soft_radius = float(mount_tol_soft_radius)
        self._mount_tol_soft_angle_rad = float(mount_tol_soft_angle_rad)
        # v11c1 — contact-force kill-switch toggle. During Phase A the
        # tire is teleported into the hub which produces enormous
        # contact-force spikes on the very first physics step; the
        # default 2500 N gate would terminate the episode in one step
        # before the policy can collect any post-mount data. We
        # disable the gate (effectively ``+inf``) while Phase A is
        # active, then restore the config default for Phase B/C.
        self.phase_a_contact_force_term = float(phase_a_contact_force_term)
        # ``None`` → resolved on training start by capturing the
        # env's current contact_force_terminate_above value.
        self._phase_bc_contact_force_term_user = phase_bc_contact_force_term
        self.phase_bc_contact_force_term = (
            float(phase_bc_contact_force_term) if phase_bc_contact_force_term is not None else 0.0
        )
        # Per-env coin (resampled each rollout boundary) so A/B blend
        # resets are stable within a rollout.
        self._blend_rng = np.random.default_rng(seed=12345)

    def _phase_for_timestep(self, t: int) -> str:
        a_end = self.phase_a_steps
        ab_end = a_end + self.a_to_b_overlap
        if t < a_end:
            return "A"
        if t < ab_end:
            # Bernoulli blend handled per-env below in ``_on_rollout_start``.
            return "AB_BLEND"
        if t < self.phase_b_steps:
            return "B"
        return "C"

    def _broadcast_terminate_on(self, value: str) -> None:
        """Push ``terminate_on`` to every env (no-op if hook missing)."""
        try:
            self.training_env.env_method("set_terminate_on", value)
        except (AttributeError, Exception):  # noqa: BLE001
            pass

    def _broadcast_contact_force_term(self, value: float) -> None:
        try:
            self.training_env.env_method("set_contact_force_term", float(value))
        except (AttributeError, Exception):  # noqa: BLE001
            pass

    def _broadcast_safety_terminations(self, enabled: bool) -> None:
        try:
            self.training_env.env_method("set_safety_terminations", bool(enabled))
        except (AttributeError, Exception):  # noqa: BLE001
            pass

    def _broadcast_mount_tol(self, r: float, a: float) -> None:
        try:
            self.training_env.env_method("set_mount_tol", r, a)
        except (AttributeError, Exception):  # noqa: BLE001
            pass

    def _push_to_envs(self, phase: str) -> None:
        if not hasattr(self.training_env, "env_method"):
            return
        if phase == "AB_BLEND":
            # Resample per-env which phase to use during the overlap.
            # Per-env terminate_on is also broadcast so the A copies see
            # the mount-terminate gate while the B copies stay on the
            # default cycle terminator.
            n = int(self.training_env.num_envs)
            for env_i in range(n):
                use_a = bool(self._blend_rng.random() < self.phase_a_mix_prob)
                sub_phase = "A" if use_a else "B"
                self.training_env.env_method(
                    "set_reverse_curriculum_phase",
                    sub_phase,
                    indices=[env_i],
                )
                t_on = (
                    self.phase_a_terminate_on if use_a
                    else self.phase_bc_terminate_on
                )
                cf_val = (
                    self.phase_a_contact_force_term if use_a
                    else self.phase_bc_contact_force_term
                )
                # v11c4 (2026-05-31) — keep vertical/collision/workspace
                # gates ON during Phase A; only the contact_force gate is
                # disabled. Episodes self-terminate around step 20 via
                # vertical_violation (the post-mount tire wobble), giving
                # ep_rew_mean ≈ +200 from R_mount instead of the
                # −1500 dense pile-up that the full 600-step "safety off"
                # variant accumulated.
                safety_on = True
                try:
                    self.training_env.env_method(
                        "set_terminate_on", t_on, indices=[env_i],
                    )
                    self.training_env.env_method(
                        "set_contact_force_term", float(cf_val), indices=[env_i],
                    )
                    self.training_env.env_method(
                        "set_safety_terminations", bool(safety_on), indices=[env_i],
                    )
                except (AttributeError, Exception):  # noqa: BLE001
                    pass
        else:
            self.training_env.env_method(
                "set_reverse_curriculum_phase", phase,
            )
            t_on = (
                self.phase_a_terminate_on if phase == "A"
                else self.phase_bc_terminate_on
            )
            self._broadcast_terminate_on(t_on)
            cf_val = (
                self.phase_a_contact_force_term if phase == "A"
                else self.phase_bc_contact_force_term
            )
            self._broadcast_contact_force_term(cf_val)
            # v11c4 — keep vertical/collision/workspace ON in all phases.
            self._broadcast_safety_terminations(True)

    def _on_training_start(self) -> None:
        # v11c — initial broadcast at t=0 (before the MountTol callback's
        # first rollout-end push so Phase A starts already locked).
        if self.phase_a_mount_tol_lock:
            self._broadcast_mount_tol(
                self._mount_tol_soft_radius,
                self._mount_tol_soft_angle_rad,
            )
        # v11c1 — capture the env's pre-training contact_force_terminate_above
        # so Phase B/C can restore it instead of using a hard-coded value.
        if self._phase_bc_contact_force_term_user is None:
            try:
                envs_cf = self.training_env.env_method("get_contact_force_term")
                if envs_cf:
                    self.phase_bc_contact_force_term = float(envs_cf[0])
            except (AttributeError, Exception):  # noqa: BLE001
                self.phase_bc_contact_force_term = 0.0

    def _on_rollout_start(self) -> None:
        t = int(self.model.num_timesteps)
        phase = self._phase_for_timestep(t)
        self._push_to_envs(phase)
        # v11c — re-broadcast Phase A soft mount_tol every rollout
        # so MountTolCurriculumCallback._on_rollout_end can't clobber
        # the lock. (Callback ordering: rollout_end runs *after* the
        # next rollout_start, so the lock here is overridden mid-
        # rollout otherwise. We re-push on every rollout boundary.)
        lock_a = self.phase_a_mount_tol_lock and phase in ("A", "AB_BLEND")
        lock_b = self.phase_b_mount_tol_lock and phase in ("B", "AB_BLEND")
        if lock_a or lock_b:
            self._broadcast_mount_tol(
                self._mount_tol_soft_radius,
                self._mount_tol_soft_angle_rad,
            )
        if phase != self._last_logged:
            self._last_logged = phase
            if self.verbose:
                print(f"[reverse-curriculum] phase = {phase} (t={t})")
        # TB diagnostics — integer encoding for sparkline.
        code = {"A": 0, "AB_BLEND": 1, "B": 2, "C": 3}[phase]
        self.logger.record("curriculum/reverse_phase_code", float(code))

    def _on_step(self) -> bool:
        return True


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
        ))
    if bool(getattr(args, "reverse_curriculum", False)):
        # v11c — Phase A terminate-on-mount + mount-tol soft lock.
        # ``--terminate-on`` carries the Phase B/C default; Phase A
        # always uses "mount" (unless --no-phase-a-terminate-on-mount).
        pa_term = (
            "mount"
            if bool(getattr(args, "phase_a_terminate_on_mount", True))
            else str(getattr(args, "terminate_on", "never")).lower()
        )
        pbc_term = str(getattr(args, "terminate_on", "never")).lower()
        cbs.append(ReverseCurriculumCallback(
            phase_a_steps=int(args.reverse_phase_a_steps),
            a_to_b_overlap=int(args.reverse_phase_a_to_b_overlap),
            phase_a_mix_prob=float(args.reverse_phase_a_mix_prob),
            phase_b_steps=int(args.reverse_phase_b_steps),
            verbose=1,
            phase_a_terminate_on=pa_term,
            phase_bc_terminate_on=pbc_term,
            phase_a_mount_tol_lock=bool(getattr(args, "phase_a_mount_tol_lock", True)),
            phase_b_mount_tol_lock=bool(getattr(args, "phase_b_mount_tol_lock", True)),
            mount_tol_soft_radius=float(args.mount_radius_soft),
            mount_tol_soft_angle_rad=float(np.deg2rad(args.mount_angle_soft_deg)),
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
    if (
        bool(getattr(args, "start_pos_easy_prob_curriculum", False))
        and str(args.start_pos_mode) == "mix"
    ):
        cbs.append(StartPosEasyProbCurriculumCallback(
            prob_start=float(args.start_pos_easy_prob_schedule_start),
            prob_mid=float(args.start_pos_easy_prob_schedule_mid),
            prob_end=float(args.start_pos_easy_prob_schedule_end),
            mid_steps=int(args.start_pos_easy_prob_schedule_mid_steps),
            end_steps=int(args.start_pos_easy_prob_schedule_end_steps),
            verbose=1,
        ))
    if (
        bool(getattr(args, "nut_fastening", False))
        and bool(getattr(args, "nut_hotstart_curriculum", True))
        and not bool(getattr(args, "nut_b_planner_residual", False))
    ):
        cbs.append(NutHotStartCurriculumCallback(
            start_alpha=float(getattr(args, "nut_hotstart_alpha_start", 1.0)),
            end_alpha=float(getattr(args, "nut_hotstart_alpha_end", 0.0)),
            hold_steps=int(getattr(args, "nut_hotstart_hold_steps", 300_000)),
            ramp_steps=int(getattr(args, "nut_hotstart_ramp_steps", 1_500_000)),
            verbose=1,
        ))
    if (
        bool(getattr(args, "nut_fastening", False))
        and bool(getattr(args, "nut_arrive_ang_curriculum", True))
    ):
        cbs.append(NutArriveAngCurriculumCallback(
            start_deg=float(getattr(args, "nut_arrive_ang_start_deg", 35.0)),
            end_deg=float(getattr(args, "nut_arrive_ang_end_deg", 12.0)),
            hold_steps=int(getattr(args, "nut_arrive_ang_hold_steps", 300_000)),
            ramp_steps=int(getattr(args, "nut_arrive_ang_ramp_steps", 1_500_000)),
            verbose=1,
        ))
    if (
        bool(getattr(args, "nut_fastening", False))
        and bool(getattr(args, "nut_arrive_pos_curriculum", False))
    ):
        cbs.append(NutArrivePosCurriculumCallback(
            start_m=float(getattr(args, "nut_arrive_pos_start_cm", 12.0)) / 100.0,
            end_m=float(getattr(args, "nut_arrive_pos_end_cm", 8.0)) / 100.0,
            hold_steps=int(getattr(args, "nut_arrive_pos_hold_steps", 400_000)),
            ramp_steps=int(getattr(args, "nut_arrive_pos_ramp_steps", 2_000_000)),
            verbose=1,
        ))
    if bool(getattr(args, "dr_range_curriculum", False)):
        cbs.append(DRRangeCurriculumCallback(
            start_m=float(getattr(args, "dr_range_start_cm", 0.0)) / 100.0,
            end_m=float(getattr(args, "dr_range_end_cm", 5.0)) / 100.0,
            hold_steps=int(getattr(args, "dr_range_hold_steps", 200_000)),
            ramp_steps=int(getattr(args, "dr_range_ramp_steps", 1_000_000)),
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
    ap.add_argument("--force-log-std", action="store_true",
                    help="With policy-only --resume: overwrite the transferred "
                         "log_std with --log-std-init (sharpening fine-tune).")
    ap.add_argument("--log-std-init", type=float, default=0.0,
                    help="Initial log-std of the Gaussian policy. 0.0 => std "
                         "1.0 (SB3 default); -0.5 => std ~0.61 (tighter "
                         "exploration so the stochastic residual stays near "
                         "the already-good nominal and clears the tight mount "
                         "gate). Transfers via policy-only resume.")

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
        "--start-pos-easy-prob-curriculum",
        action=argparse.BooleanOptionalAction,
        default=_cfg_defaults.start_pos_easy_prob_curriculum_enable,
        help=(
            "In mix mode, schedule easy-spawn probability over training "
            "(0.9→0.5 @ 1M steps →0.3 @ 2M by default). "
            "Disable with --no-start-pos-easy-prob-curriculum."
        ),
    )
    ap.add_argument(
        "--start-pos-easy-prob-schedule-start",
        type=float,
        default=_cfg_defaults.start_pos_easy_prob_schedule_start,
    )
    ap.add_argument(
        "--start-pos-easy-prob-schedule-mid",
        type=float,
        default=_cfg_defaults.start_pos_easy_prob_schedule_mid,
    )
    ap.add_argument(
        "--start-pos-easy-prob-schedule-end",
        type=float,
        default=_cfg_defaults.start_pos_easy_prob_schedule_end,
    )
    ap.add_argument(
        "--start-pos-easy-prob-schedule-mid-steps",
        type=int,
        default=_cfg_defaults.start_pos_easy_prob_schedule_mid_steps,
    )
    ap.add_argument(
        "--start-pos-easy-prob-schedule-end-steps",
        type=int,
        default=_cfg_defaults.start_pos_easy_prob_schedule_end_steps,
    )
    # **v11 (2026-05-31) — Reverse curriculum.** Independent of the
    # legacy start-pos curriculum; toggles per-env reset routing
    # between Phase A (hub-aligned hot-start), Phase B (easy-mix), and
    # Phase C (pure HOME) based on global PPO timestep.
    ap.add_argument(
        "--reverse-curriculum",
        action=argparse.BooleanOptionalAction,
        default=_cfg_defaults.reverse_curriculum_enable,
        help=(
            "Enable v11 reverse curriculum (Phase A hot-start → B easy-mix → C HOME). "
            "Use --no-reverse-curriculum to disable."
        ),
    )
    ap.add_argument(
        "--reverse-phase-a-steps",
        type=int,
        default=_cfg_defaults.reverse_phase_a_steps,
        help="End of pure-A plateau (global PPO timesteps).",
    )
    ap.add_argument(
        "--reverse-phase-a-to-b-overlap",
        type=int,
        default=_cfg_defaults.reverse_phase_a_to_b_overlap,
        help="A→B overlap window length (steps).",
    )
    ap.add_argument(
        "--reverse-phase-a-mix-prob",
        type=float,
        default=_cfg_defaults.reverse_phase_a_mix_prob,
        help="Probability of staying in Phase A during A→B overlap.",
    )
    ap.add_argument(
        "--reverse-phase-b-steps",
        type=int,
        default=_cfg_defaults.reverse_phase_b_steps,
        help="End of Phase B plateau (start of pure-C HOME).",
    )
    ap.add_argument(
        "--phase-a-terminate-on-mount",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "reverse_phase_a_terminate_on_mount", True)),
        help=(
            "v11c: in Phase A, override --terminate-on to 'mount' so the "
            "first-step mount fire collects R_mount and ends the episode "
            "with success. Disable with --no-phase-a-terminate-on-mount."
        ),
    )
    ap.add_argument(
        "--phase-a-mount-tol-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "v11c: while Phase A is active, the ReverseCurriculumCallback "
            "re-broadcasts the soft (radius_soft / angle_soft) mount tol "
            "every rollout, freezing the MountTolCurriculumCallback ramp. "
            "Guarantees the first-step mount fire even under the small "
            "Phase A jitter. Disable with --no-phase-a-mount-tol-lock."
        ),
    )
    ap.add_argument(
        "--phase-b-mount-tol-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "v11c5: while Phase B / AB_BLEND is active, re-broadcast soft "
            "mount tol every rollout so MountTolCurriculumCallback cannot "
            "tighten to hard (0.04 m) before d_A closes. Tol ramp then "
            "runs only in Phase C. Disable with --no-phase-b-mount-tol-lock."
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
        "--remount-cycle",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "remount_cycle_enable", False)),
        help=(
            "Full 6-stage Robot-A duty cycle: pick → mount → (W1 tighten) "
            "release+retract to HOME → re-grip hub tire → (W2 loosen) "
            "demount → carry to rack. Use with --terminate-on never and a "
            "larger --max-steps (~1000). Off = legacy 4-stage FSM."
        ),
    )
    ap.add_argument(
        "--nut-fastening",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "nut_fastening_task", False)),
        help=(
            "Robot-B sequential nut-fastening task: the tire is held mounted "
            "on the hub (Robot A frozen) and Robot B (UR10e nut-runner) learns "
            "to seat its tool on each hub bolt in turn (geometric reach+align, "
            "no nut physics). Forces freeze_robot_b=False (13-d action). Use "
            "with --terminate-on never and a larger --max-steps (~600)."
        ),
    )
    ap.add_argument(
        "--nut-hold-steps", type=int,
        default=int(getattr(_cfg_defaults, "nut_hold_steps", 12)),
        help="Consecutive in-gate steps before a bolt counts as fastened.",
    )
    ap.add_argument(
        "--nut-b-planner-residual",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "nut_b_planner_residual", False)),
        help=(
            "Robot-B nut APPROACH: min-jerk nominal trajectory + PPO XYZ "
            "residual (oracle path). Disables hot-start curriculum."
        ),
    )
    ap.add_argument(
        "--nut-v19",
        action="store_true",
        help=(
            "v19 precision rework bundle (requires --nut-pure-rl): env align "
            "servo during insert (geometrically exact on-axis plunge), arrive "
            "gate tightened (lat 1.5cm), seat gate coaxial (1x lat tol), "
            "Robot A kinematically frozen (rigid fixture), B collision = "
            "instant episode FAILURE, solo 3-d action space, minimal-path "
            "waste cost, 250-step stall truncation."
        ),
    )
    ap.add_argument(
        "--nut-v20",
        action="store_true",
        help=(
            "v20 bundle (requires --nut-pure-rl): all v19 features plus "
            "INSERT axial servo (socket drives to hub-face base, full bolt "
            "envelopment), seat depth tol 0.7 cm (was 2 cm), joint-movement "
            "penalty 0.06 across all phases."
        ),
    )
    ap.add_argument(
        "--nut-per-leg",
        type=lambda s: s.strip().lower() in ("1", "true", "t", "yes", "y"),
        default=None,
        help=(
            "Override per-leg episodes (default: auto-on with --nut-pure-rl). "
            "Set false for the stage-2 multi-bolt CHAIN fine-tune."
        ),
    )
    ap.add_argument(
        "--nut-path-waste", type=float, default=5.0,
        help="v19 wasted-motion cost weight (only with --nut-v19).",
    )
    ap.add_argument(
        "--nut-pure-rl",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "nut_pure_rl", False)),
        help=(
            "Robot-B nut PURE-RL (hybrid): policy controls the whole cycle "
            "(approach AND insert/hold/retract). No planner nominal, no scripted "
            "macro. APPROACH/transit = free 3-DOF XYZ; INSERT/RETRACT = bolt-axis "
            "only (±Y plunge). Bolt order enforced by FSM; collision = soft "
            "penalty. Obs widened to 12-d; per-leg watchdog disabled."
        ),
    )
    ap.add_argument(
        "--nut-planner-traj-steps", type=int,
        default=int(getattr(_cfg_defaults, "nut_planner_traj_steps", 120)),
        help="Samples along each APPROACH nominal leg.",
    )
    ap.add_argument(
        "--nut-planner-pos-residual-scale", type=float,
        default=float(getattr(_cfg_defaults, "nut_planner_pos_residual_scale", 0.05)),
        help="Per-step EE residual scale (m) on the nominal nut trajectory.",
    )
    ap.add_argument(
        "--nut-hotstart-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reverse-curriculum hot-start for the nut task: B starts near "
            "bolt 0 (alpha=1) and the start pose ramps back to full HOME "
            "(alpha=0). Gives the policy a reach gradient the flat exp-reach "
            "landscape lacks from the 1.7 m HOME standoff."
        ),
    )
    ap.add_argument("--nut-hotstart-alpha-start", type=float, default=1.0,
                    help="Hot-start alpha held during the warmup (1=at bolt).")
    ap.add_argument("--nut-hotstart-alpha-end", type=float, default=0.0,
                    help="Hot-start alpha after the ramp (0=full HOME).")
    ap.add_argument("--nut-hotstart-hold-steps", type=int, default=300_000,
                    help="Steps to hold alpha_start before ramping down.")
    ap.add_argument("--nut-hotstart-ramp-steps", type=int, default=1_500_000,
                    help="Steps to ramp alpha_start → alpha_end.")
    ap.add_argument(
        "--nut-hotstart-random-bolt",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "nut_b_hotstart_random_bolt", False)),
        help=(
            "Random-bolt premark curriculum: each reset marks earlier bolts in "
            "nut_bolt_order as already fastened and hot-starts B at a random "
            "position in the sequence. Trains every bolt-to-bolt transition "
            "evenly instead of always replaying from bolt 0 (frontier effect). "
            "Requires hot-start (alpha > 0)."
        ),
    )
    ap.add_argument(
        "--nut-arrive-ang-curriculum",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "nut_arrive_ang_curriculum", True)),
        help=(
            "Ramp the arrive-alignment gate loose→tight during the nut task: "
            "the macro triggers under a generous angle early (so the reward is "
            "reachable), then the gate tightens so the policy must align well."
        ),
    )
    ap.add_argument(
        "--nut-arrive-ang-start-deg", type=float,
        default=float(getattr(_cfg_defaults, "nut_arrive_ang_start_deg", 35.0)),
        help="Loose start angle (deg) for the arrive-alignment gate.")
    ap.add_argument(
        "--nut-arrive-ang-end-deg", type=float,
        default=float(getattr(_cfg_defaults, "nut_arrive_ang_end_deg", 12.0)),
        help="Tight end angle (deg) for the arrive-alignment gate.")
    ap.add_argument(
        "--nut-arrive-ang-hold-steps", type=int,
        default=int(getattr(_cfg_defaults, "nut_arrive_ang_hold_steps", 300_000)),
        help="Steps to hold the loose start before ramping the angle gate.")
    ap.add_argument(
        "--nut-arrive-ang-ramp-steps", type=int,
        default=int(getattr(_cfg_defaults, "nut_arrive_ang_ramp_steps", 1_500_000)),
        help="Steps to ramp the arrive-alignment gate start → end.")
    ap.add_argument(
        "--nut-arrive-pos-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ramp the arrive-position capture radius loose→tight during the nut "
            "task: insert triggers from a generous staging region early (so the "
            "in/out is reachable), then the capture sphere tightens so the "
            "policy must stage precisely. (The pure-RL lateral coaxiality gate "
            "is fixed by seat physics and is NOT ramped — only the axial "
            "capture distance.)"
        ),
    )
    ap.add_argument(
        "--nut-arrive-pos-start-cm", type=float, default=12.0,
        help="Loose start capture radius (cm) for the arrive-position gate.")
    ap.add_argument(
        "--nut-arrive-pos-end-cm", type=float, default=8.0,
        help="Tight end capture radius (cm) for the arrive-position gate.")
    ap.add_argument(
        "--nut-arrive-pos-hold-steps", type=int, default=400_000,
        help="Steps to hold the loose start before ramping the position gate.")
    ap.add_argument(
        "--nut-arrive-pos-ramp-steps", type=int, default=2_000_000,
        help="Steps to ramp the arrive-position gate start → end.")
    # --- domain randomization (hub placement error) ----------------------
    ap.add_argument(
        "--dr-hub-offset",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "USE_DOMAIN_RANDOMIZATION", False)),
        help=(
            "Enable static-pose domain randomization: perturb the hub XY by a "
            "uniform +-RANDOM_POSITION_RANGE each reset (bolts move with the "
            "hub; the nut nominal trajectory regenerates around them)."
        ),
    )
    ap.add_argument(
        "--dr-range-cm", type=float,
        default=float(getattr(_cfg_defaults, "RANDOM_POSITION_RANGE", 0.02)) * 100.0,
        help="DR hub-offset half-range (cm). Fixed value unless a curriculum ramps it.")
    ap.add_argument(
        "--dr-cargo",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(_cfg_defaults, "DR_CARGO_ENABLE", True)),
        help="Also perturb cargo XY independently. Use --no-dr-cargo for hub-only DR.")
    ap.add_argument(
        "--dr-range-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ramp the DR hub-offset half-range from --dr-range-start-cm to --dr-range-end-cm.")
    ap.add_argument("--dr-range-start-cm", type=float, default=0.0,
                    help="DR ramp start half-range (cm), held for --dr-range-hold-steps.")
    ap.add_argument("--dr-range-end-cm", type=float, default=5.0,
                    help="DR ramp end half-range (cm).")
    ap.add_argument("--dr-range-hold-steps", type=int, default=200_000,
                    help="Steps to hold --dr-range-start-cm before ramping.")
    ap.add_argument("--dr-range-ramp-steps", type=int, default=1_000_000,
                    help="Steps to ramp DR half-range start → end.")
    ap.add_argument(
        "--nut-a-hold-jitter-deg", type=float,
        default=float(np.degrees(getattr(
            _cfg_defaults, "nut_a_hold_jitter_rad", np.deg2rad(3.0),
        ))),
        help="Per-joint uniform jitter (deg) on Robot A's mount-hold fixture pose.",
    )
    ap.add_argument(
        "--w-nut-ba-clear", type=float, default=None,
        help="Robot A–B joint-center clearance shaping weight (RewardConfig).",
    )
    ap.add_argument(
        "--planner-pos-offset-scale", type=float, default=None,
        help=(
            "Robot-A per-step EE residual authority (m) on the mount nominal "
            "trajectory. Overrides the make_env_config default (0.03). Raise "
            "(e.g. 0.06) when training under hub DR so the residual can correct "
            "for the offset hub."
        ),
    )
    ap.add_argument(
        "--tighten-hold-steps", type=int,
        default=int(getattr(_cfg_defaults, "tighten_hold_steps", 40)),
        help="6-stage cycle W1 hold (steps) the arm holds after mount.",
    )
    ap.add_argument(
        "--loosen-hold-steps", type=int,
        default=int(getattr(_cfg_defaults, "loosen_hold_steps", 40)),
        help="6-stage cycle W2 hold (steps) the arm holds after re-grip.",
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

    # Physics overrides (Bulleted defaults live in EnvConfig)
    ap.add_argument("--physics-num-sub-steps", type=int, default=None)
    ap.add_argument("--contact-erp", type=float, default=None)
    ap.add_argument("--contact-cfm", type=float, default=None)
    ap.add_argument("--contact-force-done", type=float, default=None,
                    help="Terminate if normal force exceeds this (N); ≤0 disables.")
    ap.add_argument("--tire-mass", type=float, default=None,
                    help="Tire base mass in kg (default: EnvConfig.tire_mass = 1.0).")
    ap.add_argument(
        "--robot-a-kind",
        type=str,
        default=None,
        choices=("ur10", "fanuc_r2000ic"),
        help='Robot A model: "fanuc_r2000ic" (default) or legacy "ur10".',
    )
    ap.add_argument(
        "--fanuc-torque-scale",
        type=float,
        default=None,
        help=(
            "Global multiplier on FANUC per-joint torque caps (default 1.0). "
            "The R-2000iC/210F is a 210 kg-payload arm so 1.5-2.0 is physically "
            "fine; note the 100 kg carry already reaches the mount target at 1.0 "
            "(see scripts/diag_torque_tracking.py) and scales >=4 destabilise the "
            "stiff position PD."
        ),
    )
    ap.add_argument(
        "--robot-b-kind",
        type=str,
        default=None,
        choices=("panda", "ur10e"),
        help='Robot B model: "ur10e" (default) or legacy "panda".',
    )
    ap.add_argument(
        "--scene-layout",
        type=str,
        default=None,
        choices=("shipping", "fanuc_spacious"),
        help='Scene layout: "fanuc_spacious" (default) or legacy "shipping" (UR10+Panda).',
    )

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
        if args.robot_a_kind is not None:
            overrides["robot_a_kind"] = str(args.robot_a_kind)
        if getattr(args, "fanuc_torque_scale", None) is not None:
            overrides["fanuc_torque_scale"] = float(args.fanuc_torque_scale)
        if getattr(args, "robot_b_kind", None) is not None:
            overrides["robot_b_kind"] = str(args.robot_b_kind)
        if getattr(args, "scene_layout", None) is not None:
            overrides["scene_layout"] = str(args.scene_layout)
        overrides["start_pos_curriculum_enable"] = bool(args.start_pos_curriculum)
        overrides["start_pos_easy_lift"] = float(args.start_pos_easy_lift)
        overrides["start_pos_curriculum_steps"] = int(args.start_pos_curriculum_steps)
        overrides["start_pos_ramp_steps"] = int(args.start_pos_ramp_steps)
        overrides["start_pos_curriculum_mode"] = str(args.start_pos_mode)
        overrides["start_pos_easy_prob"] = float(args.start_pos_easy_prob)
        overrides["start_pos_easy_prob_curriculum_enable"] = bool(
            args.start_pos_easy_prob_curriculum
        )
        overrides["reverse_curriculum_enable"] = bool(
            getattr(args, "reverse_curriculum", False)
        )
        overrides["reverse_phase_a_steps"] = int(args.reverse_phase_a_steps)
        overrides["reverse_phase_a_to_b_overlap"] = int(
            args.reverse_phase_a_to_b_overlap
        )
        overrides["reverse_phase_a_mix_prob"] = float(
            args.reverse_phase_a_mix_prob
        )
        overrides["reverse_phase_b_steps"] = int(args.reverse_phase_b_steps)
        overrides["reverse_phase_a_terminate_on_mount"] = bool(
            getattr(args, "phase_a_terminate_on_mount", True)
        )
        overrides["terminate_on_pickup"] = bool(args.terminate_on_pickup)
        overrides["terminate_on"] = str(args.terminate_on)
        overrides["remount_cycle_enable"] = bool(args.remount_cycle)
        overrides["tighten_hold_steps"] = int(args.tighten_hold_steps)
        overrides["loosen_hold_steps"] = int(args.loosen_hold_steps)
        if bool(getattr(args, "nut_fastening", False)):
            overrides["nut_fastening_task"] = True
            # Robot B must be policy-controlled (13-d action / full obs).
            overrides["freeze_robot_b"] = False
            overrides["nut_hold_steps"] = int(args.nut_hold_steps)
            overrides["nut_b_planner_residual"] = bool(
                getattr(args, "nut_b_planner_residual", False)
            )
            overrides["nut_pure_rl"] = bool(getattr(args, "nut_pure_rl", False))
            overrides["nut_planner_traj_steps"] = int(
                getattr(args, "nut_planner_traj_steps", 120)
            )
            overrides["nut_planner_pos_residual_scale"] = float(
                getattr(args, "nut_planner_pos_residual_scale", 0.05)
            )
            overrides["nut_a_hold_jitter_rad"] = float(
                np.deg2rad(getattr(args, "nut_a_hold_jitter_deg", 3.0))
            )
            overrides["nut_b_hotstart_random_bolt"] = bool(
                getattr(args, "nut_hotstart_random_bolt", False)
            )
            # v19/v20 precision-rework bundle (align servo / rigid fixture /
            # collision=fail / solo 3-d action / minimal-path shaping /
            # stall truncation). See scripts/run_b_nut_train_v19_*.sh.
            if bool(getattr(args, "nut_v19", False)) or bool(
                getattr(args, "nut_v20", False)
            ):
                overrides["nut_b_align_servo"] = True
                overrides["nut_a_kinematic_freeze"] = True
                overrides["nut_collision_fail"] = True
                overrides["nut_b_solo_action"] = True
                overrides["nut_arrive_lat_tol"] = 0.015
                overrides["nut_seat_lat_mult"] = 1.0
                overrides["nut_stall_steps"] = 250
            if bool(getattr(args, "nut_v20", False)):
                overrides["nut_b_axial_insert_servo"] = True
                overrides["nut_insert_depth_tol"] = 0.007
                # v21 — branch-aware INSERT: a stalled plunge searches for a
                # reachable seat branch instead of freezing short. Lets the
                # workspace-edge bolts actually seat so the chain advances and
                # the approach policy is rewarded for reaching them.
                overrides["nut_b_insert_branch_search"] = True
            if getattr(args, "nut_per_leg", None) is not None:
                overrides["nut_per_leg_episode"] = bool(args.nut_per_leg)
        overrides["use_planner_residual"] = bool(args.use_planner_residual)
        overrides["attached_spawn_when_easy"] = bool(args.attached_spawn_when_easy)
        # Domain randomization (hub placement error). When a DR range curriculum
        # is active the env starts at the ramp's start value and the callback
        # grows it; otherwise the fixed --dr-range-cm is used. Either way the
        # master switch is driven by --dr-hub-offset (or implied by the ramp).
        dr_curr = bool(getattr(args, "dr_range_curriculum", False))
        dr_on = bool(getattr(args, "dr_hub_offset", False)) or dr_curr
        overrides["USE_DOMAIN_RANDOMIZATION"] = dr_on
        overrides["DR_CARGO_ENABLE"] = bool(getattr(args, "dr_cargo", True))
        if dr_curr:
            overrides["RANDOM_POSITION_RANGE"] = float(args.dr_range_start_cm) / 100.0
        elif dr_on:
            overrides["RANDOM_POSITION_RANGE"] = float(args.dr_range_cm) / 100.0
        overrides["max_steps"] = int(args.max_steps)
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
        if getattr(args, "w_nut_ba_clear", None) is not None:
            cfg.reward.w_nut_ba_clear = float(args.w_nut_ba_clear)
        if bool(getattr(args, "nut_v19", False)) or bool(
            getattr(args, "nut_v20", False)
        ):
            # Minimal-path transit shaping (v19/v20): PB stays the driver; the
            # waste cost makes the straight line the optimum.
            cfg.reward.w_nut_path_waste = float(
                getattr(args, "nut_path_waste", 5.0))
        if bool(getattr(args, "nut_v20", False)):
            cfg.reward.w_nut_joint_vel = 0.06
        if getattr(args, "planner_pos_offset_scale", None) is not None:
            # Applied after make_env_config, which otherwise forces 0.03.
            cfg.planner_pos_offset_scale = float(args.planner_pos_offset_scale)
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

    # --- Learner thread pool (CPU utilisation fix) --------------------------
    # SubprocVecEnv workers are spawned above with OMP_NUM_THREADS=1 (set in the
    # launch script) so the parallel rollout doesn't oversubscribe cores. But
    # that same env var also pins THIS (main) process — which runs the single-
    # process PPO gradient update — to one thread. The result: during every
    # update all rollout workers idle while one core does the matmuls, so ~half
    # the wall-clock leaves the box >90% idle (observed: vmstat ``r`` oscillating
    # 88↔3, ~48% user). Re-enabling multiple torch threads *only in the main
    # process* (workers already spawned, unaffected) parallelises the update and
    # roughly doubles throughput without oversubscribing the rollout. Tunable via
    # TYRO_LEARNER_THREADS (default: a slice of the box left free during update).
    if args.device == "cpu":
        import os as _os
        import torch as _torch
        _lt = int(_os.environ.get("TYRO_LEARNER_THREADS", "0") or 0)
        if _lt <= 0:
            # During the update the rollout workers are idle, so the whole box
            # is free — but the small MLP matmuls (batch≈1024) stop scaling past
            # ~16 threads, so cap there to avoid thread-spawn overhead.
            _lt = max(1, min(16, _os.cpu_count() or 1))
        _torch.set_num_threads(_lt)
        print(f"[train] learner torch threads = {_lt} "
              f"(num_envs={args.num_envs}, cpus={_os.cpu_count()})")

    # Skip the eval env entirely when EvalCallback is disabled — PyBullet
    # connect + URDF load is non-trivial and otherwise wasted.
    if args.no_eval_callback:
        eval_env = None
    else:
        eval_env = DummyVecEnv([make_env(0, cfg_factory, args.seed + 10_000)])
        eval_env = VecMonitor(eval_env, info_keywords=("is_success", "termination"))
        # Match eval to the TRAINING target. The hot-start curriculum only
        # broadcasts alpha to the training envs, so the eval env keeps the cfg
        # default (0.0 = full HOME) — which tests a HARDER start than the policy
        # is ever trained for (the curriculum floors at --nut-hotstart-alpha-end,
        # e.g. 0.3). That mismatch makes eval success_rate read ~0% even while
        # the policy succeeds at its actual target distance. Pin the eval env's
        # hot-start alpha to the curriculum's end value so eval measures the real
        # deployment goal.
        if (
            bool(getattr(args, "nut_fastening", False))
            and bool(getattr(args, "nut_hotstart_curriculum", True))
        ):
            _eval_alpha = float(getattr(args, "nut_hotstart_alpha_end", 0.0))
            try:
                eval_env.env_method("set_nut_b_hotstart_alpha", _eval_alpha)
                print(f"[eval] nut_b_hotstart_alpha pinned to "
                      f"curriculum end = {_eval_alpha:.3f}")
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    net_arch = [int(w) for w in args.net_arch.split(",") if w]
    policy_kwargs = dict(net_arch=net_arch, log_std_init=args.log_std_init)

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
            if bool(args.force_log_std):
                # Sharpening fine-tune: the transferred weights include the
                # checkpoint's log_std, which can be too noisy for a
                # deterministic-deployment polish. Clamp it back to the CLI
                # value so exploration restarts tight around the learned mean.
                with torch.no_grad():
                    model.policy.log_std.fill_(float(args.log_std_init))
                print(f"[train] log_std forced to {args.log_std_init}")
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
