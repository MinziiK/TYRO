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
            "Use ~200 for pickup-only runs, ~400 for full 4-stage FSM, "
            "~1200 for the full-cycle 7-state FSM."
        ),
    )
    ap.add_argument(
        "--full-cycle",
        action="store_true",
        default=bool(getattr(_cfg_defaults, "full_cycle", False)),
        help=(
            "Enable the extended 7-state mount/dismount FSM: pick → mount → "
            "tighten-hold → retract-to-HOME → re-approach+re-grasp → "
            "loosen-hold → return-to-rack. Requires --terminate-on never and "
            "a long --max-steps (~1200)."
        ),
    )
    ap.add_argument(
        "--mount-hold-steps",
        type=int,
        default=None,
        help="Tighten dwell (control steps) at the hub. Full cycle defaults to 40.",
    )
    ap.add_argument(
        "--loosen-hold-steps",
        type=int,
        default=None,
        help="Loosen dwell (control steps) before carrying the tire back.",
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
        overrides["full_cycle"] = bool(getattr(args, "full_cycle", False))
        if args.mount_hold_steps is not None:
            overrides["mount_hold_steps"] = int(args.mount_hold_steps)
        if args.loosen_hold_steps is not None:
            overrides["loosen_hold_steps"] = int(args.loosen_hold_steps)
        overrides["use_planner_residual"] = bool(args.use_planner_residual)
        overrides["attached_spawn_when_easy"] = bool(args.attached_spawn_when_easy)
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
