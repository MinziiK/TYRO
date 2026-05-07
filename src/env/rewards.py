"""Reward terms from `tyro_design_spec.md` §4.

Each function returns a scalar (float) and accepts already-extracted geometry
so it stays unit-tested without PyBullet. The env aggregates terms in step().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import RewardConfig
from .utils import angle_between


@dataclass
class RewardBreakdown:
    """Per-term reward log — env writes this into info['reward_terms']."""
    align_A: float = 0.0
    reach_B: float = 0.0
    coop: float = 0.0
    success: float = 0.0
    collision: float = 0.0
    action: float = 0.0
    jerk: float = 0.0
    shape_A: float = 0.0
    shape_B: float = 0.0
    total: float = 0.0
    # auxiliary (for logging / termination)
    d_A: float = 0.0
    theta_A: float = 0.0
    d_B: float = 0.0
    theta_B: float = 0.0
    is_success: bool = False


def align_reward(tire_pos, hub_pos, tire_axis, hub_axis,
                 cfg: RewardConfig) -> tuple[float, float, float]:
    """r_align^A = -w_d * d_A - w_theta * theta_A. Returns (term, d_A, theta_A)."""
    d_A = float(np.linalg.norm(np.asarray(tire_pos) - np.asarray(hub_pos)))
    theta_A = angle_between(tire_axis, hub_axis)
    term = -cfg.w_d_A * d_A - cfg.w_theta_A * theta_A
    return term, d_A, theta_A


def reach_reward(ee_b_pos, bolt_pos, ee_b_z_axis, bolt_axis,
                 cfg: RewardConfig) -> tuple[float, float, float]:
    """r_reach^B = -w_d * d_B - w_theta * theta_B."""
    d_B = float(np.linalg.norm(np.asarray(ee_b_pos) - np.asarray(bolt_pos)))
    theta_B = angle_between(ee_b_z_axis, bolt_axis)
    term = -cfg.w_d_B * d_B - cfg.w_theta_B * theta_B
    return term, d_B, theta_B


def coop_reward(d_A: float, d_B: float, cfg: RewardConfig) -> float:
    """Smooth product form (spec §4.1.3): w_c * exp(-α d_A) * exp(-β d_B)."""
    return cfg.w_c * float(np.exp(-cfg.alpha * d_A)) * float(np.exp(-cfg.beta * d_B))


def success_bonus(d_A, theta_A, d_B, cfg: RewardConfig) -> tuple[float, bool]:
    ok = (d_A < cfg.eps_A) and (theta_A < cfg.delta_A) and (d_B < cfg.eps_B)
    return (cfg.R_success if ok else 0.0), bool(ok)


def collision_penalty(in_collision: bool, cfg: RewardConfig) -> float:
    return -cfg.w_collision if in_collision else 0.0


def action_penalty(action: np.ndarray, cfg: RewardConfig) -> float:
    a = np.asarray(action, dtype=np.float64)
    return -cfg.w_action * float(np.dot(a, a))


def jerk_penalty(action: np.ndarray, prev_action: np.ndarray, cfg: RewardConfig) -> float:
    a = np.asarray(action, dtype=np.float64)
    pa = np.asarray(prev_action, dtype=np.float64)
    diff = a - pa
    return -cfg.w_jerk * float(np.dot(diff, diff))


def shaping_reward(prev_d: Optional[float], curr_d: float, weight: float) -> float:
    """Potential-based shaping: w * (d_{t-1} - d_t)."""
    if prev_d is None:
        return 0.0
    return weight * (prev_d - curr_d)


def aggregate(parts: dict, use_shaping: bool) -> float:
    """Sum terms per spec §4.2 — picks dense vs shaping path."""
    if use_shaping:
        keys = ("shape_A", "shape_B", "coop", "success", "collision", "action", "jerk")
    else:
        keys = ("align_A", "reach_B", "coop", "success", "collision", "action", "jerk")
    return float(sum(parts.get(k, 0.0) for k in keys))
