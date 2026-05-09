"""Reward terms from `tyro_design_spec.md` §4.

Each function returns a scalar (float) and accepts already-extracted geometry
so it stays unit-tested without PyBullet. The env aggregates terms in step().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import RewardConfig
from .utils import angle_between


@dataclass
class RewardBreakdown:
    """Per-term reward log — env writes this into info['reward_terms']."""
    align_A: float = 0.0
    reach_B: float = 0.0
    coop: float = 0.0
    sync_joint_A: float = 0.0
    success: float = 0.0
    collision: float = 0.0
    workspace: float = 0.0
    action: float = 0.0
    jerk: float = 0.0
    shape_A: float = 0.0
    shape_B: float = 0.0
    total: float = 0.0
    dense_total_pre_mix: float = 0.0
    # auxiliary (for logging / termination)
    d_A: float = 0.0
    theta_A: float = 0.0
    d_B: float = 0.0
    theta_B: float = 0.0
    is_success: bool = False
    axial_dot_th: float = 0.0
    lateral_th: float = 0.0
    lug_spin_err_rad: float = 0.0


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


def sync_joint_a_penalty(qdot_a: np.ndarray, cfg: RewardConfig) -> float:
    """Penalty on squared joint velocity magnitude for Robot A (UR10)."""
    q = np.asarray(qdot_a, dtype=np.float64)
    return -cfg.w_sync_joint_a * float(np.dot(q, q))


def success_bonus(
    d_A: float,
    theta_A: float,
    d_B: float,
    theta_B: float,
    cfg: RewardConfig,
    mount: Optional[Tuple[float, float, float]] = None,
) -> tuple[float, bool]:
    """Sparse success predicate.

    If ``mount`` is ``(axial_dot, lateral_norm, lug_spin_err_rad)`` and
    ``cfg.use_lug_aligned_success``, require hub-face-style proximity using those
    terms in addition to the gripper / axis checks.
    """
    if cfg.R_success <= 0.0:
        return 0.0, False

    grip_ok = (
        theta_A < cfg.delta_A
        and d_B < cfg.eps_B
        and theta_B < cfg.delta_B
    )
    if not grip_ok:
        return 0.0, False

    if cfg.use_lug_aligned_success and mount is not None:
        axial, lat, lug = mount
        pos_ok = d_A < cfg.eps_A_mounted
        mating_ok = (
            abs(axial - cfg.success_axial_dot_target) < cfg.success_axial_tolerance
            and lat < cfg.success_lateral_tolerance
            and lug < cfg.lug_spin_tolerance_rad
        )
        ok = pos_ok and mating_ok
        return ((cfg.R_success if ok else 0.0), bool(ok))

    ok = grip_ok and (d_A < cfg.eps_A)
    return (cfg.R_success if ok else 0.0), bool(ok)


def collision_penalty(in_collision: bool, cfg: RewardConfig) -> float:
    return -cfg.w_collision if in_collision else 0.0


def workspace_penalty(out_of_workspace: bool, cfg: RewardConfig) -> float:
    return -cfg.w_workspace if out_of_workspace else 0.0


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


def aggregate(parts: dict, use_shaping: bool, rcfg: RewardConfig) -> float:
    """Combine dense process reward with sparse success using configured mix."""
    common = ("coop", "sync_joint_A", "collision", "workspace", "action", "jerk")
    if use_shaping:
        dense_core = float(parts.get("shape_A", 0.0) + parts.get("shape_B", 0.0))
    else:
        dense_core = float(parts.get("align_A", 0.0) + parts.get("reach_B", 0.0))
    dense = dense_core + float(sum(parts.get(k, 0.0) for k in common))
    succ = float(parts.get("success", 0.0))
    return (
        rcfg.mix_dense * dense
        + rcfg.mix_sparse_success * succ
    )
