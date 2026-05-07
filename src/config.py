"""Central configuration for Tyro env / reward / training.

Mirrors `tyro_design_spec.md` §3 (action scale), §4.3 (reward weights), §5
(termination thresholds), §6 (curriculum). Edit values here, not in the env.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
URDF_DIR = DATA_DIR / "urdf"


@dataclass
class RewardConfig:
    # Distance / angle weights
    w_d_A: float = 1.0
    w_theta_A: float = 0.5
    w_d_B: float = 1.0
    w_theta_B: float = 0.5

    # Cooperation term: r_coop = w_c * exp(-alpha*d_A) * exp(-beta*d_B)
    w_c: float = 2.0
    alpha: float = 10.0
    beta: float = 10.0

    # Sparse success bonus
    R_success: float = 100.0
    eps_A: float = 0.01     # 1 cm — tire-hub position threshold
    delta_A: float = np.deg2rad(5.0)  # 5° — tire-hub angle threshold
    eps_B: float = 0.01     # 1 cm — gripper_B-bolt position threshold

    # Penalties
    w_collision: float = 5.0
    w_action: float = 0.01
    w_jerk: float = 0.01

    # Potential-based shaping (optional; enabled via use_shaping flag in EnvConfig)
    w_shape_A: float = 10.0
    w_shape_B: float = 10.0


@dataclass
class ActionConfig:
    # Per spec §3: Δ end-effector pose, scaled from policy [-1, 1] outputs
    pos_scale: float = 0.02   # m per step
    rot_scale: float = 0.05   # rad per step
    # 13 = 6 (Robot A Δpose) + 6 (Robot B Δpose) + 1 (gripper A binary)
    dim: int = 13


@dataclass
class ObsConfig:
    # Normalization scales (per spec §2.2)
    workspace_radius: float = 1.3   # UR10 reach
    max_joint_vel: float = 3.15     # rad/s, conservative for both arms
    dim: int = 86


@dataclass
class CurriculumConfig:
    # Domain randomization phases (spec §6). Caller advances phase via env.set_phase().
    phase: int = 1                  # 1 = fixed, 2 = ±2cm, 3 = ±5cm
    phase_ranges_cm: Tuple[float, float, float] = (0.0, 2.0, 5.0)


@dataclass
class EnvConfig:
    # Simulation
    sim_freq_hz: float = 240.0
    control_freq_hz: float = 20.0   # decimation = sim_freq / control_freq = 12
    max_steps: int = 500            # ≈ 25 s at 20 Hz
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    render: bool = False            # GUI vs DIRECT

    # Reward shaping toggle (Stage 4 in spec §4.3)
    use_shaping: bool = False

    # Robot base placement. Both robots stand on the +x side of a vertical truck
    # wheel hub (bolt axis = world +x). Yaw points each base toward the hub so
    # that joint-zero configurations face the workspace.
    robot_A_base_pos: Tuple[float, float, float] = (0.9, -0.35, 0.0)
    robot_A_base_rpy: Tuple[float, float, float] = (0.0, 0.0, np.pi)
    robot_B_base_pos: Tuple[float, float, float] = (0.9, 0.35, 0.0)
    robot_B_base_rpy: Tuple[float, float, float] = (0.0, 0.0, np.pi)

    # Hub / target placement (mounted on a virtual truck side wall, bolts protrude +x)
    hub_pos_nominal: Tuple[float, float, float] = (0.0, 0.0, 0.6)
    hub_axis_world: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    hub_radius: float = 0.15
    hub_thickness: float = 0.05

    # Tire geometry (kept simple — primitive cylinder)
    tire_outer_radius: float = 0.30
    tire_inner_radius: float = 0.16
    tire_thickness: float = 0.20

    # Bolt pattern
    n_bolts: int = 8
    bolt_circle_radius: float = 0.11
    bolt_length: float = 0.04
    bolt_radius: float = 0.008

    # URDF paths
    ur10_urdf: str = str(URDF_DIR / "ur10_robot" / "ur10_robot.urdf")
    ur10_search_path: str = str(URDF_DIR / "ur10_robot")
    panda_urdf: str = "franka_panda/panda.urdf"  # resolved via pybullet_data

    # Sub-configs
    reward: RewardConfig = field(default_factory=RewardConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    @property
    def decimation(self) -> int:
        return int(round(self.sim_freq_hz / self.control_freq_hz))


# ----------------------------------------------------------------------
# Stage helper (spec §4.3): incrementally enable reward terms.
# ----------------------------------------------------------------------
def make_reward_config(stage: int) -> RewardConfig:
    """RewardConfig with weights gated to a given training stage.

    Stage 1 — per-robot tasks only (align_A + reach_B).
    Stage 2 — + cooperation term.
    Stage 3 — + success bonus + collision/action/jerk penalties (full dense).
    Stage 4 — replace dense distance with potential shaping; coop/success/penalties stay.
    """
    rc = RewardConfig()
    if stage == 1:
        rc.w_c = 0.0
        rc.R_success = 0.0
        rc.w_collision = 0.0
        rc.w_action = 0.0
        rc.w_jerk = 0.0
    elif stage == 2:
        rc.R_success = 0.0
        rc.w_collision = 0.0
        rc.w_action = 0.0
        rc.w_jerk = 0.0
    elif stage == 3:
        pass  # all defaults
    elif stage == 4:
        # use_shaping toggle (set on EnvConfig) replaces align/reach with shape_*.
        # Keep coop/success/penalties as in stage 3.
        rc.w_d_A = 0.0
        rc.w_theta_A = 0.0
        rc.w_d_B = 0.0
        rc.w_theta_B = 0.0
    else:
        raise ValueError(f"unknown stage {stage}; valid: 1..4")
    return rc


def make_env_config(stage: int = 3, phase: int = 1, **overrides) -> EnvConfig:
    """EnvConfig wired up for a given (stage, phase). Used by train/eval scripts."""
    cfg = EnvConfig(**overrides)
    cfg.reward = make_reward_config(stage)
    cfg.use_shaping = (stage == 4)
    cfg.curriculum.phase = phase
    return cfg
