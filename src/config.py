"""Central configuration for Tyro env / reward / training.

Mirrors `tyro_design_spec.md` §3 (action scale), §4.3 (reward weights), §5
(termination thresholds), §6 (curriculum). Edit values here, not in the env.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

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

    # UR10 cooperative sync — penalize large joint velocity on A so Panda can refine.
    w_sync_joint_a: float = 0.08

    # Sparse / dense balancing (applied to sparse success vs dense process total).
    mix_sparse_success: float = 0.7
    mix_dense: float = 0.3

    # Sparse success bonus
    R_success: float = 100.0
    #: Legacy Euclidean ||tire − hub|| gate. Only used when
    #: ``use_lug_aligned_success`` is False — otherwise ``eps_A_mounted`` applies.
    eps_A: float = 0.01     # 1 cm — tire-hub position threshold (legacy tight bound)
    delta_A: float = np.deg2rad(5.0)  # 5° — tire-hub angle threshold
    eps_B: float = 0.01     # 1 cm — gripper_B-bolt position threshold
    delta_B: float = np.deg2rad(5.0)  # 5° — gripper-bolt axis threshold

    # Mounting-aligned success (tire coaxial projection + lug spin vs bolt_0 ray).
    #: Default True ⇒ ``eps_A`` is unused; success uses ``eps_A_mounted`` /
    #: ``success_axial_tolerance`` / ``success_lateral_tolerance`` / ``lug_spin_tolerance_rad``.
    use_lug_aligned_success: bool = True
    #: ``dot(t_center - hub_center, û_hub)`` when seated (often near 0; tune with scene scale).
    success_axial_dot_target: float = 0.0
    success_axial_tolerance: float = 0.08
    success_lateral_tolerance: float = 0.065
    lug_spin_tolerance_rad: float = np.deg2rad(22.0)  # < half of 36° lug pitch
    #: Loosened Euclidean ||tire−hub|| cap when lug+axial gates are checked.
    eps_A_mounted: float = 0.22

    # Penalties
    w_collision: float = 5.0
    w_workspace: float = 5.0
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
    workspace_radius: float = 2.0   # large truck tire + lift layout
    max_joint_vel: float = 3.15     # rad/s, conservative for both arms
    dim: int = 89                   # §2.1 base + mounting (ax·lat·lug) diagnostics


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

    # Bullet stability (recommended: numSubSteps 4–8, ERP ~0.15, CFM ~1e-5)
    physics_num_sub_steps: int = 6
    contact_erp: float = 0.15
    contact_cfm: float = 1e-5  # passed as PyBullet ``globalCFM`` at reset

    # Terminate when any contact reports excessive normal force (simulated breakage).
    # Set ≤ 0 to disable.
    contact_force_terminate_above: float = 2500.0

    # Reward shaping toggle (Stage 4 in spec §4.3)
    use_shaping: bool = False

    # Service layout — bases separated on −X/Y so UR10/Panda home + pre-grasp tire do not clip hub/cargo.
    robot_A_base_pos: Tuple[float, float, float] = (-0.42, -0.58, 0.0)
    robot_A_base_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_B_base_pos: Tuple[float, float, float] = (-0.42, 0.58, 0.0)
    robot_B_base_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ISO 335 mm PCD truck hub (lift); URDF studs along hub **local +Z**.
    #: Pitch −π/2 (Y) maps local +Z → **world −X** so studs face Robot A/B (bases on −X).
    #: If bolts point at the sky/axis, flip sign to ``+π/2`` instead.
    hub_pos_nominal: Tuple[float, float, float] = (1.08, 0.0, 0.82)
    hub_base_rpy: Tuple[float, float, float] = (0.0, -np.pi / 2, 0.0)
    hub_axis_world: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    hub_radius: float = 0.21   # ø420 mm flange
    hub_thickness: float = 0.06

    # Tire 295/80R22.5-ish; visuals = outer cylinder; collision = hollow annulus (scene).
    #: Tire base mass (kg). Real truck tyres are ~50–80 kg but the assembly
    #: is rigidly grasped to the UR10 EE — high mass loads every joint torque
    #: command and slows reactive control. Empirically 0.5 kg is the sweet
    #: spot for the current PD tuning: d_A baseline matches the bare-arm home
    #: pose (no controller overshoot during the constraint-creation transient
    #: on step 1), while keeping enough inertia for "tyre-aware" policy
    #: dynamics. Avoid 1.0–3.0 kg — that range hits an underdamped resonance
    #: with PyBullet's default position-control gains. Tune via ``--tire-mass``.
    tire_mass: float = 0.5
    tire_outer_radius: float = 0.525
    #: Tread ring inner radius (wheel-well cavity); keep > hub pilot & flange for slide fit.
    tire_inner_radius: float = 0.282
    tire_thickness: float = 0.30
    tire_hollow_collision: bool = True
    #: Compound annulus = N boxes in XY tiling (axle +local Z before baseOrientation).
    #: ``three_piece`` disk uses child links — tread alone may use almost the full ~16 cap.
    tire_annulus_collision_segments: int = 16
    #: Upper bound on tread+disk compound size; clipped to Bullet visual cap (~16) in ``scene``.
    tire_collision_max_primitives: int = 42
    #: If ``None``, use ``tire_inner_radius``.
    tire_collision_inner_radius: Optional[float] = None
    #: Scene spawn before EE attach — keep ahead of UR10 home & clear of torso (≠ final grasp pose).
    tire_spawn_offset_from_robot_a: Tuple[float, float, float] = (0.92, -0.02, 0.88)
    #: Lug-disk blockers around PCD (between holes). ``False`` ⇒ tread hollow ring만.
    tire_wheel_disk_enabled: bool = True
    #: ``three_piece``: 10 holes × 3 silver boxes (``src/env/models.py``). ``inter_lug_wedge``: legacy radial slabs.
    tire_wheel_disk_style: str = "three_piece"
    wheel_disk_thickness: float = 0.02
    #: Clearance hole radius on the wheel disk (>molded M22 stud for learnability).
    wheel_disk_bolt_hole_radius: float = 0.018
    #: 각 런 홀에 대한 방사 간격 허반경(폭)·PCD 접선 각도 갭 계산용 (볼트 반지름 보다 크게).
    wheel_disk_bolt_gap_clearance_radius: float = 0.015
    #: 휠 디스크 쐐기의 안쪽 반지름(축 방향 타원판에서 허브 쪽 채움). 바깥은 기본값으로 타이어 보어 ``tire_collision_inner_radius``/``tire_inner_radius``.
    wheel_disk_radial_inner: float = 0.10
    #: 쐐기 바깥 반지름 고정값(선택). ``None``이면 스폰 시 타이어 보어 반경을 써 허브·볼트와 동축 정렬합니다.
    wheel_disk_radial_outer: Optional[float] = None
    #: 볼트 0번 시작 위상 (rad). 생성기 ``--bolt-pattern-phase-deg`` 와 같은 각(n=10 ⇒ 360°/10 균등)을 적용해야 휠 틈과 스터드가 맞습니다.
    wheel_disk_bolt_phase_rad: float = 0.0

    # ISO 335 mm PCD — 10× M22 studs (generator + primitives).
    n_bolts: int = 10
    bolt_circle_radius: float = 0.1675
    bolt_length: float = 0.10
    bolt_radius: float = 0.011

    # Scene source: URDF aggregates hub + bolts; primitive path matches prior behavior.
    use_truck_hub_urdf: bool = True
    truck_wheel_station_urdf: str = str(
        URDF_DIR / "truck_assembly" / "truck_wheel_station.urdf"
    )

    # Cargo body (+X aft of hub face): Cargo_X − Hub_X ≈ +0.15 m (허브 뒤 벽 역할).
    spawn_vehicle_primitive_box: bool = True
    #: Full box size ≈ [0.5, 2.0, 1.0] m → half-extents below; center 펜더/상단 근처.
    vehicle_half_extents: Tuple[float, float, float] = (0.25, 1.0, 0.5)
    #: 카고 전면(min X)이 ``hub_pos_nominal`` 의 플랜지 대비 과도하게 겹치지 않게 +X로 띄움(nominal 기준).
    vehicle_center_world: Tuple[float, float, float] = (1.42, 0.0, 1.38)
    #: Carve cargo collision/visual where tire+hub mount (axis ≈ world +X through hub).
    cargo_use_wheel_well_cutout: bool = True
    cargo_wheel_well_radius_yz: float = 0.63
    #: World-x interval [hub_x + lo, hub_x + hi] along which YZ cylinder cut applies.
    cargo_wheel_well_x_range_from_hub: Tuple[float, float] = (-0.65, 0.42)
    #: Sub-box grid over vehicle AABB (product ≤ ~16 for stable compounds).
    cargo_collision_subdiv: Tuple[int, int, int] = (3, 2, 2)

    # Bolt surface properties (helps avoid unrealistic sticking on micro-contacts).
    bolt_lateral_friction: float = 0.8
    bolt_spinning_friction: float = 0.01

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

    @property
    def spawn_cargo_box(self) -> bool:
        """Alias for ``spawn_vehicle_primitive_box`` (cargo / wheel-well proxy)."""

        return self.spawn_vehicle_primitive_box

    @spawn_cargo_box.setter
    def spawn_cargo_box(self, v: bool) -> None:
        self.spawn_vehicle_primitive_box = bool(v)


# ----------------------------------------------------------------------
# Stage helper (spec §4.3): incrementally enable reward terms.
# ----------------------------------------------------------------------
def make_reward_config(stage: int) -> RewardConfig:
    """RewardConfig with weights gated to a given training stage.

    Stage 1 — per-robot tasks only (align_A + reach_B).
    Stage 2 — + cooperation + UR10 cooperative sync penalty.
    Stage 3 — + success bonus + collision/action/jerk penalties (full dense).
    Stage 4 — replace dense distance with potential shaping; coop/success/penalties stay.
    """
    rc = RewardConfig()
    if stage == 1:
        rc.w_c = 0.0
        rc.w_sync_joint_a = 0.0
        rc.R_success = 0.0
        rc.w_collision = 0.0
        rc.w_workspace = 0.0
        rc.w_action = 0.0
        rc.w_jerk = 0.0
        rc.mix_sparse_success = 0.0
        rc.mix_dense = 1.0
    elif stage == 2:
        rc.R_success = 0.0
        rc.w_collision = 0.0
        rc.w_workspace = 0.0
        rc.w_action = 0.0
        rc.w_jerk = 0.0
        rc.mix_sparse_success = 0.0
        rc.mix_dense = 1.0
    elif stage == 3:
        pass  # all defaults (including sparse/dense mix and sync penalty)
    elif stage == 4:
        # use_shaping toggle (set on EnvConfig) makes aggregate() pick shape_*
        # instead of align_A/reach_B for the geometric dense core.
        pass
    else:
        raise ValueError(f"unknown stage {stage}; valid: 1..4")
    return rc


def make_env_config(stage: int = 3, phase: int = 1, **overrides) -> EnvConfig:
    """EnvConfig wired up for a given (stage, phase). Used by train/eval scripts."""
    cf_key = "contact_force_terminate_above"
    cf_user_set = cf_key in overrides
    cfg = EnvConfig(**overrides)
    cfg.reward = make_reward_config(stage)
    cfg.use_shaping = (stage == 4)
    cfg.curriculum.phase = phase
    # Stages 1–2 omit collision/contact *penalties*; Bullet still reports large
    # normal forces from tire–EE fixed constraints → avoid instant episode death.
    if stage <= 2 and not cf_user_set:
        cfg.contact_force_terminate_above = 0.0
    return cfg
