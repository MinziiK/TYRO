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
    # Distance / angle weights (Stage-1 align: tire COM → hub centre)
    w_d_A: float = 1.0
    w_theta_A: float = 0.5
    w_d_B: float = 1.0
    w_theta_B: float = 0.5

    # ----- FSM-specific dense weights -------------------------------------
    #: Stage 0 — UR10 EE → tire 6 o'clock outer grasp anchor. Used as the
    #: amplitude of a *positive* exponential shaping term
    #: ``r = w_approach * exp(-d / approach_decay)`` so the dense reward is
    #: bounded in ``[0, w_approach]`` instead of growing unbounded-negative
    #: with distance — this removes the per-step negative baseline that
    #: previously incentivised the policy to self-terminate early.
    #: **2026-05-28 emergency sharpening**: the 1.5 m e-fold was so flat
    #: across d ∈ [0.6, 0.8] (kernel 0.66 → 0.59, only ~10 % drop) that
    #: the 450 k policy locked into a hover at d ≈ 0.69 m without any
    #: gradient to keep closing. Decay slashed to **0.4 m** — kernel now
    #: reads 0.22 → 0.13 across the same band (40 % drop), restoring a
    #: strong "keep descending" gradient. Combined with the new close-range
    #: bonus below (see ``w_approach_close`` / ``approach_close_decay``).
    w_approach: float = 3.0
    approach_decay: float = 0.4  # m, e-fold radius of the approach kernel
    #: **Stage-0 close-range bonus** — a second, much narrower exponential
    #: that ignites only when the EE is within ~30 cm of the grasp target.
    #: Stacks additively on ``w_approach * exp(-d / approach_decay)`` to
    #: deliver an overwhelming "land it now" signal in the final approach.
    #: Total Stage-0 dense ceiling becomes ``w_approach + w_approach_close``
    #: = 3.0 + 2.0 = 5.0 (only at d=0). At d=0.5 the bonus is 2 · e⁻²·⁵
    #: ≈ 0.16; at d=0.2 it is 2 · e⁻¹ ≈ 0.74; at d=0 it is 2.0. Together
    #: with the hard-cap relaxation to 0.55 m, the close-range slope
    #: should now pull the policy through the gate within the first
    #: ~100 k global timesteps.
    w_approach_close: float = 2.0
    approach_close_decay: float = 0.2  # m, e-fold of the close-range kernel
    #: Stage 2 — tire COM → original floor pickup point (positive exp form).
    w_return: float = 3.0
    return_decay: float = 0.8  # m  (was 0.3 — same rationale as approach)
    #: Potential-based shaping weight applied per step on Δd_approach
    #: (Stage 0) and Δd_return (Stage 2). Provides a dense gradient even
    #: when the exp kernel is near-flat far from the goal:
    #:   shape_step = w_pb * (prev_d - curr_d)
    #: 0.0 disables. Set to ~5 for Phase 1.
    w_pb_approach: float = 5.0
    w_pb_return: float = 5.0
    #: All stages — penalty on tire rotation away from the prescribed
    #: vertical pose (Euler [0°, -90°, 90°]). Acts before the strict
    #: termination gate triggers.
    w_vertical: float = 1.0
    #: Stage 2 — penalty on tire descent speed (encourages soft landing).
    w_landing_speed: float = 0.5

    # FSM transition / completion bonuses
    #: Stage 0 → 1 (successful grasp).
    R_pickup: float = 25.0
    #: Stage 1 → 2 (tire seated within ``mount_radius_tol`` of hub centre).
    R_mount: float = 50.0
    #: Stage 2 → Done (tire placed back on floor pickup zone softly).
    R_return: float = 100.0
    #: One-shot terminal penalty on any *failure* termination (vertical
    #: violation, robot collision, workspace exit, contact-force damage).
    #: Counteracts the "die fast to stop losing reward" exploit when dense
    #: shaping has any negative baseline. Applied in ``step`` whenever
    #: ``terminated and not is_success``.
    R_fail: float = -50.0

    # Cooperation term: r_coop = w_c * exp(-alpha*d_A) * exp(-beta*d_B)
    w_c: float = 2.0
    alpha: float = 10.0
    beta: float = 10.0

    # UR10 cooperative sync — penalize large joint velocity on A so Panda can refine.
    w_sync_joint_a: float = 0.08

    # Sparse / dense balancing (applied to sparse success vs dense process total).
    mix_sparse_success: float = 0.7
    mix_dense: float = 0.3

    # Sparse success bonus (legacy — for Phase 1 FSM, replaced by R_return at
    # episode close-out but kept for backwards-compatible weight gating).
    R_success: float = 100.0
    #: Legacy Euclidean ||tire − hub|| gate. Only used when
    #: ``use_lug_aligned_success`` is False — otherwise ``eps_A_mounted`` applies.
    eps_A: float = 0.01     # 1 cm — tire-hub position threshold (legacy tight bound)
    delta_A: float = np.deg2rad(5.0)  # 5° — tire-hub angle threshold
    eps_B: float = 0.01     # 1 cm — gripper_B-bolt position threshold
    delta_B: float = np.deg2rad(5.0)  # 5° — gripper-bolt axis threshold

    # Mounting-aligned success (tire coaxial projection + lug spin vs bolt_0 ray).
    #: **2026-05-28**: Phase 1 success is decided by the FSM ``landed`` event
    #: (see ``_try_stage_transitions``), so the legacy ``success_bonus``
    #: predicate is no longer evaluated. Field kept for Phase 2/3 forward
    #: compatibility but pinned to ``False`` — the dead lug/axial/lateral
    #: branch in ``rewards.success_bonus`` is now skipped entirely.
    use_lug_aligned_success: bool = False
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

    # When ``use_shaping`` is on, also keep the absolute distance penalty as a
    # dense baseline (``- w_d_* * d_*``) so the policy is not free to drift away
    # whenever the per-step shaping gradient is near zero. Mirrors
    # ``Reward_A = (Φ_t − Φ_{t−1}) − w · d_A`` (Solution A).
    use_dense_baseline_with_shaping: bool = True
    #: Scales the absolute (align_A + reach_B) term when blended with shaping;
    #: 1.0 ⇒ same magnitude as stages 1–3, drop to e.g. 0.1 if shaping should
    #: dominate.
    w_dense_baseline_scale: float = 1.0


@dataclass
class ActionConfig:
    # Per spec §3: Δ end-effector pose, scaled from policy [-1, 1] outputs
    pos_scale: float = 0.02   # m per step
    rot_scale: float = 0.05   # rad per step
    # ``dim`` is set by ``make_env_config`` (or ``EnvConfig.__post_init__``)
    # based on ``EnvConfig.freeze_robot_b``:
    #   Phase 1 (Robot B frozen)  →  6  (Δpose_A 6)         ← gripper_A dropped
    #   Phase 2/3 (Robot B active) → 13 (Δpose_A 6 + Δpose_B 6 + gripper_A 1)
    # The gripper_A channel is a no-op at the sim layer (the JOINT_FIXED
    # constraint at the auto-grasp gate holds the tire), so emitting it in
    # Phase 1 only wastes a search dimension. PPO now solves a clean 6-d
    # delta-EE-pose problem during pickup; gripper_A is re-introduced at
    # the Phase 2 transition (see ``make_env_config``).
    dim: int = 13


@dataclass
class ObsConfig:
    # Normalization scales (per spec §2.2)
    workspace_radius: float = 2.0   # large truck tire + lift layout
    max_joint_vel: float = 3.15     # rad/s, conservative for both arms
    # Set by ``make_env_config`` / ``EnvConfig.__post_init__`` to match
    # the active ``ActionConfig.dim`` — the obs layout always ends with
    # ``prev_action`` (length = action.dim) followed by 3 mount tail
    # scalars, so the total dim tracks ``action.dim``:
    #   action.dim = 6  → obs.dim = 73 + 6 + 3 = 82  (Phase 1)
    #   action.dim = 13 → obs.dim = 73 + 13 + 3 = 89 (Phase 2/3)
    # The 73 base entries (joints, EE/tire/hub/bolt poses, deltas) are
    # independent of action dim and shared across all phases.
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

    # Hub-centric service layout — the truck hub sits at the WORLD ORIGIN
    # (0, 0, 0) so all relative offsets are exact decimals (no float drift
    # against a 0.50/−0.90/0.82 baseline). World Z grows upward, the floor
    # is laid at ``floor_z = −0.82 m`` so the assembly is symmetric about
    # the hub:
    #   * Hub axis  : world −Y (bolts protrude −Y toward the robot bank)
    #   * Cargo     : centred (0, +0.25, +0.56), Y=0 face mates with hub flange
    # ===== Robot B-centric world frame =====
    # The simulation world origin is the **Panda (Robot B) base centre**.
    # All static positions below are expressed in that frame. Relative
    # geometry (A↔B separation, lift travel, reach margins) is identical
    # to the previous hub-centric layout — only the global translation
    # changed by (+0, +0.80, +0.22) so the Panda base sits at (0,0,0).
    # Rationale: simplifies Sim2Real transfer (the real Panda's base
    # frame is the natural reference for Robot B's IK / proprioception)
    # and makes ``_compute_obs`` Robot B-relative by simple subtraction.
    #
    # Layout summary (post-shift):
    #   * Robot B (Panda)  : ( 0.00,  0.00,  0.00)  ← new origin
    #   * Robot A (UR10)   : (−0.80,  0.00, −0.30)  — 0.30 m plinth above floor
    #   * Hub centre       : ( 0.00, +0.80, +0.22)  — 0.80 m forward, on the +Y robot-arm direction
    #   * Tire pickup COM  : (−1.90,  0.00, +0.225) — rack raised to Robot-A-base plane
    #   * Floor plane Z    : −0.60  (= −0.82 + 0.22)
    # Robots share the Y = 0 safety line (= Panda base Y). A↔B
    # separation = 0.80 m. Phase 1 still freezes Panda at HOME (Franka
    # "ready" pose). Reach margins (rack top now flush with UR10 base):
    #   * A → tire pickup ............ 1.10 m planar (UR10 1.32 m → +22 cm)
    #   * A → mount EE target ........ 1.13 m  (UR10 1.32 m → +19 cm)
    #   * B → worst-case lug bolt .... 0.80 m  (Panda 0.825 m → +2.5 cm)
    #   * Lift travel ................ ~0 cm  (pickup ↔ hub Z gap < 1 cm)
    #: World Z height of the ground plane (Robot B-centric).
    floor_z: float = -0.60
    robot_A_base_pos: Tuple[float, float, float] = (-0.80, 0.0, -0.30)
    robot_A_base_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Panda sits at the world origin — the entire scene is expressed
    #: in Robot B's base frame.
    robot_B_base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_B_base_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Stand primitives (cylinder + flange) under each robot base so neither
    #: appears to float above the floor. Heights derive from
    #: ``base_z − floor_z`` (UR10 = 0.20 m, Panda = 0.60 m). Set radius ≤ 0
    #: to disable.
    ur10_stand_radius: float = 0.12
    ur10_stand_rgba: Tuple[float, float, float, float] = (0.35, 0.38, 0.42, 1.0)
    panda_stand_radius: float = 0.12
    panda_stand_rgba: Tuple[float, float, float, float] = (0.35, 0.38, 0.42, 1.0)

      # ISO 335 mm PCD truck hub (lift); URDF studs along hub **local +Z**.
    #: Composed rotation: pitch −π/2 (Y) maps local +Z → world −X, then yaw
    #: +π/2 (Z) rotates that to **world −Y** (still −Y *in the rotated*
    #: hub frame). After the Robot B-centric shift the hub sits 0.80 m in
    #: front of the Panda base (+Y world), elevated 0.22 m above the base
    #: plane. Bolts protrude **toward** the Panda (−Y world direction,
    #: i.e. toward the robot bank) so Panda's tool +Z aims +Y to face
    #: them. The hub *axis* direction (``hub_axis_world``) is invariant
    #: under translation and stays (0, −1, 0).
    hub_pos_nominal: Tuple[float, float, float] = (0.0, 0.80, 0.22)
    hub_base_rpy: Tuple[float, float, float] = (0.0, -np.pi / 2, np.pi / 2)
    hub_axis_world: Tuple[float, float, float] = (0.0, -1.0, 0.0)
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
    #: Phase 1 FSM — tire pickup pose (Robot B-centric world frame).
    #: Tire spawns standing vertically with its **bore axis pointing at
    #: robot A (world +X)** so the bore hole is visible from the robot
    #: side. The dual-block rack is a Y-split V-cradle (two 30 cm rails
    #: along X, **50 cm Y-gap**, **45 cm tall** rails — see
    #: ``tire_rack_*`` below). Geometry budget:
    #:   * X = −1.90: tire thickness 0.30 m along the bore axis → tire
    #:     spans X ∈ [−2.05, −1.75]. Both rack rails span X ∈ [−2.05,
    #:     −1.75] (30 cm long), flush with the tread's front + back
    #:     faces.
    #:   * Y = 0.00: bore axis world +X, tread plane in Y-Z → tire
    #:     spans Y ∈ [−0.525, +0.525] (1.05 m diameter). Inner rail at
    #:     Y = +0.30 / outer rail at Y = −0.30. The tire's tread rests
    #:     on the inner-top corners of both rails at (Y = ±0.25,
    #:     Z = −0.15), forming a stable V-cradle. The 6 o'clock line
    #:     (Y = 0) sits in the 50 cm Y-gap; static stability is held
    #:     by ``_pin_tire_to_world`` (mass = 0 freeze) at the cradle
    #:     equilibrium until Stage 0 → 1 fires.
    #:   * Z = rail_top + √(R² − 0.25²) = −0.15 + 0.46165 = **+0.3117**.
    #:     This is the geometric resting COM where the tread surface
    #:     just touches the rail inner-top corners. The 6 o'clock anchor
    #:     ends up at Z = COM − R = **−0.2133** (≈ 6.3 cm below the
    #:     rail top, inside the 50 cm Y-gap).
    #: UR10 base ↔ pickup grasp target ≈ 1.10 m planar (the 7.7 cm Z
    #: offset from the UR10 base plane Z = −0.30 to grasp Z = −0.213 is
    #: well within the IK's reach margin). Stage 1 trajectory pickup
    #: → hub centre is ≈ 2.05 m (dominant carry skill).
    tire_pickup_pos: Tuple[float, float, float] = (-1.90, 0.0, 0.3117)
    #: Tire spawn orientation as RPY. (0, π/2, 0) sends the tire's local
    #: +Z (bore axis) to **world +X** so the bore opening faces robot A
    #: directly (the robot sits at +X relative to the tire). The mount
    #: target hub still has bore axis along world −Y (``hub_axis_world``)
    #: — Stage 1 carry includes a 90° tire rotation about world +Z to
    #: align the bore with the hub flange. See ``vertical_tol_rad`` /
    #: stage-1 vertical-check carve-out in ``tyro_env`` for the FSM
    #: implications of this spawn-vs-mount-axis split.
    tire_spawn_rpy: Tuple[float, float, float] = (0.0, np.pi / 2, 0.0)
    #: Reference axis used by ``_tire_vertical_error`` to enforce the
    #: spawn vertical pose (Stages 0 + 2). Matches the world direction of
    #: the tire's local +Z under ``tire_spawn_rpy``. Stage 1 (carry) waives
    #: this check so the policy is free to rotate the tire toward
    #: ``hub_axis_world`` for mount alignment.
    tire_spawn_axis_world: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    #: Hub-mounting target in world (= hub_pos_nominal under the new
    #: Robot B-centric layout).
    tire_mount_pos: Tuple[float, float, float] = (0.0, 0.80, 0.22)
    #: Stage 0 → 1 trigger: UR10 EE within this radius of the tire 6 o'clock
    #: outer point (= tire COM + (0, 0, -R)) ⇒ create grasp constraint.
    #: This is the **final / hard** tolerance the env asymptotes to once
    #: the curriculum schedule below has fully ramped down (see
    #: ``approach_tol_soft`` / ``*_curriculum_*``). The env reads
    #: ``self._approach_tol`` at every step — that runtime field is what
    #: actually gates Stage 0 → 1 and is updated by SB3's
    #: ``ApproachTolCurriculumCallback`` (or stays at ``approach_radius_tol``
    #: when no callback is wired, e.g. during eval / render).
    #: **2026-05-28 (palm-up HOME + curriculum widening)**: previous soft
    #: gate of 0.15 m left a 36 cm dead band against HOME EE↔grasp ≈ 51 cm,
    #: i.e. the policy saw zero pickup signal for the first ~5k steps.
    #: ``approach_tol_soft`` raised to **0.35 m** so even the initial
    #: random-policy exploration cone (Δpos ≤ 0.02 m/step × 500 steps =
    #: 10 m envelope) reliably trips Stage 0 → 1 within an episode. The
    #: hard cap stays at **0.08 m** (physical fingertip contact) for the
    #: final converged policy.
    approach_radius_tol: float = 0.08
    approach_tol_soft: float = 0.35
    #: First N (global PPO) timesteps where the gate is pinned to
    #: ``approach_tol_soft``. Past this point the linear ramp begins.
    #: **Aggressively shortened** from 100 k → **5 k env-steps** (~40 k
    #: global timesteps under 12-env vec). The diagnostic hover already
    #: emerged after 450 k global, so the soft hold doesn't need a 100 k
    #: pre-roll any more.
    approach_tol_curriculum_steps: int = 5_000
    #: Linear ramp length (global PPO timesteps) after the soft hold
    #: during which the gate is interpolated from ``approach_tol_soft``
    #: down to ``approach_radius_tol``.
    #: Shortened from 200 k → **5 k env-steps** so the curriculum fully
    #: lands inside the first ~80 k global timesteps.
    approach_tol_ramp_steps: int = 5_000
    #: Stage 1 → 2 trigger: ‖tire − hub‖ < ``mount_radius_tol`` AND tire axis
    #: aligned with hub axis (≤ ``RewardConfig.delta_A`` rad).
    #: **2026-05-28**: relaxed from 0.01 m → **0.04 m** so Stage 1 has a
    #: realistic completion gate. Required mount precision is enforced
    #: separately by the lug-aligned tolerances (``eps_A_mounted`` etc.).
    mount_radius_tol: float = 0.04
    #: Stage 2 → success: ‖tire − pickup‖ < ``return_radius_tol`` AND tire
    #: descent speed < ``landing_speed_max`` (soft landing).
    return_radius_tol: float = 0.05
    landing_speed_max: float = 0.10
    #: Vertical tolerance applied only in Stages 0 / 2 (pickup pose +
    #: post-landing pose). Episodes terminate (penalty) when the tire's
    #: bore axis deviates from ``tire_spawn_axis_world`` by more than
    #: this angle (rad). Stage 1 (carry/mount) waives the check entirely
    #: so the policy can rotate the tire 90° about world +Z to align
    #: the bore with ``hub_axis_world`` for mount.
    vertical_tol_rad: float = np.deg2rad(15.0)
    #: Legacy (kept for backwards-compat with utility scripts). The Phase 1
    #: FSM uses ``tire_pickup_pos`` directly so this offset is no longer
    #: consumed by the env, but render/calibrate scripts may still read it.
    tire_spawn_offset_from_robot_a: Tuple[float, float, float] = (0.40, 0.80, 0.88)
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

    # Cargo body: rotated +π/2 yaw so the chassis reads as a truck oriented
    #: along the world X driving corridor.  ``half_extents[1] = 1.0`` so
    #: the rotated chassis spans **world X ∈ [−1.00, +1.00] (2 m long)**
    #: — the cargo wraps around the hub on both sides like an actual
    #: truck chassis. Under the Robot B-centric frame the cargo's −Y
    #: face sits at world Y = +0.80 (= hub flange plane). The dual-block
    #: rack is well below the chassis and the UR10's tire-fetch sweep
    #: stays south of the chassis on Y ≈ 0.
    spawn_vehicle_primitive_box: bool = True
    vehicle_half_extents: Tuple[float, float, float] = (0.25, 1.0, 0.5)
    #: Yaw +π/2 → world footprint 2.00 m (X) × 0.50 m (Y) × 1.00 m (Z).
    vehicle_base_rpy: Tuple[float, float, float] = (0.0, 0.0, np.pi / 2)
    #: Centre at (0, +1.05, +0.78): Y=+1.05 ± 0.25 → Y range
    #: [+0.80, +1.30] so the −Y face mates with the hub flange at
    #: Y = +0.80. Z=+0.78 ± 0.50 → Z range [+0.28, +1.28] clears the
    #: hub crown at Z = +0.22 ± 0.03.
    vehicle_center_world: Tuple[float, float, float] = (0.0, 1.05, 0.78)
    cargo_use_wheel_well_cutout: bool = True
    #: Wheel-well cylinder coaxial with the (new) hub axis = world −Y, so the
    #: arch opens through the cargo face that now faces −Y in world frame.
    cargo_wheel_well_axis: str = "y"
    cargo_wheel_well_radius: float = 0.85
    cargo_wheel_well_radius_yz: float = 0.85
    cargo_wheel_well_along_range_from_hub: Tuple[float, float] = (-0.50, 0.50)
    cargo_wheel_well_x_range_from_hub: Tuple[float, float] = (-0.65, 0.85)
    #: Subdiv (nx, ny, nz) in cargo **local** frame. With the +π/2 yaw, local
    #: Y → world X (long 2 m chassis), local X → world Y (0.5 m thickness).
    #: ``nx=1`` keeps the 0.5 m thickness as a single slab while ``ny=6`` and
    #: ``nz=3`` carve a 6×3 façade visible from the −Y robot side; the
    #: wheel-well cutout removes the middle/bottom cells, leaving two
    #: vertical pillars plus a roof so the arch opens around the tire.
    cargo_collision_subdiv: Tuple[int, int, int] = (1, 6, 3)

    # ------------------------------------------------------------------
    # Inline (Y-split) dual-block tire rack (Phase 1 pickup support).
    # ------------------------------------------------------------------
    #: Two short rails along the X direction (30 cm long each, matching
    #: the tire's bore-axis thickness) flanking the tire in Y with a
    #: **50 cm Y-gap** centred on the bore axis (Y = 0). The rails form
    #: a V-cradle: the tire (radius 0.525 m) rests with its tread
    #: surface in contact with the inner-top corners of both rails
    #: (Y = ±0.25, Z = rail_top). **2026-05-29 (rev 3)**: revised again
    #: from the 70 cm gap so the tire physically sits on the rack
    #: instead of floating above it. The 50 cm Y-gap leaves 25 cm of
    #: tread overhang on each side, enough for a stable cradle while
    #: still giving the UR10 gripper a comfortable plunge corridor
    #: (effective free Y at the 6 o'clock contact line ≈ 50 cm).
    #:
    #: NOTE on the bore=+X spawn (``tire_spawn_rpy = (0, π/2, 0)``):
    #: with the tire bore facing robot A in +X, the 6 o'clock outer
    #: tread line at Y = 0 sits **below** the rail top plane (Z =
    #: -0.213 vs rail top Z = -0.15) — the cradle geometry pulls the
    #: tire's bottom into the gap. The gripper still approaches from
    #: above through the gap (Y = 0, gap width 0.50 m) and dives to
    #: Z ≈ -0.21 to reach the 6 o'clock anchor; no rail clipping since
    #: the gripper centreline is 25 cm from either rail face.
    #:
    #: Static stability is maintained by ``_pin_tire_to_world``
    #: (mass = 0 freeze) which holds the tire at the cradle equilibrium
    #: pose until Stage 0 → 1 fires.
    #:
    #: Geometry (Robot B-centric, ``tire_outer_radius = 0.525``,
    #: ``tire_thickness = 0.30``, rails 30 cm long × 10 cm thick ×
    #: **45 cm tall**, ≈ 1.5× the original 30 cm height):
    #:   * Inner rail Y range = [+0.25, +0.35] (centre +0.30 ± 0.05).
    #:   * Outer rail Y range = [-0.35, -0.25] (centre -0.30 ± 0.05).
    #:   * Gap in Y = [-0.25, +0.25], width **0.50 m** ✓
    #:   * Both rails X range = [-2.05, -1.75] (centre -1.90 ± 0.15) —
    #:     matches the tire's bore-axis extent X ∈ [-2.05, -1.75]
    #:     exactly, so the +X / −X ends of the rails terminate flush
    #:     with the tread's front and back faces.
    #:   * Rail Z range = [-0.60, -0.15]; centre -0.375, half-ext 0.225
    #:     (= floor + 0.45 m height). Rail top Z = **-0.15** (was -0.30).
    #:   * Tire COM Z = rail_top + √(R² − 0.25²)
    #:                = -0.15 + √(0.275625 − 0.0625)
    #:                = -0.15 + 0.46165
    #:                ≈ **+0.3117** ✓ (tire rests on rail corners)
    #:   * 6 o'clock anchor (Y=0): Z = COM_Z − R = -0.2133 — the gripper
    #:     dives ≈ 6.3 cm below rail top inside the 50 cm Y-gap.
    spawn_tire_rack: bool = True
    #: Truck-side rail (inner = closer to the truck on the +Y side).
    tire_rack_inner_center: Tuple[float, float, float] = (-1.90, 0.30, -0.375)
    #: Far-side rail (outer = on the −Y side, away from the truck).
    tire_rack_outer_center: Tuple[float, float, float] = (-1.90, -0.30, -0.375)
    #: 30 cm long (X) × 10 cm thick (Y) × **45 cm tall (Z)** rails —
    #: Z half-extent 0.225 (1.5× the original 0.15) so rail top sits at
    #: floor + 0.45 m = -0.15 m world.
    tire_rack_half_extents: Tuple[float, float, float] = (0.15, 0.05, 0.225)
    tire_rack_rgba: Tuple[float, float, float, float] = (0.22, 0.24, 0.28, 1.0)

    # Bolt surface properties (helps avoid unrealistic sticking on micro-contacts).
    bolt_lateral_friction: float = 0.8
    bolt_spinning_friction: float = 0.01

    # URDF paths
    ur10_urdf: str = str(URDF_DIR / "ur10_robot" / "ur10_robot.urdf")
    ur10_search_path: str = str(URDF_DIR / "ur10_robot")
    panda_urdf: str = "franka_panda/panda.urdf"  # resolved via pybullet_data

    #: Phase 1 only: completely freeze Robot B (Panda) at its HOME pose.
    #: When True, the env ignores action[6:12] and never calls
    #: ``robot_B.apply_delta_ee`` — Panda stays parked at HOME for every
    #: step of every episode, removing it from the learning problem.
    #: ``make_env_config`` wires this to ``True`` for Phase 1.
    freeze_robot_b: bool = False
    #: When True, UR10 IK runs **position-only** and the wrist cluster
    #: ``wrist_1 / wrist_2 / wrist_3`` is held at its ``HOME_POSE`` value
    #: every step. The remaining 3 DOFs (``shoulder_pan / shoulder_lift /
    #: elbow``) drive the EE position. Because ``shoulder_pan`` is the
    #: world-Z rotation joint and ``R_z(·)(0,0,1) = (0,0,1)``, the tool
    #: +Z axis stays parallel to world +Z for any pan value — i.e. the
    #: gripper palm always faces straight up while shoulder/elbow reach
    #: in the cylindrical XY plane. ``wrist_2 / wrist_3`` HOME values are
    #: chosen so the wrist_2 link is laid flat in the XY plane and the
    #: wrist_3 (tool roll) axis aligns with world +Z (see ``HOME_POSE``
    #: in ``UR10Robot``). Default **True** for Phase 1.
    ur10_lock_tool_up: bool = True

    # ------------------------------------------------------------------
    # Domain randomization (Phase 1 → Sim2Real bridge)
    # ------------------------------------------------------------------
    #: Master switch for *static-pose* domain randomization. When False,
    #: the env spawns the hub and cargo at exactly the nominal positions
    #: defined above (deterministic — current training default). When
    #: True, the env adds an independent uniform noise to the hub and
    #: cargo XY positions on every ``reset()``, with the magnitude set
    #: by ``RANDOM_POSITION_RANGE``. Hook lives in
    #: ``TyroEnv._maybe_apply_domain_randomization`` (skeleton only —
    #: enable in a later curriculum step once Phase 1 success is stable).
    USE_DOMAIN_RANDOMIZATION: bool = False
    #: Half-range of the uniform noise injected into the hub and cargo
    #: XY positions when ``USE_DOMAIN_RANDOMIZATION`` is True (metres).
    #: Default 2 cm — small enough that mount tolerances are still
    #: reachable but large enough to force the policy to rely on the
    #: observed hub pose rather than a memorised target.
    RANDOM_POSITION_RANGE: float = 0.02

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
def make_reward_config(stage: int, phase: int = 1) -> RewardConfig:
    """RewardConfig with weights gated to a given training stage / phase.

    Stage 1 — per-robot tasks only (align_A + reach_B).
    Stage 2 — + cooperation + UR10 cooperative sync penalty.
    Stage 3 — + success bonus + collision/action/jerk penalties (full dense).
    Stage 4 — add potential shaping (``shape_A/B``). By default the absolute
        distance penalty (``align_A + reach_B``) is *kept* as a dense baseline
        (Solution A: ``r = (Φ_t − Φ_{t−1}) − w · d``) — toggle off via
        ``RewardConfig.use_dense_baseline_with_shaping`` for pure shaping.

    ``phase`` 1 = Phase 1 FSM (UR10 pick-and-place only) — Panda's reach_B
    weights are zeroed because Panda stays parked at HOME and any non-zero
    distance penalty just adds a constant negative baseline that incentivises
    self-termination. phase ≥ 2 keeps the original weights (joint training).
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
        # Shaping + weak absolute-distance baseline (Solution A). Keep dense
        # branch from dominating the return so sparse success (R_success) still
        # drives exploration — see w_dense_baseline_scale / mix_dense below.
        rc.w_dense_baseline_scale = 0.2   # start in 0.1–0.3; tune via CLI
        rc.mix_dense = 0.2
        rc.mix_sparse_success = 0.8
    else:
        raise ValueError(f"unknown stage {stage}; valid: 1..4")

    # Phase 1 FSM gating ------------------------------------------------
    # Panda is locked at HOME during Phase 1 — neutralise every Panda-side
    # reward term so the policy isn't dragged by a constant negative bias.
    if phase == 1:
        rc.w_d_B = 0.0
        rc.w_theta_B = 0.0
        rc.w_c = 0.0  # coop = exp(-α d_A) * exp(-β d_B) — useless without B
    return rc


def make_env_config(stage: int = 3, phase: int = 1, **overrides) -> EnvConfig:
    """EnvConfig wired up for a given (stage, phase). Used by train/eval scripts."""
    cf_key = "contact_force_terminate_above"
    cf_user_set = cf_key in overrides
    freeze_b_user_set = "freeze_robot_b" in overrides
    cfg = EnvConfig(**overrides)
    cfg.reward = make_reward_config(stage, phase)
    cfg.use_shaping = (stage == 4)
    cfg.curriculum.phase = phase
    # Stages 1–2 omit collision/contact *penalties*; Bullet still reports large
    # normal forces from tire–EE fixed constraints → avoid instant episode death.
    if stage <= 2 and not cf_user_set:
        cfg.contact_force_terminate_above = 0.0
    # Phase 1 = UR10 pick-and-place only. Freeze Panda at HOME unless the
    # caller explicitly opted out via ``freeze_robot_b=False``.
    if phase == 1 and not freeze_b_user_set:
        cfg.freeze_robot_b = True

    # Action / observation dims follow ``freeze_robot_b`` — the Panda
    # action block is dropped from ``action_space`` when frozen so PPO
    # doesn't search a 6-d dead manifold, and the matching ``prev_action``
    # slot inside ``obs`` shrinks accordingly. The Phase-1 gripper_A
    # channel is also dropped (sim-side no-op under the auto-grasp
    # constraint). See ``ActionConfig`` / ``ObsConfig`` docstrings.
    if cfg.freeze_robot_b:
        cfg.action.dim = 6
        cfg.obs.dim = 82  # 73 base + 6 prev_action + 3 mount tail
    else:
        cfg.action.dim = 13
        cfg.obs.dim = 89  # 73 base + 13 prev_action + 3 mount tail
    return cfg
