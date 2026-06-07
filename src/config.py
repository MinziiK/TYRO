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
    #: **2026-05-29 (rev 3 — hover-lockin fix)**: tightened from 0.4 → 0.15 m
    #: to break the policy-lock observed in v2 around d_approach ≈ 0.20 m.
    #: With decay = 0.4, the far-term kernel at hover (d=0.20) was 0.61, only
    #: 33 % below its grasp value (d=0.08) of 0.82 — the policy could harvest
    #: ~80 % of the dense reward without ever closing the gap. Slashed to
    #: 0.15 m, the same evaluations become 0.26 (hover) vs 0.59 (grasp), a
    #: 56 % drop. Combined with the boosted R_pickup and per-step alive
    #: penalty below, this re-aligns the value gradient toward an active
    #: descent. The close-range bonus (``w_approach_close``) keeps the
    #: dense reward grow when the EE actually enters the pickup envelope.
    approach_decay: float = 0.15  # m, e-fold radius of the approach kernel
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
    #: Used in the legacy 3-stage FSM (rename: Stage 3 in v6). The dense
    #: kernel ``r = w_return * exp(-d_rack / return_decay)`` pulls the
    #: tire COM toward the cradle pickup pose. v6 tightens the decay so
    #: the policy gets a strong gradient inside the final 50 cm.
    w_return: float = 3.0
    return_decay: float = 0.5  # m (v6: 0.8 → 0.5 for sharper end-game pull)
    #: **v7 (vector-guided carry)** — Stage 1 dense reward overhaul.
    #: Legacy stage-1 ``align_A = -w_d_A·d_A - w_theta_A·θ_A`` is a
    #: *negative-only* kernel: at d_A = 2 m the policy paid -2.5/step
    #: just for standing still after pickup, so the value function
    #: prefers a fast collision-suicide over committing to carry.
    #:
    #: v7 replaces the Stage 1 dense branch with TWO positive kernels:
    #:   1. ``guide_A = w_guide * exp(-||hub - ee|| / guide_decay)``
    #:      — exponential pull on the EE → hub vector (also exposed in
    #:      the observation as a 3-d ``hub_guide_vector``).
    #:   2. ``pb_carry = w_pb_carry * (prev_d_A - d_A)`` — potential-based
    #:      shaping on the tire-to-hub distance, giving a strictly
    #:      positive increment per progress step (matches the existing
    #:      Stage-0 PB approach + Stage-2/3 PB return scheme).
    #:
    #: align_A stays computed for diagnostics but is multiplied by 0
    #: inside the Stage-1 dense subtotal — see ``tyro_env._compute_reward``.
    #: **v9 (2026-05-30)**: 3 → 8 so guide_A stays meaningful at d_A≈2 m
    #: (3·exp(-4)≈0.055/step was drowned by sync_joint / approach dense).
    w_guide: float = 8.0
    guide_decay: float = 0.5  # m (matches return_decay scale)
    #: **v13 (reverted)** — a fine seating kernel ``seat_A = w_seat *
    #: exp(-d_A/seat_decay)`` was tried to pull d_A below the gate, but the
    #: 0605 run showed the d_A≈0.25 m floor is *physical* (identical across
    #: 3 reward configs), not a missing gradient: seat_A left d_A unchanged
    #: while doubling contact force (49→124, max 330) and enabling a
    #: hover-and-farm exploit. Kept at 0 — see env/scene geometry instead.
    w_seat: float = 0.0
    seat_decay: float = 0.10  # m
    #: **v9b**: 5 → 10 — stronger Δd_A gradient toward hub when guide_A
    #: alone does not shrink d_A (v9 stuck at d_A≈2.1 m with guide≈0.12).
    w_pb_carry: float = 10.0
    #: **v11 (2026-05-31 — backtracking)** — Stage 2 demount PB shaping
    #: ``w_pb_demount * (d_hub - d_hub_prev)``. Positive when tire moves
    #: *away* from hub centre (after the demount stall). Larger weight
    #: (20) than carry so the policy gets a strong "pull-back" gradient
    #: as soon as Phase A spawns it at the hub.
    w_pb_demount: float = 20.0
    #: Stage 2 — demount: tire axial-distance from hub centre. After the
    #: ``demount_stall_steps`` stall, the dense kernel
    #: ``r = w_pull * exp(-d_hub / pull_decay)`` *inverts* — the policy is
    #: rewarded for **growing** ``d_hub`` toward ``demount_axial_distance``.
    #: Implemented as ``w_pull * (1 - exp(-d_hub / pull_decay))`` in the
    #: env so the reward asymptotes to ``w_pull`` at the goal distance.
    w_pull_demount: float = 3.0
    pull_decay: float = 0.20  # m
    #: Potential-based shaping weight applied per step on Δd_approach
    #: (Stage 0) and Δd_return (Stage 2). Provides a dense gradient even
    #: when the exp kernel is near-flat far from the goal:
    #:   shape_step = w_pb * (prev_d - curr_d)
    #: 0.0 disables. Set to ~5 for Phase 1.
    w_pb_approach: float = 5.0
    #: **v11**: 5 → 30 — Stage 3 cradle-return PB shaping must dominate
    #: any residual hub-side gradient so the policy commits to returning
    #: instead of drifting around the hub area.
    w_pb_return: float = 30.0
    #: All stages — penalty on tire rotation away from the prescribed
    #: vertical pose (Euler [0°, -90°, 90°]). Acts before the strict
    #: termination gate triggers.
    w_vertical: float = 1.0
    #: Stage 2 — penalty on tire descent speed (encourages soft landing).
    w_landing_speed: float = 0.5

    # FSM transition / completion bonuses
    #: Stage 0 → 1 (successful grasp).
    #: **2026-05-29 (rev 3 — hover-lockin fix)**: raised from 25 → 300.
    #: Empirical budget check in v2 (hover d ≈ 0.20, ep_len ≈ 350): the
    #: dense kernel paid ≈ 2.5/step × 350 = 875 across an episode (before
    #: ``mix_dense``), while a single R_pickup paid only 25 before the
    #: episode immediately ended via ``terminate_on_pickup``. Even
    #: post-mix (0.3 dense vs 0.7 sparse), the hover return ≈ 262 still
    #: dwarfed the pickup return ≈ 17.5, so the policy's value function
    #: rationally chose to hover. R_pickup boosted to **300** so that
    #: ``R_pickup * mix_sparse_success = 210`` alone exceeds a typical
    #: post-mix hover return — combined with the tightened approach
    #: kernel and step alive penalty, hover stops being a Nash equilibrium.
    R_pickup: float = 300.0
    #: Stage 1 → 2 (tire seated within ``mount_radius_tol`` of hub centre).
    #: **2026-05-30 (v6 — 4-stage FSM)**: 200 → **300** so Stage 1 mount
    #: bonus matches the Stage 0 pickup bonus. Stage 1 is the longest and
    #: hardest segment (≈ 1.9 m tire travel + 90° tire rotation + 4 cm
    #: landing tolerance) — a smaller bonus would create a value-gradient
    #: cliff right after the policy "earns" R_pickup. Equalised bonuses
    #: keep the per-step expected return roughly flat across pickup → mount.
    R_mount: float = 300.0
    #: Stage 2 → 3 (tire pulled axially OUT of the hub by at least
    #: ``demount_axial_distance`` m after a ``demount_stall_steps``-step
    #: "virtual fastener release" hold). Stage 2 simulates the gap between
    #: a real Panda finishing its bolt-down and the UR10 retracting the
    #: tire safely along the hub axis. The demount bonus is smaller than
    #: pickup/mount because the motion is shorter (≈ 30 step) and less
    #: error-prone (axial pull along a known hub axis).
    R_demount: float = 150.0
    #: Stage 3 → Done (tire placed back on the cradle pickup pose softly).
    #: **2026-05-30 (v6)**: previously ``R_return = 200``; renamed
    #: ``R_success`` semantically because this is the *final* episode-
    #: terminal bonus. Capped at 300 (not 500 as initially scoped) to
    #: keep critic-loss variance bounded — total sparse pool now sums
    #: to 300 + 300 + 150 + 300 = 1050 (×0.7 mix = 735), comfortably
    #: above the dense ceiling without saturating PPO advantage norm.
    R_success: float = 300.0
    #: Legacy alias retained for any consumer still reading ``R_return``;
    #: kept equal to ``R_success`` so code paths that have not been
    #: migrated continue to credit the same terminal bonus. Both names
    #: point at the same Stage-3 episode-terminal bonus.
    R_return: float = 300.0
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
    #: **v9**: 0.08 → 0.04; **v9b**: 0.04 → 0.02 — v9 still logged
    #: sync_joint_A ≈ -2.5~-4/step drowning carry dense despite w_guide=8.
    w_sync_joint_a: float = 0.02

    # Sparse / dense balancing (applied to sparse success vs dense process total).
    mix_sparse_success: float = 0.7
    mix_dense: float = 0.3

    # Legacy ``R_success`` (Phase 0/legacy success_bonus predicate) duplicate
    # field removed in v6 — the canonical ``R_success`` (= 300.0) is defined
    # above next to ``R_demount``. ``make_reward_config`` still writes to
    # ``rc.R_success`` to gate the legacy lug-aligned success predicate; that
    # path now toggles the same v6 terminal bonus value, which is fine
    # because Phase 1 paths don't evaluate ``success_bonus`` anymore.
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
    #: **v7**: collision penalty raised 5 → 10 to compensate for losing the
    #: episode-ending consequence (see ``EnvConfig.collision_terminates``).
    #: The per-step -10 cost accumulates while the policy stays in contact,
    #: so a sustained collision still hurts; meanwhile, the agent is no
    #: longer "rewarded" for crashing as a way to short-circuit a bad
    #: Stage-1 carry rollout.
    w_collision: float = 10.0
    w_workspace: float = 5.0
    w_action: float = 0.01
    w_jerk: float = 0.01
    #: **2026-05-29 (rev 3 — hover-lockin fix)** — Per-step "alive" cost
    #: applied directly to the episode return (bypasses ``mix_dense`` so
    #: it always bites). Without this, the agent could let the episode
    #: run to ``max_steps`` because dense+pen ≈ 0 was strictly ≥ pickup
    #: return. ``-0.05/step`` adds up to ``-25`` over a full 500-step
    #: timeout, plus an opportunity cost on every hover step — pushing
    #: the value function toward a short, decisive pickup trajectory.
    #: Goes through ``b.step_alive`` and is appended to ``b.total``
    #: AFTER the dense/sparse mix in ``_compute_reward``.
    #: **v11 (2026-05-31)**: 0.05 → 0.15 (3×). v9b dense floor still ran
    #: ≈ +0.44/step (approach_A 1.86 + lateral_th 1.91 + axial_dot_th 0.81)
    #: so -0.05/step alive cost lost the budget race. -0.15/step combined
    #: with v11 Stage 1 dense gating (lateral/axial masked) makes hover
    #: cost positive (worse than chance), so the value function is forced
    #: to value progress, not standing-still.
    w_step_alive: float = 0.15

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
    # ``prev_action`` (length = action.dim) + 3 mount tail scalars +
    # **3 v7 hub_guide_vector** scalars:
    #   action.dim = 6  → obs.dim = 73 + 6 + 3 + 3 = 85  (Phase 1)
    #   action.dim = 13 → obs.dim = 73 + 13 + 3 + 3 = 92 (Phase 2/3)
    # The 73 base entries (joints, EE/tire/hub/bolt poses, deltas) are
    # independent of action dim and shared across all phases. The
    # trailing ``hub_guide_vector = hub_pos - eeA_pos`` (3-d) gives the
    # policy a direct vector cue for Stage 1 carry — without it the
    # policy must derive the same direction from the joint/EE/hub
    # poses, which is information-theoretically equivalent but
    # empirically much harder for PPO to learn in dense form.
    dim: int = 92                   # §2.1 base + mounting + hub_guide


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
    #: **2026-05-29 (rev 4 — hover-budget cap)** — shortened from 500 → 200
    #: (≈ 10 s at 20 Hz) during pickup-only training.
    #: **2026-05-30 (v5 — full FSM)**: raised back to **400** (≈ 20 s).
    #: Now that pickup terminates at step ~30 only when ``terminate_on_pickup``
    #: is True, the v5 run disables that flag and needs budget for the
    #: full pickup (30) + carry/mount (≈ 150) + return (≈ 100) ≈ 280 step
    #: cycle plus margin. Hover risk is already contained by the new
    #: ``w_step_alive`` and the tight ``approach_decay`` (rev-3). Override
    #: via ``--max-steps`` on the CLI when running pickup-only sweeps.
    max_steps: int = 600            # ≈ 30 s at 20 Hz (v11c: 400 → 600 to fit full Pickup → Mount → Demount → Return cycle)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    render: bool = False            # GUI vs DIRECT

    # Bullet stability (recommended: numSubSteps 4–8, ERP ~0.15, CFM ~1e-5)
    physics_num_sub_steps: int = 6
    contact_erp: float = 0.15
    contact_cfm: float = 1e-5  # passed as PyBullet ``globalCFM`` at reset

    # Terminate when any contact reports excessive normal force (simulated breakage).
    # Set ≤ 0 to disable.
    contact_force_terminate_above: float = 2500.0
    #: **v7 (collision relaxation)** — when False, an in-bad-collision event
    #: applies the ``w_collision`` per-step penalty but does NOT terminate
    #: the episode. This is the cure for the v6 "collision-suicide" failure
    #: mode where the policy let the arm clip the rack on purpose to escape
    #: the -2.5/step Stage-1 negative dense baseline; with the new v7
    #: positive guide+PB carry kernels, ending the episode on contact is
    #: also a *premature* signal that prevents the policy from
    #: recovering from a glancing brush against the rack/cargo.
    #: **2026-06-01 (planner-residual rewrite)**: flipped back to ``True``.
    #: With the new Min-Jerk planner the nominal trajectory is *guaranteed*
    #: collision-free for the fixed Phase-1 scene, so any in-collision
    #: state is unambiguous policy misbehaviour. We end the episode
    #: immediately and apply ``R_fail = -50`` (auto-paid in ``step``).
    collision_terminates: bool = True

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
    #: **2026-06-01 (dual-arm relayout)** — UR10 base moved from
    #: ``(-0.80, 0, -0.30)`` to ``(-0.60, 0.15, -0.30)`` so its workspace
    #: actually covers the hub (0, 0.80, 0.22) / cooperative zone. At the
    #: old pose the EE could not reach x≈0 (hub ~1.21 m away, near the
    #: 1.3 m reach limit); +0.20 X / +0.15 Y brings the hub ~0.95 m out,
    #: inside the dexterous workspace. Panda stays at the world origin.
    #: NOTE: this translates every UR10 EE world pose by (+0.20, +0.15, 0)
    #: relative to legacy runs — HOME joint config is unchanged (joint
    #: space), but the achieved HOME EE position shifts accordingly.
    robot_A_base_pos: Tuple[float, float, float] = (-0.60, 0.15, -0.30)
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
    #:     Y = +0.40 / outer rail at Y = −0.40. The tire's tread rests
    #:     on the inner-top corners of both rails at (Y = ±0.35,
    #:     Z = 0.00), forming a stable V-cradle. The 6 o'clock line
    #:     (Y = 0) sits in the 70 cm Y-gap; static stability is held
    #:     by ``_pin_tire_to_world`` (mass = 0 freeze) at the cradle
    #:     equilibrium until Stage 0 → 1 fires.
    #:   * Z = rail_top + √(R² − 0.35²) = 0.00 + 0.39131 = **+0.3913**.
    #:     This is the geometric resting COM where the tread surface
    #:     just touches the rail inner-top corners. The 6 o'clock anchor
    #:     ends up at Z = COM − R = **−0.1337** (≈ 13.4 cm below the
    #:     rail top, inside the 70 cm Y-gap).
    #: UR10 base ↔ pickup grasp target ≈ 1.10 m planar + 0.167 m up =
    #: 1.128 m 3-D (within 1.30 m reach, 17.2 cm margin). Grasp anchor
    #: Z = −0.1337 effectively coincides with the HOME EE Z = −0.1367
    #: (3 mm apart) — the pickup is now a near-pure planar reach.
    tire_pickup_pos: Tuple[float, float, float] = (-1.90, 0.0, 0.3913)
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
    #:
    #: **2026-05-29 (rev 2) — pathological 1-step episode fix**: with the
    #: new Stage-0 start-pose curriculum, the easy spawn places the EE
    #: 0.20 m below the grasp anchor. The previous 0.35 m soft gate was
    #: *already wider* than that distance, so every easy-mode episode
    #: terminated as ``pickup_success`` on step 1 without the agent
    #: moving — PPO collected 24 k identical-reward samples per
    #: iteration with no gradient signal, and 90 % of wall time was
    #: spent in URDF-reload heavy ``reset()`` (10 FPS @ 12 envs).
    #: Tightened to **0.10 m** so easy-mode episodes require the agent
    #: to actively lift the EE ≥ 10 cm before the gate fires (~5 steps
    #: minimum at ``pos_scale = 0.02 m/step``). The smoothstep ramp
    #: now interpolates 0.10 → 0.08 m, a much tighter band; the curve
    #: still maintains a meaningful "easier early, stricter later"
    #: shape because the *start-pos* curriculum simultaneously moves
    #: the spawn from grasp − 0.20 m → HOME (54 cm away).
    approach_radius_tol: float = 0.08
    approach_tol_soft: float = 0.10
    #: First N (global PPO) timesteps where the gate is pinned to
    #: ``approach_tol_soft``. Past this point the smoothstep ramp begins.
    #: **2026-05-29**: synced with ``start_pos_curriculum_steps`` /
    #: ``start_pos_ramp_steps`` (100 k hold + 500 k smoothstep) so the
    #: easy-start UR10 EE position and the soft pickup gate fade out
    #: together — preventing the gate from collapsing to 0.08 m before
    #: the start position has lifted off the easy zone.
    approach_tol_curriculum_steps: int = 100_000
    #: Linear ramp length (global PPO timesteps) after the soft hold
    #: during which the gate is interpolated from ``approach_tol_soft``
    #: down to ``approach_radius_tol``. Matches the start-pos ramp.
    #: **2026-05-29 (rev 4)**: shortened 500k → 300k so the hard regime
    #: lands by t = 400k. With total = 2M this leaves 1.6M steps of
    #: pure hard-mode learning (vs 1.4M previously) — more budget for
    #: the policy to actually master the HOME-pose pickup.
    approach_tol_ramp_steps: int = 300_000

    # ------------------------------------------------------------------
    # Stage 0 starting-pose curriculum + early termination.
    # ------------------------------------------------------------------
    #: Master switch for the UR10 EE starting-pose curriculum. When True
    #: (default), every ``reset()`` blends the UR10 joint pose between
    #:   * easy: grasp_anchor + (0, 0, −``start_pos_easy_lift``)
    #:   * hard: the analytical ``HOME_POSE`` FK
    #: using a smoothstep coefficient driven by ``StartPosCurriculumCallback``
    #: (``src/train.py``). When False, every reset uses HOME as today.
    #: Eval / render / no-callback paths leave ``_start_pos_alpha = 1.0``
    #: (full hard) so they always start at the production HOME pose.
    start_pos_curriculum_enable: bool = True
    #: +Z distance (m) below the tire 6 o'clock grasp anchor at which the
    #: easy start pose is computed. **v9**: 0.20 → 0.10 m — shorter lift
    #: so easy resets fire pickup more often and spend more steps in Stage 1.
    start_pos_easy_lift: float = 0.10
    #: First N global PPO timesteps where the start pose stays at full
    #: easy (alpha = 0). Mirrors ``approach_tol_curriculum_steps``.
    start_pos_curriculum_steps: int = 100_000
    #: Smoothstep ramp length (global PPO timesteps) after the easy hold.
    #: alpha advances from 0 → 1 as ``smoothstep((t − hold) / ramp)``.
    #: **2026-05-29 (rev 4)**: shortened 500k → 300k so the policy
    #: locks onto HOME-pose pickups by t = 400k, leaving 1.6M steps
    #: of converged hard-mode learning.
    start_pos_ramp_steps: int = 300_000
    #: **v8 (50% mix)** — start-pos curriculum *operating mode*.
    #:   * ``"lerp"`` (legacy): the callback smoothsteps a single alpha
    #:     from 0 (easy) → 1 (HOME) over ``curriculum_steps + ramp_steps``.
    #:     Every reset uses that same alpha. This is what v4 was trained
    #:     with, but it had a known weak spot: during the alpha = 0.4–0.7
    #:     intermediate band, *no* reset produces a reliable pickup
    #:     trigger and the policy can lose the picking habit
    #:     (observed in v3 as long Stage-1 dropouts and reward hacking).
    #:   * ``"mix"`` (v8 default): every reset rolls a Bernoulli(p) and
    #:     either spawns full-easy (alpha = 0) or full-hard (alpha = 1).
    #:     The "easy" half guarantees the policy keeps tasting R_pickup
    #:     and the carry/mount dense reward forever, while the "hard"
    #:     half forces generalisation to the production HOME start.
    #:     Probability ``p`` = ``start_pos_easy_prob``.
    start_pos_curriculum_mode: str = "mix"
    #: Bernoulli probability of choosing the *easy* spawn each reset
    #: under ``start_pos_curriculum_mode = "mix"``. v8 used 0.5; v9 was
    #: 0.75; **2026-06-01 planner-residual rewrite**: forced to **1.0**
    #: so every reset routes through the new attached-hot-start path
    #: (``attached_spawn_when_easy=True``) — the policy spawns already
    #: holding the tire at the cradle pose, ``task_stage == 1``, and
    #: only has to solve the carry/mount problem. Combined with
    #: ``terminate_on='mount'`` this yields the fastest possible
    #: Mount-only training loop. Override on the CLI to mix HOME
    #: starts back in once mount converges.
    start_pos_easy_prob: float = 1.0

    #: Distance gate (m) on the Stage 0 dense ``approach_A`` term: when
    #: ``d_approach > approach_A_gate`` the dense reward is zeroed so the
    #: policy can't farm it by sitting just outside the grasp anchor. Set
    #: large (>5 m) to disable.
    approach_A_gate: float = 0.20
    #: **v11c2 (2026-05-31)** — master switch for the four "safety"
    #: termination gates (``vertical_violation``, ``collision``,
    #: ``workspace``, ``contact_force``). The ``ReverseCurriculumCallback``
    #: flips this to ``False`` while Phase A is active and restores
    #: ``True`` for Phase B/C, because Phase A's hot-start produces a
    #: chaotic first-step physics burst (tire teleported into hub +
    #: untrained policy action) that would otherwise terminate the
    #: episode in 1–25 steps before PPO can collect any post-mount
    #: training data. Eval / render paths keep the default ``True``.
    safety_terminations_enabled: bool = True
    #: When False, the trailing 3-d ``hub_guide_vector`` is omitted from
    #: the observation (legacy checkpoints trained before v7 used
    #: obs.dim = 73 + action.dim + 3 without this block).
    include_hub_guide_obs: bool = True

    #: When True, the episode terminates with **success = True** the
    #: instant the Stage 0 → 1 pickup gate fires (R_pickup paid). Legacy
    #: 3-stage flag — superseded by ``terminate_on`` in v6 (kept for
    #: backwards compatibility). If ``terminate_on != "never"``, this
    #: flag is ignored and ``terminate_on`` wins.
    terminate_on_pickup: bool = False
    #: **2026-05-30 (v6 — curriculum brake-lock)** — choose at which FSM
    #: event the episode short-circuits to ``success = True``. Lets the
    #: curriculum activate Stage 0 → 1 → 2 → 3 incrementally without
    #: editing env code. Allowed values:
    #:   * "never"   — full 4-stage cycle; success at Stage 3 landing.
    #:   * "pickup"  — success on Stage 0 → 1 (legacy v4/v5 pickup-only).
    #:   * "mount"   — success on Stage 1 → 2 (v6 first run: master mount).
    #:   * "demount" — success on Stage 2 → 3 (Stage 2 specialisation).
    #: Stage 2/3 code stays active in the env regardless; the flag only
    #: gates the early-termination point so a single ``--terminate-on
    #: never`` flip turns on the full cycle.
    #: **2026-06-01 (planner-residual rewrite)**: default changed to
    #: ``"mount"`` so a fresh-from-scratch run short-circuits the moment
    #: Stage 1 → 2 fires (R_mount paid + ``is_success = True``). Stage 2/3
    #: replanning code stays wired but is not exercised under this default
    #: — flip to ``"never"`` once mount converges.
    terminate_on: str = "mount"
    #: Stage 2 demount target — axial distance (m) the tire centre must
    #: travel *away* from the hub centre to fire the Stage 2 → 3 trigger.
    #: 30 cm is roughly 1.5× tire thickness (0.30 m / 2 + clearance), so
    #: the tire fully clears the hub flange + bolts before the policy is
    #: rewarded for "demounted".
    demount_axial_distance: float = 0.30
    #: Stage 2 demount stall — number of env steps the policy must spend
    #: in Stage 2 after the Stage 1 → 2 transition before the demount
    #: gate becomes eligible to fire. Simulates the gap between Panda
    #: completing the bolt-down and UR10 starting its safe-axial-pull.
    #: 20 step at 20 Hz = 1.0 s. During the stall, UR10 actions still
    #: physically execute (so the policy is free to start drifting away),
    #: but the gate stays closed — preventing "drive the tire away
    #: instantly" exploits that bypass the safety hold.
    demount_stall_steps: int = 20
    #: Stage 3 cradle return — Euclidean radius (m) the tire centre must
    #: land within of the cradle pickup pose for the Stage 3 → done
    #: gate to consider it a successful landing.
    rack_return_radius_tol: float = 0.05
    #: Stage 1 → 2 trigger: ‖tire − hub‖ < ``mount_radius_tol`` AND tire axis
    #: aligned with hub axis (≤ ``RewardConfig.delta_A`` rad).
    #: **Final hard gate** that the v6 mount curriculum asymptotes to.
    #: NOTE (v13b): this 0.04 m target was *geometrically unreachable* until
    #: the truck-station brake proxies were resized — the brake rotor (r 0.30)
    #: exceeded the tire bore (r 0.282), flooring the tire COM at d_A≈0.13 and
    #: causing the curriculum to stall at frac≈0.40 (tol≈0.21 m) across three
    #: reward configs. With rotor→0.22 and the caliper pushed aft, the tire
    #: now seats concentrically (verified: d_A=0 clears by ~0.02 m, and a
    #: 0.04 m offset is collision-free in every direction), so this gate and
    #: ``tire_mount_pos`` (hub centre) are valid as-is — no reward/target
    #: workaround needed. See scripts/generate_truck_wheel_station_urdf.py.
    mount_radius_tol: float = 0.04
    #: v6 mount-gate curriculum — **start radius** (m) used before the
    #: smoothstep ramp begins. 0.30 m gives the policy a generous
    #: "anywhere near the hub" entry signal so R_mount fires reliably
    #: even when carry trajectory is loose. The radius then sweeps
    #: 0.30 → ``mount_radius_tol`` over ``mount_tol_ramp_steps``.
    mount_radius_tol_soft: float = 0.30
    #: v6 mount-gate curriculum — **start axis tolerance** (rad). Default
    #: 35° lets the tire-bore vs hub-axis angle be very forgiving while
    #: the policy learns the carry trajectory; the ramp tightens it to
    #: ``RewardConfig.delta_A`` (5°) at the end.
    mount_angle_tol_soft_rad: float = np.deg2rad(35.0)
    #: v6 mount curriculum hold (global PPO steps) — gate stays at the
    #: soft pair before the smoothstep ramp engages.
    mount_tol_curriculum_steps: int = 200_000
    #: v6 mount curriculum ramp (global PPO steps) — smoothstep from
    #: ``(mount_radius_tol_soft, mount_angle_tol_soft_rad)`` down to
    #: ``(mount_radius_tol, reward.delta_A)``.
    mount_tol_ramp_steps: int = 600_000
    #: Mount-gate curriculum mode. ``"adaptive"`` (default) advances the
    #: soft→hard difficulty only while the recent success rate stays high
    #: and **rolls difficulty back** when it collapses — closing the loop
    #: that the open-loop ``"schedule"`` mode left open (a time-only ramp
    #: tightens the gate regardless of whether the policy keeps up, so a
    #: collapse never recovers). ``"schedule"`` keeps the legacy
    #: ``num_timesteps`` smoothstep for reproducing old runs.
    mount_curriculum_mode: str = "adaptive"
    #: Adaptive mode — advance difficulty (``frac += mount_adapt_step_up``)
    #: when the recent success rate is at or above this threshold.
    #: v12: raised 0.80 → 0.85. The mount success curve is a *cliff*
    #: (1.0 → 0.0 in a single tol step), so advancing off a barely-passing
    #: window drove the oscillation seen in the 20260602 run. Require a
    #: stronger cushion before tightening.
    mount_adapt_advance_sr: float = 0.85
    #: Adaptive mode — roll difficulty back (``frac -= mount_adapt_step_down``)
    #: when the recent success rate falls to or below this threshold. The
    #: gap to ``mount_adapt_advance_sr`` is the hysteresis band that keeps
    #: the gate from oscillating around a single success level. v12: lowered
    #: 0.55 → 0.50 to widen that band (0.85/0.50) and avoid retreating on a
    #: noisy dip.
    mount_adapt_rollback_sr: float = 0.50
    #: Adaptive mode — difficulty increment per adjustment when advancing.
    #: v12 first tried 0.025 but, combined with dwell=6, that quartered the
    #: advance rate (~175k steps / 0.025 frac) so the gate never reached the
    #: tol≈0.19 m cliff region within a 2M budget — the run stayed in the
    #: trivial soft-tol zone. Restored to 0.05; the anti-oscillation work is
    #: carried by dwell=6 / advance_sr=0.85 / min_episodes=60 instead.
    mount_adapt_step_up: float = 0.05
    #: Adaptive mode — difficulty decrement per adjustment when rolling
    #: back. Larger than ``step_up`` so a collapse is unwound faster than
    #: it was built up (asymmetric: retreat quickly, re-advance cautiously).
    mount_adapt_step_down: float = 0.10
    #: Adaptive mode — minimum completed episodes in the success window
    #: before any advance/rollback decision is trusted (avoids reacting to
    #: a handful of noisy early episodes). v12: 40 → 60 for a more
    #: trustworthy estimate near the cliff.
    mount_adapt_min_episodes: int = 60
    #: Adaptive mode — number of rollouts to hold ``frac`` fixed after any
    #: change, letting the policy (and the success estimate) settle before
    #: the next decision. v12: 3 → 6 — the dominant anti-oscillation knob;
    #: lets the policy consolidate each new tol before being pushed again.
    mount_adapt_dwell_rollouts: int = 6
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

    #: **2026-06-01 (UR10-feasible tire)** — when True, ``Scene._spawn_tire``
    #: loads ``tire_urdf`` (a smaller hollow-ring tyre generated by
    #: ``scripts/generate_tire_urdf.py``) instead of the full-size procedural
    #: ``models.create_tire_wheel_multibody``. The procedural tyre is left
    #: untouched and remains the default. When enabling this, also set
    #: ``tire_outer_radius`` / ``tire_inner_radius`` / ``tire_thickness`` /
    #: ``tire_mass`` to match the generated URDF so the env's grasp-anchor /
    #: mount geometry stays consistent (the generator prints the exact block).
    #: Bore axis = link local +Z, same convention as the procedural tyre.
    use_tire_urdf: bool = False
    tire_urdf: str = str(URDF_DIR / "tire" / "tire_ur10.urdf")

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
    #: **70 cm Y-gap** centred on the bore axis (Y = 0). The rails form
    #: a V-cradle: the tire (radius 0.525 m) rests with its tread
    #: surface in contact with the inner-top corners of both rails
    #: (Y = ±0.35, Z = rail_top). **2026-05-29 (rev 5)**: Y-gap widened
    #: from 50 cm to **70 cm** so the tire sinks lower into the cradle
    #: (COM Z 0.461 → 0.391, anchor Z −0.063 → −0.134). The anchor now
    #: sits within 3 mm of the HOME EE Z = −0.137, reducing pickup to
    #: an almost pure planar reach.
    #:
    #: NOTE on the bore=+X spawn (``tire_spawn_rpy = (0, π/2, 0)``):
    #: with the tire bore facing robot A in +X, the 6 o'clock outer
    #: tread line at Y = 0 sits **below** the rail top plane (Z =
    #: -0.134 vs rail top Z = 0.00) — the cradle geometry pulls the
    #: tire's bottom into the gap. The gripper still approaches from
    #: above through the gap (Y = 0, gap width 0.70 m) and dives to
    #: Z ≈ -0.13 to reach the 6 o'clock anchor; no rail clipping since
    #: the gripper centreline is 35 cm from either rail face.
    #:
    #: Static stability is maintained by ``_pin_tire_to_world``
    #: (mass = 0 freeze) which holds the tire at the cradle equilibrium
    #: pose until Stage 0 → 1 fires.
    #:
    #: Geometry (Robot B-centric, ``tire_outer_radius = 0.525``,
    #: ``tire_thickness = 0.30``, rails 30 cm long × 10 cm thick ×
    #: **60 cm tall**, 2× the original 30 cm height):
    #:   * Inner rail Y range = [+0.35, +0.45] (centre +0.40 ± 0.05).
    #:   * Outer rail Y range = [-0.45, -0.35] (centre -0.40 ± 0.05).
    #:   * Gap in Y = [-0.35, +0.35], width **0.70 m** ✓
    #:   * Both rails X range = [-2.05, -1.75] (centre -1.90 ± 0.15) —
    #:     matches the tire's bore-axis extent X ∈ [-2.05, -1.75]
    #:     exactly, so the +X / −X ends of the rails terminate flush
    #:     with the tread's front and back faces.
    #:   * Rail Z range = [-0.60, 0.00]; centre -0.30, half-ext 0.30
    #:     (= floor + 0.60 m height). Rail top Z = **0.00** (world origin).
    #:   * Tire COM Z = rail_top + √(R² − 0.35²)
    #:                = 0.00 + √(0.275625 − 0.1225)
    #:                = 0.00 + 0.39131
    #:                ≈ **+0.3913** ✓ (tire rests on rail corners)
    #:   * 6 o'clock anchor (Y=0): Z = COM_Z − R = -0.1337 — the gripper
    #:     dives ≈ 13.4 cm below rail top inside the 70 cm Y-gap.
    spawn_tire_rack: bool = True
    #: Truck-side rail (inner = closer to the truck on the +Y side).
    tire_rack_inner_center: Tuple[float, float, float] = (-1.90, 0.40, -0.30)
    #: Far-side rail (outer = on the −Y side, away from the truck).
    tire_rack_outer_center: Tuple[float, float, float] = (-1.90, -0.40, -0.30)
    #: 30 cm long (X) × 10 cm thick (Y) × **60 cm tall (Z)** rails —
    #: Z half-extent 0.30 (2× the original 0.15) so rail top sits at
    #: floor + 0.60 m = 0.00 m world.
    tire_rack_half_extents: Tuple[float, float, float] = (0.15, 0.05, 0.30)
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
    # **2026-06-01 — Minimum-Jerk planner + PPO residual control.**
    # ------------------------------------------------------------------
    # Replaces the raw 6-d Δ-EE-pose action with a hybrid scheme:
    # every reset (and every FSM stage transition) builds a fresh
    # nominal trajectory from the current EE pose to the stage's end
    # pose using a 5th-order min-jerk profile for position + SLERP for
    # orientation. The PPO action is then a small per-step *residual*
    # offset added on top of that nominal pose. The policy only has to
    # learn obstacle-avoidance / fine-alignment corrections, not the
    # entire motion synthesis problem.
    #: Master switch. When False, ``_apply_action`` falls back to the
    #: legacy raw-delta path (``robot_A.apply_delta_ee``) so the same
    #: env class can still be used by eval scripts that load v11c-era
    #: checkpoints. Default True for all fresh training runs.
    use_planner_residual: bool = True
    #: Per-step residual position offset scale (metres) applied to
    #: ``action[0:3]`` ∈ [-1, 1] before adding to the nominal pose.
    #: 0.15 m matches the spec — generous enough for genuine avoidance
    #: but small enough that the planner remains the dominant signal.
    planner_pos_offset_scale: float = 0.15
    #: Per-step residual rotation offset scale (radians, axis-angle).
    #: Only consumed when ``planner_enable_rot_offset = True``.
    planner_rot_offset_scale: float = 0.15
    #: When False (default), ``action[3:6]`` is ignored at the env layer
    #: and the nominal SLERP-interpolated quaternion drives orientation
    #: alone. This is the "Q1 — pos_only" choice: maximally stable for
    #: scratch training because the wrist DOFs are fully managed by the
    #: planner. Flip to True (and optionally retune ``rot_offset_scale``)
    #: once the position residual has converged.
    planner_enable_rot_offset: bool = False
    #: Nominal trajectory length (control steps). 100 step ≈ 5.0 s at
    #: 20 Hz control. End-pose is held constant once the index exceeds
    #: this length so the policy can still operate on a "fix to last
    #: pose" basis if it has not yet triggered the stage gate.
    planner_traj_steps: int = 100
    #: **2026-06-01 (lift-first carry fix)** — vertical clearance (m) the
    #: Stage-1 nominal trajectory lifts the EE straight up (keeping the
    #: grasp orientation fixed) *before* translating + rotating toward the
    #: mount pose. Without this waypoint the single straight cradle→mount
    #: chord drags the 1.05 m tire (grasped at its 6-o'clock point) into
    #: the outer rack rail within ~15 steps, tripping the contact-force
    #: kill switch even for a zero-residual rollout (see
    #: ``scripts/smoke_planner_residual.py``). The lift clears the tread
    #: above the rail tops (rail top Z = 0.0, grasp anchor Z ≈ −0.134) so
    #: the subsequent carry segment never sweeps the rack. Set ≤ 0 to
    #: disable (single-segment chord, legacy behaviour).
    planner_stage1_lift_height: float = 0.35
    #: When ``True`` AND the easy-spawn branch is rolled in ``reset``
    #: (i.e. Bernoulli(``start_pos_easy_prob``) returned True under
    #: ``start_pos_curriculum_mode == "mix"``), the env performs the
    #: **attached hot-start**: tire pinned to the cradle pose, UR10 EE
    #: teleported to the 6-o'clock grasp anchor, grasp constraint
    #: attached, ``task_stage = 1``, ``_pickup_bonus_paid = True``.
    #: The policy then immediately starts the mount carry — Stage 0
    #: (approach / pickup) is skipped entirely. Combined with
    #: ``terminate_on='mount'`` this gives a clean Mount-only training
    #: setup. Set to False to retain the legacy "EE teleported below
    #: grasp anchor but tire still pinned to cradle" easy spawn.
    attached_spawn_when_easy: bool = True
    #: **2026-06-01 (hub-aligned carry fix)** — when the attached hot-start
    #: fires, re-pose the tire so its bore is **already aligned with the
    #: hub axis** (``_mount_tire_quat``) at the cradle, instead of the
    #: ``tire_spawn_rpy`` (+X) pickup pose. Diagnosis (see
    #: ``scripts/smoke_planner_residual.py`` history) showed the UR10, at
    #: its far-reach cradle pose, **cannot track** the 90° bore
    #: reorientation (tool-roll) needed to carry a +X-spawned tire onto the
    #: −Y hub — the IK stalls and the mount gate never fires. Pre-aligning
    #: the bore at spawn turns the reorientation into a free teleport, so
    #: the Stage-1 carry reduces to a lift + pure translation, which the arm
    #: tracks stably (mount reached with zero residual). Only consulted on
    #: the mount-only attached-hot-start path; the full-cycle Stage-0 pickup
    #: still spawns the tire at the +X ``tire_spawn_rpy`` pose. Set False to
    #: restore the legacy +X attached spawn (and the untrackable in-flight
    #: reorientation).
    #:
    #: **Default False (2026-06-01 diagnosis)**: pre-aligning the bore to
    #: −Y at the cradle puts the tread ring in the X-Z plane, which
    #: *obstructs the UR10's +X-side approach* to the 6-o'clock grasp
    #: anchor (the arm jams into the ring). Kept as an opt-in for layouts
    #: where the robot does not approach across the tread plane.
    attached_spawn_hub_aligned: bool = False

    # ------------------------------------------------------------------
    # **2026-06-01 — Dual-arm cooperative carry (mount-only).**
    # ------------------------------------------------------------------
    # Single-arm carry of the truck tire was shown (GUI + headless sweeps)
    # to be infeasible: the UR10 is always at far reach and cannot track
    # the 90° bore reorientation. The cooperative scheme instead spawns the
    # (UR10-feasible) tire in an **open zone both arms reach** (in front of
    # the hub, away from the cargo), grasps it with BOTH arms (UR10 rigid +
    # Panda point-to-point support), and carries it onto the hub with a
    # short, collision-free, reorientation-free translation. The PPO policy
    # still only drives the UR10 residual (action stays 6-d / obs 85-d);
    # the Panda is planner-driven (not policy-controlled) and simply holds
    # the far side of the tire stable. Verified collision-free with the
    # zero-residual nominal reaching d_hub ≈ 0.19 m (inside the soft mount
    # gate). Requires ``use_tire_urdf=True`` + the small-tire dims and the
    # relayout UR10 base (see those fields).
    dual_arm_coop: bool = False
    #: Tire COM spawn pose for the cooperative hot-start (world). In the
    #: open zone in front of the hub that BOTH arms reach with IK residual
    #: ≈ 0 (UR10 ~0.62 m, Panda ~0.55 m to the rim grasp points).
    coop_spawn_pos: Tuple[float, float, float] = (-0.20, 0.50, 0.20)
    #: Unit direction (world) from the tire COM to each arm's grasp rim
    #: point (bore along −Y ⇒ tread ring in the X-Z plane; point = COM +
    #: R·dir). **UR10 grasps the bottom (6-o'clock, −Z) and Panda the top
    #: (12-o'clock, +Z)** so the two arms stay vertically separated — any
    #: same-side pairing (e.g. UR −X + Panda −Z) makes the forearms collide
    #: within a couple of steps. This pairing carried the tire collision-
    #: free to d_hub ≈ 0.09 m with a zero residual.
    coop_ur_grasp_dir: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    coop_panda_grasp_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Peak of the vertical lift arc (m) added to both arms' nominal carry
    #: so the tire rises clear of any low obstacle mid-transit.
    coop_lift_arc: float = 0.12

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
    legacy_action_dim = overrides.pop("legacy_action_dim", None)
    legacy_obs_dim = overrides.pop("legacy_obs_dim", None)
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
        if legacy_action_dim is not None:
            cfg.action.dim = int(legacy_action_dim)
        else:
            cfg.action.dim = 6
        tail = 73 + int(cfg.action.dim) + 3
        if bool(getattr(cfg, "include_hub_guide_obs", True)):
            cfg.obs.dim = tail + 3
        else:
            cfg.obs.dim = tail
    else:
        cfg.action.dim = 13
        tail = 73 + 13 + 3
        if bool(getattr(cfg, "include_hub_guide_obs", True)):
            cfg.obs.dim = tail + 3
        else:
            cfg.obs.dim = tail
    if legacy_obs_dim is not None:
        cfg.obs.dim = int(legacy_obs_dim)
    return cfg
