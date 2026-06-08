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
    #: **2026-06-03 (phase1_grad_v6 / R2)** — v5 raised this to 0.75 which
    #: encouraged hover-without-landing (reward hacking). Restored to v4.
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

    # ------------------------------------------------------------------
    # Robot B sequential nut-fastening task (``cfg.nut_fastening_task``).
    # These weights are phase-independent (the nut reward branch does not
    # consult ``make_reward_config``'s phase gating), so they stay active
    # whenever the nut task is enabled. Mirrors the Stage-0 approach
    # design: bounded positive ``exp`` kernels (so surviving a step is
    # never punished) + potential-based shaping + a sparse per-bolt bonus.
    # ------------------------------------------------------------------
    #: Positive reach kernel toward the current target bolt:
    #: ``w_nut_reach * exp(-d_B / nut_reach_decay)`` where d_B is the
    #: tool_tip→bolt distance. Bounded in [0, w_nut_reach].
    #: 2026-06-08 REBALANCE (3.0 → 0.5). The standing exp kernels (reach/
    #: lateral/align/axial + clearance) summed to a dense "you're in a good
    #: spot" reward of ~1.8/step that, over a 600-step episode (~1080), exceeded
    #: the value of fastening all 10 bolts (~945). The policy correctly learned
    #: to PARK near a bolt and farm dense reward instead of fastening (eval
    #: success = 0 across the entire 3 M-step run, ep_len pinned at 600). Fix:
    #: shrink the farmable standing kernels and lean on the (farm-proof)
    #: potential-based ``pb_nut`` term for the approach gradient, while boosting
    #: the sparse fasten bonuses so task completion dominates.
    w_nut_reach: float = 0.5
    #: Lateral (off-axis) weight inside the APPROACH reach distance
    #: ``d_stage = hypot(axial_err, w * lateral)``. Retained for the d_B gate
    #: metric/logging; the APPROACH reward now uses an explicit two-stage
    #: split (see ``nut_coax_gate``) instead of this combined distance.
    nut_reach_lateral_w: float = 2.5
    #: 2026-06-07 — two-stage APPROACH coaxial gate length (m). The axial
    #: approach terms (``nut_reach``/``nut_axial_term`` and the axial PB leg)
    #: are multiplied by ``exp(-lateral / nut_coax_gate)`` so they only switch
    #: on once the socket is roughly on the bolt axis. This stops the policy
    #: from cutting a chord across the axis to shrink the staging distance
    #: (the 15 cm off-axis stall at bolt 3). ~5 cm ⇒ axial pull is ~0 at the
    #: 10 cm bolt pitch and ~0.6 once within ~2.5 cm of the axis.
    nut_coax_gate: float = 0.05
    #: 2026-06-07 — widened 0.15 → 0.50. At the 1.7 m HOME→bolt standoff
    #: ``exp(-1.7/0.15) ≈ 1e-5`` gave the policy zero approach gradient
    #: (the first B run never reached a bolt). 0.50 keeps a usable pull
    #: from ~1 m out while still rewarding the final cm of seating.
    nut_reach_decay: float = 0.50
    #: Positive alignment kernel: ``w_nut_align * exp(-θ_B / decay)``
    #: where θ_B is the tool↔bolt axis angle (folded to [0, π/2]).
    #: 2026-06-07 — decay tightened 30° → 18° so the gradient stays firm down
    #: to the arrive-angle curriculum's end gate (12°). At 30° the kernel was
    #: nearly flat across 12–35° (exp(-12/30)=0.67 vs exp(-35/30)=0.31), so
    #: tightening the trigger gate added no matching pull to align; 18° gives
    #: a clear pull right where the gate closes (exp(-12/18)=0.51,
    #: exp(-5/18)=0.76) while still rewarding coarse alignment far out.
    w_nut_align: float = 0.4  # 2026-06-08 REBALANCE (1.5 → 0.4): anti-farm.
    nut_align_decay_rad: float = np.deg2rad(18.0)
    #: Potential-based shaping on Δd_B (pays positively each step the
    #: tool closes on the target bolt). Reset across bolt advances.
    #: 2026-06-08 REBALANCE (8.0 → 14.0): PB shaping telescopes over a
    #: trajectory (Σ = w·(d_start − d_end)) so it CANNOT be farmed by standing
    #: still — it only pays for net progress toward the bolt and is negative
    #: when retreating. Promoted to the primary approach driver now that the
    #: standing exp kernels are shrunk, so the policy still gets a strong, dense
    #: "get closer" gradient without a parkable plateau.
    w_pb_nut: float = 14.0
    #: Sparse bonus paid once per bolt successfully fastened.
    #: 2026-06-08 REBALANCE (50 → 120): fastening must dominate dense farming.
    R_fasten: float = 120.0
    #: Terminal success bonus paid once all ``n_bolts`` are fastened.
    #: 2026-06-08 REBALANCE (300 → 500).
    R_all_fastened: float = 500.0
    # --- insertion-retract shaping (nut_fastening_task) -------------------
    #: Positive lateral kernel ``w_nut_lateral * exp(-lat / decay)`` where
    #: ``lat`` is the tool_tip distance off the bolt axis. Drives the
    #: "enter exactly along the bolt (Y) axis" requirement so the socket
    #: overlaps the stud instead of brushing it sideways.
    #: 2026-06-07 — decay widened 0.03 → 0.08 and weight raised 2.0 → 4.0.
    #: At 0.03 the kernel was dead past ~6 cm off-axis (``exp(-0.12/0.03)≈0.02``),
    #: so when the socket parked ~12 cm beside the next bolt's axis (the
    #: bolt-to-bolt hand-off around the lug circle) there was no gradient
    #: pulling it coaxial — the policy stalled laterally (observed: stuck at
    #: bolt 4, lateral 12 cm). 0.08 keeps a usable coaxial pull out to ~15 cm
    #: (``exp(-0.12/0.08)=0.22``) and the higher weight makes "get on the
    #: axis" compete with the raw reach term.
    w_nut_lateral: float = 1.5  # 2026-06-08 REBALANCE (4.0 → 1.5): anti-farm,
    #: but kept the largest of the standing kernels because getting ONTO the
    #: bolt axis (small lateral) is the precision bottleneck for the arrive gate.
    nut_lateral_decay: float = 0.08
    # --- Robot-B ↔ Robot-A clearance shaping (avoid A while fastening) -----
    #: Instead of forcing an "arm-up" IK branch, teach the policy to keep
    #: Robot B's arm clear of Robot A on its own: a positive, *saturating*
    #: clearance bonus ``w_nut_ba_clear * clip((d_BA - floor)/(cap - floor),0,1)``
    #: where ``d_BA`` is the minimum distance between B's and A's **joint-center
    #: points** (skeleton separation — smoother & mesh-independent vs surface
    #: closest-points). Because the centers sit inside the links, ``d_BA`` floors
    #: near the sum of link radii at hard contact, so the bonus is normalised
    #: between ``nut_ba_clear_floor`` (≈ contact, bonus→0) and ``nut_ba_clear_cap``
    #: (comfortably clear, bonus→1). Measured staging clearances: ~0.43 m at the
    #: tightest bolts (4,5) up to ~0.61 m at the open ones. Saturating at the cap
    #: means once B is clear there is no incentive to flee further (it must still
    #: approach the hub to fasten); below the cap a smooth gradient pulls B onto
    #: higher-clearance approach corridors.
    w_nut_ba_clear: float = 0.0  # 2026-06-08 (v9) REMOVED (0.4 → 0.0): the
    #: joint-center clearance bonus is mesh-blind (floors at ~0.3 m even at hard
    #: contact), so it gave almost no avoidance gradient in the tight ~6 cm
    #: corridor at the bottom bolts while still constituting a parkable income.
    #: Experiment: drop this shaping entirely, keep only the real-contact hard
    #: penalty (w_nut_collision=40), and raise exploration (ent_coef) so the
    #: policy finds collision-free joint angles on its own.
    #: (history) 2026-06-08 REBALANCE (2.0 → 0.4): this was the single worst
    #: farm — a saturating standing bonus paid every step B sat clear of A near
    #: a bolt. The engagement gate already localised it to the work point.
    nut_ba_clear_floor: float = 0.30
    nut_ba_clear_cap: float = 0.60
    #: 2026-06-08 — engagement gate length (m) for the clearance bonus. The
    #: bonus is multiplied by ``exp(-d_engage / scale)`` where ``d_engage`` is
    #: the Euclidean tool→target-staging distance, so the "keep clear of A"
    #: reward is only earned while B is actually working the bolt — not by
    #: fleeing the hub. ~0.35 m: full at staging, ≈0.06 once ~1 m away (the
    #: camping distance that previously farmed the saturated bonus and stalled
    #: n_fastened at 1). Without this gate the policy fastened bolt 0 (free via
    #: hot-start) then camped far from A instead of approaching the next bolt.
    nut_ba_clear_engage_scale: float = 0.35
    #: Dedicated (stronger) per-step penalty for a Robot-B↔Robot-A bad
    #: collision in the nut task. The shared ``w_collision`` (10) was too weak
    #: to dominate the dense reach reward, so the policy tolerated grazing A.
    #: Applied in both APPROACH and the forced macro.
    w_nut_collision: float = 40.0
    #: 2026-06-08 (v10) — soft ANSWER-PATH adherence bonus. All bolt staging
    #: points share a fixed world Y (= staging plane), and the answer route
    #: (HOME → hub center → bolts → HOME) moves within that constant-Y plane
    #: (pure XZ transit). Reward B for keeping its tool Y near that plane during
    #: APPROACH: ``w_nut_path * exp(-|ee_y - plane_y| / nut_path_decay)``. This
    #: gently pulls B onto the in-plane hub-and-spoke route WITHOUT hard-forcing
    #: it (the user wants "follow the path approximately"), so B still explores
    #: joint angles freely while staying near the collision-free corridor.
    #: Only applied in APPROACH (the macro intentionally moves along ±Y to
    #: insert/retract, so an in-plane bonus there would fight the tighten).
    w_nut_path: float = 0.6
    nut_path_decay: float = 0.10  # m; full at plane, ~0.37 at 10 cm off-plane.
    #: 2026-06-08 (v10) — minimal-joint-change penalty: ``-w_nut_joint_vel *
    #: ||dq_B||`` over the arm joints during APPROACH. The user wants the path
    #: followed with the least joint motion; the existing action/jerk penalties
    #: act in EE-delta space, this adds a direct joint-space cost. Small so it
    #: shapes smoothness without smothering the approach drive.
    w_nut_joint_vel: float = 0.02
    #: Positive axial-progress kernel during INSERT: rewards driving the
    #: tool_tip to the bolt base (hub face) along the axis.
    w_nut_axial: float = 0.5  # 2026-06-08 REBALANCE (2.0 → 0.5): anti-farm.
    nut_axial_decay: float = 0.05
    #: Potential-based shaping on the RETRACT leg: pays positively per step
    #: the tool_tip backs out along +axis (−Y) toward clearing the stud.
    w_nut_retract: float = 6.0
    #: Sparse bonus paid when the socket arrives coaxially + aligned at a
    #: bolt's staging point (triggers the scripted insert→hold→retract
    #: macro). This is the main signal the APPROACH policy chases.
    #: 2026-06-08 REBALANCE (25 → 40).
    R_arrive: float = 40.0
    #: Sparse bonus paid when a bolt's INSERT+HOLD dwell completes (socket
    #: fully seated over the stud), before the retract leg.
    #: 2026-06-08 REBALANCE (30 → 60).
    R_insert: float = 60.0

    # Cooperation term: r_coop = w_c * exp(-alpha*d_A) * exp(-beta*d_B)
    w_c: float = 2.0
    alpha: float = 10.0
    beta: float = 10.0

    # UR10 cooperative sync — penalize large joint velocity on A so Panda can refine.
    #: **v9**: 0.08 → 0.04; **v9b**: 0.04 → 0.02 — v9 still logged
    #: sync_joint_A ≈ -2.5~-4/step drowning carry dense despite w_guide=8.
    #: **2026-06-02 (phase1_grad_v3)**: 0.02 → **0.005**. With the
    #: planner-residual control scheme, the planner sets large IK
    #: targets every step and the motor servos at correspondingly high
    #: joint velocities; v2 logged ``sync_joint_A ≈ -11/step``, single-
    #: handedly making ``dense_total_pre_mix`` negative (-10.9) despite
    #: ``guide_A + pb_carry`` being positive. With the residual policy
    #: only adding fine corrections on top, the cooperative-sync
    #: penalty no longer plays the regulariser role it did in the
    #: legacy raw-delta scheme — slashing the weight 4× restores a
    #: positive dense kernel without sacrificing arm jitter control
    #: (the planner already produces smooth nominal trajectories).
    w_sync_joint_a: float = 0.005

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
    #: **2026-06-02 (phase1_grad_v3)**: 0.15 → **0.05**. v11 used
    #: ``max_steps = 400`` so the alive cost capped at -60/ep. The
    #: current ``max_steps = 600`` makes the same 0.15/step pile up to
    #: -90/ep — a constant negative baseline that drowns out the FSM
    #: bonus signal during early exploration (when ep_len_mean ≈ 416
    #: the alive cost alone is -62, larger than the per-step expected
    #: value of any single positive dense term). 0.05/step over 600
    #: steps caps at -30/ep, restoring the v6/v9b ratio while keeping
    #: hover discouraged (still a non-trivial opportunity cost).
    #: **2026-06-03 (phase1_grad_v7 / S3)**: 0.05 → **0.15** restored.
    #: v5/v6 deterministic eval showed reward hacking — the policy
    #: hovered for the full episode (≈ 600 steps) collecting dense
    #: shaping (+ 368 ~ + 603 / ep) instead of completing the task
    #: (R_mount + R_demount + R_landing ≈ + 480). 0.15 / step × 600
    #: step = - 90 / ep, larger than any feasible hover-farming
    #: subtotal, restoring sparse > dense ordering.
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
    #: **2026-06-02 (phase1_grad_v5 / Q3)**: 600 → 900 (≈ 45 s at 20 Hz).
    #: v4 deterministic eval on hard HOME spawns used ~500 steps to reach
    #: mount with no time left for demount + cradle return; 900 gives
    #: ~400 steps after mount for Stage 2/3 without changing physics.
    #: **2026-06-03 (phase1_grad_v7 / S1)**: 900 → **600** restored.
    #: v5/v6 deterministic eval revealed dense reward farming over
    #: 900-step hover episodes (rew + 368 ~ + 603 / ep with sparse = 0).
    #: 600 caps the dense-accumulation budget, restoring sparse-bonus
    #: dominance even though hard-HOME ep cannot always finish the
    #: full Pickup → Mount → Demount → Return cycle in 600 step.
    max_steps: int = 600
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
    #: **2026-06-01 (planner-residual rewrite)**: initially flipped back
    #: to ``True`` on the assumption that the Min-Jerk EE trajectory was
    #: collision-free. **Reverted to ``False`` after the v1 training run**
    #: showed every easy-spawn ep terminated within 1–48 steps. Root
    #: cause: the EE trajectory itself is clear, but the UR10 *arm body*
    #: (shoulder / upper-arm links) sweeps into the cargo box during the
    #: long Y-direction carry from cradle (Y≈0) to hub (Y≈+0.80) — the
    #: nominal planner is end-effector-only, not whole-arm collision-free.
    #: With ``False`` the env still applies the per-step
    #: ``-w_collision`` penalty (10.0) so the residual policy is forced
    #: to learn an arm-side avoidance, but the episode survives long
    #: enough to receive the R_mount sparse signal.
    collision_terminates: bool = False

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
    #: **2026-06-05 (real floor pit)** — when ``floor_pit_enable`` is True the
    #: infinite ground plane is lowered to ``floor_z − floor_pit_depth`` (the
    #: pit BOTTOM) and the normal-height floor (top at ``floor_z``) is rebuilt
    #: only OUTSIDE the pit rectangle as four rim slabs. This carves a genuine
    #: rectangular hole into the floor where Robot A's buried base/column and
    #: its low arm links live: the arm can descend inside the pit freely, while
    #: the rim slabs physically stop any link from punching through the floor
    #: *outside* the pit. The pit XY rectangle must enclose every below-floor
    #: arm point across the pickup+carry motion (measured ≈ x∈[−3.06,−0.30],
    #: y∈[−0.98,1.03]); depth must clear the deepest link (link_1 ≈ base_z +
    #: 0.27 ⇒ −0.83 at base_z=−1.10). Robot B (origin) sits on a rim slab.
    floor_pit_enable: bool = False
    floor_pit_depth: float = 0.90
    #: Pit opening shape: "rect" (four rim slabs around a rectangle) or
    #: "circle" (a generated floor plate with a single circular hole over
    #: Robot A's column). The circular pit is sized so it contains EVERY
    #: below-floor arm point across the motion (so the solid floor outside it
    #: blocks the arm from punching through), while staying clear of Robot B.
    floor_pit_shape: str = "rect"
    floor_pit_x_range: Tuple[float, float] = (-3.30, -0.15)
    floor_pit_y_range: Tuple[float, float] = (-1.20, 1.25)
    #: Circular-pit parameters (used when floor_pit_shape == "circle").
    floor_pit_center: Tuple[float, float] = (-1.15, -0.10)
    floor_pit_radius: float = 0.70
    floor_pit_circle_segments: int = 96
    #: Half-width of the surrounding floor slabs (so the visible floor still
    #: looks effectively infinite) and their slab thickness.
    floor_pit_rim_extent: float = 12.0
    floor_pit_rim_thickness: float = 0.20
    floor_pit_rim_rgba: Tuple[float, float, float, float] = (0.55, 0.55, 0.58, 1.0)
    robot_A_base_pos: Tuple[float, float, float] = (-0.80, 0.0, -0.30)
    robot_A_base_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Panda sits at the world origin — the entire scene is expressed
    #: in Robot B's base frame.
    robot_B_base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_B_base_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Reference point for the Robot B-centric observation frame. ``None``
    #: keeps the legacy behaviour (subtract ``robot_B_base_pos``). When the
    #: spacious layout pushes Robot B far out of the carry corridor, the
    #: obs frame is pinned to world origin here so positional channels stay
    #: well-scaled inside ``workspace_radius`` regardless of where B sits.
    obs_reference_pos: Optional[Tuple[float, float, float]] = None
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
    #: Optional explicit base inertia diagonal (kg·m²). When ``None`` and
    #: ``tire_mass >= 50``, ``tire_inertia_heavy`` is used so Bullet does not
    #: infer unrealistic spin from the lightweight collision mesh density.
    tire_inertia_diagonal: Optional[Tuple[float, float, float]] = None
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
    #: **2026-06-02 (kinematic upright lock — final design)** — while
    #: the tire is grasped, re-write its base orientation every control
    #: step so roll/pitch stay at the spawn "standing" pose and only
    #: world +Z yaw can change (driven by the UR10 EE yaw so the bore
    #: can align with the hub). Position follows the EE via a cached
    #: world-frame COM offset. With the cargo penetration guard
    #: (``_sync_grasped_tire_upright`` reverts to the last safe pose
    #: when the kinematic update would push the tire INTO cargo /
    #: back-wall geometry), this approach is robust and physically
    #: meaningful: the tire is rigidly upright, cargo cannot be
    #: phased through, and the policy is penalised for trying.
    #:
    #: A geometric palm-up lock with a ``JOINT_FIXED`` rigid grasp
    #: (path B) was prototyped but rejected: PyBullet's iterative
    #: constraint solver cannot keep a 0.5 kg child rigidly bonded to
    #: a multi-link arm under fast trajectory motion, even with
    #: ``erp=1.0`` and 200 LCP iterations. Bond reaction forces of
    #: 400-650 kN (vs the ~10 N that physics actually requires) and
    #: 0.7 → 0.5 m bond-distance oscillation were observed in
    #: ``scripts/diag_palmup_locked_grasp.py``. Kinematic projection
    #: is the practical solution for this scenario.
    lock_tire_upright_when_grasped: bool = True
    #: Stages where the tire pose is re-written every step (kinematic
    #: upright lock). Default ``(0,)`` = pickup approach only. Stage 1+
    #: uses a ``JOINT_FIXED`` EE bond instead so the tire is not
    #: teleported every physics sub-step (which amplified EE/IK jitter
    #: into visible "수직 유지" shaking during carry/mount).
    kinematic_tire_lock_stages: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    #: EMA blend for kinematic tire sync (1.0 = snap, 0.35 = smooth).
    #: Low values cut the "수직 유지" teleport jitter while keeping reach.
    kinematic_tire_sync_alpha: float = 0.65
    #: **2026-06-04 (insertion smoothing)** — per-joint motor speed cap
    #: (rad/s) applied via PyBullet POSITION_CONTROL ``maxVelocity`` in
    #: ``UR10Robot.drive_arm_targets``. 0 = unlimited (legacy). The carry
    #: itself is smooth; the residual "오락가락" is the stiff PD whipping the
    #: arm through the near-singular hub insertion (hub at 88 % reach).
    #: Measured 2026-06-04: cap ≈ 1.0 rad/s cut the worst insertion EE jump
    #: 70 → 25 cm (mean 6.3 → 1.5 cm) while the baked Min-Jerk joint path
    #: (max |Δq| ≈ 0.15 rad/step ⇒ ~3 rad/s peak) still tracks given the
    #: full step budget. The Min-Jerk profile is already velocity-bounded,
    #: so this only clips the singular-direction overshoot, not the plan.
    #: **Swept 2026-06-04** (zero-action, shipping geometry): 1.0 rad/s is
    #: the sweet spot — worst insertion jump 70 → 25 cm, mean 6.3 → 3.2 cm,
    #: >15 cm jumps 11 → 2, and it mounts at the curriculum-start gate
    #: (mount@112 / 0.12 m) with the tire 5.8 cm from the hub at the hard
    #: 0.04 m gate (the policy residual closes the final ~2 cm). 1.5 rad/s
    #: failed to seat; 2.0 rad/s mounted but stayed noisy (max 38 cm). 0
    #: restores the legacy uncapped behaviour (mount@104, 70 cm snap).
    ur10_motor_max_velocity_rad_s: float = 1.0
    #: Max joint change (rad) per control step when playing back a baked
    #: planner trajectory — slew is taken from the *measured* arm state,
    #: not the previous command, so PD lag cannot cause 1+ rad snaps.
    ur10_joint_slew_max_rad: float = 0.0
    #: Baked planner: advance to the next waypoint when
    #: ``max|q - q_waypoint|`` falls below this (rad).
    planner_joint_waypoint_tol_rad: float = 0.05
    #: **2026-06-04 (waypoint arrival gate)** — OFF by default. When
    #: enabled, ``current_traj_step`` advances only when the measured EE
    #: is within ``planner_waypoint_pos_tol_m`` of the current waypoint
    #: (or the stall watchdog elapses). This was prototyped to stop the
    #: index racing ahead of a lagging arm, but with the *baked* joint
    #: trajectory (which is a pre-solved smooth path played one waypoint
    #: per control step) the natural PD lag exceeds the tolerance almost
    #: everywhere, so the gate stalled the index and the arm never
    #: reached the hub. Left in the codebase (default off) for the
    #: per-step EE-IK path; the baked path does not need it.
    planner_waypoint_gate_enable: bool = False
    planner_waypoint_pos_tol_m: float = 0.04
    #: Max control steps the index may stall at a single waypoint before
    #: it is force-advanced (watchdog). Only consulted when the gate is
    #: enabled. 10 steps ≈ 0.5 s at 20 Hz.
    planner_waypoint_max_stall: int = 10
    #: **2026-06-04 (−Y insertion standoff)** — when > 0, the Stage-1 carry
    #: routes through a pre-hub via-point offset this far along **−Y** (the
    #: wheel-well / hub axis) from the mount EE target, so the final segment
    #: is a straight +Y insertion into the well. Built for the (rejected)
    #: anti-singularity hub-reposition layout, where the cargo box sat in
    #: the carry corridor. **Default 0 (disabled)** — the shipping layout's
    #: mount/insertion subsystem is co-designed with the arch approach and
    #: this standoff is unnecessary (and the multi-via code path is left in
    #: place for any future scene redesign).
    planner_stage1_approach_standoff: float = 0.0
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
    #: **2026-06-02**: lengthened 300k → **500k** so the gate tightens
    #: more gradually (≈ 0.04 mm narrower per 100 steps). With total =
    #: 2M and 100k soft hold this hits the hard regime at t = 600k,
    #: leaving 1.4M steps of pure hard-mode learning.
    approach_tol_ramp_steps: int = 500_000

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
    #: **2026-06-02**: lengthened 300k → **500k** to match the new
    #: ``approach_tol_ramp_steps``. The two curricula are deliberately
    #: synced: tightening the pickup gate while moving the start pose
    #: away from the easy zone in lockstep prevents either curriculum
    #: from outpacing the other.
    start_pos_ramp_steps: int = 500_000
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
    #: **2026-06-02 (phase1_grad_v5 / Q3)**: 1.0 → **0.9** initial mix weight.
    #: When ``start_pos_easy_prob_curriculum_enable`` is True,
    #: ``StartPosEasyProbCurriculumCallback`` drives the live value down
    #: to 0.5 @ 1M steps and 0.3 @ 2M steps so hard HOME spawns ramp in.
    start_pos_easy_prob: float = 0.9
    #: Schedule Bernoulli easy-spawn probability in ``mix`` mode (v5).
    start_pos_easy_prob_curriculum_enable: bool = True
    start_pos_easy_prob_schedule_start: float = 0.9
    start_pos_easy_prob_schedule_mid: float = 0.5
    start_pos_easy_prob_schedule_end: float = 0.3
    start_pos_easy_prob_schedule_mid_steps: int = 1_000_000
    start_pos_easy_prob_schedule_end_steps: int = 2_000_000

    # ------------------------------------------------------------------
    # **v11 (2026-05-31) — Reverse curriculum (backtracking) scheduler.**
    # ------------------------------------------------------------------
    # Independent of the legacy ``start_pos_curriculum_mode``. When
    # ``reverse_curriculum_enable`` is True, the env reset() routes to
    # one of three phases as a step-function of the global PPO step:
    #   * Phase A (0 .. reverse_phase_a_steps): hub-aligned hot-start.
    #     Tire teleported to ``tire_mount_pos`` with bore axis aligned
    #     to hub axis, UR10 EE on the 6-o'clock grasp anchor of that
    #     tire pose, grasp constraint attached, ``task_stage = 1``.
    #     Policy starts within R_mount range every reset.
    #   * Phase A→B blend (reverse_phase_a_steps .. + a_to_b_overlap):
    #     Bernoulli mix — keeps ``reverse_phase_a_mix_prob`` of resets
    #     in Phase A while gradually introducing Phase B. Mitigates
    #     catastrophic forgetting of mount-endgame skill.
    #   * Phase B (.. reverse_phase_b_steps): legacy easy-mix start.
    #     Bernoulli(``start_pos_easy_prob``) on easy vs HOME.
    #   * Phase C (after reverse_phase_b_steps): pure HOME starts.
    reverse_curriculum_enable: bool = False
    #: End of pure-A plateau (global PPO timesteps).
    reverse_phase_a_steps: int = 250_000
    #: End of A→B overlap window. Between ``phase_a_steps`` and this
    #: value resets pick A with probability ``reverse_phase_a_mix_prob``,
    #: else fall through to Phase B sampling.
    reverse_phase_a_to_b_overlap: int = 50_000
    #: Probability of staying in Phase A during the A→B overlap.
    reverse_phase_a_mix_prob: float = 0.75
    #: End of Phase B plateau (also start of pure-C HOME).
    reverse_phase_b_steps: int = 750_000
    #: Distance budget (m) for the tire spawn perturbation around the
    #: mount target in Phase A. Sampled uniformly along the hub-axis
    #: direction inward from the goal, so the policy still has
    #: a small mount-approach to traverse on every reset.
    #: **v11c (2026-05-31)**: 0.03 → 0.01. v11 hot-start sampled
    #: backoff in [0.01, 0.03] m which combined with a 2° angular tilt
    #: pushed a non-trivial fraction of resets close to the mount-tol
    #: boundary (radius_soft = 0.30 m, angle_soft = 35°). After the
    #: first physics decimation step, the policy's untrained Δaction
    #: could shove the tire out of tol → mount event never fired and
    #: ``fsm_bonus`` averaged ≈ 0.25/step instead of the expected
    #: ≈ 1.0/step (one R_mount paid per ep). Tight jitter restores the
    #: "guaranteed first-step mount fire" assumption.
    reverse_phase_a_radial_jitter: float = 0.01
    #: Max angular perturbation (rad) applied to the tire bore axis
    #: around the hub axis during Phase A spawn.
    #: **v11c (2026-05-31)**: 2.0° → 0.5° for the same first-step
    #: mount-fire guarantee. 0.5° ≪ angle_soft = 35° so the tilt never
    #: pushes the angular check past gate.
    reverse_phase_a_angular_jitter: float = np.deg2rad(0.5)
    #: **v11c (2026-05-31)** — Phase A R_mount paid + episode continues.
    #: When ``True``, the env temporarily switches ``terminate_on`` to
    #: "mount" while the ``ReverseCurriculumCallback`` reports Phase A;
    #: Phase B/C revert to the CLI-supplied value (typically "never").
    #:
    #: **v11c1 (2026-05-31)** — default flipped to **False** after the
    #: v11c_balanced_v1 smoke run showed fps = 12 (≈ 1/8 of v11). Root
    #: cause: 1-step episodes triggered a PyBullet ``reset()`` every env
    #: step, which dominates wall-clock. Keeping ``terminate_on="never"``
    #: in Phase A lets the ``_phase_a_force_mount_first_step`` flag still
    #: pay R_mount on step 1 (sparse signal preserved), then the episode
    #: continues through Stage 2 (demount) and Stage 3 (return) up to
    #: ``max_steps`` — which is the actual goal of reverse curriculum:
    #: collect on-policy trajectories *after* the mount event so PPO can
    #: also learn demount + return. Set to ``True`` only if the legacy
    #: 1-step success pattern is desired (e.g. for a Phase A-only
    #: mount-policy distillation run).
    reverse_phase_a_terminate_on_mount: bool = False
    #: **v11c (2026-05-31)** — distance gate (m) on the Stage 0 dense
    #: ``approach_A`` term. When ``d_approach > approach_A_gate``, the
    #: dense reward is zeroed for that step so the policy can't farm
    #: it by sitting just outside the grasp anchor. Reflects the v4 /
    #: v9b plateau diagnosis (policy collected ~+1.5/step from
    #: ``approach_A`` while ``d_A`` stayed at 2.0 m). Set to a
    #: large value (>5 m) to disable.
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
    #: **2026-06-03 — hub mount pose hold (demo / visual correctness).**
    #: Phase-1 FSM continues to Stage 2 (demount) after mount; the tire
    #: stays kinematically glued to the EE until landing unless pinned.
    #: When ``pin_tire_on_mount`` is True, the tire is snapped to
    #: ``tire_mount_pos`` with bore ‖ ``hub_axis_world`` and made static
    #: (same mechanism as cradle pin) at the mount event.
    pin_tire_on_mount: bool = True
    #: After the mount gate fires, keep the arm frozen at the current
    #: joint vector for this many control steps before ending the episode
    #: (only when ``terminate_on == "mount"``). 0 = immediate terminate
    #: (legacy training). 40 ≈ 2 s at 20 Hz — enough to *see* a still
    #: mounted tire in the GUI.
    mount_hold_steps: int = 0
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
    #: **2026-06-06 (mount-seat glide)** — number of env steps over which the
    #: tire is smoothly interpolated from its grasped pose at the mount-gate
    #: fire to the exact seated hub pose, *before* the JOINT_FIXED bond /
    #: pin is applied and the ``mounted`` event is emitted. The arm reaches
    #: the stage-1 end pose ~10 cm short of the hub (kinematic reach limit
    #: carrying the tire under the hub), so an instant ``resetBasePosition``
    #: snap produced a visible ~12-14 cm tire teleport at the mount instant.
    #: Gliding the seat over ``mount_seat_glide_steps`` (0.5 s at 20 Hz)
    #: turns that snap into a smooth slide-onto-the-hub. Set to 0 for the
    #: legacy instant snap. The ``mounted`` event (and any mount-terminate)
    #: is deferred until the glide completes so termination still coincides
    #: with the tire actually being seated.
    mount_seat_glide_steps: int = 10
    #: **2026-06-06 (mount bounce fix)** — disable tire↔hub collision for the
    #: whole episode. The wheel disk's inner radius (``wheel_disk_radial_inner``
    #: = 0.10) is smaller than the hub flange radius (``hub_radius`` = 0.21)
    #: because the lug circle (``bolt_circle_radius`` = 0.1675) sits inside the
    #: flange footprint, so a perfectly-aligned seated tire's disk *overlaps*
    #: the hub cylinder by design. During the kinematic carry/seat this overlap
    #: drove a ~700 kN tire↔hub contact impulse each step that the engine kept
    #: trying to resolve while the sync yanked the tire back — the visible
    #: "tire bouncing off the hub" jitter. The final seated state is defined by
    #: the rigid ``_attach_tire_to_hub`` bond (and the gross approach is still
    #: constrained by the vehicle wheel-well cutout + cargo back wall), so the
    #: hub's own collision against the tire serves no training purpose and is
    #: filtered out. Covers all hub/truck links (flange + bolt children).
    disable_tire_hub_collision: bool = True
    #: Stage 3 cradle return — Euclidean radius (m) the tire centre must
    #: land within of the cradle pickup pose for the Stage 3 → done
    #: gate to consider it a successful landing.
    rack_return_radius_tol: float = 0.05
    #: Stage 1 → 2 trigger: ‖tire − hub‖ < ``mount_radius_tol`` AND tire axis
    #: aligned with hub axis (≤ ``RewardConfig.delta_A`` rad).
    #: **Final hard gate** that the v6 mount curriculum asymptotes to.
    #: **2026-06-06 (gate realism fix)** — raised 0.04 → 0.12 m. The
    #: carry-mount policy (planner + residual on a 100 kg kinematic carry)
    #: asymptotes to ~9.3 cm tire↔hub at best approach and then oscillates
    #: 0.10–0.16 m around the hub; a 0.04 m gate NEVER fired, so the mount
    #: event never triggered: the episode ran to ``max_steps`` (never
    #: terminated) AND the policy kept jittering the tire in/out of the hub
    #: (the visible "teleport away & back"). Because the mount event snaps +
    #: JOINT_FIXED-bonds the tire to ``tire_mount_pos`` exactly
    #: (``_attach_tire_to_hub``), the gate is only a "close enough to seat"
    #: trigger — the final seated pose is exact regardless of gate width.
    #: 0.12 m clears the 9.3 cm best approach with margin so the event fires
    #: reliably; the angle gate (``reward.delta_A`` 5°) is unchanged (the
    #: planner yaw-aligns the bore to ~0°). The old 0.04 m "100 % mount"
    #: number was a measurement artifact: ``eval_ckpt_mount.py`` forced
    #: ``set_mount_tol(0.55, 45°)`` so the event fired early and the snap
    #: then placed the tire at d=0 before the radius was measured.
    mount_radius_tol: float = 0.12
    #: v6 mount-gate curriculum — **start radius** (m) used before the
    #: smoothstep ramp begins. 0.30 m gives the policy a generous
    #: "anywhere near the hub" entry signal so R_mount fires reliably
    #: even when carry trajectory is loose. The radius then sweeps
    #: 0.30 → ``mount_radius_tol`` over ``mount_tol_ramp_steps``.
    mount_radius_tol_soft: float = 0.30
    #: v6 mount-gate curriculum — **start axis tolerance** (rad).
    #: **2026-06-02 (planner-yaw alignment)**: tightened back from
    #: 100° → **30°**. Stage 1's planner end-pose now rotates the
    #: gripper −90° about world +Z so the kinematically-locked tire
    #: bore aligns with ``hub_axis_world`` automatically (see
    #: ``TyroEnv._compute_stage_end_ee_pose`` stage==1 branch). 30°
    #: gives the policy headroom for residual yaw error during the
    #: SLERP rotation but is no longer a "free pass" — the planner
    #: does the bulk of the work. Smoothsteps to ``reward.delta_A``
    #: (5°) over ``mount_tol_ramp_steps``.
    mount_angle_tol_soft_rad: float = np.deg2rad(30.0)
    #: v6 mount curriculum hold (global PPO steps) — gate stays at the
    #: soft pair before the smoothstep ramp engages.
    #: **2026-06-02**: 200k → 300k so the policy has more time to
    #: master the new yaw-aligned mount before the gate tightens.
    mount_tol_curriculum_steps: int = 300_000
    #: v6 mount curriculum ramp (global PPO steps) — smoothstep from
    #: ``(mount_radius_tol_soft, mount_angle_tol_soft_rad)`` down to
    #: ``(mount_radius_tol, reward.delta_A)``.
    #: **2026-06-02**: 600k → 1.0M so each 1° narrowing of the angle
    #: tol gets ~40k steps of policy adaptation (was ~6k under the
    #: 100° → 5° / 600k schedule). With total = 2M this still leaves
    #: 700k steps of pure hard-mode learning at the end.
    mount_tol_ramp_steps: int = 1_000_000
    #: Stage 2 → success: ‖tire − pickup‖ < ``return_radius_tol`` AND tire
    #: descent speed < ``landing_speed_max`` (soft landing).
    return_radius_tol: float = 0.05
    landing_speed_max: float = 0.10

    # ------------------------------------------------------------------
    # **2026-06-05 — Full remount cycle (6-stage FSM).** OPT-IN extension
    # of the 4-stage FSM that matches the complete Robot-A duty cycle:
    #   S0 pick → S1 mount → S2 (hold W1=tighten) release+retract to HOME
    #   → S3 re-approach + re-grip the hub tire → S4 (hold W2=loosen)
    #   demount → S5 carry to rack + land.
    # When ``remount_cycle_enable`` is False (default) the legacy 4-stage
    # FSM (pick→mount→demount→return) is used unchanged so existing
    # training/checkpoints are unaffected.
    # ------------------------------------------------------------------
    #: Master switch for the 6-stage pick→mount→retract→regrip→demount→return
    #: cycle. Production full-task runs set this True with ``terminate_on=never``.
    remount_cycle_enable: bool = False
    #: W1 — steps the arm holds the tire seated on the hub after mount
    #: ("Robot B tightening the nuts") before releasing + retracting to HOME.
    #: 40 ≈ 2 s at 20 Hz.
    tighten_hold_steps: int = 40
    #: W2 — steps the arm holds the re-gripped hub tire ("Robot B loosening
    #: the nuts") before pulling it off the hub. 40 ≈ 2 s at 20 Hz.
    loosen_hold_steps: int = 40

    # ------------------------------------------------------------------
    # **2026-06-07 — Robot B sequential nut-fastening task.** OPT-IN
    # single-arm task that trains Robot B (UR10e + nut-runner tool) to
    # fasten the hub bolts one-by-one while the tire is held mounted on
    # the hub (Robot A frozen at HOME, tire bonded to the hub flange).
    # "Fastening" is modelled GEOMETRICALLY (no nut bodies / torque in
    # sim): the nut-runner ``tool_tip`` must come within ``nut_reach_tol``
    # of a bolt tip AND align its tool axis to the bolt axis within
    # ``nut_align_tol_rad`` for ``nut_hold_steps`` consecutive steps —
    # then that bolt counts as fastened and the target advances to the
    # next bolt. The episode succeeds once all ``n_bolts`` are fastened.
    # When False (default) this path is entirely inert.
    # ------------------------------------------------------------------
    #: Master switch for the Robot-B sequential nut-fastening task.
    nut_fastening_task: bool = False
    #: Width of the appended nut-task obs block (eeB→staging vec [3], bolt
    #: axis unit [3], alignment θ [1]). Kept as a field so the dim math and
    #: ``_compute_obs`` stay in sync.
    nut_obs_extra_dim: int = 7
    #: tool_tip → bolt-tip distance (m) admitting a bolt into the
    #: fasten gate. 4 cm matches the mount gate; the nut-runner socket
    #: only needs to seat over the stud, not contact it.
    nut_reach_tol: float = 0.04
    #: tool-axis ↔ bolt-axis angular tolerance (rad). Both ±axis count
    #: (the env folds θ→min(θ, π−θ)) so the policy may approach the bolt
    #: from either bore direction. 15° ≈ a forgiving coaxial seat.
    nut_align_tol_rad: float = np.deg2rad(15.0)
    #: Consecutive in-gate steps required before a bolt is registered as
    #: fastened (models the run-down dwell). 6 ≈ 0.3 s at 20 Hz (halved from
    #: 12 for v6 — shorter dwell to speed the per-bolt cycle).
    nut_hold_steps: int = 6
    # --- insertion-retract gate (the tighten cycle) ----------------------
    #: Max tool_tip distance off the bolt axis (m) admitted into the INSERT
    #: gate. The socket must enter coaxially (exactly along the bolt's Y
    #: axis) so it overlaps the stud; ~1.5 cm ≳ the M22 stud radius (0.011).
    nut_lateral_tol: float = 0.015
    #: Axial tolerance (m) for the INSERT depth target. The tool_tip must
    #: reach the bolt *base* (hub face, axial ≈ −L/2) within this band to
    #: count as fully seated over the stud (socket envelops the whole bolt).
    nut_insert_depth_tol: float = 0.02
    #: Extra axial clearance (m) past the bolt tip the tool_tip must back
    #: out to on RETRACT so the socket fully separates from the stud (no
    #: residual overlap ⇒ "no collision on withdrawal"). Retract distance
    #: from the seated base is therefore ≈ bolt_length + this.
    nut_retract_clear: float = 0.03
    #: Standoff (m) beyond the bolt tip for the oracle/demo APPROACH pose
    #: (pre-insert staging point on the bolt axis). Unused by the policy.
    nut_insert_standoff: float = 0.05
    #: When True, fasten bolts strictly in the ``nut_bolt_order`` sequence
    #: (the env forces the first entry at reset and advances along the list).
    #: When False the per-episode random target is kept and a single bolt is
    #: trained (used for early single-bolt curriculum).
    nut_sequential: bool = True
    #: Explicit bolt-fastening order (a permutation of the bolt indices). A
    #: *balanced* / star-like order spreads the fastening around the lug
    #: circle instead of walking it sequentially, which (a) mirrors how lug
    #: nuts are torqued in practice and (b) gives the policy a more uniform
    #: spatial spread of approach geometries early on. Entries outside the
    #: actual bolt count are skipped; any bolts missing from the list are
    #: appended in ascending index order so the sequence always covers all.
    nut_bolt_order: tuple = (0, 5, 7, 2, 3, 8, 9, 4, 6, 1)
    #: Optional path to a ``.npz`` snapshot of Robot-A mount-completion
    #: poses (arrays ``qA`` [N,6], ``tire_pos`` [N,3], ``tire_orn`` [N,4]),
    #: produced by ``scripts/extract_mount_endpose.py`` from a trained mount
    #: policy. When set and the file exists, the nut-fastening reset samples
    #: one snapshot so Robot A is frozen at the *actual* learned mount-hold
    #: pose (exact deployment-distribution match) instead of the analytic
    #: 6-o'clock anchor. Empty / missing ⇒ analytic fallback.
    nut_mount_endpose_path: str = "data/nut_mount_endpose.npz"
    #: Per-joint uniform jitter (rad) added to Robot A's frozen mount-hold
    #: joint vector each reset. Injects support-pose variety so the Robot-B
    #: policy treats A as a *distribution* of obstacle poses rather than one
    #: memorised configuration — improves robustness to the exact pose the
    #: deployed mount policy ends at. 0 ⇒ no jitter. ~3° keeps the gripper
    #: on the tread (the tire stays hub-bonded regardless).
    nut_a_hold_jitter_rad: float = np.deg2rad(3.0)
    # --- Robot-B reverse-curriculum hot-start ----------------------------
    #: When True, Robot B starts each episode partway between its HOME pose
    #: and the first target bolt's approach pose, controlled by
    #: ``nut_b_hotstart_alpha`` (1 = right at the bolt approach point,
    #: 0 = full HOME distance). The flat exp-reach landscape gives no
    #: gradient from the 1.7 m HOME standoff, so a reverse curriculum
    #: (start near the bolt, then back the start pose off to HOME) is what
    #: lets the policy first discover the insert, exactly like the mount
    #: reverse-curriculum hot-start. The training callback ramps alpha
    #: 1 → 0 over ``nut_b_hotstart_*_steps``.
    nut_b_hotstart_enable: bool = True
    #: 2026-06-08 (v10) — hot-start target = the **bolt-ring CENTER** at staging
    #: depth, not the per-bolt approach point. Measured (0, -0.21, 0): reachable
    #: (IK err 0.6 cm), no A↔B contact, and **equidistant (0.21 m) to all 10
    #: bolts** at a **fixed Y = staging depth**. This removes the bolt-0 bias
    #: (every bolt is now a symmetric pure-XZ radial reach in the constant-Y
    #: plane) and matches the corrected "XZ-only transit at fixed Y" path. The
    #: legacy per-bolt approach start floored bolt-0 as free while every other
    #: bolt was unscaffolded; the center start makes the learned skill ("reach
    #: radially to the target bolt at fixed Y") reusable for all bolts.
    nut_b_hotstart_hub_center: bool = True
    #: Current hot-start interpolation (set per-rollout by the curriculum
    #: callback; the env reads it each reset). 1 = bolt approach, 0 = HOME.
    #: Default 0 = deployment/HOME so envs *without* the curriculum callback
    #: (eval, preview) measure the true full-distance task; the training
    #: callback overrides this to 1.0 at start and ramps it back to 0.
    nut_b_hotstart_alpha: float = 0.0
    #: Per-bolt random start (curriculum coverage). When enabled (and the
    #: hot-start is active, alpha > 0), each reset seeds the chain at a
    #: uniformly random bolt k (bolts < k pre-marked fastened) instead of
    #: always bolt 0. Always starting at bolt 0 means later bolt-to-bolt
    #: transitions are sampled only after every earlier bolt is cleared —
    #: vanishingly rare early in training — so competence stalls at an
    #: advancing frontier (v2/v3 stuck at bolt 3, v4 at bolt 4) with the
    #: socket parked ~10 cm off the next bolt's axis. Uniform start gives
    #: every transition equal training mass and removes the frontier.
    #: 2026-06-08 — disabled, then RE-ENABLED. Disabling it (always start at
    #: bolt 0) was meant to encourage bolt-specific approaches, but in practice
    #: it meant ONLY bolt 0 ever received hot-start scaffolding: every episode
    #: fastened bolt 0 trivially, then had to traverse to the next bolt
    #: (bolt 5) entirely unscaffolded. At the ~1 m inter-bolt distance the
    #: approach kernels (lateral exp(-d/0.08), coax-gated reach) are exactly 0,
    #: so there was no gradient to learn that transition — n_fastened stalled
    #: at 1 for the whole run (observed v7 @440k: nut_lateral pinned ~1 m,
    #: max n_fastened = 1). Uniform random start gives EVERY bolt direct
    #: hot-start training mass (start at a random order position k, bolts < k
    #: pre-fastened, B teleported to bolt k's staging), so each bolt's approach
    #: is learned; the alpha ramp then extends the start distance to stitch the
    #: transitions into full sequential execution. This is the mechanism the
    #: curriculum needs — re-enabled to get past bolt 0.
    #: 2026-06-08 (v9) — DISABLED again for the collision-avoidance experiment:
    #: always start at bolt 0 so the honest n_fastened_policy / eval success
    #: reflect a true cold sequential run (no premark inflation), and the policy
    #: must learn the full traverse + collision-free approach under raised
    #: exploration (ent_coef) rather than leaning on per-bolt hot-start.
    nut_b_hotstart_random_bolt: bool = False
    #: Per-joint motor torque caps (N·m) for Robot B during the nut task,
    #: overriding the tire-carrying default ([400,400,300,60,60,60]). The
    #: far-arc bolts need near-full extension where the default elbow cap is
    #: below the static gravity moment (arm sags ~36 cm, cannot hold staging).
    #: B carries no payload here so higher caps are safe.
    nut_b_motor_forces: tuple = (6000.0, 6000.0, 4000.0, 1000.0, 1000.0, 1000.0)
    #: Per-step EE translation scale (m) for Robot B in the nut task. The
    #: shared 0.02 m/step makes the ≥1 m HOME→bolt traverse ~90 steps even
    #: in a straight line; 0.05 keeps the traverse tractable inside the
    #: 600-step horizon while staying smooth enough for the 1.5 cm insert.
    nut_pos_scale: float = 0.05
    # --- scripted insert→hold→retract macro ------------------------------
    #: 2026-06-07 — the policy ONLY learns to APPROACH each bolt's staging
    #: point (just outside the stud tip, on-axis). The delicate in/out is no
    #: longer learned: once the socket arrives coaxially + aligned at the
    #: staging point, the environment *forces* a deterministic
    #: insert→hold→retract macro (driven by ``apply_absolute_ee`` straight
    #: down/up the bolt axis), so the tighten cycle is always geometrically
    #: correct and collision-free. The per-bolt fasten still requires the
    #: *measured* socket to seat at the base then clear the tip.
    nut_scripted_macro: bool = True
    #: Consecutive in-gate steps at the staging point required to trigger the
    #: macro. 2026-06-07 — dropped 3 → 1. With exploration noise of ~3 cm/step
    #: (log_std −0.5 × pos_scale 0.05) the policy almost never held the old
    #: tight 3-consecutive-step gate, so it never sampled the macro reward and
    #: never learned to seat. The macro itself drives to a *cached* base IK
    #: regardless of the exact arrival pose, so a single in-capture step is a
    #: safe trigger.
    nut_arrive_steps: int = 1
    #: Capture radius (m) for the APPROACH→macro trigger: the macro fires once
    #: the socket tip is within this sphere of the on-axis staging point
    #: (combined with the alignment gate). Generous (5 cm ≫ the old ±2 cm
    #: axial / 1.5 cm lateral box) so the policy can realistically reach it
    #: under exploration; the scripted macro supplies the final precision.
    nut_arrive_pos_tol: float = 0.08
    #: Alignment gate (rad) for the trigger — tool +Z within this of the bolt
    #: axis. Live value, ramped down by ``NutArriveAngCurriculumCallback`` from
    #: ``nut_arrive_ang_start_deg`` → ``nut_arrive_ang_end_deg`` so the policy
    #: first samples the macro under a loose gate, then must align ever tighter
    #: to trigger. Default = the *tight end* so envs without the callback
    #: (eval / smoke) measure honestly at deployment difficulty, mirroring the
    #: hot-start alpha default of 0.0 (= hardest, full HOME).
    nut_arrive_ang_tol_rad: float = np.deg2rad(12.0)
    # --- arrive-alignment curriculum -------------------------------------
    #: Whether to ramp the arrive alignment gate during training.
    nut_arrive_ang_curriculum: bool = True
    #: Loose start (deg): generous so the macro reward is reachable early.
    nut_arrive_ang_start_deg: float = 35.0
    #: Tight end (deg): the alignment quality we ultimately want at trigger.
    nut_arrive_ang_end_deg: float = 12.0
    #: Steps to hold the loose start before ramping (lets the value function
    #: learn the macro is valuable while the gate is easy to hit).
    nut_arrive_ang_hold_steps: int = 300_000
    #: Steps to linearly ramp start → end after the hold.
    nut_arrive_ang_ramp_steps: int = 1_500_000
    #: Watchdog: max control steps the macro may spend in any one leg
    #: (INSERT / HOLD / RETRACT) before it force-advances, so an IK stall
    #: can't hang the episode. Generous vs the ~6-step legs.
    nut_macro_leg_max_steps: int = 30
    #: Per-control-step axial travel (m) of the forced macro. The macro
    #: IK-teleports the socket toward the leg target capped at this stride
    #: so the in/out is *visible* (several steps per leg) yet fast enough
    #: that all 10 bolts fit the episode horizon (PD tracking the full
    #: plunge took ~50 steps/bolt — too slow). ~4 cm ⇒ ~5-step legs.
    nut_macro_step_m: float = 0.04
    #: Extra axial margin (m) for the in/out cycle. The APPROACH staging
    #: point is pushed this much further out in −Y (away from the hub, more
    #: clearance before the plunge), so the forced macro plunges from
    #: further out to the hub-face base — a longer, deeper-*looking* insert
    #: stroke (the base is the deepest reachable point; the hub blocks
    #: anything past it). The RETRACT then backs out this much further past
    #: the tip too. 0 ⇒ legacy park-at-tip / seat-at-base cycle.
    nut_insert_margin: float = 0.03
    #: S2 retract gate — EE must come within this of the HOME EE pose for the
    #: empty-handed retract (S2 → S3) to fire.
    home_return_radius_tol: float = 0.12
    #: S3 regrip gate — EE must come within this of the hub-mounted tire's
    #: 6-o'clock grasp anchor for the re-grip (S3 → S4) to fire.
    regrip_radius_tol: float = 0.10
    #: Vertical tolerance applied only in Stages 0 / 2 (pickup pose +
    #: post-landing pose). Episodes terminate (penalty) when the tire's
    #: bore axis deviates from ``tire_spawn_axis_world`` by more than
    #: this angle (rad). Stage 1 (carry/mount) waives the check entirely
    #: so the policy can rotate the tire 90° about world +Z to align
    #: the bore with ``hub_axis_world`` for mount.
    vertical_tol_rad: float = np.deg2rad(15.0)
    #: **2026-06-02 (Stage 3 cradle-return gate fix)** — radius (metres)
    #: from ``tire_pickup_pos`` within which the Stage 3 vertical gate
    #: (termination + dense penalty) re-activates. Outside this radius
    #: the policy is free to slerp the tire bore from ``hub_axis_world``
    #: (-Y) back to ``tire_spawn_axis_world`` (+X) along the planner
    #: return trajectory without tripping the 15° tolerance. Inside the
    #: radius the spawn vertical pose is enforced again so the cradle
    #: landing still demands the production pose. 0.20 m comfortably
    #: covers the cradle drop neighbourhood (rack diameter ≈ 0.30 m)
    #: while leaving > 1.5 m of free-rotation runway across the carry
    #: trajectory.
    stage3_vertical_gate_radius: float = 0.20
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
    # **2026-06-01 — Cargo back wall (inner cargo block).**
    # ------------------------------------------------------------------
    # A separate static body placed *behind* the hub (between the wheel
    # well and the cargo interior) so the tire cannot be pushed through
    # the hub flange into the cargo. Independent body avoids hitting
    # PyBullet's compound primitive count limit when adding more cargo
    # subdivisions. Geometry: thin slab spanning the cargo's X-Z face,
    # placed at world Y just past the hub (= ``hub_y + back_wall_y_offset``)
    #: Master switch — when False, no extra body is spawned.
    spawn_cargo_back_wall: bool = True
    #: Y offset (m) from the hub centre to the back-wall plane. The hub
    #: thickness is 0.06 m and the tire is 0.30 m thick; once the tire
    #: COM is at the hub centre, the tire's far face sits at hub_y + 0.15.
    #: **2026-06-01 (relaxed)** — pushed 0.18 → **0.30** m so the wall
    #: sits in the middle of the 0.50 m cargo depth (world Y = 1.10):
    #:   * 15 cm clearance past the mounted-tire face (was only 3 cm)
    #:   * 20 cm clearance from the cargo's outer face (Y = 1.30)
    #: This gives the policy a small overshoot tolerance during the final
    #: mount approach without letting the tire actually penetrate the hub
    #: into the cargo interior.
    cargo_back_wall_y_offset: float = 0.30
    #: Half-extents (X, Y, Z). Y half-thickness is the wall thickness.
    cargo_back_wall_half_extents: Tuple[float, float, float] = (1.0, 0.02, 0.50)
    #: Z centre of the wall (world). Default = cargo centre Z so it spans
    #: the same vertical range as the cargo body.
    cargo_back_wall_center_z: float = 0.78
    cargo_back_wall_rgba: Tuple[float, float, float, float] = (0.45, 0.30, 0.30, 1.0)

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
    #: **2026-06-04 (open-corridor cradle)** — when True the two rails are
    #: built as thin top *bars* (``tire_rack_half_extents`` z is small) that
    #: only touch the tire at the seating height, and each bar is propped up
    #: from the floor by a vertical post placed on its OUTER face (away from
    #: the Y=0 centerline). This leaves the inner-lower region — the corridor
    #: the FANUC forearm sweeps through to reach the 6 o'clock grasp point —
    #: completely open, so ``link_5`` no longer collides with a full-height
    #: rail column. The post never enters the arm corridor (it sits beyond
    #: the tire's tread on the outer side). Off by default (legacy full
    #: column rails for the shipping layout).
    tire_rack_support_posts: bool = False
    #: Outer support-post half-extents (X, Y, Z is auto-derived to span
    #: floor → bar bottom). Only used when ``tire_rack_support_posts``.
    tire_rack_post_half_extents_xy: Tuple[float, float] = (0.10, 0.05)

    # Bolt surface properties (helps avoid unrealistic sticking on micro-contacts).
    bolt_lateral_friction: float = 0.8
    bolt_spinning_friction: float = 0.01

    # URDF paths
    ur10_urdf: str = str(URDF_DIR / "ur10_robot" / "ur10_robot.urdf")
    ur10_search_path: str = str(URDF_DIR / "ur10_robot")
    panda_urdf: str = "franka_panda/panda.urdf"  # resolved via pybullet_data

    #: Robot A model: ``"fanuc_r2000ic"`` (default) or legacy ``"ur10"``.
    robot_a_kind: str = "fanuc_r2000ic"
    #: Robot B model: ``"ur10e"`` (default) or legacy ``"panda"``.
    robot_b_kind: str = "ur10e"
    #: ``"fanuc_spacious"`` = FANUC + UR10e (default); ``"shipping"`` = UR10 + Panda.
    scene_layout: str = "fanuc_spacious"
    fanuc_urdf: str = str(URDF_DIR / "fanuc_r2000ic" / "r2000ic210f_wheeltool.urdf")
    fanuc_search_path: str = str(URDF_DIR / "fanuc_r2000ic")
    fanuc_mesh_support_path: str = str(
        URDF_DIR / "fanuc_ros" / "fanuc_r2000ic_support"
    )
    #: Larger plinth for the R-2000iC base footprint.
    fanuc_stand_radius: float = 0.28
    fanuc_stand_rgba: Tuple[float, float, float, float] = (0.32, 0.34, 0.38, 1.0)
    #: FANUC defaults to full 6-DOF planner IK (no UR10 palm-up wrist lock).
    fanuc_lock_tool_up: bool = False
    #: **2026-06-06 (post-step palm-up re-lock)** — kinematically snap the
    #: FANUC gripper back to palm-up (tool +Z = world +Z, yaw free) after the
    #: physics sub-steps whenever the achieved tool +Z has drooped past
    #: ``fanuc_palm_up_tool_z_threshold``. The 100 kg grasped tire makes the
    #: stiff position PD settle ~15° off palm-up at the far mount-insertion
    #: reach (a steady-state tilt that extra wrist torque cannot remove and
    #: that higher position gain only destabilises). This guarantees the
    #: gripper stays +Z while still allowing free rotation about vertical.
    #: The grasped tire is re-placed via the cached EE↔tire transform so the
    #: rigid bond is not shocked. Off ⇒ legacy command-only palm-up.
    fanuc_enforce_palm_up_post_step: bool = False
    #: Dot(tool +Z, world +Z) below which the post-step re-lock fires.
    #: 0.999 ≈ 2.5°, 0.9998 ≈ 1.1°.
    fanuc_palm_up_tool_z_threshold: float = 0.999
    fanuc_motor_max_velocity_rad_s: float = 1.0
    fanuc_joint_target_smooth_alpha: float = 1.0
    fanuc_joint_max_step_rad: float = 0.0
    fanuc_position_gain: float = 1.0
    fanuc_velocity_gain: float = 1.0
    fanuc_joint_slew_max_rad: float = 0.08
    #: Per-joint POSITION_CONTROL torque caps (N·m), order j1..j6. The real
    #: R-2000iC/210F is a 210 kg-payload arm so the big joints can be driven
    #: very hard; these are deliberately conservative sim defaults. Raise via
    #: ``fanuc_torque_scale`` (global multiplier) or override the list outright.
    #: NOTE (scripts/diag_torque_tracking.py): the 100 kg baked carry already
    #: reaches the mount target at scale 1.0 (EE end err = baked residual), so
    #: the carry is NOT torque-limited — the realised-path quality is governed
    #: by the velocity cap + PD gains, and scales ≥ ~4× actually DESTABILISE the
    #: stiff position PD (overshoot/whip). Increase only with care.
    fanuc_arm_motor_forces: Tuple[float, ...] = (
        2000.0, 2000.0, 1500.0, 400.0, 400.0, 200.0,
    )
    #: Global multiplier applied to ``fanuc_arm_motor_forces``. 1.0 = datasheet-
    #: conservative default; bump (e.g. 1.5–2.0) for extra static-load headroom.
    fanuc_torque_scale: float = 1.0
    #: EE link name for IK / grasp parent (FANUC + wheel tool).
    fanuc_ee_link_name: str = "wheel_tool_tip"
    #: Optional 6-joint HOME override (rad). ``None`` keeps the class default
    #: ``FanucR2000icRobot.HOME_POSE``. The spacious layout sets a folded,
    #: palm-up ready pose so the wheel tool sits ~0.9 m high (not stretched
    #: to ~2.45 m). Re-tune via ``scripts/audit_fanuc_layout.py``.
    fanuc_home_pose: Optional[Tuple[float, ...]] = None
    #: Optional 6-joint **physical reset** override (rad) — the config the arm is
    #: parked in at episode reset. DECOUPLED from ``fanuc_home_pose``: the latter
    #: still seeds ``arm.rest`` (IK warm-start) and defines the palm-up
    #: ``FINAL_LOCK_QUATERNION``, while this only sets where the arm physically
    #: sits at reset. Needed because the canonical IK-seed home folds the wrist
    #: below the floor outside the narrow column pit; this lifts it clear.
    #: ``None`` ⇒ reset to ``fanuc_home_pose`` (legacy behaviour).
    fanuc_reset_pose: Optional[Tuple[float, ...]] = None
    #: Grasp anchor: tire COM = EE + (0, 0, tire_outer_radius) in world +Z
    #: when bore is vertical (same convention as UR10 palm-up grasp).
    grasp_com_offset_world: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    ur10e_urdf: str = str(URDF_DIR / "ur10e_robot" / "ur10e.urdf")
    ur10e_search_path: str = str(URDF_DIR / "ur10e_robot")
    ur10e_mesh_support_path: str = str(URDF_DIR / "ur_ros" / "ur_e_description")
    #: Length (m) of the nut-runner socket extension bolted to the UR10e
    #: ``tool0`` flange along +Z (the bolt/approach axis). 0 = bare flange.
    #: When > 0 the EE frame resolves to the ``tool_tip`` link (the socket
    #: tip) so IK targets the nut, and the extra reach lets Robot B's base
    #: sit further -Y while still reaching every hub bolt (≈ 1:1 trade with
    #: -Y base displacement, since the tool points +Y back toward the hub).
    #: Requires a URDF carrying the matching ``nut_runner``/``tool_tip``
    #: links (see ``ur10e_with_nut_tool.urdf``).
    ur10e_nut_tool_length: float = 0.0
    ur10e_stand_radius: float = 0.12
    ur10e_stand_rgba: Tuple[float, float, float, float] = (0.35, 0.38, 0.42, 1.0)
    ur10e_motor_max_velocity_rad_s: float = 1.0
    ur10e_position_gain: float = 1.0
    ur10e_velocity_gain: float = 1.0
    #: 100 kg truck tire sim (fanuc_spacious layout). Legacy shipping uses 0.5.
    tire_mass_heavy: float = 100.0
    #: Manual inertia for ``tire_mass_heavy`` (bore spin, transverse, transverse).
    tire_inertia_heavy: Tuple[float, float, float] = (18.0, 32.0, 32.0)
    #: When True and ``tire_mass >= 50``, carry stages use JOINT_FIXED not kinematic.
    heavy_tire_fixed_grasp: bool = True
    #: Min link index for A↔B collision penalty (skip base links).
    robot_ab_collision_min_link: int = 2
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
    #: **2026-06-04**: 0.15 → **0.10**. With the waypoint arrival gate +
    #: per-step EE IK (no joint bake), the realised path tracks the
    #: Min-Jerk nominal closely; a 0.15 m residual let the policy yank
    #: the EE far enough off the plan to re-introduce the zig-zag the
    #: gate is meant to remove. 0.10 m keeps genuine obstacle-avoidance
    #: headroom while letting the smooth planner dominate the motion.
    planner_pos_offset_scale: float = 0.10
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
    #: Master switch for planner palm-up **tilt-lock** (tool +Z = world +Z,
    #: yaw from SLERP). Which FSM stages use it is set by
    #: ``planner_lock_palm_up_stages``.
    planner_lock_palm_up: bool = True
    #: **2026-06-01 (Option B, restored)** — FSM stage indices where
    #: tilt-lock is active. Default ``(0, 1, 2, 3)`` = ALL stages: the
    #: gripper stays palm-up throughout the cycle so the tire — which
    #: stands vertically when picked up — keeps standing vertically all
    #: the way to the hub and back. Yaw is still free (inherited from
    #: the planner SLERP) so the wrist can rotate about the vertical to
    #: align the bore with the hub axis. Set to ``(0,)`` to relax the
    #: lock to pickup-only (Option C, faster IK convergence but the
    #: tire flops over during carry).
    planner_lock_palm_up_stages: Tuple[int, ...] = (0, 1, 2, 3)
    #: **2026-06-06 (remount cycle)** — palm-up tilt-lock stages for the
    #: 6-stage remount cycle. Differs from the 4-stage default: S2 is the
    #: *empty-handed* retract to HOME, where baking the home EE pose to a
    #: palm-up quat (instead of the true captured home orientation) put IK
    #: in a branch that stalled ~0.24 m short of HOME, so S2→S3 never fired.
    #: S2 is EXCLUDED for that reason, and S5 (return-to-rack) is excluded for
    #: the SAME reason — baking the rack grasp pose to a palm-up quat stalled
    #: the IK ~0.6 m short of the rack so the landing never fired. Both the
    #: demount (S4) and return (S5) carry the tire KINEMATICALLY (the per-step
    #: ``_sync_grasped_tire_upright`` sets the tire orientation directly), so
    #: the planner no longer needs the palm-up lock to keep the tire upright
    #: on those legs. Kept for S0 pickup, S1 mount-carry, S3 regrip-approach,
    #: S4 demount pull-off (short near-hub move that tracks fine palm-up).
    remount_planner_lock_palm_up_stages: Tuple[int, ...] = (0, 1, 3, 4)
    #: Stage 1 carry arch height (m). 0.5 m clears the UR10 base column
    #: on a straight-line carry from cradle to hub; lower values risk
    #: clipping the base / cargo bottom.
    #: **2026-06-02 (phase1_grad_v3)**: 0.5 → **0.35**. Geometry
    #: re-check — base column top sits at z = 0 (plinth height 0.30 m
    #: above floor at z = -0.30); arch midpoint with lift 0.35 sits at
    #: z = (-0.13 - 0.305)/2 + 0.35 = +0.13, giving 13 cm clearance
    #: over the column top. The 0.5 m lift gave 28 cm clearance but
    #: pushed the via-point distance from the UR10 base to 0.72 m
    #: which, combined with the end-pose at 1.13 m (86 % of the
    #: 1.32 m max reach), saturates the UR10 IK during the descent
    #: phase of the arch — v2 logged ``ik_residual_A_mean ≈ 0.45 m``
    #: which directly inflates ``contact_force_mean`` (≈ 1.8 kN) and
    #: ``sync_joint_A`` (motor servoing hard against the IK clip).
    #: 0.35 trims the via-point distance to 0.61 m (15 cm headroom)
    #: which empirically restores clean IK tracking on the descent.
    planner_stage1_lift: float = 0.35
    #: Nominal trajectory length (control steps). 100 step ≈ 5.0 s at
    #: 20 Hz control. End-pose is held constant once the index exceeds
    #: this length so the policy can still operate on a "fix to last
    #: pose" basis if it has not yet triggered the stage gate.
    planner_traj_steps: int = 100
    #: **2026-06-02 (D4 — Stage 1/3 yaw front-loading)** — exponent ``k``
    #: in the SLERP time-warp ``s(t) = 1 - (1 - t)^k`` applied to the
    #: orientation interpolation for Stage 1 (carry) and Stage 3
    #: (return). With ``k = 2.5`` about 60 % of the 90° bore yaw
    #: rotation completes in the first 30 % of the trajectory and
    #: > 95 % by 70 %, so the tire enters the cradle gate (D1 fix)
    #: with the bore already aligned to the destination axis. This
    #: is the planner-side companion to D1: D1 demotes Stage 3
    #: vertical violation to penalty-only, D4 makes the violation
    #: itself rare by getting the yaw recovery done early. Set to
    #: ``<= 1.0`` to fall back to uniform linear SLERP (legacy
    #: behaviour).
    planner_yaw_front_load_k: float = 2.5
    #: **2026-06-03 — UR10 joint-target motion smoothing (visual jitter fix).**
    #: The planner emits a fresh IK target every 20 Hz control step; near
    #: reach-saturation the raw IK solution jitters step-to-step and the
    #: stiff PD (gains 1.0, 400/300/60 N·m caps) snaps the arm toward each
    #: new target inside the 240 Hz physics window, which reads as a
    #: visible "흔들흔들" tremor. These two knobs low-pass the *commanded*
    #: joint targets so the arm moves smoothly even when IK is noisy.
    #: Disabled by default (alpha=1.0, max_step=0.0) so trained policies
    #: see the exact control dynamics they were trained on; enable for
    #: demos / replay via overrides.
    #:
    #: EMA blend factor for the commanded joint vector:
    #:   q_cmd = alpha * q_ik + (1 - alpha) * q_cmd_prev
    #: 1.0 = no smoothing (legacy), 0.2–0.4 = visibly smooth.
    ur10_joint_target_smooth_alpha: float = 1.0
    #: Hard cap on the per-control-step joint change (rad) applied AFTER
    #: the EMA. 0.0 = disabled. ~0.05–0.10 rad caps the worst IK spikes
    #: so the arm slews instead of snapping. Acts like a crude joint
    #: speed limit (≈ alpha·freq rad/s).
    #: **2026-06-04**: tested at 0.06 but reverted to **0.0**. With the
    #: baked Min-Jerk joint trajectory (the reachable path), the per-step
    #: joint change must occasionally exceed 0.06–0.12 rad (e.g. the
    #: shoulder-pan swing from cradle to hub); a hard cap stalled the
    #: carry and the tire never reached the hub in the 600-step budget.
    #: The baked trajectory is *already* a smooth, bounded-jerk speed
    #: profile, so it provides the "real-robot" motion without an extra
    #: throttle. Keep 0.0 for training; enable in replay/eval only.
    ur10_joint_max_step_rad: float = 0.0
    #: **2026-06-03 — UR10 PD gains (visual jitter fix, part 2).**
    #: PyBullet ``POSITION_CONTROL`` positionGain/velocityGain for the
    #: arm motors. The legacy 1.0/1.0 is extremely stiff — it tries to
    #: annihilate the joint error in a single solver pass, which (with
    #: the high torque caps) overshoots and rings around a fixed target,
    #: reading as a tremor even when the IK target is perfectly still.
    #: Lowering positionGain to ~0.1–0.3 makes the motor *ease* toward
    #: the target over several physics steps (critically-damped-ish)
    #: instead of snapping. Defaults kept at 1.0/1.0 so trained policies
    #: see their original dynamics; override (with retrain) or use in
    #: replay/eval for a smooth demo.
    ur10_position_gain: float = 1.0
    ur10_velocity_gain: float = 1.0
    #: **2026-06-03 — bake joint-space planner trajectory at replan time.**
    #: When True (default), ``_replan_for_current_stage`` runs IK once per
    #: nominal waypoint with chained warm-start (previous solution → next)
    #: and stores ``_traj_q``. Each control step then *plays back* joint
    #: targets — no per-step ``apply_palm_up_pose`` (HOME warm-start IK),
    #: which was the dominant source of visible arm tremor even with zero
    #: policy residual. Set False to restore the legacy per-step IK path
    #: (needed only for A/B debugging).
    #: **2026-06-04**: kept **True** (baked reaches the hub cleanly —
    #: zero-action smoke mounts at ~step 104, d_hub→0). A per-step EE-IK
    #: path warm-started from the *current* joints stalls ~0.85 m short
    #: of the hub, so True is required for reachability. The realised-
    #: path "오락가락" came from the *residual-active* branch throwing the
    #: baked solution away and re-solving IK from scratch each step; that
    #: is now fixed by warm-starting the residual IK from the baked joint
    #: vector (see ``planner_residual_warmstart_from_baked``).
    planner_precompute_joint_traj: bool = True
    #: **2026-06-04** — moving-average window (waypoints) over the baked
    #: joint trajectory ``_traj_q``. **Default 0 (disabled).** Prototyped
    #: as a fix for the baked carry zig-zag (zero-action straightness
    #: ratio 3.15, 29 reversals), but measurement showed any window ≥ 5
    #: pulls the final approach off the 4 cm mount tolerance → the tire
    #: reaches the hub neighbourhood but never trips the mount gate
    #: (mounted@None, wanders to max_steps). Joint-space smoothing is
    #: therefore incompatible with the tight mount gate; left in the
    #: codebase (off) for experiments only. The realised carry wiggle is
    #: dominated by the arch + 90° yaw, not IK chatter — a genuine fix
    #: needs a straighter carry plan or Cartesian tracking (future work).
    planner_smooth_baked_window: int = 0
    #: **2026-06-04** — warm-start the residual-offset IK from the baked
    #: joint solution ``_traj_q[idx]`` instead of the live joint state.
    #: NOTE (measured): for the 6-DOF UR10 with 200-iter DLS IK and a
    #: reachable target, the converged solution is essentially branch-
    #: independent, so this had **no measurable effect** on the realised
    #: path (identical metrics with it on/off). Kept (cheap, harmless) as
    #: defence for poses near singularities, but it is *not* the carry-
    #: smoothness lever it was hoped to be. Default off to avoid implying
    #: otherwise; flip on for redundant/near-singular configurations.
    planner_residual_warmstart_from_baked: bool = False
    #: **2026-06-04 — DLS Cartesian servo (smoothness fix).**
    #: When True the planner-residual path drives the UR10 with a damped
    #: least-squares resolved-rate servo (``UR10Robot.drive_ee_servo_dls``)
    #: toward the nominal+residual EE pose, instead of per-step absolute
    #: IK. This is a closed-loop step on the EE error, so the commanded
    #: joint target never sits far from the achievable pose — eliminating
    #: the lag-then-burst that made the measured EE snap 40–70 cm/step near
    #: the hub (86 % of UR10 reach). The λ² damping degrades gracefully at
    #: singularities (smooth slow-down instead of a wild swing). When the
    #: baked joint trajectory exists it is still used for the zero-residual
    #: replay path; DLS engages whenever a residual is active OR
    #: ``planner_dls_always`` is set.
    #: **Default OFF**: measured 2026-06-04 — DLS makes the carry beautifully
    #: smooth (mean 1.05 cm/step, zero >15 cm jumps) but the λ² damping cannot
    #: drive the tire into the 4 cm mount gate at 96 % reach, so it must not
    #: silently replace the IK path during training. Kept as an opt-in lever.
    use_dls_cartesian_servo: bool = False
    #: Apply DLS on every step (including zero residual). Default False so
    #: zero-action replay still plays the clean baked joint trajectory.
    planner_dls_always: bool = False
    #: DLS damping λ (rad·m⁻¹ scale). Larger = smoother / more singularity-
    #: robust but more EE tracking error. 0.06 balances hub reach vs swing.
    planner_dls_damping: float = 0.06
    #: Per-control-step joint change cap (rad) inside the DLS servo. Bounds
    #: the resolved-rate command like a joint speed limit. 0.10 ≈ 2 rad/s
    #: at 20 Hz — fast enough to traverse the carry, slow enough to look
    #: like a real robot.
    planner_dls_max_joint_step: float = 0.10
    #: EE position / orientation servo gains (proportional). <1 eases the
    #: approach so the arm does not overshoot a fast-moving nominal target.
    planner_dls_pos_gain: float = 1.0
    planner_dls_orn_gain: float = 0.8
    #: **2026-06-04 — adaptive (manipulability-scheduled) DLS damping.**
    #: When True the damping is 0 away from singularities (exact tracking,
    #: no stall) and ramps up to ``planner_dls_damping`` only as the
    #: manipulability ``w = √det(J Jᵀ)`` drops below
    #: ``planner_dls_manip_threshold`` — i.e. it activates *only* in the
    #: ill-conditioned hub-insertion region. This is what lets one DLS
    #: controller be both jump-free during the carry and still seat the
    #: tire (fixed damping had to trade one for the other).
    planner_dls_adaptive: bool = True
    planner_dls_manip_threshold: float = 0.02

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

    def __post_init__(self) -> None:
        layout = str(getattr(self, "scene_layout", "fanuc_spacious")).lower()
        if layout in ("fanuc_spacious", "fanuc", "spacious"):
            apply_fanuc_spacious_layout(self)


def apply_fanuc_spacious_layout(cfg: "EnvConfig") -> None:
    """Widen hub/cargo/cradle for FANUC reach; 100 kg tire physics.

    **2026-06-06 (HUB-origin frame)** — the whole layout is now expressed with
    the HUB CENTER at the WORLD ORIGIN (0,0,0). The previous frame had B (UR10e)
    at the origin and the hub at (−0.90, 0.20, 1.30); every absolute world
    coordinate below was rigidly translated by Δ = (+0.90, −0.20, −1.30) so the
    hub lands on the origin. Because it is a pure rigid translation the physical
    scene — every reach, clearance, the pit, the grasp/carry/mount geometry, and
    every runtime-computed planner trajectory (all derived from these landmarks)
    — is byte-for-byte unchanged; only the frame origin moved. Following the
    established re-anchoring convention, ``obs_reference_pos`` stays numerically
    at (0,0,0), so it is now coincident with the hub and observations are
    hub-centric (positional obs channels shift by Δ vs the old B-origin frame —
    retrain / re-evaluate checkpoints accordingly).

    Post-translation landmark cheat-sheet (world coords, hub = origin):
        hub          (0.00,  0.00,  0.00)
        Robot B base (0.90, -0.80, -0.30)   UR10e + 0.30 m nut tool
        Robot A base (-0.40, -1.50, -2.20)  FANUC (buried in pit)
        pickup       (-2.20, -1.50, -0.30)
        vehicle      (0.00,  0.25,  0.56)
        floor_z      -1.30
    """
    cfg.scene_layout = "fanuc_spacious"
    cfg.robot_a_kind = "fanuc_r2000ic"
    cfg.robot_b_kind = "ur10e"
    cfg.fanuc_urdf = str(URDF_DIR / "fanuc_r2000ic" / "r2000ic210f_wheeltool.urdf")
    cfg.fanuc_ee_link_name = "wheel_tool_tip"
    cfg.ur10_lock_tool_up = False
    cfg.fanuc_lock_tool_up = False
    #: Folded palm-up ready pose. Defines ``arm.rest`` (the IK warm-start branch
    #: every baked trajectory uses) and the palm-up ``FINAL_LOCK_QUATERNION`` for
    #: grasp/mount — KEEP this canonical so trajectories/grasp stay valid.
    cfg.fanuc_home_pose = (0.381, -1.05, -1.522, 0.0, 2.066, -1.189)
    #: **2026-06-05 (decoupled physical reset pose)** — the canonical home above
    #: folds the wrist (link_4/5) out toward +x where, with the narrow column-only
    #: circular pit (R=0.65), it dipped ~12–21 cm below the floor JUST OUTSIDE the
    #: pit edge → a 276 kN floor-plate contact tripped contact_force termination
    #: on every home-start reset (step 1). This lifts the whole arm over the pit
    #: centre (≈19.7 cm min floor clearance, 0 self-collision pairs) WITHOUT
    #: touching the IK seed / grasp orientation. Found via scripts/find_home_pose.py.
    cfg.fanuc_reset_pose = (1.7830, -0.9519, -0.2671, 2.0266, 2.1763, -0.7752)

    #: FANUC base BURIED to Z=−1.10. **2026-06-04 (mount-reach fix)** — at
    #: −0.95 the buried wrist could position the 6-o'clock pickup anchor (low
    #: target) but could NOT dip the gripper to the MOUNT pose under the hub
    #: with the required palm-up −90° yaw: the Stage-1 carry stalled with the
    #: EE 12 cm / 15° off the mount target (pose-IK infeasible), so the tire
    #: never seated on the hub. Burying 0.15 m deeper drops the whole reach
    #: band so the gripper reaches the under-hub mount pose at < 3 cm / 4°
    #: while the pickup grasp anchor still resolves to 0.0 cm. (Verified via
    #: scripts/probe_mount_reach.py across base-Z × hub Y-Z.)
    #: X stays at −1.10 (compact cell); still +1.10 m clear of Robot B.
    #: **2026-06-05 (un-bury for a narrower/shallower pit + lower rack)** — with
    #: the hub at 1.10 the base no longer needs the full −1.10 burial to reach
    #: the under-hub mount: −0.90 still resolves the mount pose at 0.1 cm/0.1°
    #: (−0.70 fails at 5.2 cm — scripts/check_reach_target.py). Un-burying 0.20 m
    #: raises the whole arm so (a) the deepest below-floor point rises −0.76 →
    #: −0.56 (pit depth 0.85 → 0.65) and the below-floor footprint shrinks to
    #: R≈0.40 (narrower circular pit), and (b) the forearm clears the floor at a
    #: LOWER pickup (1.00 instead of 1.34 → much lower rack).
    #: (B-origin frame: world (−1.10, 0.0, −0.90) − (0.20, 0.40, 0).)
    #: **2026-06-05 (B→origin parallel translation)** — whole scene shifted
    #: −0.90 m in Y so Robot B's base lands exactly on the world origin (was at
    #: y=0.90). This is a rigid translation of EVERY absolute landmark, so all
    #: relative geometry (reach, clearances, the just-fixed pit) is invariant;
    #: only the world frame is re-anchored to Robot B. (was y=−0.40.)
    #: HUB-origin frame: world (-1.30, -1.30, -0.90) + Δ(0.90, -0.20, -1.30).
    cfg.robot_A_base_pos = (-0.40, -1.50, -2.20)
    #: Ground plane at Z=0 — flush with Robot B's base (origin), so B stands
    #: directly on the floor with no stand pillar. The FANUC base (Z=−0.95)
    #: sits *below* the plane (intentionally buried for reach/height); since
    #: base_z < floor_z no support pillar is drawn for it either.
    #: HUB-origin frame: floor 0.0 + Δz(-1.30).
    cfg.floor_z = -1.30
    #: **2026-06-05 (real pit)** — carve a genuine rectangular hole into the
    #: floor around the FANUC base so the buried column lives in a pit while
    #: the surrounding floor (rim slabs, top at z=0) physically blocks any arm
    #: link from punching through outside the pit. Depth 0.95 m clears the
    #: deepest link (link_1 ≈ base_z+0.27 = −0.83). The XY rectangle encloses
    #: the whole below-floor arm sweep across pickup+carry (≈ x∈[−3.06,−0.30],
    #: y∈[−0.98,1.03]) plus margin, while keeping Robot B (origin) and the
    #: pit's +X rim edge (−0.15) safely apart. (scripts/diag_pit_extent.py)
    #: **2026-06-05 (circular column pit)** — per request the pit is a single
    #: CIRCLE over Robot A's column, not a big rectangle. This is only feasible
    #: because the pickup (→1.04) and hub (→1.10) are raised so the forearm
    #: (link_4/5) no longer dips below the floor while reaching the pickup or
    #: the under-hub mount: with those raises EVERY below-floor arm point falls
    #: within ~0.62 m of (−1.15,−0.10) (measured via scripts/measure_pit_footprint
    #: .py). Radius must clear the FAT lower-arm (link_2) cross-section as it
    #: passes through the hole, not just the contact point: a sweep with
    #: scripts/verify_pit.py shows R=0.80 already gives 0 penetrations and R=0.90
    #: leaves the arm everywhere ≥5 cm clear of the wall, so 0.90 is used for a
    #: tracking-error margin. It still leaves ~0.13 m of solid floor between the
    #: hole rim and Robot B's pedestal at the origin. Depth 0.85 clears the
    #: deepest arm point (base pedestal ≈ −0.76). The solid floor outside the
    #: circle physically blocks any arm link from punching through.
    cfg.floor_pit_enable = True
    cfg.floor_pit_shape = "circle"
    #: **2026-06-05 (pit-rim collision fix)** — depth/radius had been shrunk to
    #: 0.65/0.65 which left link_2 (J2 shoulder) jammed against the pit-mouth
    #: rim at z=0 (peakF≈1484 N, scripts/diag_pit_link.py). The base column
    #: sits at z=−0.90, so the pit must be at least that deep; depth 0.95
    #: fully buries the column and clears link_1 (≈−0.83). Radius 0.90 clears
    #: the fat lower-arm (link_2) cross-section with ≥5 cm wall margin
    #: (scripts/verify_pit.py) and still leaves solid floor to Robot B's
    #: pedestal at the origin (+X edge at −0.45).
    cfg.floor_pit_depth = 0.95
    #: B→origin translation: y −0.35 → −1.25 (was −0.35).
    #: HUB-origin frame: (-1.35, -1.25) + Δxy(0.90, -0.20).
    cfg.floor_pit_center = (-0.45, -1.45)
    cfg.floor_pit_radius = 0.90
    #: Hub at Y=0.70, Z=0.85. **2026-06-04 (raise + pull-in for mount)** —
    #: raising the hub so a grasped tire can actually be inserted requires
    #: BOTH (a) the gripper to dip under the hub to (0, hub_y, hub_z − R), and
    #: (b) Robot B at the origin to still reach the bolts. These conflict at
    #: hub_y=0.95: any hub_z ≥ 0.85 pushes B past full stretch (≥ 103 %). The
    #: resolution is to pull the hub IN to Y=0.70 (bolts closer to B) and lift
    #: to Z=0.85: B's farthest bolt ≈ 1.20 m ≈ 92 % reach (comfortable, with
    #: orientation freedom to drive the nut) while the FANUC (base now −1.10)
    #: reaches the under-hub mount pose at ≈ 3 cm / 4°. A mounted tire's lowest
    #: point sits at 0.85 − 0.525 = 0.325 m (32.5 cm floor clearance) so it no
    #: longer clips the ground. (scripts/probe_mount_reach.py joint Y-Z sweep.)
    #: **2026-06-04 (−X carry clearance)** — hub (and the vehicle that tracks
    #: it) are pulled to X=−0.30. With Robot A and the pickup FIXED, this
    #: changes the carry path to the relocated hub and roughly doubles the
    #: tire↔vehicle clearance along the ideal baked carry (3.6 → 8.8 cm) — the
    #: 3.6 cm baseline was far too tight to survive the heavy-tire tracking
    #: error, so the carry rammed the vehicle. Shifting in X (not Y) is cheap
    #: for Robot B at the origin: hub-to-origin distance is dominated by Y/Z,
    #: so B's farthest-bolt reach moves only 92 → 95 %, whereas a +Y shift of
    #: the same size would push B past full stretch. (probe_carry_clearance.py)
    #: NOTE: a second, now-dominant blocker remains — the 100 kg tire exceeds
    #: what the torque/velocity-limited FANUC can track on the baked carry, so
    #: even with vehicle collision disabled the zero-action carry stalls ~48 cm
    #: short of the hub. That is a control/tuning + policy concern, not layout.
    #: **2026-06-04 (user-forced +Y)** — hub also pushed +Y to 0.85 and −X to
    #: −0.50 per explicit request for more carry clearance (≈ 12.6 cm). WARNING:
    #: this BREAKS the "Robot B reaches every bolt" constraint — at hub_y=0.85
    #: the farthest bolt is ≈ 107 % of B's 1.30 m reach (B can no longer drive
    #: those nuts from the origin). Kept because the user prioritised carry
    #: clearance over B's bolt reach for this iteration. (probe_carry_clearance)
    #: **2026-06-05 (user-requested further raise)** — hub lifted 0.85 → 0.95.
    #: This RAISES the mounted-tire floor clearance to 0.95 − 0.525 = 0.425 m
    #: and lifts the under-hub mount target, which (because the FANUC base is
    #: buried for that very reach) gives a little headroom to UN-bury the base
    #: later for a shallower pit. NOTE: it does NOT reduce the current pit DEPTH
    #: — the deepest arm point is link_1 ≈ base_z+0.27, fixed by base burial,
    #: not by hub height. It also pushes Robot B further past full stretch.
    #: **2026-06-05 (circular-pit rework)** — hub raised 0.95 → 1.10. Raising
    #: the under-hub mount target lifts the carrying forearm so link_4/5 stay
    #: ABOVE the floor during the carry/mount (a precondition for the column-
    #: only circular pit). FANUC pickup+mount IK both resolve to 0.0 cm at this
    #: hub Z with the buried base (scripts/check_reach_target.py). Cost: Robot B
    #: at the origin now sits at ~121 % of full stretch for the farthest bolt
    #: (cannot drive those nuts) — acceptable for Phase 1 (B frozen at HOME);
    #: B reach must be revisited before Phase ≥2 cooperative nut-driving.
    #: **2026-06-05 (lower hub+cargo −z)** — hub lowered 1.10 → 1.00 per request
    #: to drop the whole hub/cargo set. 1.00 is the floor: FANUC mount IK still
    #: resolves (1.9 cm / 2°) but 0.90 fails (5.2 cm) at base_z=−0.90. The lower
    #: mount target (0.475) keeps the below-floor geometry column-only (R≈0.41 <
    #: pit R=0.65 — scripts/measure_pit_footprint.py), and the mounted tire
    #: still clears the floor by 0.475 m.
    #: **2026-06-05 (+Y corridor widen, B↔hub spacing fixed)** — hub/cargo/
    #: bolts/UR10e base translated +0.90 m along +Y to open the carry/mount
    #: corridor (FANUC arm links were jamming cargo+hub_mount at the old
    #: y=0.45 hub). Robot A + tire rack stay fixed; ``obs_reference_pos``
    #: stays at the world origin so policy obs still treat B as the origin
    #: frame. UR10e bolt reach is unchanged because B moves with the hub
    #: (scripts/sweep_hub_y.py: dY=0.90 → SEATED, worst-bolt 0.1 cm).
    #: (B-origin frame: world (−0.50, 1.75, 1.00) − (0.20, 0.40, 0).)
    #: B→origin translation: hub y 1.35 → 0.45 (was 1.35).
    #: **2026-06-06 (raised hub + nudge for palm-up mount)** — hub lifted
    #: z 1.00 → 1.30 and pulled toward FANUC (x −0.70 → −0.90, y 0.45 → 0.20)
    #: so palm-up mount IK resolves at 86 % reach / 1.5 cm (was 100 % reach /
    #: 64 cm miss at the old pose). Cargo tracks the same Δ=(−0.20, −0.25, +0.30).
    #: Robot B moves to (−0.90, −0.30, 0.80) on a pedestal so all 10 bolts
    #: reach at 64 % (was unreachable from the origin). ``obs_reference_pos``
    #: stays at the world origin. (scripts/sweep_layout_feasibility.py,
    #: scripts/validate_candidate_layout.py.)
    #: HUB-origin frame: the hub IS the world origin now (was (-0.90,0.20,1.30)).
    cfg.hub_pos_nominal = (0.0, 0.0, 0.0)
    cfg.tire_mount_pos = cfg.hub_pos_nominal
    #: Tire / rack at X=−2.90 — a compact ~1.8 m from the (pulled-in) FANUC
    #: base. Far enough to stay out of the inner reach deadzone (baked palm-up
    #: grasp error 6.5 cm < 8 cm approach gate) yet much closer than the 3.80 m
    #: full-stretch placement. RAISED by +0.45 (pickup 0.3913 → 0.8413) so the
    #: big tire (R=0.525) sits in the locked-tool-up wrist's reachable band.
    #: **2026-06-05 (circular-pit rework)** — pickup RAISED 0.8413 → 1.04. At
    #: the old height the reaching forearm (link_4/5) dipped ~13–15 cm below the
    #: floor out at the pickup (x≈−2.9), far from the column, which a column-
    #: only pit could not contain. Raising the pickup lifts that whole reach
    #: above the floor (link_4/5 below-floor depth → ~0; verified via
    #: scripts/measure_pit_footprint.py) so the only remaining below-floor
    #: geometry is the buried column cluster. Grasp IK still resolves to 0.0 cm.
    #: NOTE: with the base un-buried to −0.90 the forearm clears the floor at a
    #: pickup of 1.00 (at the old −1.10 burial it needed 1.34), so the rack can
    #: sit much lower while the below-floor geometry stays confined to the
    #: column. (scripts/measure_pit_footprint.py base/pickup sweep.)
    #: (B-origin frame: world (−2.90, 0.0, 1.00) − (0.20, 0.40, 0).)
    #: B→origin translation: pickup y −0.40 → −1.30 (was −0.40).
    #: HUB-origin frame: (-3.10, -1.30, 1.00) + Δ(0.90, -0.20, -1.30).
    cfg.tire_pickup_pos = (-2.20, -1.50, -0.30)
    #: Vehicle body tracks the hub (kept +0.25 m behind in Y, +0.56 m above in
    #: Z so the wheel-well cutout stays centred on the hub): hub
    #: (−0.50,0.85,0.85) → vehicle (−0.50, 1.10, 1.41).
    #: (B-origin frame: world (−0.50, 2.00, 1.56) − (0.20, 0.40, 0).)
    #: B→origin translation: vehicle y 1.60 → 0.70 (was 1.60).
    #: **2026-06-06** — tracks raised hub (+0.30 z, −0.25 y, −0.20 x).
    #: HUB-origin frame: (-0.90, 0.45, 1.86) + Δ(0.90, -0.20, -1.30).
    cfg.vehicle_center_world = (0.0, 0.25, 0.56)
    cfg.cargo_back_wall_y_offset = 0.35
    #: **2026-06-05 (align wall bottom with cargo bottom)** — the back wall
    #: (he_z=0.50) was centred at the hub (1.10) so its bottom hung at 0.60,
    #: i.e. 0.56 m BELOW the cargo box bottom (vehicle_center_z 1.66 − he_z 0.50
    #: = 1.16) — it looked like it was drooping below the cargo. Centring it at
    #: 1.66 puts its bottom flush with the cargo bottom (1.16) and its z-span
    #: exactly on the cargo box [1.16, 2.16]. It still overlaps the upper ~46 cm
    #: of a hub-mounted tire (top at 1.10+0.525=1.625) so it keeps blocking +Y
    #: over-push. **2026-06-05** — tracks the lowered cargo (vehicle z 1.56),
    #: so the wall bottom (1.56−0.50=1.06) stays flush with the cargo bottom.
    #: **2026-06-06** — tracks raised cargo centre (1.86).
    #: HUB-origin frame: 1.86 + Δz(-1.30).
    cfg.cargo_back_wall_center_z = 0.56
    #: Cradle rails stand on the Z=0 floor; their TOP corner must touch the
    #: tire tread at the rail's Y offset (the tire rests on the two rails at
    #: y=±0.40, NOT on its y=0 bottom). Seating contract:
    #:     rack_top = tire_com_z − √(R² − (inner_y − he_y)²)
    #:             = 0.8413 − √(0.525² − 0.35²) = 0.8413 − 0.3913 = 0.450
    #: With the rail standing on the floor (bottom = 0): he_z = top/2 = 0.225,
    #: center = he_z = 0.225.
    #: **2026-06-04 (open-corridor cradle — fixes the link_5↔rail block)**
    #: The full-height rail columns physically blocked the FANUC forearm
    #: (``link_5``) from dipping between the rails to the tire's 6 o'clock
    #: grasp point — the baked palm-up approach stalled ~25 cm short (Y off
    #: by 0.61 m) so the Stage-0 grasp never fired (pickup success ≈ 0).
    #: Fix: build the rails as thin top *bars* that touch the tire only at
    #: the seating height (top edge at z = 0.450 = the y=±0.35 tread contact),
    #: propped up from the floor by support posts on their OUTER faces. The
    #: inner-lower region is now open, so the forearm threads straight to the
    #: grasp anchor and the grasp fires (verified @ step ~89, home start).
    #: Bar: he=(0.10, 0.05, 0.025); top = center_z + he_z = 0.425 + 0.025 =
    #: 0.450. Posts auto-span floor (0) → bar bottom (0.40) on the outer side.
    #: **2026-06-05 (pickup 1.00)** — rails seat the tire COM at z=1.00.
    #: rack_top = 1.00 − √(0.525² − 0.35²) = 1.00 − 0.3913 = 0.6087;
    #: bar center_z = 0.6087 − 0.025 = 0.584. Posts span the solid floor (0) up
    #: to the bar bottom (0.559).
    cfg.tire_rack_half_extents = (0.10, 0.05, 0.025)
    #: (B-origin frame: world (−2.90, ±0.40, 0.584) − (0.20, 0.40, 0).)
    #: B→origin translation: rack y {0.00,−0.80} → {−0.90,−1.70}.
    #: HUB-origin frame: rack centers + Δ(0.90, -0.20, -1.30).
    cfg.tire_rack_inner_center = (-2.20, -1.10, -0.716)
    cfg.tire_rack_outer_center = (-2.20, -1.90, -0.716)
    cfg.tire_rack_support_posts = True
    cfg.tire_rack_post_half_extents_xy = (0.10, 0.05)

    #: **2026-06-04 (carry arch vs vehicle)** — the Stage-1 carry goes from the
    #: pickup (z≈0.86) DOWN to the under-hub mount (z=0.325); the default 0.35 m
    #: up-arch lofted the grasped tire to z≈1.13 and rammed it into the raised
    #: vehicle underbody (bottom at z=0.91), stalling the carry. A small 0.10 m
    #: arch keeps the tire clear of Robot B while staying below the vehicle.
    cfg.planner_stage1_lift = 0.10

    #: **2026-06-05 (coaxial −Y insertion for FANUC mount)** — with the
    #: heavy tire rigidly grasped (JOINT_FIXED, see kinematic_tire_lock_stages
    #: = (0,) below), the default arch carry swings the tire toward the hub
    #: *laterally* (from −X), so the tire's outer rim (R=0.525) catches the
    #: hub_mount base (radius 0.21) and the carry jams ~0.75 m short — the
    #: tire never reaches a mountable pose (prior runs: ~0 % success). A
    #: rigid-transform collision sweep (scripts/diag_s1_stall.py) confirms
    #: that routing through a pre-hub via-point 0.70 m along −Y makes the
    #: final approach a coaxial straight +Y insertion that is collision-free
    #: through the wheel well (the hub pilot/peg, r=0.21, then fits the tire
    #: bore, r=0.282). This gives the PPO residual a feasible nominal to
    #: refine instead of one that drives the tire into the truck structure.
    cfg.planner_stage1_approach_standoff = 0.70

    #: Robot B (UR10e) sits at the WORLD ORIGIN and reaches all hub bolts (see
    #: hub placement above). Phase 1 still freezes B at HOME; the obs frame is
    #: pinned to the origin (below), which now coincides with B's base.
    #: **2026-06-05 (B shifted +Y to reach all bolts)** — at the origin the
    #: UR10e could not reach the farthest hub bolt (worst-bolt IK 7.2 cm). With
    #: the hub at (−0.5, 0.85, 1.10) a Y-sweep (scripts/sweep_robotB_y.py) shows
    #: every bolt becomes reachable for base y ∈ [0.20, 1.40]; y=0.60 is the
    #: most comfortable (worst-bolt IK ≈ 0.7 mm). The observation frame stays
    #: pinned to the world origin so positional channels are unaffected.
    #: **2026-06-05 (push B further +x/−y for tire clearance)** — B moved to
    #: (0.20, 0.40): the furthest +x/−y from the hub that still reaches every
    #: bolt (worst-bolt IK ≈ 0.9 mm; Bx=0.40 misses by 13 cm). This pulls B off
    #: the carry/mount corridor so the carried tire clears it with more margin.
    #: (scripts/sweep_b_and_hubz.py.)
    #: UR10e base follows the hub +0.90 m (+Y) so B↔bolt geometry is invariant.
    #: ``obs_reference_pos`` stays at the world origin — policy obs still use B
    #: as the reference frame even though B's sim base is at y=0.90.
    #: **2026-06-05 (B→origin)** — Robot B's base now sits exactly on the world
    #: origin (was y=0.90). The whole scene was translated −0.90 m in Y so B is
    #: the physical origin AND the observation reference, removing the prior
    #: split where the obs frame (origin) and B's sim base (y=0.90) disagreed.
    #: **2026-06-06** — B stays at the origin XY but rises onto a 1.0 m
    #: pedestal. With the raised+pulled-in hub (−0.90, 0.20, 1.30) all 10 bolts
    #: now resolve at 0.0 cm IK / 87 % reach from (0, 0, 1.0); moving B in −Y
    #: only WORSENS reach (hub is at +Y 0.20). ``obs_reference_pos`` stays at
    #: the world origin so the policy frame is unchanged.
    #: (scripts/validate_candidate_layout.py x=0 −y/z sweep.)
    #: **2026-06-08 (v6 nut layout — shorter tool + raised B)** — nut-runner
    #: shortened to ``bolt_length`` (10 cm) so B's forearm stays higher on the
    #: far arc; base Z raised to hub centre (0.0) and shifted +Y by the 20 cm
    #: tool reduction (−0.95 → −0.75) to preserve reach. Clears A↔B corridor
    #: on bolts 0–3 vs the v5 layout; bolts 4–5 remain tight (6-o'clock A
    #: grip vs 6-o'clock bolts). (scripts/diag_b_a_clearance.py)
    cfg.ur10e_urdf = str(
        URDF_DIR / "ur10e_robot" / "ur10e_with_nut_tool_10cm.urdf"
    )
    cfg.ur10e_nut_tool_length = 0.10
    cfg.robot_B_base_pos = (0.90, -0.75, 0.0)
    #: Observation reference stays numerically at the world origin — which is
    #: now the HUB. Positional obs channels are therefore hub-centric.
    cfg.obs_reference_pos = (0.0, 0.0, 0.0)

    cfg.tire_mass = cfg.tire_mass_heavy
    cfg.tire_inertia_diagonal = cfg.tire_inertia_heavy
    cfg.physics_num_sub_steps = 12
    cfg.contact_erp = 0.2
    cfg.contact_cfm = 2e-5
    cfg.contact_force_terminate_above = 50_000.0
    cfg.fanuc_position_gain = 0.85
    cfg.fanuc_velocity_gain = 1.15
    #: Keep the FANUC gripper palm-up (tool +Z = world +Z) at all times; the
    #: heavy-tire droop at the mount reach would otherwise tilt it ~15°.
    cfg.fanuc_enforce_palm_up_post_step = True
    cfg.fanuc_palm_up_tool_z_threshold = 0.9998
    #: **2026-06-05 (carry tracking)** — 0.8 rad/s left the 100 kg tire
    #: lagging the baked Min-Jerk by ~0.7 m (``ik_residual_A_mean``) while the
    #: traj index advanced every step; raise the cap so the arm can keep up.
    cfg.fanuc_motor_max_velocity_rad_s = 1.4
    cfg.fanuc_torque_scale = 1.5
    #: Finer baked trajectory (200 vs 100) → smaller per-step joint deltas.
    cfg.planner_traj_steps = 200
    #: Wait until the measured EE is near the nominal waypoint before advancing
    #: the traj index (prevents the index from outrunning the heavy load).
    #: Baked joint traj advances one waypoint/step; the EE-arrival gate throttled
    #: the index to ~1 step per 15 controls (arm lagging nominal by >6 cm) which
    #: made carry painfully slow and never reached the +Y insertion leg within
    #: 600 steps (diag_train_mount: idx 61/200 at timeout). Disable for FANUC.
    cfg.planner_waypoint_gate_enable = False
    cfg.planner_waypoint_pos_tol_m = 0.06
    cfg.planner_waypoint_max_stall = 15
    #: **2026-06-06 (kinematic carry restored for palm-up + reach)** — the
    #: dynamic JOINT_FIXED carry (klock=(0,)) made the stiff position PD fight
    #: the 100 kg tire: the arm saturated ~51 cm short of the mount target and
    #: the gripper drooped ~15° off palm-up at the insertion reach (a steady
    #: state that neither extra wrist torque nor the post-step palm-up re-lock
    #: could fix cleanly — reorienting the gripper pivots the tire hanging
    #: ~0.8 m off the wrist by ~20 cm). Re-writing the grasped tire pose every
    #: step (kinematic lock on all carry/mount/return stages) removes that
    #: load from the PD, so the arm tracks the palm-up baked joints to the
    #: mount target (end err 51 cm → 0.1 cm) and the gripper stays palm-up
    #: (worst tilt 15.6° → 5.6°, → 3.4° with the post-step re-lock below).
    #: ``kinematic_tire_sync_alpha`` (0.65) smooths the per-step teleport.
    #: **2026-06-06 (remount S4/S5)** — extended to include the 6-stage
    #: cycle's demount (S4) and return (S5) carry legs. Previously the
    #: regrip created a JOINT_FIXED bond and S4/S5 were absent here, so the
    #: arm had to *dynamically drag* the 100 kg tire off the hub against the
    #: mount-seating penetration contact (~98 kN) — the position PD saturated
    #: and the arm was pinned, so the demount never fired. Kinematic carry
    #: (teleport the tire to EE+offset each step, as S1–S3 already do) avoids
    #: both the dynamic-drag torque limit and the penetration jam. The
    #: 4-stage FSM never reaches stage 4/5 so the extra entries are inert.
    cfg.kinematic_tire_lock_stages = (0, 1, 2, 3, 4, 5)

    #: **2026-06-05 (mount gate matched to carry tracking reach)** — with
    #: the 100 kg tire the zero-residual baked carry tracks to ~0.74 m of
    #: the hub before the position-controlled arm saturates on the fine
    #: insertion segment (the baked *nominal* reaches 0.07 m, but the live
    #: arm lags). The old 0.30 m soft mount gate is therefore unreachable
    #: from the carry start, so R_mount never fired in 2 M steps
    #: (success_rate stuck at 0). Soft gate must be **narrower than**
    #: ``planner_stage1_approach_standoff`` (0.70 m): at 0.85 m the FSM
    #: fired mount at the standoff via-point before the +Y insertion leg.
    #: 0.55 m forces the coaxial insertion segment to run first.
    cfg.mount_radius_tol_soft = 0.55
    cfg.mount_angle_tol_soft_rad = np.deg2rad(45.0)
    cfg.mount_tol_ramp_steps = 1_500_000
    #: Per-step residual authority. **2026-06-06: 0.20 → 0.12.** The zero-
    #: residual baked nominal already mounts at d=0 / theta=0 (attached hot-
    #: start probe), so a wide ±0.20 m residual under std≈0.78 injected ~0.16 m
    #: of exploration noise on top of an already-correct path — overshooting
    #: the 0.04 m mount gate and crushing the *rollout* success_rate (eval
    #: stayed healthy). 0.12 m keeps enough authority for fine correction /
    #: hard-pickup recovery while halving the noise the tight gate sees.
    #: Paired with ``--log-std-init -0.5`` in run_phase1_pipeline.sh.
    #: **2026-06-07: 0.12 → 0.03.** Diagnostics showed the trained policy was
    #: using near-saturated residuals (~5 cm lateral X offset at scale 0.05)
    #: that pulled the tire off-centre during the carry/insertion — the
    #: "tire X ≠ hub-centre X during mount" the operator observed — while the
    #: baked nominal alone seats the tire to 0.4 cm. With orientation fully
    #: planner-locked and the end pose now derived from the real grasp
    #: transform (tire centre lands exactly on ``tire_mount_pos``), a large
    #: residual only adds off-axis noise. 0.03 m keeps a small fine-correction
    #: budget while letting the smooth nominal dominate the insertion so the
    #: tire tracks the hub centre coaxially the whole way in.
    cfg.planner_pos_offset_scale = 0.03
    cfg.obs.workspace_radius = 3.0
    cfg.grasp_com_offset_world = (0.0, 0.0, 1.0)


# ----------------------------------------------------------------------
# Stage helper (spec §4.3): incrementally enable reward terms.
# ----------------------------------------------------------------------
def make_reward_config(stage: int, phase: int = 1) -> RewardConfig:
    """RewardConfig with weights gated to a given training stage / phase.

    Stage 1 — per-robot tasks only (align_A + reach_B).
    Stage 2 — + cooperation + UR10 cooperative sync penalty.
    Stage 3 — + success bonus + action/jerk penalties (full dense).
    Collision + workspace penalties are active in EVERY stage as of
    2026-06-06 (see the note below; previously zeroed in stages 1/2).
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
    # **2026-06-06 (collision/workspace weights restored in stages 1/2)** —
    # these used to be zeroed because, when ``collision_terminates=True``, a
    # negative collision penalty plus collision-termination let the policy
    # *escape* the negative dense baseline by deliberately crashing (the
    # "self-termination" exploit). With ``collision_terminates=False`` (default
    # now) a collision never ends the episode, so the penalty can no longer be
    # gamed for self-termination — it only shapes the policy AWAY from hitting
    # geometry. Keeping the default ``w_collision``/``w_workspace`` therefore
    # gives the warmup stages real collision awareness with no downside.
    # (Production pipeline uses stage 3, which already had these on.)
    if stage == 1:
        rc.w_c = 0.0
        rc.w_sync_joint_a = 0.0
        rc.R_success = 0.0
        rc.w_action = 0.0
        rc.w_jerk = 0.0
        rc.mix_sparse_success = 0.0
        rc.mix_dense = 1.0
    elif stage == 2:
        rc.R_success = 0.0
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
    # Snapshot the caller's explicit overrides so they remain authoritative
    # over the layout presets below. ``scene_layout`` is excluded (re-applying
    # it would just re-select the same layout, harmless, but kept for clarity).
    user_overrides = {k: v for k, v in overrides.items() if k != "scene_layout"}
    cfg.reward = make_reward_config(stage, phase)
    cfg.use_shaping = (stage == 4)
    cfg.curriculum.phase = phase
    # Stages 1–2 omit collision/contact *penalties*; Bullet still reports large
    # normal forces from tire–EE fixed constraints → avoid instant episode death.
    if stage <= 2 and not cf_user_set:
        cfg.contact_force_terminate_above = 0.0
    # Phase 1 = Robot A pick-and-place only. Freeze Robot B at HOME unless the
    # caller explicitly opted out via ``freeze_robot_b=False``.
    if phase == 1 and not freeze_b_user_set:
        cfg.freeze_robot_b = True

    # Robot-B nut-fastening task: Robot B is the policy-controlled arm
    # (13-d action / full obs) and Robot A is a static fixture. Force the
    # un-frozen regime regardless of phase (must precede the action/obs dim
    # computation below). The contact-force termination is disabled further
    # down, *after* the layout preset runs (it would otherwise reset the
    # gate to 50 kN).
    nut_task_cfg = bool(getattr(cfg, "nut_fastening_task", False))
    if nut_task_cfg:
        cfg.freeze_robot_b = False

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
    # Nut-fastening task: append a 7-d task block (eeB→staging vector [3],
    # bolt axis unit [3], alignment θ [1]) so the policy gets the approach
    # target + insertion direction directly instead of inferring them from
    # the bolt-centre vector + quaternion. ``_compute_obs`` appends the same
    # block under ``nut_fastening_task``.
    if nut_task_cfg:
        cfg.obs.dim += int(getattr(cfg, "nut_obs_extra_dim", 7))
    if legacy_obs_dim is not None:
        cfg.obs.dim = int(legacy_obs_dim)
    layout = str(getattr(cfg, "scene_layout", "fanuc_spacious")).lower()
    if layout in ("fanuc_spacious", "fanuc", "spacious"):
        apply_fanuc_spacious_layout(cfg)
        # **2026-06-05 (override-precedence fix)** — the layout function sets a
        # whole batch of presets (incl. ``contact_force_terminate_above`` = 50 kN,
        # robot poses, torque, gains). Previously it ran AFTER ``EnvConfig(**
        # overrides)`` and silently CLOBBERED explicit CLI flags — e.g. train.ps1's
        # ``--contact-force-done 0`` was overwritten back to 50 kN. Re-apply the
        # caller's explicit overrides so CLI flags always win over layout presets.
        for k, v in user_overrides.items():
            setattr(cfg, k, v)
    # Nut-fastening: disable the contact-force termination after the layout
    # preset (which sets it to 50 kN) unless the caller set it explicitly —
    # the nut-runner tool legitimately seats against the studs / wheel face.
    if nut_task_cfg and not cf_user_set:
        cfg.contact_force_terminate_above = 0.0
    # The nut task is intimate-contact work: the socket seats on the studs at
    # the hub face while Robot A holds the tire right there. A glancing
    # socket↔tire / socket↔A contact is expected and must NOT kill the
    # episode (the forced insert macro deliberately drives the socket to the
    # hub-face base). Keep the per-step collision *penalty* but disable
    # collision *termination* so the policy can learn to work in close.
    if nut_task_cfg and "collision_terminates" not in user_overrides:
        cfg.collision_terminates = False
    return cfg
