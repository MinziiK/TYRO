"""TyroEnv — Gymnasium env implementing the Phase 1 FSM cycle.

Phase 1 task: Robot A (UR10) picks up a tire from a floor pickup zone next to
its base, transports it to the truck hub, mounts it coaxially, then returns
the tire back to the floor pickup zone for a soft landing. Robot B (Panda)
concurrently reaches and aligns its tool +Z with the target bolt.

**2026-06-01 — Hybrid control rewrite.** PPO no longer drives raw Δ-EE-pose.
Instead a Minimum-Jerk planner generates a nominal EE trajectory per FSM
stage and the policy outputs a small per-step *residual offset* (XYZ only
by default; rotation handled by SLERP). See ``cfg.use_planner_residual``,
``_generate_nominal_trajectory``, and ``_apply_action``.

The FSM tracks three task stages:
  * Stage 0 (approach/pick)  — tire pinned to the floor; reward shapes the
                               UR10 EE toward the tire's 6 o'clock outer
                               point. A successful approach within
                               ``approach_radius_tol`` swaps the world pin
                               for a JOINT_FIXED grasp and emits ``R_pickup``.
  * Stage 1 (carry/mount)    — tire bonded to UR10 EE; reward shapes tire
                               COM toward the hub centre. A mount within
                               ``mount_radius_tol`` emits ``R_mount`` and
                               advances to Stage 2.
  * Stage 2 (return/place)   — tire still grasped; reward shapes tire COM
                               back toward the original pickup pose plus a
                               soft-landing term on |v_z|. On landing the
                               grasp releases, the world pin re-engages, and
                               the episode succeeds with ``R_return``.

Always-on penalties (collision / workspace / action / jerk / vertical) apply
regardless of stage. Vertical pose violations beyond ``vertical_tol_rad``
trigger an immediate penalty termination.

Action: 13-d in [-1, 1]  → (Δpose_A 6, Δpose_B 6, gripper_A 1)
        Phase 1 collapses to 6-d (Δpose_A only) — gripper_A is a sim-side
        no-op under the auto-grasp constraint, Panda is frozen at HOME.
Observation: 89-d (spec §2.1 base + 3-d hub–tire mating diagnostics).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from scipy.spatial.transform import Rotation, Slerp

from ..config import EnvConfig
from . import rewards
from .robots import PandaRobot, Robot, make_robot_a, make_robot_b, robot_a_lock_quaternion
from .scene import Scene, SceneHandles
from .utils import (
    axisangle3_to_quat,
    quat_axis,
    quat_multiply,
    relative_axisangle,
)


# ----------------------------------------------------------------------
# **2026-06-01 — Min-Jerk planner + SLERP helpers (module-level).**
# ----------------------------------------------------------------------
def _min_jerk_positions(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """5th-order minimum-jerk position profile.

    Returns ``(n, 3)`` array of XYZ poses sampled at ``n`` equal time
    fractions on [0, 1]. The position interpolant is
    ``p(s) = start + s_curve(t) * (end - start)`` with
    ``s_curve(t) = 10 t³ − 15 t⁴ + 6 t⁵`` — the standard min-jerk
    profile satisfying p(0)=start, p(1)=end, p′(0)=p′(1)=p″(0)=p″(1)=0.
    """
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    n = max(int(n), 1)
    if n == 1:
        return end[None, :].copy()
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    s = 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5
    return start[None, :] + s[:, None] * (end - start)[None, :]


def _multi_min_jerk_positions(waypoints, n: int) -> np.ndarray:
    """Chain min-jerk segments through a list of ``(K, 3)`` waypoints.

    Steps are distributed across the ``K-1`` segments proportionally to
    Euclidean segment length (each segment gets >= 2 samples) and the
    shared segment endpoints are de-duplicated so the result is exactly
    ``(n, 3)``. Used to route the Stage-1 carry through an arch apex *and*
    a −Y insertion standoff so the tire threads into the wheel well along
    the hub axis instead of being dropped onto the cargo from above.
    """
    pts = [np.asarray(w, dtype=np.float64).reshape(3) for w in waypoints]
    n = max(int(n), 2)
    if len(pts) < 2:
        return pts[-1][None, :].copy()
    if len(pts) == 2:
        return _min_jerk_positions(pts[0], pts[1], n)
    seg_len = np.array([np.linalg.norm(pts[i + 1] - pts[i])
                        for i in range(len(pts) - 1)], dtype=np.float64)
    total = float(seg_len.sum())
    n_seg = len(seg_len)
    if total < 1e-9:
        alloc = np.full(n_seg, max(2, n // n_seg), dtype=int)
    else:
        alloc = np.maximum(2, np.round(seg_len / total * n).astype(int))
    segs = []
    for i in range(n_seg):
        s = _min_jerk_positions(pts[i], pts[i + 1], int(alloc[i]))
        segs.append(s if i == 0 else s[1:])  # drop shared endpoint
    traj = np.vstack(segs)
    if traj.shape[0] > n:
        # subsample preserving the final endpoint
        idx = np.linspace(0, traj.shape[0] - 1, n).round().astype(int)
        traj = traj[idx]
    elif traj.shape[0] < n:
        pad = np.repeat(traj[-1:], n - traj.shape[0], axis=0)
        traj = np.vstack([traj, pad])
    traj[-1] = pts[-1]
    return traj


def _slerp_quats(q_start, q_end, n: int,
                 times: Optional[np.ndarray] = None) -> np.ndarray:
    """Spherical-linear interpolation between two xyzw quaternions.

    Returns ``(n, 4)`` array of quaternions in PyBullet xyzw order.
    Inputs are normalised defensively; ``scipy.spatial.transform.Slerp``
    handles the shortest-path sign correction internally.

    ``times`` (optional, shape ``(n,)``) is a non-decreasing sequence in
    [0, 1] sampled along the slerp arc. ``None`` falls back to a uniform
    linear schedule. Used by ``_generate_nominal_trajectory`` for
    front-loaded yaw schedules (D4 fix, 2026-06-02).
    """
    q_start = np.asarray(q_start, dtype=np.float64).reshape(4)
    q_end = np.asarray(q_end, dtype=np.float64).reshape(4)
    n_a = float(np.linalg.norm(q_start))
    n_b = float(np.linalg.norm(q_end))
    if n_a > 1e-12:
        q_start = q_start / n_a
    else:
        q_start = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if n_b > 1e-12:
        q_end = q_end / n_b
    else:
        q_end = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    n = max(int(n), 1)
    if n == 1:
        return q_end[None, :].copy()

    # Rotation.from_quat / Slerp accept xyzw (scalar_first=False default).
    key_rots = Rotation.from_quat(np.stack([q_start, q_end], axis=0))
    slerp = Slerp([0.0, 1.0], key_rots)
    if times is None:
        times = np.linspace(0.0, 1.0, n, dtype=np.float64)
    else:
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        # Clamp into [0, 1] defensively so Slerp does not raise.
        times = np.clip(times, 0.0, 1.0)
        if times.shape[0] != n:
            # Pad/truncate to match n exactly.
            if times.shape[0] > n:
                times = times[:n]
            else:
                times = np.concatenate([
                    times,
                    np.full(n - times.shape[0], 1.0, dtype=np.float64),
                ])
    out = slerp(times).as_quat()
    return np.asarray(out, dtype=np.float64).reshape(n, 4)


def _quat_align_z_to(target_dir) -> np.ndarray:
    """Quaternion (xyzw) that rotates body +Z to world ``target_dir``.

    Used for building "tire bore aligned with hub axis" orientations.
    Returns identity / 180-about-X for the parallel / anti-parallel
    degeneracies so callers never see a NaN.
    """
    z_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    target = np.asarray(target_dir, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(target))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target = target / n
    v = np.cross(z_ref, target)
    c = float(np.dot(z_ref, target))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 180° about X
    axis = v / s
    half = 0.5 * float(np.arctan2(s, c))
    sh = float(np.sin(half))
    return np.array(
        [axis[0] * sh, axis[1] * sh, axis[2] * sh, float(np.cos(half))],
        dtype=np.float64,
    )


class TyroEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg: Optional[EnvConfig] = None,
                 render: bool = False, seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.cfg.render = render or self.cfg.render
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        self.client: int = -1
        self.scene: Optional[Scene] = None
        self.handles: Optional[SceneHandles] = None
        self.robot_A: Optional[Robot] = None
        self.robot_B: Optional[Robot] = None
        # Tire ↔ UR10 EE fixed joint (Stage 1/2). Recreated on each pickup.
        self._grasp_constraint: Optional[int] = None
        # Tire ↔ world fixed joint (Stage 0 and after final landing). Keeps
        # the standing-on-edge tire from tipping over when not grasped.
        self._world_pin: Optional[int] = None
        # Tire ↔ hub fixed joint (created at the mount event when
        # ``cfg.pin_tire_on_mount`` is on). Represents the tire being
        # physically seated on the hub so it stays put while Robot B
        # tightens the bolts.
        self._hub_mount_constraint: Optional[int] = None
        # Seated tire pose recorded at the mount event; re-applied each
        # hold step so the bonded tire visibly sits still on the hub.
        self._mount_seated_pos: Optional[np.ndarray] = None
        self._mount_seated_orn: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._prev_action: np.ndarray = np.zeros(self.cfg.action.dim, dtype=np.float32)
        self._prev_d_A: Optional[float] = None
        self._prev_d_B: Optional[float] = None
        # FSM potential-based shaping book-keeping. Per-stage distances
        # that drive ``w_pb_*`` shaping bonuses; reset to ``None`` at
        # every stage transition so the first sample after transition
        # doesn't dump a spurious one-shot bonus.
        self._prev_d_approach: Optional[float] = None
        self._prev_d_return: Optional[float] = None
        # FSM bookkeeping — see module docstring.
        self.task_stage: int = 0
        self._pickup_pos_world: np.ndarray = np.zeros(3, dtype=np.float64)
        self._vertical_quat: np.ndarray = np.zeros(4, dtype=np.float64)
        self._mount_bonus_paid: bool = False
        self._pickup_bonus_paid: bool = False
        # v6: Stage 2 demount bookkeeping (see also reset()).
        self._mount_done_step: Optional[int] = None
        self._demount_bonus_paid: bool = False

        # 2026-06-01 — Min-Jerk planner + residual control state.
        # Per-stage nominal trajectory (built on reset + each FSM
        # transition). ``current_traj_step`` indexes into both arrays
        # and is clamped to len-1 once the planner runs out — at that
        # point the policy holds the end pose with the residual offset.
        self._traj_pos: Optional[np.ndarray] = None      # (N, 3) world XYZ
        self._traj_quat: Optional[np.ndarray] = None     # (N, 4) world xyzw
        self._traj_q: Optional[np.ndarray] = None        # (N, 6) arm joints
        self.current_traj_step: int = 0
        # Waypoint arrival gate watchdog — counts control steps the index
        # has stalled at the current waypoint (see ``_advance_traj_index``).
        self._traj_stall: int = 0
        # T_ee_tire cached at the moment of the grasp constraint
        # creation. Lets us compute the EE world pose required to put
        # the tire at any desired world pose afterwards (Stage 1 mount
        # / Stage 2 demount / Stage 3 cradle return).
        self._grasp_t_ee_tire_pos: Optional[np.ndarray] = None
        self._grasp_t_ee_tire_quat: Optional[np.ndarray] = None
        # Kinematic upright lock (``lock_tire_upright_when_grasped``):
        # tire pose is re-written each step; no JOINT_FIXED.
        self._grasp_kinematic: bool = False
        self._grasp_yaw_ee0: Optional[float] = None
        self._grasp_com_offset_ee: Optional[np.ndarray] = None
        # **2026-06-02 (cargo penetration fix)** — last safe (no cargo / back-
        # wall penetration) tire COM and orientation under the kinematic
        # upright lock. When the desired pose at the next sync would push
        # the tire INTO the cargo / back-wall, the env reverts to this
        # cached pose so the tire visually stops at the wall instead of
        # phasing through it. ``_in_bad_collision`` then keeps firing the
        # per-step ``-w_collision`` penalty, so the policy is taught not
        # to drive into the wall in the first place.
        self._safe_tire_pos: Optional[np.ndarray] = None
        self._safe_tire_orn: Optional[np.ndarray] = None
        # Joint targets seeded by attached-hot-start; lets the first
        # planner step skip a redundant IK solve when action ≈ 0.
        self._planner_hold_arm_targets: Optional[np.ndarray] = None
        # Mount-and-hold (``cfg.mount_hold_steps``): freeze arm + pinned tire.
        self._mount_hold_left: int = 0
        self._mount_frozen_q: Optional[np.ndarray] = None
        self._mount_hold_finish_term: bool = False

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.cfg.action.dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.cfg.obs.dim,), dtype=np.float32)

        self._connect()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        if self.client >= 0:
            return
        mode = p.GUI if self.cfg.render else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        p.setTimeStep(1.0 / self.cfg.sim_freq_hz, physicsClientId=self.client)

    def close(self) -> None:
        if self.client >= 0:
            try:
                if p.isConnected(self.client):
                    p.disconnect(physicsClientId=self.client)
            except p.error:
                pass
            self.client = -1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        p.resetSimulation(physicsClientId=self.client)
        # resetSimulation() invalidates all body / constraint ids — clear cache.
        self._grasp_constraint = None
        self._grasp_kinematic = False
        self._grasp_yaw_ee0 = None
        self._grasp_com_offset_ee = None
        self._safe_tire_pos = None
        self._safe_tire_orn = None
        self._world_pin = None
        self._hub_mount_constraint = None
        self.task_stage = 0
        self._mount_bonus_paid = False
        self._pickup_bonus_paid = False
        # 2026-06-01 — clear the planner ``skip_ik`` hold state up front so
        # any stale joint vector from a previous episode cannot leak into
        # this one. ``_apply_attached_hot_start`` (run further down) sets
        # this back to a valid value when it fires.
        self._planner_hold_arm_targets = None
        self._mount_hold_left = 0
        self._mount_frozen_q = None
        self._mount_hold_finish_term = False
        # v6 (4-stage FSM) — also reset Stage-2 bookkeeping at the start
        # of every reset call so the demount stall counter never carries
        # state across episode boundaries (each new episode = fresh 20-step
        # stall budget once Stage 1 fires).
        self._mount_done_step = None
        self._demount_bonus_paid = False
        # 6-stage remount cycle bookkeeping (only consumed when
        # ``cfg.remount_cycle_enable``). S2 = empty retract to HOME after
        # the W1 tighten hold; S3 = re-approach + re-grip the hub tire;
        # S4 = W2 loosen hold then demount. The W1/W2 holds reuse the
        # ``_mount_hold_left`` freeze/clamp machinery.
        self._retract_bonus_paid = False
        self._regrip_bonus_paid = False
        self._retract_release_done = False
        self._demount_replan_done = False
        self._home_ee_pos = None
        self._home_ee_quat = None
        p.setGravity(*self.cfg.gravity, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.cfg.sim_freq_hz, physicsClientId=self.client)
        # PyBullet expects ``globalCFM`` (solver-wide CFM); there is no ``contactCFM`` kwarg.
        p.setPhysicsEngineParameter(
            numSubSteps=self.cfg.physics_num_sub_steps,
            contactERP=self.cfg.contact_erp,
            globalCFM=self.cfg.contact_cfm,
            physicsClientId=self.client,
        )

        # Optional static-pose domain randomization. When
        # ``cfg.USE_DOMAIN_RANDOMIZATION`` is False (default) both offsets
        # are zero and the hub / cargo spawn at the nominal coordinates
        # listed in ``config.py`` — Scene build is bit-identical to the
        # pre-DR path. When True, the offsets are sampled from the env's
        # seeded RNG (reproducible across runs sharing a seed) and the
        # ``Scene`` constructor consumes them before placing bodies.
        self._maybe_apply_domain_randomization()

        self.scene = Scene(
            self.client, self.cfg, self._np_random,
            hub_xy_offset=tuple(self._dr_hub_xy_offset.tolist()),
            cargo_xy_offset=tuple(self._dr_cargo_xy_offset.tolist()),
        )
        self.handles = self.scene.build()

        self.robot_A = make_robot_a(self.client, self.cfg)
        self.robot_B = make_robot_b(self.client, self.cfg)

        # When Robot A's base is intentionally buried below the ground plane
        # (e.g. the FANUC spacious layout sinks it to lower the working EE and
        # clear the inner reach deadzone), its base/lower links overlap the
        # infinite plane and would emit huge spurious contact forces — tripping
        # the contact-force termination on step 1. Disable A↔plane collision in
        # that case; the arm works well above the floor so it never needs it.
        if float(self.cfg.robot_A_base_pos[2]) < float(self.cfg.floor_z) - 1e-6:
            plane_uid = self.handles.plane
            n_links = p.getNumJoints(self.robot_A.uid, physicsClientId=self.client)
            for link in range(-1, n_links):
                p.setCollisionFilterPair(
                    self.robot_A.uid, plane_uid, link, -1, 0,
                    physicsClientId=self.client,
                )

        # Vertical reference quaternion sourced from ``cfg.tire_spawn_rpy``.
        # Used by ``_pin_tire_to_world`` at reset/landing so the tire
        # returns to the spawn pose. ``_tire_vertical_error`` compares the
        # current tire bore axis against ``cfg.tire_spawn_axis_world`` (the
        # world direction implied by this rpy).
        self._vertical_quat = np.asarray(
            p.getQuaternionFromEuler(list(self.cfg.tire_spawn_rpy)),
            dtype=np.float64,
        )
        self._pickup_pos_world = np.asarray(
            self.cfg.tire_pickup_pos, dtype=np.float64
        )

        # Runtime pickup-gate radius. Defaults to the hard cap and is
        # updated externally by ``ApproachTolCurriculumCallback`` (in
        # ``src/train.py``). Eval / render paths leave this at the hard
        # cap, matching production behaviour.
        if not hasattr(self, "_approach_tol"):
            self._approach_tol = float(self.cfg.approach_radius_tol)

        # Settle a couple of physics steps so the IK warm start is sane.
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)

        # Park both robots at HOME — Robot A is NOT yet grasping the tire.
        self.robot_A.reset_to_home()
        self.robot_B.reset_to_home()

        # Cache the canonical HOME EE pose of Robot A *before* any
        # curriculum easy-start teleports the arm. The 6-stage remount
        # cycle (S2) retracts the empty gripper back to this pose after
        # releasing the freshly-mounted tire.
        _home_ee_pos, _home_ee_quat = self.robot_A.ee_pose()
        self._home_ee_pos = np.asarray(_home_ee_pos, dtype=np.float64).copy()
        self._home_ee_quat = np.asarray(_home_ee_quat, dtype=np.float64).copy()

        # Stage 0 starting-pose curriculum. Two operating modes
        # (``cfg.start_pos_curriculum_mode``):
        #   * "lerp": legacy smoothstep blend. ``_start_pos_alpha`` is
        #     swept 0 → 1 by ``StartPosCurriculumCallback`` and every
        #     reset uses that exact alpha.
        #   * "mix" (v8 default): each reset independently samples
        #     Bernoulli(``cfg.start_pos_easy_prob``) and dispatches to
        #     full-easy (alpha = 0) *or* full-hard (alpha = 1). The
        #     callback-controlled ``_start_pos_alpha`` is ignored in
        #     this branch — the *probability* itself is the curriculum
        #     knob, which can also be schedulable via the env method
        #     ``set_start_pos_alpha`` (interpreted as ``easy_prob`` in
        #     mix mode; see ``set_start_pos_alpha``).
        # Default alpha = 1.0 (full HOME = hardest) when no callback is
        # wired (eval / render / smoke). The blend teleports the UR10
        # EE to ``lerp(easy_start, HOME, alpha)`` via IK with
        # ``FINAL_LOCK_QUATERNION`` so the gripper stays palm-up.
        if not hasattr(self, "_start_pos_alpha"):
            self._start_pos_alpha = 1.0
        # **v11 (2026-05-31) — Reverse curriculum** takes priority over
        # the legacy start-pos curriculum. When enabled, the env reset
        # routes to one of three phases driven by ``_rev_curriculum_phase``
        # (set per-rollout by the ReverseCurriculumCallback). The phase
        # is consulted *before* the world-pin step so Phase A can place
        # the tire at the hub-mount pose instead of the cradle.
        if not hasattr(self, "_rev_curriculum_phase"):
            self._rev_curriculum_phase = "C"  # safe default (HOME)
        if not hasattr(self, "_start_pos_used_hot_start_last_reset"):
            self._start_pos_used_hot_start_last_reset = False

        rev_enabled = bool(getattr(self.cfg, "reverse_curriculum_enable", False))
        hot_started = False
        if rev_enabled and str(self._rev_curriculum_phase).upper() == "A":
            # Phase A — hub-aligned hot-start. Tire placed at the mount
            # target with axis aligned to hub axis (small jitter), UR10
            # EE on the 6-o'clock grasp anchor of that tire pose, fixed
            # grasp constraint attached, task_stage forcibly set to 1.
            try:
                self._apply_reverse_phase_a_hot_start()
                hot_started = True
            except Exception as exc:  # noqa: BLE001
                # Fall through to standard reset if hot-start fails for
                # any reason (geometry not loaded, IK failure, etc.).
                print(f"[reverse-curriculum] Phase A hot-start failed: {exc}")
                hot_started = False
        self._start_pos_used_hot_start_last_reset = bool(hot_started)

        if hot_started:
            # Hot-start handled tire pose + grasp + EE pose already.
            # Don't re-pin the tire (would override the mount pose).
            self._start_pos_used_easy_last_reset = False
        else:
            # Phase A→B blend may still want Phase B sampling (easy/HOME
            # mix). Phase B and C both consult the legacy start-pos
            # curriculum below. Phase C is implemented as easy_prob=0.
            phase = str(self._rev_curriculum_phase).upper() if rev_enabled else None
            if rev_enabled and phase == "B":
                # Force mix mode with the configured easy_prob.
                self._start_pos_alpha = float(getattr(
                    self.cfg, "start_pos_easy_prob", 0.75,
                ))
                start_mode = "mix"
                start_pos_active = True
            elif rev_enabled and phase == "C":
                # Pure HOME — disable the easy spawn entirely.
                self._start_pos_alpha = 0.0
                start_mode = "mix"
                start_pos_active = True
            else:
                start_pos_active = bool(getattr(
                    self.cfg, "start_pos_curriculum_enable", False,
                ))
                start_mode = str(getattr(
                    self.cfg, "start_pos_curriculum_mode", "lerp",
                )).lower()

            # **2026-06-01** — attached-hot-start gate. When the easy
            # branch is rolled AND ``cfg.attached_spawn_when_easy`` is
            # True, route into ``_apply_attached_hot_start`` which
            # spawns at Stage 1 with the tire already grasped at the
            # cradle pose. Stage 0 is skipped entirely so the policy
            # learns mount-only from step 1. Falls through to the
            # legacy "EE teleported but tire pinned" easy spawn when
            # the new flag is disabled.
            attached_started = False
            if start_pos_active:
                if start_mode == "mix":
                    # In mix mode, ``_start_pos_alpha`` is reinterpreted
                    # as the Bernoulli easy-spawn probability.
                    if rev_enabled and phase in ("B", "C"):
                        easy_prob = float(self._start_pos_alpha)
                    else:
                        easy_prob = float(getattr(self, "_start_pos_alpha", 1.0))
                        if easy_prob >= 1.0:
                            easy_prob = float(getattr(
                                self.cfg, "start_pos_easy_prob", 0.5,
                            ))
                    if float(self._np_random.random()) < easy_prob:
                        if bool(getattr(
                            self.cfg, "attached_spawn_when_easy", False,
                        )):
                            try:
                                self._apply_attached_hot_start()
                                attached_started = True
                            except Exception as exc:  # noqa: BLE001
                                # Fall through to legacy easy spawn if
                                # the attached path fails (IK failure
                                # under DR, geometry mismatch, etc.).
                                print(
                                    f"[planner-residual] attached hot-start "
                                    f"failed, falling back to legacy easy "
                                    f"spawn: {exc}"
                                )
                                attached_started = False
                        if not attached_started:
                            self._apply_start_pos_curriculum(0.0)
                        self._start_pos_used_easy_last_reset = True
                    else:
                        self._start_pos_used_easy_last_reset = False
                else:  # "lerp"
                    if float(self._start_pos_alpha) < 1.0:
                        self._apply_start_pos_curriculum(float(self._start_pos_alpha))
                    self._start_pos_used_easy_last_reset = (
                        float(self._start_pos_alpha) < 0.5
                    )
            else:
                self._start_pos_used_easy_last_reset = False

            # Pin the tire to the floor pickup pose so it stands upright
            # on its tread edge. Removed at Stage 0 → 1 grasp, re-engaged
            # at the final landing. The attached-hot-start path manages
            # its own world-pin / grasp lifecycle and must skip this.
            if not attached_started:
                self._pin_tire_to_world(
                    self._pickup_pos_world, self._vertical_quat,
                )

        self._step_count = 0
        self._prev_action = np.zeros(self.cfg.action.dim, dtype=np.float32)
        self._prev_d_A = None
        self._prev_d_B = None
        self._prev_d_approach = None
        self._prev_d_return = None
        # v11: Stage 2 PB-demount accumulator. Reset on every episode
        # so the first Stage 2 step doesn't see a stale ``_prev_d_hub``
        # from a previous (longer) episode.
        self._prev_d_hub: Optional[float] = None
        # v6 (4-stage FSM) — Stage 2 demount bookkeeping. ``_mount_done_step``
        # records the env step at which Stage 1 → 2 fired; the demount gate
        # only becomes eligible once ``step_count - _mount_done_step >=
        # cfg.demount_stall_steps``. ``_demount_bonus_paid`` mirrors the
        # existing ``_pickup_bonus_paid`` / ``_mount_bonus_paid`` idempotence.
        self._mount_done_step: Optional[int] = None
        self._demount_bonus_paid = False


        # 2026-06-01 — initialise the planner trajectory for whatever
        # ``self.task_stage`` the spawn logic above ended up in. With the
        # default attached-hot-start path this targets the mount end-pose;
        # with a HOME spawn it targets the cradle grasp anchor (Stage 0).
        # When ``use_planner_residual = False`` this is a no-op and the
        # legacy delta-EE path takes over inside ``_apply_action``.
        self._replan_for_current_stage()

        obs = self._compute_obs()
        info = {"target_bolt_idx": self.handles.target_bolt_idx}
        return obs, info

    # ------------------------------------------------------------------
    # Action / obs masks (Phase 1 feature isolation)
    # ------------------------------------------------------------------
    def _build_action_mask(self) -> np.ndarray:
        """Cache an action mask sized to the active ``action_space``.

        Two regimes:

        * Phase 1 — ``freeze_robot_b=True`` ⇒ ``action.dim == 6``.
          Δpose_A only; gripper_A and the Panda block are sliced out of
          the action space entirely. Mask is ``ones(6)``.

        * Phase 2/3 — ``freeze_robot_b=False`` ⇒ ``action.dim == 13``.
          Mask is ``ones(13)``. (When a caller manually freezes Panda
          *while keeping* a 13-d action — non-default path — the Panda
          slice ``[6:12]`` is zeroed so action / jerk L2 penalties do
          not waste reward signal on a dead manifold.)
        """
        cached = getattr(self, "_action_mask_cache", None)
        if cached is not None:
            return cached
        dim = int(self.cfg.action.dim)
        m = np.ones(dim, dtype=np.float64)
        if bool(getattr(self.cfg, "freeze_robot_b", False)) and dim == 13:
            m[6:12] = 0.0
        # **2026-06-01** — when the planner-residual path is active AND
        # the rotation offset channel is disabled, ``action[3:6]`` has no
        # effect on the world. Zero it in the action/jerk L2 mask so the
        # policy isn't penalised for whatever residual noise it emits on
        # those dead channels. Preserves action_space dim (6 or 13).
        if (
            bool(getattr(self.cfg, "use_planner_residual", False))
            and not bool(getattr(self.cfg, "planner_enable_rot_offset", False))
            and dim >= 6
        ):
            m[3:6] = 0.0
        self._action_mask_cache = m
        return m

    def _build_obs_mask(self) -> np.ndarray:
        """Cache an obs mask that zeros Phase-1-irrelevant Panda channels.

        Layout follows ``_compute_obs``'s concatenation order. The first
        73 entries and the trailing 3 ``mount_tail`` scalars are dim-
        independent; only the ``prev_action`` slice between them changes
        length with the action space (6 in Phase 1, 13 in Phase 2/3).

          [0:6]    qA_n            — UR10 joints
          [6:12]   dqA_n           — UR10 joint vels
          [12:19]  qB_n (7)        — Robot B joints (Panda 7-D; UR10e padded)
          [19:26]  dqB_n (7)       — Robot B joint vels  ← zero in Phase 1
          [26:33]  eeA_pos_rel/orn — UR10 EE
          [33:40]  eeB_pos_rel/orn — Panda EE          ← zero in Phase 1
          [40:47]  tire_pos_rel/orn
          [47:54]  hub_pos_rel/orn
          [54:61]  bolt_pos_rel/orn                    ← zero in Phase 1
          [61:67]  Δtire-hub (pos 3 + axisangle 3)
          [67:73]  ΔeeB-bolt                           ← zero in Phase 1
          [73 : 73+A]   prev_action (A = action.dim)
          [73+A : 76+A] mount_tail (axial/lateral/lug) ← KEEP (mount FSM)
          [76+A : 79+A] hub_guide_vector (v7) ← KEEP (carry guide cue)

        ``prev_action`` is *not* masked here — in Phase 1 its Panda slice
        no longer exists (action is already 6-d), and in Phase 2/3 those
        channels are part of the real control output. The v7 hub guide
        vector is base-frame agnostic (a pure EE→hub direction) and is
        kept in every phase since the UR10 is always present.
        """
        cached = getattr(self, "_obs_mask_cache", None)
        if cached is not None:
            return cached
        m = np.ones(int(self.cfg.obs.dim), dtype=np.float64)
        if bool(getattr(self.cfg, "freeze_robot_b", False)):
            m[12:19] = 0.0   # qB
            m[19:26] = 0.0   # dqB
            m[33:40] = 0.0   # eeB
            m[54:61] = 0.0   # bolt
            m[67:73] = 0.0   # ΔeeB-bolt
        self._obs_mask_cache = m
        return m

    # ------------------------------------------------------------------
    # Curriculum hooks (callable from SB3 callbacks via env_method)
    # ------------------------------------------------------------------
    def set_mount_tol(self, radius: float, angle_rad: float) -> None:
        """v6 mount-gate curriculum entry point.

        Broadcast each rollout boundary by ``MountTolCurriculumCallback``
        with the schedule's current (radius_m, angle_rad). Eval / render
        paths leave both at the config defaults (4 cm / 5°). The pair is
        consumed by ``_try_stage_transitions`` under ``task_stage == 1``.
        """
        self._mount_radius_tol = float(radius)
        self._mount_angle_tol = float(angle_rad)

    def get_mount_tol(self) -> Tuple[float, float]:
        return (
            float(getattr(self, "_mount_radius_tol", self.cfg.mount_radius_tol)),
            float(getattr(self, "_mount_angle_tol", self.cfg.reward.delta_A)),
        )

    def set_approach_tol(self, value: float) -> None:
        """External curriculum entry point for the Stage 0 → 1 grasp gate.

        Called by ``ApproachTolCurriculumCallback`` in ``src/train.py`` at
        every PPO rollout boundary with the schedule's current value.
        Has no effect outside training (eval / render keep the hard cap
        ``cfg.approach_radius_tol``).
        """
        self._approach_tol = float(value)

    def get_approach_tol(self) -> float:
        """Mirror of ``set_approach_tol`` — useful for logging callbacks."""
        return float(getattr(self, "_approach_tol", self.cfg.approach_radius_tol))

    def set_start_pos_easy_prob(self, prob: float) -> None:
        """v8 helper — explicitly set the mix-mode easy-spawn probability.

        Identical to ``set_start_pos_alpha`` but with a name that makes
        the mix-mode semantic clear (``alpha`` originally meant a blend
        coefficient, not a probability). Both call sites store on the
        same attribute so existing callbacks continue to work.
        """
        self._start_pos_alpha = float(np.clip(prob, 0.0, 1.0))

    def set_start_pos_alpha(self, value: float) -> None:
        """External curriculum entry point for the Stage 0 starting pose.

        ``alpha ∈ [0, 1]``: 0 = full easy (EE teleported just below the
        grasp anchor), 1 = full HOME. Smoothstep blending is applied by
        the callback before this is invoked so the env can lerp linearly.
        Called every PPO rollout boundary by ``StartPosCurriculumCallback``;
        eval / render leave the default 1.0 = HOME.
        """
        self._start_pos_alpha = float(np.clip(value, 0.0, 1.0))

    def get_start_pos_alpha(self) -> float:
        """Mirror of ``set_start_pos_alpha`` — useful for logging callbacks."""
        return float(getattr(self, "_start_pos_alpha", 1.0))

    def _apply_start_pos_curriculum(self, alpha: float) -> None:
        """Teleport UR10 EE to ``lerp(easy_start, home_ee, alpha)``.

        ``easy_start`` lies ``cfg.start_pos_easy_lift`` metres below the
        tire 6 o'clock grasp anchor (alpha = 0 = maximum easy). The
        analytical HOME EE pose (alpha = 1) is taken from
        ``robot_A.ee_pose()`` right after ``reset_to_home`` parked the
        UR10 at HOME. IK is run against ``FINAL_LOCK_QUATERNION`` so the
        gripper still faces palm-up regardless of where on the segment
        the blend lands.

        After the joint teleport, ``robot_A.last_target_pos`` is re-seeded
        with the achieved EE so subsequent ``apply_delta_ee`` calls
        accumulate Δpos from the new analytical baseline (not HOME).
        Without this re-seed the first Δpos action would snap the IK
        target back to HOME-Δ, undoing the curriculum on step 1.
        """
        ur = self.robot_A
        R = float(self.cfg.tire_outer_radius)
        tire_pos = np.asarray(self.cfg.tire_pickup_pos, dtype=np.float64)
        grasp_anchor = tire_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
        lift = float(getattr(self.cfg, "start_pos_easy_lift", 0.20))
        easy = grasp_anchor + np.array([0.0, 0.0, -lift], dtype=np.float64)
        home_ee = ur.ee_pose()[0].copy()
        a = float(np.clip(alpha, 0.0, 1.0))
        target = (1.0 - a) * easy + a * home_ee

        # Warm-start IK from the current (HOME) joint state so the IK
        # solver picks the same elbow / wrist branch as HOME, avoiding
        # an arm flip even at low alpha values.
        warm_start, _ = ur.joint_state()
        ik = p.calculateInverseKinematics(
            ur.uid,
            ur.EE_LINK_INDEX,
            target.tolist(),
            list(ur.FINAL_LOCK_QUATERNION),
            lowerLimits=ur.arm.lower.tolist(),
            upperLimits=ur.arm.upper.tolist(),
            jointRanges=ur.arm.range.tolist(),
            restPoses=warm_start.tolist(),
            maxNumIterations=300,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        if ur._ik_arm_slots and len(ik) > max(ur._ik_arm_slots):
            arm_q = ik[ur._ik_arm_slots]
        else:
            arm_q = np.asarray(ur.HOME_POSE, dtype=np.float64)
        arm_q = np.clip(arm_q, ur.arm.lower, ur.arm.upper)

        for idx, q in zip(ur.arm.indices, arm_q):
            p.resetJointState(
                ur.uid, idx,
                targetValue=float(q),
                targetVelocity=0.0,
                physicsClientId=self.client,
            )
        ur.last_target_pos = ur.ee_pose()[0].copy()

    # ------------------------------------------------------------------
    # **v11 (2026-05-31) — Reverse curriculum hot-start.**
    # ------------------------------------------------------------------
    def set_terminate_on(self, value: str) -> None:
        """**v11c (2026-05-31)** — broadcast-able ``terminate_on`` toggle.

        Called by the ``ReverseCurriculumCallback`` on every phase change.
        Phase A flips the env to ``terminate_on="mount"`` so the
        first-step mount fire collects R_mount + ``is_success`` and
        the episode ends; Phase B/C restore the CLI-supplied default
        (typically ``"never"`` for full 4-stage cycle).

        Args
        ----
        value : str
            One of ``"never" | "pickup" | "mount" | "demount"``.
        """
        v = str(value).strip().lower()
        if v not in ("never", "pickup", "mount", "demount"):
            raise ValueError(f"terminate_on must be never/pickup/mount/demount, got {value!r}")
        self.cfg.terminate_on = v

    def get_terminate_on(self) -> str:
        return str(getattr(self.cfg, "terminate_on", "never")).lower()

    def set_safety_terminations(self, enabled: bool) -> None:
        """**v11c2 (2026-05-31)** — master switch for the four "safety"
        episode-end gates: vertical / collision / workspace / contact_force.

        See ``EnvConfig.safety_terminations_enabled`` for the rationale.
        """
        self.cfg.safety_terminations_enabled = bool(enabled)

    def get_safety_terminations(self) -> bool:
        return bool(getattr(self.cfg, "safety_terminations_enabled", True))

    def set_contact_force_term(self, value: float) -> None:
        """**v11c1 (2026-05-31)** — broadcast-able ``contact_force_terminate_above``.

        Called by ``ReverseCurriculumCallback`` so Phase A can disable
        the contact-force kill switch (set to ``+inf``) while the tire
        is sitting on the hub at hot-start — the first physics step
        produces 25k–70k N of contact force which would otherwise
        terminate the episode in one step. Phase B/C restore the
        config default so the safety gate is active during random-
        spawn / HOME training.
        """
        self.cfg.contact_force_terminate_above = float(value)

    def get_contact_force_term(self) -> float:
        return float(getattr(self.cfg, "contact_force_terminate_above", 0.0))

    def set_reverse_curriculum_phase(self, phase: str) -> None:
        """External entry point for the ReverseCurriculumCallback.

        ``phase`` ∈ {"A", "B", "C"}. The env consumes this on the next
        ``reset()`` call. Defaults to "C" (pure HOME) when never set,
        so eval / render paths always evaluate the production task.
        """
        ph = str(phase).strip().upper()
        if ph not in ("A", "B", "C"):
            raise ValueError(f"reverse curriculum phase must be A/B/C, got {phase!r}")
        self._rev_curriculum_phase = ph

    def get_reverse_curriculum_phase(self) -> str:
        return str(getattr(self, "_rev_curriculum_phase", "C")).upper()

    def _apply_reverse_phase_a_hot_start(self) -> None:
        """Place the tire at the hub mount pose, attach the grasp, and
        teleport the UR10 EE to the corresponding 6-o'clock anchor.

        Sets ``self.task_stage = 1`` so the policy starts inside the
        carry/mount window. Small radial + angular jitter is applied so
        every reset is a slightly different "almost mounted" pose,
        preventing memorisation of a single grasp.
        """
        R = float(self.cfg.tire_outer_radius)
        mount_target = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
        hub_axis = np.asarray(self.cfg.hub_axis_world, dtype=np.float64)
        hub_axis = hub_axis / max(float(np.linalg.norm(hub_axis)), 1e-9)

        radial_jitter = float(getattr(
            self.cfg, "reverse_phase_a_radial_jitter", 0.03,
        ))
        angular_jitter = float(getattr(
            self.cfg, "reverse_phase_a_angular_jitter", np.deg2rad(2.0),
        ))

        # Sample a backoff along the hub axis (1 .. 3 cm away from the
        # mount target) so the tire is "just outside" the bore.
        backoff = float(self._np_random.uniform(0.01, max(radial_jitter, 0.01)))
        tire_pos = mount_target - backoff * hub_axis

        # Tire orientation — align bore axis with hub axis + small
        # random tilt within ``angular_jitter`` rad. Implemented as a
        # rotation around a random axis perpendicular to hub_axis.
        if angular_jitter > 0.0:
            perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(perp, hub_axis))) > 0.95:
                perp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            perp = perp - float(np.dot(perp, hub_axis)) * hub_axis
            perp = perp / max(float(np.linalg.norm(perp)), 1e-9)
            tilt = float(self._np_random.uniform(-angular_jitter, angular_jitter))
            # Rotate hub_axis around ``perp`` by ``tilt`` to get the
            # perturbed tire axis.
            cos_t, sin_t = float(np.cos(tilt)), float(np.sin(tilt))
            tire_axis = (
                cos_t * hub_axis
                + sin_t * np.cross(perp, hub_axis)
                + (1.0 - cos_t) * float(np.dot(perp, hub_axis)) * perp
            )
            tire_axis = tire_axis / max(float(np.linalg.norm(tire_axis)), 1e-9)
        else:
            tire_axis = hub_axis.copy()

        # Build a quaternion whose +Z body axis matches ``tire_axis``.
        # Tire URDF convention (see scene.py / cfg.tire_spawn_rpy) has
        # the bore axis along the body +Z; we align that to tire_axis.
        z_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = np.cross(z_ref, tire_axis)
        c = float(np.dot(z_ref, tire_axis))
        s = float(np.linalg.norm(v))
        if s < 1e-9:
            # tire_axis is parallel (or anti-parallel) to z_ref.
            if c > 0.0:
                tire_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            else:
                tire_orn = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = v / s
            half = 0.5 * float(np.arctan2(s, c))
            sh = float(np.sin(half))
            tire_orn = np.array(
                [axis[0] * sh, axis[1] * sh, axis[2] * sh, float(np.cos(half))],
                dtype=np.float64,
            )

        # 1. Detach any existing grasp from a prior episode.
        if self._grasp_constraint is not None:
            try:
                p.removeConstraint(
                    self._grasp_constraint, physicsClientId=self.client,
                )
            except p.error:
                pass
            self._grasp_constraint = None

        # 2. Release world-pin if currently engaged (eval path may not).
        self._release_world_pin()

        # 3. Restore tire dynamic mass so the constraint behaves like a
        #    real grasp (mass=0 is reserved for the world-pin).
        p.changeDynamics(
            self.handles.tire, -1,
            mass=float(self.cfg.tire_mass),
            physicsClientId=self.client,
        )

        # 4. Snap tire to the perturbed mount pose.
        p.resetBasePositionAndOrientation(
            self.handles.tire,
            tire_pos.tolist(),
            tire_orn.tolist(),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.handles.tire,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )

        # 5. Compute UR10 EE target: 6-o'clock grasp anchor of the
        #    perturbed tire pose. In the cradle pose the anchor is
        #    ``tire_pos + (0, 0, -R)``; for arbitrary tire orientations
        #    we rotate that local anchor by the tire's body frame so
        #    the EE meets the tread at the right side of the wheel.
        # The tire bore axis is body +Z (per tire URDF), so the "6
        # o'clock" tread direction is **body −X** (any direction
        # orthogonal to bore axis works for a circular tire). For the
        # phase-A hot-start we pick the world −Z (gravity) projection
        # onto the bore plane so the grasp point is as close to "below
        # the tire" as the perturbed orientation allows.
        tire_R_mat = np.array(
            p.getMatrixFromQuaternion(list(tire_orn)), dtype=np.float64,
        ).reshape(3, 3)
        # World −Z projected onto the bore plane (orthogonal to
        # tire_axis), normalised, then scaled by R to get the offset.
        gravity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        proj = gravity_dir - float(np.dot(gravity_dir, tire_axis)) * tire_axis
        proj_norm = float(np.linalg.norm(proj))
        if proj_norm < 1e-6:
            # tire axis ~ world ±Z: fall back to world +X for the
            # tread offset (arbitrary choice on a circular tire).
            tread_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            tread_dir = tread_dir - float(np.dot(tread_dir, tire_axis)) * tire_axis
            tread_dir = tread_dir / max(float(np.linalg.norm(tread_dir)), 1e-9)
        else:
            tread_dir = proj / proj_norm
        ee_target = tire_pos + R * tread_dir

        # 6. IK the UR10 to that EE target with FINAL_LOCK_QUATERNION.
        ur = self.robot_A
        warm_start, _ = ur.joint_state()
        ik = p.calculateInverseKinematics(
            ur.uid,
            ur.EE_LINK_INDEX,
            ee_target.tolist(),
            list(ur.FINAL_LOCK_QUATERNION),
            lowerLimits=ur.arm.lower.tolist(),
            upperLimits=ur.arm.upper.tolist(),
            jointRanges=ur.arm.range.tolist(),
            restPoses=warm_start.tolist(),
            maxNumIterations=300,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        if ur._ik_arm_slots and len(ik) > max(ur._ik_arm_slots):
            arm_q = ik[ur._ik_arm_slots]
        else:
            arm_q = np.asarray(ur.HOME_POSE, dtype=np.float64)
        arm_q = np.clip(arm_q, ur.arm.lower, ur.arm.upper)
        for idx, q in zip(ur.arm.indices, arm_q):
            p.resetJointState(
                ur.uid, idx,
                targetValue=float(q),
                targetVelocity=0.0,
                physicsClientId=self.client,
            )
        ur.last_target_pos = ur.ee_pose()[0].copy()

        # 7. Attach the grasp constraint between EE and tire.
        ee_pos, ee_orn = ur.ee_pose()
        tire_R = np.array(
            p.getMatrixFromQuaternion(list(tire_orn)), dtype=np.float64,
        ).reshape(3, 3)
        local_anchor = tire_R.T @ (-R * tread_dir)
        child_pos = local_anchor.tolist()
        inv_tire_pos, inv_tire_orn = p.invertTransform(
            tire_pos.tolist(), tire_orn.tolist(),
        )
        _, child_orn = p.multiplyTransforms(
            inv_tire_pos, inv_tire_orn,
            ee_pos.tolist(), ee_orn.tolist(),
        )
        if self._use_kinematic_tire_sync():
            self._begin_kinematic_grasp()
        else:
            self._grasp_kinematic = False
            self._grasp_constraint = p.createConstraint(
                parentBodyUniqueId=ur.uid,
                parentLinkIndex=ur.EE_LINK_INDEX,
                childBodyUniqueId=self.handles.tire,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                parentFrameOrientation=[0, 0, 0, 1],
                childFramePosition=child_pos,
                childFrameOrientation=list(child_orn),
                physicsClientId=self.client,
            )
            p.changeConstraint(
                self._grasp_constraint,
                maxForce=1.0e6,
                erp=1.0,
                physicsClientId=self.client,
            )
            self._cache_grasp_relative_transform()

        # 8. Force Stage-1 entry — the pickup gate already fired by
        #    construction. The mount/demount bookkeeping still runs
        #    via ``_try_stage_transitions`` based on env state.
        self.task_stage = 1
        self._pickup_bonus_paid = True
        # If the grasp above was created as a kinematic upright lock
        # (Stage-0 sync), promote it to a rigid JOINT_FIXED bond now that
        # we are entering the carry stage — the per-step upright sync does
        # not run outside the kinematic stages, so the tire would otherwise
        # be unconstrained during the carry.
        self._maybe_promote_to_fixed_grasp()
        # Re-seed PB shaping accumulators so the first Stage-1 step
        # doesn't see a stale ``_prev_d_A``.
        self._prev_d_approach = None
        self._prev_d_return = None
        self._prev_d_A = None
        # v11c — stash a one-shot "force the mount FSM event on the
        # next ``step()``" flag. Without it, the first physics
        # decimation pushes the tire out of mount_tol (the hub-aligned
        # grasp + small policy action injects O(0.5 m) translation on
        # step 1) and the mount event never fires for the rest of the
        # episode. Phase A is meant to deliver a guaranteed R_mount
        # sparse signal per reset, so we bypass the post-step
        # geometric check exactly once.
        self._phase_a_force_mount_first_step = True

    # ------------------------------------------------------------------
    # **2026-06-01 — Min-Jerk planner + residual control helpers.**
    # ------------------------------------------------------------------
    def _apply_attached_hot_start(self) -> None:
        """Stage-1 attached hot-start at the cradle pickup pose.

        Used when ``cfg.attached_spawn_when_easy = True`` and the easy
        branch is rolled. The tire remains at the cradle ``tire_pickup_pos``
        (world-pinned until this method runs). The UR10 EE is IK'd to the
        6-o'clock grasp anchor, then a rigid grasp is created **in place**
        via :pymeth:`_create_grasp_constraint_in_place` — the tire is never
        teleported (unlike the legacy :pymeth:`_attach_tire_to_robot_A`
        snap), which avoids gripper↔tire interpenetration spikes on the
        first physics step.
        """
        R = float(self.cfg.tire_outer_radius)
        tire_pos = np.asarray(self.cfg.tire_pickup_pos, dtype=np.float64)
        # Exact 6-o'clock outer tread point for the spawn orientation.
        grasp_anchor = tire_pos + np.array([0.0, 0.0, -R], dtype=np.float64)

        ur = self.robot_A
        warm_start, _ = ur.joint_state()
        ik = p.calculateInverseKinematics(
            ur.uid,
            ur.EE_LINK_INDEX,
            grasp_anchor.tolist(),
            list(ur.FINAL_LOCK_QUATERNION),
            lowerLimits=ur.arm.lower.tolist(),
            upperLimits=ur.arm.upper.tolist(),
            jointRanges=ur.arm.range.tolist(),
            restPoses=warm_start.tolist(),
            maxNumIterations=300,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        if ur._ik_arm_slots and len(ik) > max(ur._ik_arm_slots):
            arm_q = ik[ur._ik_arm_slots]
        else:
            arm_q = np.asarray(ur.HOME_POSE, dtype=np.float64)
        arm_q = np.clip(arm_q, ur.arm.lower, ur.arm.upper)
        for idx, q in zip(ur.arm.indices, arm_q):
            p.resetJointState(
                ur.uid, idx,
                targetValue=float(q),
                targetVelocity=0.0,
                physicsClientId=self.client,
            )
        forces = [400.0, 400.0, 300.0, 60.0, 60.0, 60.0]
        p.setJointMotorControlArray(
            ur.uid, ur.arm.indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=arm_q.tolist(),
            forces=forces,
            positionGains=[1.0] * 6,
            velocityGains=[1.0] * 6,
            physicsClientId=self.client,
        )
        ur.last_target_pos = ur.ee_pose()[0].copy()

        # Restore tire dynamic mass, then kinematic upright grasp in place.
        self._release_world_pin()
        self.task_stage = 1
        self._pickup_bonus_paid = True
        self._begin_kinematic_grasp()
        # The hot-start lands directly in the carry stage (task_stage = 1),
        # which is not a kinematic-sync stage, so the per-step upright lock
        # never runs. Promote the kinematic grasp to a rigid JOINT_FIXED
        # bond exactly as the pickup→carry FSM transition does
        # (``_try_stage_transitions`` → ``_maybe_promote_to_fixed_grasp``);
        # otherwise the tire is left completely unconstrained during the
        # carry and the arm flies to the hub empty-handed (S1 stall).
        self._maybe_promote_to_fixed_grasp()
        self._prev_d_approach = None
        self._prev_d_return = None
        self._prev_d_A = None

        # Short settle — motors already hold ``arm_q`` so the 0.5 kg load
        # does not sag the arm before the first policy step.
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)
        ur.last_target_pos = ur.ee_pose()[0].copy()
        hold_q, _ = ur.joint_state()
        self._planner_hold_arm_targets = hold_q.copy()

    def _cache_grasp_relative_transform(self) -> None:
        """Record T_ee_tire = inv(T_world_ee) ∘ T_world_tire at this moment.

        Called from :pymeth:`_attach_tire_to_robot_A` right after the
        rigid grasp is created. The transform is invariant for the
        lifetime of that grasp (JOINT_FIXED is, by definition, a rigid
        offset in the parent's frame) so the planner can pre-compute
        every future EE end-point from it without re-querying physics.
        """
        ee_pos, ee_orn = self.robot_A.ee_pose()
        tire_pos, tire_orn = self.scene.tire_pose()
        inv_ee_pos, inv_ee_orn = p.invertTransform(
            np.asarray(ee_pos, dtype=np.float64).tolist(),
            np.asarray(ee_orn, dtype=np.float64).tolist(),
        )
        t_pos, t_orn = p.multiplyTransforms(
            inv_ee_pos, inv_ee_orn,
            np.asarray(tire_pos, dtype=np.float64).tolist(),
            np.asarray(tire_orn, dtype=np.float64).tolist(),
        )
        self._grasp_t_ee_tire_pos = np.asarray(t_pos, dtype=np.float64)
        self._grasp_t_ee_tire_quat = np.asarray(t_orn, dtype=np.float64)

    def _is_tire_grasped(self) -> bool:
        return (
            self._grasp_constraint is not None
            or self._grasp_kinematic
        )

    def _enforce_robot_a_palm_up(self) -> None:
        """Post-step: keep the FANUC gripper palm-up (tool +Z = world +Z).

        Snaps the arm joints back onto the palm-up manifold (yaw free) when
        the heavy-tire load has drooped the stiff position PD off vertical,
        then re-places the grasped tire via the cached EE↔tire transform so
        the rigid ``JOINT_FIXED`` bond is not shocked. Gated by
        ``cfg.fanuc_enforce_palm_up_post_step``. No-op when the robot lacks
        the IK-based re-lock (e.g. Panda Robot B).
        """
        if not bool(getattr(self.cfg, "fanuc_enforce_palm_up_post_step", False)):
            return
        if self.robot_A is None:
            return
        enforce = getattr(self.robot_A, "enforce_palm_up_pose", None)
        if enforce is None:
            return
        thr = float(getattr(self.cfg, "fanuc_palm_up_tool_z_threshold", 0.999))
        corrected = bool(enforce(thr))
        if not corrected:
            return
        # The kinematic upright sync (stage 0) re-places the tire from the EE
        # itself, so only the JOINT_FIXED bond needs an explicit re-place.
        if self._grasp_kinematic or self._grasp_constraint is None:
            return
        t_pos = getattr(self, "_grasp_t_ee_tire_pos", None)
        t_quat = getattr(self, "_grasp_t_ee_tire_quat", None)
        if t_pos is None or t_quat is None:
            return
        ee_pos, ee_orn = self.robot_A.ee_pose()
        new_pos, new_orn = p.multiplyTransforms(
            np.asarray(ee_pos, dtype=np.float64).tolist(),
            np.asarray(ee_orn, dtype=np.float64).tolist(),
            np.asarray(t_pos, dtype=np.float64).tolist(),
            np.asarray(t_quat, dtype=np.float64).tolist(),
        )
        p.resetBasePositionAndOrientation(
            self.handles.tire, list(new_pos), list(new_orn),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.handles.tire, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )

    def _use_tire_upright_lock(self) -> bool:
        return bool(getattr(self.cfg, "lock_tire_upright_when_grasped", True))

    def _kinematic_tire_stages(self) -> Tuple[int, ...]:
        return tuple(
            int(x) for x in getattr(
                self.cfg, "kinematic_tire_lock_stages", (0,),
            )
        )

    def _grasp_com_offset_from_ee(self, ee_orn: np.ndarray) -> np.ndarray:
        """World-frame offset from EE link to tire COM at grasp."""
        R = float(self.cfg.tire_outer_radius)
        off = np.asarray(
            getattr(self.cfg, "grasp_com_offset_world", (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        scale = off * R if np.linalg.norm(off) > 1e-9 else np.array([0.0, 0.0, R])
        if abs(float(scale[0])) < 1e-9 and abs(float(scale[1])) < 1e-9:
            return np.array([0.0, 0.0, float(scale[2])], dtype=np.float64)
        tire_R = np.array(
            p.getMatrixFromQuaternion(list(ee_orn)), dtype=np.float64,
        ).reshape(3, 3)
        return tire_R @ scale

    def _use_kinematic_tire_sync(self) -> bool:
        """Per-step tire teleport upright lock (Stage 0 only by default)."""
        return (
            self._use_tire_upright_lock()
            and int(self.task_stage) in self._kinematic_tire_stages()
        )

    def _maybe_promote_to_fixed_grasp(self) -> None:
        """Switch Stage 0 kinematic lock → JOINT_FIXED for carry/mount."""
        if not self._grasp_kinematic:
            return
        if int(self.task_stage) in self._kinematic_tire_stages():
            return
        self._grasp_kinematic = False
        self._create_grasp_constraint_in_place(force_fixed=True)

    @staticmethod
    def _yaw_about_world_z_from_quat(quat: np.ndarray) -> float:
        """Yaw (rad) of tool +X projected onto the world XY plane."""
        R = np.array(
            p.getMatrixFromQuaternion(list(quat)), dtype=np.float64,
        ).reshape(3, 3)
        ex = R[:, 0]
        return float(np.arctan2(ex[1], ex[0]))

    def _upright_tire_quat_for_ee(self, ee_orn: np.ndarray) -> np.ndarray:
        """Standing spawn pose + world-Z yaw = EE yaw − yaw at grasp."""
        psi = self._yaw_about_world_z_from_quat(ee_orn)
        psi0 = float(self._grasp_yaw_ee0) if self._grasp_yaw_ee0 is not None else psi
        delta = psi - psi0
        r_spawn = Rotation.from_euler(
            "xyz", list(self.cfg.tire_spawn_rpy), degrees=False,
        )
        r_tgt = Rotation.from_euler("z", delta, degrees=False) * r_spawn
        return np.asarray(r_tgt.as_quat(), dtype=np.float64)

    def _cache_grasp_kinematic_offsets(self) -> None:
        """World-frame tire-COM offset at grasp + reference EE yaw.

        Storing the offset in WORLD frame (rather than EE frame) means
        the per-step kinematic sync only rotates it about world +Z by
        the EE yaw delta. If EE pitches/rolls due to IK failure, the
        tire COM does NOT follow that tilt — it stays at the same
        height + horizontal radius, just spinning about vertical. This
        prevents the workspace-violation terminations we hit when the
        full EE rotation matrix sent the tire flying sideways.
        """
        ee_pos, ee_orn = self.robot_A.ee_pose()
        tire_pos, _ = self.scene.tire_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_orn = np.asarray(ee_orn, dtype=np.float64)
        tire_pos = np.asarray(tire_pos, dtype=np.float64)
        self._grasp_com_offset_ee = (tire_pos - ee_pos).astype(np.float64)
        self._grasp_yaw_ee0 = self._yaw_about_world_z_from_quat(ee_orn)

    def _begin_kinematic_grasp(self) -> None:
        """Grasp without JOINT_FIXED — upright lock drives tire each step."""
        self._release_grasp()
        self._cache_grasp_relative_transform()
        self._cache_grasp_kinematic_offsets()
        self._grasp_kinematic = True
        # Seed the "last safe pose" with the current tire pose so that if
        # the very first sync sees a penetration (e.g. spawn already
        # overlapping), the revert has somewhere to fall back to.
        try:
            tire_pos0, tire_orn0 = self.scene.tire_pose()
            self._safe_tire_pos = np.asarray(tire_pos0, dtype=np.float64).copy()
            self._safe_tire_orn = np.asarray(tire_orn0, dtype=np.float64).copy()
        except Exception:
            self._safe_tire_pos = None
            self._safe_tire_orn = None
        self._sync_grasped_tire_upright()

    def _sync_grasped_tire_upright(self) -> None:
        """Re-write tire pose: COM = EE + yaw-rotated world offset; orn upright.

        **2026-06-02 (cargo penetration fix)** — before committing the new
        kinematic pose, check whether it would push the tire INTO the cargo
        body or the cargo back wall. If so, revert to the last safe pose
        (``_safe_tire_pos`` / ``_safe_tire_orn``). This stops the tire from
        phasing through scene geometry when the policy drives the EE past
        the cargo wall. Combined with the new tire-vs-cargo branch in
        ``_in_bad_collision`` (per-step ``-w_collision`` penalty), the
        policy is taught to keep the tire out of the wall in the first
        place. ``getClosestPoints`` uses ≥0 contact penetration depth
        (cp[8] negative) to flag overlap.
        """
        if not self._grasp_kinematic or self.robot_A is None:
            return
        if self._grasp_com_offset_ee is None:
            return
        ee_pos, ee_orn = self.robot_A.ee_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_orn = np.asarray(ee_orn, dtype=np.float64)
        psi = self._yaw_about_world_z_from_quat(ee_orn)
        psi0 = float(self._grasp_yaw_ee0) if self._grasp_yaw_ee0 is not None else psi
        delta = psi - psi0
        c, s = float(np.cos(delta)), float(np.sin(delta))
        offs = self._grasp_com_offset_ee
        offs_yaw = np.array([
            c * offs[0] - s * offs[1],
            s * offs[0] + c * offs[1],
            offs[2],
        ], dtype=np.float64)
        tire_pos = ee_pos + offs_yaw
        tire_orn = self._upright_tire_quat_for_ee(ee_orn)
        alpha = float(getattr(self.cfg, "kinematic_tire_sync_alpha", 1.0))
        if (
            alpha < 1.0
            and self._safe_tire_pos is not None
            and self._safe_tire_orn is not None
        ):
            tire_pos = (
                (1.0 - alpha) * self._safe_tire_pos + alpha * tire_pos
            )
            q0 = np.asarray(self._safe_tire_orn, dtype=np.float64)
            q1 = np.asarray(tire_orn, dtype=np.float64)
            q_blend = (1.0 - alpha) * q0 + alpha * q1
            qn = float(np.linalg.norm(q_blend))
            if qn > 1e-12:
                tire_orn = q_blend / qn
        uid = self.handles.tire

        # ----- Penetration guard ---------------------------------------
        # Tentatively place tire at the desired pose, query closest points
        # against cargo + back wall, and if either reports a deep
        # penetration (> ``pen_tol``), revert to the cached safe pose.
        # Only the cargo body and the back-wall slab are checked: the
        # truck/hub is *expected* to make contact at the mount event, and
        # the cradle rails / plane are legitimate Stage-3 supports.
        p.resetBasePositionAndOrientation(
            uid, tire_pos.tolist(), tire_orn.tolist(),
            physicsClientId=self.client,
        )
        obstacles: List[int] = []
        if self.handles.vehicle is not None:
            obstacles.append(int(self.handles.vehicle))
        bw = getattr(self.handles, "cargo_back_wall", None)
        if bw is not None:
            obstacles.append(int(bw))
        # Penetration tolerance: 5 mm. Tire-cargo contacts at the cargo
        # face (e.g. mounted at the hub flange) generate sub-millimetre
        # overlap from PyBullet's contact margin and must not trip the
        # revert path; a deep clip (≥ 5 mm) reliably means we drove the
        # tire into the wall.
        pen_tol = -0.005
        penetrated = False
        for obs_uid in obstacles:
            cps = p.getClosestPoints(
                bodyA=uid,
                bodyB=obs_uid,
                distance=0.0,
                physicsClientId=self.client,
            )
            for cp in cps:
                if len(cp) > 8 and float(cp[8]) < pen_tol:
                    penetrated = True
                    break
            if penetrated:
                break

        if penetrated and self._safe_tire_pos is not None and self._safe_tire_orn is not None:
            p.resetBasePositionAndOrientation(
                uid,
                self._safe_tire_pos.tolist(),
                self._safe_tire_orn.tolist(),
                physicsClientId=self.client,
            )
        else:
            self._safe_tire_pos = tire_pos.copy()
            self._safe_tire_orn = tire_orn.copy()

        p.resetBaseVelocity(
            uid, linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )

    def _ee_pose_for_tire_pose(self, tire_pos, tire_quat
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Given a desired world tire pose, compute the required EE pose.

        Uses the most recently cached ``T_ee_tire`` from the active
        grasp. Returns ``(ee_pos, ee_quat)`` both in world frame. The
        identity ``T_world_ee * T_ee_tire = T_world_tire`` is solved
        for ``T_world_ee = T_world_tire ∘ inv(T_ee_tire)``.
        """
        inv_t_pos, inv_t_orn = p.invertTransform(
            np.asarray(self._grasp_t_ee_tire_pos, dtype=np.float64).tolist(),
            np.asarray(self._grasp_t_ee_tire_quat, dtype=np.float64).tolist(),
        )
        ee_pos, ee_orn = p.multiplyTransforms(
            np.asarray(tire_pos, dtype=np.float64).tolist(),
            np.asarray(tire_quat, dtype=np.float64).tolist(),
            inv_t_pos, inv_t_orn,
        )
        return (
            np.asarray(ee_pos, dtype=np.float64),
            np.asarray(ee_orn, dtype=np.float64),
        )

    def _compute_stage_end_ee_pose(self, stage: int
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """End-point ``(EE_pos, EE_quat)`` for the given stage's nominal traj.

        The mapping follows the Phase 1 FSM:

        * Stage 0 — HOME → tire cradle grasp anchor, palm-up.
        * Stage 1 — current EE → mount end-pose
                    (tire at ``tire_mount_pos`` with bore aligned to
                     ``hub_axis_world``). Computed via the cached
                     T_ee_tire from the active grasp.
        * Stage 2 — pull tire backward along ``hub_axis_world`` by
                    ``1.2 × demount_axial_distance`` (20 % overshoot
                    so the demount gate fires reliably).
        * Stage 3 — return tire to cradle pickup pose (vertical spawn
                    orientation, on rails).

        Stages 1–3 require the grasp to be active; if T_ee_tire is not
        cached (e.g. eval path that bypassed the grasp), the helpers
        fall back to a palm-up EE at the tire target — still a
        sensible direction but no orientation guarantee.
        """
        R = float(self.cfg.tire_outer_radius)
        palm_up = robot_a_lock_quaternion(self.robot_A)
        have_grasp = (
            self._grasp_t_ee_tire_pos is not None
            and self._grasp_t_ee_tire_quat is not None
        )

        # 6-stage remount cycle (opt-in). Remap the inserted stages onto
        # the legacy pose generators:
        #   S2 (retract) → empty-handed HOME EE pose (handled inline below).
        #   S3 (regrip)  → same EE pose that grasps the hub-mounted tire
        #                  (== legacy Stage 1 mount end pose).
        #   S4 (demount) → legacy Stage 2 pull-off pose.
        #   S5 (return)  → legacy Stage 3 cradle-return pose.
        if bool(getattr(self.cfg, "remount_cycle_enable", False)):
            if stage == 2:
                home_pos = getattr(self, "_home_ee_pos", None)
                home_quat = getattr(self, "_home_ee_quat", None)
                if home_pos is not None and home_quat is not None:
                    return (
                        np.asarray(home_pos, dtype=np.float64).copy(),
                        np.asarray(home_quat, dtype=np.float64).copy(),
                    )
                # Fallback: FK of the canonical rest pose is unavailable
                # here, so hold at the current EE (no-op retract).
                cur_pos, cur_quat = self.robot_A.ee_pose()
                return (
                    np.asarray(cur_pos, dtype=np.float64),
                    np.asarray(cur_quat, dtype=np.float64),
                )
            stage = {3: 1, 4: 2, 5: 3}.get(int(stage), int(stage))

        if stage == 0:
            tire_p = np.asarray(self.cfg.tire_pickup_pos, dtype=np.float64)
            end_pos = tire_p + np.array([0.0, 0.0, -R], dtype=np.float64)
            return end_pos, palm_up

        if stage == 1:
            # Mount end-pose: gripper UNDER the tire (6-o'clock), palm-up
            # (tool +Z ‖ world +Z).  Use ``palm_up`` (= FINAL_LOCK from
            # ``fanuc_home_pose`` FK) with **0° extra yaw** — do NOT apply
            # the UR10-era −90° Z rotation here.  On FANUC that rotation
            # kept palm-up (Z-rot preserves tool +Z) but swung tool +X
            # 90° away from ``hub_axis_world`` (scripts/diag_mount_yaw_palm.py:
            # 0° → bore ‖ hub 0.5°, −90° → 90° off).  ``_tilt_lock_palm_up_quat``
            # still enforces tool +Z = world +Z on every planner step.
            R = float(self.cfg.tire_outer_radius)
            hub_pos = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
            end_pos = hub_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
            end_quat = np.asarray(palm_up, dtype=np.float64)
            return end_pos, end_quat

        if stage == 2:
            # **2026-06-05 (demount = exact reverse of the Stage-1 insertion)**
            # Stage 1's final leg is a coaxial straight push from the
            # ``mount + standoff`` via-point (``−Y`` by
            # ``planner_stage1_approach_standoff``) into the hub, holding the
            # mount EE orientation fixed. The demount must be the EXACT reverse:
            # take the Stage-1 mount end-pose and translate it straight back
            # along the hub axis by the SAME standoff, keeping the orientation
            # identical. This yields a pure −Y straight pull-off (no yaw, no
            # x/z drift) instead of the previous ``_quat_align_z_to`` re-orient
            # that bent the path. Falls back to ``demount_axial_distance`` only
            # if no standoff is configured.
            s1_pos, s1_quat = self._compute_stage_end_ee_pose(1)
            so = self._hub_axis_standoff_vector()
            if so is None:
                hub_axis = np.asarray(self.cfg.hub_axis_world, dtype=np.float64)
                hub_axis = hub_axis / max(float(np.linalg.norm(hub_axis)), 1e-9)
                so = hub_axis * float(self.cfg.demount_axial_distance) * 1.2
            return (
                np.asarray(s1_pos, dtype=np.float64) + so,
                np.asarray(s1_quat, dtype=np.float64),
            )

        # Stage 3 — cradle return, tire stands on rails.
        tire_end_pos = np.asarray(self.cfg.tire_pickup_pos, dtype=np.float64)
        tire_end_quat = np.asarray(
            p.getQuaternionFromEuler(list(self.cfg.tire_spawn_rpy)),
            dtype=np.float64,
        )
        if have_grasp:
            return self._ee_pose_for_tire_pose(tire_end_pos, tire_end_quat)
        end_pos = tire_end_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
        return end_pos, palm_up

    def _generate_nominal_trajectory(self, start_pose, end_pose,
                                     total_steps: int = 100,
                                     lift: float = 0.0,
                                     orient_front_load_k: float = 0.0,
                                     approach_standoff: Optional[np.ndarray] = None,
                                     departure_standoff: Optional[np.ndarray] = None,
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a ``(N, 3) / (N, 4)`` nominal EE trajectory.

        ``start_pose`` and ``end_pose`` are each ``(pos, quat)`` tuples
        with ``pos`` shape (3,) and ``quat`` shape (4,) in PyBullet
        xyzw. Position is interpolated with a 5th-order min-jerk
        polynomial; orientation is interpolated with SLERP.

        **2026-06-01 (post-v3 hot-fix)** — ``lift`` adds an arched
        via-point at ``(start + end) / 2 + (0, 0, lift)``. A pure
        straight-line min-jerk through the air can pass *through* the
        UR10 base column for the cradle→hub carry: with cradle at
        ``(-1.90, 0, -0.13)`` and hub at ``(0, 0.80, -0.30)``, the
        midpoint sits ~0.43 m from the base origin and the IK has
        no shoulder solution there (``ik_residual_A_mean = 0.84 m``
        in the v3 run). Lifting the via-point by ~0.5 m forces the
        arm into a clear over-the-shoulder swing instead. ``lift = 0``
        falls back to the original single-segment min-jerk, which is
        still correct for short pickup / cradle-return motions where
        no obstacle lies between the endpoints.

        **2026-06-02 (D4 — orientation front-load).** ``orient_front_
        load_k`` warps the SLERP time parameter as
        ``s(t) = 1 - (1 - t)^k`` (k > 1 ⇒ rotation accelerated at
        start, settled at end). With k = 2.5 about 60 % of the
        rotation completes in the first 30 % of the trajectory, and
        > 95 % by 70 % — the policy/planner reaches the carry
        endpoint with the bore already aligned to the destination
        axis, leaving the final straight-shot for fine position
        control only. ``k <= 0`` falls back to uniform linear SLERP.
        Used for Stage 1 (carry: +X → -Y) and Stage 3 (return:
        -Y → +X) so the bore yaw recovers well *before* the tire
        enters the cradle gate (D1 fix, see ``step()``).
        """
        start_pos, start_quat = start_pose
        end_pos, end_quat = end_pose
        n = int(total_steps)

        start_arr = np.asarray(start_pos, dtype=np.float64).reshape(3)
        end_arr = np.asarray(end_pos, dtype=np.float64).reshape(3)
        has_approach = False
        has_departure = False
        if approach_standoff is not None:
            so = np.asarray(approach_standoff, dtype=np.float64).reshape(3)
            has_approach = float(np.linalg.norm(so)) > 1e-6
        if departure_standoff is not None:
            ds = np.asarray(departure_standoff, dtype=np.float64).reshape(3)
            has_departure = float(np.linalg.norm(ds)) > 1e-6
        if has_approach or has_departure:
            # Multi-via path supporting:
            #   * Stage 1 — [start, apex?, end+standoff, end] (+Y insertion)
            #   * Stage 3 — [start, start+standoff, apex?, end] (−Y extraction
            #     first, mirror of Stage 1)
            waypoints = [start_arr]
            if has_departure:
                ds = np.asarray(departure_standoff, dtype=np.float64).reshape(3)
                waypoints.append(start_arr + ds)
            if float(lift) > 1e-6:
                waypoints.append(
                    0.5 * (start_arr + end_arr)
                    + np.array([0.0, 0.0, float(lift)], dtype=np.float64))
            if has_approach:
                so = np.asarray(approach_standoff, dtype=np.float64).reshape(3)
                waypoints.append(end_arr + so)
            waypoints.append(end_arr)
            traj_pos = _multi_min_jerk_positions(waypoints, n)
        elif float(lift) > 1e-6:
            # Original symmetric arch: split the n steps equally between
            # the start→apex and apex→end segments (preserves the tested
            # carry timing the mount subsystem is tuned to).
            mid_pos = 0.5 * (start_arr + end_arr) + np.array(
                [0.0, 0.0, float(lift)], dtype=np.float64)
            half = max(1, n // 2)
            seg1 = _min_jerk_positions(start_arr, mid_pos, half + 1)
            seg2 = _min_jerk_positions(mid_pos, end_arr, n - half + 1)
            traj_pos = np.vstack([seg1[:-1], seg2[:-1], seg2[-1:]])
            traj_pos = traj_pos[:n] if traj_pos.shape[0] > n else traj_pos
            if traj_pos.shape[0] < n:
                pad = np.repeat(traj_pos[-1:], n - traj_pos.shape[0], axis=0)
                traj_pos = np.vstack([traj_pos, pad])
        else:
            traj_pos = _min_jerk_positions(start_arr, end_arr, n)

        if float(orient_front_load_k) > 1.0 + 1e-6:
            t_lin = np.linspace(0.0, 1.0, n, dtype=np.float64)
            k = float(orient_front_load_k)
            t_warped = 1.0 - np.power(1.0 - t_lin, k)
            traj_quat = _slerp_quats(start_quat, end_quat, n, times=t_warped)
        else:
            traj_quat = _slerp_quats(start_quat, end_quat, n)
        return traj_pos, traj_quat

    def compute_all_stage_trajectories(
        self, start_stage: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Nominal EE position paths for every remaining FSM stage.

        Chains each stage's start pose to the previous stage's end pose
        (the first stage starts at the current EE pose). Pure: does not
        mutate ``self._traj_*`` or any sim state — for GUI visualisation.
        Returns ``{stage: (N, 3) world XYZ}``.
        """
        out: Dict[int, np.ndarray] = {}
        if not bool(getattr(self.cfg, "use_planner_residual", True)):
            return out
        if self.robot_A is None:
            return out
        s0 = int(self.task_stage if start_stage is None else start_stage)
        start_pos, start_quat = self.robot_A.ee_pose()
        start_pos = np.asarray(start_pos, dtype=np.float64)
        start_quat = np.asarray(start_quat, dtype=np.float64)
        n = int(getattr(self.cfg, "planner_traj_steps", 100))
        for stage in range(s0, 4):
            end_pos, end_quat = self._compute_stage_end_ee_pose(stage)
            end_pos = np.asarray(end_pos, dtype=np.float64)
            end_quat = np.asarray(end_quat, dtype=np.float64)
            lift = 0.0
            if stage in (1, 3):
                # Stage 3 (return) mirrors the Stage-1 carry arch in reverse:
                # standoff → apex → cradle. Same lift keeps it the visual
                # reverse of the orange carry.
                lift = float(getattr(self.cfg, "planner_stage1_lift", 0.2))
            front_load_k = 0.0
            if stage in (1, 3):
                front_load_k = float(getattr(
                    self.cfg, "planner_yaw_front_load_k", 2.5,
                ))
            sq, eq = start_quat, end_quat
            if self._planner_palm_up_active(stage):
                sq = self._tilt_lock_palm_up_quat(sq)
                eq = self._tilt_lock_palm_up_quat(eq)
            standoff = self._stage_approach_standoff(stage)
            depart = self._stage_departure_standoff(stage)
            traj_pos, traj_quat = self._generate_nominal_trajectory(
                (start_pos, sq), (end_pos, eq),
                total_steps=n, lift=lift, orient_front_load_k=front_load_k,
                approach_standoff=standoff,
                departure_standoff=depart,
            )
            out[stage] = np.asarray(traj_pos, dtype=np.float64)
            start_pos = np.asarray(traj_pos[-1], dtype=np.float64)
            start_quat = np.asarray(traj_quat[-1], dtype=np.float64)
        return out

    def _hub_axis_standoff_vector(self) -> Optional[np.ndarray]:
        """``hub_axis_world * planner_stage1_approach_standoff``, or None."""
        d = float(getattr(self.cfg, "planner_stage1_approach_standoff", 0.0))
        if d <= 1e-6:
            return None
        hub_axis = np.asarray(self.cfg.hub_axis_world, dtype=np.float64)
        hub_axis = hub_axis / max(float(np.linalg.norm(hub_axis)), 1e-9)
        return hub_axis * d

    def _stage_approach_standoff(self, stage: int) -> Optional[np.ndarray]:
        """Pre-hub via offset for Stage 1 (+Y insertion leg at trajectory end)."""
        st = int(stage)
        if bool(getattr(self.cfg, "remount_cycle_enable", False)):
            st = {3: 1, 4: 2, 5: 3}.get(st, st)
        if st != 1:
            return None
        return self._hub_axis_standoff_vector()

    def _stage_departure_standoff(self, stage: int) -> Optional[np.ndarray]:
        """Departure via offset (disabled).

        **2026-06-05** — the coaxial −Y extraction leg now lives entirely in
        Stage 2 (demount), whose end-pose is the Stage-1 mount pose translated
        −Y by the standoff. Stage 3 therefore STARTS already at the standoff
        via-point and only has to arc back to the cradle (the reverse of the
        Stage-1 carry arch), so it needs no extra departure standoff. Returning
        None keeps Stage 3 a clean ``standoff → apex → cradle`` mirror of the
        carry instead of inserting a second redundant −Y leg.
        """
        return None

    def _replan_for_current_stage(self) -> None:
        """Rebuild the planner trajectory from the current EE pose.

        Called on:
          * reset (after every spawn branch)
          * each FSM stage transition (picked_up / mounted / demounted)

        Becomes a no-op if ``cfg.use_planner_residual = False`` so the
        legacy delta-EE path still works for backwards-compatibility
        evaluations (loaded v11c checkpoints, eval scripts).
        """
        if not bool(getattr(self.cfg, "use_planner_residual", True)):
            self._traj_pos = None
            self._traj_quat = None
            self._traj_q = None
            self.current_traj_step = 0
            self._traj_stall = 0
            return
        start_pos, start_quat = self.robot_A.ee_pose()
        end_pos, end_quat = self._compute_stage_end_ee_pose(int(self.task_stage))
        n = int(getattr(self.cfg, "planner_traj_steps", 100))
        # Stage 1 carry must arch over the UR10 base column to avoid
        # the IK dead-zone near the shoulder (see ``_generate_nominal
        # _trajectory`` docstring + diag_mount_pose.py). 0.5 m lift is
        # a tested compromise: high enough to clear the base+plinth at
        # z = -0.30 + ~0.30 (plinth) ≈ 0 m, low enough that the arched
        # midpoint stays in reach.
        lift = 0.0
        if int(self.task_stage) in (1, 3):
            # Stage 3 (return) arcs standoff → apex → cradle, the reverse of
            # the Stage-1 carry arch (same lift).
            lift = float(getattr(self.cfg, "planner_stage1_lift", 0.2))
        # **2026-06-02 (D4 — yaw front-loading)** — Stage 1 (cradle→hub
        # carry) and Stage 3 (hub→cradle return) both require a 90 °
        # yaw rotation about world +Z to keep the kinematically locked
        # tire bore aligned with the destination axis (hub axis −Y on
        # the way out, spawn axis +X on the way back). With uniform
        # SLERP the rotation completes only at the trajectory endpoint,
        # which means the bore is *still drifting* when the tire enters
        # the Stage 3 cradle gate (the source of the 55 % vertical_
        # violation rate in v3). Front-loading concentrates ~60 % of
        # the yaw in the first 30 % of the trajectory so the tire is
        # already at the destination yaw when it reaches the gate.
        # k = 2.5 is a balance between aggressive enough to clear the
        # cradle gate and gentle enough not to spike joint velocities
        # at trajectory start. ``planner_yaw_front_load_k <= 1.0``
        # disables the warp (linear SLERP fallback).
        front_load_k = 0.0
        # Carry (+X→−Y) and return (−Y→+X) legs need the 90° yaw front-
        # loaded. In the 6-stage cycle the return leg is Stage 5.
        _front_load_stages = (1, 5) if bool(
            getattr(self.cfg, "remount_cycle_enable", False)
        ) else (1, 3)
        if int(self.task_stage) in _front_load_stages:
            front_load_k = float(getattr(
                self.cfg, "planner_yaw_front_load_k", 2.5,
            ))
        # **2026-06-01 (Option C)** — tilt-lock endpoint projection only on
        # stages listed in ``planner_lock_palm_up_stages`` (default: 0).
        if self._planner_palm_up_active():
            start_quat = self._tilt_lock_palm_up_quat(
                np.asarray(start_quat, dtype=np.float64),
            )
            end_quat = self._tilt_lock_palm_up_quat(
                np.asarray(end_quat, dtype=np.float64),
            )
        standoff = self._stage_approach_standoff(int(self.task_stage))
        depart = self._stage_departure_standoff(int(self.task_stage))
        traj_pos, traj_quat = self._generate_nominal_trajectory(
            (np.asarray(start_pos, dtype=np.float64),
             np.asarray(start_quat, dtype=np.float64)),
            (np.asarray(end_pos, dtype=np.float64),
             np.asarray(end_quat, dtype=np.float64)),
            total_steps=n,
            lift=lift,
            orient_front_load_k=front_load_k,
            approach_standoff=standoff,
            departure_standoff=depart,
        )
        self._traj_pos = traj_pos
        self._traj_quat = traj_quat
        self.current_traj_step = 0
        self._traj_stall = 0
        if bool(getattr(self.cfg, "planner_precompute_joint_traj", True)):
            self._traj_q = self._bake_planner_joint_trajectory(traj_pos, traj_quat)
            self._traj_q = self._smooth_baked_joint_trajectory(self._traj_q)
        else:
            self._traj_q = None

    def _smooth_baked_joint_trajectory(self, traj_q: Optional[np.ndarray]
                                       ) -> Optional[np.ndarray]:
        """Centred moving-average over the baked joint sequence.

        Removes per-waypoint IK micro-branch chatter that otherwise reads
        as a high-frequency EE zig-zag when the baked joints are replayed
        one-per-control-step. The endpoints are preserved (shrinking
        window at the edges) so the start pose and the mount end pose are
        not pulled off-target. No-op for window ≤ 1 or short trajectories.
        """
        if traj_q is None:
            return None
        w = int(getattr(self.cfg, "planner_smooth_baked_window", 0))
        n = int(traj_q.shape[0])
        if w <= 1 or n < 3:
            return traj_q
        half = w // 2
        out = np.empty_like(traj_q)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = traj_q[lo:hi].mean(axis=0)
        # Pin the exact endpoints so the mount target pose is unchanged.
        out[0] = traj_q[0]
        out[-1] = traj_q[-1]
        return out

    def _bake_planner_joint_trajectory(
        self,
        traj_pos: np.ndarray,
        traj_quat: np.ndarray,
    ) -> np.ndarray:
        """IK each nominal waypoint once; chain warm-start for continuity.

        Runs inside a PyBullet state snapshot so intermediate
        ``resetJointState`` calls do not disturb the live sim. If a
        waypoint jumps to a distant IK branch (|Δq| > 0.8 rad), that
        index is re-solved once from ``HOME_POSE`` as rest.
        """
        ur = self.robot_A
        n = int(traj_pos.shape[0])
        # Warm-start every waypoint from HOME (``arm.rest``) so the baked
        # solution matches the live per-step ``apply_palm_up_pose`` path
        # (which also rest-warm-starts and reliably tracks the nominal to
        # the mount target). Chained current-state warm-start was found to
        # stall ~0.85 m short of the hub. We still progress the snapshot
        # joint state between waypoints so PyBullet's IK seed advances
        # along the trajectory.
        traj_q = np.zeros((n, ur.arm.lower.size), dtype=np.float64)
        rest = ur.arm.rest.copy()
        palm_up = self._planner_palm_up_active()
        state_id = p.saveState(physicsClientId=self.client)
        try:
            for i in range(n):
                pos = np.asarray(traj_pos[i], dtype=np.float64)
                quat = np.asarray(traj_quat[i], dtype=np.float64)
                # Match the live per-step path: when palm-up tilt-lock is
                # active it rewrites the nominal quat each step to be exactly
                # tool +Z = world +Z (yaw from the SLERP nominal). Baking
                # against the raw SLERP quat instead left the arm in a tilted
                # IK branch that stalled ~0.85 m short of the hub.
                if palm_up:
                    quat = self._tilt_lock_palm_up_quat(
                        np.asarray(quat, dtype=np.float64)
                    )
                q = ur.solve_arm_joints_in_snapshot(pos, quat, rest)
                traj_q[i] = q
                for jidx, qv in zip(ur.arm.indices, q):
                    p.resetJointState(
                        ur.uid, jidx,
                        targetValue=float(qv), targetVelocity=0.0,
                        physicsClientId=self.client,
                    )
        finally:
            p.restoreState(stateId=state_id, physicsClientId=self.client)
            p.removeState(stateUniqueId=state_id, physicsClientId=self.client)
        return traj_q

    # ------------------------------------------------------------------
    # Domain randomization scaffold (Phase 1 → Sim2Real bridge)
    # ------------------------------------------------------------------
    def _maybe_apply_domain_randomization(self) -> None:
        """Sample per-reset spawn noise on hub + cargo when enabled.

        Gating knobs (see ``src/config.py``):
          * ``USE_DOMAIN_RANDOMIZATION`` (bool, default ``False``)
            master switch. Off ⇒ both offsets are zero, scene build is
            bit-identical to the deterministic curriculum default.
          * ``RANDOM_POSITION_RANGE`` (float, metres, default ``0.02``)
            half-width of the uniform XY noise applied independently to
            the hub and cargo origins.

        Offsets are stored on ``self`` and consumed by the next
        ``Scene(...)`` constructor call inside ``reset()``. Sampling
        uses the env's seeded ``np_random`` so DR rollouts are exactly
        reproducible across runs that share a seed.
        """
        # Default to zero so the Scene build path stays deterministic
        # whenever DR is off (or RANDOM_POSITION_RANGE is non-positive).
        self._dr_hub_xy_offset = np.zeros(2, dtype=np.float64)
        self._dr_cargo_xy_offset = np.zeros(2, dtype=np.float64)

        if not bool(getattr(self.cfg, "USE_DOMAIN_RANDOMIZATION", False)):
            return

        rng = float(getattr(self.cfg, "RANDOM_POSITION_RANGE", 0.0))
        if rng <= 0.0:
            return

        self._dr_hub_xy_offset = self._np_random.uniform(
            -rng, rng, size=2
        ).astype(np.float64, copy=False)
        self._dr_cargo_xy_offset = self._np_random.uniform(
            -rng, rng, size=2
        ).astype(np.float64, copy=False)

    def _create_grasp_constraint_in_place(
        self, *, force_fixed: bool = False,
    ) -> None:
        """Bond the tire to the UR10 EE at the *current* world poses.

        Unlike :pymeth:`_attach_tire_to_robot_A`, this never teleports the
        tire. The child frame is derived from the live EE↔tire transform so
        the constraint is consistent with whatever geometry the simulator
        already has — critical for the attached-hot-start path where the
        tire must stay on the cradle pose while the EE meets the 6-o'clock
        grasp anchor. Also caches ``T_ee_tire`` for the planner end-pose
        inverse transform.
        """
        if self._grasp_constraint is not None:
            try:
                p.removeConstraint(
                    self._grasp_constraint, physicsClientId=self.client,
                )
            except p.error:
                pass
            self._grasp_constraint = None

        ee_pos, ee_orn = self.robot_A.ee_pose()
        tire_pos, tire_orn = self.scene.tire_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_orn = np.asarray(ee_orn, dtype=np.float64)
        tire_pos = np.asarray(tire_pos, dtype=np.float64)
        tire_orn = np.asarray(tire_orn, dtype=np.float64)

        if (
            not force_fixed
            and self._use_kinematic_tire_sync()
        ):
            self._begin_kinematic_grasp()
            return

        inv_tire_pos, inv_tire_orn = p.invertTransform(
            tire_pos.tolist(), tire_orn.tolist(),
        )
        child_pos, child_orn = p.multiplyTransforms(
            inv_tire_pos, inv_tire_orn,
            ee_pos.tolist(), ee_orn.tolist(),
        )

        self._grasp_kinematic = False
        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_A.uid,
            parentLinkIndex=self.robot_A.EE_LINK_INDEX,
            childBodyUniqueId=self.handles.tire,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=list(child_pos),
            childFrameOrientation=list(child_orn),
            physicsClientId=self.client,
        )
        p.changeConstraint(
            self._grasp_constraint,
            maxForce=1.0e6,
            erp=1.0,
            physicsClientId=self.client,
        )
        self._cache_grasp_relative_transform()

    def _attach_tire_to_robot_A(self) -> None:
        """Snap tire to UR10 EE and lock it with a rigid JOINT_FIXED so it
        is lifted vertically (X-Z tread plane) without sag or wobble.

        Geometry — exactly the spec the policy sees during training:
          * Tire orientation = Euler ``(0, −90°, 90°)`` → bore axis world −Y,
            tread in the world X-Z plane. Tire stands vertically.
          * Tire COM = EE + ``(0, 0, R)`` so the 6 o'clock outer tread point
            coincides with the EE position (margin = tire outer radius).
          * Constraint anchor on the tire = ``(−R, 0, 0)`` in tire local
            coordinates — by construction this is the 6 o'clock outer point
            in the body frame, independent of any drift in ``ee_pos``.
          * ``childFrameOrientation`` is derived from the explicit vertical
            ``tire_orn`` (rpy [0, −π/2, π/2]) and current ``ee_orn`` so the
            fixed joint records the exact EE↔tire pose at attach time.
          * ``changeConstraint(maxForce=1e6)`` stiffens the constraint so
            the 0.5 kg tire cannot pull itself off the EE under gravity.
        """
        ee_pos, ee_orn = self.robot_A.ee_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        R = float(self.cfg.tire_outer_radius)

        # 1. Tire orientation sourced from ``cfg.tire_spawn_rpy`` so the
        #    bonded-rigid grasp reproduces the exact pose the tire was
        #    spawned in (e.g. bore axis = world +X when rpy = (0, π/2, 0)).
        tire_orn = np.asarray(
            p.getQuaternionFromEuler(list(self.cfg.tire_spawn_rpy)),
            dtype=np.float64,
        )

        # 2. Tire COM offset from EE (tool tip or palm-up grasp anchor).
        tire_pos = ee_pos + self._grasp_com_offset_from_ee(ee_orn)

        # 3. Force the tire to the prescribed pose (overrides any spawn
        #    drift before the grasp).
        p.resetBasePositionAndOrientation(
            self.handles.tire,
            tire_pos.tolist(),
            tire_orn.tolist(),
            physicsClientId=self.client,
        )
        if self._grasp_constraint is not None:
            try:
                p.removeConstraint(self._grasp_constraint, physicsClientId=self.client)
            except p.error:
                pass

        # 4. childFramePosition = the 6 o'clock outer point expressed in
        #    tire local coordinates. Geometrically that point is at world
        #    (0, 0, -R) relative to the tire COM; inverse-rotate to tire
        #    local frame so the anchor is correct regardless of
        #    ``tire_spawn_rpy`` (independent of bore orientation).
        tire_R = np.array(
            p.getMatrixFromQuaternion(list(tire_orn)), dtype=np.float64,
        ).reshape(3, 3)
        local_anchor = tire_R.T @ np.array([0.0, 0.0, -R], dtype=np.float64)
        child_pos = local_anchor.tolist()

        # childFrameOrientation = relative orientation EE-in-tire-local so
        # the constraint records the EE wrist roll at attach time.
        inv_tire_pos, inv_tire_orn = p.invertTransform(
            tire_pos.tolist(), tire_orn.tolist(),
        )
        _, child_orn = p.multiplyTransforms(
            inv_tire_pos, inv_tire_orn,
            ee_pos.tolist(), ee_orn.tolist(),
        )

        if self._use_kinematic_tire_sync():
            self._begin_kinematic_grasp()
            return

        self._grasp_kinematic = False
        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_A.uid,
            parentLinkIndex=self.robot_A.EE_LINK_INDEX,
            childBodyUniqueId=self.handles.tire,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=child_pos,
            childFrameOrientation=list(child_orn),
            physicsClientId=self.client,
        )
        p.changeConstraint(
            self._grasp_constraint,
            maxForce=1.0e6,
            erp=1.0,
            physicsClientId=self.client,
        )
        self._cache_grasp_relative_transform()

    # ------------------------------------------------------------------
    # FSM constraint helpers — world-pin (floor) ↔ grasp (UR10 EE)
    # ------------------------------------------------------------------
    def _pin_tire_to_world(self, pos: np.ndarray, orn: np.ndarray) -> None:
        """Park the tire at ``(pos, orn)`` and make it fully static.

        Implementation: snap pose + zero velocities + ``changeDynamics
        (mass=0)``. A mass-zero rigid body in PyBullet is treated as static
        infrastructure (immune to gravity, collision impulses, constraint
        forces) so the standing-on-edge tire never tips over while waiting
        to be grasped or after a soft landing.

        ``_world_pin`` is repurposed as a non-zero sentinel meaning "static"
        — actual constraint creation is no longer required.
        """
        pos_list = np.asarray(pos, dtype=np.float64).tolist()
        orn_list = np.asarray(orn, dtype=np.float64).tolist()
        p.resetBasePositionAndOrientation(
            self.handles.tire, pos_list, orn_list,
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.handles.tire,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.handles.tire, -1, mass=0.0,
            physicsClientId=self.client,
        )
        self._world_pin = 1  # sentinel — "tire is static"

    def _release_world_pin(self) -> None:
        """Restore the tire to its configured dynamic mass before grasp."""
        if self._world_pin is not None:
            p.changeDynamics(
                self.handles.tire, -1,
                mass=float(self.cfg.tire_mass),
                physicsClientId=self.client,
            )
            self._world_pin = None

    def _attach_tire_to_hub(self) -> None:
        """Physically bond the seated tire to the hub (fixed constraint).

        Represents the tire being inserted onto the hub flange/pilot:
        the tire keeps its dynamic mass but is rigidly held by the (fixed-
        base) hub, so it stays put while Robot B tightens the bolts. The
        tire is first snapped to the exact seated pose (``tire_mount_pos``,
        bore ‖ ``hub_axis_world``) so the bond records a clean transform.
        """
        mount_pos = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
        hub_axis = np.asarray(self.cfg.hub_axis_world, dtype=np.float64)
        hub_axis = hub_axis / max(float(np.linalg.norm(hub_axis)), 1e-9)
        mount_orn = _quat_align_z_to(hub_axis)
        self._mount_seated_pos = mount_pos.copy()
        self._mount_seated_orn = np.asarray(mount_orn, dtype=np.float64).copy()

        # Restore dynamic mass (may have been mass=0 from a prior world-pin)
        # so the bonded tire reacts physically while held on the hub.
        self._release_world_pin()
        p.resetBasePositionAndOrientation(
            self.handles.tire, mount_pos.tolist(), mount_orn.tolist(),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.handles.tire, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )

        hub_ref = self.scene.handles.hub
        hub_uid = int(hub_ref.uid)
        hub_link = int(hub_ref.link_index)
        hub_pos, hub_orn = self.scene.hub_pose()
        inv_hub_pos, inv_hub_orn = p.invertTransform(
            np.asarray(hub_pos, dtype=np.float64).tolist(),
            np.asarray(hub_orn, dtype=np.float64).tolist(),
        )
        # Tire seated pose expressed in the hub link frame → parentFrame.
        parent_pos, parent_orn = p.multiplyTransforms(
            inv_hub_pos, inv_hub_orn,
            mount_pos.tolist(), mount_orn.tolist(),
        )
        if self._hub_mount_constraint is not None:
            try:
                p.removeConstraint(self._hub_mount_constraint, physicsClientId=self.client)
            except p.error:
                pass
            self._hub_mount_constraint = None
        self._hub_mount_constraint = p.createConstraint(
            parentBodyUniqueId=hub_uid,
            parentLinkIndex=hub_link,
            childBodyUniqueId=self.handles.tire,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(parent_pos),
            parentFrameOrientation=list(parent_orn),
            childFramePosition=[0.0, 0.0, 0.0],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )
        p.changeConstraint(
            self._hub_mount_constraint,
            maxForce=1.0e6, erp=1.0,
            physicsClientId=self.client,
        )

    def _begin_mount_hold(self) -> None:
        """Seat the tire on the hub and freeze the arm in a holding pose.

        The UR10 keeps its (kinematic) grasp pose frozen at the mount
        config so the gripper visually stays on the tire — the "holding
        steady while Robot B tightens the bolts" feel — while the tire is
        physically bonded to the hub via ``_attach_tire_to_hub``.
        """
        if bool(getattr(self.cfg, "pin_tire_on_mount", True)):
            self._attach_tire_to_hub()
        # Stop the kinematic EE→tire sync from fighting the hub bond, but
        # keep the arm where it is so the gripper still appears to hold.
        self._grasp_kinematic = False
        self._mount_frozen_q, _ = self.robot_A.joint_state()
        self._mount_hold_left = int(getattr(self.cfg, "mount_hold_steps", 0))
        self._mount_hold_finish_term = False

    def _pin_tire_at_hub_mount(self) -> None:
        """Bond the seated tire to the hub (alias kept for terminate path)."""
        self._attach_tire_to_hub()

    def _release_grasp(self) -> None:
        if self._grasp_constraint is not None:
            try:
                p.removeConstraint(
                    self._grasp_constraint, physicsClientId=self.client,
                )
            except p.error:
                pass
            self._grasp_constraint = None
        self._grasp_kinematic = False
        self._grasp_yaw_ee0 = None
        self._grasp_com_offset_ee = None
        self._safe_tire_pos = None
        self._safe_tire_orn = None

    def _release_hub_mount(self) -> None:
        if self._hub_mount_constraint is not None:
            try:
                p.removeConstraint(
                    self._hub_mount_constraint, physicsClientId=self.client,
                )
            except p.error:
                pass
            self._hub_mount_constraint = None

    def _try_stage_transitions(self) -> Dict[str, Any]:
        """Drive task_stage 0 → 1 → 2 → 3 → done. Returns a dict of FSM events.

        Events emitted in ``info`` keys (v6 4-stage FSM):
          * ``picked_up`` — Stage 0 → 1 (grasp constraint born, R_pickup paid)
          * ``mounted``   — Stage 1 → 2 (tire seated on hub, R_mount paid)
          * ``demounted`` — Stage 2 → 3 (tire pulled clear of hub after the
                            ``demount_stall_steps`` stall, R_demount paid)
          * ``landed``    — Stage 3 → done success (tire back on cradle pose,
                            R_success / R_return paid)

        The Stage 2 demount stall (``cfg.demount_stall_steps``) means the
        demount gate ignores ``d_hub > demount_axial_distance`` for the
        first N steps after Stage 1 fires — physically modelling the
        gap between Panda finishing its bolt-down and UR10 starting its
        retract.
        """
        events: Dict[str, Any] = {
            "picked_up": False, "mounted": False,
            "demounted": False, "landed": False,
            "retracted": False, "regripped": False,
        }
        remount = bool(getattr(self.cfg, "remount_cycle_enable", False))
        ee_pos, _ = self.robot_A.ee_pose()
        tire_pos, _ = self.scene.tire_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        tire_pos = np.asarray(tire_pos, dtype=np.float64)
        R = float(self.cfg.tire_outer_radius)

        if self.task_stage == 0:
            grasp_target = tire_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
            if float(np.linalg.norm(ee_pos - grasp_target)) < float(self._approach_tol):
                self._release_world_pin()
                self._attach_tire_to_robot_A()
                self.task_stage = 1
                self._pickup_bonus_paid = True
                events["picked_up"] = True
                self._maybe_promote_to_fixed_grasp()
                self._prev_d_approach = None
                self._prev_d_return = None

        elif self.task_stage == 1:
            hub_pos, _ = self.scene.hub_pose()
            mount_target = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
            d_mount = float(np.linalg.norm(tire_pos - mount_target))
            tire_axis = self.scene.tire_axis()
            hub_axis = self.scene.hub_axis()
            theta = float(np.arccos(
                np.clip(np.dot(tire_axis, hub_axis), -1.0, 1.0)
            ))
            # v6: runtime-tunable mount gate. ``_mount_radius_tol`` and
            # ``_mount_angle_tol`` default to the config values but the
            # ``MountTolCurriculumCallback`` (src/train.py) overrides them
            # via env_method() each rollout boundary so the gate can fade
            # from loose (e.g. 0.20 m / 25°) to hard (0.04 m / 5°) over
            # the early curriculum steps.
            mount_tol = float(getattr(self, "_mount_radius_tol",
                                       self.cfg.mount_radius_tol))
            mount_ang_tol = float(getattr(self, "_mount_angle_tol",
                                           self.cfg.reward.delta_A))
            mounted = (d_mount < mount_tol) and (theta < mount_ang_tol)
            # v11c — one-shot Phase A force-mount override. Set by
            # ``_apply_reverse_phase_a_hot_start`` so the first
            # ``step()`` after a hot-start always fires the mount
            # event, regardless of how far the first physics
            # decimation translated the tire. This guarantees the
            # R_mount sparse signal per Phase A reset.
            if getattr(self, "_phase_a_force_mount_first_step", False):
                self._phase_a_force_mount_first_step = False
                mounted = True
            if mounted and not self._mount_bonus_paid:
                self._mount_bonus_paid = True
                self.task_stage = 2
                self._mount_done_step = int(self._step_count)
                events["mounted"] = True
                if remount:
                    # 6-stage cycle: seat + bond the tire to the hub and
                    # freeze the arm holding it while "Robot B tightens the
                    # nuts" (W1). After the hold elapses (handled in the
                    # Stage-2 block below) the grasp is released and the
                    # empty gripper retracts to HOME.
                    self._attach_tire_to_hub()
                    self._grasp_kinematic = False
                    self._mount_frozen_q, _ = self.robot_A.joint_state()
                    self._mount_hold_left = int(
                        getattr(self.cfg, "tighten_hold_steps", 0)
                    )
                    self._mount_hold_finish_term = False
                else:
                    hold_steps = int(getattr(self.cfg, "mount_hold_steps", 0))
                    term_on = str(getattr(self.cfg, "terminate_on", "never")).lower()
                    if hold_steps > 0 and term_on == "mount":
                        self._begin_mount_hold()
                    elif bool(getattr(self.cfg, "pin_tire_on_mount", True)) and term_on == "mount":
                        self._pin_tire_at_hub_mount()
                self._prev_d_approach = None
                self._prev_d_return = None
                # v11: reset Stage 2 PB shaping accumulator.
                self._prev_d_hub = None

        elif self.task_stage == 2 and remount:
            # 6-stage cycle S2 — empty-handed retract to HOME after W1.
            # ``step()`` decrements ``_mount_hold_left`` and freezes the arm
            # while the hold is active; once it elapses we release the grasp
            # (tire stays bonded to the hub) and let the planner drive the
            # empty gripper back to the cached HOME EE pose.
            if self._mount_hold_left <= 0:
                if not self._retract_release_done:
                    self._release_grasp()
                    self._retract_release_done = True
                    # Replan now (task_stage == 2 → HOME end pose) so the
                    # planner pulls the empty arm away from the hub.
                    self._replan_for_current_stage()
                home = getattr(self, "_home_ee_pos", None)
                if home is not None:
                    d_home = float(np.linalg.norm(ee_pos - np.asarray(home, dtype=np.float64)))
                    if (d_home < float(getattr(self.cfg, "home_return_radius_tol", 0.12))
                            and not self._retract_bonus_paid):
                        self._retract_bonus_paid = True
                        self.task_stage = 3
                        events["retracted"] = True
                        self._prev_d_approach = None
                        self._prev_d_return = None

        elif self.task_stage == 3 and remount:
            # 6-stage cycle S3 — re-approach + re-grip the hub-mounted tire.
            # The tire sits at ``tire_mount_pos`` (bore ‖ hub axis); its
            # 6 o'clock outer point is the grasp anchor. On contact we bond
            # the tire back to the EE, release the hub bond and start the
            # W2 loosen hold.
            grasp_target = tire_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
            if (float(np.linalg.norm(ee_pos - grasp_target))
                    < float(getattr(self.cfg, "regrip_radius_tol", 0.10))
                    and not self._regrip_bonus_paid):
                self._create_grasp_constraint_in_place(force_fixed=True)
                self._release_hub_mount()
                self._regrip_bonus_paid = True
                self.task_stage = 4
                events["regripped"] = True
                # Hold steady while "Robot B loosens the nuts" (W2).
                self._grasp_kinematic = False
                self._mount_frozen_q, _ = self.robot_A.joint_state()
                self._mount_hold_left = int(getattr(self.cfg, "loosen_hold_steps", 0))
                self._mount_hold_finish_term = False
                self._mount_done_step = int(self._step_count)
                self._prev_d_approach = None
                self._prev_d_return = None
                self._prev_d_hub = None

        elif self.task_stage == 4 and remount:
            # 6-stage cycle S4 — demount: after W2, pull the tire clear of
            # the hub along the hub axis.
            if self._mount_hold_left <= 0:
                if not self._demount_replan_done:
                    # Replan toward the demount pull-off pose once W2 ends.
                    self._replan_for_current_stage()
                    self._demount_replan_done = True
                hub_pos, _ = self.scene.hub_pose()
                d_hub = float(np.linalg.norm(
                    tire_pos - np.asarray(hub_pos, dtype=np.float64)
                ))
                if (d_hub > float(self.cfg.demount_axial_distance)
                        and not self._demount_bonus_paid):
                    self._demount_bonus_paid = True
                    self.task_stage = 5
                    events["demounted"] = True
                    self._prev_d_approach = None
                    self._prev_d_return = None

        elif self.task_stage == 5 and remount:
            # 6-stage cycle S5 — carry to the rack + soft landing (success).
            d_return = float(np.linalg.norm(tire_pos - self._pickup_pos_world))
            lin_vel, _ = p.getBaseVelocity(
                self.handles.tire, physicsClientId=self.client,
            )
            descend_speed = abs(float(lin_vel[2]))
            landed = (
                d_return < float(getattr(
                    self.cfg, "rack_return_radius_tol",
                    self.cfg.return_radius_tol,
                ))
                and descend_speed < self.cfg.landing_speed_max
            )
            if landed:
                self._release_grasp()
                _, cur_orn = self.scene.tire_pose()
                self._pin_tire_to_world(tire_pos, cur_orn)
                events["landed"] = True

        elif self.task_stage == 2:
            # v6 demount: tire must travel ≥ ``demount_axial_distance`` away
            # from the hub centre. Stall enforces a minimum step count
            # between mount and demount eligibility (virtual fastener-release
            # interval). Once the stall elapses, ``d_hub`` is computed every
            # step and the gate fires as soon as the threshold is crossed.
            hub_pos, _ = self.scene.hub_pose()
            d_hub = float(np.linalg.norm(tire_pos - np.asarray(hub_pos, dtype=np.float64)))
            stall = int(getattr(self.cfg, "demount_stall_steps", 0))
            stall_elapsed = (
                self._mount_done_step is not None
                and (self._step_count - self._mount_done_step) >= stall
            )
            if stall_elapsed and (not self._demount_bonus_paid):
                if d_hub > float(self.cfg.demount_axial_distance):
                    self._demount_bonus_paid = True
                    self.task_stage = 3
                    events["demounted"] = True
                    self._prev_d_approach = None
                    self._prev_d_return = None

        elif self.task_stage == 3:
            d_return = float(np.linalg.norm(tire_pos - self._pickup_pos_world))
            lin_vel, _ = p.getBaseVelocity(
                self.handles.tire, physicsClientId=self.client,
            )
            descend_speed = abs(float(lin_vel[2]))
            landed = (
                d_return < float(getattr(
                    self.cfg, "rack_return_radius_tol",
                    self.cfg.return_radius_tol,
                ))
                and descend_speed < self.cfg.landing_speed_max
            )
            if landed:
                self._release_grasp()
                _, cur_orn = self.scene.tire_pose()
                self._pin_tire_to_world(tire_pos, cur_orn)
                events["landed"] = True

        # **2026-06-01 — planner replan on stage transitions.** Whenever
        # the FSM advances we rebuild the nominal trajectory from the
        # *current* EE pose to the new stage's end-pose so the policy
        # always operates against a fresh min-jerk reference. ``landed``
        # is the terminal event and needs no replan (episode is over).
        skip_mount_replan = (
            events.get("mounted")
            and (
                # In the 6-stage cycle the arm is frozen holding during the
                # W1 tighten hold; the replan toward HOME is issued inline
                # when the hold elapses (Stage-2 block above).
                remount
                or (
                    int(getattr(self.cfg, "mount_hold_steps", 0)) > 0
                    and str(getattr(self.cfg, "terminate_on", "never")).lower() == "mount"
                )
            )
        )
        # ``retracted`` (S2 → S3) needs a fresh plan toward the regrip
        # anchor. ``demounted`` toward the rack-return / pull-off pose.
        if (events.get("picked_up") or events.get("demounted")
                or events.get("retracted")):
            self._replan_for_current_stage()
        elif events.get("mounted") and not skip_mount_replan:
            self._replan_for_current_stage()

        return events

    def _tire_vertical_error(self) -> float:
        """Angle (rad) between current tire bore axis and the *spawn*
        reference axis (``cfg.tire_spawn_axis_world``, e.g. world +X).

        Stages 0 / 2 use this directly to enforce the pickup-zone pose.
        Stage 1 carve-out (see ``step``/``_compute_reward``) waives the
        penalty + termination so the policy can rotate the tire toward
        ``hub_axis_world`` for mount without being penalised.
        """
        cur_axis = self.scene.tire_axis()
        ref_axis = np.array(self.cfg.tire_spawn_axis_world, dtype=np.float64)
        ref_axis = ref_axis / max(float(np.linalg.norm(ref_axis)), 1e-9)
        c = float(np.clip(np.dot(cur_axis, ref_axis), -1.0, 1.0))
        return float(np.arccos(c))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.cfg.action.dim)
        action = np.clip(action, -1.0, 1.0)
        if self._mount_hold_left > 0 and self._mount_frozen_q is not None:
            self.robot_A.drive_arm_targets(self._mount_frozen_q)
        else:
            self._apply_action(action)

        for _ in range(self.cfg.decimation):
            p.stepSimulation(physicsClientId=self.client)
        if self._mount_hold_left <= 0:
            self._enforce_robot_a_palm_up()
        if self._grasp_kinematic and self._use_kinematic_tire_sync():
            self._sync_grasped_tire_upright()

        self._step_count += 1
        if self._mount_hold_left > 0:
            # Clamp the bonded tire to its seated pose so it visibly sits
            # still on the hub while Robot B "tightens the bolts" — the
            # JOINT_FIXED bond alone drifts a few cm under tread/bolt
            # contact, so we re-assert the seated pose + zero velocity.
            if self._mount_seated_pos is not None:
                p.resetBasePositionAndOrientation(
                    self.handles.tire,
                    self._mount_seated_pos.tolist(),
                    self._mount_seated_orn.tolist(),
                    physicsClientId=self.client,
                )
                p.resetBaseVelocity(
                    self.handles.tire, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                    physicsClientId=self.client,
                )
            self._mount_hold_left -= 1
            if self._mount_hold_left == 0:
                self._mount_hold_finish_term = True
        # FSM transitions run AFTER physics steps so the trigger checks see
        # the realised post-action world (EE position / tire pose / velocity).
        fsm_events = self._try_stage_transitions()
        # Evaluate world-state checks once per step; reuse for both reward
        # penalty computation and termination logic so they cannot diverge.
        in_collision = self._in_bad_collision()
        out_of_ws = self._out_of_workspace()
        cforce_max = self._max_contact_normal_force()
        damaged = (
            self.cfg.contact_force_terminate_above > 0.0
            and cforce_max >= self.cfg.contact_force_terminate_above
        )
        vertical_err = self._tire_vertical_error()
        # Stage 1 (carry/mount) waives the vertical-pose gate so the
        # policy can rotate the tire 90° about world +Z to align the
        # bore axis with ``hub_axis_world`` for mount. Stage 2 (demount)
        # also waives — the tire is still bore-aligned to the hub.
        #
        # **2026-06-02 (D1 — Stage 3 cradle vertical violation no longer
        # terminates).** Previously Stage 3 inside ``stage3_vertical_gate
        # _radius`` (0.20 m of cradle) terminated the episode with
        # ``R_fail = -50`` whenever ``vertical_err > vertical_tol_rad``.
        # In ``phase1_grad_v3`` this accounted for ~55 % of all
        # terminations because the planner needs the entire return
        # trajectory to slerp the bore from world −Y (hub axis) back
        # to world +X (spawn axis) — when the tire crossed the cradle
        # gate before the SLERP completed, the gate fired and the
        # episode died before any cradle-landing signal could be
        # collected.
        #
        # The dense ``vertical_pen`` term in ``_compute_reward`` keeps
        # shaping the tire toward the spawn yaw (still gated by
        # ``stage3_vertical_gate_radius`` so it only acts inside the
        # cradle neighbourhood), but the *episode survives*. Termination
        # is now restricted to Stage 0 (HOME → cradle approach), which
        # is the only safety-critical stage left: the tire is mass=0
        # pinned at the cradle by ``tire_initial_pin``, so a violation
        # here would indicate a real fault (kinematic lock corruption).
        vertical_violated = (
            self.task_stage == 0
            and vertical_err > self.cfg.vertical_tol_rad
        )
        mount_residuals = self.scene.tire_hub_mount_residuals()
        obs = self._compute_obs(mount_residuals)
        reward, breakdown = self._compute_reward(
            action, in_collision, out_of_ws, mount_residuals,
            fsm_events, vertical_err,
        )
        terminated, truncated, term_info = self._check_termination(
            breakdown, in_collision, out_of_ws, damaged,
            vertical_violated, fsm_events,
        )
        # Failure-termination penalty: counteracts the "die fast to stop
        # losing reward" exploit when dense shaping has any negative
        # baseline. Applied iff the episode ends *non-successfully* via one
        # of the safety gates (vertical / collision / workspace /
        # contact-force). ``truncated`` (max_steps reached) does NOT trigger
        # this — that's a legitimate timeout, not a failure mode.
        if terminated and not breakdown.is_success:
            fail_pen = float(self.cfg.reward.R_fail)
            reward += fail_pen
            breakdown.fail_pen = fail_pen
            breakdown.total = float(reward)

        # IK tracking residual = ||target_pos - achieved_pos|| after physics
        # settled. A persistent gap means IK saturated (joint limits or
        # unreachable target) and the policy's Δ command isn't being executed.
        ik_res_A = self._ik_residual(self.robot_A)
        ik_res_B = self._ik_residual(self.robot_B)

        info: Dict[str, Any] = {
            "reward_terms": breakdown.__dict__,
            "step": self._step_count,
            "contact_force_max": float(cforce_max),
            "ik_residual_A": ik_res_A,
            "ik_residual_B": ik_res_B,
            "task_stage": int(self.task_stage),
            "tire_vertical_err_rad": float(vertical_err),
            **fsm_events,
            **term_info,
        }
        self._prev_action = action.copy()
        self._prev_d_A = breakdown.d_A
        self._prev_d_B = breakdown.d_B
        return obs, reward, terminated, truncated, info

    def _ik_residual(self, robot) -> float:
        """Norm of (last IK target EE position − achieved EE position)."""
        target = robot.last_target_pos
        if target is None:
            return 0.0
        achieved, _ = robot.ee_pose()
        return float(np.linalg.norm(achieved - target))

    def _planner_palm_up_active(self, stage: Optional[int] = None) -> bool:
        """True when palm-up tilt-lock applies to the given FSM stage."""
        if not bool(getattr(self.cfg, "planner_lock_palm_up", True)):
            return False
        st = int(self.task_stage if stage is None else stage)
        allowed = tuple(
            int(x) for x in getattr(
                self.cfg, "planner_lock_palm_up_stages", (0,),
            )
        )
        return st in allowed

    def _tilt_lock_palm_up_quat(
        self, nominal_quat: np.ndarray,
    ) -> np.ndarray:
        """Project ``nominal_quat`` onto the tilt-locked subspace.

        Returns a unit quaternion whose:
          * tool +Z axis is **exactly** world +Z (palm-up, no tilt), and
          * tool +X axis is the projection of the nominal's tool +X
            onto the world XY plane (so yaw tracks the planner SLERP).

        This is "Option B" from the 2026-06-01 design discussion: the
        wrist is allowed to **yaw** freely about the vertical so the
        tire bore can be rotated 90° from its spawn direction (world +X)
        to the hub direction (world −Y) for mount, while the palm still
        always faces the sky. If the nominal's tool +X is exactly
        vertical (yaw indeterminate at the singularity), we fall back
        to ``FINAL_LOCK_QUATERNION``'s yaw.
        """
        Rn = np.array(
            p.getMatrixFromQuaternion(list(nominal_quat)), dtype=np.float64,
        ).reshape(3, 3)
        nx_world = Rn[:, 0]  # tool +X expressed in world frame
        # Project to XY plane.
        nx_xy = np.array([nx_world[0], nx_world[1], 0.0], dtype=np.float64)
        n_xy = float(np.linalg.norm(nx_xy))
        if n_xy < 1e-6:
            # Singular: nominal tool +X is (nearly) vertical. Use the
            # canonical palm-up yaw from the home pose.
            return robot_a_lock_quaternion(self.robot_A)
        tool_x_w = nx_xy / n_xy
        tool_z_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tool_y_w = np.cross(tool_z_w, tool_x_w)
        # Build rotation matrix R_target (world<-tool) with columns
        # = world-frame basis vectors of the tool axes.
        Rt = np.column_stack([tool_x_w, tool_y_w, tool_z_w])
        # scipy returns (x, y, z, w), matching PyBullet's quat order.
        return np.asarray(
            Rotation.from_matrix(Rt).as_quat(), dtype=np.float64,
        )

    def _advance_traj_index(self, nominal_pos: np.ndarray) -> None:
        """Advance ``current_traj_step`` with a waypoint **arrival gate**.

        2026-06-04 — the index used to increment every control step
        regardless of whether the arm had reached the current nominal
        waypoint. Near reach-saturation the stiff PD lags the commanded
        target by 10–40 cm, so the index raced ahead of the arm and the
        realised EE path zig-zagged behind the Min-Jerk plan (the
        "오락가락" carry). Now the index only advances when the *measured*
        EE is within ``planner_waypoint_pos_tol_m`` of the current
        waypoint, OR a stall watchdog (``planner_waypoint_max_stall``
        control steps) force-advances it so an unreachable pose can't
        freeze the trajectory and time the episode out. The result is a
        realised path that tracks the plan closely → smooth, robot-like
        motion.

        Default-OFF: the baked Min-Jerk joint trajectory is played one
        waypoint per control step and reaches the hub cleanly, so the
        gate is only useful for the per-step EE-IK path. When disabled
        the index simply advances every step.
        """
        if not bool(getattr(self.cfg, "planner_waypoint_gate_enable", False)):
            self.current_traj_step += 1
            self._traj_stall = 0
            return
        tol = float(getattr(self.cfg, "planner_waypoint_pos_tol_m", 0.04))
        max_stall = int(getattr(self.cfg, "planner_waypoint_max_stall", 10))
        try:
            ee_now, _ = self.robot_A.ee_pose()
            err = float(np.linalg.norm(
                np.asarray(ee_now, dtype=np.float64)
                - np.asarray(nominal_pos, dtype=np.float64)
            ))
        except Exception:
            err = 0.0
        self._traj_stall = int(getattr(self, "_traj_stall", 0)) + 1
        if err <= tol or self._traj_stall >= max(1, max_stall):
            self.current_traj_step += 1
            self._traj_stall = 0

    def _apply_action(self, action: np.ndarray) -> None:
        """Dispatch the policy action to the robots.

        Two regimes, selected by ``cfg.use_planner_residual``:

        * **Planner + residual (2026-06-01 default).** ``action[0:3]``
          is treated as a per-step *residual XYZ offset* in metres,
          scaled by ``cfg.planner_pos_offset_scale`` and added to the
          current nominal pose ``self._traj_pos[idx]``. Orientation is
          driven by the SLERP-interpolated nominal quaternion alone
          unless ``cfg.planner_enable_rot_offset`` flips on the
          ``action[3:6]`` rotation residual (axis-angle, scaled by
          ``cfg.planner_rot_offset_scale``). The combined absolute
          (pos, quat) is forwarded to :pymeth:`UR10Robot.apply_absolute_ee`,
          which runs full 6-DOF IK without the legacy ``ur10_lock_tool_up``
          hack — that lock was the cure for the broken delta-accumulator
          path; the planner handles orientation now.

        * **Legacy raw-delta.** When the flag is off, behaviour is bit-
          identical to the v11c-era path so old eval scripts / loaded
          checkpoints keep working.

        Robot B (Panda) is unchanged: frozen at HOME in Phase 1, fully
        delta-driven in Phase 2/3.
        """
        use_planner = bool(getattr(self.cfg, "use_planner_residual", True))
        traj_ready = (
            use_planner
            and self._traj_pos is not None
            and self._traj_quat is not None
        )

        if traj_ready:
            n = int(self._traj_pos.shape[0])
            idx = int(min(self.current_traj_step, n - 1))
            nominal_pos = np.asarray(self._traj_pos[idx], dtype=np.float64)
            nominal_quat = np.asarray(self._traj_quat[idx], dtype=np.float64)

            pos_scale = float(getattr(
                self.cfg, "planner_pos_offset_scale", 0.15,
            ))
            pos_off = np.asarray(action[0:3], dtype=np.float64) * pos_scale
            final_pos = nominal_pos + pos_off

            # **2026-06-01 (palm-up tilt-lock, Option B)** — keep the
            # gripper's tool +Z axis aligned with world +Z (palm faces
            # up), but **let the wrist yaw freely** to track the
            # planner's SLERP nominal. This satisfies the user-visible
            # "gripper +Z always points up" constraint while preserving
            # the 90°-yaw rotation needed to align the tire bore (world
            # +X at spawn) with the hub axis (world −Y) during Stage 1
            # mount. Implementation: extract a clean yaw from the
            # nominal SLERP target and rebuild a tilt-corrected
            # quaternion whose tool +Z is exactly world +Z.
            #
            # Derivation: ``FINAL_LOCK_QUATERNION`` has tool +Z = world
            # +Z and tool +X aligned with world −Y (yaw_base = −π/2).
            # The nominal's *intended* yaw is recovered by mapping its
            # tool +X onto the world XY plane and taking
            # ``atan2(world_x_y, world_x_x)`` then offsetting from the
            # base yaw. Multiplying ``Rz(Δyaw) ⊗ FINAL_LOCK_QUATERNION``
            # gives the desired palm-up + correctly-yawed target.
            lock_palm_up = self._planner_palm_up_active()
            if lock_palm_up:
                nominal_quat = self._tilt_lock_palm_up_quat(nominal_quat)

            enable_rot = bool(getattr(
                self.cfg, "planner_enable_rot_offset", False,
            ))
            if enable_rot and len(action) >= 6 and not lock_palm_up:
                rot_scale = float(getattr(
                    self.cfg, "planner_rot_offset_scale", 0.15,
                ))
                aa = np.asarray(action[3:6], dtype=np.float64) * rot_scale
                rot_quat = axisangle3_to_quat(aa)
                # Apply residual rotation in the world frame (left-mul).
                # Quaternion *multiplication*, never addition — additive
                # quat math is the classic source of NaN drift here.
                final_quat = quat_multiply(rot_quat, nominal_quat)
            else:
                final_quat = nominal_quat.copy()

            # Defensive normalisation — Slerp output is unit-norm but
            # the rot-offset compose can drift on the 6th decimal.
            qn = float(np.linalg.norm(final_quat))
            if qn > 1e-9:
                final_quat = final_quat / qn
            else:
                final_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

            residual_active = float(np.max(np.abs(action[0:3]))) >= 1e-4
            use_dls = bool(getattr(self.cfg, "use_dls_cartesian_servo", False)) and (
                residual_active
                or bool(getattr(self.cfg, "planner_dls_always", False))
            )
            use_baked = (
                not use_dls
                and self._traj_q is not None
                and bool(getattr(self.cfg, "planner_precompute_joint_traj", True))
                and not residual_active
            )
            if use_dls:
                # 2026-06-04 — closed-loop DLS resolved-rate servo toward
                # the nominal+residual EE pose. Replaces per-step absolute
                # IK to kill the 40–70 cm EE snaps near the hub singularity.
                self.robot_A.drive_ee_servo_dls(
                    final_pos, final_quat,
                    damping=float(getattr(self.cfg, "planner_dls_damping", 0.06)),
                    max_joint_step=float(getattr(
                        self.cfg, "planner_dls_max_joint_step", 0.10)),
                    pos_gain=float(getattr(self.cfg, "planner_dls_pos_gain", 1.0)),
                    orn_gain=float(getattr(self.cfg, "planner_dls_orn_gain", 0.8)),
                    adaptive=bool(getattr(self.cfg, "planner_dls_adaptive", True)),
                    manip_threshold=float(getattr(
                        self.cfg, "planner_dls_manip_threshold", 0.02)),
                )
                self._advance_traj_index(nominal_pos)
            elif use_baked:
                # 2026-06-03 — play back joint targets baked at replan
                # time (chained warm-start IK). Avoids per-step HOME-
                # anchored IK in ``apply_palm_up_pose``, which was the
                # main source of visible tremor with zero policy residual.
                q_cmd = np.asarray(self._traj_q[idx], dtype=np.float64)
                self.robot_A.last_target_pos = nominal_pos.copy()
                self.robot_A.drive_arm_targets(q_cmd)
                self._advance_traj_index(nominal_pos)
            else:
                hold_q = getattr(self, "_planner_hold_arm_targets", None)
                skip_ik = (
                    hold_q is not None
                    and int(self.current_traj_step) == 0
                    and not residual_active
                )
                # When a baked joint trajectory exists, warm-start the
                # residual-offset IK from the baked solution for this
                # waypoint so the realised path hugs the clean baked
                # branch (+ small offset) instead of snapping elsewhere.
                warm_q = None
                if (
                    self._traj_q is not None
                    and bool(getattr(
                        self.cfg,
                        "planner_residual_warmstart_from_baked", True,
                    ))
                ):
                    warm_q = np.asarray(self._traj_q[idx], dtype=np.float64)
                if skip_ik:
                    self.robot_A.last_target_pos = final_pos.copy()
                    self.robot_A.drive_arm_targets(hold_q)
                    self._planner_hold_arm_targets = None
                elif lock_palm_up:
                    self.robot_A.apply_palm_up_pose(
                        final_pos, final_quat, warm_q=warm_q,
                    )
                else:
                    self.robot_A.apply_absolute_ee(final_pos, final_quat)
                self._advance_traj_index(nominal_pos)
        else:
            # Legacy raw-delta path (kept for backwards-compatibility).
            ps = self.cfg.action.pos_scale
            rs = self.cfg.action.rot_scale
            d_pos_A = action[0:3] * ps
            if bool(getattr(self.cfg, "ur10_lock_tool_up", True)):
                d_rot_A = np.zeros(3, dtype=np.float64)
            else:
                d_rot_A = action[3:6] * rs
            self.robot_A.apply_delta_ee(d_pos_A, d_rot_A)

        # Robot B: in Phase 1 (``freeze_robot_b=True``) Panda is parked at
        # HOME and ``reset_to_home`` re-seeds the motor target every step
        # so it doesn't sag under gravity. In Phase 2/3 ``action[6:12]``
        # carries Δpose_B and is forwarded to IK.
        if self.cfg.freeze_robot_b:
            self.robot_B.reset_to_home()
        else:
            ps = self.cfg.action.pos_scale
            rs = self.cfg.action.rot_scale
            d_pos_B = action[6:9] * ps
            d_rot_B = action[9:12] * rs
            self.robot_B.apply_delta_ee(d_pos_B, d_rot_B)
        # Gripper A: sim-side no-op (auto-grasp constraint holds the tire).
        # In Phase 1 the channel is dropped from the action vector entirely;
        # in Phase 2/3 it lives at ``action[12]`` and only feeds back into
        # the obs via ``prev_action`` so the policy can still emit intent.

    # Fixed Robot B joint slots in obs (Panda=7; UR10e=6 padded with zeros).
    _ROBOT_B_OBS_ARM_DOFS: int = 7

    @staticmethod
    def _pad_obs_vector(v: np.ndarray, size: int) -> np.ndarray:
        """Pad or truncate a 1-D obs slice to a fixed width for checkpoint compat."""
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        n = int(size)
        if v.size == n:
            return v
        if v.size > n:
            return v[:n].copy()
        out = np.zeros(n, dtype=np.float64)
        out[: v.size] = v
        return out

    # ------------------------------------------------------------------
    # Observation (spec §2.1 base + mating scalars ⇒ 89-d)
    # ------------------------------------------------------------------
    def _compute_obs(self,
                     mount_residuals: Optional[Tuple[float, float, float]] = None,
                     ) -> np.ndarray:
        # ``mount_residuals`` may be ``None`` only on the first call from
        # ``reset()`` (before ``step()`` shares its precomputed value).
        obs_cfg = self.cfg.obs
        ws = obs_cfg.workspace_radius
        vmax = obs_cfg.max_joint_vel

        qA, dqA = self.robot_A.joint_state()
        qB, dqB = self.robot_B.joint_state()

        # joint pos to [-1,1] using each robot's own limits
        qA_n = 2 * (qA - self.robot_A.arm.lower) / np.maximum(self.robot_A.arm.range, 1e-6) - 1
        qB_n = 2 * (qB - self.robot_B.arm.lower) / np.maximum(self.robot_B.arm.range, 1e-6) - 1
        dqA_n = np.clip(dqA / vmax, -1.0, 1.0)
        dqB_n = np.clip(dqB / vmax, -1.0, 1.0)
        b_dof = int(self._ROBOT_B_OBS_ARM_DOFS)
        qB_n = self._pad_obs_vector(qB_n, b_dof)
        dqB_n = self._pad_obs_vector(dqB_n, b_dof)

        eeA_pos, eeA_orn = self.robot_A.ee_pose()
        eeB_pos, eeB_orn = self.robot_B.ee_pose()
        tire_pos, tire_orn = self.scene.tire_pose()
        hub_pos, hub_orn = self.scene.hub_pose()
        bolt_pos, bolt_orn = self.scene.bolt_pose()

        # ----- Robot B-centric observation frame -----
        # All positional channels below are expressed as vectors from
        # the Panda (Robot B) base to the body of interest. Under the
        # Robot B-centric world layout introduced in ``config.py`` the
        # Panda base sits at ``(0,0,0)`` so this is numerically a no-op
        # today, but the explicit subtraction (1) documents the obs
        # convention and (2) makes the policy invariant to any future
        # global translation of the world frame (the real Panda's base
        # is the canonical reference for Sim2Real transfer).
        obs_ref = getattr(self.cfg, "obs_reference_pos", None)
        rb_base = np.asarray(
            obs_ref if obs_ref is not None else self.cfg.robot_B_base_pos,
            dtype=np.float64,
        )
        eeA_pos_rel = eeA_pos - rb_base
        eeB_pos_rel = eeB_pos - rb_base
        tire_pos_rel = tire_pos - rb_base
        hub_pos_rel = hub_pos - rb_base
        bolt_pos_rel = bolt_pos - rb_base

        # Relative tire→hub (position diff + axis-angle rotation error).
        # Position differences are translation-invariant so the choice
        # of base frame does not affect them.
        rel_th_pos = tire_pos - hub_pos
        rel_th_rot = relative_axisangle(tire_orn, hub_orn)
        # Relative gripperB→bolt
        rel_eb_pos = eeB_pos - bolt_pos
        rel_eb_rot = relative_axisangle(eeB_orn, bolt_orn)

        if mount_residuals is None:
            mount_residuals = self.scene.tire_hub_mount_residuals()
        axial_th, lateral_th, lug_spin = mount_residuals
        ax_t = self.cfg.reward.success_axial_dot_target
        n_b = max(3, int(self.cfg.n_bolts))
        lug_scale = max((math.pi / float(n_b)), 1e-6)
        lug_ch = float(np.clip(lug_spin / lug_scale, 0.0, 1.0))
        mount_tail = np.array(
            [
                (axial_th - ax_t) / ws,
                lateral_th / ws,
                lug_ch,
            ],
            dtype=np.float64,
        )

        # v7: hub_guide_vector = (hub_pos - eeA_pos) — direct 3-d direction
        # from the UR10 EE toward the hub centre, normalised by workspace
        # radius so the channel lives in roughly the same scale as the
        # other relative-position observations. The norm of this vector
        # is also exposed indirectly via the Stage-1 ``guide_A`` reward
        # term (env-side); putting the raw 3-d vector in the observation
        # lets the policy condition the carry trajectory on a direct
        # bearing cue rather than re-deriving it from the joint state.
        hub_guide_vector = (hub_pos - eeA_pos) / ws

        parts = [
            qA_n, dqA_n,                              # 12
            qB_n, dqB_n,                              # 14
            eeA_pos_rel / ws, eeA_orn,                # 7
            eeB_pos_rel / ws, eeB_orn,                # 7
            tire_pos_rel / ws, tire_orn,              # 7
            hub_pos_rel / ws, hub_orn,                # 7
            bolt_pos_rel / ws, bolt_orn,              # 7
            rel_th_pos / ws, rel_th_rot / np.pi,      # 6
            rel_eb_pos / ws, rel_eb_rot / np.pi,      # 6
            self._prev_action,                         # action.dim
            mount_tail,                               # 3
        ]
        if bool(getattr(self.cfg, "include_hub_guide_obs", True)):
            parts.append(hub_guide_vector)            # 3  ← v7 vector guide
        obs = np.concatenate(parts).astype(np.float32)

        # Phase 1 feature isolation: zero the Panda-side channels so the
        # policy sees them as a constant 0 signal during Phase 1, but
        # the dimension stays 89-d for checkpoint compatibility. Mount
        # diagnostics (last 3) are preserved — they remain meaningful
        # for the Stage-1 mounting reward even when Panda is frozen.
        obs_mask = self._build_obs_mask()
        if not bool(np.all(obs_mask == 1.0)):
            obs = obs * obs_mask.astype(np.float32)

        assert obs.shape[0] == obs_cfg.dim, f"obs dim {obs.shape[0]} != {obs_cfg.dim}"
        return obs

    # ------------------------------------------------------------------
    # Reward (Phase 1 FSM — stage-dispatched dense + always-on penalties)
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray, in_collision: bool,
                        out_of_workspace: bool,
                        mount_residuals: Tuple[float, float, float],
                        fsm_events: Dict[str, Any],
                        vertical_err: float,
                        ) -> Tuple[float, rewards.RewardBreakdown]:
        rcfg = self.cfg.reward
        b = rewards.RewardBreakdown()

        ee_pos, _ = self.robot_A.ee_pose()
        tire_pos, _ = self.scene.tire_pose()
        hub_pos, _ = self.scene.hub_pose()
        tire_axis = self.scene.tire_axis()
        hub_axis = self.scene.hub_axis()
        _, dqA = self.robot_A.joint_state()

        # Align term: tire ↔ hub geometry. Computed in every stage for
        # diagnostic logging, only injected into the dense reward in
        # Stage 1 (carry).
        b.align_A, b.d_A, b.theta_A = rewards.align_reward(
            tire_pos, hub_pos, tire_axis, hub_axis, rcfg)

        # Panda-side terms (reach_B) only matter when Robot B is active.
        # Phase 1 freezes Panda at HOME and zeroes ``w_d_B`` / ``w_theta_B``
        # in ``make_reward_config`` — skip the bolt/EE queries entirely.
        if not bool(getattr(self.cfg, "freeze_robot_b", False)):
            eeB_pos, eeB_orn = self.robot_B.ee_pose()
            bolt_pos, _bolt_orn = self.scene.bolt_pose()
            bolt_axis = self.scene.bolt_axis()
            eeB_z = quat_axis(eeB_orn, "z")
            b.reach_B, b.d_B, b.theta_B = rewards.reach_reward(
                eeB_pos, bolt_pos, eeB_z, bolt_axis, rcfg)
        # else: ``b.reach_B`` / ``b.d_B`` / ``b.theta_B`` stay at 0.0.

        ax_th, lat_th, lug_e = mount_residuals
        b.axial_dot_th = float(ax_th)
        b.lateral_th = float(lat_th)
        b.lug_spin_err_rad = float(lug_e)

        # Stage-specific dense terms ---------------------------------------
        # Positive shaping (exponential kernel) — keeps the dense reward
        # bounded in ``[0, w]`` so the policy isn't punished by simply
        # surviving a step. This was the fix for the "self-terminate to
        # stop losing reward" failure mode observed when stage-0 used the
        # raw negative-distance form ``- w * d``.
        R = float(self.cfg.tire_outer_radius)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        tire_pos = np.asarray(tire_pos, dtype=np.float64)
        # Potential-based (PB) shaping accumulator — added on top of the
        # bounded ``exp`` kernel so distant states still see a non-zero
        # gradient as the EE / tire close in on the goal. The PB term is
        # ``w_pb * (prev_d - curr_d)``; positive when progressing.
        pb_step = 0.0
        remount = bool(getattr(self.cfg, "remount_cycle_enable", False))
        ts = int(self.task_stage)

        if remount and ts == 2:
            # 6-stage S2 — retract empty gripper toward HOME. Reuse the
            # bounded return kernel (slot ``return_A``) keyed on the
            # EE→HOME distance so the policy is rewarded for pulling the
            # arm clear of the hub after releasing the mounted tire.
            home = getattr(self, "_home_ee_pos", None)
            d_home = (
                float(np.linalg.norm(ee_pos - np.asarray(home, dtype=np.float64)))
                if home is not None else 0.0
            )
            decay = max(float(rcfg.return_decay), 1e-3)
            b.return_A = float(rcfg.w_return) * float(np.exp(-d_home / decay))
            b.d_return = d_home
            if self._prev_d_return is not None and rcfg.w_pb_return > 0.0:
                pb_step = float(rcfg.w_pb_return) * float(self._prev_d_return - d_home)
            self._prev_d_return = d_home
            b.align_A = 0.0
            b.approach_A = 0.0
            b.d_approach = 0.0
        elif remount and ts == 3:
            # 6-stage S3 — re-approach the hub-mounted tire's 6 o'clock
            # grasp anchor. Reuse the Stage-0 approach kernel.
            grasp_target = tire_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
            d_approach = float(np.linalg.norm(ee_pos - grasp_target))
            decay = max(float(rcfg.approach_decay), 1e-3)
            b.approach_A = float(rcfg.w_approach) * float(np.exp(-d_approach / decay))
            b.d_approach = d_approach
            if self._prev_d_approach is not None and rcfg.w_pb_approach > 0.0:
                pb_step = float(rcfg.w_pb_approach) * float(
                    self._prev_d_approach - d_approach
                )
            self._prev_d_approach = d_approach
            b.align_A = 0.0
        elif self.task_stage == 0:
            # Stage 0 — approach: shape UR10 EE toward the tire's 6 o'clock outer point.
            grasp_target = tire_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
            d_approach = float(np.linalg.norm(ee_pos - grasp_target))
            decay = max(float(rcfg.approach_decay), 1e-3)
            far_term = float(rcfg.w_approach) * float(np.exp(-d_approach / decay))
            close_w = float(getattr(rcfg, "w_approach_close", 0.0))
            close_decay = max(
                float(getattr(rcfg, "approach_close_decay", 0.2)), 1e-3,
            )
            close_term = close_w * float(np.exp(-d_approach / close_decay))
            # v11c — distance gate on the Stage-0 dense kernel. When the
            # EE is farther than ``approach_A_gate`` from the grasp
            # anchor, both ``far_term`` and ``close_term`` are zeroed so
            # the policy cannot plateau by hovering at d_approach ≈
            # 0.20–0.30 m collecting +1.5/step (v4 / v9b failure mode).
            # Below the gate the kernel is unchanged.
            gate = float(getattr(self.cfg, "approach_A_gate", 999.0))
            if d_approach > gate:
                far_term = 0.0
                close_term = 0.0
            b.approach_A = far_term + close_term
            b.d_approach = d_approach
            if self._prev_d_approach is not None and rcfg.w_pb_approach > 0.0:
                pb_step = float(rcfg.w_pb_approach) * float(
                    self._prev_d_approach - d_approach
                )
            self._prev_d_approach = d_approach
            b.align_A = 0.0
        elif self.task_stage == 1:
            # Stage 1 — carry/mount (v7 vector-guided overhaul).
            # Two positive kernels replace the legacy negative align_A:
            #   1. guide_A: w_guide * exp(-||hub - ee|| / guide_decay)
            #      strong pull on the EE-to-hub vector. Bounded in
            #      [0, w_guide]; non-vanishing across the full 2-m reach
            #      because guide_decay = 0.5 m.
            #   2. pb_carry: w_pb_carry * (prev_d_A - d_A)
            #      potential-based shaping on tire-to-hub distance — pays
            #      positively for every step that closes the carry gap.
            ee_to_hub = float(np.linalg.norm(
                np.asarray(hub_pos, dtype=np.float64) - ee_pos
            ))
            decay_g = max(float(getattr(rcfg, "guide_decay", 0.5)), 1e-3)
            b.guide_A = float(rcfg.w_guide) * float(np.exp(-ee_to_hub / decay_g))
            b.d_guide = ee_to_hub
            if self._prev_d_A is not None and float(rcfg.w_pb_carry) > 0.0:
                pb_step = float(rcfg.w_pb_carry) * float(self._prev_d_A - b.d_A)
            self._prev_d_A = b.d_A
            b.pb_carry = float(pb_step)
            # Move the PB contribution from the generic pb_shape slot to
            # the carry-dedicated slot so dense_total_A doesn't double-
            # count it (stage_dense_A already adds b.pb_carry below).
            pb_step = 0.0
            # Suppress the legacy align_A signal so the policy isn't
            # pulled by a negative-only kernel during carry.
            b.align_A = 0.0
            b.approach_A = 0.0
            b.d_approach = 0.0
        elif self.task_stage == 2 or (remount and ts == 4):
            # Stage 2 — demount: reward GROWING distance from hub centre
            # so the policy axially pulls the tire clear of the studs.
            # (6-stage cycle: S4 reuses this kernel after the W2 hold.)
            # Kernel: r = w_pull * (1 - exp(-d_hub / pull_decay))
            #   d_hub = 0    →  0.0
            #   d_hub = τ    →  ~0.63 · w_pull
            #   d_hub = 3τ   →  ~0.95 · w_pull
            # Saturates near ``w_pull`` once the tire reaches the demount
            # target distance (~30 cm with τ = 20 cm). During the stall,
            # the kernel still emits the same value — but the demount gate
            # is closed, so the policy cannot collect R_demount by racing
            # away during the safety hold (it still earns the dense pull
            # term, which is the desired behaviour — start the retract).
            hub_pos_v, _ = self.scene.hub_pose()
            d_hub = float(np.linalg.norm(
                tire_pos - np.asarray(hub_pos_v, dtype=np.float64)
            ))
            pull_w = float(getattr(rcfg, "w_pull_demount", 0.0))
            pull_decay = max(
                float(getattr(rcfg, "pull_decay", 0.20)), 1e-3,
            )
            b.demount = pull_w * (1.0 - float(np.exp(-d_hub / pull_decay)))
            b.d_demount = d_hub
            # **v11**: PB shaping on Δd_hub so each backward step pays
            # immediately, mirroring Stage-1 pb_carry. Sign: positive
            # when tire moves *away* from hub. Stored in ``pb_carry``
            # slot to keep diagnostics readable (Stage 1 has already
            # reset _prev_d_A on the mount transition; Stage 2 reuses
            # ``_prev_d_hub`` initialised below).
            pb_demount_w = float(getattr(rcfg, "w_pb_demount", 0.0))
            if pb_demount_w > 0.0:
                if not hasattr(self, "_prev_d_hub") or self._prev_d_hub is None:
                    self._prev_d_hub = d_hub
                pb_demount_step = pb_demount_w * (d_hub - self._prev_d_hub)
                self._prev_d_hub = d_hub
                b.pb_carry = float(pb_demount_step)
            else:
                b.pb_carry = 0.0
            # Track stall countdown for log-friendly diagnostics.
            if self._mount_done_step is not None:
                stall_left = max(
                    0,
                    int(getattr(self.cfg, "demount_stall_steps", 0))
                    - (self._step_count - self._mount_done_step),
                )
            else:
                stall_left = 0
            b.stage2_stall_left = int(stall_left)
            b.align_A = 0.0
            b.approach_A = 0.0
        else:  # task_stage == 3 (cradle return / soft landing)
            d_return = float(np.linalg.norm(tire_pos - self._pickup_pos_world))
            decay = max(float(rcfg.return_decay), 1e-3)
            b.return_A = float(rcfg.w_return) * float(np.exp(-d_return / decay))
            b.d_return = d_return
            if self._prev_d_return is not None and rcfg.w_pb_return > 0.0:
                pb_step = float(rcfg.w_pb_return) * float(
                    self._prev_d_return - d_return
                )
            self._prev_d_return = d_return
            lin_vel, _ = p.getBaseVelocity(
                self.handles.tire, physicsClientId=self.client,
            )
            v_descend = abs(float(lin_vel[2]))
            b.landing = -rcfg.w_landing_speed * v_descend
            b.d_v_descend = v_descend
            b.align_A = 0.0
            b.approach_A = 0.0
        b.pb_shape = float(pb_step)

        # FSM transition bonuses (paid exactly once per event).
        # v6 (4-stage FSM): picked_up → mounted → demounted → landed.
        if fsm_events.get("picked_up"):
            b.fsm_bonus += float(rcfg.R_pickup)
        if fsm_events.get("mounted"):
            b.fsm_bonus += float(rcfg.R_mount)
        # 6-stage cycle intermediate bonuses (fall back to mount/pickup
        # magnitudes when dedicated fields are not set).
        if fsm_events.get("retracted"):
            b.fsm_bonus += float(getattr(rcfg, "R_retract",
                                         getattr(rcfg, "R_mount", 0.0)))
        if fsm_events.get("regripped"):
            b.fsm_bonus += float(getattr(rcfg, "R_regrip",
                                         getattr(rcfg, "R_pickup", 0.0)))
        if fsm_events.get("demounted"):
            b.fsm_bonus += float(getattr(rcfg, "R_demount", 0.0))
        if fsm_events.get("landed"):
            # Final success bonus — prefer the new ``R_success`` field, fall
            # back to the legacy ``R_return`` alias for backwards compat.
            b.fsm_bonus += float(getattr(rcfg, "R_success",
                                         getattr(rcfg, "R_return", 0.0)))
            b.is_success = True

        # Vertical-pose dense penalty (squared angle, bounded by tol²).
        # Mirrors the Stage-1 carve-out applied to the termination gate
        # in ``step``: during carry/mount the tire is permitted to rotate
        # toward ``hub_axis_world`` (-Y) and so the penalty is zeroed.
        # v6: Stage 2 (demount) also waives the gate — the tire stays
        # mated to the hub during the stall and the policy needs the
        # freedom to retract along the hub axis without an axis-pose
        # whip-back penalty.
        # **2026-06-02 (Stage 3 cradle-return penalty fix)** — Stage 3
        # also waives the dense penalty until the tire approaches the
        # cradle (``stage3_vertical_gate_radius``). Without this the
        # policy received a -w_vertical * (90°)² = ~-2.5 / step
        # penalty over the entire return leg, drowning out the
        # ``return_A`` shaping signal and leaving no learnable gradient
        # toward the cradle landing.
        # The "cradle-return" stage is Stage 3 in the legacy FSM and
        # Stage 5 in the 6-stage remount cycle. Likewise, every stage where
        # the tire is held bore-aligned to the hub (carry/mount/retract/
        # regrip/demount) must waive the vertical penalty.
        return_stage = 5 if remount else 3
        hub_aligned_stages = (1, 2, 3, 4) if remount else (1, 2)
        stage3_gate_on = True
        if self.task_stage == return_stage:
            tire_pos_pen, _ = self.scene.tire_pose()
            d_cradle_pen = float(np.linalg.norm(
                np.asarray(tire_pos_pen, dtype=np.float64)
                - self._pickup_pos_world
            ))
            stage3_gate_on = d_cradle_pen < float(getattr(
                self.cfg, "stage3_vertical_gate_radius", 0.20,
            ))
        if self.task_stage in hub_aligned_stages or not stage3_gate_on:
            b.vertical_pen = 0.0
        else:
            b.vertical_pen = -rcfg.w_vertical * float(vertical_err ** 2)

        # Always-on penalties --------------------------------------------
        b.coop = rewards.coop_reward(b.d_A, b.d_B, rcfg)
        b.sync_joint_A = rewards.sync_joint_a_penalty(dqA, rcfg)
        # Legacy ``success_bonus`` removed (2026-05-28). Phase 1 success is
        # decided by the FSM ``landed`` event in ``_try_stage_transitions``;
        # the lug-aligned predicate is no longer evaluated to save compute.
        b.success = 0.0
        b.collision = rewards.collision_penalty(in_collision, rcfg)
        b.workspace = rewards.workspace_penalty(out_of_workspace, rcfg)
        # Phase 1 action/jerk mask — when Robot B is frozen, the policy
        # cannot influence the world via action[6:12]. Including those
        # channels in the L2 penalty would force PPO to drive them to
        # zero for no return, wasting gradient signal. The mask zeros
        # the Panda slice while preserving action/obs dimensions (13-d
        # / 89-d) so checkpoints remain forward-compatible to Phase 2/3.
        action_mask = self._build_action_mask()
        b.action = rewards.action_penalty(action, rcfg, mask=action_mask)
        b.jerk = rewards.jerk_penalty(
            action, self._prev_action, rcfg, mask=action_mask,
        )
        b.shape_A = rewards.shaping_reward(self._prev_d_A, b.d_A, rcfg.w_shape_A)
        b.shape_B = rewards.shaping_reward(self._prev_d_B, b.d_B, rcfg.w_shape_B)

        # Dense subtotal (for logging). v7 carry overhaul:
        #   0 → approach_A   (EE → grasp anchor, exp positive)
        #   1 → guide_A + pb_carry  (NEW: EE→hub exp + PB Δd_A)
        #   2 → demount      (axial pull from hub)
        #   3 → return_A + landing  (cradle return + soft-landing penalty)
        # Stage 0/2/3 reuse ``b.pb_shape`` for their respective PB terms;
        # Stage 1's PB carry lives in ``b.pb_carry`` to keep diagnostics
        # readable. Only one of (pb_shape, pb_carry) is non-zero per step.
        # v11: Stage 2 reuses ``b.pb_carry`` for the PB-demount term
        # (positive when Δd_hub > 0). Stage 3's PB return continues to
        # live in ``b.pb_shape`` (handled in the Stage 3 branch via
        # ``pb_step``), so we don't double-count here.
        stage_dense_A = {
            0: float(b.approach_A),
            1: float(b.guide_A) + float(b.pb_carry),
            2: float(b.demount) + float(b.pb_carry),
            3: float(b.return_A) + float(b.landing),
        }[int(self.task_stage)]
        common = ("coop", "sync_joint_A", "collision", "workspace",
                  "action", "jerk", "vertical_pen")
        b.dense_total_pre_mix = float(
            stage_dense_A + float(b.pb_shape) + float(b.reach_B)
            + sum(getattr(b, k) for k in common)
        )

        # Final aggregate — dense process + sparse stage bonus.
        # Per-step alive cost is appended AFTER the mix so it always bites
        # by exactly ``w_step_alive`` per step regardless of dense / sparse
        # weighting. This breaks the v2 hover-lockin where the policy's
        # value function preferred soaking 350 step of dense reward over
        # firing a single 25-pt R_pickup; with -0.05/step the agent pays
        # up to -25 just for letting an episode run to ``max_steps``.
        b.step_alive = -float(getattr(rcfg, "w_step_alive", 0.0))
        b.total = float(
            rcfg.mix_dense * b.dense_total_pre_mix
            + rcfg.mix_sparse_success * float(b.fsm_bonus)
            + b.step_alive
        )
        return b.total, b

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _in_bad_collision(self) -> bool:
        # Robot bodies vs ground plane (and, with a real pit, the rim slabs
        # that form the floor surface outside the pit — an arm link punching
        # through them is just as bad as punching the infinite plane).
        floor_bodies = [self.handles.plane]
        floor_bodies.extend(getattr(self.handles, "floor_rim", []) or [])
        for robot in (self.robot_A, self.robot_B):
            for floor_uid in floor_bodies:
                cps = p.getContactPoints(bodyA=robot.uid, bodyB=floor_uid,
                                         physicsClientId=self.client)
                for cp in cps:
                    # link 0 / -1 is the base; foot contact is fine, mid-arm isn't.
                    if cp[3] > 1:    # link index on robot side
                        return True
        # Robot A vs Robot B — ignore base-link grazing (large arms).
        min_link = int(getattr(self.cfg, "robot_ab_collision_min_link", 2))
        cps = p.getContactPoints(bodyA=self.robot_A.uid, bodyB=self.robot_B.uid,
                                 physicsClientId=self.client)
        for cp in cps:
            if cp[3] > min_link or cp[4] > min_link:
                return True
        if self.handles.vehicle is not None:
            for robot in (self.robot_A, self.robot_B):
                cpsv = p.getContactPoints(
                    bodyA=robot.uid, bodyB=self.handles.vehicle,
                    physicsClientId=self.client,
                )
                for cp in cpsv:
                    if cp[3] > 1 or cp[4] > 1:
                        return True
        # **2026-06-02 (cargo penetration fix)** — also count the static
        # back-wall slab (separate body) so the policy is penalised when an
        # arm link clips it. Without this the back wall existed visually but
        # was invisible to the collision penalty.
        back_wall = getattr(self.handles, "cargo_back_wall", None)
        if back_wall is not None:
            for robot in (self.robot_A, self.robot_B):
                cpsw = p.getContactPoints(
                    bodyA=robot.uid, bodyB=back_wall,
                    physicsClientId=self.client,
                )
                for cp in cpsw:
                    if cp[3] > 1:
                        return True
        # **2026-06-02 (tire-vs-cargo)** — flag tire penetration into the
        # cargo / back-wall as a bad-collision event. The kinematic
        # upright lock teleports the tire AFTER the physics step, so a
        # ``getContactPoints`` call here would only see the pre-sync
        # contacts (stale). ``getClosestPoints`` returns geometric
        # overlap for the CURRENT body poses regardless of the last
        # ``stepSimulation``, so it correctly fires once the kinematic
        # tire ends up inside the wall.
        # NOTE: tire-vs-truck (hub) is *expected* during the Stage 1→2
        # mount event and during Stage 3 cradle landing — we deliberately
        # do NOT include those bodies here.
        tire_uid = self.handles.tire
        tire_obstacles: List[int] = []
        if self.handles.vehicle is not None:
            tire_obstacles.append(int(self.handles.vehicle))
        if back_wall is not None:
            tire_obstacles.append(int(back_wall))
        # Same penetration tolerance the kinematic-sync revert uses
        # (5 mm). Anything shallower is treated as PyBullet's contact
        # margin / numerical fuzz and ignored.
        pen_tol = -0.005
        for obs_uid in tire_obstacles:
            cps_t = p.getClosestPoints(
                bodyA=tire_uid, bodyB=obs_uid, distance=0.0,
                physicsClientId=self.client,
            )
            for cp in cps_t:
                if len(cp) > 8 and float(cp[8]) < pen_tol:
                    return True
        return False

    def _max_contact_normal_force(self) -> float:
        """Largest |normal force| across contacts between the policy-relevant
        bodies (robots + tire). Excludes two noise sources:

        * **Scene-internal contacts** (e.g. subdivided cargo collision boxes
          abutting each other) — huge solver forces but not penalisable.
        * **Self-collisions** (URDF_USE_SELF_COLLISION makes gripper fingers
          touch each other; the simulation reports tens of thousands of
          newtons that have nothing to do with the policy's actions).
        """
        relevant = {
            self.robot_A.uid,
            self.robot_B.uid,
            self.handles.tire,
        }
        mx = 0.0
        pts = p.getContactPoints(physicsClientId=self.client)
        for cp in pts:
            if len(cp) <= 9:
                continue
            bodyA, bodyB = cp[1], cp[2]
            if (bodyA not in relevant) and (bodyB not in relevant):
                continue
            # Skip self-collisions — gripper-finger contacts produce huge
            # spike forces from the URDF self-collision flag.
            if bodyA == bodyB:
                continue
            # **2026-06-01 (planner-residual)** — while the JOINT_FIXED
            # grasp is active, PyBullet reports large normal forces on
            # *every* tire-vs-X contact (gripper, rail, hub, plane,
            # cargo). The tire is being **deliberately** transported by
            # the planner trajectory — none of those contacts represent
            # "policy damage". v1 only filtered tire↔UR10 which still
            # let the cradle rails (3.6 kN) and hub mount-touchdown
            # (>10 kN) terminate the episode in 1–48 steps with the
            # default 2500 N gate, producing the v1 monitor.csv pattern
            # of all-contact-force terminations.
            #
            # **2026-06-01 (v2)** — broaden the filter: while the grasp
            # is active, ignore *all* contacts involving the tire body.
            # Robot-vs-plane / robot-vs-vehicle / robot-vs-robot contacts
            # are still counted, so genuine arm crashes still terminate.
            if self._is_tire_grasped():
                tire_uid = self.handles.tire
                if bodyA == tire_uid or bodyB == tire_uid:
                    continue
            try:
                mx = max(mx, abs(float(cp[9])))
            except (TypeError, ValueError):
                pass
        return mx

    def _check_termination(self, b: rewards.RewardBreakdown,
                           in_collision: bool, out_of_workspace: bool,
                           contact_damage: bool,
                           vertical_violated: bool,
                           fsm_events: Dict[str, Any],
                           ) -> Tuple[bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {"is_success": b.is_success}
        # Final Phase-1 success — tire landed on the cradle (Stage 3 done).
        if fsm_events.get("landed"):
            info["is_success"] = True
            return True, False, {**info, "termination": "success"}

        # v6 curriculum brake-lock — short-circuit success on intermediate
        # FSM events when ``cfg.terminate_on`` is set. The downstream
        # stages' code remains active; only the early-termination gate
        # is moved. Resolution order: explicit ``terminate_on`` enum
        # wins; legacy ``terminate_on_pickup=True`` falls back to
        # ``terminate_on = "pickup"`` for backwards compat.
        terminate_on = str(getattr(self.cfg, "terminate_on", "never")).lower()
        if terminate_on == "never" and bool(
            getattr(self.cfg, "terminate_on_pickup", False)
        ):
            terminate_on = "pickup"
        early_event = {
            "pickup":   ("picked_up", "pickup_success"),
            "mount":    ("mounted", "mount_success"),
            "demount":  ("demounted", "demount_success"),
        }.get(terminate_on)
        if early_event is not None:
            evt_key, term_tag = early_event
            if evt_key == "mounted" and int(getattr(self.cfg, "mount_hold_steps", 0)) > 0:
                if getattr(self, "_mount_hold_finish_term", False):
                    self._mount_hold_finish_term = False
                    b.is_success = True
                    info["is_success"] = True
                    return True, False, {**info, "termination": term_tag}
            elif fsm_events.get(evt_key):
                b.is_success = True
                info["is_success"] = True
                return True, False, {**info, "termination": term_tag}
        # v11c2 (2026-05-31) — safety-termination master switch. Phase A
        # of the reverse curriculum teleports the tire into the hub which
        # produces a chaotic first-step physics burst (vertical wobble,
        # huge contact forces, transient workspace overshoot). With
        # ``safety_terminations_enabled = False`` the ``vertical``,
        # ``collision``, ``workspace`` and ``contact_force`` gates are
        # skipped so the episode can survive long enough for the policy
        # to collect post-mount data. Phase B/C re-enable the gates via
        # ``set_safety_terminations(True)`` so the production task keeps
        # its damage / wobble protection.
        safety_on = bool(getattr(self.cfg, "safety_terminations_enabled", True))
        if safety_on:
            # Strict vertical-pose gate — tire must stay upright through pick,
            # carry and place. Violation = penalty termination.
            if vertical_violated:
                return True, False, {**info, "termination": "vertical_violation"}
            # v7: collision termination is now opt-in via ``collision_terminates``.
            # Default (False) lets the episode survive a glancing rack/cargo
            # contact while the policy still pays the per-step ``w_collision``
            # penalty (raised to -10 to compensate). The previous "collision
            # ends the episode" behaviour effectively rewarded the agent for
            # crashing to escape the Stage-1 negative dense baseline.
            if in_collision and bool(getattr(self.cfg, "collision_terminates", True)):
                return True, False, {**info, "termination": "collision"}
            if out_of_workspace:
                return True, False, {**info, "termination": "workspace"}
            if contact_damage:
                return True, False, {**info, "termination": "contact_force"}
        if self._step_count >= self.cfg.max_steps:
            return False, True, {**info, "termination": "max_steps"}
        return False, False, info

    def _out_of_workspace(self) -> bool:
        """Workspace check with floor-aware lower bound.

        Lower clamp = ``floor_z - 0.05`` (5 cm below floor) so the EE/tire/
        pickup pose at z = floor + R does not falsely trip the gate, while
        a body that has fallen well through the floor still does.
        """
        ws = self.cfg.obs.workspace_radius
        z_lo = float(self.cfg.floor_z) - 0.05
        for getter in (self.robot_A.ee_pose, self.robot_B.ee_pose,
                       self.scene.tire_pose):
            pos, _ = getter()
            if np.linalg.norm(pos[:2]) > ws * 1.5 or pos[2] > 2.5 or pos[2] < z_lo:
                return True
        return False

    # ------------------------------------------------------------------
    # Curriculum hook (env consumers call this between rollouts)
    # ------------------------------------------------------------------
    def set_phase(self, phase: int) -> None:
        self.cfg.curriculum.phase = int(phase)
