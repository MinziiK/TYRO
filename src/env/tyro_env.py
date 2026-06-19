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

from typing import Any, Dict, List, Optional, Tuple
import contextlib
import math
import os

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
    angle_between,
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
        # Eval hook: when set, ``reset()`` uses this hub XY offset instead of
        # sampling from ``RANDOM_POSITION_RANGE`` (see ``set_dr_hub_xy_offset``).
        self._dr_hub_xy_override: Optional[np.ndarray] = None

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
        # Nut-task two-stage APPROACH shaping book-keeping: separate
        # potentials for "get onto the bolt axis" (lateral) and "slide to
        # the staging point along the axis" (axial). Reset on every target
        # switch / macro hand-off so the first post-transition sample does
        # not pay a spurious potential jump.
        self._prev_lateral_B: Optional[float] = None
        self._prev_axial_err_B: Optional[float] = None
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
        # ``carry_tire_rigid_sync`` visual lock: True while the tire mass has
        # been zeroed for the rigid carry (so physics never drifts it between
        # the per-sub-step snaps). Restored to ``tire_mass`` on carry exit.
        self._carry_mass_zeroed: bool = False
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
        # Stage-1 carry: first traj index whose nominal EE has risen at
        # least ``planner_carry_lift_skip_min_dz`` above the grasp Z.
        # Waypoints before this are yaw-alignment near the tire; skipped
        # on replan when ``planner_skip_s1_yaw_preamble`` is enabled.
        self._carry_lift_from_idx: int = 0
        # EE Z at grasp time; Stage-1 keeps baked replay until the arm
        # clears this height + ``planner_carry_lift_skip_min_dz``.
        self._s1_grasp_ee_z: Optional[float] = None
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
        # GUI mode defaults to real-time stepping, which advances physics
        # during wall-clock sleeps (e.g. eval viewer pacing) and desynchronises
        # the policy loop. Force manual stepping only.
        if mode == p.GUI:
            p.setRealTimeSimulation(0, physicsClientId=self.client)
            # Hide the default side panels (left parameter tree + right
            # RGB/depth/segmentation preview strips) for a clean demo view.
            # Keep only the 3D viewport. No effect on physics or headless.
            try:
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
                    physicsClientId=self.client)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0,
                    physicsClientId=self.client)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
                    physicsClientId=self.client)
            except p.error:
                pass

    def _draw_world_axes(self, length: float = 1.0) -> None:
        """Draw the world-origin RGB XYZ axis triad (GUI only).

        Disabling ``COV_ENABLE_GUI`` to hide the side panels also hides
        PyBullet's built-in axis overlay, so re-draw an explicit triad with
        debug lines: X=red, Y=green, Z=blue. ``resetSimulation`` (run every
        ``reset``) clears debug items, so this is re-issued from ``reset``.
        No-op in DIRECT / headless.
        """
        if not (bool(getattr(self.cfg, "render", False)) and self.client >= 0):
            return
        # Skip while a render-freeze is active (e.g. the E2E handoff reset):
        # issuing debug-draw calls into a frozen visualiser is unnecessary and
        # has been observed to destabilise long software-rendered sessions.
        if getattr(self, "_render_freeze_depth", 0) > 0:
            return
        try:
            ids = getattr(self, "_axis_dbg_ids", [None, None, None])
            specs = (
                ([0, 0, 0], [length, 0, 0], [1, 0, 0]),
                ([0, 0, 0], [0, length, 0], [0, 1, 0]),
                ([0, 0, 0], [0, 0, length], [0, 0, 1]),
            )
            new_ids = []
            for i, (a, b, c) in enumerate(specs):
                kw = dict(lineWidth=2.0, physicsClientId=self.client)
                if ids[i] is not None:
                    new_ids.append(p.addUserDebugLine(
                        a, b, c, replaceItemUniqueId=ids[i], **kw))
                else:
                    new_ids.append(p.addUserDebugLine(a, b, c, **kw))
            self._axis_dbg_ids = new_ids
        except p.error:
            pass

    def close(self) -> None:
        if self.client >= 0:
            try:
                if p.isConnected(self.client):
                    p.disconnect(physicsClientId=self.client)
            except p.error:
                pass
            self.client = -1

    # ------------------------------------------------------------------
    # GUI render freeze (nestable). Internal IK / collision probing teleports
    # Robot B all over via resetJointState and restores it; in GUI mode those
    # intermediate poses get drawn as a "flicker / pose snaps away and back".
    # Suppress the visualiser for the duration of any such probe so only the
    # final committed pose is ever shown. Reference-counted so nested probes
    # don't prematurely re-enable rendering. No effect in DIRECT / headless.
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _render_frozen(self):
        gui = bool(getattr(self.cfg, "render", False)) and self.client >= 0
        depth = getattr(self, "_render_freeze_depth", 0)
        if gui and depth == 0:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                                           physicsClientId=self.client)
            except p.error:
                gui = False
        self._render_freeze_depth = depth + 1
        try:
            yield
        finally:
            self._render_freeze_depth -= 1
            if gui and self._render_freeze_depth == 0:
                try:
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                                               physicsClientId=self.client)
                except p.error:
                    pass

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
        self._axis_dbg_ids = [None, None, None]
        # Invalidate the obs/action mask caches so they track the active cfg
        # layout. This matters when a single env is reconfigured between phases
        # (e.g. the E2E viewer switches mount→nut, changing obs/action dims);
        # the masks are cheap to rebuild, so always refreshing is safe.
        self._obs_mask_cache = None
        self._action_mask_cache = None
        # resetSimulation() invalidates all body / constraint ids — clear cache.
        self._grasp_constraint = None
        self._grasp_kinematic = False
        self._carry_mass_zeroed = False
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
        self._carry_lift_from_idx = 0
        self._s1_grasp_ee_z = None
        self._mount_hold_left = 0
        self._mount_frozen_q = None
        self._mount_hold_finish_term = False
        # 2026-06-06 — mount-seat glide state. While ``_mount_seat_active``
        # the tire is kinematically interpolated from ``_mount_seat_t0_*``
        # (its grasped pose at gate fire) to ``_mount_seat_tgt_*`` (the exact
        # seated hub pose) over ``_mount_seat_total`` env steps; the
        # ``mounted`` event + bond are deferred until ``_mount_seat_left``
        # reaches 0. Removes the instant snap teleport at the mount instant.
        self._mount_seat_active = False
        self._mount_seat_left = 0
        self._mount_seat_total = 0
        self._mount_seat_t0_pos = None
        self._mount_seat_t0_orn = None
        self._mount_seat_tgt_pos = None
        self._mount_seat_tgt_orn = None
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
        # Robot B sequential nut-fastening task bookkeeping (only consumed
        # when ``cfg.nut_fastening_task``). ``_nut_target_idx`` is the bolt
        # currently being fastened; ``_nut_fastened`` is the ordered list of
        # completed bolt indices; ``_nut_hold_count`` counts consecutive
        # in-gate steps; ``_nut_done`` latches once every bolt is fastened.
        # ``_nut_frozen_qA`` holds Robot A's parked joint vector so it can be
        # re-driven each step (the arm is a static fixture in this task).
        self._nut_target_idx = 0
        self._nut_fastened: List[int] = []
        self._nut_premark = 0
        # Index into ``_nut_order()`` of the current target bolt.
        self._nut_seq_pos = 0
        self._nut_hold_count = 0
        self._nut_done = False
        self._nut_frozen_qA = None
        # Coaxial tool orientation B holds during APPROACH (captured from the
        # reachable hot-start pose) when ``nut_b_lock_coaxial`` is set. None
        # ⇒ fall back to the analytic minimal-rotation coaxial quat.
        self._nut_lock_quat = None
        # v14 — nominal APPROACH trajectory (joint-space lerp; coaxial quat locked).
        self._nut_traj_pos: Optional[np.ndarray] = None
        self._nut_traj_q: Optional[np.ndarray] = None
        self._nut_traj_step = 0
        # Per-bolt sub-FSM: 0 = APPROACH (policy drives B toward the bolt's
        # staging point just outside the stud tip), 1 = MACRO (the env
        # scripts a deterministic insert→hold→retract straight down/up the
        # bolt axis; the policy is ignored). ``_nut_macro_stage`` indexes the
        # macro leg (0 INSERT, 1 HOLD, 2 RETRACT); ``_nut_macro_step`` is the
        # per-leg watchdog counter; ``_nut_arrive_count`` counts consecutive
        # in-gate steps at the staging point before the macro triggers.
        self._nut_subphase = 0
        self._nut_macro_stage = 0
        self._nut_macro_step = 0
        self._nut_arrive_count = 0
        self._nut_macro_quat: Optional[np.ndarray] = None
        # Joint-space lerp for the current macro leg (IK once per leg, then
        # interpolate so the forced slide can't flip IK branches mid-plunge).
        self._nut_macro_q_from: Optional[np.ndarray] = None
        self._nut_macro_q_to: Optional[np.ndarray] = None
        self._nut_macro_leg_len = 1
        # Per-bolt cached macro endpoints (insert=base, retract=clear), solved
        # once per reset so the forced slide always reaches a known-good pose.
        self._nut_base_q: List[Optional[np.ndarray]] = []
        self._nut_retract_q: List[Optional[np.ndarray]] = []
        self._prev_axial_B: Optional[float] = None
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
        self._maybe_disable_tire_hub_collision()
        # resetSimulation() above cleared any debug items; redraw the world
        # XYZ axis triad (GUI only, no-op headless).
        self._draw_world_axes()

        # **2026-06-09 (DR mount-target sync)** — the Stage-1 mount planner
        # bakes its nominal end-pose from ``cfg.tire_mount_pos`` (a static
        # config value). Under hub DR the scene hub translates by
        # ``_dr_hub_xy_offset`` but this target did not move, so the baked
        # nominal pointed at the *nominal* hub and the whole offset had to be
        # absorbed by the RL residual alone (un-correctable past the residual
        # cap, and the seated-tire bond target drifted off the real hub).
        # Re-derive the seat target each reset = base target + the live hub XY
        # offset, so A's nominal trajectory tracks the offset hub (mirrors how
        # Robot B reads live bolt coords). Z is preserved. When DR is off the
        # offset is zero ⇒ bit-identical to the pre-patch path.
        if not hasattr(self, "_tire_mount_pos_base"):
            self._tire_mount_pos_base = tuple(
                float(x) for x in self.cfg.tire_mount_pos
            )
        dr_xy = getattr(self, "_dr_hub_xy_offset", np.zeros(2))
        self.cfg.tire_mount_pos = (
            self._tire_mount_pos_base[0] + float(dr_xy[0]),
            self._tire_mount_pos_base[1] + float(dr_xy[1]),
            self._tire_mount_pos_base[2],
        )

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
        # **2026-06-07 — Robot B nut-fastening task** short-circuits the
        # entire Robot-A start-pose / world-pin block. The tire is bonded
        # mounted on the hub and Robot A is parked as a static fixture, so
        # none of the cradle-pin / easy-spawn / hot-start logic applies.
        nut_task = bool(getattr(self.cfg, "nut_fastening_task", False))
        if nut_task:
            self._apply_nut_fastening_setup()
            self._start_pos_used_easy_last_reset = False
            self._start_pos_used_hot_start_last_reset = False
        else:
            self._reset_robot_a_start_pose()

        self._step_count = 0
        self._prev_action = np.zeros(self.cfg.action.dim, dtype=np.float32)
        self._prev_d_A = None
        self._prev_d_B = None
        self._prev_axial_B = None
        self._prev_lateral_B = None
        self._prev_axial_err_B = None
        self._prev_d_approach = None
        self._prev_d_return = None
        self._prev_d_hub: Optional[float] = None
        self._mount_done_step: Optional[int] = None
        self._demount_bonus_paid = False

        self._replan_for_current_stage()

        obs = self._compute_obs()
        info = {"target_bolt_idx": self.handles.target_bolt_idx}
        return obs, info

    def _reset_robot_a_start_pose(self) -> None:
        """Robot-A start-pose curriculum + tire world-pin (non-nut tasks).

        Extracted verbatim from ``reset`` so the nut-fastening task can
        bypass the entire cradle-pin / easy-spawn / reverse-curriculum
        block. Mutates the same ``self`` state the inline version did.
        """
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

    # ------------------------------------------------------------------
    # Robot B sequential nut-fastening task
    # ------------------------------------------------------------------
    #: Bolt visual states (RGBA).
    _NUT_COLOR_PENDING = (0.55, 0.55, 0.60, 1.0)
    _NUT_COLOR_TARGET = (0.95, 0.85, 0.10, 1.0)
    _NUT_COLOR_RETRACT = (0.95, 0.55, 0.10, 1.0)
    _NUT_COLOR_FASTENED = (0.10, 0.80, 0.25, 1.0)

    def _set_bolt_color(self, idx: int, rgba: Tuple[float, float, float, float]) -> None:
        try:
            bref = self.handles.bolts[idx]
            p.changeVisualShape(
                bref.uid, bref.link_index, rgbaColor=rgba,
                physicsClientId=self.client,
            )
        except (IndexError, p.error):
            pass

    def _sample_mount_hold_qA(self) -> Optional[np.ndarray]:
        """Sample a recorded Robot-A mount-completion joint vector.

        Loads (and caches) the ``.npz`` snapshot at ``cfg.nut_mount_endpose
        _path`` written by ``scripts/extract_mount_endpose.py``. Returns a
        randomly chosen ``qA`` row, or ``None`` if the file is unset /
        missing / malformed (caller falls back to analytic IK).
        """
        if not hasattr(self, "_mount_hold_qA_bank"):
            self._mount_hold_qA_bank = None
            path = str(getattr(self.cfg, "nut_mount_endpose_path", "") or "")
            if path and os.path.exists(path):
                try:
                    data = np.load(path)
                    q = np.asarray(data["qA"], dtype=np.float64)
                    if q.ndim == 1:
                        q = q[None, :]
                    n_dof = int(self.robot_A.arm.n)
                    if q.ndim == 2 and q.shape[1] == n_dof and q.shape[0] > 0:
                        self._mount_hold_qA_bank = q
                        print(f"[nut] loaded {q.shape[0]} mount-hold A poses "
                              f"from {path}")
                    else:
                        print(f"[nut] mount-endpose shape {q.shape} != "
                              f"(*, {n_dof}); using analytic A hold pose")
                except Exception as exc:  # noqa: BLE001
                    print(f"[nut] failed to load {path}: {exc}; analytic fallback")
        bank = self._mount_hold_qA_bank
        if bank is None or len(bank) == 0:
            return None
        i = int(self._np_random.integers(0, len(bank)))
        return bank[i].copy()

    def _ik_mount_hold_qA(
        self,
        ee_target: np.ndarray,
        warm_start_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """IK Robot A to the seated-tire 6-o'clock support grasp."""
        ur = self.robot_A
        if warm_start_q is not None:
            warm_start = np.asarray(warm_start_q, dtype=np.float64)
        else:
            warm_start, _ = ur.joint_state()
        ik = p.calculateInverseKinematics(
            ur.uid, ur.EE_LINK_INDEX,
            ee_target.tolist(), list(ur.FINAL_LOCK_QUATERNION),
            lowerLimits=ur.arm.lower.tolist(),
            upperLimits=ur.arm.upper.tolist(),
            jointRanges=ur.arm.range.tolist(),
            restPoses=warm_start.tolist(),
            maxNumIterations=300, residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        if ur._ik_arm_slots and len(ik) > max(ur._ik_arm_slots):
            return ik[ur._ik_arm_slots]
        return np.asarray(ur.HOME_POSE, dtype=np.float64)

    def _apply_nut_fastening_setup(self) -> None:
        """Set up the Robot-B nut-fastening start state.

        Models the W1 "tighten" window of the real duty cycle: Robot A has
        just mounted the tire and **keeps holding it seated on the hub**
        while Robot B fastens the bolts.

        * Bonds the tire seated on the hub (fixed constraint) + per-step
          clamp (in ``step``) so it stays put.
        * Positions Robot A's gripper on the seated tire's 6-o'clock tread
          point (the mount-hold grasp) and freezes it there as a static
          support fixture — re-driven to the cached joint vector each step
          so it does not sag and visibly keeps supporting the wheel.
        * Initialises the sequential bolt target to index 0 and recolours
          the bolts (target = yellow, the rest = pending grey).
        """
        # Branch-continuity IK for B: the socket sweeps the whole lug ring, so
        # the HOME-biased IK warm-start fights the policy / snaps branches on
        # the far arc (bolts 4–6). Use current-joint warm-start instead.
        self.robot_B._ik_warmstart_current = True
        # Raise B's motor torque caps for the nut task. The far-arc bolts
        # (4–6) need a near-full-extension reach where the default 300 N·m
        # elbow cap is below the static gravity moment — the arm sags ~36 cm
        # and physically cannot hold the staging pose, so the policy can never
        # dwell to trigger the macro (root cause of the mid-ring stall). B
        # carries no payload here (geometric fastening), so higher caps are
        # safe and realistic. Confirmed: sag 36.8→4.2 cm at bolt 4.
        self.robot_B._motor_forces_override = list(
            getattr(self.cfg, "nut_b_motor_forces",
                    (6000.0, 6000.0, 4000.0, 1000.0, 1000.0, 1000.0))
        )

        R = float(self.cfg.tire_outer_radius)
        # 1. Seat + bond the tire on the hub (sets _mount_seated_pos/orn).
        self._attach_tire_to_hub()
        tire_pos = np.asarray(self._mount_seated_pos, dtype=np.float64)
        tire_orn = np.asarray(self._mount_seated_orn, dtype=np.float64)
        tire_axis = quat_axis(tire_orn, "z")
        tire_axis = tire_axis / max(float(np.linalg.norm(tire_axis)), 1e-9)

        # 2. 6-o'clock grasp anchor of the seated tire: world −Z projected
        #    onto the bore plane (orthogonal to the bore axis), scaled by R
        #    to reach the tread. Mirrors the reverse-curriculum hot-start.
        gravity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        proj = gravity_dir - float(np.dot(gravity_dir, tire_axis)) * tire_axis
        proj_norm = float(np.linalg.norm(proj))
        if proj_norm < 1e-6:
            tread_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            tread_dir = tread_dir - float(np.dot(tread_dir, tire_axis)) * tire_axis
            tread_dir = tread_dir / max(float(np.linalg.norm(tread_dir)), 1e-9)
        else:
            tread_dir = proj / proj_norm
        ee_target = tire_pos + R * tread_dir

        # 3. Resolve Robot A's mount-hold joint vector.
        #    * Nominal hub: sample from the recorded mount-completion bank
        #      (deployment-distribution match).
        #    * Hub DR active: IK to the *current* seated-tire 6-o'clock
        #      anchor so A tracks the offset hub/tire; the bank row (if any)
        #      seeds the IK rest pose only.
        #    Per-joint jitter is added in both cases so B sees a spread of
        #    support poses rather than a single memorised configuration.
        ur = self.robot_A
        hub_off = float(np.linalg.norm(
            getattr(self, "_dr_hub_xy_offset", np.zeros(2)),
        ))
        bank_q = self._sample_mount_hold_qA()
        if hub_off > 1e-5:
            arm_q = self._ik_mount_hold_qA(ee_target, warm_start_q=bank_q)
        elif bank_q is not None:
            arm_q = bank_q
        else:
            arm_q = self._ik_mount_hold_qA(ee_target)
        # Per-joint robustness jitter (uniform, symmetric).
        jit = float(getattr(self.cfg, "nut_a_hold_jitter_rad", 0.0))
        if jit > 0.0:
            arm_q = np.asarray(arm_q, dtype=np.float64) + self._np_random.uniform(
                -jit, jit, size=np.asarray(arm_q).shape
            )
        arm_q = np.clip(arm_q, ur.arm.lower, ur.arm.upper)
        for idx, q in zip(ur.arm.indices, arm_q):
            p.resetJointState(
                ur.uid, idx, targetValue=float(q), targetVelocity=0.0,
                physicsClientId=self.client,
            )
        ur.last_target_pos = ur.ee_pose()[0].copy()
        self._nut_frozen_qA = np.asarray(arm_q, dtype=np.float64).copy()

        # 4. Sequential bolt bookkeeping.
        #
        # Per-bolt random start (curriculum coverage fix): always starting the
        # chain at bolt 0 means later bolt-to-bolt transitions (4→5, 5→6, …)
        # are only ever sampled in episodes that already cleared every earlier
        # bolt — exponentially rare early in training. The learned competence
        # then stalls at an advancing frontier (observed: v2/v3 stuck at bolt
        # 3, v4 at bolt 4), with the socket parked ~10 cm off the next bolt's
        # axis (alignment fine, lateral-closing untrained). Seeding the start
        # bolt uniformly over the ring gives every transition equal training
        # mass, so the frontier disappears. Earlier bolts are marked already
        # fastened; the hot-start teleports B to the chosen bolt's approach.
        n_bolts = len(self.handles.bolts)
        order = self._nut_order()
        # Sequence position to begin the episode at. Normally 0 (start at the
        # first bolt in the order); the reverse-curriculum random-bolt option
        # may seed a later position so every transition gets training mass.
        start_pos = 0
        if (
            bool(getattr(self.cfg, "nut_b_hotstart_random_bolt", False))
            and bool(getattr(self.cfg, "nut_b_hotstart_enable", False))
            and float(getattr(self.cfg, "nut_b_hotstart_alpha", 0.0)) > 1e-3
            and n_bolts > 0
        ):
            start_pos = int(self._np_random.integers(0, len(order)))
        self._nut_seq_pos = start_pos
        self._nut_target_idx = order[start_pos]
        # Bolts earlier in the order are pre-marked fastened.
        self._nut_fastened = list(order[:start_pos])
        # Count of bolts the episode STARTS with already fastened (random-bolt
        # curriculum). ``n_fastened`` includes these, so subtract this to get
        # the true policy-fastened count (the honest progress metric — the raw
        # n_fastened was inflated by ~the mean random start position).
        self._nut_premark = len(self._nut_fastened)
        self._nut_hold_count = 0
        self._nut_subphase = 0
        self._nut_macro_stage = 0
        self._nut_macro_step = 0
        self._nut_arrive_count = 0
        self._nut_macro_quat = None
        self._nut_macro_q_from = None
        self._nut_macro_q_to = None
        self._nut_macro_leg_len = 1
        self._prev_axial_B = None
        self._nut_done = False
        # v21 — branch-aware INSERT: last macro step a seat branch search ran.
        self._nut_last_reseat_step = -(10 ** 9)
        self._nut_lock_quat = None
        self._nut_traj_pos = None
        self._nut_traj_q = None
        self._nut_traj_step = 0
        # v19 — per-episode reset of the stall-truncation / path-waste state.
        self._nut_stall_key = None
        self._nut_stall_count = 0
        self._nut_prog_best = None
        self._nut_prev_ee = None
        self.task_stage = 0

        n = len(self.handles.bolts)
        for i in range(n):
            self._set_bolt_color(
                i,
                self._NUT_COLOR_FASTENED if i in self._nut_fastened
                else self._NUT_COLOR_PENDING,
            )
        if n > 0:
            self._set_bolt_color(self._nut_target_idx, self._NUT_COLOR_TARGET)
            self.handles.target_bolt_idx = self._nut_target_idx

        # 4a. Filter Robot-B socket ↔ hub/tire collision. The fastening is
        #     geometric (no nut bodies / torque), so the forced insert must be
        #     able to slide the socket coaxially to the hub-face base without
        #     a hard contact shoving it off-axis. Mirrors the tire↔hub mount
        #     filter — the gate's measured seating still proves coverage.
        self._filter_nut_socket_collisions()

        # 4b. Cache the macro INSERT (hub-face base) and RETRACT joint
        #     solutions per bolt. These are pose-independent (the hub is
        #     fixed), so solving them once with a thorough roll-free search
        #     here — rather than per-leg with luck-of-the-seed IK — makes the
        #     forced insert reliably reach the base coaxially every time.
        if bool(getattr(self.cfg, "nut_scripted_macro", True)):
            self._precompute_nut_macro_solutions(n)

        # 5. Robot-B start pose: planner nominal trajectory (v14) or
        #    reverse-curriculum hot-start (legacy).
        if bool(getattr(self.cfg, "nut_b_planner_residual", False)):
            self._generate_nut_approach_traj()
        else:
            self._apply_nut_b_hotstart()

    def _quat_align_tool_z(self, want_z: np.ndarray,
                           seed_quat: Optional[np.ndarray] = None) -> np.ndarray:
        """Quaternion (x,y,z,w) whose body +Z axis maps to ``want_z``.

        Roll about ``want_z`` is unconstrained (a nut-runner spins freely);
        we pick the minimal rotation from world +Z so the wrist stays in a
        natural pose. ``seed_quat`` is unused (kept for signature symmetry).
        """
        want_z = np.asarray(want_z, dtype=np.float64)
        want_z = want_z / max(float(np.linalg.norm(want_z)), 1e-9)
        z0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        c = float(np.dot(z0, want_z))
        if c > 1.0 - 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if c < -1.0 + 1e-9:
            # 180° about any axis ⊥ z0.
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = np.cross(z0, want_z)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        ang = math.acos(max(-1.0, min(1.0, c)))
        return axisangle3_to_quat(axis * ang)

    def _nut_hotstart_branch_seed(self, idx: int, pos: np.ndarray,
                                  want_z: np.ndarray) -> Optional[np.ndarray]:
        """Per-bolt arm-joint seed that reaches the staging pose coaxially.

        PyBullet's DLS IK cannot hop into the joint branch required for the
        bottom-arc bolts (4–7) from a HOME/random seed — it saturates 60–90 cm
        short, which previously mis-classified those bolts as 'unreachable'
        and silently degraded the hot-start to HOME. Direct joint-space
        optimisation (scipy least_squares over the 6 arm joints, multi-start)
        reaches EVERY bolt with <1 mm error, so we solve it once per bolt here
        and cache the result. The cached branch then seeds the regular
        iterative IK at every reset (hub DR only shifts the target a few cm,
        which the iterative refine absorbs).
        """
        cache = getattr(self, "_nut_hotstart_seed_q", None)
        if cache is None:
            cache = {}
            # v19 — disk-backed seed sharing: the multi-start solve below costs
            # seconds per bolt, and EVERY worker (88 in training) used to redo
            # it on first encounter. Load the npz once; solves merge back in.
            path = str(getattr(self.cfg, "nut_hotstart_seed_cache", "") or "")
            if path:
                try:
                    import os
                    if os.path.exists(path):
                        data = np.load(path)
                        for k in data.files:
                            cache[int(k)] = np.asarray(
                                data[k], dtype=np.float64)
                except Exception:
                    pass
            self._nut_hotstart_seed_q = cache
        if idx in cache:
            return cache[idx]
        try:
            from scipy.optimize import least_squares
        except ImportError:
            cache[idx] = None
            return None
        rb = self.robot_B
        lo, hi = rb.arm.lower, rb.arm.upper
        q_save, _ = rb.joint_state()
        q_save = np.asarray(q_save, dtype=np.float64)
        pos = np.asarray(pos, dtype=np.float64)
        want_z = np.asarray(want_z, dtype=np.float64)
        want_z = want_z / max(float(np.linalg.norm(want_z)), 1e-9)

        def _fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            for s, qq in zip(rb.arm.indices, q):
                p.resetJointState(rb.uid, int(s), float(qq),
                                  targetVelocity=0.0,
                                  physicsClientId=self.client)
            ee, eq = rb.ee_pose()
            return np.asarray(ee, dtype=np.float64), np.asarray(
                quat_axis(eq, "z"), dtype=np.float64)

        def _resid(q: np.ndarray) -> np.ndarray:
            ee, gz = _fk(q)
            # Sign-free coaxiality: the socket may bore from either direction.
            mis = min(float(np.linalg.norm(gz - want_z)),
                      float(np.linalg.norm(gz + want_z)))
            return np.concatenate([(ee - pos) * 10.0, [mis * 2.0]])

        best_q, best_pe = None, 1e9
        for si in range(40):
            q0 = q_save if si == 0 else self._np_random.uniform(lo, hi)
            try:
                r = least_squares(_resid, q0, bounds=(lo, hi), xtol=1e-10,
                                  ftol=1e-10, max_nfev=300, diff_step=1e-4)
            except Exception:
                continue
            ee, gz = _fk(r.x)
            pe = float(np.linalg.norm(ee - pos))
            ang = min(float(np.linalg.norm(gz - want_z)),
                      float(np.linalg.norm(gz + want_z)))
            if pe < best_pe:
                best_pe, best_q = pe, np.asarray(r.x, dtype=np.float64).copy()
            if pe < 0.003 and ang < 0.05:
                break
        # Restore the live config (caller owns the actual teleport).
        for s, qq in zip(rb.arm.indices, q_save):
            p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=self.client)
        cache[idx] = best_q if best_pe < 0.02 else None
        # Persist (atomic rename; concurrent writers race benignly — the
        # winning file is always a valid superset solved by SOME worker).
        path = str(getattr(self.cfg, "nut_hotstart_seed_cache", "") or "")
        if path and cache[idx] is not None:
            try:
                import os
                import tempfile
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                payload = {str(k): v for k, v in cache.items()
                           if v is not None}
                fd, tmp = tempfile.mkstemp(
                    dir=os.path.dirname(path) or ".", suffix=".npz")
                os.close(fd)
                np.savez(tmp, **payload)
                # np.savez appends .npz when missing; mkstemp already has it.
                os.replace(tmp, path)
            except Exception:
                pass
        return cache[idx]

    def _apply_nut_b_hotstart(self) -> None:
        """Reverse-curriculum start pose for Robot B (nut task).

        ``alpha`` interpolates B's tool_tip start between its HOME pose
        (``alpha = 0``) and the **target bolt's approach point**, just outside
        the stud tip (``alpha = 1``), with tool +Z aligned to the bolt axis so
        the socket is oriented to insert. The target bolt at reset is the first
        entry of ``nut_bolt_order`` (bolt 0), so the curriculum's easy start is
        directly over bolt 0. IK-teleports B and re-seeds its delta accumulator
        so the first policy step continues from there.
        """
        if not bool(getattr(self.cfg, "nut_b_hotstart_enable", False)):
            return
        alpha = float(getattr(self.cfg, "nut_b_hotstart_alpha", 0.0))
        alpha = max(0.0, min(1.0, alpha))
        if alpha <= 1e-3:
            return  # full HOME start — nothing to do.

        rb = self.robot_B
        idx = int(self._nut_target_idx)
        bolt_pos = np.asarray(self.scene.bolt_pose(idx)[0], dtype=np.float64)
        a = np.asarray(self.scene.bolt_axis(idx), dtype=np.float64)
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        if bool(getattr(self.cfg, "nut_b_hotstart_hub_center", False)):
            # 2026-06-08 (v10) — start at the bolt-ring CENTER, backed off to
            # the staging depth (NOT the hub face, which is unreachable with the
            # +Y tool orientation). This point sits at a FIXED Y = staging depth
            # and is equidistant (~0.21 m) to every bolt, so the approach to any
            # bolt is a symmetric pure-XZ radial reach in the constant-Y plane —
            # no bolt-0 bias, and the same skill transfers to all bolts.
            n_b = len(self.handles.bolts)
            ring_center = np.mean(
                [np.asarray(self.scene.bolt_pose(i)[0], dtype=np.float64)
                 for i in range(n_b)], axis=0,
            )
            approach = ring_center + a * self._nut_staging_axial()
        else:
            # Legacy: on-axis at the staging depth just outside the TARGET
            # bolt's stud tip (directly over bolt 0 with the custom order).
            approach = bolt_pos + a * self._nut_staging_axial()
        home_ee = np.asarray(rb.ee_pose()[0], dtype=np.float64)
        start_pos = (1.0 - alpha) * home_ee + alpha * approach
        want_z = -a  # tool +Z points into the bolt (insert direction)
        base_quat = self._quat_align_tool_z(want_z)

        # A single IK jump from HOME to the target frequently sticks in a folded
        # local minimum (the target sits in the far/orientation-constrained part
        # of the workspace), and the minimal-rotation roll is not always the
        # reachable branch — the hub-CENTER start in particular needs a specific
        # roll the single-quat solve misses (lands ~1 m off). So search a few
        # rolls about ``want_z`` (a nut-runner spins freely, so roll is free),
        # refine each iteratively from the HOME warm-start, and keep the
        # solution with the smallest realised reach error.
        warm, _ = rb.joint_state()
        warm = np.asarray(warm, dtype=np.float64)
        n_roll = 12
        best_ik = warm.copy()
        best_err = 1e9
        prefer_up = bool(getattr(self.cfg, "nut_b_hotstart_elbow_up", False))
        up_gate = float(getattr(self.cfg, "nut_b_hotstart_reach_gate", 0.04))
        best_score = -1e9
        best_score_ik = None
        if not hasattr(self, "_nut_hotstart_ik_cache"):
            self._nut_hotstart_ik_cache: Dict[int, Tuple[float, np.ndarray]] = {}
        ik_cache = self._nut_hotstart_ik_cache

        def _arm_height_score() -> float:
            zs = []
            for s in rb.arm.indices:
                ls = p.getLinkState(rb.uid, int(s),
                                    physicsClientId=self.client)
                zs.append(float(ls[0][2]))
            return float(np.mean(zs)) if zs else 0.0

        def _refine_from_seed(
                seed: np.ndarray, target_quat,
                ) -> Tuple[np.ndarray, float]:
            """One hot-start IK refine (12 PyBullet iters) from ``seed``."""
            ik = np.asarray(seed, dtype=np.float64).copy()
            seed_arm = np.clip(
                ik[: len(rb.arm.indices)]
                if len(ik) >= len(rb.arm.indices) else ik,
                rb.arm.lower, rb.arm.upper,
            )
            for s, q in zip(rb.arm.indices, seed_arm):
                p.resetJointState(rb.uid, int(s), float(q),
                                  targetVelocity=0.0,
                                  physicsClientId=self.client)
            for _ in range(12):
                sol = p.calculateInverseKinematics(
                    rb.uid, rb.EE_LINK_INDEX,
                    start_pos.tolist(), list(target_quat),
                    lowerLimits=rb.arm.lower.tolist(),
                    upperLimits=rb.arm.upper.tolist(),
                    jointRanges=rb.arm.range.tolist(),
                    restPoses=np.asarray(ik, dtype=np.float64).tolist(),
                    maxNumIterations=300, residualThreshold=1e-6,
                    physicsClientId=self.client,
                )
                sol = np.asarray(sol, dtype=np.float64)
                if not (rb._ik_arm_slots and len(sol) > max(rb._ik_arm_slots)):
                    break
                arm_sol = np.clip(
                    sol[rb._ik_arm_slots], rb.arm.lower, rb.arm.upper)
                for s, q in zip(rb.arm.indices, arm_sol):
                    p.resetJointState(rb.uid, int(s), float(q),
                                      targetVelocity=0.0,
                                      physicsClientId=self.client)
                ik = sol
                if float(np.linalg.norm(
                        np.asarray(rb.ee_pose()[0]) - start_pos)) < 0.01:
                    break
            err = float(np.linalg.norm(
                np.asarray(rb.ee_pose()[0]) - start_pos))
            return ik, err

        def _record_candidate(ik: np.ndarray, err: float, roll: float) -> None:
            nonlocal best_err, best_ik, best_score, best_score_ik, best_roll
            if err < best_err:
                best_err = err
                best_ik = np.asarray(ik, dtype=np.float64).copy()
                best_roll = float(roll)
            if prefer_up and err < up_gate:
                score = _arm_height_score()
                if score > best_score:
                    best_score = score
                    best_score_ik = np.asarray(ik, dtype=np.float64).copy()

        def _apply_ik_teleport(ik: np.ndarray) -> None:
            if not (rb._ik_arm_slots and len(ik) > max(rb._ik_arm_slots)):
                return
            arm_q = np.clip(ik[rb._ik_arm_slots], rb.arm.lower, rb.arm.upper)
            for s, q in zip(rb.arm.indices, arm_q):
                p.resetJointState(
                    rb.uid, int(s), targetValue=float(q), targetVelocity=0.0,
                    physicsClientId=self.client,
                )
            rb.last_target_pos = rb.ee_pose()[0].copy()
            rb._cmd_q = arm_q.copy()
            rb.drive_arm_targets(arm_q)
            self._nut_lock_quat = np.asarray(
                rb.ee_pose()[1], dtype=np.float64,
            ).copy()

        def _target_quat_for_roll(roll: float):
            roll_q = np.asarray(
                axisangle3_to_quat(want_z * float(roll)), dtype=np.float64)
            _, tq = p.multiplyTransforms(
                [0.0, 0.0, 0.0], roll_q.tolist(),
                [0.0, 0.0, 0.0], base_quat.tolist(),
            )
            return tq

        best_roll = 0.0
        # Fast path — one refine from the cached (roll, arm seed). Skips the
        # scipy branch seed + 12-roll grid that cost ~1.3 s/reset.
        cached = ik_cache.get(idx)
        if cached is not None:
            c_roll, c_seed = cached
            ik, err = _refine_from_seed(
                np.asarray(c_seed), _target_quat_for_roll(c_roll))
            if err < 0.05:
                _apply_ik_teleport(ik)
                return

        # Slow path — full roll × seed search (first visit per bolt / cache miss).
        seed_pool = [warm]
        if bool(getattr(self.cfg, "nut_pure_rl", False)):
            branch_seed = self._nut_hotstart_branch_seed(idx, approach, want_z)
            if branch_seed is not None:
                seed_pool.insert(0, np.asarray(branch_seed, dtype=np.float64))
            for _ in range(4):
                seed_pool.append(
                    self._np_random.uniform(rb.arm.lower, rb.arm.upper)
                )
        for ri in range(n_roll):
            roll = 2.0 * math.pi * ri / n_roll
            target_quat = _target_quat_for_roll(roll)
            for seed in seed_pool:
                ik, err = _refine_from_seed(seed, target_quat)
                _record_candidate(ik, err, roll)
                for s, q in zip(rb.arm.indices, warm):
                    p.resetJointState(rb.uid, int(s), float(q),
                                      targetVelocity=0.0,
                                      physicsClientId=self.client)
            if not prefer_up and best_err < 0.01:
                break
        if prefer_up and best_score_ik is not None:
            best_ik = best_score_ik
        for s, q in zip(rb.arm.indices, warm):
            p.resetJointState(rb.uid, int(s), float(q), targetVelocity=0.0,
                              physicsClientId=self.client)
        if (
            bool(getattr(self.cfg, "nut_pure_rl", False))
            and best_err > 0.05
        ):
            return
        _apply_ik_teleport(best_ik)
        if best_err < 0.05 and rb._ik_arm_slots:
            arm_q = np.clip(
                best_ik[rb._ik_arm_slots], rb.arm.lower, rb.arm.upper)
            ik_cache[idx] = (best_roll, arm_q.copy())

    def _nut_gate_metrics(self, idx: int) -> Tuple[float, float]:
        """(d_B, theta_B) for the nut-runner tool_tip vs bolt ``idx``.

        ``theta_B`` is folded into [0, π/2] so the tool may seat over the
        stud from either bore direction.
        """
        eeB_pos, eeB_orn = self.robot_B.ee_pose()
        bolt_pos, _ = self.scene.bolt_pose(idx)
        bolt_axis = self.scene.bolt_axis(idx)
        eeB_z = quat_axis(eeB_orn, "z")
        d_B = float(np.linalg.norm(
            np.asarray(eeB_pos, dtype=np.float64)
            - np.asarray(bolt_pos, dtype=np.float64)
        ))
        theta = float(angle_between(eeB_z, bolt_axis))
        theta_B = min(theta, math.pi - theta)
        return d_B, theta_B

    def _nut_axial_lateral(self, idx: int) -> Tuple[float, float, float]:
        """(axial, lateral, theta) of the nut-runner tool_tip vs bolt ``idx``.

        With ``a`` = the (unit) bolt axis (points from the hub face toward
        the free stud tip) and ``v = tool_tip − bolt_center``:

        * ``axial   = v·a`` — signed depth along the stud. The hub-face
          **base** is at ``−L/2``, the free **tip** at ``+L/2``; values
          ``> L/2`` mean the socket has cleared the tip entirely.
        * ``lateral = ‖v − axial·a‖`` — perpendicular offset from the stud
          axis (0 ⇒ perfectly coaxial, i.e. entered exactly along the axis).
        * ``theta``  — angle between tool +Z and the stud axis, folded to
          ``[0, π/2]`` (either bore direction is acceptable).
        """
        eeB_pos, eeB_orn = self.robot_B.ee_pose()
        bolt_pos, _ = self.scene.bolt_pose(idx)
        a = np.asarray(self.scene.bolt_axis(idx), dtype=np.float64)
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        v = np.asarray(eeB_pos, dtype=np.float64) - np.asarray(
            bolt_pos, dtype=np.float64
        )
        axial = float(np.dot(v, a))
        lateral = float(np.linalg.norm(v - axial * a))
        eeB_z = quat_axis(eeB_orn, "z")
        theta = float(angle_between(eeB_z, a))
        theta = min(theta, math.pi - theta)
        return axial, lateral, theta

    def _nut_axis_unit(self, idx: int) -> np.ndarray:
        a = np.asarray(self.scene.bolt_axis(idx), dtype=np.float64)
        return a / max(float(np.linalg.norm(a)), 1e-9)

    def _nut_stage_target_axial(self) -> float:
        """Signed axial depth the policy is currently driving the socket to.

        * APPROACH (``_nut_subphase == 0``) → the staging point (just past the
          tip).
        * INSERT / HOLD (subphase 1, macro_stage ≤ 1) → the hub-face **base**
          (``−L/2``), the deepest reachable seat.
        * RETRACT (subphase 1, macro_stage 2) → the cleared point past the tip.

        This is the per-leg goal for the pure-RL axial potential (and the
        ``axial_err`` obs channel) so the policy always knows which way to plunge.
        """
        if int(self._nut_subphase) == 0:
            return float(self._nut_staging_axial())
        if int(self._nut_macro_stage) >= 2:
            return float(self._nut_retract_axial())
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        return -0.5 * L

    def _nut_obs_block(self, eeB_pos: np.ndarray, ws: float) -> np.ndarray:
        """Nut-task observation block for the current target bolt.

        Base 7-d (always):

        * [0:3] ``(staging_point − tool_tip) / ws`` — the direct approach
          target vector (on-axis, just outside the stud tip). This is exactly
          what the APPROACH reach reward optimises, given to the policy
          instead of forcing it to infer the staging point from the
          bolt-centre vector + orientation quaternion.
        * [3:6] bolt axis unit vector (world) — the insertion direction.
        * [6]   ``θ / (π/2)`` — tool +Z ↔ bolt-axis angle (0 aligned, 1 perp).

        Pure-RL extra 5-d (``nut_pure_rl`` only; the policy now drives the whole
        insert→hold→retract, so it needs to sense the in/out state):

        * [7]  ``axial / L``    — signed depth along the stud (base −0.5, tip
          +0.5, staging ≈ +1.3).
        * [8]  ``lateral / ws`` — off-axis (coaxiality) error.
        * [9]  ``subphase``     — 0 APPROACH, 1 INSERT/HOLD/RETRACT.
        * [10] ``stage / 2``    — macro stage 0 insert / 1 hold / 2 retract.
        * [11] ``axial_err / L``— signed (target − axial), i.e. how far / which
          way to plunge to reach the current leg's goal depth.
        """
        idx = int(self._nut_target_idx)
        a = self._nut_axis_unit(idx)
        staging = self._nut_point_on_axis(idx, self._nut_staging_axial())
        vec_to_staging = (staging - np.asarray(eeB_pos, dtype=np.float64)) / ws
        axial, lateral, theta = self._nut_axial_lateral(idx)
        theta_n = float(np.clip(theta / (0.5 * math.pi), 0.0, 1.0))
        base = np.concatenate(
            [vec_to_staging, a, np.array([theta_n], dtype=np.float64)]
        ).astype(np.float64)
        if not bool(getattr(self.cfg, "nut_pure_rl", False)):
            return base
        L = max(float(getattr(self.cfg, "bolt_length", 0.10)), 1e-6)
        tgt = self._nut_stage_target_axial()
        extra = np.array([
            float(axial) / L,
            float(lateral) / ws,
            float(self._nut_subphase),
            float(self._nut_macro_stage) / 2.0,
            float(tgt - axial) / L,
        ], dtype=np.float64)
        return np.concatenate([base, extra]).astype(np.float64)

    def _nut_point_on_axis(self, idx: int, axial: float) -> np.ndarray:
        """World point at signed depth ``axial`` along bolt ``idx``'s axis."""
        bolt_pos = np.asarray(self.scene.bolt_pose(idx)[0], dtype=np.float64)
        return bolt_pos + self._nut_axis_unit(idx) * float(axial)

    def _nut_staging_axial(self) -> float:
        """Signed depth of the APPROACH staging point (just past the tip).

        Pushed an extra ``nut_insert_margin`` further out in −Y (away from
        the hub) so the socket parks with clearance before the plunge.
        """
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        standoff = float(getattr(self.cfg, "nut_insert_standoff", 0.05))
        margin = float(getattr(self.cfg, "nut_insert_margin", 0.0))
        return 0.5 * L + standoff + margin

    def _nut_ref_center(self) -> np.ndarray:
        """Bolt-ring centroid at the staging depth (hub-and-spoke center)."""
        n_b = len(self.handles.bolts)
        centroid = np.mean(
            [np.asarray(self.scene.bolt_pose(i)[0], dtype=np.float64)
             for i in range(n_b)], axis=0,
        )
        return centroid + self._nut_axis_unit(0) * self._nut_staging_axial()

    def _nut_retract_axial(self) -> float:
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        retract_clear = float(getattr(self.cfg, "nut_retract_clear", 0.03))
        margin = float(getattr(self.cfg, "nut_insert_margin", 0.0))
        return 0.5 * L + retract_clear + 0.03 + margin

    def _capture_nut_lock_quat_if_needed(self) -> None:
        """Capture a reachable coaxial tool quat for APPROACH orientation lock."""
        if self._nut_lock_quat is not None:
            return
        if not bool(getattr(self.cfg, "nut_b_lock_coaxial", True)):
            return
        idx = int(self._nut_target_idx)
        pos = np.asarray(
            self._nut_point_on_axis(idx, self._nut_staging_axial()), dtype=np.float64,
        )
        want_z = -self._nut_axis_unit(idx)
        q = self._ik_b_rollfree(pos, want_z)
        if q is not None:
            rb = self.robot_B
            q_save, _ = rb.joint_state()
            for sl, qq in zip(rb.arm.indices, q):
                p.resetJointState(
                    rb.uid, int(sl), float(qq), targetVelocity=0.0,
                    physicsClientId=self.client,
                )
            self._nut_lock_quat = np.asarray(
                rb.ee_pose()[1], dtype=np.float64,
            ).copy()
            for sl, qq in zip(rb.arm.indices, q_save):
                p.resetJointState(
                    rb.uid, int(sl), float(qq), targetVelocity=0.0,
                    physicsClientId=self.client,
                )
            rb._cmd_q = None
        else:
            self._nut_lock_quat = self._quat_align_tool_z(want_z)

    def _generate_nut_approach_traj(self) -> None:
        """Min-jerk nominal EE path for the current bolt's APPROACH leg."""
        idx = int(self._nut_target_idx)
        rb = self.robot_B
        from_pos = np.asarray(rb.ee_pose()[0], dtype=np.float64)
        staging = np.asarray(
            self._nut_point_on_axis(idx, self._nut_staging_axial()), dtype=np.float64,
        )
        n_steps = int(getattr(self.cfg, "nut_planner_traj_steps", 120))
        n_steps = max(n_steps, 2)

        policy_fastened = len(self._nut_fastened) - int(self._nut_premark)
        if policy_fastened <= 0:
            center = self._nut_ref_center()
            wps = [from_pos, center, staging]
        else:
            # Minimal bolt→bolt move (2026-06-09). Previously the socket backed
            # out to the *previous* bolt's deep retract Y-plane and shuffled
            # across — a large, repetitive in/out detour that looked clumsy.
            # Instead arc directly from the current (retract) pose to the next
            # staging point. Both lie ~one standoff outside the ring face, so a
            # straight chord between adjacent bolts can graze the studs in
            # between; bow the midpoint a little further OUT along the bolt axis
            # (``nut_transit_clear``) so the short arc clears the stud tips.
            axis_out = self._nut_axis_unit(idx)  # points away from the hub face
            clear = float(getattr(self.cfg, "nut_transit_clear", 0.04))
            mid = 0.5 * (from_pos + staging) + axis_out * clear
            wps = [from_pos, mid, staging]

        self._nut_traj_pos = _multi_min_jerk_positions(wps, n_steps)
        self._nut_traj_step = 0
        self._capture_nut_lock_quat_if_needed()
        self._nut_traj_q = self._nut_cart_traj_to_joint_traj(
            self._nut_traj_pos, idx,
        )

    def _nut_cart_traj_to_joint_traj(
        self, positions: np.ndarray, idx: int,
    ) -> np.ndarray:
        """IK-chain a Cartesian min-jerk path into a branch-stable joint lerp."""
        rb = self.robot_B
        want_z = -self._nut_axis_unit(idx)
        q_save, _ = rb.joint_state()
        q_cur = np.asarray(q_save, dtype=np.float64)
        qs = []
        for pos in np.asarray(positions, dtype=np.float64):
            for sl, qq in zip(rb.arm.indices, q_cur):
                p.resetJointState(
                    rb.uid, int(sl), float(qq), targetVelocity=0.0,
                    physicsClientId=self.client,
                )
            q = self._ik_b_rollfree(pos, want_z, n_roll=8, n_seed=2)
            if q is not None:
                q_cur = np.asarray(q, dtype=np.float64)
            qs.append(q_cur.copy())
        for sl, qq in zip(rb.arm.indices, q_save):
            p.resetJointState(
                rb.uid, int(sl), float(qq), targetVelocity=0.0,
                physicsClientId=self.client,
            )
        rb._cmd_q = None
        rb.last_target_pos = rb.ee_pose()[0].copy()
        return np.asarray(qs, dtype=np.float64)

    def _nut_macro_target(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """(pos, quat) the scripted macro commands for the current leg.

        All targets sit exactly on the bolt axis with tool +Z = −axis so
        the forced insert→hold→retract is perfectly coaxial:

        * leg 0 INSERT / 1 HOLD → the hub-face **base** (axial −L/2),
        * leg 2 RETRACT         → a point a margin **past the tip** so the
          socket fully clears the stud (axial L/2 + retract_clear + margin).
        """
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        retract_clear = float(getattr(self.cfg, "nut_retract_clear", 0.03))
        margin = float(getattr(self.cfg, "nut_insert_margin", 0.0))
        if self._nut_macro_stage >= 2:
            # RETRACT: clear the tip + clearance, then a further ``margin``
            # out (+0.03 keeps the target safely past the gate threshold).
            axial_t = 0.5 * L + retract_clear + 0.03 + margin
        else:
            # INSERT: drive to the hub-face base — the deepest reachable
            # point (the hub blocks anything past it). The −Y approach margin
            # makes this a *longer plunge* (deeper-looking insert), not a
            # deeper endpoint; commanding past the base just stalls the PD.
            axial_t = -0.5 * L
        pos = self._nut_point_on_axis(idx, axial_t)
        # Coaxial orientation captured at arrival (preserves the approach
        # roll, only snapping tool +Z exactly onto the axis) so the macro
        # is a pure axial slide with no wrist flip.
        quat = self._nut_macro_quat
        if quat is None:
            quat = self._quat_align_tool_z(-self._nut_axis_unit(idx))
        return pos, quat

    def _coaxial_quat_preserving_roll(self, idx: int) -> np.ndarray:
        """Snap B's *current* tool +Z exactly onto −axis, keeping its roll.

        Minimal rotation mapping the live tool +Z onto the (anti-)bolt axis,
        composed onto the current orientation, so the forced insert keeps the
        wrist roll the approach ended at (no 180° flip) while being perfectly
        coaxial.
        """
        a = self._nut_axis_unit(idx)
        _, cur_quat = self.robot_B.ee_pose()
        cur_quat = np.asarray(cur_quat, dtype=np.float64)
        cur_z = quat_axis(cur_quat, "z")
        cur_z = cur_z / max(float(np.linalg.norm(cur_z)), 1e-9)
        # Pick the axis sign nearer the current +Z so we don't force a flip.
        want_z = -a if float(np.dot(cur_z, -a)) >= float(np.dot(cur_z, a)) else a
        c = float(np.clip(np.dot(cur_z, want_z), -1.0, 1.0))
        if c > 1.0 - 1e-9:
            return cur_quat
        if c < -1.0 + 1e-9:
            axis = np.cross(cur_z, np.array([1.0, 0.0, 0.0]))
            if float(np.linalg.norm(axis)) < 1e-6:
                axis = np.cross(cur_z, np.array([0.0, 1.0, 0.0]))
            ang = math.pi
        else:
            axis = np.cross(cur_z, want_z)
            ang = math.acos(c)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        q_corr = axisangle3_to_quat(axis * ang)
        return quat_multiply(q_corr, cur_quat)

    def _quat_z_roll(self, want_z: np.ndarray, roll: float) -> np.ndarray:
        """Quaternion with tool +Z = ``want_z`` rolled by ``roll`` about +Z."""
        z = np.asarray(want_z, dtype=np.float64)
        z = z / max(float(np.linalg.norm(z)), 1e-9)
        ref = (np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        x0 = np.cross(ref, z)
        x0 = x0 / max(float(np.linalg.norm(x0)), 1e-9)
        y0 = np.cross(z, x0)
        cr, sr = math.cos(roll), math.sin(roll)
        xr = cr * x0 + sr * y0
        yr = -sr * x0 + cr * y0
        m = np.column_stack([xr, yr, z])
        return np.asarray(Rotation.from_matrix(m).as_quat(), dtype=np.float64)

    def _ik_b_rollfree(self, pos: np.ndarray, want_z: np.ndarray,
                       n_roll: int = 16, n_seed: int = 4) -> Optional[np.ndarray]:
        """Roll-free IK: arm joints placing tool_tip at ``pos`` with tool +Z
        = ``want_z``, roll unconstrained (a nut-runner spins freely).

        Sweeps candidate rolls (the fixed-roll solution is often unreachable
        at the hub-face base), evaluating each via a transient
        ``resetJointState`` + FK read, and returns the min-cost joints. The
        live config is restored before returning so the caller's captured
        ``q_from`` stays valid. Mirrors the validated preview oracle IK.
        """
        rb = self.robot_B
        want_z = np.asarray(want_z, dtype=np.float64)
        want_z = want_z / max(float(np.linalg.norm(want_z)), 1e-9)
        pos = np.asarray(pos, dtype=np.float64)
        lo, hi = rb.arm.lower, rb.arm.upper
        q_save, _ = rb.joint_state()
        rest = np.asarray(getattr(rb.arm, "rest", q_save), dtype=np.float64)
        # Seed pool: rest pose, current config, then random restarts so the
        # solver can hop into a different elbow/wrist branch that reaches the
        # (hub-side) base coaxially — a single deterministic seed gets stuck.
        seeds = [rest, np.asarray(q_save, dtype=np.float64)]
        for _ in range(max(0, n_seed - 2)):
            seeds.append(self._np_random.uniform(lo, hi))
        best_q, best_cost = None, 1e9
        for ri in range(n_roll):
            quat = self._quat_z_roll(want_z, 2.0 * math.pi * ri / n_roll)
            for seed in seeds:
                ik = p.calculateInverseKinematics(
                    rb.uid, rb.EE_LINK_INDEX, pos.tolist(), quat.tolist(),
                    lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                    jointRanges=rb.arm.range.tolist(),
                    restPoses=np.asarray(seed, dtype=np.float64).tolist(),
                    maxNumIterations=400, residualThreshold=1e-6,
                    physicsClientId=self.client,
                )
                ik = np.asarray(ik, dtype=np.float64)
                if not (rb._ik_arm_slots and len(ik) > max(rb._ik_arm_slots)):
                    continue
                q = np.clip(ik[rb._ik_arm_slots], lo, hi)
                for sl, qq in zip(rb.arm.indices, q):
                    p.resetJointState(rb.uid, int(sl), float(qq),
                                      targetVelocity=0.0,
                                      physicsClientId=self.client)
                ee, eq = rb.ee_pose()
                dp = float(np.linalg.norm(np.asarray(ee, dtype=np.float64) - pos))
                gz = quat_axis(eq, "z")
                ang = float(angle_between(gz, want_z))
                ang = min(ang, math.pi - ang)
                cost = dp + 0.02 * ang
                if cost < best_cost:
                    best_cost, best_q = cost, q.copy()
            if best_cost < 0.01:
                break
        # Restore the live config so the captured q_from is unperturbed.
        for sl, qq in zip(rb.arm.indices, q_save):
            p.resetJointState(rb.uid, int(sl), float(qq), targetVelocity=0.0,
                              physicsClientId=self.client)
        rb._cmd_q = None
        return best_q

    def _nut_b_forearm_link(self) -> Optional[int]:
        """Cached PyBullet link index of Robot B's ``forearm_link`` (the link
        that collides with A's low arm). ``None`` if not found."""
        if hasattr(self, "_b_forearm_link_cache"):
            return self._b_forearm_link_cache
        idx = None
        rb = self.robot_B
        try:
            for li in range(p.getNumJoints(rb.uid, physicsClientId=self.client)):
                nm = p.getJointInfo(rb.uid, li,
                                    physicsClientId=self.client)[12].decode()
                if nm == "forearm_link":
                    idx = li
                    break
        except p.error:
            idx = None
        self._b_forearm_link_cache = idx
        return idx

    def _nut_order(self) -> List[int]:
        """Resolved bolt-fastening order (a full permutation of all bolt
        indices). Filters ``cfg.nut_bolt_order`` to valid indices, drops
        duplicates, then appends any bolts missing from the list in ascending
        order so the sequence always covers every bolt exactly once.
        """
        n = len(self.handles.bolts)
        raw = getattr(self.cfg, "nut_bolt_order", tuple(range(n)))
        order: List[int] = []
        seen = set()
        for k in raw:
            k = int(k)
            if 0 <= k < n and k not in seen:
                order.append(k)
                seen.add(k)
        for k in range(n):
            if k not in seen:
                order.append(k)
        return order

    def _nut_ba_link_indices(self, uid: int) -> List[int]:
        """Cached arm-link indices (> ``robot_ab_collision_min_link``) of a
        robot, used for the joint-center clearance metric."""
        cache = getattr(self, "_ba_link_idx_cache", None)
        if cache is None:
            cache = {}
            self._ba_link_idx_cache = cache
        if uid in cache:
            return cache[uid]
        min_link = int(getattr(self.cfg, "robot_ab_collision_min_link", 2))
        idxs = [li for li in range(p.getNumJoints(uid, physicsClientId=self.client))
                if li > min_link]
        cache[uid] = idxs
        return idxs

    def _nut_ba_clearance(self) -> float:
        """Minimum distance (m) between Robot A's and Robot B's **joint-center
        points** (link frame origins), over the arm links past the base.

        Joint-center (skeleton) separation is a smoother, mesh-independent
        proxy for "how far B's arm is from A" than surface closest-points: link
        origins move continuously with the joints so the reward gradient has no
        mesh-facet discontinuities. Note the centers sit *inside* the links, so
        even at hard contact this distance floors at roughly the sum of the
        link radii (~0.3 m here) — the reward normalises against that floor.
        """
        if self.robot_A is None or self.robot_B is None:
            return 1.0
        a_idx = self._nut_ba_link_indices(self.robot_A.uid)
        b_idx = self._nut_ba_link_indices(self.robot_B.uid)
        a_pts = [np.asarray(p.getLinkState(
            self.robot_A.uid, li, computeForwardKinematics=True,
            physicsClientId=self.client)[4], dtype=np.float64) for li in a_idx]
        b_pts = [np.asarray(p.getLinkState(
            self.robot_B.uid, li, computeForwardKinematics=True,
            physicsClientId=self.client)[4], dtype=np.float64) for li in b_idx]
        best = 1e9
        for pa in a_pts:
            for pb in b_pts:
                d = float(np.linalg.norm(pa - pb))
                if d < best:
                    best = d
        return best if best < 1e8 else 1.0

    def _filter_nut_socket_collisions(self) -> None:
        """Disable Robot-B ↔ hub & tire collision for the geometric nut task.

        The socket must slide coaxially to the hub-face base; a hard contact
        with the hub flange / mounted tire would shove it off-axis and stall
        the forced macro. Since there are no nut bodies and the seating is
        purely geometric, filtering these pairs is the analogue of the
        tire↔hub mount filter. All B links vs all hub links + the tire.
        """
        if self.handles is None or self.robot_B is None:
            return
        b_uid = int(self.robot_B.uid)
        try:
            n_b = p.getNumJoints(b_uid, physicsClientId=self.client)
        except p.error:
            return
        targets: List[Tuple[int, int]] = []
        hub_uid = int(self.handles.hub.uid)
        try:
            n_hub = p.getNumJoints(hub_uid, physicsClientId=self.client)
        except p.error:
            n_hub = 0
        for link in range(-1, n_hub):
            targets.append((hub_uid, link))
        if self.handles.tire is not None:
            targets.append((int(self.handles.tire), -1))
        for b_link in range(-1, n_b):
            for other_uid, other_link in targets:
                try:
                    p.setCollisionFilterPair(
                        b_uid, other_uid, b_link, other_link, 0,
                        physicsClientId=self.client,
                    )
                except p.error:
                    pass

        # Spurious self-collision: the ``nut_runner`` tool is bolted onto the
        # wrist, so it geometrically overlaps the wrist cluster — but that pair
        # is NOT auto-filtered (only the kinematically-adjacent joint pair is).
        # On the far arc (bolts 4–6) the wrist roll needed to stay coaxial
        # deepens the tool↔wrist_3 overlap, and PyBullet answers with a huge
        # penalty contact (≈16 kN vs ≈4 kN baseline) that shoves the arm ~36 cm
        # off the commanded pose — so it can never dwell at staging to trigger
        # the macro. Disable collision between the tool and the wrist links it
        # is mounted on (a standard adjacent-link filter).
        name_by_link = {}
        for li in range(n_b):
            try:
                info = p.getJointInfo(b_uid, li, physicsClientId=self.client)
                name_by_link[info[12].decode()] = li
            except p.error:
                pass
        tool_links = [
            name_by_link[n] for n in ("nut_runner", "tool_tip", "tool0")
            if n in name_by_link
        ]
        wrist_links = [
            name_by_link[n] for n in (
                "wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link",
            ) if n in name_by_link
        ]
        for tl in tool_links:
            for wl in wrist_links:
                try:
                    p.setCollisionFilterPair(
                        b_uid, b_uid, tl, wl, 0, physicsClientId=self.client,
                    )
                except p.error:
                    pass

    def _precompute_nut_macro_solutions(self, n: int) -> None:
        """Solve + cache the INSERT (base) and RETRACT joint vectors per bolt.

        Pose-independent (the hub is fixed), so a single thorough roll-free
        search per bolt here guarantees the forced macro always reaches the
        hub-face base coaxially — instead of a per-leg IK whose quality
        depends on the live seed/branch.
        """
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        retract_clear = float(getattr(self.cfg, "nut_retract_clear", 0.03))
        margin = float(getattr(self.cfg, "nut_insert_margin", 0.0))
        self._nut_base_q = [None] * n
        self._nut_retract_q = [None] * n
        for i in range(n):
            want_z = -self._nut_axis_unit(i)
            base_pos = self._nut_point_on_axis(i, -0.5 * L)
            retr_pos = self._nut_point_on_axis(
                i, 0.5 * L + retract_clear + 0.03 + margin
            )
            self._nut_base_q[i] = self._ik_b_rollfree(base_pos, want_z)
            self._nut_retract_q[i] = self._ik_b_rollfree(retr_pos, want_z)

    def _setup_nut_macro_leg(self, target_pos: np.ndarray) -> None:
        """Plan a joint-space lerp toward the current leg's cached endpoint.

        ``q_to`` is the pre-solved (base / retract) joint vector for the
        target bolt; the leg interpolates ``q_from → q_to`` over ``leg_len``
        steps. Interpolating in joint space between reachable coaxial
        endpoints keeps the forced slide branch-stable (no mid-plunge flip).
        """
        rb = self.robot_B
        q_from, _ = rb.joint_state()
        idx = int(self._nut_target_idx)
        if self._nut_macro_stage >= 2:
            cached = (self._nut_retract_q[idx]
                      if idx < len(self._nut_retract_q) else None)
        else:
            cached = (self._nut_base_q[idx]
                      if idx < len(self._nut_base_q) else None)
        q_from = np.asarray(q_from, dtype=np.float64)
        # Fall back to a live roll-free solve if the cache is missing.
        if cached is None:
            cached = self._ik_b_rollfree(target_pos, -self._nut_axis_unit(idx))
        cur_pos = np.asarray(rb.ee_pose()[0], dtype=np.float64)
        dist = float(np.linalg.norm(np.asarray(target_pos) - cur_pos))
        stride = max(float(getattr(self.cfg, "nut_macro_step_m", 0.04)), 1e-3)
        self._nut_macro_q_from = q_from
        self._nut_macro_q_to = (
            np.asarray(cached, dtype=np.float64) if cached is not None
            else q_from
        )
        self._nut_macro_leg_len = int(max(1, math.ceil(dist / stride)))

    def _drive_nut_macro(self) -> None:
        """Teleport Robot B along the current macro leg's joint-space lerp.

        Called from ``_apply_action`` when ``_nut_subphase == 1``. Reads the
        leg progress from ``_nut_macro_step`` (incremented in the post-physics
        FSM), interpolates ``q_from → q_to`` (smoothstep), resets the arm to
        that joint vector AND commands the motors there (so the decimation
        physics steps don't PD it back). The result is a smooth, branch-stable
        coaxial slide visible over ``leg_len`` steps.
        """
        rb = self.robot_B
        if self._nut_macro_q_to is None or self._nut_macro_q_from is None:
            return
        leg_len = int(max(1, self._nut_macro_leg_len))
        t = float(np.clip(self._nut_macro_step / leg_len, 0.0, 1.0))
        s = t * t * (3.0 - 2.0 * t)  # smoothstep
        arm_q = (1.0 - s) * self._nut_macro_q_from + s * self._nut_macro_q_to
        arm_q = np.clip(arm_q, rb.arm.lower, rb.arm.upper)
        for slot, q in zip(rb.arm.indices, arm_q):
            p.resetJointState(
                rb.uid, int(slot), targetValue=float(q), targetVelocity=0.0,
                physicsClientId=self.client,
            )
        rb._cmd_q = None
        rb.drive_arm_targets(arm_q)
        rb.last_target_pos = rb.ee_pose()[0].copy()

    def _advance_nut_fastening(self) -> Dict[str, Any]:
        """APPROACH + scripted-macro FSM for sequential nut fastening.

        The policy is responsible for **APPROACH only**: drive the socket to
        the target bolt's staging point (on-axis, just outside the stud tip),
        coaxially (small ``lateral``) and aligned (small ``theta``). Once it
        parks there for ``nut_arrive_steps`` consecutive steps the env emits
        ``arrived`` and hands off to a **forced macro** (``_nut_subphase==1``)
        that drives a deterministic insert→hold→retract straight down/up the
        bolt axis (see ``_drive_nut_macro``). The macro's legs are gated on
        the *measured* socket pose so the tighten is geometrically real:

        * leg 0 INSERT  — seated when axial ≈ −L/2 (hub-face base), coaxial.
        * leg 1 HOLD    — dwell ``nut_hold_steps`` steps → emits ``inserted``.
        * leg 2 RETRACT — cleared when axial ≥ L/2 + clearance, coaxial →
          emits ``fastened`` and advances to the next bolt (APPROACH).

        A per-leg watchdog (``nut_macro_leg_max_steps``) force-advances a
        stalled leg so IK saturation can't hang the episode.

        Returns FSM events consumed by reward + termination.
        """
        events: Dict[str, Any] = {
            "arrived": False,
            "inserted": False,
            "fastened": False,
            "fastened_idx": -1,
            "all_fastened": False,
            "n_fastened": len(self._nut_fastened),
        }
        if self._nut_done:
            return events

        n = len(self.handles.bolts)
        idx = int(self._nut_target_idx)
        self.handles.target_bolt_idx = idx

        L = float(getattr(self.cfg, "bolt_length", 0.10))
        axial, lateral, theta = self._nut_axial_lateral(idx)
        lat_tol = float(getattr(self.cfg, "nut_lateral_tol", 0.015))
        ang_tol = float(getattr(self.cfg, "nut_align_tol_rad", np.deg2rad(15.0)))
        depth_tol = float(getattr(self.cfg, "nut_insert_depth_tol", 0.02))
        retract_clear = float(getattr(self.cfg, "nut_retract_clear", 0.03))
        hold_need = int(getattr(self.cfg, "nut_hold_steps", 12))
        arrive_need = int(getattr(self.cfg, "nut_arrive_steps", 1))
        arrive_pos_tol = float(getattr(self.cfg, "nut_arrive_pos_tol", 0.05))
        arrive_ang_tol = float(
            getattr(self.cfg, "nut_arrive_ang_tol_rad", np.deg2rad(35.0))
        )
        leg_max = int(getattr(self.cfg, "nut_macro_leg_max_steps", 50))
        # Pure-RL: the policy must actually achieve the seat/clear gates itself.
        # Disable the per-leg watchdog so a stall can't force-advance the leg and
        # hand the policy a free R_insert / R_fasten it didn't earn (which would
        # be farmable: arrive, wait out the watchdog, collect). A genuinely stuck
        # bolt then just runs out the horizon with no sparse reward, as it should.
        if bool(getattr(self.cfg, "nut_pure_rl", False)):
            leg_max = 1_000_000_000
        staging_axial = self._nut_staging_axial()

        if self._nut_subphase == 0:
            # APPROACH — trigger the macro once the socket tip is inside a
            # generous capture sphere of the on-axis staging point and roughly
            # aligned. Tight precision is the scripted macro's job (it drives
            # to a cached base IK), so a loose, reachable capture is what lets
            # the policy actually sample the macro reward under exploration.
            d_stage = float(math.hypot(axial - staging_axial, lateral))
            parked = d_stage < arrive_pos_tol and theta < arrive_ang_tol
            # Pure-RL insert is axis-only (±Y): require coaxial alignment at
            # arrive before handing off to insert. v19 tightens this to a
            # dedicated gate ("the nut runner must be exactly above the bolt,
            # aligned along Y, before it may plunge"); the align servo then
            # erases the residual offset during the slide.
            if bool(getattr(self.cfg, "nut_pure_rl", False)):
                arrive_lat = float(getattr(
                    self.cfg, "nut_arrive_lat_tol", 2.0 * lat_tol))
                parked = parked and lateral < arrive_lat
            if parked:
                self._nut_arrive_count += 1
            else:
                self._nut_arrive_count = 0
            if self._nut_arrive_count >= arrive_need:
                # Hand off to the forced insert→hold→retract macro. Capture a
                # coaxial orientation that preserves the approach roll.
                self._nut_subphase = 1
                self._nut_macro_stage = 0
                self._nut_macro_step = 0
                self._nut_hold_count = 0
                self._nut_arrive_count = 0
                # v22 — switch B into the collision-free seat branch BEFORE the
                # plunge. The lug bolt is recessed inside the mounted tire; the
                # approach branch's forearm would otherwise clip the tire mid-
                # plunge and trip nut_collision_fail ~6 cm short of the seat.
                # The warm-started axial servo keeps this clean branch to seat.
                # Clear stale endpoints first so a failed switch (no clean
                # branch for this bolt) can't reuse the previous bolt's lerp.
                self._nut_clean_approach_q = None
                self._nut_clean_plunge_from = None
                self._nut_clean_plunge_to = None
                self._nut_clean_stage_retract = None
                self._nut_clean_prep_path = None
                if bool(getattr(self.cfg, "nut_b_clean_branch_insert", False)):
                    if self._nut_prepare_clean_branch(idx):
                        if bool(getattr(self, "_nut_clean_use_prep", False)):
                            self._nut_macro_stage = -1
                        else:
                            # Path clips the tire — snap once to staging.
                            self._nut_snap_to_clean_staging()
                            self._nut_macro_stage = 0
                    else:
                        self._nut_macro_stage = 0
                self._nut_macro_quat = self._coaxial_quat_preserving_roll(idx)
                # Plan the INSERT leg (current staging → hub-face base).
                self._setup_nut_macro_leg(self._nut_macro_target(idx)[0])
                # Pure-RL: fresh axial-PB baseline so the first INSERT step
                # doesn't book a spurious Δ from the (stale) approach phase.
                self._prev_axial_err_B = None
                # v21 — fresh branch-aware plunge bookkeeping for this leg.
                self._nut_last_reseat_step = -(10 ** 9)
                events["arrived"] = True
                self._set_bolt_color(idx, self._NUT_COLOR_RETRACT)
            return events

        # ---- _nut_subphase == 1: scripted macro (env-driven) --------------
        margin = float(getattr(self.cfg, "nut_insert_margin", 0.0))
        self._nut_macro_step += 1
        stalled = self._nut_macro_step >= leg_max
        if self._nut_macro_stage == -1:
            # PREP — smooth joint lerp from the policy's approach config into
            # the collision-free staging branch (no teleport). Advance to INSERT
            # once the lerp completes (or immediately if prep_len==0).
            prep_len = int(getattr(self.cfg, "nut_clean_prep_len", 30))
            if prep_len <= 0 or self._nut_macro_step >= prep_len:
                self._nut_macro_stage = 0
                self._nut_macro_step = 0
        elif self._nut_macro_stage == 0:
            # INSERT — wait until the socket has seated at the hub-face base
            # (deepest reachable; the −Y approach margin makes the plunge a
            # longer/deeper stroke, but the endpoint is still the base).
            seat_lat = lat_tol * float(
                getattr(self.cfg, "nut_seat_lat_mult", 2.0))
            seated = (
                abs(axial - (-0.5 * L)) < depth_tol and lateral < seat_lat
            )
            # v21 — branch-aware plunge. The env-driven servo warm-starts IK
            # from the APPROACH branch; at the workspace edge that branch can
            # saturate ~1-2 cm short of the seat, so the gate never fires and
            # the leg stalls. The bolt IS reachable from a different branch, so
            # once the plunge has clearly overrun a normal stroke without
            # seating, search for a reaching seat branch and switch B into it;
            # the servo then seats. Retries periodically while stuck. Policy
            # never owned this DOF, so this is pure env-side control (no retrain).
            reseat_after = int(getattr(self.cfg, "nut_insert_reseat_after", 40))
            if (not seated
                    and bool(getattr(self.cfg, "nut_b_insert_branch_search", False))
                    and bool(getattr(self.cfg, "nut_pure_rl", False))
                    and self._nut_macro_step >= reseat_after
                    and (self._nut_macro_step - self._nut_last_reseat_step)
                    >= reseat_after):
                self._nut_last_reseat_step = int(self._nut_macro_step)
                if self._nut_switch_to_seat_branch(idx):
                    axial, lateral, _th = self._nut_axial_lateral(idx)
                    seated = (abs(axial - (-0.5 * L)) < depth_tol
                              and lateral < seat_lat)
            if seated or stalled:
                self._nut_macro_stage = 1
                self._nut_macro_step = 0
                self._nut_hold_count = 0
                # HOLD stays seated: freeze the lerp at the base joints.
                if self._nut_macro_q_to is not None:
                    self._nut_macro_q_from = self._nut_macro_q_to.copy()
                self._nut_macro_leg_len = 1
        elif self._nut_macro_stage == 1:
            # HOLD — dwell seated for the tighten window.
            self._nut_hold_count += 1
            if self._nut_hold_count >= hold_need or stalled:
                self._nut_macro_stage = 2
                self._nut_macro_step = 0
                self._nut_hold_count = 0
                # Plan the RETRACT leg (base → clear past the tip + margin).
                self._setup_nut_macro_leg(self._nut_macro_target(idx)[0])
                events["inserted"] = True
                self._prev_axial_B = None
                # Pure-RL: the stage target jumps base → cleared point, so reset
                # the axial-PB baseline to avoid a one-step reward spike.
                self._prev_axial_err_B = None
        else:
            # RETRACT — wait until the socket clears the tip + clearance, and
            # the extra ``margin`` further out (symmetric to the deeper insert).
            cleared = (
                axial >= (0.5 * L + retract_clear + margin)
                and lateral < 2.5 * lat_tol
            )
            if cleared or stalled:
                self._nut_fastened.append(idx)
                self._set_bolt_color(idx, self._NUT_COLOR_FASTENED)
                self._nut_subphase = 0
                self._nut_macro_stage = 0
                self._nut_macro_step = 0
                self._nut_hold_count = 0
                self._nut_arrive_count = 0
                self._nut_macro_quat = None
                self._nut_macro_q_from = None
                self._nut_macro_q_to = None
                self._nut_macro_leg_len = 1
                events["fastened"] = True
                events["fastened_idx"] = idx
                events["n_fastened"] = len(self._nut_fastened)
                if len(self._nut_fastened) >= n:
                    self._nut_done = True
                    events["all_fastened"] = True
                else:
                    # Advance along the explicit fastening order, skipping any
                    # positions whose bolt is already fastened.
                    order = self._nut_order()
                    self._nut_seq_pos += 1
                    while (self._nut_seq_pos < len(order)
                           and order[self._nut_seq_pos] in self._nut_fastened):
                        self._nut_seq_pos += 1
                    self._nut_seq_pos = min(self._nut_seq_pos, len(order) - 1)
                    self._nut_target_idx = order[self._nut_seq_pos]
                    self.handles.target_bolt_idx = self._nut_target_idx
                    self._set_bolt_color(
                        self._nut_target_idx, self._NUT_COLOR_TARGET
                    )
                    # Reset PB shaping baselines so the bolt switch doesn't
                    # inject a spurious large Δ on the next step.
                    self._prev_d_B = None
                    self._prev_axial_B = None
                    self._prev_lateral_B = None
                    self._prev_axial_err_B = None
                    if bool(getattr(self.cfg, "nut_b_planner_residual", False)):
                        self._generate_nut_approach_traj()
        return events

    def _nut_best_seat_q(self, idx: int) -> Optional[np.ndarray]:
        """Strongest reachable coaxial seat config for bolt ``idx``.

        Runs ``nut_insert_reseat_tries`` independent roll-free, multi-seed IK
        restarts to the hub-face base (``axial = −L/2``) and returns the joint
        vector achieving the DEEPEST coaxial socket pose. The extra restarts
        (vs a single ``_ik_b_rollfree`` call) are what reliably find the
        elbow/wrist branch that reaches the workspace-edge bolts — a single
        warm-started solve stays in the (short-reaching) approach branch.
        Returns ``None`` if no candidate is found. Restores the live config.
        """
        rb = self.robot_B
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        axis = self._nut_axis_unit(idx)
        bolt = np.asarray(self.scene.bolt_pose(idx)[0], dtype=np.float64)
        seat_pt = bolt + axis * (-0.5 * L)
        want_z = -axis / max(float(np.linalg.norm(axis)), 1e-9)
        lo, hi = rb.arm.lower, rb.arm.upper
        tries = max(1, int(getattr(self.cfg, "nut_insert_reseat_tries", 8)))
        q_save, _ = rb.joint_state()
        best_q, best_cost = None, 1e9
        # Mirror the validated oracle seat IK (``scripts._best_b_ik``): seed only
        # from the rest pose + fresh random restarts (NOT the live config — that
        # biases every solve back into the short-reaching stuck branch). Each
        # ``try`` is an independent random sweep; the deepest+coaxial wins.
        for tri in range(tries):
            rng = np.random.default_rng(
                11 + idx * 131 + tri * 17 + int(self._step_count))
            for ri in range(16):
                quat = np.asarray(
                    self._quat_z_roll(want_z, 2.0 * math.pi * ri / 16),
                    dtype=np.float64,
                ).tolist()
                for k in range(4):
                    seed = (rb.arm.rest if k == 0
                            else rng.uniform(lo, hi)).tolist()
                    ik = p.calculateInverseKinematics(
                        rb.uid, rb.EE_LINK_INDEX, seat_pt.tolist(), quat,
                        lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                        jointRanges=rb.arm.range.tolist(), restPoses=seed,
                        maxNumIterations=400, residualThreshold=1e-6,
                        physicsClientId=self.client,
                    )
                    ik = np.asarray(ik, dtype=np.float64)
                    if not (rb._ik_arm_slots and len(ik) > max(rb._ik_arm_slots)):
                        continue
                    q = np.clip(ik[rb._ik_arm_slots], lo, hi)
                    for s, qq in zip(rb.arm.indices, q):
                        p.resetJointState(rb.uid, int(s), float(qq),
                                          targetVelocity=0.0,
                                          physicsClientId=self.client)
                    ee, eq = rb.ee_pose()
                    dp = float(np.linalg.norm(
                        np.asarray(ee, dtype=np.float64) - seat_pt))
                    gz = quat_axis(eq, "z")
                    ang = float(angle_between(gz, want_z))
                    ang = min(ang, math.pi - ang)
                    cost = dp + 0.02 * ang
                    if cost < best_cost:
                        best_cost = cost
                        best_q = np.asarray(q, dtype=np.float64).copy()
                if best_cost < 0.01:
                    break
            if best_cost < 0.01:
                break
        # Restore the live config; the caller commits the branch only if useful.
        for s, qq in zip(rb.arm.indices, q_save):
            p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=self.client)
        rb._cmd_q = None
        return best_q

    def _nut_switch_to_seat_branch(self, idx: int) -> bool:
        """On a stalled plunge, switch Robot B into a reachable seat branch.

        Finds the deepest coaxial seat config (:meth:`_nut_best_seat_q`) and,
        only if it is *strictly deeper* than the stuck plunge AND reaches the
        seat band, commits it (``resetJointState`` — a one-time branch swap at a
        config the oracle proved collision-free) and re-seeds the servo target.
        Returns ``True`` iff B was switched into a seated branch.
        """
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        depth_tol = float(getattr(self.cfg, "nut_insert_depth_tol", 0.02))
        lat_tol = float(getattr(self.cfg, "nut_lateral_tol", 0.015))
        seat_lat = lat_tol * float(getattr(self.cfg, "nut_seat_lat_mult", 2.0))
        rb = self.robot_B
        q_save, _ = rb.joint_state()
        # Freeze the GUI during the seat-branch IK search (resetJointState
        # restarts would otherwise flicker on screen); no effect headless.
        with self._render_frozen():
            q = self._nut_best_seat_q(idx)
        if q is None:
            return False
        for s, qq in zip(rb.arm.indices, q):
            p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=self.client)
        ax, lat, _th = self._nut_axial_lateral(idx)
        if abs(ax - (-0.5 * L)) < depth_tol and lat < seat_lat:
            rb._cmd_q = None
            rb.drive_arm_targets(np.asarray(q, dtype=np.float64))
            rb.last_target_pos = np.asarray(
                rb.ee_pose()[0], dtype=np.float64).copy()
            return True
        # Best branch still can't seat — restore the live (stuck) config so the
        # plunge servo keeps trying rather than freezing in a worse pose.
        for s, qq in zip(rb.arm.indices, q_save):
            p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=self.client)
        rb._cmd_q = None
        return False

    # ------------------------------------------------------------------
    # v22 — collision-aware clean-branch INSERT
    # ------------------------------------------------------------------
    def _nut_tire_penetration(self) -> float:
        """Most-negative tire-vs-RobotB closest-point distance over non-base B
        links (``< 0`` ⇒ penetration depth). ``getClosestPoints`` is geometric
        (independent of the last ``stepSimulation``), so it is correct right
        after a ``resetJointState`` teleport — exactly what the IK solve needs.
        """
        tire = getattr(self.handles, "tire", None)
        if tire is None:
            return 0.0
        worst = 0.0
        for cp in p.getClosestPoints(bodyA=tire, bodyB=self.robot_B.uid,
                                     distance=0.05, physicsClientId=self.client):
            if len(cp) > 8 and int(cp[4]) > 1:
                worst = min(worst, float(cp[8]))
        return worst

    def _nut_shortest_arm_target(
            self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """FK-identical ``q_to`` reached via the shortest per-joint path from
        ``q_from`` (each revolute joint may shift by ``±2πk``)."""
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        out = q_to.copy()
        rb = self.robot_B
        lo, hi = rb.arm.lower, rb.arm.upper
        two_pi = 2.0 * np.pi
        for i in range(out.shape[0]):
            delta = q_to[i] - q_from[i]
            wrapped = delta - two_pi * np.round(delta / two_pi)
            cand = q_from[i] + wrapped
            if lo[i] <= cand <= hi[i]:
                out[i] = cand
        return out

    def _nut_joint_path_cost(
            self, q_from: np.ndarray, q_to: np.ndarray) -> float:
        """Sum of squared shortest-path joint deltas (rad²)."""
        q_from = np.asarray(q_from, dtype=np.float64)
        q_short = self._nut_shortest_arm_target(q_from, q_to)
        d = q_short - q_from
        return float(np.dot(d, d))

    def _nut_clean_seat_q(
            self, idx: int,
            seed_q: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Collision-AWARE coaxial seat config for bolt ``idx``.

        scipy least_squares with a residual that jointly minimises
        [seat-point error, coaxiality, tire penetration beyond the 5 mm
        ``_in_bad_collision`` tolerance], multi-restart over joint branches.
        A tire-free coaxial full-seat provably exists for every lug bolt; this
        finds it where the collision-blind ``_ik_b_rollfree`` / ``_nut_best_seat_q``
        return a branch whose forearm clips the mounted tire. Restores the live
        config; returns ``None`` if no clean seat is found.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError:
            return None
        rb = self.robot_B
        cl = self.client
        L = float(getattr(self.cfg, "bolt_length", 0.10))
        axis = self._nut_axis_unit(idx)
        bolt = np.asarray(self.scene.bolt_pose(idx)[0], dtype=np.float64)
        seat_pt = bolt + axis * (-0.5 * L)
        want_z = -axis / max(float(np.linalg.norm(axis)), 1e-9)
        lo, hi = rb.arm.lower, rb.arm.upper
        depth_tol = float(getattr(self.cfg, "nut_insert_depth_tol", 0.007))
        q_save, _ = rb.joint_state()

        def set_q(q):
            for s, qq in zip(rb.arm.indices, q):
                p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                                  physicsClientId=cl)

        def resid(q):
            set_q(q)
            ee, eq = rb.ee_pose()
            ee = np.asarray(ee, dtype=np.float64)
            gz = np.asarray(quat_axis(eq, "z"), dtype=np.float64)
            mis = min(float(np.linalg.norm(gz - want_z)),
                      float(np.linalg.norm(gz + want_z)))
            viol = max(0.0, -(self._nut_tire_penetration()) - 0.005)
            return np.concatenate([(ee - seat_pt) * 10.0, [mis * 2.0, viol * 40.0]])

        rng = np.random.default_rng(977 + idx * 131)
        n_restart = int(getattr(self.cfg, "nut_clean_seat_restarts", 80))
        seed_q = (np.asarray(seed_q, dtype=np.float64)
                  if seed_q is not None else None)
        best_q, best_key = None, None
        for r in range(n_restart):
            if seed_q is not None and r == 0:
                q0 = seed_q.copy()
            elif seed_q is not None and r < 8:
                q0 = np.clip(
                    seed_q + rng.uniform(-0.35, 0.35, size=len(seed_q)),
                    lo, hi)
            elif r == (0 if seed_q is None else 8):
                q0 = np.asarray(rb.arm.rest, dtype=np.float64)
            else:
                q0 = rng.uniform(lo, hi)
            try:
                sol = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10,
                                    ftol=1e-10, max_nfev=400, diff_step=2e-3)
            except Exception:
                continue
            q = np.clip(sol.x, lo, hi)
            set_q(q)
            ax, lat, _t = self._nut_axial_lateral(idx)
            seated = (abs(ax - (-0.5 * L)) < depth_tol and lat < 0.015)
            clean = self._nut_tire_penetration() >= -0.005
            path_cost = (self._nut_joint_path_cost(seed_q, q)
                         if seed_q is not None else 0.0)
            key = (bool(seated and clean), bool(seated), -path_cost,
                   float(self._nut_tire_penetration()))
            if best_key is None or key > best_key:
                best_key, best_q = key, q.copy()
            if best_key[0]:
                break
        set_q(q_save)
        rb._cmd_q = None
        if best_q is not None and seed_q is not None:
            best_q = self._nut_shortest_arm_target(seed_q, best_q)
        return best_q if (best_key is not None and best_key[0]) else None

    def _nut_clean_staging_q(
            self, idx: int,
            seed_q: Optional[np.ndarray] = None,
            seat_q: Optional[np.ndarray] = None,
            return_raw: bool = False):
        """Staging config in the SAME clean branch as :meth:`_nut_clean_seat_q`.

        IK to the on-axis staging point (just outside the stud tip), seeded from
        the clean seat config and penalising tire penetration, so the approach
        sits in the collision-free branch. The axial plunge (``apply_absolute_ee``
        warm-starts from the current joints) then stays in that branch all the
        way to the seat. Cached per bolt (in-memory + optional disk) unless
        ``nut_clean_approach_seed`` is active (staging then depends on the live
        approach winding).
        """
        approach_seed = bool(getattr(self.cfg, "nut_clean_approach_seed", False))
        seed_q = (np.asarray(seed_q, dtype=np.float64)
                  if seed_q is not None else None)
        cache = getattr(self, "_nut_clean_stage_cache", None)
        if cache is None:
            cache = {}
            path = str(getattr(self.cfg, "nut_clean_seat_cache", "") or "")
            if path and not approach_seed:
                try:
                    import os
                    if os.path.exists(path):
                        data = np.load(path)
                        for k in data.files:
                            cache[int(k)] = np.asarray(data[k], dtype=np.float64)
                except Exception:
                    pass
            self._nut_clean_stage_cache = cache
        if not approach_seed and idx in cache:
            return cache[idx]
        try:
            from scipy.optimize import least_squares
        except ImportError:
            if not approach_seed:
                cache[idx] = None
            return None
        if seat_q is None:
            seat_q = self._nut_clean_seat_q(idx, seed_q=seed_q)
        if seat_q is None:
            if not approach_seed:
                cache[idx] = None
            return None
        seat_q = np.asarray(seat_q, dtype=np.float64)
        rb = self.robot_B
        cl = self.client
        axis = self._nut_axis_unit(idx)
        stage_pt = self._nut_point_on_axis(idx, self._nut_staging_axial())
        want_z = -axis / max(float(np.linalg.norm(axis)), 1e-9)
        lo, hi = rb.arm.lower, rb.arm.upper
        q_save, _ = rb.joint_state()

        def set_q(q):
            for s, qq in zip(rb.arm.indices, q):
                p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                                  physicsClientId=cl)

        def resid(q):
            set_q(q)
            ee, eq = rb.ee_pose()
            ee = np.asarray(ee, dtype=np.float64)
            gz = np.asarray(quat_axis(eq, "z"), dtype=np.float64)
            mis = min(float(np.linalg.norm(gz - want_z)),
                      float(np.linalg.norm(gz + want_z)))
            viol = max(0.0, -(self._nut_tire_penetration()) - 0.005)
            return np.concatenate([(ee - stage_pt) * 10.0, [mis * 2.0, viol * 40.0]])

        rng = np.random.default_rng(613 + idx * 97)
        best_q, best_key = None, None
        for r in range(40):
            if seed_q is not None and r == 0:
                q0 = seed_q.copy()
            elif seed_q is not None and r < 6:
                q0 = np.clip(
                    seed_q + rng.uniform(-0.25, 0.25, size=len(seed_q)),
                    lo, hi)
            elif r == (0 if seed_q is None else 6):
                q0 = np.asarray(seat_q, dtype=np.float64)
            else:
                q0 = np.clip(seat_q + rng.uniform(-0.3, 0.3, size=len(seat_q)),
                               lo, hi)
            try:
                sol = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10,
                                    ftol=1e-10, max_nfev=300, diff_step=2e-3)
            except Exception:
                continue
            q = np.clip(sol.x, lo, hi)
            set_q(q)
            ee = np.asarray(rb.ee_pose()[0], dtype=np.float64)
            pe = float(np.linalg.norm(ee - stage_pt))
            clean = self._nut_tire_penetration() >= -0.005
            path_cost = (self._nut_joint_path_cost(seed_q, q)
                         if seed_q is not None else 0.0)
            key = (bool(pe < 0.01 and clean), bool(clean), -path_cost, -pe)
            if best_key is None or key > best_key:
                best_key, best_q = key, q.copy()
            if best_key[0]:
                break
        set_q(q_save)
        rb._cmd_q = None
        result_raw = (best_q if (best_key is not None and best_key[1]) else None)
        result = result_raw
        if result is not None and seed_q is not None:
            result = self._nut_shortest_arm_target(seed_q, result)
        if not approach_seed:
            cache[idx] = result
        path = str(getattr(self.cfg, "nut_clean_seat_cache", "") or "")
        if path and not approach_seed and result is not None:
            try:
                import os
                import tempfile
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                payload = {str(k): v for k, v in cache.items() if v is not None}
                fd, tmp = tempfile.mkstemp(
                    dir=os.path.dirname(path) or ".", suffix=".npz")
                os.close(fd)
                np.savez(tmp, **payload)
                os.replace(tmp, path)
            except Exception:
                pass
        if return_raw:
            return result, result_raw
        return result

    def _nut_joint_lerp_path_tire_clean(
            self, q_from: np.ndarray, q_to: np.ndarray, steps: int) -> bool:
        """True iff a joint-space smoothstep from ``q_from``→``q_to`` stays
        tire-free at every sampled configuration."""
        rb = self.robot_B
        q_save, _ = rb.joint_state()
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        lo, hi = rb.arm.lower, rb.arm.upper
        n = max(1, int(steps))
        ok = True
        for i in range(n + 1):
            t = float(i) / float(n)
            s = t * t * (3.0 - 2.0 * t)
            q = np.clip((1.0 - s) * q_from + s * q_to, lo, hi)
            for slot, qq in zip(rb.arm.indices, q):
                p.resetJointState(rb.uid, int(slot), float(qq),
                                  targetVelocity=0.0,
                                  physicsClientId=self.client)
            if self._nut_tire_penetration() < -0.005:
                ok = False
                break
        for slot, qq in zip(rb.arm.indices, q_save):
            p.resetJointState(rb.uid, int(slot), float(qq), targetVelocity=0.0,
                              physicsClientId=self.client)
        rb._cmd_q = None
        return ok

    def _nut_snap_to_clean_staging(self) -> None:
        """Instant fallback: jump B to the cached clean staging config."""
        stage_q = getattr(self, "_nut_clean_plunge_from", None)
        if stage_q is not None:
            self._nut_set_arm_q(np.asarray(stage_q, dtype=np.float64))

    def _nut_pathB_clear(self, q_from: np.ndarray, q_to: np.ndarray,
                         steps: int, tire_margin: float = -0.005,
                         body_margin: float = 0.008) -> bool:
        """True iff the joint smoothstep ``q_from→q_to`` keeps Robot B clear of
        the tire, Robot A, the floor and the walls/vehicle at every sample.

        Per-body tolerance (mirrors the live ``_in_bad_collision`` gate but
        geometrically, valid right after a teleport):

        * tire — allowed to sit close (the staging capture parks the socket just
          outside the stud); reject only on real penetration (< ``tire_margin``,
          5 mm, same as the kinematic-sync revert).
        * Robot A / floor / walls / vehicle — must never be near; reject within
          ``body_margin`` (8 mm) so the planned path will not trip the
          zero-tolerance ``nut_collision_fail`` during execution.
        """
        rb = self.robot_B
        lo, hi = rb.arm.lower, rb.arm.upper
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        q_save, _ = rb.joint_state()
        tire = getattr(self.handles, "tire", None)
        floor_bodies = [getattr(self.handles, "plane", None)]
        floor_bodies.extend(getattr(self.handles, "floor_rim", []) or [])
        floor_bodies = [b for b in floor_bodies if b is not None]
        min_link = int(getattr(self.cfg, "robot_ab_collision_min_link", 2))
        vehicle = getattr(self.handles, "vehicle", None)
        back_wall = getattr(self.handles, "cargo_back_wall", None)
        n = max(1, int(steps))
        ok = True

        def too_close(body, link_filter, margin):
            for cp in p.getClosestPoints(bodyA=rb.uid, bodyB=body,
                                         distance=float(margin) + 0.02,
                                         physicsClientId=self.client):
                if link_filter(int(cp[3]), int(cp[4])) and float(cp[8]) < margin:
                    return True
            return False

        for i in range(n + 1):
            t = float(i) / float(n)
            s = t * t * (3.0 - 2.0 * t)
            q = np.clip((1.0 - s) * q_from + s * q_to, lo, hi)
            for slot, qq in zip(rb.arm.indices, q):
                p.resetJointState(rb.uid, int(slot), float(qq),
                                  targetVelocity=0.0, physicsClientId=self.client)
            hit = False
            if tire is not None and too_close(
                    tire, lambda lb, lo_: lb > 1, tire_margin):
                hit = True
            if not hit:
                for fb in floor_bodies:
                    if too_close(fb, lambda lb, lo_: lb > 1, body_margin):
                        hit = True
                        break
            if not hit and too_close(
                    self.robot_A.uid,
                    lambda lb, la: lb > min_link or la > min_link, body_margin):
                hit = True
            if not hit and vehicle is not None and too_close(
                    vehicle, lambda lb, lo_: lb > 1, body_margin):
                hit = True
            if not hit and back_wall is not None and too_close(
                    back_wall, lambda lb, lo_: lb > 1, body_margin):
                hit = True
            if hit:
                ok = False
                break
        for slot, qq in zip(rb.arm.indices, q_save):
            p.resetJointState(rb.uid, int(slot), float(qq), targetVelocity=0.0,
                              physicsClientId=self.client)
        rb._cmd_q = None
        return ok

    def _nut_clean_prep_path_q(
            self, idx: int, approach_q: np.ndarray, stage_q: np.ndarray,
            seg_steps: int):
        """Collision-free waypoint path so approach→staging avoids the tire.

        When the direct joint lerp ``approach→stage_q`` clips the tire (which
        otherwise forces a one-frame snap), plan a short collision-free joint
        path through *outward* waypoints ``W`` on the bolt axis — further from
        the hub than the staging point, hence tire-clear. Tries a single
        waypoint first; if no single-waypoint path is clear, prepends an axial
        pre-lift of the approach (back the socket straight out along the bolt
        axis, an almost-always-clear short move) and searches again.

        Returns a list ``[W1, ...]`` of INTERMEDIATE configs (endpoints excluded)
        such that ``approach→W1→…→stage_q`` is clear by margin at every sample,
        or ``None``. Pure planning: IK + geometric collision sampling only, so
        the macro endpoints — and the trained chain — are untouched.
        """
        rb = self.robot_B
        approach_q = np.asarray(approach_q, dtype=np.float64)
        stage_q = np.asarray(stage_q, dtype=np.float64)
        quat = self._coaxial_quat_preserving_roll(idx)
        base_ax = self._nut_staging_axial()
        axis = self._nut_axis_unit(idx)
        seg = max(1, int(seg_steps))
        stage_pt = self._nut_point_on_axis(idx, base_ax)
        try:
            radial = stage_pt - self._nut_ref_center()
            radial = radial - float(np.dot(radial, axis)) * axis
            rn = float(np.linalg.norm(radial))
            radial = radial / rn if rn > 1e-6 else np.zeros(3)
        except Exception:
            radial = np.zeros(3)
        q_save, _ = rb.joint_state()

        def _set(q):
            for s, qq in zip(rb.arm.indices, q):
                p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                                  physicsClientId=self.client)

        # Outward staging waypoint candidates (axial standoff × radial lift).
        def _outward_W():
            for out in (0.10, 0.16, 0.24, 0.32, 0.42, 0.55):
                for rad in (0.0, 0.08, 0.16, 0.28):
                    w_pos = (self._nut_point_on_axis(idx, base_ax + out)
                             + radial * rad)
                    for warm in (stage_q, approach_q):
                        try:
                            _set(warm)
                            W = rb.solve_arm_joints_in_snapshot(
                                w_pos, quat, warm)
                        except Exception:
                            continue
                        yield np.asarray(W, dtype=np.float64)

        result = None
        # --- single waypoint: approach → W → stage ---------------------------
        for W in _outward_W():
            if (self._nut_pathB_clear(approach_q, W, seg)
                    and self._nut_pathB_clear(W, stage_q, seg)):
                result = [W]
                break
        # --- two waypoints: approach → A_lift → W → stage --------------------
        if result is None:
            _set(approach_q)
            ee_app = np.asarray(rb.ee_pose()[0], dtype=np.float64)
            eq_app = np.asarray(rb.ee_pose()[1], dtype=np.float64)
            for lift in (0.10, 0.18, 0.28):
                a_pos = ee_app + axis * lift
                try:
                    _set(approach_q)
                    A_lift = np.asarray(
                        rb.solve_arm_joints_in_snapshot(a_pos, eq_app,
                                                        approach_q),
                        dtype=np.float64)
                except Exception:
                    continue
                if not self._nut_pathB_clear(approach_q, A_lift, seg):
                    continue
                for W in _outward_W():
                    if (self._nut_pathB_clear(A_lift, W, seg)
                            and self._nut_pathB_clear(W, stage_q, seg)):
                        result = [A_lift, W]
                        break
                if result is not None:
                    break
        _set(q_save)
        rb._cmd_q = None
        return result

    def _nut_prepare_clean_branch(self, idx: int) -> bool:
        """Cache clean-branch endpoints at APPROACH→INSERT handoff (no teleport).

        Saves the live approach joint vector and the collision-free staging/seat
        configs solved by :meth:`_nut_clean_staging_q` / :meth:`_nut_clean_seat_q`.
        :meth:`_nut_drive_clean_macro` then smoothsteps approach→staging (PREP
        leg, ``macro_stage==-1``) before INSERT/HOLD/RETRACT. Returns ``False``
        if no clean staging config exists.
        """
        # The whole handoff is pure planning: IK restarts and collision sweeps
        # teleport B all over via resetJointState, then restore. In GUI mode
        # those intermediate poses get drawn — the "flicker / pose snaps away and
        # back" the user sees. Freeze the visualiser for the duration so only the
        # final committed pose is shown (no effect in DIRECT/headless).
        with self._render_frozen():
            return self._nut_prepare_clean_branch_impl(idx)

    def _nut_prepare_clean_branch_impl(self, idx: int) -> bool:
        rb = self.robot_B
        approach_q, _ = rb.joint_state()
        approach_q = np.asarray(approach_q, dtype=np.float64)
        seed = (approach_q if bool(getattr(self.cfg, "nut_clean_approach_seed",
                                            False)) else None)
        seat_q = self._nut_clean_seat_q(idx, seed_q=seed)
        if seat_q is None:
            return False
        stage_out = self._nut_clean_staging_q(
            idx, seed_q=seed, seat_q=seat_q, return_raw=(seed is not None))
        if seed is not None:
            stage_prep, stage_raw = stage_out
        else:
            stage_prep = stage_out
            stage_raw = stage_out
        if stage_prep is None:
            return False
        stage_q = np.asarray(stage_prep, dtype=np.float64)
        seat_q = np.asarray(seat_q, dtype=np.float64)
        # Lightweight winding-cleanup (no IK re-solve): re-express the raw IK
        # staging on the joint winding nearest the live approach config so the
        # scripted PREP leg no longer spins a wrist joint a full turn. The
        # RETRACT/resume endpoint is moved to the SAME winding (kept consistent,
        # unlike v23's approach-seed hybrid) so the policy resumes on a single,
        # FK-identical staging config. obs at the resume boundary changes only by
        # FK-identical ±2πk; verify the chain still holds before relying on it.
        if (not bool(getattr(self.cfg, "nut_clean_approach_seed", False))
                and bool(getattr(self.cfg, "nut_clean_shortest_macro", False))):
            stage_q = self._nut_shortest_arm_target(approach_q, stage_q)
            stage_raw = stage_q
        seat_insert = self._nut_shortest_arm_target(stage_q, seat_q)
        plunge_len = int(getattr(self.cfg, "nut_clean_plunge_len", 25))
        if not self._nut_joint_lerp_path_tire_clean(stage_q, seat_insert,
                                                      plunge_len):
            seat_insert = seat_q
        self._nut_clean_approach_q = approach_q.copy()
        self._nut_clean_plunge_from = stage_q.copy()
        self._nut_clean_plunge_to = seat_insert.copy()
        # RETRACT ends on the raw IK staging winding so v22 policies (trained
        # on the un-expressed branch) stay in-distribution until v23 fine-tune
        # converges; PREP/INSERT still use the approach-nearest winding.
        self._nut_clean_stage_retract = np.asarray(stage_raw, dtype=np.float64).copy()
        skip = float(getattr(self.cfg, "nut_clean_prep_skip_rad", 0.15))
        self._nut_clean_skip_prep = (
            float(np.linalg.norm(approach_q - stage_q)) < skip
        )
        prep_len = int(getattr(self.cfg, "nut_clean_prep_len", 30))
        self._nut_clean_prep_path = None
        if prep_len <= 0 or self._nut_clean_skip_prep:
            self._nut_clean_use_prep = False
        else:
            self._nut_clean_use_prep = self._nut_joint_lerp_path_tire_clean(
                approach_q, stage_q, prep_len)
            # "Real robot" cleanup: the direct PREP lerp clips the tire and
            # would snap in one frame. Route it through a collision-free outward
            # waypoint path so the approach→staging move is smooth and
            # continuous. Endpoints unchanged → policy/chain unaffected.
            if (not self._nut_clean_use_prep
                    and bool(getattr(self.cfg,
                                     "nut_clean_macro_smooth", False))):
                path = self._nut_clean_prep_path_q(
                    idx, approach_q, stage_q,
                    max(1, prep_len // 3))
                if path:
                    self._nut_clean_prep_path = [
                        np.asarray(w, dtype=np.float64).copy() for w in path]
                    self._nut_clean_use_prep = True
        return self._nut_clean_plunge_to is not None

    def _nut_set_arm_q(self, arm_q: np.ndarray) -> None:
        """Teleport + motor-command Robot B's arm to ``arm_q`` (joint space)."""
        rb = self.robot_B
        arm_q = np.clip(np.asarray(arm_q, dtype=np.float64),
                        rb.arm.lower, rb.arm.upper)
        for slot, q in zip(rb.arm.indices, arm_q):
            p.resetJointState(
                rb.uid, int(slot), targetValue=float(q), targetVelocity=0.0,
                physicsClientId=self.client,
            )
        rb._cmd_q = None
        rb.drive_arm_targets(arm_q)
        rb.last_target_pos = np.asarray(rb.ee_pose()[0], dtype=np.float64).copy()

    def _nut_drive_clean_macro(self) -> None:
        """Joint-space PREP→INSERT→HOLD→RETRACT inside the clean branch.

        * PREP    (stage -1): smooth lerp approach → clean staging (no teleport)
        * INSERT  (stage  0): lerp staging → seat
        * HOLD    (stage  1): freeze at seat
        * RETRACT (stage  2): lerp seat → staging

        All legs are env-scripted joint-space smoothsteps so B never hands
        control back to the Cartesian IK servo mid-macro (which would snap an
        isolated clean branch back into a tire-clipping natural branch).
        """
        stage_q = getattr(self, "_nut_clean_plunge_from", None)
        seat_q = getattr(self, "_nut_clean_plunge_to", None)
        if stage_q is None or seat_q is None:
            return
        stage_q = np.asarray(stage_q, dtype=np.float64)
        seat_q = np.asarray(seat_q, dtype=np.float64)
        stg = int(self._nut_macro_stage)
        if stg == -1:
            approach_q = getattr(self, "_nut_clean_approach_q", None)
            if approach_q is None:
                return
            leg_len = int(getattr(self.cfg, "nut_clean_prep_len", 30))
            approach_q = np.asarray(approach_q, dtype=np.float64)
            path = getattr(self, "_nut_clean_prep_path", None)
            if path:
                # Multi-segment smooth PREP: approach→W1→…→staging through the
                # planned collision-free waypoints (no tire-clip snap). Split the
                # leg into equal segments and smoothstep within each.
                nodes = [approach_q] + [np.asarray(w, dtype=np.float64)
                                        for w in path] + [stage_q]
                n_seg = len(nodes) - 1
                step = int(self._nut_macro_step)
                u = float(np.clip(step / max(1, leg_len), 0.0, 1.0)) * n_seg
                k = min(int(u), n_seg - 1)
                t = u - k
                s = t * t * (3.0 - 2.0 * t)
                self._nut_set_arm_q((1.0 - s) * nodes[k] + s * nodes[k + 1])
                return
            q_from = approach_q
            q_to = stage_q
        elif stg == 0:
            leg_len = int(getattr(self.cfg, "nut_clean_plunge_len", 25))
            q_from, q_to = stage_q, seat_q
        elif stg == 2:
            leg_len = int(getattr(self.cfg, "nut_clean_plunge_len", 25))
            stage_retract = np.asarray(
                getattr(self, "_nut_clean_stage_retract", stage_q),
                dtype=np.float64,
            )
            q_from, q_to = seat_q, stage_retract
        else:               # HOLD: hold the seat config
            self._nut_set_arm_q(seat_q)
            return
        t = float(np.clip(self._nut_macro_step / max(1, leg_len), 0.0, 1.0))
        s = t * t * (3.0 - 2.0 * t)
        self._nut_set_arm_q((1.0 - s) * q_from + s * q_to)

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
        # Nut-fastening task: Robot A is a static fixture and the gripper
        # channel is a no-op, so only the Robot-B Δpose block [6:12] is a
        # live control manifold. Zero the A slices in the L2 action/jerk
        # mask so PPO isn't penalised for residual noise on dead channels.
        if bool(getattr(self.cfg, "nut_fastening_task", False)) and dim == 13:
            m[0:6] = 0.0
            m[12] = 0.0
            # Coaxial-lock: B's tool orientation is env-controlled (fixed to the
            # bolt axis), so the rotation residual channels [9:12] are dead.
            if bool(getattr(self.cfg, "nut_b_lock_coaxial", True)):
                m[9:12] = 0.0
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

    def set_nut_b_hotstart_alpha(self, value: float) -> None:
        """Curriculum entry point for the Robot-B nut hot-start.

        ``alpha ∈ [0, 1]``: 1 = B starts at bolt 0's approach point (easy),
        0 = B starts at full HOME distance (hard). Broadcast each rollout
        boundary by ``NutHotStartCurriculumCallback``; read at reset by
        ``_apply_nut_b_hotstart``.
        """
        self.cfg.nut_b_hotstart_alpha = float(np.clip(value, 0.0, 1.0))

    def set_nut_arrive_ang_tol(self, value_rad: float) -> None:
        """Curriculum entry point for the arrive-alignment gate (rad).

        Broadcast each rollout boundary by ``NutArriveAngCurriculumCallback``
        as it ramps the gate from the loose start to the tight end; read by
        ``_advance_nut_fastening`` at the APPROACH→macro trigger.
        """
        self.cfg.nut_arrive_ang_tol_rad = float(max(1e-3, value_rad))

    def set_nut_arrive_pos_tol(self, value_m: float) -> None:
        """Curriculum entry point for the arrive-position capture radius (m).

        Broadcast each rollout boundary by ``NutArrivePosCurriculumCallback``
        as it ramps the staging capture sphere from a loose start to a tight
        end; read by ``_advance_nut_fastening`` at the APPROACH→insert trigger
        (``d_stage < nut_arrive_pos_tol``). Note this is the *axial* capture
        distance; the pure-RL coaxiality (lateral) gate is fixed by the seat
        physics (axis-only insert can't change lateral) and is NOT ramped.
        """
        self.cfg.nut_arrive_pos_tol = float(max(1e-3, value_m))

    def set_random_position_range(self, value_m: float) -> None:
        """Curriculum entry point for the DR hub-offset half-range (metres).

        Broadcast each rollout boundary by ``DRRangeCurriculumCallback`` as it
        ramps the offset from 0 up to the target (e.g. 0.05 m); read at reset by
        ``_maybe_apply_domain_randomization``. Enabling a non-zero range also
        flips the DR master switch on so the offset actually takes effect.
        """
        v = float(max(0.0, value_m))
        self.cfg.RANDOM_POSITION_RANGE = v
        if v > 0.0:
            self.cfg.USE_DOMAIN_RANDOMIZATION = True

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
        self._s1_grasp_ee_z = float(ur.ee_pose()[0][2])

    def _capture_s1_grasp_ee_z(self) -> None:
        """Record grasp-height EE Z for Stage-1 baked-replay gating."""
        try:
            ee, _ = self.robot_A.ee_pose()
            self._s1_grasp_ee_z = float(np.asarray(ee, dtype=np.float64)[2])
        except Exception:
            self._s1_grasp_ee_z = None

    def _stage1_force_baked(self) -> bool:
        """Hold baked replay through pickup lift + early carry preamble."""
        if int(self.task_stage) != 1:
            return False
        lift_from = int(getattr(self, "_carry_lift_from_idx", 0))
        extra = int(getattr(
            self.cfg, "planner_stage1_force_baked_extra_steps", 25,
        ))
        if lift_from > 0 and int(self.current_traj_step) < lift_from + extra:
            return True
        z_ref = getattr(self, "_s1_grasp_ee_z", None)
        if z_ref is None:
            return lift_from > 0 and int(self.current_traj_step) < lift_from
        z_min = max(
            float(getattr(self.cfg, "planner_carry_lift_skip_min_dz", 0.022)),
            self._stage1_pickup_lift_dz(),
        )
        try:
            ee, _ = self.robot_A.ee_pose()
            return float(ee[2]) < float(z_ref) + z_min
        except Exception:
            return True

    def _stage1_pickup_lift_dz(self) -> float:
        return float(getattr(self.cfg, "planner_stage1_pickup_lift_dz", 0.10))

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

    def _replace_grasped_tire_rigid(self) -> bool:
        """Hard re-place the grasped tire onto ``EE * T_ee_tire`` (visual).

        Uses the transform cached at grasp time (``_grasp_t_ee_tire_*``) to
        snap the tire back to its exact gripper-relative pose, cancelling the
        ±cm swing the soft JOINT_FIXED bond exhibits during fast carry
        acceleration. Velocity is zeroed so the next physics sub-step does
        not re-inject the lag. Returns ``True`` iff a re-place happened.
        """
        t_pos = getattr(self, "_grasp_t_ee_tire_pos", None)
        t_quat = getattr(self, "_grasp_t_ee_tire_quat", None)
        if t_pos is None or t_quat is None or self.robot_A is None:
            return False
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
        return True

    def _is_tire_grasped(self) -> bool:
        return (
            self._grasp_constraint is not None
            or self._grasp_kinematic
        )

    def _enforce_robot_a_palm_up(self) -> bool:
        """Post-step: keep the FANUC gripper palm-up (tool +Z = world +Z).

        Snaps the arm joints back onto the palm-up manifold (yaw free) when
        the heavy-tire load has drooped the stiff position PD off vertical,
        then re-places the grasped tire via the cached EE↔tire transform so
        the rigid ``JOINT_FIXED`` bond is not shocked. Gated by
        ``cfg.fanuc_enforce_palm_up_post_step``. No-op when the robot lacks
        the IK-based re-lock (e.g. Panda Robot B).

        Returns ``True`` iff a palm-up correction was applied. The correction
        is a ``resetJointState`` (no ``stepSimulation``), so the contact
        cache that ``getContactPoints`` reads is stale for the new arm pose;
        the caller refreshes it with ``performCollisionDetection`` when this
        returns ``True`` so the same-step collision/force gates see the
        corrected pose instead of lagging one step.
        """
        if not bool(getattr(self.cfg, "fanuc_enforce_palm_up_post_step", False)):
            return False
        if self.robot_A is None:
            return False
        enforce = getattr(self.robot_A, "enforce_palm_up_pose", None)
        if enforce is None:
            return False
        thr = float(getattr(self.cfg, "fanuc_palm_up_tool_z_threshold", 0.999))
        corrected = bool(enforce(thr))
        if not corrected:
            return False
        # The kinematic upright sync (stage 0) re-places the tire from the EE
        # itself, so only the JOINT_FIXED bond needs an explicit re-place.
        if not (self._grasp_kinematic or self._grasp_constraint is None):
            t_pos = getattr(self, "_grasp_t_ee_tire_pos", None)
            t_quat = getattr(self, "_grasp_t_ee_tire_quat", None)
            if t_pos is not None and t_quat is not None:
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
        return True

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
            tire_p, _ = self.scene.tire_pose()
            tire_p = np.asarray(tire_p, dtype=np.float64).reshape(3)
            end_pos = tire_p + np.array([0.0, 0.0, -R], dtype=np.float64)
            return end_pos, palm_up

        if stage == 1:
            # Mount end-pose: gripper UNDER the tire (6-o'clock), palm-up
            # (tool +Z ‖ world +Z) with a **−90° yaw about world +Z**.
            #
            # **2026-06-06 (yaw restored — fixes theta_A frozen at 90°).**
            # The tire is carried upright (``+Z``) and its yaw rigidly tracks
            # the gripper's yaw (true for BOTH the JOINT_FIXED grasp and the
            # kinematic upright lock, which both rotate the bore about world
            # +Z). The tire spawns with bore ‖ world +X; the hub axis is
            # world −Y (``hub_axis_world``). So aligning the bore onto the hub
            # needs a −90° world-Z yaw (+X → −Y). Commit 1074858 dropped this
            # yaw (believing 0° already gave bore ‖ hub per an old
            # diag_mount_yaw_palm.py reading), which left the bore stuck at +X
            # → ``theta_A = 90°`` every step → the mount gate (θ < δ_A) could
            # NEVER fire (training ran full 600-step episodes, success_rate 0).
            # A yaw about world +Z preserves tool +Z, so palm-up / upright is
            # unchanged — only the bore swings onto the hub axis.
            R = float(self.cfg.tire_outer_radius)
            hub_pos = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
            yaw_rot = axisangle3_to_quat(
                np.array([0.0, 0.0, -np.pi / 2.0], dtype=np.float64)
            )
            end_quat = np.asarray(quat_multiply(
                np.asarray(yaw_rot, dtype=np.float64),
                np.asarray(palm_up, dtype=np.float64),
            ), dtype=np.float64)
            # **2026-06-06 (seat-gap fix)** — derive the END position from the
            # actual cached grasp offset instead of the ``hub - R·ẑ``
            # heuristic. The heuristic assumed the tire centre sits exactly R
            # straight below the EE, but the real ``T_ee_tire`` offset differs
            # by ~3 cm, so the baked nominal left the tire ~3 cm short of the
            # hub even at zero residual (the leftover seat-glide slide). With
            # the grasp transform, the EE end pose places the grasped tire
            # centre exactly on ``tire_mount_pos`` (bore aligned by ``end_quat``
            # as before). Falls back to the heuristic if no grasp is cached.
            if have_grasp:
                rel_pos = np.asarray(self._grasp_t_ee_tire_pos, dtype=np.float64)
                world_off = Rotation.from_quat(end_quat).apply(rel_pos)
                end_pos = hub_pos - np.asarray(world_off, dtype=np.float64)
            else:
                end_pos = hub_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
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
            #   * Stage 1 — [start, +Z lift?, apex?, end+standoff, end]
            #   * Stage 3 — [start, start+standoff, apex?, end] (−Y extraction
            #     first, mirror of Stage 1)
            waypoints = [start_arr]
            if has_departure:
                ds = np.asarray(departure_standoff, dtype=np.float64).reshape(3)
                waypoints.append(start_arr + ds)
            if float(lift) > 1e-6:
                arch_from = np.asarray(waypoints[-1], dtype=np.float64)
                waypoints.append(
                    0.5 * (arch_from + end_arr)
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

    def _generate_stage0_pickup_trajectory(
        self,
        start_pose: Tuple[np.ndarray, np.ndarray],
        end_pose: Tuple[np.ndarray, np.ndarray],
        total_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 0 pickup: min-jerk XY with Z lowered ahead of horizontal travel."""
        start_pos, start_quat = start_pose
        end_pos, end_quat = end_pose
        n = int(total_steps)
        start_arr = np.asarray(start_pos, dtype=np.float64).reshape(3)
        end_arr = np.asarray(end_pos, dtype=np.float64).reshape(3)
        if bool(getattr(self.cfg, "planner_stage0_z_xy_coupled", True)):
            z_ahead = float(getattr(
                self.cfg, "planner_stage0_z_ahead_frac", 0.32,
            ))
            z_ahead = float(np.clip(z_ahead, 0.0, 0.75))
            if z_ahead <= 1e-6:
                traj_pos = _min_jerk_positions(start_arr, end_arr, n)
            else:
                xy0 = start_arr[:2]
                xy1 = end_arr[:2]
                z0 = float(start_arr[2])
                z1 = float(end_arr[2])
                xy_span = float(np.linalg.norm(xy1 - xy0))
                xy_traj = _min_jerk_positions(
                    np.array([xy0[0], xy0[1], 0.0], dtype=np.float64),
                    np.array([xy1[0], xy1[1], 0.0], dtype=np.float64),
                    n,
                )
                z_denom = max(1e-6, 1.0 - z_ahead)
                traj_pos = np.zeros((n, 3), dtype=np.float64)
                for i in range(n):
                    if xy_span > 1e-6:
                        frac_xy = float(
                            np.linalg.norm(xy_traj[i, :2] - xy0) / xy_span
                        )
                    else:
                        frac_xy = float(i) / max(n - 1, 1)
                    frac_xy = min(1.0, max(0.0, frac_xy))
                    z_frac = min(1.0, frac_xy / z_denom)
                    traj_pos[i, 0] = xy_traj[i, 0]
                    traj_pos[i, 1] = xy_traj[i, 1]
                    traj_pos[i, 2] = z0 + z_frac * (z1 - z0)
                traj_pos[-1] = end_arr
        else:
            hover_dz = float(getattr(self.cfg, "planner_stage0_hover_dz", 0.12))
            xy_frac = float(getattr(self.cfg, "planner_stage0_early_xy_frac", 0.30))
            z_frac = float(getattr(self.cfg, "planner_stage0_early_z_frac", 0.55))
            hover = end_arr + np.array([0.0, 0.0, hover_dz], dtype=np.float64)
            xy_frac = float(np.clip(xy_frac, 0.05, 0.95))
            z_frac = float(np.clip(z_frac, 0.05, 0.95))
            early = start_arr.copy()
            early[:2] = start_arr[:2] + xy_frac * (hover[:2] - start_arr[:2])
            early[2] = start_arr[2] + z_frac * (hover[2] - start_arr[2])
            traj_pos = _multi_min_jerk_positions(
                [start_arr, early, hover, end_arr], n,
            )
        traj_quat = _slerp_quats(start_quat, end_quat, n)
        return traj_pos, traj_quat

    def _stage0_live_grasp_pos(self) -> np.ndarray:
        """Live 6-o'clock outer grasp anchor for the cradle tire."""
        tire_pos, _ = self.scene.tire_pose()
        R = float(self.cfg.tire_outer_radius)
        return (
            np.asarray(tire_pos, dtype=np.float64).reshape(3)
            + np.array([0.0, 0.0, -R], dtype=np.float64)
        )

    def _apply_stage0_terminal_grasp_servo(self, quat: np.ndarray) -> None:
        """Kinematic descent to the live grasp anchor after traj playback."""
        grasp = self._stage0_live_grasp_pos()
        warm_q = getattr(self.robot_A, "_cmd_q", None)
        if warm_q is None:
            warm_q, _ = self.robot_A.joint_state()
        q_cmd = self._solve_robot_a_planner_q(grasp, quat, warm_q=warm_q)
        self.robot_A.last_target_pos = grasp.copy()
        self.robot_A.apply_kinematic_arm_targets(q_cmd)

    def _stage0_should_terminal_servo(self) -> bool:
        """True when the arm should kinematically descend to the grasp anchor."""
        if int(self.task_stage) != 0:
            return False
        if self._traj_pos is None:
            return False
        if not bool(getattr(self.cfg, "planner_stage0_terminal_grasp_servo", True)):
            return False
        n = int(self._traj_pos.shape[0])
        if int(self.current_traj_step) >= max(0, n - 1):
            return True
        if not bool(getattr(self.cfg, "planner_stage0_terminal_early_xy", False)):
            return False
        ee, _ = self.robot_A.ee_pose()
        grasp = self._stage0_live_grasp_pos()
        ee = np.asarray(ee, dtype=np.float64)
        dxy = float(np.linalg.norm(ee[:2] - grasp[:2]))
        dz = float(ee[2] - grasp[2])
        xy_tol = float(getattr(
            self.cfg, "planner_stage0_terminal_xy_tol", 0.18,
        ))
        z_tol = float(getattr(
            self.cfg, "planner_stage0_terminal_z_tol", 0.015,
        ))
        return dxy < xy_tol and dz > z_tol

    def _find_pickup_lift_traj_index(self, start_z: float) -> int:
        """First index whose nominal Z clears ``start_z + pickup_lift_dz``."""
        if self._traj_pos is None:
            return 0
        z_min = self._stage1_pickup_lift_dz()
        n = int(self._traj_pos.shape[0])
        for i in range(n):
            pos = np.asarray(self._traj_pos[i], dtype=np.float64)
            if float(pos[2]) >= float(start_z) + z_min - 1e-4:
                return i
        return max(0, n - 1)

    def _generate_stage1_carry_trajectory(
        self,
        start_pose: Tuple[np.ndarray, np.ndarray],
        end_pose: Tuple[np.ndarray, np.ndarray],
        total_steps: int,
        *,
        lift: float,
        orient_front_load_k: float,
        approach_standoff: Optional[np.ndarray],
        departure_standoff: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1 carry with vertical lift first, yaw rotation deferred."""
        start_pos, start_quat = start_pose
        end_pos, end_quat = end_pose
        traj_pos, _ = self._generate_nominal_trajectory(
            start_pose,
            end_pose,
            total_steps=total_steps,
            lift=lift,
            orient_front_load_k=0.0,
            approach_standoff=approach_standoff,
            departure_standoff=departure_standoff,
        )
        n = int(total_steps)
        start_z = float(np.asarray(start_pos, dtype=np.float64).reshape(3)[2])
        lift_end = self._find_pickup_lift_traj_index(start_z)
        lift_end = min(max(0, int(lift_end)), n - 1)
        traj_quat = np.repeat(
            np.asarray(start_quat, dtype=np.float64).reshape(1, 4),
            n,
            axis=0,
        )
        n_rot = n - lift_end - 1
        if n_rot > 0:
            t_lin = np.linspace(0.0, 1.0, n_rot, dtype=np.float64)
            k = float(orient_front_load_k)
            if k > 1.0 + 1e-6:
                t_lin = 1.0 - np.power(1.0 - t_lin, k)
            rot_quats = _slerp_quats(start_quat, end_quat, n_rot, times=t_lin)
            traj_quat[lift_end + 1:] = rot_quats
        traj_quat[-1] = np.asarray(end_quat, dtype=np.float64).reshape(4)
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
        """Departure via offset before the carry/return arch.

        Stage 1: straight +Z lift to clear the tire from the rack before any
        lateral hub motion. Stage 3 keeps ``None`` (extraction leg lives in
        Stage 2 demount).
        """
        st = int(stage)
        if bool(getattr(self.cfg, "remount_cycle_enable", False)):
            st = {3: 1, 4: 2, 5: 3}.get(st, st)
        if st == 1:
            dz = self._stage1_pickup_lift_dz()
            if dz > 1e-6:
                return np.array([0.0, 0.0, dz], dtype=np.float64)
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
        start_pose = (
            np.asarray(start_pos, dtype=np.float64),
            np.asarray(start_quat, dtype=np.float64),
        )
        end_pose = (
            np.asarray(end_pos, dtype=np.float64),
            np.asarray(end_quat, dtype=np.float64),
        )
        if (
            int(self.task_stage) == 0
            and (
                bool(getattr(self.cfg, "planner_stage0_z_xy_coupled", True))
                or float(getattr(self.cfg, "planner_stage0_hover_dz", 0.12)) > 1e-6
            )
        ):
            traj_pos, traj_quat = self._generate_stage0_pickup_trajectory(
                start_pose, end_pose, total_steps=n,
            )
        elif int(self.task_stage) == 1 and self._stage1_pickup_lift_dz() > 1e-6:
            traj_pos, traj_quat = self._generate_stage1_carry_trajectory(
                start_pose,
                end_pose,
                total_steps=n,
                lift=lift,
                orient_front_load_k=front_load_k,
                approach_standoff=standoff,
                departure_standoff=depart,
            )
        else:
            traj_pos, traj_quat = self._generate_nominal_trajectory(
                start_pose,
                end_pose,
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
            stage0_cart = (
                int(self.task_stage) == 0
                and bool(getattr(
                    self.cfg, "planner_stage0_cartesian_replay", True,
                ))
            )
            if stage0_cart:
                self._traj_q = None
            else:
                self._traj_q = self._bake_planner_joint_trajectory(
                    traj_pos, traj_quat,
                )
                self._traj_q = self._smooth_baked_joint_trajectory(self._traj_q)
            hold_q = getattr(self, "_planner_hold_arm_targets", None)
            if hold_q is not None and self._traj_q is not None:
                prefix = int(getattr(
                    self.cfg, "planner_hold_rebake_prefix", 20,
                ))
                if int(self.task_stage) == 1:
                    lift_idx = self._find_carry_lift_traj_index(
                        np.asarray(start_pos, dtype=np.float64),
                    )
                    prefix = max(prefix, int(lift_idx) + 1)
                self._traj_q = self._rebake_traj_prefix_from_joint_q(
                    np.asarray(hold_q, dtype=np.float64), prefix=prefix,
                )
            elif int(self.task_stage) == 0 and self._traj_q is not None:
                q0, _ = self.robot_A.joint_state()
                self._traj_q = self._rebake_traj_prefix_from_joint_q(
                    np.asarray(q0, dtype=np.float64),
                    prefix=int(self._traj_q.shape[0]),
                    pin_near_grasp=False,
                )
                # Re-solve the tail so the baked FK reaches the grasp anchor.
                tail = int(getattr(self.cfg, "planner_stage0_rebake_tail", 25))
                self._traj_q = self._rebake_traj_suffix_to_grasp(
                    tail=max(1, tail),
                )
        else:
            self._traj_q = None
        self._carry_lift_from_idx = 0
        if int(self.task_stage) == 1 and self._traj_pos is not None:
            lift_idx = self._find_carry_lift_traj_index(
                np.asarray(start_pos, dtype=np.float64),
            )
            self._carry_lift_from_idx = int(lift_idx)
            if bool(getattr(self.cfg, "planner_skip_s1_yaw_preamble", True)):
                self.current_traj_step = int(lift_idx)

    def _find_carry_lift_traj_index(
        self,
        grasp_ee: Optional[np.ndarray] = None,
    ) -> int:
        """First traj index whose nominal Z clears the grasp height + margin."""
        if self._traj_pos is None:
            return 0
        if grasp_ee is None:
            grasp_ee, _ = self.robot_A.ee_pose()
        grasp_ee = np.asarray(grasp_ee, dtype=np.float64).reshape(3)
        z_min = max(
            float(getattr(self.cfg, "planner_carry_lift_skip_min_dz", 0.022)),
            self._stage1_pickup_lift_dz(),
        )
        n = int(self._traj_pos.shape[0])
        for i in range(n):
            pos = np.asarray(self._traj_pos[i], dtype=np.float64)
            if float(pos[2]) >= float(grasp_ee[2]) + z_min:
                return i
        return max(0, n - 1)

    def _rebake_traj_prefix_from_joint_q(
        self,
        q0: np.ndarray,
        *,
        prefix: int = 20,
        pin_near_grasp: bool = True,
    ) -> np.ndarray:
        """Re-chain the early baked joints from the live grasp pose.

        ``_bake_planner_joint_trajectory`` warm-starts every waypoint from
        HOME, so ``_traj_q[0]`` can land on a different IK branch than the
        actual grasp joints even when the nominal EE pose matches. That
        mismatch makes the first few policy steps (and the GUI lerp) visibly
        bounce away from the tire before the carry arch begins.
        """
        if self._traj_q is None or self._traj_pos is None or self._traj_quat is None:
            return self._traj_q
        ur = self.robot_A
        n = int(self._traj_q.shape[0])
        end = min(max(1, int(prefix)), n)
        out = np.asarray(self._traj_q, dtype=np.float64).copy()
        q0 = np.clip(np.asarray(q0, dtype=np.float64).reshape(-1),
                     ur.arm.lower, ur.arm.upper)
        out[0] = q0
        palm_up = self._planner_palm_up_active()
        state_id = p.saveState(physicsClientId=self.client)
        try:
            for jidx, qv in zip(ur.arm.indices, q0):
                p.resetJointState(
                    ur.uid, jidx,
                    targetValue=float(qv), targetVelocity=0.0,
                    physicsClientId=self.client,
                )
            grasp_ee, grasp_quat = ur.ee_pose()
            grasp_ee = np.asarray(grasp_ee, dtype=np.float64)
            grasp_quat = np.asarray(grasp_quat, dtype=np.float64)
            self._traj_pos[0] = grasp_ee.copy()
            self._traj_quat[0] = grasp_quat.copy()
            pos_tol = float(getattr(
                self.cfg, "planner_hold_rebake_pos_tol", 0.045,
            ))
            z_tol = float(getattr(
                self.cfg, "planner_hold_rebake_z_tol", 0.035,
            ))
            rest = q0.copy()
            for i in range(1, end):
                pos = np.asarray(self._traj_pos[i], dtype=np.float64)
                near_grasp = False
                if pin_near_grasp:
                    near_grasp = (
                        float(np.linalg.norm(pos - grasp_ee)) < pos_tol
                        and float(pos[2]) < float(grasp_ee[2]) + z_tol
                    )
                if near_grasp:
                    out[i] = q0.copy()
                    continue
                quat = np.asarray(self._traj_quat[i], dtype=np.float64)
                if palm_up:
                    quat = self._tilt_lock_palm_up_quat(quat)
                q = ur.solve_arm_joints_in_snapshot(pos, quat, rest)
                out[i] = q
                rest = q.copy()
                for jidx, qv in zip(ur.arm.indices, q):
                    p.resetJointState(
                        ur.uid, jidx,
                        targetValue=float(qv), targetVelocity=0.0,
                        physicsClientId=self.client,
                    )
        finally:
            p.restoreState(stateId=state_id, physicsClientId=self.client)
            p.removeState(stateUniqueId=state_id, physicsClientId=self.client)
        return out

    def _solve_robot_a_planner_q(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        warm_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Chained IK for planner replay (Stage 0 cartesian path)."""
        ur = self.robot_A
        if warm_q is None:
            warm_q, _ = ur.joint_state()
        return ur.solve_arm_joints_in_snapshot(
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
            np.asarray(warm_q, dtype=np.float64),
        )

    def _rebake_traj_suffix_to_grasp(self, *, tail: int = 25) -> np.ndarray:
        """Re-chain the tail baked joints so FK reaches the live grasp pose."""
        if self._traj_q is None or self._traj_pos is None or self._traj_quat is None:
            return self._traj_q
        ur = self.robot_A
        n = int(self._traj_q.shape[0])
        tail_n = min(max(1, int(tail)), n)
        grasp = self._stage0_live_grasp_pos()
        self._traj_pos[-1] = grasp.copy()
        out = np.asarray(self._traj_q, dtype=np.float64).copy()
        palm_up = self._planner_palm_up_active()
        start_i = max(0, n - tail_n)
        state_id = p.saveState(physicsClientId=self.client)
        try:
            seed_i = max(0, start_i - 1)
            rest = out[seed_i].copy()
            for jidx, qv in zip(ur.arm.indices, rest):
                p.resetJointState(
                    ur.uid, int(jidx),
                    targetValue=float(qv), targetVelocity=0.0,
                    physicsClientId=self.client,
                )
            for i in range(start_i, n):
                pos = grasp if i == n - 1 else np.asarray(
                    self._traj_pos[i], dtype=np.float64,
                )
                quat = np.asarray(self._traj_quat[i], dtype=np.float64)
                if palm_up:
                    quat = self._tilt_lock_palm_up_quat(quat)
                q = ur.solve_arm_joints_in_snapshot(pos, quat, rest)
                out[i] = q
                rest = q.copy()
                for jidx, qv in zip(ur.arm.indices, q):
                    p.resetJointState(
                        ur.uid, int(jidx),
                        targetValue=float(qv), targetVelocity=0.0,
                        physicsClientId=self.client,
                    )
        finally:
            p.restoreState(stateId=state_id, physicsClientId=self.client)
            p.removeState(stateUniqueId=state_id, physicsClientId=self.client)
        return out

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

        override = getattr(self, "_dr_hub_xy_override", None)
        if override is not None:
            self._dr_hub_xy_offset = np.asarray(
                override, dtype=np.float64
            )[:2].copy()
        else:
            self._dr_hub_xy_offset = self._np_random.uniform(
                -rng, rng, size=2
            ).astype(np.float64, copy=False)
        # Cargo is perturbed independently of the hub only when enabled. The
        # nut-fastening DR fine-tune sets ``DR_CARGO_ENABLE = False`` so the
        # measured robustness is attributable purely to hub placement error.
        if bool(getattr(self.cfg, "DR_CARGO_ENABLE", True)):
            self._dr_cargo_xy_offset = self._np_random.uniform(
                -rng, rng, size=2
            ).astype(np.float64, copy=False)

    def set_dr_hub_xy_offset(
        self, offset_xy: Optional[np.ndarray],
    ) -> None:
        """Pin (or clear) the hub XY offset used on the next ``reset()``.

        Pass ``None`` to restore uniform sampling from
        ``cfg.RANDOM_POSITION_RANGE``. Passing a 2-vector enables DR and
        forces that exact hub translation every reset until cleared.
        """
        if offset_xy is None:
            self._dr_hub_xy_override = None
            return
        self._dr_hub_xy_override = np.asarray(
            offset_xy, dtype=np.float64
        )[:2].copy()
        self.cfg.USE_DOMAIN_RANDOMIZATION = True

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
    def _maybe_disable_tire_hub_collision(self) -> None:
        """Filter out tire↔hub collision when ``disable_tire_hub_collision``.

        The seated wheel disk (inner radius 0.10) geometrically overlaps the
        hub flange cylinder (radius 0.21) because the lug circle (0.1675) sits
        inside the flange footprint. Under the kinematic carry/seat this drove
        a ~700 kN per-step tire↔hub contact impulse (the visible "bouncing off
        the hub" jitter). The rigid ``_attach_tire_to_hub`` bond defines the
        final seated state and the vehicle wheel-well cutout + cargo back wall
        still constrain the gross approach, so the hub's collision against the
        tire is filtered for the whole episode. Covers every hub/truck link
        (flange base + bolt children).
        """
        if not bool(getattr(self.cfg, "disable_tire_hub_collision", False)):
            return
        if self.handles is None:
            return
        tire_uid = int(self.handles.tire)
        hub_uid = int(self.handles.hub.uid)
        try:
            n_links = p.getNumJoints(hub_uid, physicsClientId=self.client)
        except p.error:
            n_links = 0
        for link in range(-1, n_links):
            p.setCollisionFilterPair(
                tire_uid, hub_uid, -1, link, 0,
                physicsClientId=self.client,
            )

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

    def _seated_tire_orn(self) -> np.ndarray:
        """Seated bore-aligned orientation that PRESERVES the carried roll.

        **2026-06-06 (seat-spin fix)** — the carried tire (``_upright_tire_
        quat_for_ee`` = ``Rz(yaw)·Ry(90°)``) holds a stable roll about the hub
        axis (measured +180°), but the previous seated target
        (``_quat_align_z_to(hub_axis)`` = minimal ``Rx(90°)``) sat at −90°
        roll. Snapping/gliding to that target spun the tire ~90° about the hub
        axis as it seated — the visible "tire rotating about Y while entering
        the hub". This builds the seated orientation by applying only the
        *minimal tilt correction* that maps the tire's current bore exactly
        onto the hub axis, leaving the roll about that axis untouched. Since
        the bore is already aligned to <0.3° at the mount gate, the correction
        is tiny and seating becomes a pure axial insertion (no spin).
        """
        hub_axis = np.asarray(self.cfg.hub_axis_world, dtype=np.float64)
        hub_axis = hub_axis / max(float(np.linalg.norm(hub_axis)), 1e-9)
        _, cur_orn = self.scene.tire_pose()
        R = Rotation.from_quat(np.asarray(cur_orn, dtype=np.float64))
        bore = R.apply([0.0, 0.0, 1.0])
        bore = bore / max(float(np.linalg.norm(bore)), 1e-9)
        v = np.cross(bore, hub_axis)
        s = float(np.linalg.norm(v))
        c = float(np.dot(bore, hub_axis))
        if s < 1e-9:
            # Already (anti)parallel: identity for aligned, 180° about any
            # bore-perpendicular axis for the degenerate flip.
            if c > 0.0:
                return np.asarray(cur_orn, dtype=np.float64)
            perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(perp, hub_axis))) > 0.9:
                perp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            perp = perp - hub_axis * float(np.dot(perp, hub_axis))
            perp = perp / max(float(np.linalg.norm(perp)), 1e-9)
            delta = Rotation.from_rotvec(np.pi * perp)
        else:
            axis = v / s
            angle = float(np.arctan2(s, c))
            delta = Rotation.from_rotvec(angle * axis)
        seated = delta * R
        return np.asarray(seated.as_quat(), dtype=np.float64)

    def _attach_tire_to_hub(self) -> None:
        """Physically bond the seated tire to the hub (fixed constraint).

        Represents the tire being inserted onto the hub flange/pilot:
        the tire keeps its dynamic mass but is rigidly held by the (fixed-
        base) hub, so it stays put while Robot B tightens the bolts. The
        tire is first snapped to the exact seated pose (``tire_mount_pos``,
        bore ‖ ``hub_axis_world``, roll preserved from the carry) so the bond
        records a clean transform.
        """
        mount_pos = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
        hub_axis = np.asarray(self.cfg.hub_axis_world, dtype=np.float64)
        hub_axis = hub_axis / max(float(np.linalg.norm(hub_axis)), 1e-9)
        mount_orn = self._seated_tire_orn()
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

    def _begin_mount_seat_glide(self, steps: int) -> None:
        """Start a smooth kinematic slide of the tire onto the seated hub pose.

        Records the tire's current grasped pose at the mount-gate fire and the
        exact seated pose (``tire_mount_pos``, bore ‖ ``hub_axis_world``);
        ``_advance_mount_seat_glide`` then interpolates between them over
        ``steps`` env steps before ``_finalize_mount`` applies the bond. This
        replaces the legacy instant snap (which teleported the tire the full
        residual reach gap ~12-14 cm in one frame).
        """
        cur_pos, cur_orn = self.scene.tire_pose()
        mount_pos = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
        # Roll-preserving seated orientation (see ``_seated_tire_orn``): only
        # the tiny residual tilt is corrected, the carried roll about the hub
        # axis is kept. With the bore already aligned to <0.3° at the gate the
        # orientation barely changes, so the glide is a pure axial slide onto
        # the hub — the tire no longer spins about the hub axis while the
        # gripper holds still.
        mount_orn = self._seated_tire_orn()
        self._mount_seat_t0_pos = np.asarray(cur_pos, dtype=np.float64).copy()
        self._mount_seat_t0_orn = np.asarray(cur_orn, dtype=np.float64).copy()
        self._mount_seat_tgt_pos = mount_pos.copy()
        self._mount_seat_tgt_orn = np.asarray(mount_orn, dtype=np.float64).copy()
        self._mount_seat_total = int(max(1, steps))
        self._mount_seat_left = int(max(1, steps))
        self._mount_seat_active = True

    def _advance_mount_seat_glide(self) -> None:
        """Advance the in-progress mount-seat glide by one env step.

        Interpolates the tire pose with smoothstep easing (position lerp,
        normalized-lerp of the near-aligned bore quaternions) and clamps its
        velocity to zero so the slide reads as a clean insertion.
        """
        if not self._mount_seat_active:
            return
        self._mount_seat_left = max(0, int(self._mount_seat_left) - 1)
        done = self._mount_seat_total - self._mount_seat_left
        frac = float(np.clip(done / max(1, self._mount_seat_total), 0.0, 1.0))
        s = frac * frac * (3.0 - 2.0 * frac)
        pos = (1.0 - s) * self._mount_seat_t0_pos + s * self._mount_seat_tgt_pos
        q0 = self._mount_seat_t0_orn
        q1 = self._mount_seat_tgt_orn
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        q = (1.0 - s) * q0 + s * q1
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        p.resetBasePositionAndOrientation(
            self.handles.tire, pos.tolist(), q.tolist(),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.handles.tire, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )

    def _finalize_mount(self, remount: bool, events: Dict[str, Any]) -> None:
        """Commit the mount: bond/pin the tire and emit the ``mounted`` event.

        Called either immediately at the gate fire (legacy snap, when
        ``mount_seat_glide_steps == 0``) or after the seat glide completes.
        """
        self._mount_bonus_paid = True
        self.task_stage = 2
        self._mount_done_step = int(self._step_count)
        events["mounted"] = True
        self._mount_seat_active = False
        if remount:
            # 6-stage cycle: seat + bond the tire to the hub and freeze the
            # arm holding it while "Robot B tightens the nuts" (W1). After the
            # hold elapses (handled in the Stage-2 block) the grasp is released
            # and the empty gripper retracts to HOME.
            self._attach_tire_to_hub()
            self._grasp_kinematic = False
            self._mount_frozen_q, _ = self.robot_A.joint_state()
            self._mount_hold_left = int(getattr(self.cfg, "tighten_hold_steps", 0))
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
            d_grasp = float(np.linalg.norm(ee_pos - grasp_target))
            dz_above = float(ee_pos[2] - grasp_target[2])
            z_cap = float(getattr(
                self.cfg, "planner_stage0_pickup_max_dz", 0.030,
            ))
            if (
                d_grasp < float(self._approach_tol)
                and dz_above <= z_cap
            ):
                self._release_world_pin()
                self._attach_tire_to_robot_A()
                self.task_stage = 1
                self._pickup_bonus_paid = True
                events["picked_up"] = True
                hold_q, _ = self.robot_A.joint_state()
                self._planner_hold_arm_targets = np.asarray(
                    hold_q, dtype=np.float64,
                ).copy()
                self._capture_s1_grasp_ee_z()
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
            # 2026-06-06 — mount-seat glide. The arm reaches the stage-1 end
            # pose ~10 cm short of the hub, so finalizing the mount with an
            # instant ``resetBasePosition`` snap teleported the tire ~12-14 cm
            # onto the hub in a single frame. Instead, slide the tire smoothly
            # onto the seated pose over ``mount_seat_glide_steps`` env steps,
            # then bond and emit ``mounted`` (so any mount-terminate still
            # coincides with the tire actually being seated).
            if self._mount_seat_active:
                self._advance_mount_seat_glide()
                if self._mount_seat_left <= 0:
                    self._finalize_mount(remount, events)
            elif mounted and not self._mount_bonus_paid:
                glide_steps = int(getattr(self.cfg, "mount_seat_glide_steps", 0))
                if glide_steps > 0:
                    self._begin_mount_seat_glide(glide_steps)
                else:
                    self._finalize_mount(remount, events)

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
                # **2026-06-06 (kinematic demount carry)** — free the tire
                # from the hub bond, then re-grasp it KINEMATICALLY (not a
                # rigid JOINT_FIXED bond). A fixed bond forced the position
                # PD to dynamically drag the 100 kg tire off the hub against
                # the ~98 kN mount-seating penetration, which saturated the
                # arm and pinned it (demount never fired). The kinematic lock
                # teleports the tire to EE+offset each step (as S1–S3 do), so
                # S4/S5 carry the tire without the drag/jam. ``stage 4`` is in
                # ``kinematic_tire_lock_stages`` so the per-step sync runs.
                self._release_hub_mount()
                self._begin_kinematic_grasp()
                self._regrip_bonus_paid = True
                self.task_stage = 4
                events["regripped"] = True
                # Hold steady while "Robot B loosens the nuts" (W2). Keep the
                # kinematic grasp (do NOT promote to fixed) so the demount
                # pull-off below stays teleport-driven.
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
        nut_task = bool(getattr(self.cfg, "nut_fastening_task", False))
        # Action application runs all the nut-task IK / collision probes
        # (clean-branch planning, seat-branch search, macro target solves) that
        # teleport Robot B via resetJointState. Freeze the GUI for the whole
        # apply phase so none of those intermediate poses flicker on screen;
        # rendering is restored before the physics step so only the committed
        # motion is drawn (no effect in DIRECT/headless).
        with self._render_frozen():
            if self._mount_hold_left > 0 and self._mount_frozen_q is not None:
                self.robot_A.drive_arm_targets(self._mount_frozen_q)
            else:
                self._apply_action(action)

        # Visual-only carry rigidity: during carry (task_stage == 1) the tire
        # is held by a kinematic grasp (no physical constraint), so during the
        # decimation sub-steps it free-falls / drifts as a loose rigid body and
        # is only snapped back at the END of the step. The GUI renders every
        # sub-step, so that free body motion shows up as the tire "jittering /
        # moving on top of Robot A". When enabled, re-snap the tire onto the
        # cached EE↔tire transform after EVERY sub-step so it stays perfectly
        # rigid to the gripper on screen (no effect on headless physics result).
        carry_rigid = (
            bool(getattr(self.cfg, "carry_tire_rigid_sync", False))
            and not nut_task
            and int(self.task_stage) == 1
            and not self._mount_seat_active
            and self._is_tire_grasped()
        )
        # Zero the tire mass during rigid carry so ``stepSimulation`` never
        # moves it (no per-sub-step free-fall for the GUI to render); restore
        # the dynamic mass the moment rigid carry ends (mount-seat / Stage 2)
        # so the seating physics behave normally.
        if carry_rigid and not self._carry_mass_zeroed:
            try:
                p.changeDynamics(self.handles.tire, -1, mass=0.0,
                                 physicsClientId=self.client)
            except p.error:
                pass
            self._carry_mass_zeroed = True
        elif self._carry_mass_zeroed and not carry_rigid:
            try:
                p.changeDynamics(self.handles.tire, -1,
                                 mass=float(self.cfg.tire_mass),
                                 physicsClientId=self.client)
            except p.error:
                pass
            self._carry_mass_zeroed = False
        carry_rigid_synced = False
        if carry_rigid:
            # Snap before the first sub-step too, so the very first rendered
            # frame of the step already shows the tire rigid to the gripper.
            self._replace_grasped_tire_rigid()
        for _ in range(self.cfg.decimation):
            p.stepSimulation(physicsClientId=self.client)
            if carry_rigid:
                carry_rigid_synced = self._replace_grasped_tire_rigid()
        palm_up_corrected = False
        # Nut task: Robot A is a frozen fixture — no palm-up re-lock IK.
        if self._mount_hold_left <= 0 and not nut_task:
            palm_up_corrected = self._enforce_robot_a_palm_up()
            if carry_rigid and palm_up_corrected:
                # Palm-up re-lock moved the arm; re-snap the tire to match.
                carry_rigid_synced = self._replace_grasped_tire_rigid()
        # Non-rigid path keeps the legacy post-step kinematic upright sync.
        if (not carry_rigid and self._grasp_kinematic
                and self._use_kinematic_tire_sync()
                and not self._mount_seat_active):
            self._sync_grasped_tire_upright()
        # The palm-up re-lock / kinematic tire sync above move bodies via
        # resetJointState / resetBasePositionAndOrientation WITHOUT a physics
        # step, so the contact cache is stale for the new poses. Refresh it so
        # this step's collision/contact-force gates see the corrected world
        # (``getClosestPoints`` checks are already pose-current; this fixes the
        # ``getContactPoints``-based robot-link checks).
        if palm_up_corrected or carry_rigid_synced:
            p.performCollisionDetection(physicsClientId=self.client)

        self._step_count += 1
        # Nut task: re-assert the bonded tire's seated pose each step. The
        # JOINT_FIXED hub bond alone drifts a few mm under the nut-runner
        # tool contact, so clamp it back so it visibly stays mounted.
        if nut_task and self._mount_seated_pos is not None:
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
        if nut_task:
            fsm_events = self._advance_nut_fastening()
        else:
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
            not nut_task
            and self.task_stage == 0
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
        if nut_task:
            self._prev_axial_B = breakdown.nut_axial
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
        # 6-stage remount cycle uses its own stage set: S2 (empty retract to
        # HOME) is excluded so baking targets the true home orientation
        # (palm-up would land IK in a branch that stalls ~0.24 m short).
        if bool(getattr(self.cfg, "remount_cycle_enable", False)):
            allowed = tuple(
                int(x) for x in getattr(
                    self.cfg, "remount_planner_lock_palm_up_stages",
                    (0, 1, 3, 4, 5),
                )
            )
        else:
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
        # Nut-fastening task: Robot A is a static fixture (re-driven to its
        # cached HOME joint vector each step) and only Robot B is policy-
        # controlled via the ``action[6:12]`` Δpose block. Short-circuit the
        # whole planner/residual machinery — it is Robot-A-only.
        if bool(getattr(self.cfg, "nut_fastening_task", False)):
            if self._nut_frozen_qA is not None:
                if bool(getattr(self.cfg, "nut_a_kinematic_freeze", False)):
                    # v19 — A is a RIGID fixture: hard-reset its joints to the
                    # frozen hold pose every step so Robot-B contact can never
                    # push the arm (PD holding alone is compliant and visibly
                    # yields when B brushes it).
                    ra = self.robot_A
                    for s, q in zip(ra.arm.indices, self._nut_frozen_qA):
                        p.resetJointState(ra.uid, int(s), float(q),
                                          targetVelocity=0.0,
                                          physicsClientId=self.client)
                self.robot_A.drive_arm_targets(self._nut_frozen_qA)
            # MACRO leg (subphase 1): the env forces a coaxial insert→hold→
            # retract straight along the bolt axis; the policy is ignored.
            if (
                bool(getattr(self.cfg, "nut_scripted_macro", True))
                and int(self._nut_subphase) == 1
            ):
                self._drive_nut_macro()
                return
            # APPROACH leg (subphase 0): policy drives B toward staging.
            rb = self.robot_B
            if bool(getattr(self.cfg, "nut_b_planner_residual", False)):
                if self._nut_traj_q is None:
                    self._generate_nut_approach_traj()
                n = int(self._nut_traj_q.shape[0])
                t = min(int(self._nut_traj_step), n - 1)
                q_nom = np.asarray(self._nut_traj_q[t], dtype=np.float64)
                scale = float(getattr(
                    self.cfg, "nut_planner_pos_residual_scale", 0.05,
                ))
                residual = np.asarray(action[6:9], dtype=np.float64) * scale
                lock_quat = self._nut_lock_quat
                if lock_quat is None:
                    lock_quat = self._quat_align_tool_z(
                        -self._nut_axis_unit(int(self._nut_target_idx))
                    )
                if float(np.linalg.norm(residual)) > 1e-9:
                    for slot, q in zip(rb.arm.indices, q_nom):
                        p.resetJointState(
                            rb.uid, int(slot), targetValue=float(q),
                            targetVelocity=0.0, physicsClientId=self.client,
                        )
                    ee = np.asarray(rb.ee_pose()[0], dtype=np.float64)
                    rb.apply_absolute_ee(ee + residual, lock_quat)
                else:
                    q_nom = np.clip(q_nom, rb.arm.lower, rb.arm.upper)
                    for slot, q in zip(rb.arm.indices, q_nom):
                        p.resetJointState(
                            rb.uid, int(slot), targetValue=float(q),
                            targetVelocity=0.0, physicsClientId=self.client,
                        )
                    rb._cmd_q = None
                    rb.drive_arm_targets(q_nom)
                    rb.last_target_pos = rb.ee_pose()[0].copy()
                self._nut_traj_step += 1
            elif bool(getattr(self.cfg, "nut_b_lock_coaxial", True)):
                ps = float(getattr(self.cfg, "nut_pos_scale",
                                   self.cfg.action.pos_scale))
                # v22 — clean-branch macro: drive INSERT→HOLD→RETRACT entirely
                # in joint space inside the proven collision-free branch. The
                # Cartesian servo re-solves IK from the live joints and can snap
                # the (kinematically isolated) clean seat branch back to the
                # natural tire-clipping branch on HOLD/RETRACT (bolt 7: 17 cm
                # jump → nut_collision). Only fires once a clean branch was
                # actually committed at arrive (endpoints cached).
                if (
                    bool(getattr(self.cfg, "nut_b_clean_branch_insert", False))
                    and int(self._nut_subphase) == 1
                    and getattr(self, "_nut_clean_plunge_to", None) is not None
                ):
                    self._nut_drive_clean_macro()
                    return
                # v19 solo action: the whole action IS B's 3-d Δposition.
                a_off = 0 if int(self.cfg.action.dim) == 3 else 6
                d_pos_B = np.asarray(
                    action[a_off:a_off + 3], dtype=np.float64) * ps
                # Pure-RL insert/retract: plunge and retract ONLY along the bolt
                # axis (±Y in the hub frame). Approach/transit (subphase 0)
                # keeps full 3-DOF so B can move freely between bolts; once the
                # arrive gate fires the in/out is a coaxial slide the policy learns.
                if (
                    bool(getattr(self.cfg, "nut_pure_rl", False))
                    and int(self._nut_subphase) == 1
                ):
                    idx_t = int(self._nut_target_idx)
                    axis = self._nut_axis_unit(idx_t)
                    d_pos_B = float(np.dot(d_pos_B, axis)) * axis
                    # v19 align servo — the env zeroes the residual lateral
                    # offset (rate-limited) so the plunge is a geometrically
                    # exact on-axis slide; the policy keeps the axial DOF.
                    ee_now = np.asarray(rb.ee_pose()[0], dtype=np.float64)
                    bolt_pos = np.asarray(
                        self.scene.bolt_pose(idx_t)[0], dtype=np.float64)
                    # v20 — INSERT axial servo: drive the socket to the hub-
                    # face base so the nut runner fully envelops the stud.
                    if (
                        bool(getattr(self.cfg, "nut_b_axial_insert_servo", False))
                        and int(self._nut_macro_stage) == 0
                    ):
                        L_ax = float(getattr(self.cfg, "bolt_length", 0.10))
                        target_ax = -0.5 * L_ax
                        ax_now = float(np.dot(ee_now - bolt_pos, axis))
                        ax_err = target_ax - ax_now
                        ax_rate = float(getattr(
                            self.cfg, "nut_b_axial_insert_servo_rate", 0.008))
                        if ax_err < -1e-6:
                            ax_drive = -min(ax_rate, -ax_err)
                            d_pos_B = ax_drive * axis
                    # v19 align servo — zero lateral offset (rate-limited).
                    if bool(getattr(self.cfg, "nut_b_align_servo", False)):
                        v = ee_now - bolt_pos
                        lat_vec = v - float(np.dot(v, axis)) * axis
                        lat = float(np.linalg.norm(lat_vec))
                        rate = float(getattr(
                            self.cfg, "nut_b_align_servo_rate", 0.005))
                        if lat > 1e-6:
                            d_pos_B = d_pos_B - lat_vec * min(1.0, rate / lat)
                if rb.last_target_pos is None:
                    rb.last_target_pos = rb.ee_pose()[0].copy()
                rb.last_target_pos = rb.last_target_pos + d_pos_B
                # v19 — clamp the INSERT/RETRACT target onto the working band
                # of the bolt axis. Without this an over-plunging policy
                # accumulates an IK target metres past the hub-face base; the
                # arm physically wedges into the hub (collision => instant
                # fail under nut_collision_fail) even though the intended
                # motion was a few cm. Band: just past the seat depth to just
                # outside the staging point.
                if (
                    bool(getattr(self.cfg, "nut_pure_rl", False))
                    and int(self._nut_subphase) == 1
                    and bool(getattr(self.cfg, "nut_b_align_servo", False))
                ):
                    idx_t = int(self._nut_target_idx)
                    axis = self._nut_axis_unit(idx_t)
                    bolt_pos = np.asarray(
                        self.scene.bolt_pose(idx_t)[0], dtype=np.float64)
                    L = float(getattr(self.cfg, "bolt_length", 0.10))
                    v = rb.last_target_pos - bolt_pos
                    ax = float(np.dot(v, axis))
                    ax_cl = float(np.clip(
                        ax, -0.5 * L - 0.01,
                        self._nut_staging_axial() + 0.05,
                    ))
                    if ax_cl != ax:
                        rb.last_target_pos = (
                            rb.last_target_pos + (ax_cl - ax) * axis
                        )
                lock_quat = self._nut_lock_quat
                if lock_quat is None:
                    lock_quat = self._quat_align_tool_z(
                        -self._nut_axis_unit(int(self._nut_target_idx))
                    )
                rb.apply_absolute_ee(rb.last_target_pos, lock_quat)
            else:
                ps = float(getattr(self.cfg, "nut_pos_scale",
                                   self.cfg.action.pos_scale))
                rs = self.cfg.action.rot_scale
                d_pos_B = np.asarray(action[6:9], dtype=np.float64) * ps
                d_rot_B = np.asarray(action[9:12], dtype=np.float64) * rs
                self.robot_B.apply_delta_ee(d_pos_B, d_rot_B)
            return

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
            lift_from = int(getattr(self, "_carry_lift_from_idx", 0))
            force_baked = bool(getattr(
                self.cfg, "planner_force_baked_while_approaching", True,
            )) and (int(self.task_stage) == 0 or self._stage1_force_baked())
            use_dls = bool(getattr(self.cfg, "use_dls_cartesian_servo", False)) and (
                (residual_active and not force_baked)
                or bool(getattr(self.cfg, "planner_dls_always", False))
            )
            use_baked = (
                not use_dls
                and self._traj_q is not None
                and bool(getattr(self.cfg, "planner_precompute_joint_traj", True))
                and (not residual_active or force_baked)
            )
            stage0_kinematic = (
                force_baked
                and int(self.task_stage) == 0
                and bool(getattr(
                    self.cfg, "planner_stage0_kinematic_baked", True,
                ))
            )
            stage1_kinematic = (
                force_baked
                and int(self.task_stage) == 1
                and bool(getattr(
                    self.cfg, "planner_stage1_kinematic_until_lift", True,
                ))
                and int(self.current_traj_step) < int(
                    getattr(self, "_carry_lift_from_idx", 0)
                ) + int(getattr(
                    self.cfg, "planner_stage1_force_baked_extra_steps", 25,
                ))
            )
            stage0_cartesian = (
                force_baked
                and int(self.task_stage) == 0
                and bool(getattr(
                    self.cfg, "planner_stage0_cartesian_replay", True,
                ))
            )
            stage0_terminal = self._stage0_should_terminal_servo()
            if stage0_terminal:
                self._apply_stage0_terminal_grasp_servo(final_quat)
                at_tail = int(self.current_traj_step) >= n - 1
                if at_tail:
                    max_extra = int(getattr(
                        self.cfg, "planner_stage0_terminal_grasp_max_steps", 80,
                    ))
                    cap = max(0, n - 1) + max(1, max_extra)
                    if int(self.current_traj_step) < cap:
                        self.current_traj_step += 1
            elif stage0_cartesian:
                self.robot_A.last_target_pos = nominal_pos.copy()
                if bool(getattr(self.cfg, "planner_stage0_dls_replay", True)):
                    self.robot_A.drive_ee_servo_dls(
                        final_pos, final_quat,
                        damping=float(getattr(
                            self.cfg, "planner_dls_damping", 0.06)),
                        max_joint_step=float(getattr(
                            self.cfg, "planner_stage0_dls_max_joint_step",
                            0.14,
                        )),
                        pos_gain=float(getattr(
                            self.cfg, "planner_stage0_dls_pos_gain", 1.5,
                        )),
                        orn_gain=float(getattr(
                            self.cfg, "planner_dls_orn_gain", 0.8)),
                        adaptive=bool(getattr(
                            self.cfg, "planner_dls_adaptive", True)),
                        manip_threshold=float(getattr(
                            self.cfg, "planner_dls_manip_threshold", 0.02)),
                    )
                else:
                    warm_q = getattr(self.robot_A, "_cmd_q", None)
                    if warm_q is None:
                        warm_q, _ = self.robot_A.joint_state()
                    q_cmd = self._solve_robot_a_planner_q(
                        final_pos, final_quat, warm_q=warm_q,
                    )
                    self.robot_A.apply_kinematic_arm_targets(q_cmd)
                self._advance_traj_index(nominal_pos)
            elif use_dls:
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
                hold_q = getattr(self, "_planner_hold_arm_targets", None)
                lift_idx = int(getattr(self, "_carry_lift_from_idx", 0))
                skip_hold = (
                    hold_q is not None
                    and lift_idx > 0
                    and int(self.current_traj_step) == lift_idx
                )
                q_cmd = np.asarray(
                    hold_q if skip_hold else self._traj_q[idx],
                    dtype=np.float64,
                )
                self.robot_A.last_target_pos = nominal_pos.copy()
                if stage0_kinematic or stage1_kinematic:
                    self.robot_A.apply_kinematic_arm_targets(q_cmd)
                else:
                    self.robot_A.drive_arm_targets(q_cmd)
                if skip_hold:
                    self._planner_hold_arm_targets = None
                self._advance_traj_index(nominal_pos)
            else:
                hold_q = getattr(self, "_planner_hold_arm_targets", None)
                skip_ik = (
                    hold_q is not None
                    and (
                        int(self.current_traj_step) == 0
                        or (
                            int(getattr(self, "_carry_lift_from_idx", 0)) > 0
                            and int(self.current_traj_step)
                            == int(getattr(self, "_carry_lift_from_idx", 0))
                        )
                    )
                    and (
                        not residual_active
                        or bool(getattr(
                            self.cfg,
                            "planner_hold_skip_ik_ignore_residual",
                            True,
                        ))
                    )
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
        if bool(getattr(self.cfg, "nut_fastening_task", False)):
            parts.append(self._nut_obs_block(eeB_pos, ws))  # 7 (12 pure-RL) ← nut
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
    # Reward — Robot B sequential nut-fastening task
    # ------------------------------------------------------------------
    def _compute_nut_reward(self, action: np.ndarray, in_collision: bool,
                            out_of_workspace: bool,
                            fsm_events: Dict[str, Any],
                            b: rewards.RewardBreakdown,
                            ) -> Tuple[float, rewards.RewardBreakdown]:
        """Dense shaping for the APPROACH leg + sparse macro bonuses.

        The policy only controls APPROACH, so the dense terms shape that:

        * **APPROACH** (``_nut_subphase == 0``) — reach toward the bolt's
          *staging point* (on-axis, just past the tip) + a *coaxial*
          ``lateral`` kernel (arrive exactly along the axis) + an axial
          kernel pulling the socket onto the standoff band + alignment +
          PB shaping on Δ(distance-to-staging). All positive bounded ``exp``
          kernels so surviving a step is never punished — and ``reach_decay``
          is wide enough to give a gradient from the full HOME standoff.
        * **MACRO** (``_nut_subphase == 1``) — the env forces the coaxial
          insert→hold→retract, so dense shaping is moot: only the safety
          penalties stay on (action/jerk are meaningless here since the
          policy is ignored, so they are dropped).

        Sparse: ``R_arrive`` (socket parked at staging → macro triggered),
        ``R_insert`` (seat dwell done), ``R_fasten`` (bolt cleared),
        ``R_all_fastened`` (episode success).
        """
        rcfg = self.cfg.reward
        idx = int(self._nut_target_idx)
        _, theta_B = self._nut_gate_metrics(idx)
        axial, lateral, theta = self._nut_axial_lateral(idx)
        staging_axial = self._nut_staging_axial()
        staging_pos = np.asarray(
            self._nut_point_on_axis(idx, staging_axial), dtype=np.float64,
        )
        ee_pos = np.asarray(self.robot_B.ee_pose()[0], dtype=np.float64)
        # v13 — Euclidean tool→staging distance (PB potential; no decay).
        d_stage = float(np.linalg.norm(ee_pos - staging_pos))
        b.d_B = float(d_stage)
        b.theta_B = float(theta_B)
        b.nut_target_idx = idx
        b.n_fastened = len(self._nut_fastened)
        b.n_fastened_policy = len(self._nut_fastened) - int(
            getattr(self, "_nut_premark", 0)
        )
        b.nut_lateral = float(lateral)
        b.nut_axial = float(axial)
        b.nut_subphase = int(self._nut_subphase)

        macro = int(self._nut_subphase) == 1

        if not macro:
            # v13 APPROACH — farm-proof: PB progress only (+ penalties).
            # Standing exp kernels (reach/lateral/axial/align/path) removed —
            # they were farmed repeatedly (align→path→lateral whack-a-mole).
            b.nut_align = 0.0
            b.nut_lateral_term = 0.0
            b.nut_reach = 0.0
            b.nut_axial_term = 0.0
            b.nut_path = 0.0

            wpb = float(rcfg.w_pb_nut)
            pb_nut = 0.0
            progress = 0.0
            if wpb > 0.0 and self._prev_d_B is not None:
                progress = float(self._prev_d_B - d_stage)
                pb_nut = wpb * progress
            b.pb_nut = float(pb_nut)
            self._prev_d_B = float(d_stage)

            # One-sided corridor: penalise hub-side (+Y) excursions past staging.
            w_corr = float(getattr(rcfg, "w_nut_corridor", 0.0))
            plane_y = float(staging_pos[1])
            margin = float(getattr(rcfg, "nut_corridor_margin", 0.02))
            y_excursion = max(0.0, float(ee_pos[1]) - (plane_y + margin))
            b.nut_path_dev = float(y_excursion)
            b.nut_path = -w_corr * y_excursion if w_corr > 0.0 else 0.0

            # v19 — wasted-motion cost: every metre of EE travel that does not
            # close distance to the staging target is paid for. PB telescopes
            # (path-independent), so this term is what makes the MINIMAL
            # (straight) transit the optimum instead of any wandering path.
            w_waste = float(getattr(rcfg, "w_nut_path_waste", 0.0))
            prev_ee = getattr(self, "_nut_prev_ee", None)
            if w_waste > 0.0 and prev_ee is not None:
                moved = float(np.linalg.norm(ee_pos - prev_ee))
                waste = max(0.0, moved - max(0.0, progress))
                b.nut_path += -w_waste * waste
            self._nut_prev_ee = ee_pos.copy()

            w_jv = float(getattr(rcfg, "w_nut_joint_vel", 0.0))
            if w_jv > 0.0:
                _, dqB = self.robot_B.joint_state()
                b.nut_joint_vel = -w_jv * float(
                    np.linalg.norm(np.asarray(dqB, dtype=np.float64))
                )
        elif bool(getattr(self.cfg, "nut_pure_rl", False)):
            # PURE-RL INSERT/HOLD/RETRACT — the policy drives the in/out itself,
            # so shape it with a per-leg axial potential (drive the socket to the
            # current stage target depth along the bolt axis) plus a coaxiality
            # cost. Both are non-farmable: the PB telescopes (net progress only)
            # and the lateral term is a pure negative penalty.
            b.nut_align = 0.0
            b.nut_reach = 0.0
            b.nut_axial_term = 0.0
            b.nut_path = 0.0
            b.nut_path_dev = 0.0
            b.nut_joint_vel = 0.0
            self._prev_d_B = None

            # v20 — when the INSERT axial servo is driving the plunge (macro
            # stage 0), the env (not the policy) owns the axial DOF, so the
            # axial-PB reward and the joint-velocity penalty would be paid for
            # motion the policy did not command. Gate them off in that window
            # (same reasoning the scripted macro uses) and keep the PB baseline
            # frozen so RETRACT doesn't book a spurious one-step jump.
            insert_servo = (
                bool(getattr(self.cfg, "nut_b_axial_insert_servo", False))
                and int(self._nut_macro_stage) == 0
            )

            if insert_servo:
                b.pb_nut = 0.0
            else:
                tgt_axial = self._nut_stage_target_axial()
                axial_err = abs(float(axial) - float(tgt_axial))
                wpb_ax = float(getattr(rcfg, "w_nut_pb_axial", 0.0))
                pb_ax = 0.0
                if wpb_ax > 0.0 and self._prev_axial_err_B is not None:
                    pb_ax = wpb_ax * float(self._prev_axial_err_B - axial_err)
                b.pb_nut = float(pb_ax)
                self._prev_axial_err_B = float(axial_err)

            w_lat = float(getattr(rcfg, "w_nut_lateral_pen", 0.0))
            b.nut_lateral_term = -w_lat * float(lateral) if w_lat > 0.0 else 0.0

            # v20 — joint-movement penalty in policy-driven HOLD/RETRACT (not
            # the servo-driven INSERT plunge). Shapes smooth motion without
            # capping joint velocity.
            w_jv = float(getattr(rcfg, "w_nut_joint_vel", 0.0))
            if w_jv > 0.0 and not insert_servo:
                _, dqB = self.robot_B.joint_state()
                b.nut_joint_vel = -w_jv * float(
                    np.linalg.norm(np.asarray(dqB, dtype=np.float64))
                )
        else:
            # MACRO — env-driven; no policy-shaping dense terms.
            b.nut_align = 0.0
            b.nut_lateral_term = 0.0
            b.nut_reach = 0.0
            b.nut_axial_term = 0.0
            b.pb_nut = 0.0
            b.nut_path = 0.0
            b.nut_path_dev = 0.0
            b.nut_joint_vel = 0.0
            self._prev_d_B = None

        # v19 — stalled-progress bookkeeping for the early-truncation gate.
        # Tracks the best value of the phase's own progress metric (approach:
        # distance-to-staging; insert/retract: axial error to the stage
        # target); any phase/bolt change resets the window.
        ns = int(getattr(self.cfg, "nut_stall_steps", 0))
        if ns > 0:
            key = (int(self._nut_subphase), idx, int(self._nut_macro_stage))
            metric = d_stage if not macro else abs(
                float(axial) - float(self._nut_stage_target_axial()))
            eps = float(getattr(self.cfg, "nut_stall_eps", 0.001))
            best = getattr(self, "_nut_prog_best", None)
            if (getattr(self, "_nut_stall_key", None) != key
                    or best is None or metric < best - eps):
                self._nut_stall_key = key
                self._nut_prog_best = float(metric)
                self._nut_stall_count = 0
            else:
                self._nut_stall_count = int(
                    getattr(self, "_nut_stall_count", 0)) + 1

        # Sparse FSM bonuses.
        if fsm_events.get("arrived"):
            b.fsm_bonus += float(getattr(rcfg, "R_arrive", 25.0))
        if fsm_events.get("inserted"):
            b.fsm_bonus += float(getattr(rcfg, "R_insert", 30.0))
        if fsm_events.get("fastened"):
            b.fsm_bonus += float(rcfg.R_fasten)
        if fsm_events.get("all_fastened"):
            b.fsm_bonus += float(rcfg.R_all_fastened)
            b.is_success = True

        # B↔A clearance shaping (both phases): a saturating positive bonus for
        # keeping Robot B's arm away from Robot A. Lets the policy discover a
        # collision-free fastening configuration on its own (replaces the
        # forced "arm-up" IK branch). Uses joint-center (skeleton) separation,
        # normalised between a floor (≈ joint-center distance at hard contact)
        # and a cap (well-separated): 0 at/below floor, 1 at/above cap.
        #
        # 2026-06-08 — ENGAGEMENT GATE. The ungated bonus was farmable: after the
        # hot-started bolt-0 freebie, the policy discovered that fleeing the hub
        # (B↔A dist ≈ 1.1 m, bonus saturated) paid +0.6/step forever, which beat
        # the hard, locally-dead approach gradient toward the next bolt (at
        # lateral ≈ 1 m both the lateral kernel and the coax-gated reach are ~0,
        # so nothing pulled B back in). Result: bolt 0 fastened, then camp →
        # n_fastened stuck at 1. The clearance reward is only meaningful WHILE B
        # is actually at the bolt working, so gate it by proximity to the target
        # staging point: ``exp(-d_engage / scale)``. Camping far away ⇒ gate ≈ 0
        # ⇒ no farmable bonus; at/near staging ⇒ gate ≈ 1 ⇒ full "keep the arm
        # clear of A while fastening" incentive (the user's actual intent).
        floor = float(getattr(rcfg, "nut_ba_clear_floor", 0.30))
        cap = float(getattr(rcfg, "nut_ba_clear_cap", 0.60))
        span = max(cap - floor, 1e-3)
        w_ba = float(getattr(rcfg, "w_nut_ba_clear", 0.0))
        d_ba = self._nut_ba_clearance()
        b.nut_ba_dist = float(d_ba)
        engage_scale = max(
            float(getattr(rcfg, "nut_ba_clear_engage_scale", 0.35)), 1e-3,
        )
        # True (unweighted) Euclidean tool→staging distance; 0 at the work point.
        d_engage = float(math.hypot(axial - staging_axial, lateral))
        engage = float(np.exp(-d_engage / engage_scale))
        b.nut_ba_clear = (
            w_ba * float(np.clip((d_ba - floor) / span, 0.0, 1.0)) * engage
        )

        # Safety penalties stay on in both phases. Action/jerk L2 only makes
        # sense while the policy controls B (APPROACH); during the forced
        # macro the policy is ignored, so penalising its (dead) outputs would
        # inject spurious negative reward across the whole tighten window.
        # Use the stronger dedicated nut collision penalty (the shared
        # w_collision was too weak to dominate the dense reach reward).
        w_nut_col = float(getattr(rcfg, "w_nut_collision",
                                  float(rcfg.w_collision)))
        b.collision = -w_nut_col if in_collision else 0.0
        b.workspace = rewards.workspace_penalty(out_of_workspace, rcfg)
        # Action/jerk L2 only makes sense while the POLICY controls B. That's
        # the APPROACH always, and (pure-RL only) the insert/retract too. Under
        # the scripted macro the policy is ignored, so penalising its dead
        # outputs would inject spurious cost across the whole tighten window.
        policy_active = (not macro) or bool(getattr(self.cfg, "nut_pure_rl", False))
        # v20 — the INSERT axial servo overrides the policy action during the
        # plunge (macro stage 0), so its action/jerk outputs are dead there too.
        if (
            macro
            and bool(getattr(self.cfg, "nut_pure_rl", False))
            and bool(getattr(self.cfg, "nut_b_axial_insert_servo", False))
            and int(self._nut_macro_stage) == 0
        ):
            policy_active = False
        if policy_active:
            action_mask = self._build_action_mask()
            b.action = rewards.action_penalty(action, rcfg, mask=action_mask)
            b.jerk = rewards.jerk_penalty(
                action, self._prev_action, rcfg, mask=action_mask,
            )
        else:
            b.action = 0.0
            b.jerk = 0.0

        dense = (
            b.nut_reach + b.nut_align + b.nut_lateral_term + b.nut_axial_term
            + b.pb_nut + b.nut_ba_clear + b.collision + b.workspace
            + b.action + b.jerk + b.nut_path + b.nut_joint_vel
        )
        b.dense_total_pre_mix = float(dense)
        b.step_alive = -float(getattr(rcfg, "w_step_alive", 0.0))
        b.total = float(
            rcfg.mix_dense * b.dense_total_pre_mix
            + rcfg.mix_sparse_success * float(b.fsm_bonus)
            + b.step_alive
        )
        return b.total, b

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

        # Nut-fastening task uses a dedicated Robot-B reward branch (the
        # Robot-A stage-dense dispatch below does not apply — A is frozen).
        if bool(getattr(self.cfg, "nut_fastening_task", False)):
            return self._compute_nut_reward(
                action, in_collision, out_of_workspace, fsm_events, b,
            )

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
        # **2026-06-06 (remount-cycle dense dispatch)** — the legacy dict
        # only keyed 0..3, so the 6-stage cycle (a) mis-read S2/S3 (they
        # compute ``return_A``/``approach_A`` above, but the legacy dict
        # pulled ``demount``/``return_A`` for keys 2/3) and (b) raised
        # ``KeyError`` the instant the FSM reached S4/S5. Map each remount
        # stage to the term its upstream branch actually populated:
        #   S0 approach_A · S1 guide+pb_carry · S2 return_A (retract→HOME)
        #   S3 approach_A (regrip) · S4 demount+pb_carry · S5 return+landing
        if remount:
            stage_dense_A = {
                0: float(b.approach_A),
                1: float(b.guide_A) + float(b.pb_carry),
                2: float(b.return_A),
                3: float(b.approach_A),
                4: float(b.demount) + float(b.pb_carry),
                5: float(b.return_A) + float(b.landing),
            }[ts]
        else:
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
        # **2026-06-06 (tire-vs-robotB)** — the carried tire must not clip
        # Robot B (UR10e + 30 cm nut-runner). B is frozen at HOME in Phase 1
        # but its tool reaches toward the hub (+Y), and the ±0.20 m policy
        # residual can push the carried tire toward it. Like the cargo
        # check above this uses ``getClosestPoints`` so it is correct under
        # the kinematic upright lock (which teleports the tire post-step).
        # tire-vs-robotA is intentionally NOT checked here — that is the
        # gripper grasp contact.
        if self.robot_B is not None:
            cps_b = p.getClosestPoints(
                bodyA=tire_uid, bodyB=self.robot_B.uid, distance=0.0,
                physicsClientId=self.client,
            )
            for cp in cps_b:
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
            #
            # **2026-06-06 (remount-cycle)** — also mask while the tire is
            # bonded to the hub (``_hub_mount_constraint``). On mount the
            # grasp is released but the tire is seated + JOINT_FIXED-bonded
            # to the wheel-station flange; the rigid bond (erp=1, maxForce=
            # 1e6) reports a spurious ~98 kN tire↔wheel_station penetration
            # reaction every step — a seating artifact, not policy damage.
            # Without this mask the 50 kN gate killed *every* Phase B episode
            # one step after mount (stage 1→2), so the policy never collected
            # any tighten-hold / retract / regrip / demount / return data.
            tire_held = self._is_tire_grasped() or (
                self._hub_mount_constraint is not None
            )
            if tire_held:
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
        if bool(getattr(self.cfg, "nut_fastening_task", False)):
            # v19 — hard process rule: Robot B colliding with the fixture
            # (Robot A / floor / walls — whatever _in_bad_collision flags)
            # fails the cycle immediately. Checked FIRST so a same-step
            # fasten cannot mask the violation.
            if bool(getattr(self.cfg, "nut_collision_fail", False)) \
                    and in_collision:
                b.is_success = False
                info["is_success"] = False
                return True, False, {**info, "termination": "nut_collision"}
            # v19 — stalled-progress early truncation (saves the ~800-step
            # horizon burn of episodes parked with no approach/insert
            # progress). Truncation (not termination) so the value bootstrap
            # stays unbiased.
            ns = int(getattr(self.cfg, "nut_stall_steps", 0))
            if ns > 0 and int(getattr(self, "_nut_stall_count", 0)) >= ns:
                info["is_success"] = False
                return False, True, {**info, "termination": "nut_stall"}
        # Nut-fastening task — success once every bolt is fastened.
        if fsm_events.get("all_fastened"):
            b.is_success = True
            info["is_success"] = True
            return True, False, {**info, "termination": "all_fastened"}
        # Pure-RL per-leg curriculum: one bolt per episode. Terminate on the
        # first policy-driven fasten so the horizon is spent on approach+insert
        # for the hot-started bolt, not on failed transit to the next bolt.
        if (
            bool(getattr(self.cfg, "nut_per_leg_episode", False))
            and fsm_events.get("fastened")
        ):
            b.is_success = True
            info["is_success"] = True
            return True, False, {**info, "termination": "nut_per_leg_fasten"}
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
