"""TyroEnv — Gymnasium env implementing the Phase 1 FSM cycle.

Phase 1 task: Robot A (UR10) picks up a tire from a floor pickup zone next to
its base, transports it to the truck hub, mounts it coaxially, then returns
the tire back to the floor pickup zone for a soft landing. Robot B (Panda)
concurrently reaches and aligns its tool +Z with the target bolt.

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

from ..config import EnvConfig
from . import rewards
from .robots import PandaRobot, UR10Robot
from .scene import Scene, SceneHandles
from .utils import quat_axis, relative_axisangle


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
        self.robot_A: Optional[UR10Robot] = None
        self.robot_B: Optional[PandaRobot] = None
        # Tire ↔ UR10 EE fixed joint (Stage 1/2). Recreated on each pickup.
        self._grasp_constraint: Optional[int] = None
        # Tire ↔ world fixed joint (Stage 0 and after final landing). Keeps
        # the standing-on-edge tire from tipping over when not grasped.
        self._world_pin: Optional[int] = None
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
            p.disconnect(physicsClientId=self.client)
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
        self._world_pin = None
        self.task_stage = 0
        self._mount_bonus_paid = False
        self._pickup_bonus_paid = False
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

        self.robot_A = UR10Robot(self.client, self.cfg)
        self.robot_B = PandaRobot(self.client, self.cfg)

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

        # Pin the tire to the floor pickup pose so it stands upright on its
        # tread edge without toppling over. This is removed at Stage 0 → 1
        # grasp and re-engaged at the final landing.
        self._pin_tire_to_world(self._pickup_pos_world, self._vertical_quat)

        self._step_count = 0
        self._prev_action = np.zeros(self.cfg.action.dim, dtype=np.float32)
        self._prev_d_A = None
        self._prev_d_B = None
        self._prev_d_approach = None
        self._prev_d_return = None

        obs = self._compute_obs()
        info = {"target_bolt_idx": self.handles.target_bolt_idx}
        return obs, info

    # ------------------------------------------------------------------
    # Action / obs masks (Phase 1 feature isolation)
    # ------------------------------------------------------------------
    def _build_action_mask(self) -> np.ndarray:
        """Cache an action mask sized to the active ``action_space``.

        Two regimes:

        * Phase 1 — ``freeze_robot_b=True`` ⇒ ``action.dim == 7``.
          The Panda block has been *sliced out* of the action space, so
          the mask is simply ``ones(7)``. The gripper_A channel sits at
          ``action[6]``.

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
        self._action_mask_cache = m
        return m

    def _build_obs_mask(self) -> np.ndarray:
        """Cache an obs mask that zeros Phase-1-irrelevant Panda channels.

        Layout follows ``_compute_obs``'s concatenation order. The first
        73 entries and the trailing 3 ``mount_tail`` scalars are dim-
        independent; only the ``prev_action`` slice between them changes
        length with the action space (7 in Phase 1, 13 in Phase 2/3).

          [0:6]    qA_n            — UR10 joints
          [6:12]   dqA_n           — UR10 joint vels
          [12:19]  qB_n (7)        — Panda joints      ← zero in Phase 1
          [19:26]  dqB_n (7)       — Panda joint vels  ← zero in Phase 1
          [26:33]  eeA_pos_rel/orn — UR10 EE
          [33:40]  eeB_pos_rel/orn — Panda EE          ← zero in Phase 1
          [40:47]  tire_pos_rel/orn
          [47:54]  hub_pos_rel/orn
          [54:61]  bolt_pos_rel/orn                    ← zero in Phase 1
          [61:67]  Δtire-hub (pos 3 + axisangle 3)
          [67:73]  ΔeeB-bolt                           ← zero in Phase 1
          [73 : 73+A]   prev_action (A = action.dim)
          [73+A : 76+A] mount_tail (axial/lateral/lug) ← KEEP (mount FSM)

        ``prev_action`` is *not* masked here — in Phase 1 its Panda slice
        no longer exists (action is already 7-d), and in Phase 2/3 those
        channels are part of the real control output.
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

        # 2. Tire COM exactly one outer radius above the gripper in world +Z,
        #    so the 6 o'clock tread point sits on the EE (R-margin).
        tire_pos = ee_pos + np.array([0.0, 0.0, R], dtype=np.float64)

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
        # 5. Stiffen the fixed joint so tire cannot sag under its own
        #    weight (default maxForce is finite in Bullet; raising it
        #    several orders of magnitude removes the visible droop).
        p.changeConstraint(
            self._grasp_constraint,
            maxForce=1.0e6,
            physicsClientId=self.client,
        )

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

    def _release_grasp(self) -> None:
        if self._grasp_constraint is not None:
            try:
                p.removeConstraint(
                    self._grasp_constraint, physicsClientId=self.client,
                )
            except p.error:
                pass
            self._grasp_constraint = None

    def _try_stage_transitions(self) -> Dict[str, Any]:
        """Drive task_stage 0 → 1 → 2 → done. Returns a dict of FSM events.

        Events emitted in ``info`` keys:
          * ``picked_up``  — Stage 0 → 1 (grasp constraint born)
          * ``mounted``    — Stage 1 → 2 (tire seated on hub)
          * ``landed``     — Stage 2 → done success (tire back on floor)
        """
        events: Dict[str, Any] = {
            "picked_up": False, "mounted": False, "landed": False,
        }
        ee_pos, _ = self.robot_A.ee_pose()
        tire_pos, _ = self.scene.tire_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        tire_pos = np.asarray(tire_pos, dtype=np.float64)
        R = float(self.cfg.tire_outer_radius)

        if self.task_stage == 0:
            # Trigger: EE within ``approach_radius_tol`` of the tire's 6 o'clock
            # outer point (world = tire COM + (0, 0, -R) for a vertical tire).
            grasp_target = tire_pos + np.array([0.0, 0.0, -R], dtype=np.float64)
            if float(np.linalg.norm(ee_pos - grasp_target)) < float(self._approach_tol):
                self._release_world_pin()
                self._attach_tire_to_robot_A()
                self.task_stage = 1
                self._pickup_bonus_paid = True
                events["picked_up"] = True
                # New stage → reset PB shaping baseline so the first
                # post-transition step does not pay a phantom Δd bonus.
                self._prev_d_approach = None
                self._prev_d_return = None

        elif self.task_stage == 1:
            hub_pos, _ = self.scene.hub_pose()
            mount_target = np.asarray(self.cfg.tire_mount_pos, dtype=np.float64)
            # Mount completion uses the explicit mount target (world coord).
            # Hub origin and mount target coincide under hub-centric layout.
            d_mount = float(np.linalg.norm(tire_pos - mount_target))
            tire_axis = self.scene.tire_axis()
            hub_axis = self.scene.hub_axis()
            theta = float(np.arccos(
                np.clip(np.dot(tire_axis, hub_axis), -1.0, 1.0)
            ))
            mounted = (
                d_mount < self.cfg.mount_radius_tol
                and theta < self.cfg.reward.delta_A
            )
            if mounted and not self._mount_bonus_paid:
                self._mount_bonus_paid = True
                self.task_stage = 2
                events["mounted"] = True
                self._prev_d_approach = None
                self._prev_d_return = None

        elif self.task_stage == 2:
            d_return = float(np.linalg.norm(tire_pos - self._pickup_pos_world))
            # Soft-landing check: tire descent speed (linear velocity Z).
            lin_vel, _ = p.getBaseVelocity(
                self.handles.tire, physicsClientId=self.client,
            )
            descend_speed = abs(float(lin_vel[2]))
            landed = (
                d_return < self.cfg.return_radius_tol
                and descend_speed < self.cfg.landing_speed_max
            )
            if landed:
                # Release grasp and pin to world at the tire's CURRENT pose so
                # there is no visible teleport. Reset velocities to zero.
                self._release_grasp()
                _, cur_orn = self.scene.tire_pose()
                self._pin_tire_to_world(tire_pos, cur_orn)
                events["landed"] = True

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
        self._apply_action(action)

        for _ in range(self.cfg.decimation):
            p.stepSimulation(physicsClientId=self.client)

        self._step_count += 1
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
        # bore axis with ``hub_axis_world`` for mount. The gate stays
        # active in Stage 0 (pickup pose enforced) and Stage 2 (post-
        # landing pose enforced).
        vertical_violated = (
            self.task_stage != 1
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

    def _apply_action(self, action: np.ndarray) -> None:
        ps = self.cfg.action.pos_scale
        rs = self.cfg.action.rot_scale
        # Robot A: 6 — always the leading slice regardless of action dim.
        d_pos_A = action[0:3] * ps
        d_rot_A = action[3:6] * rs
        self.robot_A.apply_delta_ee(d_pos_A, d_rot_A)
        # Robot B: 6 channels live at ``action[6:12]`` only when the env
        # is in the full 13-d action layout (Phase 2/3). In Phase 1 the
        # action space is sliced to 7-d so the Panda block is *not in
        # the action vector at all* — we just pin Panda at HOME.
        # ``reset_to_home`` only seeds joint positions; without a motor
        # target the arm would slowly droop under gravity over the 500-
        # step episode, so we re-seed every control tick.
        if self.cfg.freeze_robot_b:
            self.robot_B.reset_to_home()
        else:
            d_pos_B = action[6:9] * ps
            d_rot_B = action[9:12] * rs
            self.robot_B.apply_delta_ee(d_pos_B, d_rot_B)
        # Gripper A (Tier-1: ignored at sim level, the constraint holds the
        # tire). Lives at ``action[-1]`` in both layouts (index 6 for 7-d,
        # index 12 for 13-d). Kept in obs as ``prev_action`` so the policy
        # can still emit the binary intent.

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
        rb_base = np.asarray(self.cfg.robot_B_base_pos, dtype=np.float64)
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

        obs = np.concatenate([
            qA_n, dqA_n,                              # 12
            qB_n, dqB_n,                              # 14
            eeA_pos_rel / ws, eeA_orn,                # 7
            eeB_pos_rel / ws, eeB_orn,                # 7
            tire_pos_rel / ws, tire_orn,              # 7
            hub_pos_rel / ws, hub_orn,                # 7
            bolt_pos_rel / ws, bolt_orn,              # 7
            rel_th_pos / ws, rel_th_rot / np.pi,      # 6
            rel_eb_pos / ws, rel_eb_rot / np.pi,      # 6
            self._prev_action,                         # 7 (Phase 1) / 13 (Phase 2+)
            mount_tail,                               # 3
        ]).astype(np.float32)

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
        eeB_pos, eeB_orn = self.robot_B.ee_pose()
        bolt_pos, _bolt_orn = self.scene.bolt_pose()
        bolt_axis = self.scene.bolt_axis()
        eeB_z = quat_axis(eeB_orn, "z")

        # Align term is computed in every stage for diagnostic logging, but
        # only injected into the dense reward for the carrying stage (1).
        b.align_A, b.d_A, b.theta_A = rewards.align_reward(
            tire_pos, hub_pos, tire_axis, hub_axis, rcfg)
        b.reach_B, b.d_B, b.theta_B = rewards.reach_reward(
            eeB_pos, bolt_pos, eeB_z, bolt_axis, rcfg)
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

        if self.task_stage == 0:
            # Shape UR10 EE toward the tire's 6 o'clock outer point.
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
            # Suppress the Stage-1 align signal so the policy isn't pulled
            # toward the hub before grasping.
            b.align_A = 0.0
        elif self.task_stage == 1:
            # Original alignment reward dominates in Stage 1.
            b.approach_A = 0.0
            b.d_approach = 0.0
        else:  # task_stage == 2
            # Shape tire COM back toward its original pickup pose; add a
            # soft-landing penalty on tire descent speed.
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
            # Stage-1 align no longer applicable; zero it out for clarity.
            b.align_A = 0.0
            b.approach_A = 0.0
        b.pb_shape = float(pb_step)

        # FSM transition bonuses (paid exactly once per event).
        if fsm_events.get("picked_up"):
            b.fsm_bonus += float(rcfg.R_pickup)
        if fsm_events.get("mounted"):
            b.fsm_bonus += float(rcfg.R_mount)
        if fsm_events.get("landed"):
            b.fsm_bonus += float(rcfg.R_return)
            b.is_success = True

        # Vertical-pose dense penalty (squared angle, bounded by tol²).
        # Mirrors the Stage-1 carve-out applied to the termination gate
        # in ``step``: during carry/mount the tire is permitted to rotate
        # toward ``hub_axis_world`` (-Y) and so the penalty is zeroed.
        if self.task_stage == 1:
            b.vertical_pen = 0.0
        else:
            b.vertical_pen = -rcfg.w_vertical * float(vertical_err ** 2)

        # Always-on penalties --------------------------------------------
        b.coop = rewards.coop_reward(b.d_A, b.d_B, rcfg)
        b.sync_joint_A = rewards.sync_joint_a_penalty(dqA, rcfg)
        # Legacy sparse success path still produces a separate dense bonus
        # for back-compat (e.g. analytics). It does not drive Phase-1 done.
        b.success, _legacy_succ = rewards.success_bonus(
            b.d_A, b.theta_A, b.d_B, b.theta_B, rcfg,
            mount=(ax_th, lat_th, lug_e),
        )
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

        # Dense subtotal (for logging). Stage-aware: each stage contributes
        # exactly one A-side term plus the PB shaping bonus and the
        # always-on reach_B + safety penalties.
        stage_dense_A = {
            0: float(b.approach_A),
            1: float(b.align_A),
            2: float(b.return_A) + float(b.landing),
        }[int(self.task_stage)]
        common = ("coop", "sync_joint_A", "collision", "workspace",
                  "action", "jerk", "vertical_pen")
        b.dense_total_pre_mix = float(
            stage_dense_A + float(b.pb_shape) + float(b.reach_B)
            + sum(getattr(b, k) for k in common)
        )

        # Final aggregate — dense process + sparse stage bonus.
        b.total = float(
            rcfg.mix_dense * b.dense_total_pre_mix
            + rcfg.mix_sparse_success * float(b.fsm_bonus)
        )
        return b.total, b

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _in_bad_collision(self) -> bool:
        # Robot bodies vs ground plane.
        for robot in (self.robot_A, self.robot_B):
            cps = p.getContactPoints(bodyA=robot.uid, bodyB=self.handles.plane,
                                     physicsClientId=self.client)
            for cp in cps:
                # link 0 / -1 is the base; foot contact is fine, but mid-arm isn't.
                if cp[3] > 1:    # link index on robot side
                    return True
        # Robot A vs Robot B
        cps = p.getContactPoints(bodyA=self.robot_A.uid, bodyB=self.robot_B.uid,
                                 physicsClientId=self.client)
        if len(cps) > 0:
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
        # Phase 1 FSM final success — tire safely landed on the floor.
        if fsm_events.get("landed"):
            info["is_success"] = True
            return True, False, {**info, "termination": "success"}
        # Strict vertical-pose gate — tire must stay upright through pick,
        # carry and place. Violation = penalty termination.
        if vertical_violated:
            return True, False, {**info, "termination": "vertical_violation"}
        # Trigger on the raw geometric condition, not on the penalty value —
        # the latter is gated to 0 in early stages (w_collision = 0) and would
        # otherwise let the episode keep running through a clipping contact.
        if in_collision:
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
