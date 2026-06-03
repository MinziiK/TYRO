"""Thin wrappers around UR10 and Franka Panda for delta-EE-pose control via IK.

Each Robot exposes (joint_pos, joint_vel, ee_pos, ee_quat) for observation and
takes a `(Δpos[3], Δrpy[3])` command per control step (spec §3). IK is computed
once per control step against the current EE pose; joint targets drive position
servos which the env then steps through `decimation` sim sub-steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from ..config import EnvConfig
from .utils import (
    quat_multiply, rpy_to_quat, axisangle3_to_quat, relative_axisangle,
)


@dataclass
class JointGroup:
    """Subset of joints we drive with position control."""
    indices: List[int]
    lower: np.ndarray
    upper: np.ndarray
    rest: np.ndarray
    range: np.ndarray   # upper - lower

    @property
    def n(self) -> int:
        return len(self.indices)


class Robot:
    """Base wrapper. Subclasses set urdf_path, joint indices, EE link, home pose."""

    # filled by subclass
    NAME = "robot"
    EE_LINK_INDEX: int = 0
    HOME_POSE: Sequence[float] = ()

    def __init__(self, client: int, base_pos, base_orn,
                 urdf_path: str, search_path: Optional[str] = None,
                 use_fixed_base: bool = True):
        self.client = client
        self.base_pos = np.asarray(base_pos, dtype=np.float64)
        self.base_orn = np.asarray(base_orn, dtype=np.float64)

        if search_path:
            p.setAdditionalSearchPath(search_path, physicsClientId=client)

        self.uid: int = p.loadURDF(
            urdf_path,
            basePosition=list(base_pos),
            baseOrientation=list(base_orn),
            useFixedBase=use_fixed_base,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=client,
        )

        self.arm: JointGroup = self._build_joint_group(self._arm_joint_indices())
        # Cache: position of each arm joint within calculateInverseKinematics()'s
        # output, which spans all controllable (non-fixed) joints in PyBullet's
        # enumeration order. Avoids the fragile assumption that arm joints occupy
        # the first len(arm) slots.
        self._ik_arm_slots: List[int] = self._compute_ik_arm_slots()
        # Most recent IK target EE position (set by ``apply_delta_ee``). The env
        # compares this to the post-step achieved EE pose to log IK tracking
        # residual — useful for catching reach-saturation in DR rollouts.
        self.last_target_pos: Optional[np.ndarray] = None
        self._disable_non_arm_motors()
        self.reset_to_home()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _arm_joint_indices(self) -> List[int]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Joint utilities
    # ------------------------------------------------------------------
    def _joints_by_name(self, suffixes: Sequence[str]) -> List[int]:
        """Resolve PyBullet joint indices by matching name (full or suffix).

        Robust to URDF re-ordering and library version drift, unlike hard-coded
        positional indices.
        """
        n = p.getNumJoints(self.uid, physicsClientId=self.client)
        all_names: List[str] = []
        for j in range(n):
            raw = p.getJointInfo(self.uid, j, physicsClientId=self.client)[1]
            all_names.append(
                raw.decode("utf8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            )
        out: List[int] = []
        for sfx in suffixes:
            match = -1
            for j, name in enumerate(all_names):
                if name == sfx or name.endswith(sfx):
                    match = j
                    break
            if match < 0:
                raise RuntimeError(
                    f"{self.NAME}: joint matching '{sfx}' not found; available: {all_names}"
                )
            out.append(match)
        return out

    def _compute_ik_arm_slots(self) -> List[int]:
        """For each arm joint, its position in calculateInverseKinematics()'s output."""
        n = p.getNumJoints(self.uid, physicsClientId=self.client)
        controllable: List[int] = []
        for j in range(n):
            info = p.getJointInfo(self.uid, j, physicsClientId=self.client)
            if info[2] != p.JOINT_FIXED:
                controllable.append(j)
        try:
            return [controllable.index(j) for j in self.arm.indices]
        except ValueError as exc:
            raise RuntimeError(
                f"{self.NAME}: arm joint not in controllable set "
                f"(arm={self.arm.indices}, controllable={controllable})"
            ) from exc


    def _disable_non_arm_motors(self) -> None:
        """Disable default velocity controllers on non-arm joints so they don't
        fight against position targets we send."""
        n = p.getNumJoints(self.uid, physicsClientId=self.client)
        non_arm = [j for j in range(n) if j not in self.arm.indices]
        if non_arm:
            p.setJointMotorControlArray(
                self.uid, non_arm,
                controlMode=p.VELOCITY_CONTROL,
                forces=[0.0] * len(non_arm),
                physicsClientId=self.client,
            )

    def _build_joint_group(self, indices: List[int]) -> JointGroup:
        lower, upper, rest = [], [], []
        for j in indices:
            info = p.getJointInfo(self.uid, j, physicsClientId=self.client)
            if info[2] == p.JOINT_FIXED:
                raise ValueError(
                    f"{self.NAME}: joint index {j} ('{info[1]}') is JOINT_FIXED "
                    f"but was selected as an arm joint"
                )
            ll, ul = info[8], info[9]
            if ll >= ul:        # no limit declared — use a wide range
                ll, ul = -np.pi, np.pi
            lower.append(ll); upper.append(ul)
        rest = list(self.HOME_POSE) if self.HOME_POSE else [0.0] * len(indices)
        return JointGroup(
            indices=indices,
            lower=np.array(lower, dtype=np.float64),
            upper=np.array(upper, dtype=np.float64),
            rest=np.array(rest, dtype=np.float64),
            range=np.array(upper, dtype=np.float64) - np.array(lower, dtype=np.float64),
        )

    def reset_to_home(self) -> None:
        for idx, q in zip(self.arm.indices, self.arm.rest):
            p.resetJointState(self.uid, idx, targetValue=float(q),
                              targetVelocity=0.0, physicsClientId=self.client)
        # Seed the absolute IK target with the HOME EE forward kinematics.
        # ``apply_delta_ee`` accumulates Δpos onto this analytical target
        # rather than rebasing on the (gravity-perturbed) measured EE pose
        # every step — without this seed, the first ``cur_pos`` read after
        # reset would already include sub-millimetre sag from the 5-step
        # settle in ``TyroEnv.reset`` and lock the policy onto a sagging
        # baseline for the rest of the episode.
        self.last_target_pos = self.ee_pose()[0].copy()

    def joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        states = p.getJointStates(self.uid, self.arm.indices,
                                  physicsClientId=self.client)
        q = np.array([s[0] for s in states], dtype=np.float64)
        dq = np.array([s[1] for s in states], dtype=np.float64)
        return q, dq

    # ------------------------------------------------------------------
    # End-effector pose
    # ------------------------------------------------------------------
    def ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        ls = p.getLinkState(self.uid, self.EE_LINK_INDEX,
                            computeForwardKinematics=True,
                            physicsClientId=self.client)
        return np.asarray(ls[4], dtype=np.float64), np.asarray(ls[5], dtype=np.float64)

    # ------------------------------------------------------------------
    # Delta-pose action
    # ------------------------------------------------------------------
    def apply_delta_ee(self, delta_pos: np.ndarray, delta_axisangle: np.ndarray) -> None:
        """Move EE by Δpos / Δrotation (axis*angle, world frame).

        Sets joint position targets via IK; physics step is owned by the env.
        """
        cur_pos, cur_orn = self.ee_pose()
        target_pos = cur_pos + np.asarray(delta_pos, dtype=np.float64)
        d_quat = axisangle3_to_quat(delta_axisangle)
        target_orn = quat_multiply(d_quat, cur_orn)
        self.last_target_pos = target_pos

        ik = p.calculateInverseKinematics(
            self.uid, self.EE_LINK_INDEX,
            list(target_pos), list(target_orn),
            lowerLimits=self.arm.lower.tolist(),
            upperLimits=self.arm.upper.tolist(),
            jointRanges=self.arm.range.tolist(),
            restPoses=self.arm.rest.tolist(),
            maxNumIterations=50,
            residualThreshold=1e-3,
            physicsClientId=self.client,
        )
        # IK returns a target for every controllable joint of the body, in the
        # order PyBullet enumerates them. Index each arm joint's slot explicitly
        # — assuming arm joints occupy [0:n] silently breaks when the URDF has
        # a non-arm controllable joint (gripper, tool) ahead of them.
        ik = np.asarray(ik, dtype=np.float64)
        max_slot = max(self._ik_arm_slots) if self._ik_arm_slots else -1
        if len(ik) > max_slot:
            arm_targets = ik[self._ik_arm_slots]
        else:
            arm_targets = self._fallback_targets()
        # Clamp to limits before sending.
        arm_targets = np.clip(arm_targets, self.arm.lower, self.arm.upper)

        # Stiffer PD (positionGain 0.1 → 1.0; default velocityGain stays 1.0)
        # so the arm actually holds the commanded joint posture when a
        # mass (e.g. the 0.5 kg tire bonded to the UR10 EE) is attached.
        # Without this the joints visibly droop under gravity, dragging
        # the tire down and rolling it relative to the world — exactly
        # the "wobble / sag" symptom we want to eliminate.
        n = self.arm.n
        p.setJointMotorControlArray(
            self.uid, self.arm.indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=arm_targets.tolist(),
            forces=[150.0] * n,
            positionGains=[1.0] * n,
            velocityGains=[1.0] * n,
            physicsClientId=self.client,
        )

    def _fallback_targets(self) -> np.ndarray:
        q, _ = self.joint_state()
        return q


class UR10Robot(Robot):
    NAME = "ur10"
    EE_LINK_INDEX = 7  # robot_ee_link
    # 2026-05-28 (post-laydown): the previous HOME_POSE was designed for
    # a vertical-tire layout. After ``tire_spawn_rpy = (0, π/2, 0)`` was
    # adopted (bore along +X to face Robot A), that HOME parked the EE at
    # (−1.90, 0.00, +0.27) — *inside* the tire bore (YZ dist = 4.5 cm <
    # bore radius 28.2 cm, X mid-thickness), so the gripper was born
    # impaled on the tire and every episode terminated on contact_force.
    #
    # New compact tool-up HOME (Robot B-centric coords, robot_A_base at
    # (−0.80, 0, −0.30); tire COM (−1.90, 0, +0.225), bore axis +X,
    # outer radius 0.525 m, inner 0.282 m, thickness 0.30 m):
    #
    #   EE world pose = (−1.55, −0.05, +0.55), tool +Z ‖ world +Z.
    #
    # That places the gripper +33 cm above the tire equator and +35 cm
    # in front of the tire's front face (X = −1.75), entirely clear of:
    #   * tire AABB         X ∈ [−2.05, −1.75]
    #   * rack rail AABBs   X ∈ [−2.05, −1.75], |Y| ∈ [0.05, 0.35]
    #   * vehicle / cargo   Y > +0.60
    # AABB overlap with every blocker was verified zero by
    # scripts/find_compact_home.py.
    #
    # EE↔grasp_target = 92.1 cm at HOME, comfortably OUTSIDE both the
    # soft gate (62 cm) and hard cap (55 cm) of the approach_tol
    # curriculum — the policy now has room to learn an active descent.
    #
    # 2026-05-29 — tool-up "palms-up cradle" HOME:
    # The previous compact HOME let the gripper face an angled direction
    # rather than world +Z; for an underhand cradle pickup the gripper
    # cup must look straight up. New joint vector (rad):
    #
    #   shoulder_pan  = π        ( +180° — rotate 0° toward −X, facing tire)
    #   shoulder_lift = −1.1344  ( ≈ −65° — shoulder tipped up)
    #   elbow         = +1.1344  ( ≈ +65° — elbow folded, avoids singularity)
    #   shoulder_pan  = +180°   shoulder_lift = −65°   elbow = +130°
    #   wrist_1 = −155°   wrist_2 = 0°    wrist_3 = −90°
    HOME_POSE = (3.14159, -1.13446, 2.26893, -2.70526, 0.0, -1.57080)

    #: ``FINAL_LOCK_QUATERNION`` — the FULL 3-D EE orientation that IK is
    #: pinned to when ``EnvConfig.ur10_lock_tool_up`` is True. Equal to the
    #: FK of ``HOME_POSE`` under the palm-up pose
    #: ``[180, −65, 130, −155, 0, −90]°``: tool +Z ‖ world +Z (palm up),
    #: tool +X ‖ world −Y (finger-closure points away from the truck),
    #: tool +Y ‖ world +X. The wrist_2 link still lies flat in XY but
    #: rotated 180° about world Z vs the previous "+180/+90" variant.
    #: Re-measure with ``scripts/verify_home.py`` whenever ``HOME_POSE``
    #: changes. ``TOOL_UP_QUATERNION`` is kept as a backwards-compatible
    #: alias.
    FINAL_LOCK_QUATERNION = (
        9.381810e-07, -9.381835e-07, -0.7071077, 0.7071058,
    )
    TOOL_UP_QUATERNION = FINAL_LOCK_QUATERNION

    def _arm_motor_forces(self) -> List[float]:
        return [400.0, 400.0, 300.0, 60.0, 60.0, 60.0]

    def __init__(self, client: int, cfg: EnvConfig):
        self._lock_tool_up = bool(getattr(cfg, "ur10_lock_tool_up", True))
        # 2026-06-03 — commanded-joint-target motion smoothing (see
        # ``EnvConfig.ur10_joint_target_smooth_alpha`` / ``..._max_step_rad``).
        self._smooth_alpha = float(
            getattr(cfg, "ur10_joint_target_smooth_alpha", 1.0)
        )
        self._max_step = float(getattr(cfg, "ur10_joint_max_step_rad", 0.0))
        self._pgain = float(getattr(cfg, "ur10_position_gain", 1.0))
        self._vgain = float(getattr(cfg, "ur10_velocity_gain", 1.0))
        # 2026-06-04 — per-joint motor speed cap (rad/s). 0 = unlimited.
        # Passed to PyBullet POSITION_CONTROL ``maxVelocity`` so the stiff
        # PD cannot whip the arm during the near-singular hub insertion
        # (the residual source of the 70 cm/step "오락가락" EE snap once the
        # carry itself is smooth). Throttles velocity *inside* the physics
        # sub-steps — unlike ``_max_step`` which clamps the commanded
        # target delta but lets the PD overshoot toward it.
        self._max_joint_vel = float(
            getattr(cfg, "ur10_motor_max_velocity_rad_s", 0.0)
        )
        self._joint_slew_max = float(
            getattr(cfg, "ur10_joint_slew_max_rad", 0.08),
        )
        # Last commanded joint vector (post-smoothing). ``None`` ⇒ seed from
        # the current measured state on the next drive call (and after reset).
        self._cmd_q: Optional[np.ndarray] = None
        super().__init__(
            client=client,
            base_pos=cfg.robot_A_base_pos,
            base_orn=rpy_to_quat(cfg.robot_A_base_rpy),
            urdf_path=cfg.ur10_urdf,
            search_path=cfg.ur10_search_path,
        )

    def reset_to_home(self) -> None:
        super().reset_to_home()
        # Re-seed the smoothing filter at the HOME pose so the first
        # post-reset command doesn't slew from a stale target.
        self._cmd_q = None

    def _smooth_arm_targets(self, arm_targets: np.ndarray) -> np.ndarray:
        """Low-pass the raw IK joint targets before they hit the motors.

        ``q_cmd = alpha·q_ik + (1-alpha)·q_cmd_prev`` (EMA), then an
        optional hard per-step slew clamp. A no-op when ``alpha == 1.0``
        and ``max_step == 0.0`` (the training default), so trained
        policies are unaffected. The filter state seeds from the current
        measured joint vector on the first call after a reset.
        """
        alpha = self._smooth_alpha
        max_step = self._max_step
        if alpha >= 1.0 and max_step <= 0.0:
            return arm_targets
        if self._cmd_q is None:
            self._cmd_q, _ = self.joint_state()
        prev = self._cmd_q
        if alpha < 1.0:
            cmd = alpha * arm_targets + (1.0 - alpha) * prev
        else:
            cmd = arm_targets.astype(np.float64, copy=True)
        if max_step > 0.0:
            delta = np.clip(cmd - prev, -max_step, max_step)
            cmd = prev + delta
        cmd = np.clip(cmd, self.arm.lower, self.arm.upper)
        self._cmd_q = cmd.copy()
        return cmd

    def apply_delta_ee(self, delta_pos: np.ndarray, delta_axisangle: np.ndarray) -> None:
        """Δpos drives EE translation. When ``ur10_lock_tool_up`` is True:

        * IK is run with a **fixed 6-D target** — position from the
          policy's Δpos, orientation **always** equal to
          :pyattr:`FINAL_LOCK_QUATERNION` (the FK of ``HOME_POSE`` where
          tool +Z ‖ world +Z and the wrist_2 link is laid flat in XY).
        * **No post-IK clamp.** All six joints are free for IK to solve.
          The wrist_2 joint auto-compensates for ``shoulder_lift +
          elbow + wrist_1`` motion so that the tool +Z axis stays
          aligned with world +Z; wrist_3 is whatever satisfies the
          fixed finger closure direction baked into the quaternion.
        * Geometric intuition: ``shoulder_pan`` rotation is invisible to
          tool +Z (rotating about world Z fixes the +Z vector). The
          ``shoulder_lift / elbow / wrist_1`` chain pitches the wrist
          cluster up/down; ``wrist_2`` is the one DOF needed to undo
          that pitch and keep the gripper palm flat.

        ``delta_axisangle`` is ignored in lock mode (the env also zeroes
        it before calling — defence-in-depth).

        **Drift-free absolute target accumulator.** Δpos is added to
        ``self.last_target_pos`` (seeded at HOME-EE FK in
        ``reset_to_home``) rather than to the measured EE pose. Gravity
        sag, IK numerical jitter, and joint-limit clamping all cause the
        measured EE to lag the commanded target by 1–30 cm depending on
        load; rebasing on ``cur_pos`` every step would compound that lag
        into a runaway drift (~0.3 cm/step under the current PD gains).
        Using a pure mathematical accumulator means the motor always
        fights *back* to the original analytical target — sag becomes a
        bounded steady-state offset instead of an unbounded sink."""
        if self.last_target_pos is None:
            self.last_target_pos = self.ee_pose()[0].copy()
        self.last_target_pos = self.last_target_pos + np.asarray(
            delta_pos, dtype=np.float64
        )
        target_pos = self.last_target_pos

        if self._lock_tool_up:
            target_orn = list(self.FINAL_LOCK_QUATERNION)
        else:
            _cur_pos, cur_orn = self.ee_pose()
            d_quat = axisangle3_to_quat(delta_axisangle)
            target_orn = list(quat_multiply(d_quat, cur_orn))

        ik = p.calculateInverseKinematics(
            self.uid, self.EE_LINK_INDEX,
            list(target_pos), target_orn,
            lowerLimits=self.arm.lower.tolist(),
            upperLimits=self.arm.upper.tolist(),
            jointRanges=self.arm.range.tolist(),
            restPoses=self.arm.rest.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        max_slot = max(self._ik_arm_slots) if self._ik_arm_slots else -1
        if len(ik) > max_slot:
            arm_targets = ik[self._ik_arm_slots]
        else:
            arm_targets = self._fallback_targets()
        arm_targets = np.clip(arm_targets, self.arm.lower, self.arm.upper)
        # Per-joint torque caps (N·m) realistic to the UR10 datasheet, in the
        # same order as ``_arm_joint_indices`` returns them (shoulder_pan,
        # shoulder_lift, elbow, wrist_1, wrist_2, wrist_3). The original
        # uniform 150 N·m was too weak for the large lower joints (real
        # UR10 shoulder/elbow caps are 150–330 N·m) and overspec'ed for
        # the wrists (real ≤ 56 N·m), causing visible gravity sag at HOME
        # while leaving the wrists too "twitchy" to converge against PD.
        # New caps: 400/400/300 for shoulder/elbow give a healthy buffer
        # over the worst-case static gravity moment (~120 N·m at HOME
        # under a 0.5 kg tire load) and 60 for the wrists matches the
        # real torque-limited region while still beating wrist inertia.
        forces = self._arm_motor_forces()
        pgains = [self._pgain] * len(forces)
        vgains = [self._vgain] * len(forces)
        p.setJointMotorControlArray(
            self.uid, self.arm.indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=arm_targets.tolist(),
            forces=forces,
            positionGains=pgains,
            velocityGains=vgains,
            physicsClientId=self.client,
        )

    def drive_arm_targets_toward(
        self,
        target_q: np.ndarray,
        max_step_rad: float,
    ) -> None:
        """Slew measured joints toward ``target_q`` (caps per-step jump)."""
        cap = float(max_step_rad)
        if cap <= 0.0:
            self.drive_arm_targets(target_q)
            return
        cur_q, _ = self.joint_state()
        delta = np.clip(
            np.asarray(target_q, dtype=np.float64) - cur_q,
            -cap, cap,
        )
        self.drive_arm_targets(cur_q + delta)

    def drive_arm_targets(self, arm_targets: np.ndarray) -> None:
        """Send (optionally smoothed) joint targets to the arm motors."""
        arm_targets = np.clip(
            np.asarray(arm_targets, dtype=np.float64),
            self.arm.lower, self.arm.upper,
        )
        arm_targets = self._smooth_arm_targets(arm_targets)
        forces = self._arm_motor_forces()
        if self._max_joint_vel > 0.0:
            # setJointMotorControlArray ignores maxVelocity, so issue a
            # per-joint control2 call to enforce the speed cap.
            for i, j in enumerate(self.arm.indices):
                p.setJointMotorControl2(
                    self.uid, j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=float(arm_targets[i]),
                    force=float(forces[i]),
                    positionGain=self._pgain,
                    velocityGain=self._vgain,
                    maxVelocity=self._max_joint_vel,
                    physicsClientId=self.client,
                )
        else:
            p.setJointMotorControlArray(
                self.uid, self.arm.indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=arm_targets.tolist(),
                forces=forces,
                positionGains=[self._pgain] * len(forces),
                velocityGains=[self._vgain] * len(forces),
                physicsClientId=self.client,
            )

    def solve_arm_joints_in_snapshot(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        warm_q: np.ndarray,
    ) -> np.ndarray:
        """IK only (caller owns save/restore). Chained warm-start for baking."""
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        target_orn = np.asarray(quat, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(target_orn))
        if n > 1e-12:
            target_orn = target_orn / n
        else:
            target_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        warm = np.clip(
            np.asarray(warm_q, dtype=np.float64).reshape(-1),
            self.arm.lower, self.arm.upper,
        )
        ik = p.calculateInverseKinematics(
            self.uid, self.EE_LINK_INDEX,
            target_pos.tolist(), target_orn.tolist(),
            lowerLimits=self.arm.lower.tolist(),
            upperLimits=self.arm.upper.tolist(),
            jointRanges=self.arm.range.tolist(),
            restPoses=warm.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        max_slot = max(self._ik_arm_slots) if self._ik_arm_slots else -1
        if len(ik) > max_slot:
            arm_targets = ik[self._ik_arm_slots]
        else:
            arm_targets = self._fallback_targets()
        return np.clip(arm_targets, self.arm.lower, self.arm.upper)

    def apply_palm_up_locked(self, pos) -> None:
        """**Iterative palm-up IK** with save/restore state isolation.

        2026-06-02 (path B — closed-form-style). Runs PyBullet's IK in
        a state-snapshot, projects the result onto the palm-up manifold

            wrist_1 = -pi/2 - shoulder_lift - elbow
            wrist_2 = 0
            wrist_3 = -pi/2

        re-seeds the next pass from the projected joint vector, and
        repeats until the position error is small or 5 passes elapse.
        The state-snapshot keeps the rest of the simulation (notably
        the ``JOINT_FIXED`` tire bond) frozen during iteration, so the
        intermediate ``resetJointState`` calls don't shock the bond
        with phantom impulses.

        The converged joint vector is then driven via stiff PD —
        because the IK target is a smooth function of ``pos`` and the
        previous step's converged state, consecutive control steps
        produce smooth joint trajectories (no sudden wrist snaps),
        which keeps bond reaction forces in the physically expected
        range (a few N for the 0.5 kg tire).

        The post-step ``enforce_palm_up_wrists`` safety net (called
        from ``TyroEnv.step``) handles the rare case where PD
        undershoots and the achieved wrist values drift off the
        palm-up manifold.
        """
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        target_orn = list(self.FINAL_LOCK_QUATERNION)
        self.last_target_pos = target_pos.copy()

        state_id = p.saveState(physicsClientId=self.client)
        try:
            # Warm-start from CURRENT joint state so the converged IK
            # target is in the same branch as the previous step. This
            # keeps PD targets continuous step-to-step (essential for
            # the JOINT_FIXED tire bond — large per-step joint jumps
            # generate kN-scale phantom impulses on the bond).
            cur_q, _ = self.joint_state()
            warm = cur_q.copy()
            # Project the warm start onto the palm-up manifold so the
            # very first IK pass is already close to the manifold.
            warm[3] = -math.pi / 2.0 - warm[1] - warm[2]
            warm[4] = 0.0
            warm[5] = -math.pi / 2.0
            warm = np.clip(warm, self.arm.lower, self.arm.upper)
            arm_targets: Optional[np.ndarray] = None
            for _pass in range(5):
                ik = p.calculateInverseKinematics(
                    self.uid, self.EE_LINK_INDEX,
                    target_pos.tolist(), target_orn,
                    lowerLimits=self.arm.lower.tolist(),
                    upperLimits=self.arm.upper.tolist(),
                    jointRanges=self.arm.range.tolist(),
                    restPoses=warm.tolist(),
                    maxNumIterations=200,
                    residualThreshold=1e-5,
                    physicsClientId=self.client,
                )
                ik = np.asarray(ik, dtype=np.float64)
                max_slot = max(self._ik_arm_slots) if self._ik_arm_slots else -1
                if len(ik) > max_slot:
                    arm_targets = ik[self._ik_arm_slots].copy()
                else:
                    arm_targets = self._fallback_targets()
                # Project onto the palm-up manifold.
                arm_targets[3] = -math.pi / 2.0 - arm_targets[1] - arm_targets[2]
                arm_targets[4] = 0.0
                arm_targets[5] = -math.pi / 2.0
                arm_targets = np.clip(arm_targets, self.arm.lower, self.arm.upper)
                # Set the (snapshot-protected) joint state for the next
                # IK pass and for the convergence FK check below.
                for idx, qv in zip(self.arm.indices, arm_targets):
                    p.resetJointState(
                        self.uid, idx,
                        targetValue=float(qv), targetVelocity=0.0,
                        physicsClientId=self.client,
                    )
                ee_now, _ = self.ee_pose()
                if float(np.linalg.norm(ee_now - target_pos)) < 1e-4:
                    break
                warm = arm_targets.copy()
            if arm_targets is None:
                arm_targets = self._fallback_targets()
        finally:
            p.restoreState(stateId=state_id, physicsClientId=self.client)
            p.removeState(stateUniqueId=state_id, physicsClientId=self.client)

        self.drive_arm_targets(arm_targets)

    def enforce_palm_up_wrists(self, tool_z_threshold: float = 0.999) -> None:
        """Geometric safety net: snap wrist_1/2/3 to the palm-up formula
        ONLY when the IK-driven gripper has drifted off the palm-up
        manifold by more than ``acos(tool_z_threshold)`` (default ≈ 2.5°).

        ``tool_z_threshold`` is the dot product of the gripper's tool +Z
        axis with the world +Z axis; 1.0 = perfectly palm-up, 0.999 =
        ~2.5° tilt. Most steps will be inside the threshold (IK with
        HOME warm-start typically holds palm-up to numerical precision),
        so the wrist override almost never fires — meaning
        ``JOINT_FIXED`` rigid grasp + IK position accuracy work
        normally and the tire is physically held upright by the bond.
        On the rare step where IK saturates and palm-up slips, this
        method snaps the wrists back via
        ``resetJointState`` (with ``JOINT_FIXED`` the tire snaps with
        them; the resulting physics impulse is small because the
        deviation was small to begin with).
        """
        ee_pos, ee_orn = self.ee_pose()
        R = np.array(p.getMatrixFromQuaternion(list(ee_orn))).reshape(3, 3)
        tool_z_dot_world_z = float(R[2, 2])
        if tool_z_dot_world_z >= tool_z_threshold:
            return  # palm-up is healthy; do nothing.

        states = p.getJointStates(
            self.uid, self.arm.indices, physicsClientId=self.client,
        )
        q = np.array([s[0] for s in states], dtype=np.float64)
        lift = float(q[1])
        elbow = float(q[2])
        w1 = float(np.clip(
            -math.pi / 2.0 - lift - elbow,
            float(self.arm.lower[3]), float(self.arm.upper[3]),
        ))
        w2 = float(np.clip(
            0.0, float(self.arm.lower[4]), float(self.arm.upper[4]),
        ))
        w3 = float(np.clip(
            -math.pi / 2.0,
            float(self.arm.lower[5]), float(self.arm.upper[5]),
        ))
        for slot, val in zip((3, 4, 5), (w1, w2, w3)):
            p.resetJointState(
                self.uid,
                self.arm.indices[slot],
                targetValue=float(val),
                targetVelocity=0.0,
                physicsClientId=self.client,
            )
        wrist_indices = [self.arm.indices[s] for s in (3, 4, 5)]
        p.setJointMotorControlArray(
            self.uid, wrist_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[w1, w2, w3],
            forces=[60.0, 60.0, 60.0],
            positionGains=[1.0, 1.0, 1.0],
            velocityGains=[1.0, 1.0, 1.0],
            physicsClientId=self.client,
        )

    def apply_palm_up_pose(self, pos, target_orn, warm_q=None) -> None:
        """Absolute-pose IK with **HOME-based warm start**, for palm-up.

        ``warm_q`` (2026-06-04): optional explicit IK warm-start joint
        vector. When the planner has a baked joint trajectory, the env
        passes ``_traj_q[idx]`` here so the residual-offset IK stays in
        the same clean branch as the reachable baked path (instead of
        re-solving from the live joints, which let the realised carry
        path snap to far IK branches → "오락가락"). ``None`` falls back to
        the current joint state.

        2026-06-01 (Option B): ``apply_absolute_ee`` warm-starts IK from
        the *current* joint state, which is intended to keep the solver
        in the same branch as the previous step. But when we tilt-lock
        the orientation to palm-up, the current state may be in a
        partially-tilted branch (because SLERP from a non-palm-up
        previous target left the arm there) and IK then converges to
        an upside-down "shoulder rotated 180°" solution that satisfies
        ``tool +Z = world −Z`` instead of ``+Z``. Diagnostic smoke
        showed ``tool_z · world_z`` flipping to −0.58 mid-trajectory.
        Switching the warm start to ``HOME_POSE`` (the canonical
        palm-up rest pose) consistently anchors IK to the upright
        branch. Residual / maxIter / PD gains identical to
        ``apply_delta_ee`` lock path which had clean palm-up
        convergence in the v0 smoke.
        """
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        target_orn = np.asarray(target_orn, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(target_orn))
        if n > 1e-12:
            target_orn = target_orn / n
        else:
            target_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.last_target_pos = target_pos.copy()

        # Warm-start from the baked joint vector when provided (keeps IK in
        # the reachable baked branch), else from the *current* arm state
        # (not HOME) so consecutive planner steps stay in the same IK
        # branch — HOME rest was the main source of step-to-step wrist
        # snaps / visible shaking.
        if warm_q is not None:
            rest_q = np.clip(
                np.asarray(warm_q, dtype=np.float64).reshape(-1),
                self.arm.lower, self.arm.upper,
            )
        else:
            rest_q, _ = self.joint_state()
        ik = p.calculateInverseKinematics(
            self.uid, self.EE_LINK_INDEX,
            target_pos.tolist(), target_orn.tolist(),
            lowerLimits=self.arm.lower.tolist(),
            upperLimits=self.arm.upper.tolist(),
            jointRanges=self.arm.range.tolist(),
            restPoses=rest_q.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        max_slot = max(self._ik_arm_slots) if self._ik_arm_slots else -1
        if len(ik) > max_slot:
            arm_targets = ik[self._ik_arm_slots]
        else:
            arm_targets = self._fallback_targets()
        arm_targets = np.clip(arm_targets, self.arm.lower, self.arm.upper)
        if self._joint_slew_max > 0.0:
            self.drive_arm_targets_toward(arm_targets, self._joint_slew_max)
        else:
            self.drive_arm_targets(arm_targets)

    def apply_absolute_ee(self, pos, quat) -> None:
        """Direct absolute (world-frame) EE-pose IK + joint control.

        This is the new API for the **Minimum-Jerk planner + PPO residual**
        control path (2026-06-01). Unlike :pymeth:`apply_delta_ee`, it does
        **not** accumulate onto ``last_target_pos`` and it **ignores**
        ``ur10_lock_tool_up`` — the caller (the planner) is fully
        responsible for orientation. The legacy lock was a hack for the
        broken delta path; planner mode supersedes it.

        ``pos`` and ``quat`` are world-frame absolute targets (quat in
        PyBullet xyzw). The full 6-DOF IK is run against them, joint
        targets are clamped to limits and pushed to PyBullet via the
        same stiff PD that the delta path uses.

        IK warm-start uses the **current** joint state as the rest pose
        (not ``HOME_POSE``) so the solver stays in the same IK branch as
        the previous step. Falling back to ``HOME_POSE`` lets the IK
        teleport across branches whenever the arm strays from HOME,
        producing step-1 spikes of tens of cm even for tiny pose deltas.
        """
        target_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        target_orn = np.asarray(quat, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(target_orn))
        if n > 1e-12:
            target_orn = target_orn / n
        else:
            target_orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.last_target_pos = target_pos.copy()

        # Warm-start IK from the *current* arm state so the solver picks
        # the same elbow / wrist branch as the previous step.
        cur_q, _ = self.joint_state()
        ik = p.calculateInverseKinematics(
            self.uid, self.EE_LINK_INDEX,
            target_pos.tolist(), target_orn.tolist(),
            lowerLimits=self.arm.lower.tolist(),
            upperLimits=self.arm.upper.tolist(),
            jointRanges=self.arm.range.tolist(),
            restPoses=cur_q.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        max_slot = max(self._ik_arm_slots) if self._ik_arm_slots else -1
        if len(ik) > max_slot:
            arm_targets = ik[self._ik_arm_slots]
        else:
            arm_targets = self._fallback_targets()
        arm_targets = np.clip(arm_targets, self.arm.lower, self.arm.upper)
        self.drive_arm_targets(arm_targets)

    # ------------------------------------------------------------------
    # 2026-06-04 — Damped least-squares resolved-rate Cartesian servo.
    # ------------------------------------------------------------------
    def _movable_joint_indices(self) -> List[int]:
        """All non-fixed joints in PyBullet enumeration order (cached).

        ``calculateJacobian`` needs position/velocity/accel vectors over
        *every* movable DOF, and returns columns in this same order; the
        arm columns are then sliced via ``_ik_arm_slots``.
        """
        cached = getattr(self, "_movable_cache", None)
        if cached is not None:
            return cached
        n = p.getNumJoints(self.uid, physicsClientId=self.client)
        movable = [
            j for j in range(n)
            if p.getJointInfo(self.uid, j, physicsClientId=self.client)[2]
            != p.JOINT_FIXED
        ]
        self._movable_cache = movable
        return movable

    def drive_ee_servo_dls(
        self,
        target_pos,
        target_quat,
        damping: float = 0.06,
        max_joint_step: float = 0.10,
        pos_gain: float = 1.0,
        orn_gain: float = 0.8,
        adaptive: bool = True,
        manip_threshold: float = 0.02,
    ) -> None:
        """Resolved-rate EE servo with damped least squares (DLS).

        2026-06-04 — replaces the per-step *absolute* IK (which can teleport
        across IK branches and, near reach saturation, makes the measured
        EE snap 40–70 cm in a single control step) with a **closed-loop**
        step on the EE error:

            e = [pos_gain·(p* − p);  orn_gain·axisangle(q → q*)]
            Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e            (damped least squares)
            q_cmd = clip(q + clip(Δq, ±max_joint_step))

        Because the command is always ``current_q + small Δq``, the PD
        target never sits far from the achievable pose, so there is no
        lag-then-burst.

        **Adaptive damping (Nakamura & Hanafusa 1986).** A *fixed* λ forces
        a single compromise: small λ tracks accurately but snaps at the
        singularity, large λ is smooth everywhere but never reaches the hub
        (the EE stalls 40–60 cm short). With ``adaptive=True`` the damping
        is scheduled on the manipulability ``w = √det(J Jᵀ)``:

            λ² = 0                                   if w ≥ w0
            λ² = damping² · (1 − w/w0)²              if w <  w0

        So far from singularities (w ≥ w0 = ``manip_threshold``) the servo
        tracks the plan exactly (zero damping → no stall), and damping
        switches on *only* as the arm enters the ill-conditioned hub-
        insertion region, smoothly bounding the EE velocity there. This is
        what lets a single controller be both jump-free during the approach
        **and** able to seat the tire.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
        nq = float(np.linalg.norm(target_quat))
        target_quat = (
            target_quat / nq if nq > 1e-12
            else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        self.last_target_pos = target_pos.copy()

        cur_pos, cur_orn = self.ee_pose()
        e_pos = (target_pos - np.asarray(cur_pos, dtype=np.float64)) * float(pos_gain)
        e_orn = relative_axisangle(cur_orn, target_quat) * float(orn_gain)
        err = np.concatenate([e_pos, e_orn])

        movable = self._movable_joint_indices()
        states = p.getJointStates(self.uid, movable, physicsClientId=self.client)
        q_full = [float(s[0]) for s in states]
        zeros = [0.0] * len(movable)
        jac_lin, jac_ang = p.calculateJacobian(
            self.uid, self.EE_LINK_INDEX,
            [0.0, 0.0, 0.0], q_full, zeros, zeros,
            physicsClientId=self.client,
        )
        J_full = np.vstack([
            np.asarray(jac_lin, dtype=np.float64),
            np.asarray(jac_ang, dtype=np.float64),
        ])  # (6, n_movable)
        J = J_full[:, self._ik_arm_slots]  # (6, n_arm)

        JJt = J @ J.T
        if bool(adaptive):
            # Manipulability-scheduled damping: zero away from singularities
            # (exact tracking → no stall), ramping up only as w → 0.
            det = float(max(np.linalg.det(JJt), 0.0))
            w = math.sqrt(det)
            w0 = float(manip_threshold)
            if w0 > 1e-12 and w < w0:
                scale = (1.0 - w / w0)
                lam2 = (float(damping) ** 2) * scale * scale
            else:
                lam2 = 0.0
        else:
            lam2 = float(damping) ** 2
        try:
            y = np.linalg.solve(JJt + lam2 * np.eye(6), err)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(JJt + lam2 * np.eye(6), err, rcond=None)[0]
        dq = J.T @ y

        cap = float(max_joint_step)
        if cap > 0.0:
            dq = np.clip(dq, -cap, cap)
        cur_q, _ = self.joint_state()
        q_cmd = np.clip(cur_q + dq, self.arm.lower, self.arm.upper)
        self.drive_arm_targets(q_cmd)

    def _arm_joint_indices(self) -> List[int]:
        # Match by URDF joint name — robust to non-arm joints (Robotiq gripper,
        # robot_ee_fixed_joint) appearing in PyBullet's enumeration.
        return self._joints_by_name([
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])


def make_robot_a(client: int, cfg: EnvConfig) -> Robot:
    """Factory for Robot A (UR10 default, optional FANUC R-2000iC)."""
    kind = str(getattr(cfg, "robot_a_kind", "ur10")).lower().replace("-", "_")
    if kind in ("fanuc", "fanuc_r2000ic", "fanuc_r2000ic210f", "r2000ic", "r2000ic210f"):
        return FanucR2000icRobot(client, cfg)
    return UR10Robot(client, cfg)


def robot_a_lock_quaternion(robot: Robot) -> np.ndarray:
    """Palm-up / stage planner lock quaternion for Robot A."""
    return np.asarray(
        getattr(robot, "FINAL_LOCK_QUATERNION", (0.0, 0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


class FanucR2000icRobot(UR10Robot):
    """FANUC R-2000iC/210F (ROS-Industrial URDF) as Robot A.

    Reuses UR10 motor smoothing, DLS servo, and planner IK helpers.
    Palm-up wrist constraints are UR-specific — those methods delegate
    to full 6-DOF :pymeth:`apply_absolute_ee`.
    """

    NAME = "fanuc_r2000ic"
    EE_LINK_INDEX = 8  # ``tool0`` child link (overwritten after load if needed)
    #: Compact HOME — EE clear of tire bore at shipping layout; re-tune via
    #: ``scripts/measure_fanuc_home.py`` when the scene layout changes.
    HOME_POSE = (0.0, -0.5, 0.4, 0.0, -0.9, 0.0)
    FINAL_LOCK_QUATERNION = (0.0, 0.0, 0.0, 1.0)
    TOOL_UP_QUATERNION = FINAL_LOCK_QUATERNION

    def __init__(self, client: int, cfg: EnvConfig):
        self._lock_tool_up = bool(
            getattr(cfg, "fanuc_lock_tool_up",
                    getattr(cfg, "ur10_lock_tool_up", False))
        )
        self._smooth_alpha = float(
            getattr(cfg, "fanuc_joint_target_smooth_alpha",
                    getattr(cfg, "ur10_joint_target_smooth_alpha", 1.0))
        )
        self._max_step = float(
            getattr(cfg, "fanuc_joint_max_step_rad",
                    getattr(cfg, "ur10_joint_max_step_rad", 0.0))
        )
        self._pgain = float(
            getattr(cfg, "fanuc_position_gain",
                    getattr(cfg, "ur10_position_gain", 1.0))
        )
        self._vgain = float(
            getattr(cfg, "fanuc_velocity_gain",
                    getattr(cfg, "ur10_velocity_gain", 1.0))
        )
        self._max_joint_vel = float(
            getattr(cfg, "fanuc_motor_max_velocity_rad_s",
                    getattr(cfg, "ur10_motor_max_velocity_rad_s", 1.0))
        )
        self._joint_slew_max = float(
            getattr(cfg, "fanuc_joint_slew_max_rad",
                    getattr(cfg, "ur10_joint_slew_max_rad", 0.08))
        )
        self._cmd_q: Optional[np.ndarray] = None

        urdf = str(getattr(cfg, "fanuc_urdf", ""))
        search = str(getattr(cfg, "fanuc_search_path", ""))
        if search:
            p.setAdditionalSearchPath(search, physicsClientId=client)
        ic_meshes = getattr(cfg, "fanuc_mesh_support_path", None)
        if ic_meshes:
            p.setAdditionalSearchPath(str(ic_meshes), physicsClientId=client)

        Robot.__init__(
            self,
            client=client,
            base_pos=cfg.robot_A_base_pos,
            base_orn=rpy_to_quat(cfg.robot_A_base_rpy),
            urdf_path=urdf,
            search_path=None,
        )
        self.EE_LINK_INDEX = self._link_index_for_child_link("tool0")
        self._cmd_q = None
        self.reset_to_home()
        home_pos, home_quat = self.ee_pose()
        self.FINAL_LOCK_QUATERNION = tuple(float(x) for x in home_quat)
        self.TOOL_UP_QUATERNION = self.FINAL_LOCK_QUATERNION
        print(
            f"[{self.NAME}] EE_LINK_INDEX={self.EE_LINK_INDEX}  "
            f"HOME tool0={tuple(round(v, 3) for v in home_pos)}"
        )

    def _link_index_for_child_link(self, link_name: str) -> int:
        for j in range(p.getNumJoints(self.uid, physicsClientId=self.client)):
            info = p.getJointInfo(self.uid, j, physicsClientId=self.client)
            child = info[12]
            name = child.decode() if isinstance(child, (bytes, bytearray)) else str(child)
            if name == link_name:
                return j
        raise RuntimeError(f"{self.NAME}: link '{link_name}' not found on URDF")

    def _arm_joint_indices(self) -> List[int]:
        return self._joints_by_name([f"joint_{i}" for i in range(1, 7)])

    def _arm_motor_forces(self) -> List[float]:
        return [2000.0, 2000.0, 1500.0, 400.0, 400.0, 200.0]

    def apply_palm_up_locked(self, pos) -> None:
        self.apply_absolute_ee(pos, self.FINAL_LOCK_QUATERNION)

    def apply_palm_up_pose(self, pos, target_orn, warm_q=None) -> None:
        orn = target_orn if target_orn is not None else self.FINAL_LOCK_QUATERNION
        self.apply_absolute_ee(pos, orn)

    def enforce_palm_up_wrists(self, tool_z_threshold: float = 0.999) -> None:
        return


class PandaRobot(Robot):
    NAME = "panda"
    EE_LINK_INDEX = 11  # panda_grasptarget
    # Hub-centric layout (base at (+0.40, −0.80, −0.22)): EE parked at
    # (+0.15, −0.30, 0.00) with **tool +Z aligned to world +Y** (R_x(−π/2)),
    # so the wrist already faces the hub bolts head-on with no 90° twist.
    # Phase 1 freeze pose: standard Franka "ready" config
    #   (0, -π/4, 0, -3π/4, 0, π/2, π/4)
    # EE is parked ~0.30 m forward (+X local = +X world) and ~0.49 m
    # above the panda base, *facing away* from Robot A. With Panda base
    # at (+1.20, -0.80, -0.22) the EE world pose is roughly
    # (+1.50, -0.80, +0.27) — completely outside Robot A's carry
    # envelope (X ∈ [-1.65, +0.15]) so it cannot interfere during
    # Phase 1 tire transport even if the freeze were lifted.
    # Phase 2/3 will re-calibrate via scripts/calibrate_home_pose.py
    # when Panda becomes active for bolt tightening.
    HOME_POSE = (0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854)

    def __init__(self, client: int, cfg: EnvConfig):
        # panda urdf is shipped with pybullet_data; ensure search path is set.
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        super().__init__(
            client=client,
            base_pos=cfg.robot_B_base_pos,
            base_orn=rpy_to_quat(cfg.robot_B_base_rpy),
            urdf_path=cfg.panda_urdf,
        )

    def _arm_joint_indices(self) -> List[int]:
        return self._joints_by_name([f"panda_joint{i}" for i in range(1, 8)])
