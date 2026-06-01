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
from .utils import quat_multiply, rpy_to_quat, axisangle3_to_quat


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

    def __init__(self, client: int, cfg: EnvConfig):
        self._lock_tool_up = bool(getattr(cfg, "ur10_lock_tool_up", True))
        super().__init__(
            client=client,
            base_pos=cfg.robot_A_base_pos,
            base_orn=rpy_to_quat(cfg.robot_A_base_rpy),
            urdf_path=cfg.ur10_urdf,
            search_path=cfg.ur10_search_path,
        )

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
        forces = [400.0, 400.0, 300.0, 60.0, 60.0, 60.0]
        # ``positionGain`` restored to 1.0 — combined with the new
        # force caps it gives stiff hold without the 21 cm steady-state
        # sag that 0.3 produced. The drift-free target accumulator
        # above means there is no longer a positive feedback loop
        # between sag and IK target, so we can afford the stiffer PD.
        pgains = [1.0] * len(forces)
        vgains = [1.0] * len(forces)
        p.setJointMotorControlArray(
            self.uid, self.arm.indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=arm_targets.tolist(),
            forces=forces,
            positionGains=pgains,
            velocityGains=vgains,
            physicsClientId=self.client,
        )

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
        # UR10 joint torque caps + PD — same tuning as ``apply_delta_ee``.
        forces = [400.0, 400.0, 300.0, 60.0, 60.0, 60.0]
        pgains = [1.0] * len(forces)
        vgains = [1.0] * len(forces)
        p.setJointMotorControlArray(
            self.uid, self.arm.indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=arm_targets.tolist(),
            forces=forces,
            positionGains=pgains,
            velocityGains=vgains,
            physicsClientId=self.client,
        )

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
