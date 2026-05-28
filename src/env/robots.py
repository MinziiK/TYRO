"""Thin wrappers around UR10 and Franka Panda for delta-EE-pose control via IK.

Each Robot exposes (joint_pos, joint_vel, ee_pos, ee_quat) for observation and
takes a `(Δpos[3], Δrpy[3])` command per control step (spec §3). IK is computed
once per control step against the current EE pose; joint targets drive position
servos which the env then steps through `decimation` sim sub-steps.
"""
from __future__ import annotations

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
    # Dual-block-rack layout (hub at origin, base at (−0.40, −0.80, −0.62),
    # pickup COM at (−1.50, −0.80, −0.145), 6 o'clock outer point at
    # (−1.50, −0.80, −0.67)):
    #
    # HOME places the EE at (−1.50, −0.80, −0.05) — 62 cm directly above
    # the grasp target, centred in the 20 cm-wide X gap between the two
    # support blocks. Tool +Z is aligned to world +Z to within 0.00°
    # (EE RPY = (0°, 0°, −90°)), so the gripper cup points at the sky.
    # The Stage-0 task is reduced to a pure −Z descent through the gap;
    # no wrist twist is needed to capture the 6 o'clock tread point.
    #
    # The 62 cm clearance exceeds ``approach_radius_tol`` (60 cm) so the
    # FSM does *not* trigger Stage 0 → 1 on env-step 0. wrist_3 = +0.2327
    # is the (−6.0505 mod 2π) wrap of the IK output, equivalent under
    # joint kinematics but kept inside [−π, π] for readability.
    #
    # Re-run scripts/calibrate_home_pose.py to refresh after any change
    # to the base position, rack geometry, or approach radius tolerance.
    HOME_POSE = (2.9089, -0.0246, -0.5518, -2.5652, 0.2327, 0.0000)

    def __init__(self, client: int, cfg: EnvConfig):
        super().__init__(
            client=client,
            base_pos=cfg.robot_A_base_pos,
            base_orn=rpy_to_quat(cfg.robot_A_base_rpy),
            urdf_path=cfg.ur10_urdf,
            search_path=cfg.ur10_search_path,
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
