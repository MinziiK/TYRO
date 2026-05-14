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
        # order PyBullet enumerates them. Pick out the arm slots.
        ik = np.asarray(ik, dtype=np.float64)
        arm_targets = ik[: self.arm.n] if len(ik) >= self.arm.n else self._fallback_targets()
        # Clamp to limits before sending.
        arm_targets = np.clip(arm_targets, self.arm.lower, self.arm.upper)

        p.setJointMotorControlArray(
            self.uid, self.arm.indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=arm_targets.tolist(),
            forces=[150.0] * self.arm.n,
            physicsClientId=self.client,
        )

    def _fallback_targets(self) -> np.ndarray:
        q, _ = self.joint_state()
        return q


class UR10Robot(Robot):
    NAME = "ur10"
    EE_LINK_INDEX = 7  # robot_ee_link
    HOME_POSE = (0.0, -1.2, 1.4, -1.7, -1.57, 0.0)

    def __init__(self, client: int, cfg: EnvConfig):
        super().__init__(
            client=client,
            base_pos=cfg.robot_A_base_pos,
            base_orn=rpy_to_quat(cfg.robot_A_base_rpy),
            urdf_path=cfg.ur10_urdf,
            search_path=cfg.ur10_search_path,
        )

    def _arm_joint_indices(self) -> List[int]:
        # joints 1..6 from URDF inspection; verify by joint type.
        return [1, 2, 3, 4, 5, 6]


class PandaRobot(Robot):
    NAME = "panda"
    EE_LINK_INDEX = 11  # panda_grasptarget
    HOME_POSE = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

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
        return [0, 1, 2, 3, 4, 5, 6]
