"""TyroEnv — Gymnasium env implementing the Tier-1 task in `tyro_design_spec.md`.

Two robots, one centralized policy:
  - Robot A (UR10) holds a tire and must align its axis to the truck hub axis.
  - Robot B (Franka Panda) must reach the target bolt and orient its gripper z
    along the bolt axis.

Action: 13-d in [-1, 1]  → (Δpose_A 6, Δpose_B 6, gripper_A 1)
Observation: 86-d (see spec §2.1)
Reward: sum of alignment / reach / cooperation / success / penalties (spec §4)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
        self._grasp_constraint: Optional[int] = None
        self._step_count: int = 0
        self._prev_action: np.ndarray = np.zeros(self.cfg.action.dim, dtype=np.float32)
        self._prev_d_A: Optional[float] = None
        self._prev_d_B: Optional[float] = None

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
        p.setGravity(*self.cfg.gravity, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.cfg.sim_freq_hz, physicsClientId=self.client)

        # Build scene first; robots are placed independently of the scene origin.
        self.scene = Scene(self.client, self.cfg, self._np_random)
        self.handles = self.scene.build()

        self.robot_A = UR10Robot(self.client, self.cfg)
        self.robot_B = PandaRobot(self.client, self.cfg)

        # Settle a couple of physics steps so the IK warm start is sane.
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)

        # Snap the tire to Robot A's EE and bind it with a fixed constraint —
        # this is the Tier-1 simplification (spec §1: alignment task, gripping
        # not part of training scope).
        self._attach_tire_to_robot_A()

        self._step_count = 0
        self._prev_action = np.zeros(self.cfg.action.dim, dtype=np.float32)
        self._prev_d_A = None
        self._prev_d_B = None

        obs = self._compute_obs()
        info = {"target_bolt_idx": self.handles.target_bolt_idx}
        return obs, info

    def _attach_tire_to_robot_A(self) -> None:
        ee_pos, ee_orn = self.robot_A.ee_pose()
        # Move tire so its center sits at the EE, axis aligned with EE local-z.
        tire_orn = ee_orn  # share orientation
        p.resetBasePositionAndOrientation(self.handles.tire,
                                          ee_pos.tolist(), tire_orn.tolist(),
                                          physicsClientId=self.client)
        if self._grasp_constraint is not None:
            try:
                p.removeConstraint(self._grasp_constraint, physicsClientId=self.client)
            except p.error:
                pass
        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_A.uid,
            parentLinkIndex=self.robot_A.EE_LINK_INDEX,
            childBodyUniqueId=self.handles.tire,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client,
        )

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
        obs = self._compute_obs()
        reward, breakdown = self._compute_reward(action)
        terminated, truncated, term_info = self._check_termination(breakdown)

        info: Dict[str, Any] = {
            "reward_terms": breakdown.__dict__,
            "step": self._step_count,
            **term_info,
        }
        self._prev_action = action.copy()
        self._prev_d_A = breakdown.d_A
        self._prev_d_B = breakdown.d_B
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray) -> None:
        ps = self.cfg.action.pos_scale
        rs = self.cfg.action.rot_scale
        # Robot A: 6
        d_pos_A = action[0:3] * ps
        d_rot_A = action[3:6] * rs
        self.robot_A.apply_delta_ee(d_pos_A, d_rot_A)
        # Robot B: 6
        d_pos_B = action[6:9] * ps
        d_rot_B = action[9:12] * rs
        self.robot_B.apply_delta_ee(d_pos_B, d_rot_B)
        # Gripper A (Tier-1: ignored at sim level, the constraint holds the tire).
        # Bit kept in obs as previous_action so the policy can still output it.

    # ------------------------------------------------------------------
    # Observation (spec §2.1: 86-d)
    # ------------------------------------------------------------------
    def _compute_obs(self) -> np.ndarray:
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

        # Relative tire→hub (position diff + axis-angle rotation error)
        rel_th_pos = tire_pos - hub_pos
        rel_th_rot = relative_axisangle(tire_orn, hub_orn)
        # Relative gripperB→bolt
        rel_eb_pos = eeB_pos - bolt_pos
        rel_eb_rot = relative_axisangle(eeB_orn, bolt_orn)

        obs = np.concatenate([
            qA_n, dqA_n,                          # 12
            qB_n, dqB_n,                          # 14
            eeA_pos / ws, eeA_orn,                # 7
            eeB_pos / ws, eeB_orn,                # 7
            tire_pos / ws, tire_orn,              # 7
            hub_pos / ws, hub_orn,                # 7
            bolt_pos / ws, bolt_orn,              # 7
            rel_th_pos / ws, rel_th_rot / np.pi,  # 6
            rel_eb_pos / ws, rel_eb_rot / np.pi,  # 6
            self._prev_action.astype(np.float64),  # 13
        ]).astype(np.float32)

        assert obs.shape[0] == obs_cfg.dim, f"obs dim {obs.shape[0]} != {obs_cfg.dim}"
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray
                        ) -> Tuple[float, rewards.RewardBreakdown]:
        rcfg = self.cfg.reward
        b = rewards.RewardBreakdown()

        tire_pos, _ = self.scene.tire_pose()
        hub_pos, _ = self.scene.hub_pose()
        tire_axis = self.scene.tire_axis()
        hub_axis = self.scene.hub_axis()
        eeB_pos, eeB_orn = self.robot_B.ee_pose()
        bolt_pos, _ = self.scene.bolt_pose()
        bolt_axis = self.scene.bolt_axis()
        eeB_z = quat_axis(eeB_orn, "z")

        b.align_A, b.d_A, b.theta_A = rewards.align_reward(
            tire_pos, hub_pos, tire_axis, hub_axis, rcfg)
        b.reach_B, b.d_B, b.theta_B = rewards.reach_reward(
            eeB_pos, bolt_pos, eeB_z, bolt_axis, rcfg)
        b.coop = rewards.coop_reward(b.d_A, b.d_B, rcfg)
        b.success, b.is_success = rewards.success_bonus(b.d_A, b.theta_A, b.d_B, rcfg)
        b.collision = rewards.collision_penalty(self._in_bad_collision(), rcfg)
        b.action = rewards.action_penalty(action, rcfg)
        b.jerk = rewards.jerk_penalty(action, self._prev_action, rcfg)
        b.shape_A = rewards.shaping_reward(self._prev_d_A, b.d_A, rcfg.w_shape_A)
        b.shape_B = rewards.shaping_reward(self._prev_d_B, b.d_B, rcfg.w_shape_B)

        b.total = rewards.aggregate(b.__dict__, use_shaping=self.cfg.use_shaping)
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
        return False

    def _check_termination(self, b: rewards.RewardBreakdown
                           ) -> Tuple[bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {"is_success": b.is_success}
        if b.is_success:
            return True, False, {**info, "termination": "success"}
        if b.collision < 0:
            return True, False, {**info, "termination": "collision"}
        if self._out_of_workspace():
            return True, False, {**info, "termination": "workspace"}
        if self._step_count >= self.cfg.max_steps:
            return False, True, {**info, "termination": "max_steps"}
        return False, False, info

    def _out_of_workspace(self) -> bool:
        ws = self.cfg.obs.workspace_radius
        for getter in (self.robot_A.ee_pose, self.robot_B.ee_pose,
                       self.scene.tire_pose):
            pos, _ = getter()
            if np.linalg.norm(pos[:2]) > ws * 1.5 or pos[2] > 2.5 or pos[2] < -0.05:
                return True
        return False

    # ------------------------------------------------------------------
    # Curriculum hook (env consumers call this between rollouts)
    # ------------------------------------------------------------------
    def set_phase(self, phase: int) -> None:
        self.cfg.curriculum.phase = int(phase)
