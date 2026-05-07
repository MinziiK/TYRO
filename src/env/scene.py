"""Tire / hub / bolts scene built from primitives (no external mesh).

Hub is a fixed disk with N bolts protruding along its local +z. The tire is a
free-falling cylinder that Robot A grasps via a fixed constraint at reset.
A target bolt is selected per episode and exposed to the env for reward calc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from ..config import EnvConfig
from .utils import quat_axis, rpy_to_quat


@dataclass
class SceneHandles:
    """Body / link ids returned by Scene.build()."""
    plane: int
    hub: int
    tire: int
    bolts: List[int]
    target_bolt_idx: int

    @property
    def target_bolt(self) -> int:
        return self.bolts[self.target_bolt_idx]


class Scene:
    """Manages tire + hub + bolt creation, randomization, and queries."""

    def __init__(self, client: int, cfg: EnvConfig, np_random: np.random.Generator):
        self.client = client
        self.cfg = cfg
        self.np_random = np_random
        self.handles: Optional[SceneHandles] = None
        self._hub_orn_world: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    def build(self) -> SceneHandles:
        """Load plane, then create hub + bolts + tire from primitives."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        plane = p.loadURDF("plane.urdf", physicsClientId=self.client)

        hub_pos = np.array(self.cfg.hub_pos_nominal, dtype=np.float64)
        hub_pos += self._sample_offset_xyz()  # curriculum randomization
        # Hub local-z aligned with world +x: rotate -90° about world-y.
        hub_orn = rpy_to_quat([0.0, -np.pi / 2, 0.0])
        self._hub_orn_world = hub_orn

        hub = self._make_cylinder(
            radius=self.cfg.hub_radius,
            height=self.cfg.hub_thickness,
            mass=0.0,
            base_pos=hub_pos,
            base_orn=hub_orn,
            rgba=(0.4, 0.4, 0.45, 1.0),
        )

        bolts = self._make_bolts(hub_pos, hub_orn)
        target_idx = int(self.np_random.integers(0, self.cfg.n_bolts))
        # Recolor the target so a human watching can see the goal.
        p.changeVisualShape(bolts[target_idx], -1, rgbaColor=(0.95, 0.85, 0.1, 1.0),
                            physicsClientId=self.client)

        tire = self._spawn_tire()

        self.handles = SceneHandles(plane=plane, hub=hub, tire=tire,
                                    bolts=bolts, target_bolt_idx=target_idx)
        return self.handles

    # ------------------------------------------------------------------
    # Geometry queries used by reward / observation
    # ------------------------------------------------------------------
    def hub_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos, orn = p.getBasePositionAndOrientation(self.handles.hub,
                                                   physicsClientId=self.client)
        return np.asarray(pos), np.asarray(orn)

    def tire_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos, orn = p.getBasePositionAndOrientation(self.handles.tire,
                                                   physicsClientId=self.client)
        return np.asarray(pos), np.asarray(orn)

    def bolt_pose(self, idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        bidx = self.handles.target_bolt_idx if idx is None else idx
        pos, orn = p.getBasePositionAndOrientation(self.handles.bolts[bidx],
                                                   physicsClientId=self.client)
        return np.asarray(pos), np.asarray(orn)

    def hub_axis(self) -> np.ndarray:
        """Unit vector along the hub face normal (local +z, world frame)."""
        _, orn = self.hub_pose()
        return quat_axis(orn, "z")

    def tire_axis(self) -> np.ndarray:
        """Unit vector along the tire rotational axis (local +z, world frame)."""
        _, orn = self.tire_pose()
        return quat_axis(orn, "z")

    def bolt_axis(self, idx: Optional[int] = None) -> np.ndarray:
        _, orn = self.bolt_pose(idx)
        return quat_axis(orn, "z")

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _make_cylinder(self, radius: float, height: float, mass: float,
                       base_pos, base_orn, rgba) -> int:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                     physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                  rgbaColor=rgba, physicsClientId=self.client)
        return p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=list(base_pos),
                                 baseOrientation=list(base_orn),
                                 physicsClientId=self.client)

    def _make_bolts(self, hub_pos, hub_orn) -> List[int]:
        bolts: List[int] = []
        n = self.cfg.n_bolts
        rc = self.cfg.bolt_circle_radius
        bolt_h = self.cfg.bolt_length
        bolt_r = self.cfg.bolt_radius

        # Hub local frame axes in world.
        # Hub is rotated so local-z points along world +x (bolt protrusion direction).
        R = np.array(p.getMatrixFromQuaternion(list(hub_orn))).reshape(3, 3)
        local_x_world = R[:, 0]
        local_y_world = R[:, 1]
        local_z_world = R[:, 2]

        # Bolts protrude half their length past the hub face.
        face_offset = self.cfg.hub_thickness / 2 + bolt_h / 2

        for k in range(n):
            theta = 2 * np.pi * k / n
            offset_in_face = rc * (np.cos(theta) * local_x_world
                                   + np.sin(theta) * local_y_world)
            bolt_pos = np.asarray(hub_pos) + offset_in_face + face_offset * local_z_world
            bolts.append(self._make_cylinder(
                radius=bolt_r, height=bolt_h, mass=0.0,
                base_pos=bolt_pos, base_orn=hub_orn,
                rgba=(0.7, 0.7, 0.7, 1.0),
            ))
        return bolts

    def _spawn_tire(self) -> int:
        """Place the tire roughly in front of Robot A so its gripper can grab it."""
        # World position: in front of Robot A's nominal reach, tire axis = world +x.
        base_pos_A = np.asarray(self.cfg.robot_A_base_pos, dtype=np.float64)
        tire_pos = base_pos_A + np.array([-0.4, 0.0, 0.55])
        tire_orn = rpy_to_quat([0.0, -np.pi / 2, 0.0])

        col = p.createCollisionShape(p.GEOM_CYLINDER,
                                     radius=self.cfg.tire_outer_radius,
                                     height=self.cfg.tire_thickness,
                                     physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=self.cfg.tire_outer_radius,
                                  length=self.cfg.tire_thickness,
                                  rgbaColor=(0.1, 0.1, 0.1, 1.0),
                                  physicsClientId=self.client)
        tire = p.createMultiBody(baseMass=5.0,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=list(tire_pos),
                                 baseOrientation=list(tire_orn),
                                 physicsClientId=self.client)
        # Damping so the tire doesn't oscillate when held by the constraint.
        p.changeDynamics(tire, -1, linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=self.client)
        return tire

    def _sample_offset_xyz(self) -> np.ndarray:
        """Curriculum-aware random ±r offset on the hub position (spec §6)."""
        ranges_cm = self.cfg.curriculum.phase_ranges_cm
        idx = max(0, min(len(ranges_cm) - 1, self.cfg.curriculum.phase - 1))
        r_m = ranges_cm[idx] / 100.0
        if r_m <= 0:
            return np.zeros(3)
        return self.np_random.uniform(-r_m, r_m, size=3)
