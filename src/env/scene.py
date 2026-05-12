"""Tire / hub / bolts scene: aggregated truck-wheel URDF or legacy primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from ..config import EnvConfig
from . import models
from .utils import quat_axis, rpy_to_quat

# PyBullet ``createVisualShapeArray`` rejects compounds above ~16 child shapes per body
# (collision arrays may accept more depending on build; keep totals aligned).
_PYB_COMPOUND_SHAPE_CAP = 16


@dataclass(frozen=True)
class BodyLinkRef:
    """Pose uses world link-frame position/orientation (``getLinkState`` [4], [5])."""

    uid: int
    link_index: int  # -1 = base / single-link body


def ref_world_pose(client: int, ref: BodyLinkRef) -> Tuple[np.ndarray, np.ndarray]:
    if ref.link_index < 0:
        pos, orn = p.getBasePositionAndOrientation(ref.uid, physicsClientId=client)
        return np.asarray(pos), np.asarray(orn)
    ls = p.getLinkState(
        ref.uid,
        ref.link_index,
        computeForwardKinematics=True,
        physicsClientId=client,
    )
    return np.asarray(ls[4], dtype=np.float64), np.asarray(ls[5], dtype=np.float64)


@dataclass
class SceneHandles:
    plane: int
    hub: BodyLinkRef
    tire: int
    bolts: List[BodyLinkRef]
    truck_uid: Optional[int]
    vehicle: Optional[int]
    target_bolt_idx: int

    @property
    def target_bolt(self) -> BodyLinkRef:
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
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client
        )
        plane = p.loadURDF("plane.urdf", physicsClientId=self.client)

        hub_pos = np.array(self.cfg.hub_pos_nominal, dtype=np.float64)
        hub_pos += self._sample_offset_xyz()
        hub_rpy = tuple(float(x) for x in self.cfg.hub_base_rpy)
        hub_orn = rpy_to_quat(hub_rpy)
        self._hub_orn_world = hub_orn

        vehicle_id: Optional[int] = None
        if self.cfg.spawn_vehicle_primitive_box:
            vehicle_id = self._make_vehicle_box(hub_pos)

        truck_uid: Optional[int] = None
        bolts: List[BodyLinkRef] = []

        if self.cfg.use_truck_hub_urdf:
            truck_uid = p.loadURDF(
                self.cfg.truck_wheel_station_urdf,
                basePosition=hub_pos.tolist(),
                baseOrientation=hub_orn.tolist(),
                useFixedBase=True,
                physicsClientId=self.client,
            )
            hub_ref = BodyLinkRef(truck_uid, -1)
            bolt_links = self._bolt_child_link_indices(truck_uid)
            bolts = [BodyLinkRef(truck_uid, li) for li in bolt_links]
        else:
            hub_uid = self._make_cylinder(
                radius=self.cfg.hub_radius,
                height=self.cfg.hub_thickness,
                mass=0.0,
                base_pos=hub_pos,
                base_orn=hub_orn,
                rgba=(0.4, 0.4, 0.45, 1.0),
            )
            hub_ref = BodyLinkRef(hub_uid, -1)
            bolts = self._make_bolts_primitive(hub_pos, hub_orn)

        if not bolts:
            raise RuntimeError("Scene built with zero bolts — check URDF or config.")

        target_idx = int(self.np_random.integers(0, len(bolts)))
        tb = bolts[target_idx]
        p.changeVisualShape(
            tb.uid,
            tb.link_index,
            rgbaColor=(0.95, 0.85, 0.1, 1.0),
            physicsClientId=self.client,
        )

        for bref in bolts:
            p.changeDynamics(
                bref.uid,
                bref.link_index,
                lateralFriction=self.cfg.bolt_lateral_friction,
                spinningFriction=self.cfg.bolt_spinning_friction,
                physicsClientId=self.client,
            )

        tire = self._spawn_tire()

        self.handles = SceneHandles(
            plane=plane,
            hub=hub_ref,
            tire=tire,
            bolts=bolts,
            truck_uid=truck_uid,
            vehicle=vehicle_id,
            target_bolt_idx=target_idx,
        )
        return self.handles

    def _bolt_child_link_indices(self, uid: int) -> List[int]:
        """Ordered child link indices for joints named ``bolt_*`` on the truck URDF."""
        n_j = p.getNumJoints(uid, physicsClientId=self.client)
        pairs: List[Tuple[int, int]] = []
        for ji in range(n_j):
            info = p.getJointInfo(uid, ji, physicsClientId=self.client)
            # Index 12 is the child link name; for this URDF, linkIndex == jointIndex ji.
            raw = info[12]
            name = raw.decode("utf8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            if not name.startswith("bolt"):
                continue
            try:
                k = int(name.split("_", 1)[1])
            except (ValueError, IndexError):
                k = ji
            pairs.append((k, ji))
        pairs.sort(key=lambda t: t[0])
        return [c for _, c in pairs]

    # ------------------------------------------------------------------
    # Geometry queries used by reward / observation
    # ------------------------------------------------------------------
    def hub_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.handles is not None
        return ref_world_pose(self.client, self.handles.hub)

    def tire_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.handles is not None
        pos, orn = p.getBasePositionAndOrientation(
            self.handles.tire, physicsClientId=self.client
        )
        return np.asarray(pos), np.asarray(orn)

    def bolt_pose(self, idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        assert self.handles is not None
        bidx = self.handles.target_bolt_idx if idx is None else idx
        return ref_world_pose(self.client, self.handles.bolts[bidx])

    def hub_axis(self) -> np.ndarray:
        _, orn = self.hub_pose()
        return quat_axis(orn, "z")

    def tire_axis(self) -> np.ndarray:
        _, orn = self.tire_pose()
        return quat_axis(orn, "z")

    def bolt_axis(self, idx: Optional[int] = None) -> np.ndarray:
        _, orn = self.bolt_pose(idx)
        return quat_axis(orn, "z")

    def tire_hub_mount_residuals(self) -> tuple[float, float, float]:
        """`(axial_dot, lateral_norm, lug_spin_err_rad)` for mating diagnostics.

        * **axial** — `(t − h)·û_hub` (signed slide along flange normal).
        * **lateral** — perpendicular distance hubs↔ tire center (√ of Gram).
        * **lug_spin_err_rad** — minimal |rotation| aligning first lug-gap ray to bolt_0 ray
          about û_hub modulo ``2π/n_bolts``.
        """
        assert self.handles is not None
        ht = np.asarray(self.hub_pose()[0], dtype=np.float64)
        tt = np.asarray(self.tire_pose()[0], dtype=np.float64)
        ah = self.hub_axis()
        nh = np.linalg.norm(ah)
        if nh < 1e-9:
            return 0.0, float(np.linalg.norm(tt - ht)), 0.0
        ah /= nh
        dh = tt - ht
        axial = float(np.dot(dh, ah))
        lateral = float(np.linalg.norm(dh - axial * ah))

        nbolt = max(3, int(self.cfg.n_bolts))
        period = 2.0 * math.pi / float(nbolt)
        b0, _ = ref_world_pose(self.client, self.handles.bolts[0])
        vb = np.asarray(b0, dtype=np.float64) - ht
        eb_p = vb - ah * np.dot(vb, ah)
        eb_n = np.linalg.norm(eb_p)
        style = (self.cfg.tire_wheel_disk_style or "three_piece").lower()
        if style == "three_piece":
            phi0 = float(self.cfg.wheel_disk_bolt_phase_rad)
        else:
            phi0 = float(self.cfg.wheel_disk_bolt_phase_rad + period * 0.5)
        t_orn = self.tire_pose()[1]
        R = np.array(
            p.getMatrixFromQuaternion(list(t_orn)), dtype=np.float64
        ).reshape(3, 3)
        lh_local = np.array([
            math.cos(phi0),
            math.sin(phi0),
            0.0,
        ])
        lh_w = R @ lh_local
        eu_p = lh_w - ah * np.dot(lh_w, ah)
        eu_n = np.linalg.norm(eu_p)

        lug_err = 0.0
        if eb_n >= 1e-6 and eu_n >= 1e-6:
            eb_u = eb_p / eb_n
            eu_u = eu_p / eu_n
            crs = np.cross(eb_u, eu_u)
            sin_t = float(np.dot(ah, crs))
            cos_t = float(np.dot(eb_u, eu_u))
            phi = math.atan2(sin_t, cos_t)
            folded = phi - period * round(phi / period)
            lug_err = abs(float(folded))

        return axial, lateral, lug_err

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _make_cylinder(self, radius: float, height: float, mass: float,
                       base_pos, base_orn, rgba) -> int:
        col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.client,
        )
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=rgba,
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(base_pos),
            baseOrientation=list(base_orn),
            physicsClientId=self.client,
        )

    def _make_bolts_primitive(self, hub_pos, hub_orn) -> List[BodyLinkRef]:
        bolts: List[BodyLinkRef] = []
        n = self.cfg.n_bolts
        rc = self.cfg.bolt_circle_radius
        bolt_h = self.cfg.bolt_length
        bolt_r = self.cfg.bolt_radius

        R = np.array(p.getMatrixFromQuaternion(list(hub_orn))).reshape(3, 3)
        local_x_world = R[:, 0]
        local_y_world = R[:, 1]
        local_z_world = R[:, 2]

        face_offset = self.cfg.hub_thickness / 2 + bolt_h / 2

        for k in range(n):
            theta = 2 * np.pi * k / n
            offset_in_face = rc * (
                np.cos(theta) * local_x_world + np.sin(theta) * local_y_world
            )
            bolt_pos = np.asarray(hub_pos) + offset_in_face + face_offset * local_z_world
            uid = self._make_cylinder(
                radius=bolt_r,
                height=bolt_h,
                mass=0.0,
                base_pos=bolt_pos,
                base_orn=hub_orn,
                rgba=(0.7, 0.7, 0.7, 1.0),
            )
            bolts.append(BodyLinkRef(uid, -1))
        return bolts

    def _make_vehicle_box(self, hub_center: np.ndarray) -> int:
        he = np.asarray(self.cfg.vehicle_half_extents, dtype=np.float64)
        nom = np.asarray(self.cfg.hub_pos_nominal, dtype=np.float64)
        drift = hub_center - nom
        pos = np.asarray(self.cfg.vehicle_center_world, dtype=np.float64) + drift

        if not self.cfg.cargo_use_wheel_well_cutout:
            return self._make_vehicle_box_solid(pos, he)

        cells = self._cargo_keep_cells(hub_center, pos, he)
        if not cells:
            return self._make_vehicle_box_solid(pos, he)

        half_extents: list[list[float]] = []
        positions: list[list[float]] = []
        orientations: list[list[float]] = []
        rgba = (0.25, 0.35, 0.5, 0.35)
        rgba_list = [list(rgba) for _ in cells]
        for half, off in cells:
            half_extents.append(half)
            positions.append(off.tolist())
            orientations.append([0.0, 0.0, 0.0, 1.0])

        n = len(cells)
        col = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_BOX] * n,
            halfExtents=half_extents,
            collisionFramePositions=positions,
            collisionFrameOrientations=orientations,
            physicsClientId=self.client,
        )
        vis = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * n,
            halfExtents=half_extents,
            rgbaColors=rgba_list,
            visualFramePositions=positions,
            visualFrameOrientations=orientations,
            physicsClientId=self.client,
        )
        if col < 0 or vis < 0:
            return self._make_vehicle_box_solid(pos, he)
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos.tolist(),
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )

    def _make_vehicle_box_solid(self, pos: np.ndarray, he: np.ndarray) -> int:
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=he.tolist(),
            physicsClientId=self.client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=he.tolist(),
            rgbaColor=(0.25, 0.35, 0.5, 0.35),
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos.tolist(),
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )

    def _cargo_keep_cells(
        self, hub_center: np.ndarray, cargo_center: np.ndarray, he: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (half_extents, offset_from_cargo_center) for sub-boxes outside wheel well."""
        hx, hy, hz = float(he[0]), float(he[1]), float(he[2])
        cx, cy, cz = float(cargo_center[0]), float(cargo_center[1]), float(cargo_center[2])
        xmin, xmax = cx - hx, cx + hx
        ymin, ymax = cy - hy, cy + hy
        zmin, zmax = cz - hz, cz + hz

        nx, ny, nz = self.cfg.cargo_collision_subdiv
        dx = (xmax - xmin) / float(nx)
        dy = (ymax - ymin) / float(ny)
        dz = (zmax - zmin) / float(nz)

        x_lo = float(hub_center[0]) + float(self.cfg.cargo_wheel_well_x_range_from_hub[0])
        x_hi = float(hub_center[0]) + float(self.cfg.cargo_wheel_well_x_range_from_hub[1])
        R = float(self.cfg.cargo_wheel_well_radius_yz)
        r_sq = R * R
        hy0, hz0 = float(hub_center[1]), float(hub_center[2])

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x0 = xmin + ix * dx
                    y0 = ymin + iy * dy
                    z0 = zmin + iz * dz
                    pcx = x0 + 0.5 * dx
                    pcy = y0 + 0.5 * dy
                    pcz = z0 + 0.5 * dz
                    off = np.array([pcx - cx, pcy - cy, pcz - cz], dtype=np.float64)
                    half = np.array([0.5 * dx, 0.5 * dy, 0.5 * dz], dtype=np.float64)
                    if self._cargo_cell_in_wheel_well(
                        pcx, pcy, pcz, x_lo, x_hi, hy0, hz0, r_sq,
                    ):
                        continue
                    out.append((half, off))
        return out

    @staticmethod
    def _cargo_cell_in_wheel_well(
        pcx: float, pcy: float, pcz: float,
        x_lo: float, x_hi: float, hy0: float, hz0: float, r_sq: float,
    ) -> bool:
        if not (x_lo <= pcx <= x_hi):
            return False
        dy = pcy - hy0
        dz = pcz - hz0
        return (dy * dy + dz * dz) < r_sq

    def _spawn_tire(self) -> int:
        base_pos_A = np.asarray(self.cfg.robot_A_base_pos, dtype=np.float64)
        tire_pos = base_pos_A + np.asarray(self.cfg.tire_spawn_offset_from_robot_a,
                                              dtype=np.float64)
        tire_orn = rpy_to_quat([0.0, -np.pi / 2, 0.0])
        return models.create_tire_wheel_multibody(
            self.client,
            self.cfg,
            base_position=tire_pos.tolist(),
            base_orientation=tire_orn.tolist(),
            visual_primitive_cap=_PYB_COMPOUND_SHAPE_CAP,
        )

    def _sample_offset_xyz(self) -> np.ndarray:
        ranges_cm = self.cfg.curriculum.phase_ranges_cm
        idx = max(0, min(len(ranges_cm) - 1, self.cfg.curriculum.phase - 1))
        r_m = ranges_cm[idx] / 100.0
        if r_m <= 0:
            return np.zeros(3)
        return self.np_random.uniform(-r_m, r_m, size=3)
