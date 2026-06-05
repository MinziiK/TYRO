"""Tire / hub / bolts scene: aggregated truck-wheel URDF or legacy primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    #: Body UIDs of the two static support blocks that cradle the
    #: vertical tire during Stage 0 (front + rear). Empty when
    #: ``spawn_tire_rack`` is False.
    tire_rack: List[int] = field(default_factory=list)
    #: **2026-06-02 (cargo penetration fix)** — UID of the static thin slab
    #: behind the hub (``cfg.spawn_cargo_back_wall``). Previously the
    #: ``_make_cargo_back_wall`` return value was discarded, which left the
    #: body in the simulation but invisible to ``_in_bad_collision`` /
    #: ``_max_contact_normal_force``. Storing it here so the env can include
    #: it in collision checks (both for the robot arms and for the
    #: kinematically-driven tire). ``None`` when the back wall is disabled.
    cargo_back_wall: Optional[int] = None
    #: **2026-06-05 (real floor pit)** — UIDs of the four rim slabs that
    #: rebuild the normal-height floor *outside* the pit rectangle when
    #: ``cfg.floor_pit_enable`` is True. Empty when the pit is disabled (the
    #: floor is then a single infinite plane). These are solid bodies the arm
    #: links must not punch through, so the env includes them in its
    #: robot-vs-floor bad-collision check alongside ``plane``.
    floor_rim: List[int] = field(default_factory=list)

    @property
    def target_bolt(self) -> BodyLinkRef:
        return self.bolts[self.target_bolt_idx]


class Scene:
    """Manages tire + hub + bolt creation, randomization, and queries."""

    def __init__(
        self,
        client: int,
        cfg: EnvConfig,
        np_random: np.random.Generator,
        *,
        hub_xy_offset: Tuple[float, float] = (0.0, 0.0),
        cargo_xy_offset: Tuple[float, float] = (0.0, 0.0),
    ):
        self.client = client
        self.cfg = cfg
        self.np_random = np_random
        self.handles: Optional[SceneHandles] = None
        self._hub_orn_world: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        # ``False`` after a tire-build fallback — disables lug-spin residual.
        self.has_wheel_disk: bool = True
        # Static-pose domain randomization offsets (XY only). Z stays
        # pinned to the nominal config value so the tire never floats off
        # contact or clips the floor.  ``TyroEnv._maybe_apply_domain_-
        # randomization`` samples these once per ``reset()`` and forwards
        # them here; when DR is off they default to (0, 0) and the build
        # path is bit-identical to the pre-DR scene.
        self._hub_xy_offset: np.ndarray = np.array(
            [float(hub_xy_offset[0]), float(hub_xy_offset[1]), 0.0],
            dtype=np.float64,
        )
        self._cargo_xy_offset: np.ndarray = np.array(
            [float(cargo_xy_offset[0]), float(cargo_xy_offset[1]), 0.0],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    def build(self) -> SceneHandles:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client
        )
        floor_z = float(getattr(self.cfg, "floor_z", 0.0))
        pit_enable = bool(getattr(self.cfg, "floor_pit_enable", False))
        floor_rim_uids: List[int] = []
        if pit_enable:
            # Real pit: drop the infinite plane to the pit BOTTOM and rebuild
            # the floor surface (top at floor_z) only OUTSIDE the pit rectangle
            # as four rim slabs, leaving a genuine rectangular hole.
            pit_depth = float(getattr(self.cfg, "floor_pit_depth", 0.90))
            plane = p.loadURDF(
                "plane.urdf",
                basePosition=[0.0, 0.0, floor_z - pit_depth],
                physicsClientId=self.client,
            )
            shape = str(getattr(self.cfg, "floor_pit_shape", "rect")).lower()
            if shape == "circle":
                floor_rim_uids = [self._make_floor_with_circular_hole(floor_z)]
            else:
                floor_rim_uids = self._make_floor_rim(floor_z)
        else:
            plane = p.loadURDF(
                "plane.urdf",
                basePosition=[0.0, 0.0, floor_z],
                physicsClientId=self.client,
            )

        hub_pos = np.array(self.cfg.hub_pos_nominal, dtype=np.float64)
        # ``_sample_offset_xyz`` = the legacy curriculum-phase XY noise
        # (config.curriculum.phase_ranges_cm). ``_hub_xy_offset`` = the
        # new static-pose DR (cfg.USE_DOMAIN_RANDOMIZATION). The two
        # paths are additive; both default to zero so behaviour is
        # unchanged unless explicitly enabled.
        hub_pos += self._sample_offset_xyz()
        hub_pos += self._hub_xy_offset
        hub_rpy = tuple(float(x) for x in self.cfg.hub_base_rpy)
        hub_orn = rpy_to_quat(hub_rpy)
        self._hub_orn_world = hub_orn

        vehicle_id: Optional[int] = None
        if self.cfg.spawn_vehicle_primitive_box:
            vehicle_id = self._make_vehicle_box(hub_pos)
        # **2026-06-01 (cargo back wall)** — independent thin slab placed
        # at ``hub_y + cargo_back_wall_y_offset`` (default 0.18 m, just
        # past a fully-mounted tire's far face at hub_y + 0.15). Stops
        # the policy from pushing the tire all the way through the hub
        # into the cargo interior. Implemented as a separate static
        # body to avoid the PyBullet compound-primitive count limit
        # we hit when subdividing the cargo body itself.
        cargo_back_wall_uid: Optional[int] = None
        if (
            self.cfg.spawn_vehicle_primitive_box
            and bool(getattr(self.cfg, "spawn_cargo_back_wall", False))
        ):
            cargo_back_wall_uid = self._make_cargo_back_wall(hub_pos)

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

        # Inline (Y-split) dual-block tire rack — must be created *before*
        # ``_spawn_tire`` so the tire's static world-pin (mass=0 at
        # COM = rack_top + R) already sees the cradle rails underneath it
        # on step 1. The 10 cm Y-gap between the rails is what makes
        # Stage 0 grasp geometrically legal: gripper threads along the tire
        # bore axis (layout-dependent Y centerline) to the 6 o'clock outer
        # point without colliding with either rail.
        rack_uids = self._make_split_tire_rack()

        tire = self._spawn_tire()

        # Visual support pillars for any robot base mounted above the floor
        # (e.g. the sandwich layout where UR10@Z=0.20 and Panda@Z=0.60 both
        # sit on plinths). Each call is a no-op when its base z≈0 or the
        # corresponding ``*_stand_radius`` config is non-positive.
        self._make_ur10_stand()
        self._make_panda_stand()

        self.handles = SceneHandles(
            plane=plane,
            hub=hub_ref,
            tire=tire,
            bolts=bolts,
            truck_uid=truck_uid,
            vehicle=vehicle_id,
            target_bolt_idx=target_idx,
            tire_rack=rack_uids,
            cargo_back_wall=cargo_back_wall_uid,
            floor_rim=floor_rim_uids,
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
        tt_pos, t_orn = self.tire_pose()
        ht_pos, _ = self.hub_pose()
        ht = np.asarray(ht_pos, dtype=np.float64)
        tt = np.asarray(tt_pos, dtype=np.float64)
        ah = self.hub_axis()
        nh = np.linalg.norm(ah)
        if nh < 1e-9:
            return 0.0, float(np.linalg.norm(tt - ht)), 0.0
        ah /= nh
        dh = tt - ht
        axial = float(np.dot(dh, ah))
        lateral = float(np.linalg.norm(dh - axial * ah))

        # No wheel-disk geometry (tire fell back to a solid cylinder) → there
        # is nothing whose lug holes can mis-align with the studs, so the
        # rotational residual is meaningless. Return 0 to avoid biasing reward
        # / success gates with a phantom signal.
        if not self.has_wheel_disk:
            return axial, lateral, 0.0

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
        # ``drift`` already absorbs both the curriculum-phase noise and
        # the new ``_hub_xy_offset`` because ``hub_center`` is the
        # post-offset hub position. We add ``_cargo_xy_offset`` on top
        # so cargo can be perturbed *independently* of the hub (the spec
        # treats them as separate randomized objects).
        drift = hub_center - nom
        pos = (
            np.asarray(self.cfg.vehicle_center_world, dtype=np.float64)
            + drift
            + self._cargo_xy_offset
        )
        cargo_rpy = tuple(float(x) for x in getattr(
            self.cfg, "vehicle_base_rpy", (0.0, 0.0, 0.0)
        ))
        cargo_orn = rpy_to_quat(cargo_rpy)

        if not self.cfg.cargo_use_wheel_well_cutout:
            return self._make_vehicle_box_solid(pos, he, cargo_orn)

        cells = self._cargo_keep_cells(hub_center, pos, cargo_orn, he)
        if not cells:
            return self._make_vehicle_box_solid(pos, he, cargo_orn)

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
            return self._make_vehicle_box_solid(pos, he, cargo_orn)
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos.tolist(),
            baseOrientation=cargo_orn.tolist(),
            physicsClientId=self.client,
        )

    def _make_cargo_back_wall(self, hub_center: np.ndarray) -> int:
        """Static thin slab behind the hub blocking +Y push past the flange.

        World-axis aligned (no yaw) for simplicity — only the +Y face
        matters for blocking. Half-extents from
        ``cfg.cargo_back_wall_half_extents``; placed at
        ``(hub_x, hub_y + offset, back_wall_center_z)``.
        """
        he = list(self.cfg.cargo_back_wall_half_extents)
        offset = float(self.cfg.cargo_back_wall_y_offset)
        wall_pos = [
            float(hub_center[0]),
            float(hub_center[1]) + offset,
            float(self.cfg.cargo_back_wall_center_z),
        ]
        rgba = list(self.cfg.cargo_back_wall_rgba)
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=he, physicsClientId=self.client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=he, rgbaColor=rgba,
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=wall_pos,
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )

    def _make_vehicle_box_solid(self, pos: np.ndarray, he: np.ndarray,
                                 orn: Optional[np.ndarray] = None) -> int:
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
        if orn is None:
            orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos.tolist(),
            baseOrientation=np.asarray(orn, dtype=np.float64).tolist(),
            physicsClientId=self.client,
        )

    def _cargo_keep_cells(
        self,
        hub_center: np.ndarray,
        cargo_center: np.ndarray,
        cargo_orn: np.ndarray,
        he: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (half_extents, offset_from_cargo_center) for sub-boxes outside the wheel-well.

        Sub-box offsets are expressed in the cargo's **local frame** (so they
        sit correctly when the body carries a non-identity ``baseOrientation``).
        The wheel-well cylinder is defined in the **world frame** at
        ``hub_center`` along ``cargo_wheel_well_axis`` (world X or Y) — each
        local cell centre is rotated into world coordinates before the cutout
        test, so the arch always opens on the side of the truck the hub
        actually protrudes from.
        """
        hx, hy, hz = float(he[0]), float(he[1]), float(he[2])
        nx, ny, nz = self.cfg.cargo_collision_subdiv
        dx = (2.0 * hx) / float(nx)
        dy = (2.0 * hy) / float(ny)
        dz = (2.0 * hz) / float(nz)

        Rcargo = np.array(
            p.getMatrixFromQuaternion(list(cargo_orn)), dtype=np.float64
        ).reshape(3, 3)

        axis = str(getattr(self.cfg, "cargo_wheel_well_axis", "x")).lower()
        R = float(getattr(self.cfg, "cargo_wheel_well_radius",
                          self.cfg.cargo_wheel_well_radius_yz))
        r_sq = R * R
        if axis == "y":
            along_range = getattr(
                self.cfg, "cargo_wheel_well_along_range_from_hub", (-1.05, 1.05),
            )
            a_lo = float(hub_center[1]) + float(along_range[0])
            a_hi = float(hub_center[1]) + float(along_range[1])
            h_perp_a = float(hub_center[0])
            h_perp_b = float(hub_center[2])
        else:
            along_range = getattr(
                self.cfg, "cargo_wheel_well_x_range_from_hub", (-0.65, 0.42),
            )
            a_lo = float(hub_center[0]) + float(along_range[0])
            a_hi = float(hub_center[0]) + float(along_range[1])
            h_perp_a = float(hub_center[1])
            h_perp_b = float(hub_center[2])

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Sub-box centre in cargo LOCAL frame (relative to cargo
                    # centre). Local AABB spans (-hx..+hx, -hy..+hy, -hz..+hz).
                    off_local = np.array([
                        -hx + (ix + 0.5) * dx,
                        -hy + (iy + 0.5) * dy,
                        -hz + (iz + 0.5) * dz,
                    ], dtype=np.float64)
                    half = np.array([0.5 * dx, 0.5 * dy, 0.5 * dz], dtype=np.float64)
                    # Rotate to world to evaluate the wheel-well cylinder.
                    p_world = cargo_center + Rcargo @ off_local
                    pcx, pcy, pcz = float(p_world[0]), float(p_world[1]), float(p_world[2])
                    if axis == "y":
                        in_well = self._cell_in_cylinder(
                            pcy, pcx, pcz, a_lo, a_hi, h_perp_a, h_perp_b, r_sq,
                        )
                    else:
                        in_well = self._cell_in_cylinder(
                            pcx, pcy, pcz, a_lo, a_hi, h_perp_a, h_perp_b, r_sq,
                        )
                    if in_well:
                        continue
                    out.append((half, off_local))
        return out

    @staticmethod
    def _cell_in_cylinder(
        p_along: float, p_perp_a: float, p_perp_b: float,
        a_lo: float, a_hi: float,
        h_perp_a: float, h_perp_b: float,
        r_sq: float,
    ) -> bool:
        """Is the point inside a finite cylinder coaxial with ``p_along``?"""
        if not (a_lo <= p_along <= a_hi):
            return False
        da = p_perp_a - h_perp_a
        db = p_perp_b - h_perp_b
        return (da * da + db * db) < r_sq

    @staticmethod
    def _cargo_cell_in_wheel_well(
        pcx: float, pcy: float, pcz: float,
        x_lo: float, x_hi: float, hy0: float, hz0: float, r_sq: float,
    ) -> bool:
        """Legacy alias retained for backward-compat. Uses axis=x semantics."""
        if not (x_lo <= pcx <= x_hi):
            return False
        dy = pcy - hy0
        dz = pcz - hz0
        return (dy * dy + dz * dz) < r_sq

    def _make_floor_with_circular_hole(self, floor_z: float) -> int:
        """Solid floor plate (top at floor_z) with ONE circular hole.

        Generates a watertight-ish triangle mesh of a large square plate with a
        cylindrical hole punched over Robot A's column, writes it to a temp OBJ,
        and loads it as a static concave trimesh (collision + visual). The arm
        can descend through the hole into the pit (the infinite plane below sits
        at ``floor_z − depth``); everywhere else the plate is solid so no arm
        link can punch through the floor. Returns the body UID.
        """
        import os
        import tempfile

        cx, cy = (float(v) for v in self.cfg.floor_pit_center)
        R = float(self.cfg.floor_pit_radius)
        depth = float(getattr(self.cfg, "floor_pit_depth", 0.85))
        L = float(getattr(self.cfg, "floor_pit_rim_extent", 12.0))
        N = int(getattr(self.cfg, "floor_pit_circle_segments", 96))
        N -= N % 4
        N = max(N, 8)
        top = floor_z
        bot = floor_z - depth  # hole wall spans the full pit depth

        # Build matched inner-circle and outer-square rings sharing the SAME
        # angular order: for each angle, the outer point is where the ray from
        # the pit centre hits the square boundary. This keeps the ring strip
        # un-twisted (a per-edge square sampling vs angular circle sampling
        # would skew triangles across the hole and create bogus collisions).
        circ = []
        sq = []
        for i in range(N):
            a = 2.0 * math.pi * i / N
            ca, sa = math.cos(a), math.sin(a)
            circ.append((cx + R * ca, cy + R * sa))
            tx = ((L if ca > 0 else -L) - cx) / ca if abs(ca) > 1e-9 else 1e18
            ty = ((L if sa > 0 else -L) - cy) / sa if abs(sa) > 1e-9 else 1e18
            t = min(tx, ty)
            sq.append((cx + t * ca, cy + t * sa))

        verts: List[Tuple[float, float, float]] = []

        def add(x, y, z):
            verts.append((x, y, z))
            return len(verts)  # 1-based for OBJ

        # Vertex rings
        out_t = [add(x, y, top) for (x, y) in sq]
        in_t = [add(x, y, top) for (x, y) in circ]
        in_b = [add(x, y, bot) for (x, y) in circ]
        faces: List[Tuple[int, int, int]] = []
        for i in range(N):
            j = (i + 1) % N
            # Top annulus (square outer -> circular inner)
            faces.append((in_t[i], out_t[i], in_t[j]))
            faces.append((out_t[i], out_t[j], in_t[j]))
            # Inner cylindrical wall (top circle -> bottom circle)
            faces.append((in_t[i], in_t[j], in_b[i]))
            faces.append((in_t[j], in_b[j], in_b[i]))

        fd, path = tempfile.mkstemp(suffix=".obj", prefix="floor_pit_")
        with os.fdopen(fd, "w") as f:
            for (x, y, z) in verts:
                f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
            for (a, b, c) in faces:
                f.write(f"f {a} {b} {c}\n")

        rgba = tuple(getattr(
            self.cfg, "floor_pit_rim_rgba", (0.55, 0.55, 0.58, 1.0),
        ))
        col = p.createCollisionShape(
            p.GEOM_MESH, fileName=path, meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.client,
        )
        vis = p.createVisualShape(
            p.GEOM_MESH, fileName=path, meshScale=[1, 1, 1], rgbaColor=rgba,
            physicsClientId=self.client,
        )
        uid = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )
        try:
            os.remove(path)  # mesh is already parsed into the shapes
        except OSError:
            pass
        return uid

    def _make_floor_rim(self, floor_z: float) -> List[int]:
        """Rebuild the floor surface OUTSIDE the pit rectangle as four slabs.

        With ``cfg.floor_pit_enable`` the infinite plane is dropped to the pit
        bottom; this lays four large box slabs (top at ``floor_z``) around the
        pit rectangle ``[px0,px1] × [py0,py1]`` so everywhere except the pit
        keeps a normal-height, solid floor. The slabs are static (mass 0) and
        their top faces sit exactly at ``floor_z``. Returns their body UIDs.
        """
        px0, px1 = (float(v) for v in self.cfg.floor_pit_x_range)
        py0, py1 = (float(v) for v in self.cfg.floor_pit_y_range)
        L = float(getattr(self.cfg, "floor_pit_rim_extent", 12.0))
        th = float(getattr(self.cfg, "floor_pit_rim_thickness", 0.20))
        rgba = tuple(getattr(
            self.cfg, "floor_pit_rim_rgba", (0.55, 0.55, 0.58, 1.0),
        ))
        hz = 0.5 * th
        cz = floor_z - hz  # top face flush with floor_z
        # (x0, x1, y0, y1) for the four bands surrounding the pit opening.
        bands = [
            (-L, px0, -L, L),    # −X strip (full height in Y)
            (px1, L, -L, L),     # +X strip (full height in Y)
            (px0, px1, -L, py0),  # −Y strip (between the X strips)
            (px0, px1, py1, L),   # +Y strip (between the X strips)
        ]
        uids: List[int] = []
        for (x0, x1, y0, y1) in bands:
            if (x1 - x0) <= 1e-4 or (y1 - y0) <= 1e-4:
                continue
            hx = 0.5 * (x1 - x0)
            hy = 0.5 * (y1 - y0)
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                physicsClientId=self.client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=rgba,
                physicsClientId=self.client,
            )
            uid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[cx, cy, cz],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
            uids.append(uid)
        return uids

    def _make_robot_stand(
        self,
        base_pos: Tuple[float, float, float],
        radius: float,
        rgba: Tuple[float, float, float, float],
    ) -> Optional[int]:
        """Fixed flange+column under a robot base so it is not visually floating.

        Skipped when ``radius <= 0`` or the base sits flush with the floor.
        Stand height = ``base_z − floor_z`` so it spans from the ground plane
        up to the robot mount whatever the world Z reference is (e.g. with
        the hub-centric origin the floor is at Z=−0.82, not Z=0).
        """
        r = float(radius)
        floor_z = float(getattr(self.cfg, "floor_z", 0.0))
        h = float(base_pos[2]) - floor_z
        if r <= 0.0 or h <= 1e-3:
            return None
        orn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        bx, by = float(base_pos[0]), float(base_pos[1])
        flange_r = r * 1.55
        # Flange sits flush on the floor.
        self._make_cylinder(
            radius=flange_r,
            height=0.04,
            mass=0.0,
            base_pos=np.array([bx, by, floor_z + 0.02], dtype=np.float64),
            base_orn=orn,
            rgba=tuple(min(1.0, c + 0.08) for c in rgba[:3]) + (rgba[3],),
        )
        col_h = max(h - 0.04, 0.1)
        return self._make_cylinder(
            radius=r,
            height=col_h,
            mass=0.0,
            base_pos=np.array(
                [bx, by, floor_z + 0.04 + 0.5 * col_h], dtype=np.float64,
            ),
            base_orn=orn,
            rgba=rgba,
        )

    def _make_panda_stand(self) -> Optional[int]:
        kind = str(getattr(self.cfg, "robot_b_kind", "ur10e")).lower()
        if kind in ("ur10e", "ur10_e", "ur_e"):
            radius = float(getattr(self.cfg, "ur10e_stand_radius", 0.12))
            rgba = tuple(getattr(
                self.cfg, "ur10e_stand_rgba", (0.35, 0.38, 0.42, 1.0),
            ))
        else:
            radius = float(getattr(self.cfg, "panda_stand_radius", 0.0))
            rgba = tuple(getattr(self.cfg, "panda_stand_rgba", (0.4, 0.4, 0.45, 1.0)))
        return self._make_robot_stand(
            base_pos=self.cfg.robot_B_base_pos,
            radius=radius,
            rgba=rgba,
        )

    def _make_ur10_stand(self) -> Optional[int]:
        kind = str(getattr(self.cfg, "robot_a_kind", "fanuc_r2000ic")).lower()
        if "fanuc" in kind or "r2000" in kind:
            radius = float(getattr(self.cfg, "fanuc_stand_radius", 0.28))
            rgba = tuple(getattr(
                self.cfg, "fanuc_stand_rgba", (0.32, 0.34, 0.38, 1.0),
            ))
        else:
            radius = float(getattr(self.cfg, "ur10_stand_radius", 0.0))
            rgba = tuple(getattr(self.cfg, "ur10_stand_rgba", (0.4, 0.4, 0.45, 1.0)))
        return self._make_robot_stand(
            base_pos=self.cfg.robot_A_base_pos,
            radius=radius,
            rgba=rgba,
        )

    def _make_split_tire_rack(self) -> List[int]:
        """Build the inline (Y-split) dual-block tire cradle (Inner + Outer rails).

        Two static rails (mass=0) parallel to the X axis, flanking the
        tire bore on opposite sides of the Y=0 centerline.
        They support the vertical tire's tread on both Y faces while
        leaving a hollow Y-gap between them so the gripper can
        thread straight along the robot-tire Y centerline to the 6
        o'clock outer point and grasp it without clipping either rail.

        Returns a list of PyBullet body UIDs (empty when disabled).

        Geometry contract — must hold whenever the rack is enabled:
          * ``inner_y - he_y ≥ tire_com_y + gap_half``
          * ``outer_y + he_y ≤ tire_com_y − gap_half``
          * ``floor_z + 2·he_z == tire_com_z − tire_outer_radius``
          * ``|inner_x - outer_x| < 1e-9`` (rails share the same X line)
        Failing any of these means the rack either collides with the
        tire's bore region or leaves no straight-line corridor for the
        gripper — both fatal for Stage 0.
        """
        if not bool(getattr(self.cfg, "spawn_tire_rack", True)):
            return []
        he = tuple(float(x) for x in self.cfg.tire_rack_half_extents)
        rgba = tuple(float(x) for x in self.cfg.tire_rack_rgba)
        centers = (
            tuple(float(x) for x in self.cfg.tire_rack_inner_center),
            tuple(float(x) for x in self.cfg.tire_rack_outer_center),
        )
        uids: List[int] = []
        for cx, cy, cz in centers:
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=list(he),
                physicsClientId=self.client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=list(he),
                rgbaColor=list(rgba),
                physicsClientId=self.client,
            )
            uid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[cx, cy, cz],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
            # Higher friction so the cradle keeps the tread from slipping
            # off when the gripper bumps the tire on approach. Spinning
            # friction stays mild — we don't want grip-induced torque
            # spikes from a phantom contact.
            p.changeDynamics(
                uid,
                -1,
                lateralFriction=1.0,
                spinningFriction=0.005,
                physicsClientId=self.client,
            )
            uids.append(uid)

        # Open-corridor cradle: prop each thin top bar up from the floor with
        # a vertical post placed BEHIND the bar (−X, the far side away from
        # the FANUC base at +X). The arm threads in from the +X side and dips
        # between the bars to the 6 o'clock grasp point, so a post on the −X
        # back face never enters the approach corridor (neither the forearm
        # nor the tool chuck reaches behind the tire). See
        # cfg.tire_rack_support_posts.
        if bool(getattr(self.cfg, "tire_rack_support_posts", False)):
            floor_z = float(getattr(self.cfg, "floor_z", 0.0))
            post_xy = tuple(
                float(x) for x in getattr(
                    self.cfg, "tire_rack_post_half_extents_xy", (0.10, 0.05))
            )
            for cx, cy, cz in centers:
                bar_bottom = cz - he[2]
                post_h = bar_bottom - floor_z
                if post_h <= 1e-4:
                    continue  # bar already rests on the floor; no post needed
                # Post spans the bar's full Y width; placed flush against the
                # bar's −X (back) face and running down to the floor.
                post_he = [post_xy[0], he[1], post_h / 2.0]
                post_cx = cx - he[0] - post_xy[0]
                post_cz = floor_z + post_h / 2.0
                pcol = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=post_he,
                    physicsClientId=self.client,
                )
                pvis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=post_he, rgbaColor=list(rgba),
                    physicsClientId=self.client,
                )
                puid = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=pcol,
                    baseVisualShapeIndex=pvis,
                    basePosition=[post_cx, cy, post_cz],
                    baseOrientation=[0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client,
                )
                p.changeDynamics(
                    puid, -1, lateralFriction=1.0, spinningFriction=0.005,
                    physicsClientId=self.client,
                )
                uids.append(puid)
        return uids

    def _spawn_tire(self) -> int:
        # Phase 1 FSM: tire starts on the dual-block rack next to UR10,
        # standing vertically. The env will pin/release this body through
        # FSM transitions (Stage 0 = pinned to rack, Stage 1/2 = grasped).
        tire_pos = np.asarray(self.cfg.tire_pickup_pos, dtype=np.float64)
        # Spawn orientation comes from ``cfg.tire_spawn_rpy`` (default
        # (0, π/2, 0) → bore axis = world +X, facing robot A). The grasp
        # constraint reproduces this exact orientation when Stage 0 → 1
        # fires.
        spawn_rpy = tuple(float(x) for x in self.cfg.tire_spawn_rpy)
        tire_orn = np.asarray(
            p.getQuaternionFromEuler(list(spawn_rpy)),
            dtype=np.float64,
        )
        uid, has_disk = models.create_tire_wheel_multibody(
            self.client,
            self.cfg,
            base_position=tire_pos.tolist(),
            base_orientation=tire_orn.tolist(),
            visual_primitive_cap=_PYB_COMPOUND_SHAPE_CAP,
        )
        self.has_wheel_disk = bool(has_disk)
        return uid

    def _sample_offset_xyz(self) -> np.ndarray:
        """Hub spatial DR: XY only, Z held fixed.

        A vertical hub jitter would risk the tire (radius ~0.525 m) clipping
        the floor or floating off contact, which is a physics artefact rather
        than a policy-relevant perturbation. Robustness to vertical hub
        positioning is better tested via cargo / floor-height experiments.
        """
        ranges_cm = self.cfg.curriculum.phase_ranges_cm
        idx = max(0, min(len(ranges_cm) - 1, self.cfg.curriculum.phase - 1))
        r_m = ranges_cm[idx] / 100.0
        if r_m <= 0:
            return np.zeros(3)
        xy = self.np_random.uniform(-r_m, r_m, size=2)
        return np.array([float(xy[0]), float(xy[1]), 0.0], dtype=np.float64)
