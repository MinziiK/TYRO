"""Mesh-free tire+wheel / hub primitives (PyBullet boxes & cylinders).

`create_tire_wheel_multibody` keeps the tyre tread ring and silver wheel-disk
pieces on separate multibody links so each compound stays ≤ Bullet’s visual
shape array limit (~16), while modelling ~10 lug holes via three boxes per hole.
"""
from __future__ import annotations

import math
import warnings
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from ..config import EnvConfig
from .utils import rpy_to_quat

PyVec3 = Sequence[float]


def _tire_inertia_diagonal(cfg: EnvConfig) -> Optional[list]:
    """Explicit base inertia when mass does not match collision mesh density."""
    custom = getattr(cfg, "tire_inertia_diagonal", None)
    if custom is not None:
        return [float(x) for x in custom]
    if float(cfg.tire_mass) >= 50.0:
        heavy = getattr(cfg, "tire_inertia_heavy", (18.0, 32.0, 32.0))
        return [float(x) for x in heavy]
    return None


def _iter_box_chunks(
    he: list,
    po: list,
    ori: list,
    chunk_max: int,
) -> Iterator[Tuple[list, list, list]]:
    step = max(1, int(chunk_max))
    for i in range(0, len(he), step):
        yield he[i:i + step], po[i:i + step], ori[i:i + step]


def tire_annulus_boxes(
    rout: float,
    rin: float,
    half_axial: float,
    segments: int,
) -> Optional[tuple[list[list[float]], list[list[float]], list[list[float]]]]:
    """Hollow tyre tread: tiling boxes in XY, axle +local Z (before base ori)."""

    dr = float(rout - rin)
    segs = int(segments)
    if segs < 4 or dr <= 1e-6 or rin >= rout:
        return None
    r_med = rin + 0.5 * dr
    dtheta = 2.0 * math.pi / segs
    sy_slope = math.sin(0.5 * dtheta) * 1.06
    half_extents: list[list[float]] = []
    positions: list[list[float]] = []
    orientations: list[list[float]] = []
    hz = float(half_axial)
    for i in range(segs):
        theta = (i + 0.5) * dtheta
        half_extents.append(
            [
                max(0.5 * dr, 1e-4),
                max(r_med * sy_slope, 1e-4),
                max(hz, 1e-4),
            ]
        )
        positions.append([
            float(r_med * math.cos(theta)),
            float(r_med * math.sin(theta)),
            0.0,
        ])
        orientations.append(
            np.asarray(rpy_to_quat([0.0, 0.0, float(theta)]), dtype=float).tolist())
    return half_extents, positions, orientations


def wheel_disk_inter_lug_wedges(
    bolt_circle_radius: float,
    n_segments: int,
    r_outer: float,
    r_inner: float,
    half_axial_disk: float,
    gap_radius: float,
    pattern_phase_rad: float,
) -> Optional[tuple[list[list[float]], list[list[float]], list[list[float]]]]:
    """Legacy: radial slabs centred *between* bolt centres (lug gaps)."""

    bolt_r = float(bolt_circle_radius)
    inner = float(r_inner)
    outer = float(r_outer)
    ns = int(n_segments)
    if ns < 2 or bolt_r <= 0.0 or inner <= 0.0 or outer <= inner + 1e-4:
        return None

    wedge_step = (2.0 * math.pi) / float(ns)
    phase = float(pattern_phase_rad)
    gap_clr = float(gap_radius)
    theta_gap = 2.0 * math.asin(min(abs(gap_clr) / bolt_r, 0.9999))
    theta_gap = max(1e-4, min(theta_gap, wedge_step * 0.95))
    theta_seg = wedge_step - theta_gap
    if theta_seg <= 1e-5:
        return None

    r_med = 0.5 * (inner + outer)
    rad_half = max(0.5 * (outer - inner), 1e-4)
    hz = max(float(half_axial_disk), 1e-4)
    tau_half = max(r_med * math.sin(0.5 * theta_seg) * 1.06, 1e-4)

    half_extents: list[list[float]] = []
    positions: list[list[float]] = []
    orientations: list[list[float]] = []

    for k in range(ns):
        theta_mid = phase + wedge_step * (float(k) + 0.5)
        half_extents.append([rad_half, tau_half, hz])
        positions.append([
            r_med * math.cos(theta_mid),
            r_med * math.sin(theta_mid),
            0.0,
        ])
        orientations.append(
            np.asarray(rpy_to_quat([0.0, 0.0, float(theta_mid)]), dtype=float).tolist())
    return half_extents, positions, orientations


def wheel_disk_three_boxes_per_hole(
    *,
    bolt_circle_radius: float,
    n_holes: int,
    bolt_hole_radius: float,
    r_outer: float,
    r_inner: float,
    half_axial_disk: float,
    pattern_phase_rad: float,
) -> Optional[tuple[list[list[float]], list[list[float]], list[list[float]]]]:
    """~10 lug holes × 3 silver boxes (two flanks + inner sill) ⇒ arched clearance."""

    ns = max(3, int(n_holes))
    pcd = float(bolt_circle_radius)
    bo = float(bolt_hole_radius)
    outer = float(r_outer)
    inner = float(r_inner)
    hz = max(float(half_axial_disk), 1e-4)
    if pcd <= 0.0 or outer <= inner + 1e-4:
        return None

    beta = math.asin(min(bo / pcd, 0.9999)) * 1.18
    d_ang = beta + max(1.02 * bo / pcd, 0.035 / pcd)
    r_mid = 0.5 * (inner + outer)
    rad_leg = max(0.5 * (outer - inner), 1e-4) * 0.88

    tau_side = max(pcd * math.sin(beta) * 0.92, bo * 0.95, 1e-4)
    r_back = inner + max(0.42 * max(pcd - inner, 1e-3), bo * 1.05)
    tau_back = max(pcd * math.sin(min(beta + 0.04, math.pi / 2.8)), bo * 1.1, 1e-4)
    hr_back = max(0.5 * (pcd - r_back + bo), bo * 0.7, 1e-4)

    he_all: list[list[float]] = []
    po_all: list[list[float]] = []
    or_all: list[list[float]] = []

    wedge_step = 2.0 * math.pi / float(ns)
    phase = float(pattern_phase_rad)

    for i in range(ns):
        theta = phase + wedge_step * float(i)
        for sig in (-1.0, 1.0):
            th_wall = theta + sig * d_ang
            he_all.append([rad_leg * 0.95, tau_side * 0.55, hz])
            po_all.append([
                r_mid * math.cos(th_wall),
                r_mid * math.sin(th_wall),
                0.0,
            ])
            or_all.append(
                np.asarray(
                    rpy_to_quat([0.0, 0.0, float(th_wall)]), dtype=float
                ).tolist())
        he_all.append([hr_back * 1.05, tau_back * 0.92, hz])
        po_all.append([
            r_back * math.cos(theta),
            r_back * math.sin(theta),
            0.0,
        ])
        or_all.append(
            np.asarray(
                rpy_to_quat([0.0, 0.0, float(theta)]), dtype=float,
            ).tolist())

    return he_all, po_all, or_all


def _merged_rgbs(ncol_tread: int, ncol_piece: int, tr: Iterable, dk: Iterable) -> list[list[float]]:
    return [list(tr) for _ in range(ncol_tread)] + [list(dk) for _ in range(ncol_piece)]


def _create_box_compound_or_none(
    client: int,
    he: list,
    po: list,
    ori: list,
    rgbs: list[list[float]],
) -> Tuple[int, int]:
    ncol = len(he)
    col = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * ncol,
        halfExtents=he,
        collisionFramePositions=po,
        collisionFrameOrientations=ori,
        physicsClientId=client,
    )
    vis = p.createVisualShapeArray(
        shapeTypes=[p.GEOM_BOX] * ncol,
        halfExtents=he,
        rgbaColors=rgbs,
        visualFramePositions=po,
        visualFrameOrientations=ori,
        physicsClientId=client,
    )
    return col, vis


def create_tire_wheel_multibody(
    client: int,
    cfg: EnvConfig,
    *,
    base_position: PyVec3,
    base_orientation: PyVec3,
    visual_primitive_cap: int = 16,
) -> Tuple[int, bool]:
    """Integrated tyre+tread ring + silver wheel-disk (compound or multi-link).

    Returns
    -------
    (uid, has_wheel_disk)
        ``has_wheel_disk`` is False when we fall back to a solid cylinder; the
        caller should disable lug-spin reward terms in that case.
    """

    rout = cfg.tire_outer_radius
    rin = (
        float(cfg.tire_collision_inner_radius)
        if cfg.tire_collision_inner_radius is not None
        else float(cfg.tire_inner_radius)
    )
    half_axial = 0.5 * cfg.tire_thickness
    tire_mass = float(cfg.tire_mass)
    tire_rgba = [0.08, 0.08, 0.09, 1.0]
    disk_rgba = [0.72, 0.73, 0.76, 1.0]

    cap = max(6, min(int(visual_primitive_cap), 16))
    tread_seg = cfg.tire_annulus_collision_segments
    disk_n = int(cfg.n_bolts if cfg.tire_wheel_disk_enabled else 0)
    style = (cfg.tire_wheel_disk_style or "three_piece").lower()

    tread_prims: Optional[tuple] = None
    disk_prims: Optional[tuple] = None

    if cfg.tire_hollow_collision:
        if disk_n and style == "three_piece":
            tread_seg = max(4, min(tread_seg, cap))
        elif disk_n and style == "inter_lug_wedge":
            tread_seg = max(4, min(tread_seg, max(4, cap - disk_n)))

        tread_prims = tire_annulus_boxes(
            rout, rin, half_axial, tread_seg,
        )
        if disk_n and cfg.tire_wheel_disk_enabled:
            hz_disk = 0.5 * float(cfg.wheel_disk_thickness)
            r_disk_out = (
                float(cfg.wheel_disk_radial_outer)
                if cfg.wheel_disk_radial_outer is not None
                else rin
            )
            r_disk_out = max(r_disk_out, rin)
            r_disk_in = max(1e-3, min(float(cfg.wheel_disk_radial_inner), rin - 5e-3))
            if style == "three_piece":
                disk_prims = wheel_disk_three_boxes_per_hole(
                    bolt_circle_radius=cfg.bolt_circle_radius,
                    n_holes=disk_n,
                    bolt_hole_radius=float(cfg.wheel_disk_bolt_hole_radius),
                    r_outer=r_disk_out,
                    r_inner=r_disk_in,
                    half_axial_disk=hz_disk,
                    pattern_phase_rad=float(cfg.wheel_disk_bolt_phase_rad),
                )
            else:
                disk_prims = wheel_disk_inter_lug_wedges(
                    bolt_circle_radius=cfg.bolt_circle_radius,
                    n_segments=disk_n,
                    r_outer=r_disk_out,
                    r_inner=r_disk_in,
                    half_axial_disk=hz_disk,
                    gap_radius=cfg.wheel_disk_bolt_gap_clearance_radius,
                    pattern_phase_rad=float(cfg.wheel_disk_bolt_phase_rad),
                )

    # Single compound (legacy path) when disk+tread fit one array
    if tread_prims is not None and (
        disk_prims is None
        or (
            style == "inter_lug_wedge"
            and len(tread_prims[0]) + len(disk_prims[0]) <= cap
        )
    ):
        merged_he, merged_po, merged_or = [], [], []
        if tread_prims:
            merged_he.extend(tread_prims[0])
            merged_po.extend(tread_prims[1])
            merged_or.extend(tread_prims[2])
        ncol_tread = len(tread_prims[0]) if tread_prims else 0
        if disk_prims:
            merged_he.extend(disk_prims[0])
            merged_po.extend(disk_prims[1])
            merged_or.extend(disk_prims[2])
        ncol_disk = len(disk_prims[0]) if disk_prims else 0
        ncol = len(merged_he)
        if ncol > 0 and ncol <= cap:
            rgbs = _merged_rgbs(ncol_tread, ncol_disk, tire_rgba, disk_rgba)
            if len(rgbs) != ncol:
                rgbs = [list(tire_rgba)] * ncol
            col, vis = _create_box_compound_or_none(
                client, merged_he, merged_po, merged_or, rgbs,
            )
            if col >= 0 and vis >= 0:
                inertia = _tire_inertia_diagonal(cfg)
                mb_kwargs = dict(
                    baseMass=tire_mass,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=list(base_position),
                    baseOrientation=list(base_orientation),
                    physicsClientId=client,
                )
                if inertia is not None:
                    mb_kwargs["baseInertiaDiagonal"] = inertia
                uid = p.createMultiBody(**mb_kwargs)
                p.changeDynamics(
                    uid, -1, linearDamping=0.5, angularDamping=0.5,
                    physicsClientId=client,
                )
                return uid, (disk_prims is not None and len(disk_prims[0]) > 0)

    # Multi-link: tread on base, wheel disk split into ≤cap box arrays per link
    if tread_prims is None or disk_prims is None or len(disk_prims[0]) == 0:
        return _fallback_solid_tire_with_warning(
            client, rout, cfg.tire_thickness, base_position, base_orientation,
            mass=tire_mass,
            reason="tread or disk primitive build returned None/empty",
        )

    the, tpo, tor = tread_prims
    ncol_tr = len(the)
    tr_rgbs = [list(tire_rgba)] * ncol_tr
    t_col, t_vis = _create_box_compound_or_none(client, the, tpo, tor, tr_rgbs)
    if t_col < 0 or t_vis < 0:
        return _fallback_solid_tire_with_warning(
            client, rout, cfg.tire_thickness, base_position, base_orientation,
            mass=tire_mass,
            reason="tread compound creation failed in PyBullet",
        )

    d_he, d_po, d_ori = disk_prims
    chunk_max = max(4, cap - 1)
    disk_cols: list[int] = []
    disk_vis: list[int] = []
    for che, cpo, cor in _iter_box_chunks(d_he, d_po, d_ori, chunk_max):
        nk = len(che)
        drg = [list(disk_rgba)] * nk
        dc, dv = _create_box_compound_or_none(client, che, cpo, cor, drg)
        if dc < 0 or dv < 0:
            return _fallback_solid_tire_with_warning(
                client, rout, cfg.tire_thickness, base_position, base_orientation,
                mass=tire_mass,
                reason="wheel-disk compound chunk creation failed in PyBullet",
            )
        disk_cols.append(dc)
        disk_vis.append(dv)

    n_links = len(disk_cols)
    masses = [0.02] * n_links
    # Attach fixed children to the base (-1 in ``createMultiBody`` tree).
    parent = [-1] * n_links
    pos = [[0.0, 0.0, 0.0]] * n_links
    orn = [[0.0, 0.0, 0.0, 1.0]] * n_links
    in_pos = [[0.0, 0.0, 0.0]] * n_links
    in_orn = [[0.0, 0.0, 0.0, 1.0]] * n_links
    joint_types = [p.JOINT_FIXED] * n_links
    joint_axis = [[0.0, 0.0, 1.0]] * n_links

    mb_kwargs = dict(
        baseMass=tire_mass,
        baseCollisionShapeIndex=t_col,
        baseVisualShapeIndex=t_vis,
        basePosition=list(base_position),
        baseOrientation=list(base_orientation),
        linkMasses=masses,
        linkCollisionShapeIndices=disk_cols,
        linkVisualShapeIndices=disk_vis,
        linkPositions=pos,
        linkOrientations=orn,
        linkInertialFramePositions=in_pos,
        linkInertialFrameOrientations=in_orn,
        linkParentIndices=parent,
        linkJointTypes=joint_types,
        linkJointAxis=joint_axis,
        physicsClientId=client,
    )
    inertia = _tire_inertia_diagonal(cfg)
    if inertia is not None:
        mb_kwargs["baseInertiaDiagonal"] = inertia
    uid = p.createMultiBody(**mb_kwargs)
    p.changeDynamics(
        uid, -1, linearDamping=0.5, angularDamping=0.5, physicsClientId=client,
    )
    return uid, True


def _fallback_solid_tire_with_warning(
    client: int, radius: float, height: float, pos: PyVec3, orn: PyVec3,
    *, mass: float, reason: str,
) -> Tuple[int, bool]:
    warnings.warn(
        f"create_tire_wheel_multibody: falling back to solid cylinder ({reason}). "
        "Lug-spin reward terms must be disabled — has_wheel_disk=False is returned.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _fallback_solid_tire(client, radius, height, pos, orn, mass=mass), False


def _fallback_solid_tire(
    client: int, radius: float, height: float, pos: PyVec3, orn: PyVec3,
    *, mass: float = 1.0,
) -> int:
    col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client,
    )
    vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height,
        rgbaColor=[0.1, 0.1, 0.12, 1.0],
        physicsClientId=client,
    )
    uid = p.createMultiBody(
        baseMass=5.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(pos),
        baseOrientation=list(orn),
        physicsClientId=client,
    )
    p.changeDynamics(
        uid, -1, linearDamping=0.5, angularDamping=0.5, physicsClientId=client,
    )
    return uid
