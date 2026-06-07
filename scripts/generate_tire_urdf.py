"""Generate a UR10-feasible tire URDF (hollow tread ring, mesh-free).

The original procedural tire (``src/env/models.create_tire_wheel_multibody``)
models a full truck tyre — outer radius 0.525 m (Ø 1.05 m). Diagnosis of the
Min-Jerk planner + residual carry (2026-06-01) showed the UR10 (reach ~1.3 m)
cannot stably grasp / reorient an object that large: the COM hangs ~0.525 m
out on a lever from the 6-o'clock grasp, so any reorientation swings the tire
into the rack / arm, and the far-reach 90° bore reorientation is not even
IK-trackable.

This generator emits a **smaller** tire sized for the UR10 while still being
mountable on the *existing* truck hub:

  * ``--inner`` (bore radius) defaults to 0.23 m, just clearing the hub
    flange (``EnvConfig.hub_radius = 0.21``) and the bolt-circle outer edge
    (``bolt_circle_radius 0.1675 + bolt_radius 0.011 = 0.1785``), so the
    hollow tread slides over the studs/flange when the COM reaches the hub.
  * ``--outer`` defaults to 0.30 m → the 6-o'clock grasp lever drops from
    0.525 m to 0.30 m (-43 %).
  * ``--width`` (axial) defaults to 0.16 m, ``--mass`` 1.5 kg.

Convention (matches the env): the **bore axis is the link local +Z** — same
as the procedural tire and ``cfg.tire_spawn_rpy`` / ``_quat_align_z_to``
(which align local +Z to a world direction). The tread ring is built from
``--segments`` boxes tiled in the local X-Y plane, mirroring
``models.tire_annulus_boxes`` so collision behaviour is consistent. The link
origin is the geometric centre, so the body base pose == tire COM (same as the
procedural multibody), keeping every grasp / mount offset in the env valid.

Usage
-----
    python -m scripts.generate_tire_urdf            # defaults → data/urdf/tire/tire_ur10.urdf
    python -m scripts.generate_tire_urdf --outer 0.28 --inner 0.23 --width 0.14

After generating, enable it in training/eval via the env config:
    use_tire_urdf=True, tire_urdf=<path>,
    tire_outer_radius=<outer>, tire_inner_radius=<inner>,
    tire_thickness=<width>, tire_mass=<mass>
(the script prints the exact override block to copy).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "urdf" / "tire" / "tire_ur10.urdf"


def _hollow_cylinder_inertia(mass: float, r_in: float, r_out: float, length: float):
    """Principal inertia of a hollow cylinder about its centre.

    Bore axis = local Z (Izz), so Ixx == Iyy about the transverse axes.
    """
    rr = r_in * r_in + r_out * r_out
    izz = 0.5 * mass * rr
    ixx = (1.0 / 12.0) * mass * (3.0 * rr + length * length)
    return ixx, ixx, izz


def _annulus_boxes(r_out: float, r_in: float, half_axial: float, segments: int):
    """Tread-ring boxes tiled in the local X-Y plane (bore axis = local Z).

    Mirrors ``src/env/models.tire_annulus_boxes`` so the URDF tyre collides
    like the procedural one. Returns a list of
    ``(half_extents, xyz, yaw)`` tuples.
    """
    dr = float(r_out - r_in)
    segs = int(segments)
    if segs < 4 or dr <= 1e-6 or r_in >= r_out:
        raise ValueError("invalid annulus: need segments>=4 and 0 < inner < outer")
    r_med = r_in + 0.5 * dr
    dtheta = 2.0 * math.pi / segs
    # 1.06 overlap factor matches models.py so neighbouring boxes seal the ring.
    sy = math.sin(0.5 * dtheta) * 1.06
    out = []
    for i in range(segs):
        theta = (i + 0.5) * dtheta
        he = (max(0.5 * dr, 1e-4), max(r_med * sy, 1e-4), max(half_axial, 1e-4))
        xyz = (r_med * math.cos(theta), r_med * math.sin(theta), 0.0)
        out.append((he, xyz, theta))
    return out


def build_urdf(outer: float, inner: float, width: float, mass: float,
               segments: int, rgba) -> str:
    half_axial = 0.5 * float(width)
    boxes = _annulus_boxes(outer, inner, half_axial, segments)
    ixx, iyy, izz = _hollow_cylinder_inertia(mass, inner, outer, width)
    r, g, b, a = rgba

    blocks = []
    blocks.append('<?xml version="1.0"?>')
    blocks.append(
        f'<!-- UR10-feasible tire: outer={outer} inner={inner} width={width} '
        f'mass={mass} segments={segments}. Bore axis = link local +Z. -->'
    )
    blocks.append('<robot name="tire_ur10">')
    blocks.append('  <material name="tire_rubber">')
    blocks.append(f'    <color rgba="{r} {g} {b} {a}"/>')
    blocks.append('  </material>')
    blocks.append('  <link name="tire">')
    blocks.append('    <inertial>')
    blocks.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
    blocks.append(f'      <mass value="{mass}"/>')
    blocks.append(
        f'      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" '
        f'iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>'
    )
    blocks.append('    </inertial>')
    for he, xyz, yaw in boxes:
        sx, sy, sz = 2.0 * he[0], 2.0 * he[1], 2.0 * he[2]
        origin = (
            f'      <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" '
            f'rpy="0 0 {yaw:.6f}"/>'
        )
        box = f'        <box size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'
        blocks.append('    <visual>')
        blocks.append(origin)
        blocks.append('      <geometry>')
        blocks.append(box)
        blocks.append('      </geometry>')
        blocks.append('      <material name="tire_rubber"/>')
        blocks.append('    </visual>')
        blocks.append('    <collision>')
        blocks.append(origin)
        blocks.append('      <geometry>')
        blocks.append(box)
        blocks.append('      </geometry>')
        blocks.append('    </collision>')
    blocks.append('  </link>')
    blocks.append('</robot>')
    return "\n".join(blocks) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outer", type=float, default=0.30,
                    help="Tread outer radius (m). Grasp lever = this value.")
    ap.add_argument("--inner", type=float, default=0.23,
                    help="Bore inner radius (m). Must exceed hub flange "
                         "(0.21) + clearance to mount over the existing hub.")
    ap.add_argument("--width", type=float, default=0.16,
                    help="Axial tread width (m).")
    ap.add_argument("--mass", type=float, default=1.5, help="Tire mass (kg).")
    ap.add_argument("--segments", type=int, default=24,
                    help="Number of boxes tiling the tread ring.")
    ap.add_argument("--rgba", type=str, default="0.12,0.12,0.14,1.0",
                    help="Tread colour r,g,b,a.")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.inner <= 0.215:
        print(f"[warn] inner={args.inner} m may not clear the hub flange "
              f"(0.21 m) — the tire could collide before seating.")
    if args.outer <= args.inner:
        raise SystemExit("--outer must be greater than --inner")

    rgba = tuple(float(x) for x in args.rgba.split(","))
    urdf = build_urdf(args.outer, args.inner, args.width, args.mass,
                      args.segments, rgba)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(urdf)
    print(f"[generate_tire_urdf] wrote {out}")
    print("\n# Companion EnvConfig overrides to actually use this tire:")
    print(f"#   use_tire_urdf=True, tire_urdf={str(out)!r},")
    print(f"#   tire_outer_radius={args.outer}, tire_inner_radius={args.inner},")
    print(f"#   tire_thickness={args.width}, tire_mass={args.mass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
