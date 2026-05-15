#!/usr/bin/env python3
"""Emit ``truck_wheel_station.urdf`` (hub + N studs along **hub link +Z**).

Bolts lie on a circle in the flange (hub XY); URDF cylinder axis is hub +Z.
World-facing of the flange/studs is set in **scene** via ``EnvConfig.hub_base_rpy``
(URDF stays Z-along-stud). Default pitch **−π/2** maps local +Z → world **−X** (toward −X robots).

Bolts ``--bolt-radius`` / ``--bolt-collision-radius-factor`` adjust visual vs collision shafts.

Run from repo root (``conda activate tyro``):

  python scripts/generate_truck_wheel_station_urdf.py \\
      --bolt-collision-radius-factor 0.95 --output data/urdf/truck_assembly/truck_wheel_station.urdf
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def _fmt(x: float) -> str:
    """URDF-safe float: never omit a zero component — ``\"0\".rstrip(\"0\")\"`` must not go empty."""
    if not isinstance(x, float):
        return str(x)
    if abs(x) < 1e-14:
        return "0"
    s = f"{x:.8g}"
    if "e" in s.lower():
        return s
    if "." in s:
        t = s.rstrip("0").rstrip(".")
        return t if t else "0"
    return s


def inertia_cylinder_about_z(mass: float, radius: float, half_length: float) -> tuple[float, float]:
    """Rough diagonal inertia for a cylinder — returns ``(ixx == iyy, izz)``."""
    h = half_length * 2
    ixx = 0.083333333 * mass * (3 * radius ** 2 + h ** 2)
    izz = 0.5 * mass * radius ** 2
    return ixx, izz


def main() -> int:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    default_out = repo / "data" / "urdf" / "truck_assembly" / "truck_wheel_station.urdf"

    ap.add_argument("--output", type=Path, default=default_out)
    ap.add_argument("--n-bolts", type=int, default=10)
    ap.add_argument(
        "--bolt-pattern-phase-deg",
        type=float,
        default=0.0,
        help=(
            "First stud azimuth offset in XY (deg): theta_k = phase + k*360/n "
            "(must match EnvConfig.wheel_disk_bolt_phase_rad)."
        ),
    )
    ap.add_argument("--bolt-circle-radius", type=float, default=0.1675,
                    help="PCD radius (half of 335 mm pitch circle diameter).")
    ap.add_argument("--bolt-length", type=float, default=0.10)
    ap.add_argument("--bolt-radius", type=float, default=0.011,
                    help="M22-ish stud radius (visual + nominal collision before factor).")
    ap.add_argument("--bolt-collision-radius-factor", type=float, default=0.95)
    ap.add_argument("--hub-radius", type=float, default=0.21,
                    help="Flange cylinder radius (~ø420 mm / 2).")
    ap.add_argument("--hub-thickness", type=float, default=0.06)
    ap.add_argument("--hub-mass", type=float, default=75.0)
    ap.add_argument(
        "--no-hub-pilot",
        action="store_true",
        help="Omit cylindrical hub pilot (centering boss) protruding toward the wheel.",
    )
    ap.add_argument(
        "--hub-pilot-radius",
        type=float,
        default=0.16,
        help="Pilot OD/2 (~sliding fit inside tire wheel center bore vs EnvConfig.tire_inner_radius).",
    )
    ap.add_argument("--hub-pilot-length", type=float, default=0.046)
    ap.add_argument(
        "--no-brake-proxy",
        action="store_true",
        help="Skip simple brake rotor + caliper collision proxies aft of the flange (−hub Z).",
    )
    ap.add_argument("--brake-rotor-radius", type=float, default=0.30)
    ap.add_argument("--brake-rotor-half-thickness", type=float, default=0.011)
    ap.add_argument(
        "--brake-caliper-y",
        type=float,
        default=0.27,
        help="Caliper box center offset on hub local +Y (side bank).",
    )
    ap.add_argument(
        "--brake-caliper-size",
        type=str,
        default="0.26,0.11,0.16",
        help="Comma-separated full box extents x,y,z for caliper collision/visual.",
    )
    args = ap.parse_args()

    n = args.n_bolts
    br = args.bolt_radius
    bl = args.bolt_length
    r_col = max(br * args.bolt_collision_radius_factor, 1e-4)
    z_joint = args.hub_thickness / 2 + bl / 2
    bx, bz = inertia_cylinder_about_z(0.05, br, bl / 2)

    phase_rad = math.radians(args.bolt_pattern_phase_deg)
    bolts_xml: list[str] = []
    for k in range(n):
        theta = phase_rad + (2 * math.pi * k / n)
        x = args.bolt_circle_radius * math.cos(theta)
        y = args.bolt_circle_radius * math.sin(theta)
        bolts_xml.append(
            f"""
  <link name="bolt_{k}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="{bx}" ixy="0.0" ixz="0.0" iyy="{bx}" iyz="0.0" izz="{bz}" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{br}" length="{bl}"/>
      </geometry>
      <material name="bolt"><color rgba="0.7 0.7 0.7 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{r_col}" length="{bl}"/>
      </geometry>
    </collision>
  </link>
  <joint name="bolt_{k}_joint" type="fixed">
    <parent link="hub_mount"/>
    <child link="bolt_{k}"/>
    <origin xyz="{_fmt(x)} {_fmt(y)} {_fmt(z_joint)}" rpy="0 0 0"/>
            </joint>"""
        )

    hx = 0.083333333 * args.hub_mass * (3 * args.hub_radius ** 2 + args.hub_thickness ** 2)

    extra = ""
    ht = args.hub_thickness
    if not args.no_hub_pilot:
        pr = args.hub_pilot_radius
        lp = args.hub_pilot_length
        zp = ht / 2 + lp / 2
        prc = max(pr * 0.98, 1e-4)
        extra += f"""
  <!-- Pilot boss: tire center bore slides over this; studs pass through wheel holes. -->
  <link name="hub_pilot">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="0.00012" ixy="0.0" ixz="0.0" iyy="0.00012" iyz="0.0" izz="0.00012"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{_fmt(pr)}" length="{_fmt(lp)}"/>
      </geometry>
      <material name="pilot_steel"><color rgba="0.55 0.55 0.58 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{_fmt(prc)}" length="{_fmt(lp)}"/>
      </geometry>
    </collision>
  </link>
  <joint name="hub_pilot_joint" type="fixed">
    <parent link="hub_mount"/>
    <child link="hub_pilot"/>
    <origin xyz="0 0 {_fmt(zp)}" rpy="0 0 0"/>
  </joint>"""

    if not args.no_brake_proxy:
        rr = args.brake_rotor_radius
        rh = args.brake_rotor_half_thickness
        rotor_len = max(2.0 * rh, 1e-4)
        z_rotor = -(ht / 2 + 0.012 + rotor_len / 2)
        try:
            sx, sy, sz = [float(x.strip()) for x in args.brake_caliper_size.split(",")]
        except ValueError:
            raise SystemExit("--brake-caliper-size must be three comma floats") from None
        cy = args.brake_caliper_y
        z_cal = -(ht / 2 + 0.04 + sz / 2)
        extra += f"""
  <!-- Aft flange: coarse rotor + sided caliper proxies (keep tire from clipping through brakes). -->
  <link name="brake_rotor_proxy">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.5"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.06"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{_fmt(rr)}" length="{_fmt(rotor_len)}"/>
      </geometry>
      <material name="rotor_gray"><color rgba="0.35 0.35 0.37 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{_fmt(rr * 0.99)}" length="{_fmt(rotor_len)}"/>
      </geometry>
    </collision>
  </link>
  <joint name="brake_rotor_joint" type="fixed">
    <parent link="hub_mount"/>
    <child link="brake_rotor_proxy"/>
    <origin xyz="0 0 {_fmt(z_rotor)}" rpy="0 0 0"/>
  </joint>
  <link name="brake_caliper_proxy">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.015"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{_fmt(sx)} {_fmt(sy)} {_fmt(sz)}"/>
      </geometry>
      <material name="caliper_paint"><color rgba="0.45 0.48 0.52 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{_fmt(sx * 0.99)} {_fmt(sy * 0.99)} {_fmt(sz * 0.99)}"/>
      </geometry>
    </collision>
  </link>
  <joint name="brake_caliper_joint" type="fixed">
    <parent link="hub_mount"/>
    <child link="brake_caliper_proxy"/>
    <origin xyz="0 {_fmt(cy)} {_fmt(z_cal)}" rpy="0 0 0"/>
  </joint>"""

    xml = f"""<?xml version="1.0"?>
<robot name="truck_wheel_station">

<!-- Base = hub flange; local +Z protrudes bolts (toward mating tire). -->
  <link name="hub_mount">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{args.hub_mass}"/>
      <inertia ixx="{hx}" ixy="0.0" ixz="0.0" iyy="{hx}" iyz="0.0" izz="{0.5 * args.hub_mass * args.hub_radius ** 2}" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{args.hub_radius}" length="{args.hub_thickness}"/>
      </geometry>
      <material name="steel"><color rgba="0.4 0.4 0.45 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{args.hub_radius}" length="{args.hub_thickness}"/>
      </geometry>
    </collision>
  </link>
{''.join(bolts_xml)}{extra}

</robot>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml.strip() + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}  bolts={n}  phase_deg={args.bolt_pattern_phase_deg:g} "
        f"pilot={'off' if args.no_hub_pilot else 'on'} "
        f"brake_proxy={'off' if args.no_brake_proxy else 'on'} "
        f"collision_radius={r_col:.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
