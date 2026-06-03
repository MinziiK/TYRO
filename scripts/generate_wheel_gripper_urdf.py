#!/usr/bin/env python3
"""Generate a primitive wheel-gripper tool URDF for FANUC tool0.

The tool attaches at ``tool0`` via fixed joint; EE for IK is ``wheel_tool_tip``.

Usage::
    python scripts/generate_wheel_gripper_urdf.py
    python scripts/generate_wheel_gripper_urdf.py --output data/urdf/wheel_gripper/wheel_gripper.urdf
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def _fmt(x: float) -> str:
    if abs(x) < 1e-14:
        return "0"
    s = f"{x:.8g}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def build_wheel_gripper_xml(
    *,
    chuck_radius: float = 0.12,
    chuck_thickness: float = 0.08,
    pad_radius: float = 0.04,
    pad_length: float = 0.14,
    pad_gap: float = 0.52,
    tip_offset_z: float = 0.06,
    mass: float = 8.0,
) -> str:
    """Chuck disk + 3 radial pads; tip at bore contact plane (+Z from tool0)."""
    lines = [
        '<?xml version="1.0"?>',
        '<robot name="wheel_gripper">',
        "  <!-- Mount frame: parent FANUC tool0 -->",
        "  <link name=\"wg_base_link\">",
        "    <inertial>",
        f"      <origin xyz=\"0 0 {_fmt(chuck_thickness * 0.5)}\" rpy=\"0 0 0\"/>",
        f"      <mass value=\"{_fmt(mass)}\"/>",
        "      <inertia ixx=\"0.05\" ixy=\"0\" ixz=\"0\" iyy=\"0.05\" iyz=\"0\" izz=\"0.05\"/>",
        "    </inertial>",
        "  </link>",
        "  <link name=\"wg_chuck\">",
        "    <visual>",
        f"      <origin xyz=\"0 0 {_fmt(chuck_thickness * 0.5)}\" rpy=\"0 0 0\"/>",
        "      <geometry>",
        f"        <cylinder radius=\"{_fmt(chuck_radius)}\" length=\"{_fmt(chuck_thickness)}\"/>",
        "      </geometry>",
        "      <material name=\"grey\"><color rgba=\"0.45 0.47 0.50 1\"/></material>",
        "    </visual>",
        "    <collision>",
        f"      <origin xyz=\"0 0 {_fmt(chuck_thickness * 0.5)}\" rpy=\"0 0 0\"/>",
        "      <geometry>",
        f"        <cylinder radius=\"{_fmt(chuck_radius)}\" length=\"{_fmt(chuck_thickness)}\"/>",
        "      </geometry>",
        "    </collision>",
        "    <inertial>",
        f"      <origin xyz=\"0 0 {_fmt(chuck_thickness * 0.5)}\" rpy=\"0 0 0\"/>",
        f"      <mass value=\"{_fmt(mass * 0.6)}\"/>",
        "      <inertia ixx=\"0.02\" ixy=\"0\" ixz=\"0\" iyy=\"0.02\" iyz=\"0\" izz=\"0.02\"/>",
        "    </inertial>",
        "  </link>",
        f"  <joint name=\"wg_base-chuck\" type=\"fixed\">",
        f"    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>",
        "    <parent link=\"wg_base_link\"/>",
        "    <child link=\"wg_chuck\"/>",
        "  </joint>",
    ]
    for i in range(3):
        ang = i * (2.0 * math.pi / 3.0)
        px = pad_gap * math.cos(ang)
        py = pad_gap * math.sin(ang)
        pad_name = f"wg_pad_{i}"
        lines += [
            f"  <link name=\"{pad_name}\">",
            "    <visual>",
            f"      <origin xyz=\"0 0 {_fmt(pad_length * 0.5)}\" rpy=\"0 0 0\"/>",
            "      <geometry>",
            f"        <cylinder radius=\"{_fmt(pad_radius)}\" length=\"{_fmt(pad_length)}\"/>",
            "      </geometry>",
            "      <material name=\"yellow\"><color rgba=\"0.85 0.70 0.10 1\"/></material>",
            "    </visual>",
            "    <collision>",
            f"      <origin xyz=\"0 0 {_fmt(pad_length * 0.5)}\" rpy=\"0 0 0\"/>",
            "      <geometry>",
            f"        <cylinder radius=\"{_fmt(pad_radius)}\" length=\"{_fmt(pad_length)}\"/>",
            "      </geometry>",
            "    </collision>",
            "    <inertial>",
            f"      <origin xyz=\"0 0 {_fmt(pad_length * 0.5)}\" rpy=\"0 0 0\"/>",
            f"      <mass value=\"{_fmt(mass * 0.13)}\"/>",
            "      <inertia ixx=\"0.005\" ixy=\"0\" ixz=\"0\" iyy=\"0.005\" iyz=\"0\" izz=\"0.005\"/>",
            "    </inertial>",
            "  </link>",
            f"  <joint name=\"{pad_name}-joint\" type=\"fixed\">",
            f"    <origin xyz=\"{_fmt(px)} {_fmt(py)} {_fmt(chuck_thickness)}\" "
            f"rpy=\"0 0 {_fmt(ang)}\"/>",
            "    <parent link=\"wg_chuck\"/>",
            f"    <child link=\"{pad_name}\"/>",
            "  </joint>",
        ]
    lines += [
        "  <!-- Tool tip: bore contact / grasp anchor (world +Z from tool0 when palm-up) -->",
        "  <link name=\"wheel_tool_tip\">",
        "    <inertial>",
        f"      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>",
        "      <mass value=\"0.001\"/>",
        "      <inertia ixx=\"1e-6\" ixy=\"0\" ixz=\"0\" iyy=\"1e-6\" iyz=\"0\" izz=\"1e-6\"/>",
        "    </inertial>",
        "  </link>",
        "  <joint name=\"wg_chuck-tip\" type=\"fixed\">",
        f"    <origin xyz=\"0 0 {_fmt(chuck_thickness + tip_offset_z)}\" rpy=\"0 0 0\"/>",
        "    <parent link=\"wg_chuck\"/>",
        "    <child link=\"wheel_tool_tip\"/>",
        "  </joint>",
        "</robot>",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--output",
        type=Path,
        default=repo / "data" / "urdf" / "wheel_gripper" / "wheel_gripper.urdf",
    )
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_wheel_gripper_xml(), encoding="utf-8")
    print(f"[wheel_gripper] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
