#!/usr/bin/env python3
"""Merge FANUC r2000ic210f URDF with wheel_gripper at tool0.

Usage::
    python scripts/poc_fanuc_urdf.py --fetch --convert
    python scripts/generate_wheel_gripper_urdf.py
    python scripts/merge_fanuc_wheeltool.py
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
FANUC_URDF = _REPO / "data" / "urdf" / "fanuc_r2000ic" / "r2000ic210f.urdf"
GRIPPER_URDF = _REPO / "data" / "urdf" / "wheel_gripper" / "wheel_gripper.urdf"
OUT = _REPO / "data" / "urdf" / "fanuc_r2000ic" / "r2000ic210f_wheeltool.urdf"


def _prefix_links(root: ET.Element, prefix: str) -> None:
    rename: dict[str, str] = {}
    for link in root.findall("link"):
        old = link.get("name", "")
        if old:
            new = f"{prefix}{old}"
            rename[old] = new
            link.set("name", new)
    for joint in root.findall("joint"):
        old = joint.get("name", "")
        if old:
            joint.set("name", f"{prefix}{old}")
        for tag in ("parent", "child"):
            el = joint.find(tag)
            if el is not None and el.get("link") in rename:
                el.set("link", rename[el.get("link", "")])


def merge() -> Path:
    if not FANUC_URDF.is_file():
        raise FileNotFoundError(f"Missing {FANUC_URDF}; run poc_fanuc_urdf.py --convert")
    if not GRIPPER_URDF.is_file():
        raise FileNotFoundError(f"Missing {GRIPPER_URDF}; run generate_wheel_gripper_urdf.py")

    fanuc = ET.parse(FANUC_URDF).getroot()
    grip = ET.parse(GRIPPER_URDF).getroot()
    # Gripper links already use wg_* names — no extra prefix.

    # Remove duplicate tool0 link from fanuc (re-added below with mount).
    for link in list(fanuc.findall("link")):
        if link.get("name") == "tool0":
            fanuc.remove(link)
    for joint in list(fanuc.findall("joint")):
        if joint.get("name") == "link_6-tool0":
            fanuc.remove(joint)

    mount = ET.Element("joint", {"name": "tool0-wheel_gripper", "type": "fixed"})
    ET.SubElement(mount, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(mount, "parent", {"link": "tool0"})
    ET.SubElement(mount, "child", {"link": "wg_base_link"})
    ET.SubElement(mount, "axis", {"xyz": "0 0 1"})

    # Re-add minimal tool0 link for ROS-Industrial frame chain.
    tool0 = ET.Element("link", {"name": "tool0"})
    fanuc.append(tool0)
    j6t = ET.Element("joint", {"name": "link_6-tool0", "type": "fixed"})
    ET.SubElement(
        j6t, "origin",
        {"xyz": "0 0 0", "rpy": "3.141592653589793 -1.5707963267948966 0"},
    )
    ET.SubElement(j6t, "parent", {"link": "link_6"})
    ET.SubElement(j6t, "child", {"link": "tool0"})
    fanuc.append(j6t)
    fanuc.append(mount)

    for child in list(grip):
        if child.tag in ("link", "joint"):
            fanuc.append(child)

    fanuc.set("name", "fanuc_r2000ic210f_wheeltool")
    xml = ET.tostring(fanuc, encoding="unicode")
    xml = '<?xml version="1.0"?>\n' + xml
    OUT.write_text(xml, encoding="utf-8")
    print(f"[merge] wrote {OUT}")
    return OUT


def main() -> int:
    merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
