#!/usr/bin/env python3
"""Fetch and convert UR10e URDF for PyBullet (TYRO Robot B).

Usage::
    python scripts/fetch_ur10e.py --fetch
    python scripts/fetch_ur10e.py --convert
    python scripts/fetch_ur10e.py --load --gui
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
UR_REPO = _REPO / "data" / "urdf" / "ur_ros"
UE_PKG = UR_REPO / "ur_e_description"
OUT_DIR = _REPO / "data" / "urdf" / "ur10e_robot"
URDF_OUT = OUT_DIR / "ur10e.urdf"


def fetch_ur() -> None:
    if UE_PKG.is_dir():
        print(f"[fetch] already present: {UE_PKG}")
        return
    UR_REPO.parent.mkdir(parents=True, exist_ok=True)
    print("[fetch] cloning ros-industrial/universal_robot (kinetic-devel, sparse)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "-b", "kinetic-devel",
         "https://github.com/ros-industrial/universal_robot.git", str(UR_REPO)],
        check=True, cwd=str(_REPO),
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "ur_e_description"],
        check=True, cwd=str(UR_REPO),
    )
    print(f"[fetch] OK: {UE_PKG}")


def _stage_xacro() -> Path:
    staged = OUT_DIR / "_xacro_staging"
    staged.mkdir(parents=True, exist_ok=True)
    src_urdf = UE_PKG / "urdf"
    for name in src_urdf.glob("*.xacro"):
        text = name.read_text(encoding="utf-8")
        text = text.replace(
            "$(find ur_e_description)/urdf/",
            "",
        ).replace("$(find ur_e_description)/", f"{UE_PKG.as_posix()}/")
        text = text.replace("package://ur_e_description/", f"{UE_PKG.as_posix()}/")
        (staged / name.name).write_text(text, encoding="utf-8")
    wrapper = staged / "ur10e_poc.xacro"
    wrapper.write_text(
        """<?xml version="1.0"?>
<robot name="ur10e" xmlns:xacro="http://wiki.ros.org/xacro">
  <xacro:include filename="ur10e.urdf.xacro"/>
  <xacro:ur10e_robot prefix="" joint_limited="false"/>
</robot>
""",
        encoding="utf-8",
    )
    return wrapper


def convert_xacro() -> Path:
    if not UE_PKG.is_dir():
        raise FileNotFoundError(f"Missing {UE_PKG}; run --fetch first.")
    try:
        import xacro  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install xacro") from exc
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = _stage_xacro()
    import os
    os.chdir(wrapper.parent)
    doc = xacro.process_file(wrapper.name)
    xml = doc.toprettyxml(indent="  ")
    rel = Path(os.path.relpath(UE_PKG, OUT_DIR)).as_posix()
    xml = re.sub(
        r'filename="([^"]*ur_e_description[^"]*)"',
        lambda m: f'filename="{rel}/{Path(m.group(1).split("ur_e_description/")[-1]).as_posix()}"'
        if "ur_e_description" in m.group(1) else m.group(0),
        xml,
    )
    xml = xml.replace(UE_PKG.as_posix(), rel)
    URDF_OUT.write_text(xml, encoding="utf-8")
    print(f"[convert] wrote {URDF_OUT} ({URDF_OUT.stat().st_size // 1024} KB)")
    return URDF_OUT


def load_pybullet(gui: bool = False) -> None:
    import pybullet as p
    import pybullet_data
    if not URDF_OUT.is_file():
        raise FileNotFoundError("run --fetch --convert first")
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(str(UE_PKG), physicsClientId=cid)
    p.setAdditionalSearchPath(str(OUT_DIR), physicsClientId=cid)
    uid = p.loadURDF(str(URDF_OUT), useFixedBase=True, physicsClientId=cid)
    print(f"[load] uid={uid} joints={p.getNumJoints(uid, physicsClientId=cid)}")
    if gui:
        import time
        for _ in range(6000):
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1.0 / 120.0)
    p.disconnect(cid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch_ur()
    if args.convert:
        convert_xacro()
    if args.load:
        load_pybullet(gui=args.gui)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
