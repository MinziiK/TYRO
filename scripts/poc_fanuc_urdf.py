"""PoC step 1: FANUC R-2000iC/210F URDF for PyBullet (TYRO).

R-2000iD has no ROS-Industrial support; R-2000iC/210F is the closest
open-source match (reach ~2.65 m, payload 210 kg).

Usage
-----
    python scripts/poc_fanuc_urdf.py --fetch          # clone ROS-Industrial fanuc
    python scripts/poc_fanuc_urdf.py --convert      # xacro -> urdf (needs xacro)
    python scripts/poc_fanuc_urdf.py --load         # headless PyBullet load
    python scripts/poc_fanuc_urdf.py --load --gui   # GUI smoke test

Outputs under ``data/urdf/fanuc_r2000ic/``.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FANUC_REPO = _REPO / "data" / "urdf" / "fanuc_ros"
IC_PKG = FANUC_REPO / "fanuc_r2000ic_support"
OUT_DIR = _REPO / "data" / "urdf" / "fanuc_r2000ic"
XACRO_TOP = IC_PKG / "urdf" / "r2000ic210f.xacro"
URDF_OUT = OUT_DIR / "r2000ic210f.urdf"


def fetch_fanuc() -> None:
    """Shallow clone ROS-Industrial fanuc (sparse: iC support + resources)."""
    if IC_PKG.is_dir():
        print(f"[fetch] already present: {IC_PKG}")
        return
    FANUC_REPO.parent.mkdir(parents=True, exist_ok=True)
    print("[fetch] cloning ros-industrial/fanuc (noetic-devel, sparse)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "-b", "noetic-devel",
         "https://github.com/ros-industrial/fanuc.git", str(FANUC_REPO)],
        check=True, cwd=str(_REPO),
    )
    subprocess.run(
        ["git", "sparse-checkout", "set",
         "fanuc_r2000ic_support", "fanuc_resources"],
        check=True, cwd=str(FANUC_REPO),
    )
    print(f"[fetch] OK: {IC_PKG}")


def _rewrite_package_uris(text: str, pkg_root: Path) -> str:
    """Replace package://fanuc_r2000ic_support/... with relative paths."""

    def repl(m: re.Match) -> str:
        rel = m.group(1).lstrip("/")
        return str((pkg_root / rel).as_posix())

    text = re.sub(
        r'package://fanuc_r2000ic_support/([^"\']+)',
        lambda m: repl(m),
        text,
    )
    text = re.sub(
        r'package://fanuc_resources/([^"\']+)',
        lambda m: str((FANUC_REPO / "fanuc_resources" / m.group(1).lstrip("/")).as_posix()),
        text,
    )
    return text


def _resolve_find(text: str) -> str:
    ic = IC_PKG.as_posix()
    res = (FANUC_REPO / "fanuc_resources").as_posix()
    return (
        text.replace("$(find fanuc_r2000ic_support)", ic)
        .replace("$(find fanuc_resources)", res)
    )


def _stage_xacro_tree() -> Path:
    """Stage xacro files with absolute includes (no ROS $(find))."""
    staged = OUT_DIR / "_xacro_staging"
    staged.mkdir(parents=True, exist_ok=True)
    res_urdf = FANUC_REPO / "fanuc_resources" / "urdf"
    colours = staged / "common_colours.xacro"
    materials = staged / "common_materials.xacro"
    macro = staged / "r2000ic210f_macro.xacro"
    colours.write_text(
        (res_urdf / "common_colours.xacro").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    mat = (res_urdf / "common_materials.xacro").read_text(encoding="utf-8")
    mat = mat.replace(
        "$(find fanuc_resources)/urdf/common_colours.xacro",
        "common_colours.xacro",
    )
    materials.write_text(mat, encoding="utf-8")
    mac = (IC_PKG / "urdf" / "r2000ic210f_macro.xacro").read_text(encoding="utf-8")
    mac = mac.replace(
        "$(find fanuc_resources)/urdf/common_materials.xacro",
        "common_materials.xacro",
    )
    # xacro expands package:// via $(find pkg) — replace before processing (no ROS).
    mac = mac.replace("package://fanuc_r2000ic_support/", f"{IC_PKG.as_posix()}/")
    macro.write_text(mac, encoding="utf-8")
    wrapper = staged / "r2000ic210f_poc.xacro"
    wrapper.write_text(
        """<?xml version="1.0"?>
<robot name="fanuc_r2000ic210f" xmlns:xacro="http://wiki.ros.org/xacro">
  <xacro:include filename="r2000ic210f_macro.xacro"/>
  <xacro:fanuc_r2000ic210f prefix=""/>
</robot>
""",
        encoding="utf-8",
    )
    return wrapper


def convert_xacro() -> Path:
    """Expand r2000ic210f.xacro to a flat URDF for PyBullet."""
    if not IC_PKG.is_dir():
        raise FileNotFoundError(f"Missing {IC_PKG}; run with --fetch first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import xacro  # type: ignore
    except ImportError as exc:
        print("[convert] `xacro` Python module not installed.")
        print("  Install: pip install xacro")
        raise SystemExit(1) from exc
    wrapper = _stage_xacro_tree()
    import os
    os.chdir(wrapper.parent)
    doc = xacro.process_file(wrapper.name)
    xml = doc.toprettyxml(indent="  ")
    xml = _rewrite_package_uris(xml, IC_PKG)
    # Absolute mesh paths from staging → repo-relative (portable across machines).
    rel_ic = Path(os.path.relpath(IC_PKG, OUT_DIR)).as_posix()
    xml = xml.replace(IC_PKG.as_posix(), rel_ic)
    URDF_OUT.write_text(xml, encoding="utf-8")
    print(f"[convert] wrote {URDF_OUT} ({URDF_OUT.stat().st_size // 1024} KB)")
    return URDF_OUT


# A presentable non-zero pose (rad) so the arm is extended, not folded.
DEMO_POSE = [0.0, -0.5, 0.4, 0.0, -0.9, 0.0]


def load_pybullet(gui: bool = False, save_image: str | None = None,
                  pose: bool = True) -> None:
    """Load converted URDF in PyBullet; print joint/link summary.

    If ``save_image`` is given, render a camera frame to PNG (works headless).
    """
    import pybullet as p
    import pybullet_data

    if not URDF_OUT.is_file():
        raise FileNotFoundError(f"Missing {URDF_OUT}; run --fetch --convert first.")
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(str(IC_PKG), physicsClientId=cid)
    p.setAdditionalSearchPath(str(OUT_DIR), physicsClientId=cid)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    uid = p.loadURDF(
        str(URDF_OUT),
        basePosition=[0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
        physicsClientId=cid,
    )
    if uid < 0:
        raise RuntimeError("loadURDF failed — check mesh paths in URDF")
    nj = p.getNumJoints(uid, physicsClientId=cid)
    movable = [j for j in range(nj)
               if p.getJointInfo(uid, j, physicsClientId=cid)[2] != p.JOINT_FIXED]
    print(f"[load] uid={uid}  joints={nj}  movable={len(movable)}")
    for j in range(nj):
        info = p.getJointInfo(uid, j, physicsClientId=cid)
        name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
        if info[2] != p.JOINT_FIXED:
            lo, hi = info[8], info[9]
            print(f"  j{j:2d} {name:30s} limits=[{lo:+.2f}, {hi:+.2f}]")
    if pose:
        for k, j in enumerate(movable):
            if k < len(DEMO_POSE):
                p.resetJointState(uid, j, DEMO_POSE[k], physicsClientId=cid)
    ls = p.getLinkState(uid, nj - 1, computeForwardKinematics=True,
                        physicsClientId=cid)
    print(f"[load] tool0 pos={tuple(round(v, 3) for v in ls[4])}")

    cam = dict(cameraDistance=4.0, cameraYaw=50, cameraPitch=-25,
               cameraTargetPosition=[0, 0, 1.0])
    if gui:
        p.resetDebugVisualizerCamera(**cam, physicsClientId=cid)
    if save_image:
        view = p.computeViewMatrixFromYawPitchRoll(
            cam["cameraTargetPosition"], cam["cameraDistance"],
            cam["cameraYaw"], cam["cameraPitch"], 0, 2)
        proj = p.computeProjectionMatrixFOV(60, 1.33, 0.1, 100)
        w, h = 960, 720
        img = p.getCameraImage(w, h, view, proj,
                               renderer=p.ER_TINY_RENDERER,
                               physicsClientId=cid)
        try:
            from PIL import Image
            import numpy as np
            rgb = np.reshape(img[2], (h, w, 4))[:, :, :3].astype("uint8")
            Image.fromarray(rgb).save(save_image)
            print(f"[load] saved screenshot -> {save_image}")
        except ImportError:
            print("[load] Pillow not installed; skip PNG (pip install pillow)")
    if gui:
        print("[load] GUI open — close window to exit.")
        import time
        for _ in range(100000):
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1.0 / 120.0)
    p.disconnect(cid)
    print("[load] OK")


def main() -> int:
    ap = argparse.ArgumentParser(description="FANUC R-2000iC URDF PoC for TYRO")
    ap.add_argument("--fetch", action="store_true", help="Clone ROS-Industrial fanuc")
    ap.add_argument("--convert", action="store_true", help="xacro -> urdf")
    ap.add_argument("--load", action="store_true", help="PyBullet load test")
    ap.add_argument("--gui", action="store_true", help="With --load, open GUI")
    ap.add_argument("--save-image", metavar="PNG", default=None,
                    help="Render a screenshot to PNG (works headless)")
    ap.add_argument("--zero-pose", action="store_true",
                    help="Keep folded zero pose instead of demo pose")
    args = ap.parse_args()
    if not any([args.fetch, args.convert, args.load]):
        ap.print_help()
        return 0
    if args.fetch:
        fetch_fanuc()
    if args.convert:
        convert_xacro()
    if args.load:
        load_pybullet(gui=args.gui, save_image=args.save_image,
                      pose=not args.zero_pose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
