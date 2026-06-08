#!/usr/bin/env python3
"""Measure Robot-B ↔ Robot-A clearance per bolt at the seated (INSERT) pose.

Robot A is frozen at its mount-hold pose (holding the tire). For each bolt we
drive Robot B to the precomputed coaxial INSERT joint vector (tool_tip at the
hub-face base) and report the closest distance between ANY B link and ANY A
link — negative = penetration. This pinpoints which bolts collide with A and
which link pair is responsible, so we can reason about tool-length / B-height
fixes quantitatively.

Optional config sweeps:
  --tool-length L     override ur10e_nut_tool_length (needs matching URDF;
                      here we only shift the geometric insert target, see note)
  --b-dz DZ           raise/lower Robot B base by DZ (m)
  --b-dy DY           shift Robot B base in Y by DY (m)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _make_short_tool_urdf(base_urdf: str, length: float) -> str:
    """Write a temp URDF identical to ``base_urdf`` but with the nut-runner
    socket re-sized to ``length`` m (cylinder + tool_tip TCP offset). Returns
    the temp path. This makes the tool-length test geometrically real (the
    collision mesh and IK TCP both move), unlike toggling the config number.
    """
    import re
    import tempfile

    txt = Path(base_urdf).read_text()
    half = length / 2.0
    # nut_runner cylinder: length="0.30" -> length, origin xyz="0 0 0.15" -> half
    txt = re.sub(r'(<cylinder length=")0\.30(" radius="0\.02"/>)',
                 rf'\g<1>{length:.4f}\g<2>', txt)
    txt = re.sub(r'(<origin rpy="0 0 0" xyz="0 0 )0\.15("/>)',
                 rf'\g<1>{half:.4f}\g<2>', txt)
    # tool_tip TCP offset: xyz="0 0 0.30" -> length
    txt = re.sub(r'(<origin rpy="0 0 0" xyz="0 0 )0\.30("/>)',
                 rf'\g<1>{length:.4f}\g<2>', txt)
    fd = tempfile.NamedTemporaryFile(
        prefix=f"ur10e_tool{int(length*100)}_", suffix=".urdf",
        delete=False, dir=str(Path(base_urdf).parent))
    fd.write(txt.encode())
    fd.close()
    return fd.name


def _link_names(uid: int, client: int) -> dict[int, str]:
    names = {-1: "base"}
    for li in range(p.getNumJoints(uid, physicsClientId=client)):
        info = p.getJointInfo(uid, li, physicsClientId=client)
        names[li] = info[12].decode()
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dz", type=float, default=0.0)
    ap.add_argument("--b-dy", type=float, default=0.0)
    ap.add_argument("--tool-length", type=float, default=None)
    ap.add_argument("--seed", type=int, default=5000)
    args = ap.parse_args()

    overrides = dict(
        nut_fastening_task=True,
        scene_layout="fanuc_spacious",
        terminate_on="never",
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
        contact_force_terminate_above=0.0,
    )
    cfg = make_env_config(stage=3, phase=1, **overrides)

    if args.b_dz or args.b_dy:
        bx, by, bz = cfg.robot_B_base_pos
        cfg.robot_B_base_pos = (bx, by + args.b_dy, bz + args.b_dz)
        print(f"[probe] Robot B base -> {cfg.robot_B_base_pos}")
    tmp_urdf = None
    if args.tool_length is not None:
        tmp_urdf = _make_short_tool_urdf(cfg.ur10e_urdf, float(args.tool_length))
        cfg.ur10e_urdf = tmp_urdf
        cfg.ur10e_nut_tool_length = float(args.tool_length)
        print(f"[probe] tool length -> {args.tool_length:.2f} m "
              f"(real URDF: {Path(tmp_urdf).name})")

    env = TyroEnv(cfg=cfg, render=False, seed=args.seed)
    env.set_nut_b_hotstart_alpha(0.0)
    env.reset(seed=args.seed)

    a_uid = int(env.robot_A.uid)
    b_uid = int(env.robot_B.uid)
    a_names = _link_names(a_uid, env.client)
    b_names = _link_names(b_uid, env.client)
    rb = env.robot_B
    n = len(env.handles.bolts)

    print(f"[probe] hub={np.round(env.scene.hub_pose()[0],3)}  "
          f"A_base={cfg.robot_A_base_pos}  B_base={cfg.robot_B_base_pos}")
    L = float(getattr(cfg, "bolt_length", 0.10))
    # Sample the approach CORRIDOR per bolt: roll-free IK at several axial
    # offsets from staging (outside the tip) to the seated base, then take the
    # min B<->A clearance over the corridor. This captures transit collisions
    # the seated-only pose misses.
    axials = [0.5 * L + 0.25, 0.5 * L + 0.12, 0.5 * L + 0.04,
              0.0, -0.5 * L]
    print(f"[probe] bolts={n}  min B<->A clearance over approach corridor "
          f"({len(axials)} samples/bolt)")
    print(f"{'bolt':>4} {'minClear(cm)':>12}  culprit B-link <-> A-link")

    worst = []
    for i in range(n):
        want_z = -env._nut_axis_unit(i)
        bolt_min, bl_w, al_w = 9.0, "-", "-"
        for ax in axials:
            tgt_pos = env._nut_point_on_axis(i, ax)
            try:
                q = env._ik_b_rollfree(tgt_pos, want_z)
            except Exception:
                continue
            if q is None:
                continue
            for s, qq in zip(rb.arm.indices, q):
                p.resetJointState(b_uid, int(s), float(qq),
                                  physicsClientId=env.client)
            cps = p.getClosestPoints(b_uid, a_uid, distance=0.6,
                                     physicsClientId=env.client)
            if not cps:
                continue
            cp = min(cps, key=lambda c: float(c[8]))
            md = float(cp[8])
            if md < bolt_min:
                bolt_min = md
                bl_w = b_names.get(cp[3], str(cp[3]))
                al_w = a_names.get(cp[4], str(cp[4]))
        flag = "  <== COLLISION" if bolt_min < 0 else (
            "  <- tight" if bolt_min < 0.03 else "")
        print(f"{i:>4} {bolt_min*100:>12.1f}  {bl_w} <-> {al_w}{flag}")
        worst.append((i, bolt_min))

    env.close()
    if tmp_urdf is not None:
        try:
            Path(tmp_urdf).unlink()
        except OSError:
            pass
    coll = [i for i, dd in worst if dd < 0]
    tight = [i for i, dd in worst if 0 <= dd < 0.03]
    mn = min(dd for _, dd in worst)
    print(f"\n[probe] colliding bolts (<0): {coll}")
    print(f"[probe] tight bolts (<3cm):   {tight}")
    print(f"[probe] GLOBAL min corridor clearance: {mn*100:.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
