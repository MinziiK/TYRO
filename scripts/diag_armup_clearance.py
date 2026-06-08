#!/usr/bin/env python3
"""Does an 'arm-up' IK branch let Robot B clear Robot A at every bolt?

For each bolt we enumerate MANY roll-free IK branches (roll x diverse seeds,
including elbow-up biased seeds) at the seated + approach-corridor points,
keep the reachable ones, and report the branch with the MAXIMUM B<->A
clearance (not the min-position-error one the planner usually picks). If the
max clearance is positive for every bolt, a collision-free configuration
EXISTS and we only need to bias the policy toward it.

Also reports the elbow/forearm world height of the best branch so we can see
whether 'arm up' is what wins.

Supports the same geometry overrides as diag_b_a_clearance:
  --b-dz --b-dy --tool-length
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _make_short_tool_urdf(base_urdf: str, length: float) -> str:
    txt = Path(base_urdf).read_text()
    half = length / 2.0
    txt = re.sub(r'(<cylinder length=")0\.30(" radius="0\.02"/>)',
                 rf'\g<1>{length:.4f}\g<2>', txt)
    txt = re.sub(r'(<origin rpy="0 0 0" xyz="0 0 )0\.15("/>)',
                 rf'\g<1>{half:.4f}\g<2>', txt)
    txt = re.sub(r'(<origin rpy="0 0 0" xyz="0 0 )0\.30("/>)',
                 rf'\g<1>{length:.4f}\g<2>', txt)
    fd = tempfile.NamedTemporaryFile(
        prefix=f"ur10e_tool{int(length*100)}_", suffix=".urdf",
        delete=False, dir=str(Path(base_urdf).parent))
    fd.write(txt.encode())
    fd.close()
    return fd.name


def _quat_from_z_roll(z, roll):
    z = np.asarray(z, float) / max(np.linalg.norm(z), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = np.cross(ref, z); x0 /= max(np.linalg.norm(x0), 1e-9)
    y0 = np.cross(z, x0)
    cr, sr = np.cos(roll), np.sin(roll)
    xr, yr = cr * x0 + sr * y0, -sr * x0 + cr * y0
    m = np.column_stack([xr, yr, z])
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s]
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s]
    s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
            (m[1, 0] - m[0, 1]) / s]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dz", type=float, default=0.0)
    ap.add_argument("--b-dy", type=float, default=0.0)
    ap.add_argument("--tool-length", type=float, default=None)
    ap.add_argument("--reach-tol", type=float, default=0.02)
    ap.add_argument("--align-tol-deg", type=float, default=8.0)
    ap.add_argument("--n-seed", type=int, default=12)
    ap.add_argument("--n-roll", type=int, default=16)
    args = ap.parse_args()

    cfg = make_env_config(
        stage=3, phase=1, nut_fastening_task=True,
        scene_layout="fanuc_spacious", terminate_on="never",
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
        contact_force_terminate_above=0.0,
    )
    if args.b_dz or args.b_dy:
        bx, by, bz = cfg.robot_B_base_pos
        cfg.robot_B_base_pos = (bx, by + args.b_dy, bz + args.b_dz)
    tmp_urdf = None
    if args.tool_length is not None:
        tmp_urdf = _make_short_tool_urdf(cfg.ur10e_urdf, float(args.tool_length))
        cfg.ur10e_urdf = tmp_urdf
        cfg.ur10e_nut_tool_length = float(args.tool_length)

    env = TyroEnv(cfg=cfg, render=False, seed=5000)
    env.set_nut_b_hotstart_alpha(0.0)
    env.reset(seed=5000)

    a_uid, b_uid = int(env.robot_A.uid), int(env.robot_B.uid)
    rb = env.robot_B
    lo, hi = rb.arm.lower, rb.arm.upper
    n = len(env.handles.bolts)
    L = float(getattr(cfg, "bolt_length", 0.10))
    rng = np.random.default_rng(3)

    # forearm link index (for height readout)
    fore_idx = None
    for li in range(p.getNumJoints(b_uid, physicsClientId=env.client)):
        if p.getJointInfo(b_uid, li, physicsClientId=env.client)[12].decode() \
                == "forearm_link":
            fore_idx = li
            break

    # Elbow-UP biased seeds for the UR10e (shoulder lifts, elbow bends up).
    up_seeds = [
        np.array([0.0, -2.2, -1.6, -1.0, 1.57, 0.0]),
        np.array([0.0, -2.6, -1.2, -1.5, 1.57, 0.0]),
        np.array([0.3, -2.4, -1.8, -0.5, 1.2, 0.0]),
        np.array([-0.3, -2.0, -1.4, -1.8, 1.9, 0.0]),
    ]

    axials = [0.5 * L + 0.20, 0.5 * L + 0.05, 0.0, -0.5 * L]
    print(f"[armup] B_base={cfg.robot_B_base_pos}  "
          f"tool={'%.2f' % args.tool_length if args.tool_length else '0.30(URDF)'}  "
          f"reach_tol={args.reach_tol*100:.0f}mm align_tol={args.align_tol_deg:.0f}deg")
    print(f"{'bolt':>4} {'maxClear(cm)':>12} {'foreZ(m)':>9} "
          f"{'shoulderLift/elbow(rad)':>24}  branch")

    per_bolt_max = []
    for i in range(n):
        want_z = -np.asarray(env.scene.bolt_axis(i), float)
        want_z /= max(np.linalg.norm(want_z), 1e-9)
        bolt_best = -9.0
        best_info = None
        for ax in axials:
            tgt = (np.asarray(env.scene.bolt_pose(i)[0], float)
                   + (-want_z) * 0.0 + want_z * 0.0)  # bolt center
            tgt = np.asarray(env.scene.bolt_pose(i)[0], float) \
                + np.asarray(env.scene.bolt_axis(i), float) \
                / max(np.linalg.norm(env.scene.bolt_axis(i)), 1e-9) * ax
            for ri in range(args.n_roll):
                quat = _quat_from_z_roll(want_z, 2 * np.pi * ri / args.n_roll)
                seeds = [rb.arm.rest] + up_seeds + [
                    rng.uniform(lo, hi) for _ in range(args.n_seed)]
                for seed in seeds:
                    ik = p.calculateInverseKinematics(
                        b_uid, rb.EE_LINK_INDEX, tgt.tolist(), quat,
                        lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                        jointRanges=rb.arm.range.tolist(),
                        restPoses=np.clip(seed, lo, hi).tolist(),
                        maxNumIterations=200, residualThreshold=1e-6,
                        physicsClientId=env.client)
                    q = np.clip(np.asarray(ik, float)[rb._ik_arm_slots], lo, hi)
                    for s, qq in zip(rb.arm.indices, q):
                        p.resetJointState(b_uid, int(s), float(qq),
                                          physicsClientId=env.client)
                    ee, eq = rb.ee_pose()
                    dp = float(np.linalg.norm(np.asarray(ee, float) - tgt))
                    if dp > args.reach_tol:
                        continue
                    gz = np.asarray(p.getMatrixFromQuaternion(list(eq)),
                                    float).reshape(3, 3)[:, 2]
                    ang = np.degrees(np.arccos(
                        np.clip(np.dot(gz, want_z), -1, 1)))
                    if ang > args.align_tol_deg:
                        continue
                    cps = p.getClosestPoints(b_uid, a_uid, distance=0.5,
                                             physicsClientId=env.client)
                    md = min((float(c[8]) for c in cps), default=0.5)
                    if md > bolt_best:
                        bolt_best = md
                        fz = (p.getLinkState(b_uid, fore_idx,
                                             physicsClientId=env.client)[0][2]
                              if fore_idx is not None else float("nan"))
                        best_info = (fz, q[1], q[2])
        per_bolt_max.append(bolt_best)
        if best_info is None:
            print(f"{i:>4} {'unreachable':>12}")
            continue
        fz, sl, el = best_info
        flag = "" if bolt_best > 0.02 else ("  <- tight" if bolt_best > 0
                                            else "  <== still collide")
        print(f"{i:>4} {bolt_best*100:>12.1f} {fz:>9.2f} "
              f"{sl:>11.2f}/{el:>11.2f}{flag}")

    env.close()
    if tmp_urdf is not None:
        try:
            Path(tmp_urdf).unlink()
        except OSError:
            pass
    mn = min(per_bolt_max)
    bad = [i for i, d in enumerate(per_bolt_max) if d <= 0]
    print(f"\n[armup] worst over all bolts (best-branch clearance): {mn*100:.1f} cm")
    if bad:
        print(f"[armup] bolts with NO collision-free branch: {bad}")
    else:
        print("[armup] EVERY bolt has a collision-free reaching branch -> "
              "feasible; bias the policy toward arm-up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
