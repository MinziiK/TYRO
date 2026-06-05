#!/usr/bin/env python3
"""Validate a concrete raised-hub layout end-to-end (kinematics only).

Checks, for a candidate (fanuc_base, hub, pickup, ur10e_base):
  1. FANUC palm-up PICKUP IK  (pickup + (0,0,-R), tool +Z = world +Z)
  2. FANUC palm-up MOUNT  IK  (hub    + (0,0,-R), tool +Z = world +Z)
  3. UR10e worst-bolt reach   (10 studs on the hub PCD, perpendicular to -Y)
     swept over a pedestal height so we can pick the lowest base_z that
     reaches every bolt.

Pure DIRECT kinematics, no full scene — fast and side-effect free.
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

from src.config import EnvConfig, apply_fanuc_spacious_layout  # noqa: E402
from src.env.robots import (  # noqa: E402
    make_robot_a, make_robot_b, robot_a_lock_quaternion,
)

FANUC_REACH = 2.65
UR10E_REACH = 1.30


def settle_ee(rob, cid, target_pos, target_quat=None, steps=250, restarts=3):
    """IK + settle. If target_quat is None, position-only IK. Returns
    (ee_pos, pos_err, ang_err_deg|nan)."""
    best = None
    seed = rob.arm.rest.copy()
    for _ in range(restarts):
        kw = dict(
            lowerLimits=rob.arm.lower.tolist(),
            upperLimits=rob.arm.upper.tolist(),
            jointRanges=rob.arm.range.tolist(),
            restPoses=seed.tolist(),
            maxNumIterations=300,
            residualThreshold=1e-5,
            physicsClientId=cid,
        )
        if target_quat is not None:
            ik = p.calculateInverseKinematics(
                rob.uid, rob.EE_LINK_INDEX, list(target_pos),
                list(target_quat), **kw)
        else:
            ik = p.calculateInverseKinematics(
                rob.uid, rob.EE_LINK_INDEX, list(target_pos), **kw)
        ik = np.asarray(ik, dtype=np.float64)
        arm_targets = np.clip(ik[rob._ik_arm_slots], rob.arm.lower, rob.arm.upper)
        for _ in range(steps):
            p.setJointMotorControlArray(
                rob.uid, rob.arm.indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=arm_targets.tolist(),
                forces=[400.0] * rob.arm.n,
                positionGains=[1.0] * rob.arm.n,
                velocityGains=[1.0] * rob.arm.n,
                physicsClientId=cid,
            )
            p.stepSimulation(physicsClientId=cid)
        ee_pos, ee_quat = rob.ee_pose()
        ee_pos = np.asarray(ee_pos, float)
        perr = float(np.linalg.norm(ee_pos - np.asarray(target_pos, float)))
        if target_quat is not None:
            dq = abs(float(np.dot(ee_quat, np.asarray(target_quat, float))))
            aerr = float(np.degrees(2 * np.arccos(min(1.0, dq))))
        else:
            aerr = float("nan")
        if best is None or perr < best[1]:
            best = (ee_pos, perr, aerr)
        seed = arm_targets
    return best


def bolt_positions(hub, rc, n=10):
    """n studs on the PCD, in the plane perpendicular to hub axis (-Y) → X-Z."""
    hub = np.asarray(hub, float)
    pts = []
    for i in range(n):
        th = 2 * np.pi * i / n
        pts.append(hub + np.array([rc * np.cos(th), 0.0, rc * np.sin(th)]))
    return pts


def fanuc_checks(fanuc_base, hub, pickup, R):
    cfg = EnvConfig()
    apply_fanuc_spacious_layout(cfg)
    cfg.robot_A_base_pos = tuple(fanuc_base)
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, 0, physicsClientId=cid)
    rob = make_robot_a(cid, cfg)
    palm = robot_a_lock_quaternion(rob)
    base = np.asarray(fanuc_base, float)

    pick_tgt = np.asarray(pickup, float) + np.array([0, 0, -R])
    mnt_tgt = np.asarray(hub, float) + np.array([0, 0, -R])
    _, pe, pa = settle_ee(rob, cid, pick_tgt, palm)
    _, me, ma = settle_ee(rob, cid, mnt_tgt, palm)
    p.disconnect(cid)
    pr = np.linalg.norm(pick_tgt - base)
    mr = np.linalg.norm(mnt_tgt - base)
    print("FANUC (Robot A) palm-up checks  base =", tuple(round(x, 2) for x in fanuc_base))
    print(f"  PICKUP tgt={np.round(pick_tgt,3)} reach={100*pr/FANUC_REACH:3.0f}% "
          f"err={pe*100:5.2f}cm/{pa:4.1f}d  [{'OK' if pe<0.02 and pa<5 else 'BAD'}]")
    print(f"  MOUNT  tgt={np.round(mnt_tgt,3)} reach={100*mr/FANUC_REACH:3.0f}% "
          f"err={me*100:5.2f}cm/{ma:4.1f}d  [{'OK' if me<0.02 and ma<5 else 'BAD'}]")


def ur10e_worst_bolt(rob, cid, bolts, base):
    worst = 0.0
    worst_reach = 0.0
    for b in bolts:
        _, e, _ = settle_ee(rob, cid, b, None, steps=220, restarts=4)
        worst = max(worst, e)
        worst_reach = max(worst_reach, float(np.linalg.norm(b - base)))
    return worst, worst_reach


def ur10e_checks(hub, rc, base_xy, base_z_grid):
    bolts = bolt_positions(hub, rc, 10)
    print("\nUR10e (Robot B) bolt-reach sweep  base_xy =",
          tuple(round(x, 2) for x in base_xy))
    print("  (position-only IK to 10 studs on the PCD; worst-bolt err)")
    for bz in base_z_grid:
        cfg = EnvConfig()
        apply_fanuc_spacious_layout(cfg)
        cfg.robot_B_base_pos = (base_xy[0], base_xy[1], bz)
        cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=cid)
        rob = make_robot_b(cid, cfg)
        base = np.array([base_xy[0], base_xy[1], bz], float)
        worst, worst_reach = ur10e_worst_bolt(rob, cid, bolts, base)
        p.disconnect(cid)
        tag = "OK" if worst < 0.02 else "miss"
        print(f"  base_z={bz:+.2f}  worst-bolt reach={100*worst_reach/UR10E_REACH:3.0f}% "
              f"worst_err={worst*100:5.2f}cm  [{tag}]")


def ur10e_grid(hub, rc, dx_grid, dy_grid, z_grid):
    """Sweep Robot B base over (hub_x+dx, hub_y+dy, z); find configs that reach
    every bolt. dy<0 places B on the -Y (outboard / FANUC) side of the hub."""
    bolts = bolt_positions(hub, rc, 10)
    hub = np.asarray(hub, float)
    print("\nUR10e (Robot B) PLACEMENT grid  (find base that reaches all bolts)")
    print(f"  hub={tuple(round(x,2) for x in hub)}  bolt PCD r={rc}")
    feasible = []
    for dx in dx_grid:
        for dy in dy_grid:
            for bz in z_grid:
                bx, by = hub[0] + dx, hub[1] + dy
                cfg = EnvConfig()
                apply_fanuc_spacious_layout(cfg)
                cfg.robot_B_base_pos = (bx, by, bz)
                cid = p.connect(p.DIRECT)
                p.setGravity(0, 0, 0, physicsClientId=cid)
                rob = make_robot_b(cid, cfg)
                base = np.array([bx, by, bz], float)
                worst, worst_reach = ur10e_worst_bolt(rob, cid, bolts, base)
                p.disconnect(cid)
                if worst < 0.02:
                    feasible.append((bx, by, bz, worst_reach, worst))
                    print(f"  base=({bx:+.2f},{by:+.2f},{bz:+.2f}) "
                          f"reach={100*worst_reach/UR10E_REACH:3.0f}% "
                          f"worst_err={worst*100:4.1f}cm  [OK]")
    if not feasible:
        print("  (no all-bolt-reachable base in grid)")
    else:
        feasible.sort(key=lambda r: abs(r[3] / UR10E_REACH - 0.65))
        bx, by, bz, wr, we = feasible[0]
        print(f"  >>> best (≈65% reach): base=({bx:.2f},{by:.2f},{bz:.2f}) "
              f"reach={100*wr/UR10E_REACH:.0f}% err={we*100:.1f}cm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanuc-base", type=float, nargs=3, default=[-1.30, -1.30, -0.90])
    ap.add_argument("--hub", type=float, nargs=3, default=[-0.70, 0.30, 1.30])
    ap.add_argument("--pickup", type=float, nargs=3, default=[-3.10, -1.30, 1.30])
    ap.add_argument("--ur10e-base-xy", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--ur10e-base-z", type=float, nargs="*",
                    default=[0.0, 0.4, 0.7, 1.0, 1.2, 1.4])
    ap.add_argument("--bolt-rc", type=float, default=0.1675)
    ap.add_argument("--b-grid", action="store_true",
                    help="Sweep Robot B base placement to reach all bolts.")
    args = ap.parse_args()

    R = 0.525
    print("=" * 84)
    print("CANDIDATE LAYOUT VALIDATION")
    print(f"  fanuc_base={args.fanuc_base}  hub={args.hub}  pickup={args.pickup}")
    print("=" * 84)
    fanuc_checks(args.fanuc_base, args.hub, args.pickup, R)
    if args.b_grid:
        ur10e_grid(args.hub, args.bolt_rc,
                   dx_grid=[-0.1, 0.0, 0.1, 0.2],
                   dy_grid=[-0.7, -0.6, -0.5, -0.4],
                   z_grid=[0.6, 0.8, 1.0, 1.2, 1.3])
    else:
        ur10e_checks(args.hub, args.bolt_rc, args.ur10e_base_xy, args.ur10e_base_z)
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
