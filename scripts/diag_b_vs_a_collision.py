#!/usr/bin/env python3
"""Measure Robot-B <-> Robot-A (and tire) clearance at each bolt's seated pose.

For the nut-fastening setup (A frozen at the mount-hold pose, tire bonded on
the hub) this drives Robot B's tool tip to each bolt along the bolt axis with
roll-free IK, then reports the minimum distance between any Robot-B link and
any Robot-A link / the tire. Negative = penetration.

Optionally sweeps a shorter nut-runner tool length and/or a raised Robot-B
base Z to test whether the far-arc collisions can be removed.

Usage:
  python -m scripts.diag_b_vs_a_collision
  python -m scripts.diag_b_vs_a_collision --tool-len 0.15 --b-dz 0.20
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


def _quat_axis(quat, axis="z"):
    R = np.asarray(p.getMatrixFromQuaternion(list(quat)), float).reshape(3, 3)
    return R[:, {"x": 0, "y": 1, "z": 2}[axis]]


def _quat_from_z_roll(z, roll):
    z = np.asarray(z, float) / max(np.linalg.norm(z), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = np.cross(ref, z); x0 /= max(np.linalg.norm(x0), 1e-9)
    y0 = np.cross(z, x0)
    cr, sr = np.cos(roll), np.sin(roll)
    xr, yr = cr * x0 + sr * y0, -sr * x0 + cr * y0
    m = np.column_stack([xr, yr, z])
    q = p.getQuaternionFromEuler([0, 0, 0])
    # build quat from rotation matrix
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s; zz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; zz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s; zz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s; zz = 0.25 * s
    return [x, y, zz, w]


def _best_b_ik(env, target, want_z, seed_key=0):
    robot = env.robot_B
    want_z = np.asarray(want_z, float) / max(float(np.linalg.norm(want_z)), 1e-9)
    target = np.asarray(target, dtype=np.float64)
    lo, hi = robot.arm.lower, robot.arm.upper
    rng = np.random.default_rng(11 + seed_key)
    best_q, best_cost = None, 1e9
    for ri in range(16):
        quat = _quat_from_z_roll(want_z, 2 * np.pi * ri / 16)
        for k in range(4):
            seed = (robot.arm.rest if k == 0 else rng.uniform(lo, hi)).tolist()
            ik = p.calculateInverseKinematics(
                robot.uid, robot.EE_LINK_INDEX, target.tolist(), quat,
                lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                jointRanges=robot.arm.range.tolist(), restPoses=seed,
                maxNumIterations=400, residualThreshold=1e-6,
                physicsClientId=env.client)
            q = np.clip(np.asarray(ik, float)[robot._ik_arm_slots], lo, hi)
            st = p.saveState(physicsClientId=env.client)
            for s, qq in zip(robot.arm.indices, q):
                p.resetJointState(robot.uid, int(s), float(qq),
                                  physicsClientId=env.client)
            ee, eq = robot.ee_pose()
            dp = float(np.linalg.norm(np.asarray(ee, float) - target))
            gz = _quat_axis(eq, "z")
            ang = float(np.degrees(np.arccos(np.clip(np.dot(gz, want_z), -1, 1))))
            p.restoreState(st, physicsClientId=env.client)
            p.removeState(st, physicsClientId=env.client)
            cost = dp + 0.02 * ang
            if cost < best_cost:
                best_cost, best_q = cost, q
        if best_cost < 0.01:
            break
    return best_q, best_cost


def _link_names(uid, cid):
    names = {-1: "base"}
    for j in range(p.getNumJoints(uid, physicsClientId=cid)):
        info = p.getJointInfo(uid, j, physicsClientId=cid)
        names[j] = info[12].decode() if isinstance(info[12], bytes) else str(info[12])
    return names


def measure(tool_len=None, b_dz=0.0, verbose=True):
    overrides = dict(
        nut_fastening_task=True, scene_layout="fanuc_spacious",
        terminate_on="never", contact_force_terminate_above=0.0,
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
    )
    if tool_len is not None:
        overrides["ur10e_nut_tool_length"] = float(tool_len)
    cfg = make_env_config(stage=3, phase=1, **overrides)
    if b_dz:
        bx, by, bz = cfg.robot_B_base_pos
        cfg.robot_B_base_pos = (bx, by, bz + float(b_dz))
    env = TyroEnv(cfg=cfg, render=False, seed=7)
    env.set_nut_b_hotstart_alpha(1.0)
    env.reset(seed=7)
    cid = env.client
    rb, ra = env.robot_B, env.robot_A
    tire = env.handles.tire
    a_names = _link_names(ra.uid, cid)
    b_names = _link_names(rb.uid, cid)
    n = len(env.handles.bolts)
    L = float(cfg.bolt_length)
    standoff = float(getattr(cfg, "nut_insert_standoff", 0.05))
    eff_tool = cfg.ur10e_nut_tool_length

    print(f"\n=== tool_len={eff_tool:.2f}  b_base={cfg.robot_B_base_pos}  "
          f"(b_dz={b_dz:+.2f}) ===")
    rows = []
    for i in range(n):
        bp = np.asarray(env.scene.bolt_pose(i)[0], float)
        ax = np.asarray(env.scene.bolt_axis(i), float)
        ax /= max(np.linalg.norm(ax), 1e-9)
        want_z = -ax
        insert_pos = bp - ax * (0.5 * L)  # seated at hub face
        q, cost = _best_b_ik(env, insert_pos, want_z, seed_key=i)
        if q is None:
            rows.append((i, None, None, None, None)); continue
        for s, qq in zip(rb.arm.indices, q):
            p.resetJointState(rb.uid, int(s), float(qq), physicsClientId=cid)
        # closest distance B-link <-> A-link and B-link <-> tire
        min_a, who_a = 9.0, "-"
        cps = p.getClosestPoints(bodyA=rb.uid, bodyB=ra.uid, distance=0.5,
                                 physicsClientId=cid)
        for cp in cps:
            d = float(cp[8])
            if d < min_a:
                min_a = d; who_a = f"B:{b_names.get(cp[3],cp[3])}<->A:{a_names.get(cp[4],cp[4])}"
        min_t, who_t = 9.0, "-"
        cpt = p.getClosestPoints(bodyA=rb.uid, bodyB=tire, distance=0.5,
                                 physicsClientId=cid)
        for cp in cpt:
            d = float(cp[8])
            if d < min_t:
                min_t = d; who_t = f"B:{b_names.get(cp[3],cp[3])}<->tire"
        rows.append((i, cost, min_a, who_a, min_t))
        flag = "  <-- COLLIDE" if (min_a < 0 or min_t < 0) else ""
        if verbose:
            print(f"  bolt {i:2d}: IKcost={cost*100:5.1f}cm  "
                  f"d(A)={min_a*100:+6.1f}cm [{who_a}]  "
                  f"d(tire)={min_t*100:+6.1f}cm{flag}")
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool-len", type=float, default=None)
    ap.add_argument("--b-dz", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep a few tool-len x b-dz combos")
    args = ap.parse_args()
    if args.sweep:
        measure(tool_len=None, b_dz=0.0)          # baseline
        for tl in (0.20, 0.15, 0.10):
            measure(tool_len=tl, b_dz=0.0)
        for dz in (0.20, 0.40):
            measure(tool_len=None, b_dz=dz)
        measure(tool_len=0.15, b_dz=0.30)
    else:
        measure(tool_len=args.tool_len, b_dz=args.b_dz)


if __name__ == "__main__":
    raise SystemExit(main())
