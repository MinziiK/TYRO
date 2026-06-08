#!/usr/bin/env python3
"""Search for a Robot-B base placement that is collision-free vs Robot A.

Keeps Robot A's REAL hold-pose distribution fixed (the recorded mount-end bank
in nut_mount_endpose.npz) — because in deployment A returns to its learned
mount-completion pose, NOT a hand-edited one. For each candidate B base we
check, against EVERY recorded A hold pose:

  (a) reachability   — roll-free IK seats all 10 bolts (pos err < tol)
  (b) clearance      — min B-link <-> A-link distance over the approach
                       corridor of all bolts, worst-case over the A bank

A base is FEASIBLE if every bolt is reachable AND the worst-case corridor
clearance is positive (ideally > margin). Reports feasible bases ranked by
clearance so we can move B there and retrain.

B's base is moved at runtime via resetBasePositionAndOrientation (IK respects
the live base), so no URDF reload per candidate.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


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


class BSolver:
    """Light roll-free IK for Robot B at the current (possibly moved) base."""

    def __init__(self, env, n_roll=8, n_seed=2):
        self.env = env
        self.rb = env.robot_B
        self.n_roll, self.n_seed = n_roll, n_seed
        self.rng = np.random.default_rng(7)

    def solve(self, target, want_z):
        rb, env = self.rb, self.env
        lo, hi = rb.arm.lower, rb.arm.upper
        want_z = np.asarray(want_z, float) / max(np.linalg.norm(want_z), 1e-9)
        target = np.asarray(target, float)
        best_q, best_cost = None, 1e9
        for ri in range(self.n_roll):
            quat = _quat_from_z_roll(want_z, 2 * np.pi * ri / self.n_roll)
            for k in range(self.n_seed):
                seed = (rb.arm.rest if k == 0 else
                        self.rng.uniform(lo, hi)).tolist()
                ik = p.calculateInverseKinematics(
                    rb.uid, rb.EE_LINK_INDEX, target.tolist(), quat,
                    lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                    jointRanges=rb.arm.range.tolist(), restPoses=seed,
                    maxNumIterations=200, residualThreshold=1e-6,
                    physicsClientId=env.client)
                q = np.clip(np.asarray(ik, float)[rb._ik_arm_slots], lo, hi)
                for s, qq in zip(rb.arm.indices, q):
                    p.resetJointState(rb.uid, int(s), float(qq),
                                      physicsClientId=env.client)
                ee, eq = rb.ee_pose()
                dp = float(np.linalg.norm(np.asarray(ee, float) - target))
                gz = np.asarray(p.getMatrixFromQuaternion(list(eq)),
                                float).reshape(3, 3)[:, 2]
                ang = float(np.degrees(np.arccos(
                    np.clip(np.dot(gz, want_z), -1, 1))))
                cost = dp + 0.02 * ang
                if cost < best_cost:
                    best_cost, best_q = cost, q
            if best_cost < 0.005:
                break
        return best_q, best_cost


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach-tol", type=float, default=0.02)
    ap.add_argument("--n-apose", type=int, default=6,
                    help="how many recorded A hold poses to test (worst-case)")
    ap.add_argument("--quick", action="store_true",
                    help="coarser IK + fewer corridor samples")
    args = ap.parse_args()

    cfg = make_env_config(
        stage=3, phase=1, nut_fastening_task=True,
        scene_layout="fanuc_spacious", terminate_on="never",
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
        contact_force_terminate_above=0.0,
    )
    env = TyroEnv(cfg=cfg, render=False, seed=5000)
    env.set_nut_b_hotstart_alpha(0.0)
    env.reset(seed=5000)

    a_uid, b_uid = int(env.robot_A.uid), int(env.robot_B.uid)
    ra, rb = env.robot_A, env.robot_B
    n = len(env.handles.bolts)
    L = float(getattr(cfg, "bolt_length", 0.10))
    base0 = np.asarray(cfg.robot_B_base_pos, float)
    base_orn = list(p.getQuaternionFromEuler(list(cfg.robot_B_base_rpy)))

    # A hold-pose bank (worst-case set). Fall back to current frozen pose.
    bank = getattr(env, "_mount_hold_qA_bank", None)
    if bank is None or len(bank) == 0:
        bank = np.asarray(env._nut_frozen_qA, float)[None, :]
    apose = bank[:max(1, min(args.n_apose, len(bank)))]
    print(f"[search] testing against {len(apose)} A hold poses; "
          f"reach_tol={args.reach_tol*100:.0f}mm")

    solver = BSolver(env, n_roll=(6 if args.quick else 8),
                     n_seed=(2 if args.quick else 2))
    axials = ([0.5 * L + 0.20, 0.0, -0.5 * L] if args.quick else
              [0.5 * L + 0.25, 0.5 * L + 0.10, 0.0, -0.5 * L])

    # Precompute per-bolt (axis, want_z, corridor targets).
    want = [(-env._nut_axis_unit(i)) for i in range(n)]
    corr = [[env._nut_point_on_axis(i, ax) for ax in axials] for i in range(n)]

    # Candidate B bases (relative to current).
    dxs = [0.0, 0.2, 0.4, 0.6]
    dys = [-0.3, -0.15, 0.0, 0.15]
    dzs = [-0.2, 0.0, 0.2, 0.4]
    cands = [base0 + np.array([dx, dy, dz])
             for dx, dy, dz in itertools.product(dxs, dys, dzs)]
    print(f"[search] {len(cands)} candidate bases "
          f"(dx{dxs} x dy{dys} x dz{dzs})")

    results = []
    for ci, base in enumerate(cands):
        p.resetBasePositionAndOrientation(b_uid, base.tolist(), base_orn,
                                          physicsClientId=env.client)
        # Reachability + per-bolt B configs along corridor.
        reach_ok = True
        max_reach_err = 0.0
        bolt_qs = []  # [bolt][axial] -> q (or None)
        for i in range(n):
            qs = []
            seat_err = 9.0
            for j, tgt in enumerate(corr[i]):
                q, cost = solver.solve(tgt, want[i])
                qs.append(q)
                if abs(axials[j] + 0.5 * L) < 1e-6:  # seated sample
                    seat_err = cost
            bolt_qs.append(qs)
            max_reach_err = max(max_reach_err, seat_err)
            if seat_err > args.reach_tol:
                reach_ok = False
        if not reach_ok:
            results.append((base, False, max_reach_err, None, None, None))
            continue
        # Worst-case corridor clearance over all A poses.
        worst_clear, worst_bolt = 9.0, -1
        for qa in apose:
            for s, qq in zip(ra.arm.indices, qa):
                p.resetJointState(a_uid, int(s), float(qq),
                                  physicsClientId=env.client)
            for i in range(n):
                for q in bolt_qs[i]:
                    if q is None:
                        continue
                    for s, qq in zip(rb.arm.indices, q):
                        p.resetJointState(rb.uid, int(s), float(qq),
                                          physicsClientId=env.client)
                    cps = p.getClosestPoints(b_uid, a_uid, distance=0.5,
                                             physicsClientId=env.client)
                    if not cps:
                        continue
                    md = min(float(c[8]) for c in cps)
                    if md < worst_clear:
                        worst_clear, worst_bolt = md, i
        results.append((base, True, max_reach_err, worst_clear, worst_bolt, None))
        tag = "FEASIBLE" if worst_clear > 0 else "collide"
        print(f"  [{ci+1:2d}/{len(cands)}] base={np.round(base,2)}  "
              f"reach_err={max_reach_err*100:4.1f}mm  "
              f"worstClear={worst_clear*100:6.1f}cm @bolt{worst_bolt}  {tag}")

    env.close()

    feas = [r for r in results if r[1] and r[3] is not None and r[3] > 0]
    feas.sort(key=lambda r: -r[3])
    print("\n================ TOP FEASIBLE B BASES ================")
    if not feas:
        print("  NONE collision-free. Best (still colliding):")
        coll = [r for r in results if r[1] and r[3] is not None]
        coll.sort(key=lambda r: -r[3])
        for base, _, re_, wc, wb, _ in coll[:5]:
            print(f"  base={np.round(base,3)}  worstClear={wc*100:.1f}cm "
                  f"@bolt{wb}  reach_err={re_*100:.1f}mm")
    else:
        for base, _, re_, wc, wb, _ in feas[:8]:
            print(f"  base={np.round(base,3)}  worstClear={wc*100:.1f}cm "
                  f"(min@bolt{wb})  reach_err={re_*100:.1f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
