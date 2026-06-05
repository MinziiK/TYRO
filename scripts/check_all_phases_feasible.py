#!/usr/bin/env python3
"""End-to-end physical feasibility for the fanuc_spacious layout.

Checks, with the REAL applied config:

  A) Rack / tire geometry — is the tire actually resting on the rails, and how
     much clear corridor (Y-gap) is left for the gripper?

  B) Phase 1 (FANUC, Robot A) — every FSM stage end-EE pose:
        S0 grasp anchor, S1 mount, S2 demount, S3 cradle return.
     Reported as position-only IK residual (orientation-free reach) AND the
     baked-trajectory end error (what the sim replays under the palm-up lock).

  C) Phase 2/3 (UR10e, Robot B) — every hub bolt, with ORIENTATION:
     full 6-DOF IK driving tool0 +Z anti-parallel to the bolt axis (the pose a
     nut-runner must hold). Position residual AND tool-axis error in degrees.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pybullet as p  # noqa: E402

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402

FANUC_REACH = 2.655
UR10E_REACH = 1.30


def pr(*a):
    print(*a)
    sys.stdout.flush()


def _quat_axis(quat, axis="z"):
    R = np.asarray(p.getMatrixFromQuaternion(list(quat)), float).reshape(3, 3)
    return R[:, {"x": 0, "y": 1, "z": 2}[axis]]


def pos_only_reach(robot, pos, restarts=48):
    rng = np.random.default_rng(11)
    lo, hi = robot.arm.lower, robot.arm.upper
    best = 1e9
    for k in range(restarts):
        seed = (robot.arm.rest if k == 0 else rng.uniform(lo, hi)).tolist()
        ik = p.calculateInverseKinematics(
            robot.uid, robot.EE_LINK_INDEX, np.asarray(pos, float).tolist(),
            lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
            jointRanges=robot.arm.range.tolist(), restPoses=seed,
            maxNumIterations=400, residualThreshold=1e-6,
            physicsClientId=robot.client,
        )
        q = np.clip(np.asarray(ik, float)[robot._ik_arm_slots], lo, hi)
        st = p.saveState(physicsClientId=robot.client)
        for s, qq in zip(robot.arm.indices, q):
            p.resetJointState(robot.uid, int(s), float(qq),
                              physicsClientId=robot.client)
        ee, _ = robot.ee_pose()
        best = min(best, float(np.linalg.norm(np.asarray(ee, float)
                                              - np.asarray(pos, float))))
        p.restoreState(st, physicsClientId=robot.client)
        p.removeState(st, physicsClientId=robot.client)
    return best


def _quat_from_z_roll(z, roll):
    """Unit quat (x,y,z,w) whose tool +Z = ``z``, rolled ``roll`` rad about it."""
    z = np.asarray(z, float) / max(np.linalg.norm(z), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = np.cross(ref, z); x0 /= max(np.linalg.norm(x0), 1e-9)
    y0 = np.cross(z, x0)
    cr, sr = np.cos(roll), np.sin(roll)
    xr, yr = cr * x0 + sr * y0, -sr * x0 + cr * y0
    m = np.column_stack([xr, yr, z])
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; zz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; zz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; zz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; zz = 0.25 * s
    return [x, y, zz, w]


def pose_reach(robot, pos, axis, n_roll=16, n_seed=4):
    """Roll-FREE 6-DOF IK -> (best pos residual m, tool-z axis error deg).

    A nut-runner spins freely about the bolt axis, so the only orientation
    constraint is that the tool +Z is anti-parallel to the bolt axis (i.e.
    aligned with ``axis``). We sweep the free roll DOF about ``axis`` and try
    several IK seeds, keeping the solution with the smallest pos+axis error.
    Fixing the roll (the old behaviour) produced spurious wrist-limit FAILs.
    """
    rng = np.random.default_rng(11)
    lo, hi = robot.arm.lower, robot.arm.upper
    want_z = np.asarray(axis, float) / max(np.linalg.norm(axis), 1e-9)
    best_pos, best_ang = 1e9, 180.0
    target = np.asarray(pos, float)
    for ri in range(n_roll):
        quat = _quat_from_z_roll(want_z, 2 * np.pi * ri / n_roll)
        for k in range(n_seed):
            seed = (robot.arm.rest if k == 0 else rng.uniform(lo, hi)).tolist()
            ik = p.calculateInverseKinematics(
                robot.uid, robot.EE_LINK_INDEX,
                target.tolist(), quat,
                lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                jointRanges=robot.arm.range.tolist(), restPoses=seed,
                maxNumIterations=400, residualThreshold=1e-6,
                physicsClientId=robot.client,
            )
            q = np.clip(np.asarray(ik, float)[robot._ik_arm_slots], lo, hi)
            st = p.saveState(physicsClientId=robot.client)
            for s, qq in zip(robot.arm.indices, q):
                p.resetJointState(robot.uid, int(s), float(qq),
                                  physicsClientId=robot.client)
            ee, eq = robot.ee_pose()
            dp = float(np.linalg.norm(np.asarray(ee, float) - target))
            got_z = _quat_axis(eq, "z")
            ang = float(np.degrees(np.arccos(np.clip(np.dot(got_z, want_z), -1, 1))))
            p.restoreState(st, physicsClientId=robot.client)
            p.removeState(st, physicsClientId=robot.client)
            if dp + 0.02 * ang < best_pos + 0.02 * best_ang:
                best_pos, best_ang = dp, ang
        if best_pos < 0.01 and best_ang < 3.0:
            break
    return best_pos, best_ang


def bake_stage(env, stage):
    env.task_stage = stage
    env._replan_for_current_stage()
    return np.asarray(env._traj_pos, float), np.asarray(env._traj_q, float)


def baked_end_err(env, stage, warm_q=None):
    if warm_q is not None:
        for s, qi in zip(env.robot_A.arm.indices, warm_q):
            p.resetJointState(env.robot_A.uid, int(s), float(qi),
                              physicsClientId=env.client)
    nom, q = bake_stage(env, stage)
    for s, qi in zip(env.robot_A.arm.indices, q[-1]):
        p.resetJointState(env.robot_A.uid, int(s), float(qi),
                          physicsClientId=env.client)
    ee, _ = env.robot_A.ee_pose()
    return float(np.linalg.norm(np.asarray(ee, float) - nom[-1])), nom[-1], q


def main() -> int:
    cfg = make_env_config(stage=1, phase=1, scene_layout="fanuc_spacious")
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.0)
    env.reset()
    h = env.scene.handles
    R = float(cfg.tire_outer_radius)

    pr("=" * 74)
    pr("A) RACK / TIRE GEOMETRY")
    pr("=" * 74)
    tp, _ = env.scene.tire_pose()
    he = cfg.tire_rack_half_extents
    iy = cfg.tire_rack_inner_center[1]
    oy = cfg.tire_rack_outer_center[1]
    gap = (iy - he[1]) - (oy + he[1])  # inner gap between the two rails
    pr(f"  tire COM=({tp[0]:.2f},{tp[1]:.2f},{tp[2]:.3f})  R={R}")
    for u in h.tire_rack:
        aabb = p.getAABB(u, physicsClientId=env.client)
        cps = p.getClosestPoints(h.tire, u, distance=0.5, physicsClientId=env.client)
        dmin = min((c[8] for c in cps), default=9.9)
        pr(f"  rail u{u}: top={aabb[1][2]:+.3f} bottom={aabb[0][2]:+.3f}  "
           f"tire<->rail={dmin*100:+.2f}cm "
           f"({'seated' if abs(dmin) < 0.02 else ('FLOATING' if dmin > 0 else 'overlap')})")
    pr(f"  gripper Y-corridor between rails = {gap*100:.0f} cm "
       f"(rail face thickness 2*he_y={2*he[1]*100:.0f}cm each)")

    pr("\n" + "=" * 74)
    pr("B) PHASE 1 — FANUC stage end-EE reach (pos-only resid + baked end err)")
    pr("=" * 74)
    env.close()
    # fresh env with easy-start so the grasp transform is cached for S1-3
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    a_base = np.asarray(cfg.robot_A_base_pos, float)
    names = {0: "S0 grasp", 1: "S1 mount", 2: "S2 demount", 3: "S3 return"}
    warm = None
    for st in (0, 1, 2, 3):
        berr, tgt, q = baked_end_err(env, st, warm_q=warm)
        warm = q[-1]
        env.robot_A.reset_to_home()
        pr_only = pos_only_reach(env.robot_A, tgt)
        d = float(np.linalg.norm(np.asarray(tgt, float) - a_base))
        pr(f"  {names[st]:11s} tgt=({tgt[0]:+.2f},{tgt[1]:+.2f},{tgt[2]:+.2f}) "
           f"d={d:.2f}m ({100*d/FANUC_REACH:3.0f}%)  "
           f"pos-only={pr_only*100:5.1f}cm  baked_end={berr*100:5.1f}cm")
    env.close()

    pr("\n" + "=" * 74)
    pr("C) PHASE 2/3 — UR10e bolt approach with ORIENTATION (nut-runner pose)")
    pr("   tool +Z driven anti-parallel to bolt axis (approach direction);")
    pr("   roll about the bolt axis is FREE (nut-runner spins) -> swept.")
    pr("=" * 74)
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.0)
    env.reset()
    b_base = np.asarray(cfg.robot_B_base_pos, float)
    worst_pos, worst_ang = 0.0, 0.0
    for i in range(cfg.n_bolts):
        bp, bq = env.scene.bolt_pose(i)
        baxis = _quat_axis(bq, "z")
        # The nut-runner approaches the bolt head-on: the tool +Z points INTO
        # the bolt, i.e. tool +Z = -bolt_axis (the approach direction). The
        # spin about that axis is a free DOF, so pose_reach sweeps the roll.
        approach = -baxis / max(np.linalg.norm(baxis), 1e-9)
        env.robot_B.reset_to_home()
        dp, ang = pose_reach(env.robot_B, np.asarray(bp, float), approach)
        d = float(np.linalg.norm(np.asarray(bp, float) - b_base))
        worst_pos = max(worst_pos, dp); worst_ang = max(worst_ang, ang)
        flag = "OK" if (dp < 0.02 and ang < 5) else ("pos!" if dp >= 0.02 else "ang!")
        pr(f"  bolt{i:2d} d={d:.2f}m ({100*d/UR10E_REACH:3.0f}%)  approach=("
           f"{approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f})  "
           f"pos={dp*100:5.1f}cm  tool_ang={ang:5.1f}deg  {flag}")
    pr(f"\n  WORST  pos={worst_pos*100:.1f}cm  tool_ang={worst_ang:.1f}deg  "
       f"=> {'ALL nut poses feasible' if (worst_pos<0.02 and worst_ang<5) else 'CHECK'}")
    env.close()
    pr("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
