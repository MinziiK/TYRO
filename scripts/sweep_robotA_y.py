#!/usr/bin/env python3
"""Sweep a −Y shift of the FANUC (Robot A) + tire rack, keeping the hub /
cargo / Robot B (origin) fixed. This widens the A↔hub gap (relieving the
arm-link jam + giving the tire room to align) WITHOUT moving the hub away
from Robot B, so the UR10e bolt reach is unchanged.

The seating physics depend only on the A↔hub *relative* geometry, which is
identical to moving the hub +Y; the NEW constraint introduced by moving A is
whether the FANUC can still reach both the pickup (rack) and the mount. We
report FANUC position-only IK residual for every stage end pose, plus the
zero-action forward-roll mount gate.

Usage:
    python scripts/sweep_robotA_y.py --shifts 0.3 0.5 0.7 0.9
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

FANUC_REACH = 2.655


def _shift_y(t, dy):
    t = list(t)
    t[1] = float(t[1]) + float(dy)
    return tuple(t)


def pos_only_reach(robot, pos, restarts=32):
    rng = np.random.default_rng(11)
    lo, hi = robot.arm.lower, robot.arm.upper
    best = 1e9
    for k in range(restarts):
        seed = (robot.arm.rest if k == 0 else rng.uniform(lo, hi)).tolist()
        ik = p.calculateInverseKinematics(
            robot.uid, robot.EE_LINK_INDEX, np.asarray(pos, float).tolist(),
            lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
            jointRanges=robot.arm.range.tolist(), restPoses=seed,
            maxNumIterations=300, residualThreshold=1e-6,
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


def eval_shift(dy: float, roll: bool) -> str:
    cfg = make_env_config(
        stage=3, phase=1, scene_layout="fanuc_spacious",
        start_pos_curriculum_enable=True, start_pos_curriculum_mode="mix",
        attached_spawn_when_easy=True, terminate_on="never",
    )
    # Move Robot A + tire rack/pickup −Y; hub/cargo/bolts/Robot B unchanged.
    cfg.robot_A_base_pos = _shift_y(cfg.robot_A_base_pos, -dy)
    cfg.tire_rack_inner_center = _shift_y(cfg.tire_rack_inner_center, -dy)
    cfg.tire_rack_outer_center = _shift_y(cfg.tire_rack_outer_center, -dy)
    cfg.tire_pickup_pos = _shift_y(cfg.tire_pickup_pos, -dy)

    env = TyroEnv(cfg=cfg, render=False, seed=0)
    a_base = np.asarray(cfg.robot_A_base_pos, float)

    # FANUC pos-only reach at every stage end (easy-start caches grasp T).
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    names = {0: "S0pick", 1: "S1mount", 2: "S2demnt", 3: "S3retn"}
    reach = {}
    warm = None
    for st in (0, 1, 2, 3):
        env.task_stage = st
        if warm is not None:
            for s, qi in zip(env.robot_A.arm.indices, warm):
                p.resetJointState(env.robot_A.uid, int(s), float(qi),
                                  physicsClientId=env.client)
        env._replan_for_current_stage()
        q = np.asarray(env._traj_q, float)
        tgt = np.asarray(env._traj_pos, float)[-1]
        warm = q[-1]
        env.robot_A.reset_to_home()
        reach[st] = (pos_only_reach(env.robot_A, tgt),
                     float(np.linalg.norm(tgt - a_base)))

    seat = ""
    if roll:
        env.robot_A.reset_to_home()
        env.set_start_pos_easy_prob(0.999)
        env.reset()
        env.set_mount_tol(float(cfg.mount_radius_tol_soft),
                          float(cfg.mount_angle_tol_soft_rad))
        mount_target = np.asarray(cfg.tire_mount_pos, float)
        act = np.zeros(env.action_space.shape, dtype=np.float32)
        d_mount, th = 1e9, 180.0
        for _ in range(int(cfg.max_steps)):
            _, _, term, trunc, _ = env.step(act)
            tp = np.asarray(env.scene.tire_pose()[0], float)
            dm = float(np.linalg.norm(tp - mount_target))
            if dm < d_mount:
                d_mount = dm
                th = float(np.degrees(np.arccos(np.clip(
                    np.dot(env.scene.tire_axis(), env.scene.hub_axis()),
                    -1, 1))))
            if term or trunc:
                break
        ok = (d_mount < cfg.mount_radius_tol_soft
              and np.deg2rad(th) < cfg.mount_angle_tol_soft_rad)
        seat = (f"  | roll: min_d_mount={d_mount:.3f} theta={th:4.1f}d "
                f"-> {'SEATED' if ok else 'jam'}")
    env.close()

    parts = [f"dY=-{dy:.2f} (A_y={a_base[1]:+.2f})"]
    worst = 0.0
    for st in (0, 1, 2, 3):
        r, d = reach[st]
        worst = max(worst, r)
        parts.append(f"{names[st]}:{r*100:4.1f}cm/{100*d/FANUC_REACH:3.0f}%")
    flag = "FANUC-OK" if worst < 0.03 else "FANUC-FAIL"
    return "  ".join(parts) + f"  [{flag}]" + seat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--no-roll", action="store_true",
                    help="Skip the (slow) forward rollout seating check.")
    args = ap.parse_args()
    print("=" * 78)
    print("ROBOT-A + RACK  −Y SHIFT SWEEP (hub/cargo/Robot-B fixed)")
    print("  pos-only reach per FANUC stage end (cm residual / % of reach)")
    print("  UR10e bolt reach UNCHANGED (hub fixed) so not reported")
    print("=" * 78)
    for dy in args.shifts:
        try:
            print(eval_shift(float(dy), roll=not args.no_roll))
        except Exception as e:  # noqa: BLE001
            print(f"dY=-{dy:.2f}  ERROR {type(e).__name__}: {e}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
