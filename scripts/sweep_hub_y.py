#!/usr/bin/env python3
"""Sweep a +Y shift of the hub/cargo SET (hub + bolts + vehicle + back wall)
for the fanuc_spacious layout and report, per shift:

  * FANUC Stage-1 mount: settle the arm at the baked final joint config with
    the heavy tire JOINT_FIXED-grasped, then measure
      - tire mount gate (d_mount, theta)
      - residual EE error vs the planner target (how far physics pushed the
        arm off the seated pose)
      - peak arm-link contact force against cargo box + hub_mount (the jam)
  * UR10e bolt reach: worst-bolt position-only IK residual (Phase-2 proxy;
    bolts move +Y away from B at the origin so this gets worse).

The shift moves ``hub_pos_nominal``, ``tire_mount_pos`` and
``vehicle_center_world`` by +dY together so the set translates rigidly while
the FANUC base and tire rack stay fixed.

Usage:
    python scripts/sweep_hub_y.py
    python scripts/sweep_hub_y.py --shifts 0 0.2 0.3 0.4 0.5
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


def _shift_y(t, dy):
    t = list(t)
    t[1] = float(t[1]) + float(dy)
    return tuple(t)


def _quat_axis(quat, axis="z"):
    R = np.asarray(p.getMatrixFromQuaternion(list(quat)), float).reshape(3, 3)
    return R[:, {"x": 0, "y": 1, "z": 2}[axis]]


def pos_only_reach(robot, pos, restarts=24):
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


def eval_shift(dy: float) -> str:
    cfg = make_env_config(
        stage=3, phase=1, scene_layout="fanuc_spacious",
        start_pos_curriculum_enable=True, start_pos_curriculum_mode="mix",
        attached_spawn_when_easy=True, terminate_on="never",
    )
    cfg.hub_pos_nominal = _shift_y(cfg.hub_pos_nominal, dy)
    cfg.tire_mount_pos = _shift_y(cfg.tire_mount_pos, dy)
    cfg.vehicle_center_world = _shift_y(cfg.vehicle_center_world, dy)
    # Move Robot B +Y with the hub so the B<->hub bolt geometry is fixed
    # (UR10e reach unchanged). Keep the observation frame pinned to the
    # world origin so "B is the origin" still holds for the policy obs.
    cfg.robot_B_base_pos = _shift_y(cfg.robot_B_base_pos, dy)
    cfg.obs_reference_pos = tuple(cfg.obs_reference_pos)

    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    env.set_mount_tol(float(cfg.mount_radius_tol_soft),
                      float(cfg.mount_angle_tol_soft_rad))
    cli = env.client
    mount_target = np.asarray(cfg.tire_mount_pos, float)

    # Faithful forward roll of the zero-action (planner-only) policy, exactly
    # as training would, capturing the closest mount approach + the peak
    # arm-link contact against cargo/hub during the carry+insertion.
    veh = env.handles.vehicle
    hub_uid = env.handles.hub.uid
    fanuc = env.robot_A.uid
    rim_set = set(getattr(env.handles, "floor_rim", []) or [])
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    d_mount = 1e9
    th_at_min = 180.0
    f_cargo = f_hub = f_rim = 0.0
    cap = int(getattr(cfg, "max_steps", 600))
    for _ in range(cap):
        _, _, term, trunc, info = env.step(act)
        tp = np.asarray(env.scene.tire_pose()[0], float)
        dm = float(np.linalg.norm(tp - mount_target))
        if dm < d_mount:
            d_mount = dm
            th_at_min = float(np.degrees(np.arccos(np.clip(
                np.dot(env.scene.tire_axis(), env.scene.hub_axis()), -1, 1))))
        for cp in p.getContactPoints(physicsClientId=cli):
            a, b, f = cp[1], cp[2], cp[9]
            pair = {a, b}
            if fanuc not in pair:
                continue
            if veh in pair:
                f_cargo = max(f_cargo, f)
            if hub_uid in pair:
                f_hub = max(f_hub, f)
            other = (pair - {fanuc}).pop() if len(pair) == 2 else fanuc
            if other in rim_set:
                f_rim = max(f_rim, f)
        if term or trunc:
            break
    ee_err = 0.0
    th = th_at_min

    # UR10e worst bolt pos-only reach
    env.robot_B.reset_to_home()
    worst = 0.0
    for i in range(cfg.n_bolts):
        bp, _ = env.scene.bolt_pose(i)
        worst = max(worst, pos_only_reach(env.robot_B, np.asarray(bp, float)))
        env.robot_B.reset_to_home()
    env.close()

    seated = "SEATED" if (d_mount < cfg.mount_radius_tol_soft
                          and np.deg2rad(th) < cfg.mount_angle_tol_soft_rad) else "jam"
    return (f"dY=+{dy:.2f}  hub_y={mount_target[1]:+.2f}  "
            f"min_d_mount={d_mount:.3f}  theta@min={th:4.1f}d  "
            f"armVScargo={f_cargo:5.0f}N  armVShub={f_hub:5.0f}N  "
            f"armVSpit={f_rim:5.0f}N  "
            f"UR10e_bolt={worst*100:4.1f}cm  -> {seated}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=float, nargs="+",
                    default=[0.0, 0.2, 0.3, 0.4, 0.5])
    args = ap.parse_args()
    print("=" * 78)
    print("HUB/CARGO SET +Y SHIFT SWEEP (fanuc_spacious)")
    print("  arm-link vs cargo/hub contact = the insertion jam; lower is better")
    print("  UR10e_worst_bolt = Phase-2 reach (grows as set moves +Y from origin)")
    print("=" * 78)
    for dy in args.shifts:
        try:
            print(eval_shift(float(dy)))
        except Exception as e:  # noqa: BLE001
            print(f"dY=+{dy:.2f}  ERROR {type(e).__name__}: {e}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
