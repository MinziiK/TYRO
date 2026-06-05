#!/usr/bin/env python3
"""Inspect the gripper/tire orientation during the carry+mount of Stage 1.

The user reports the gripper appears to carry the tire sideways ("3 o'clock")
instead of cradling it palm-up (tool +Z pointing world +Z) and inserting it
along the (horizontal) hub axis.

This logs, at each FSM-stage boundary and at the moment of closest mount
approach:
  * EE tool axes in world (col 0=+X, 1=+Y, 2=+Z of the EE frame)
  * angle(tool +Z, world +Z)
  * tire bore axis in world  vs  hub axis in world
  * the cached grasp transform T_ee_tire (how the tire sits in the EE frame)
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


def _R(quat):
    return np.asarray(p.getMatrixFromQuaternion(list(quat)), float).reshape(3, 3)


def _ang(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    d = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    return float(np.degrees(np.arccos(d)))


def _vec(v):
    return "(" + ",".join(f"{x:+.2f}" for x in v) + ")"


def snapshot(env, tag):
    ee_pos, ee_quat = env.robot_A.ee_pose()
    Ree = _R(ee_quat)
    tx, ty, tz = Ree[:, 0], Ree[:, 1], Ree[:, 2]
    tire_axis = np.asarray(env.scene.tire_axis(), float)
    hub_axis = np.asarray(env.scene.hub_axis(), float)
    print(f"  [{tag}]")
    print(f"    EE_pos={_vec(ee_pos)}  tool+X={_vec(tx)} tool+Y={_vec(ty)} tool+Z={_vec(tz)}")
    print(f"    ang(tool+Z,worldZ)={_ang(tz,[0,0,1]):5.1f}d   "
          f"ang(tool+Z,worldX)={_ang(tz,[1,0,0]):5.1f}d   "
          f"ang(tool+Z,worldY)={_ang(tz,[0,1,0]):5.1f}d")
    print(f"    tire_axis={_vec(tire_axis)}  hub_axis={_vec(hub_axis)}  "
          f"ang(tire,hub)={_ang(tire_axis,hub_axis):5.1f}d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount-tol", type=float, default=0.55)
    ap.add_argument("--mount-ang-tol-deg", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious")
    env = TyroEnv(cfg=cfg, render=False, seed=args.seed)
    env.set_start_pos_easy_prob(0.999)
    env.reset(seed=args.seed)
    env.set_mount_tol(args.mount_tol, np.deg2rad(args.mount_ang_tol_deg))

    # grasp transform (how the tire is held relative to the EE)
    gp = np.asarray(getattr(env, "_grasp_t_ee_tire_pos", [0, 0, 0]), float)
    gq = np.asarray(getattr(env, "_grasp_t_ee_tire_quat", [0, 0, 0, 1]), float)
    Rg = _R(gq)
    print("=" * 74)
    print("GRASP TRANSFORM T_ee_tire (tire pose expressed in EE frame):")
    print(f"  t_ee_tire pos={_vec(gp)}")
    print(f"  tire +Z axis in EE frame = {_vec(Rg[:,2])}  "
          f"(this is the bore; if ~(0,0,1) tire bore == tool axis)")
    print("=" * 74)

    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    snapshot(env, "start")
    stage_prev = int(env.task_stage)
    best_d = 1e9
    mount_target = np.asarray(cfg.tire_mount_pos, float)
    hub_axis = np.asarray(env.scene.hub_axis(), float)
    steps = 0
    print("-" * 74)
    print("  Stage-1 bore-vs-hub alignment & approach (every 15 steps):")
    while True:
        _, _, term, trunc, info = env.step(zero)
        steps += 1
        st = int(env.task_stage)
        if st == 1 and steps % 15 == 0:
            ee_pos = np.asarray(env.robot_A.ee_pose()[0], float)
            tp = np.asarray(env.scene.tire_pose()[0], float)
            ba = _ang(env.scene.tire_axis(), hub_axis)
            dy = float(tp[1] - mount_target[1])
            dxz = float(np.linalg.norm(tp[[0, 2]] - mount_target[[0, 2]]))
            print(f"    step{steps:3d}: ang(bore,hub)={ba:5.1f}d  "
                  f"tire_dY_to_hub={dy:+.2f}m  tire_radial(xz)={dxz:.2f}m")
        if st != stage_prev:
            snapshot(env, f"enter stage{st} (step{steps})")
            stage_prev = st
        tp = np.asarray(env.scene.tire_pose()[0], float)
        d = float(np.linalg.norm(tp - mount_target))
        if d < best_d:
            best_d = d
            best_step = steps
            best_tag = f"closest mount d={d:.3f} (step{steps})"
        if term or trunc:
            break
    print("-" * 74)
    # replay to closest-approach for a clean snapshot
    env.reset(seed=args.seed)
    env.set_mount_tol(args.mount_tol, np.deg2rad(args.mount_ang_tol_deg))
    for _ in range(best_step):
        env.step(zero)
    snapshot(env, best_tag)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
