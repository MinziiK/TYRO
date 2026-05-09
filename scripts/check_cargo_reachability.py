#!/usr/bin/env python3
"""Check UR10 IK reach toward the hub opening with cargo (wheel-well box) spawned.

Loads TyroEnv with ``spawn_vehicle_primitive_box=True``, samples a Cartesian target
near the flange, runs ``calculateInverseKinematics``, and reports residual / nominal
UR10 self-collisions via contact probe.

Usage::

    python scripts/check_cargo_reachability.py --render
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

from src.config import make_env_config
from src.env import TyroEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = make_env_config(stage=3, phase=1, render=args.render)
    cfg.spawn_vehicle_primitive_box = True
    cfg.use_truck_hub_urdf = True
    env = TyroEnv(cfg=cfg, render=args.render, seed=args.seed)
    env.reset(seed=args.seed)

    hub_pos, hub_orn = env.scene.hub_pose()
    axle = env.scene.hub_axis()

    offset = hub_pos + axle * (-0.12)
    ee_orn = hub_orn

    ik = p.calculateInverseKinematics(
        env.robot_A.uid,
        env.robot_A.EE_LINK_INDEX,
        list(offset),
        list(ee_orn),
        lowerLimits=env.robot_A.arm.lower.tolist(),
        upperLimits=env.robot_A.arm.upper.tolist(),
        jointRanges=env.robot_A.arm.range.tolist(),
        restPoses=env.robot_A.arm.rest.tolist(),
        maxNumIterations=80,
        residualThreshold=1e-4,
        physicsClientId=env.client,
    )
    ik = np.asarray(ik[: env.robot_A.arm.n], dtype=np.float64)
    for ji, targ in zip(env.robot_A.arm.indices, ik):
        p.resetJointState(env.robot_A.uid, ji, targ, targetVelocity=0.0,
                          physicsClientId=env.client)
    for _ in range(env.cfg.decimation * 8):
        p.stepSimulation(physicsClientId=env.client)

    achieved, ach_orn = env.robot_A.ee_pose()
    err = np.linalg.norm(achieved - offset)

    self_c = p.getContactPoints(
        bodyA=env.robot_A.uid,
        bodyB=env.robot_A.uid,
        physicsClientId=env.client,
    )
    cargo_hits = 0
    if env.handles is not None and env.handles.vehicle is not None:
        vc = p.getContactPoints(
            bodyA=env.robot_A.uid,
            bodyB=env.handles.vehicle,
            physicsClientId=env.client,
        )
        cargo_hits = len(vc)

    print(f"[cargo_reach] hub offset target={offset}")
    print(f"[cargo_reach] |IK pos error|={err:.4f} m  UR10 self-contacts={len(self_c)}  cargo contacts={cargo_hits}")
    print("  Self-contact count > 0 often means unrealistic pose or tight wheel-well obstruction.")

    if args.render:
        print("Inspect reach in GUI — close PyBullet window to exit.")
        while p.isConnected(env.client):
            time.sleep(0.05)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
