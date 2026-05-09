#!/usr/bin/env python3
"""Static check: Panda gripper +Z vs target bolt shaft (+Z link frame).

Uses world link-frame pose (``getLinkState`` indices 4–5), same convention as EE.
Bolt ``angle_between`` with gripper axis should shrink toward zero when oriented well.

Usage (repo root, ``conda activate tyro``)::

    python scripts/check_alignment.py --render
    python scripts/check_alignment.py --render --stage 1
    python scripts/check_alignment.py --render --no-truck-hub-urdf
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

from src.config import make_env_config
from src.env import TyroEnv
from src.env.utils import angle_between, quat_axis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5,
                    help="Zero-action settle steps between prints.")
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4],
                    help="Reward stage forwarded to ``make_env_config`` (Phase-1 warmup uses 1).")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                    help="Curriculum spatial-randomisation phase.")
    ap.add_argument(
        "--no-truck-hub-urdf",
        action="store_true",
        help="Use primitive cylinder hub instead of ``truck_wheel_station.urdf``.",
    )
    args = ap.parse_args()

    cfg = make_env_config(
        stage=args.stage,
        phase=args.phase,
        render=args.render,
        use_truck_hub_urdf=(not args.no_truck_hub_urdf),
    )
    env = TyroEnv(cfg=cfg, render=args.render, seed=args.seed)
    env.reset(seed=args.seed)

    for t in range(args.steps):
        for _ in range(env.cfg.decimation):
            p.stepSimulation(physicsClientId=env.client)
        ee_pos, ee_orn = env.robot_B.ee_pose()
        bolt_pos, bolt_orn = env.scene.bolt_pose()
        gz = quat_axis(ee_orn, "z")
        bz = quat_axis(bolt_orn, "z")
        ang = float(angle_between(gz, bz))
        d = float(np.linalg.norm(ee_pos - bolt_pos))
        print(f"step {t:02d}  |ee-bolt|={d:.4f}m  angle(grip_z, bolt_z)={np.rad2deg(ang):.2f} deg")

        p.addUserDebugLine(
            ee_pos.tolist(),
            (ee_pos + 0.12 * gz).tolist(),
            lineColorRGB=(0.2, 0.9, 0.3),
            lineWidth=2.0,
            lifeTime=0.35,
            physicsClientId=env.client,
        )
        p.addUserDebugLine(
            bolt_pos.tolist(),
            (bolt_pos + 0.12 * bz).tolist(),
            lineColorRGB=(0.9, 0.4, 0.1),
            lineWidth=2.0,
            lifeTime=0.35,
            physicsClientId=env.client,
        )
        time.sleep(0.15)

    if args.render:
        print("Close PyBullet window to exit.")
        while p.isConnected(env.client):
            time.sleep(0.05)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
