#!/usr/bin/env python3
"""Check that the gripper tool +Z axis stays aligned with world +Z.

Runs a zero-action replay in the current config (typically:
  stage=3/phase=1/fanuc_spacious, easy-start, mounted-termination)
and logs:
  - per-step angle(deg) between robot tool +Z and world +Z
  - FSM stage transitions

Optionally overrides mount gate to force Phase-A mount success.
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


def _tool_z_world_angle_deg(robot) -> float:
    # robot.ee_pose() returns (pos, quat)
    _, quat = robot.ee_pose()
    R = np.asarray(p.getMatrixFromQuaternion(list(quat)), float).reshape(3, 3)
    tool_z = R[:, 2]  # tool +Z axis in world
    world_z = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(tool_z, world_z), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--scene-layout", type=str, default="fanuc_spacious")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--mount-tol", type=float, default=None)
    ap.add_argument("--mount-ang-tol-deg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = make_env_config(
        stage=args.stage, phase=args.phase, scene_layout=args.scene_layout,
    )
    env = TyroEnv(cfg=cfg, render=False, seed=args.seed)
    env.set_start_pos_easy_prob(0.999)

    if args.mount_tol is not None or args.mount_ang_tol_deg is not None:
        cur_r, cur_a = env.get_mount_tol()
        new_r = float(args.mount_tol) if args.mount_tol is not None else cur_r
        new_a = (
            np.deg2rad(float(args.mount_ang_tol_deg))
            if args.mount_ang_tol_deg is not None else cur_a
        )
        env.set_mount_tol(new_r, new_a)
        print(f"[z-align] override mount gate: r={new_r:.3f}m a={np.rad2deg(new_a):.1f}deg")

    zero = np.zeros(env.action_space.shape, dtype=np.float32)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        stage_prev = int(getattr(env, "task_stage", -1))
        max_ang = 0.0
        min_ang = 180.0
        print("=" * 72)
        print(f"[z-align] ep{ep} start_stage={stage_prev} start_min_theta={_tool_z_world_angle_deg(env.robot_A):.2f}deg")
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _, r, terminated, truncated, info = env.step(zero)
            steps += 1
            ang = _tool_z_world_angle_deg(env.robot_A)
            max_ang = max(max_ang, ang)
            min_ang = min(min_ang, ang)
            stage_now = int(getattr(env, "task_stage", -1))
            if stage_now != stage_prev:
                print(f"[z-align] stage transition {stage_prev} -> {stage_now} at step={steps}")
                stage_prev = stage_now
            if terminated or truncated:
                break

        print(f"[z-align] ep{ep} finished steps={steps} success={info.get('is_success', False)} "
              f"tool+Z angle: min={min_ang:.2f}deg max={max_ang:.2f}deg")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

