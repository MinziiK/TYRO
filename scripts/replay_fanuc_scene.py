"""Load the full TYRO scene with FANUC R-2000iC as Robot A (GUI smoke test).

Prerequisites
-------------
    python scripts/poc_fanuc_urdf.py --fetch --convert

Usage (tyro conda env)
----------------------
    python scripts/replay_fanuc_scene.py --render
    python scripts/replay_fanuc_scene.py --render --easy-start
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config
from src.env import TyroEnv


def main() -> int:
    ap = argparse.ArgumentParser(description="TYRO scene + FANUC Robot A")
    ap.add_argument("--render", action="store_true", help="Open PyBullet GUI")
    ap.add_argument("--easy-start", action="store_true")
    ap.add_argument("--home-start", action="store_true")
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    cfg = make_env_config(
        stage=1,
        phase=1,
        robot_a_kind="fanuc_r2000ic",
        ur10_lock_tool_up=False,
        fanuc_lock_tool_up=False,
    )
    env = TyroEnv(cfg=cfg, render=bool(args.render), seed=42)
    if args.easy_start:
        env.set_start_pos_easy_prob(0.999)
    elif args.home_start:
        env.set_start_pos_easy_prob(0.0)

    obs, _ = env.reset()
    print(f"[fanuc-scene] robot_A={env.robot_A.NAME}  obs.shape={obs.shape}")
    ee, _ = env.robot_A.ee_pose()
    print(f"[fanuc-scene] EE tool0 @ reset: {tuple(round(v, 3) for v in ee)}")

    zero = np.zeros(env.action_space.shape, dtype=np.float32)

    if args.render:
        import time
        import pybullet as p
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=55,
            cameraPitch=-28,
            cameraTargetPosition=[-0.5, 0.4, 0.5],
            physicsClientId=env.client,
        )
        print("[fanuc-scene] GUI open — close window or Ctrl-C to exit.")
        for _ in range(args.max_steps):
            if not p.isConnected(env.client):
                break
            env.step(zero)
            time.sleep(1.0 / cfg.control_freq_hz)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
