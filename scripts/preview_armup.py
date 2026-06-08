#!/usr/bin/env python3
"""Visualise Robot B's 'arm-up' hot-start pose at every hub bolt (GUI).

For each bolt we set it as the nut target and apply the arm-up-preferring
hot-start (``nut_prefer_arm_up`` + ``alpha = 1``), which IK-teleports B to the
on-axis approach pose using the elbow-up branch. The GUI cycles through the
bolts so you can confirm the forearm/elbow rides high (clear of Robot A's low
arm). The camera is free to orbit while it cycles.

No PPO / torch is loaded here (just env reset + IK), so the live GUI is stable.

Usage:
  python scripts/preview_armup.py                # cycle all bolts, 2.5 s each
  python scripts/preview_armup.py --dwell 4      # 4 s per bolt
  python scripts/preview_armup.py --bolts 3 4 5  # only the 6-o'clock bolts
  python scripts/preview_armup.py --no-arm-up    # compare WITHOUT the bias
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _frame_camera(env: TyroEnv) -> None:
    hub, _ = env.scene.hub_pose()
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2, cameraYaw=55, cameraPitch=-18,
        cameraTargetPosition=np.asarray(hub, dtype=np.float64).tolist(),
        physicsClientId=env.client,
    )


def _ba_clearance(env: TyroEnv) -> float:
    cps = p.getClosestPoints(int(env.robot_B.uid), int(env.robot_A.uid),
                             distance=0.6, physicsClientId=env.client)
    return min((float(c[8]) for c in cps), default=0.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--dwell", type=float, default=2.5,
                    help="seconds to hold each bolt's pose")
    ap.add_argument("--bolts", type=int, nargs="*", default=None,
                    help="subset of bolt indices to show (default: all)")
    ap.add_argument("--no-arm-up", action="store_true",
                    help="disable the arm-up IK bias (for A/B comparison)")
    args = ap.parse_args()

    cfg = make_env_config(
        stage=3, phase=1, nut_fastening_task=True,
        scene_layout="fanuc_spacious", terminate_on="never",
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
        contact_force_terminate_above=0.0,
    )
    cfg.nut_prefer_arm_up = not args.no_arm_up

    env = TyroEnv(cfg=cfg, render=True, seed=args.seed)
    # Heavy nut reset (re-creates bodies + many IK solves). Freeze GL render
    # across it, then re-enable for the live view.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                               physicsClientId=env.client)
    env.set_nut_b_hotstart_alpha(1.0)
    env.reset(seed=args.seed)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                               physicsClientId=env.client)
    _frame_camera(env)

    fore_idx = env._nut_b_forearm_link()
    n = len(env.handles.bolts)
    bolts = args.bolts if args.bolts else list(range(n))

    print(f"[armup-gui] arm_up_bias={'ON' if cfg.nut_prefer_arm_up else 'OFF'}  "
          f"bolts={bolts}  dwell={args.dwell}s")
    print(f"{'bolt':>4} {'foreZ(m)':>9} {'B<->A clear(cm)':>16}")
    print("[armup-gui] orbit the camera freely; Ctrl-C to exit.")

    try:
        while p.isConnected(env.client):
            for i in bolts:
                if not p.isConnected(env.client):
                    break
                env._nut_target_idx = int(i)
                env._apply_nut_b_hotstart()
                fz = (float(p.getLinkState(int(env.robot_B.uid), fore_idx,
                                           physicsClientId=env.client)[0][2])
                      if fore_idx is not None else float("nan"))
                clr = _ba_clearance(env)
                tag = "" if clr > 0.0 else "  <== COLLIDE"
                print(f"{i:>4} {fz:>9.2f} {clr*100:>16.1f}{tag}")
                t_end = time.time() + args.dwell
                while time.time() < t_end and p.isConnected(env.client):
                    time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
