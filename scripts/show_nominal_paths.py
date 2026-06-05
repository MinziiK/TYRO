"""Open the GUI and draw the training nominal EE trajectories at once.

No animation: builds the env exactly as training does (``make_env_config``),
resets once (attached hot-start → the stage the policy actually trains on),
draws every remaining FSM stage's baked Min-Jerk nominal EE path as static
debug lines, frames the camera on the hub, and holds the window open.

    blue=approach(0)  orange=carry/mount(1)  magenta=demount(2)  green=return(3)

Run (headless server → noVNC display):
    DISPLAY=:2 python scripts/show_nominal_paths.py
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

STAGE_COLORS = {
    0: [0.10, 0.45, 1.00],  # blue   — approach / pickup
    1: [1.00, 0.55, 0.00],  # orange — carry to hub (mount, trained stage)
    2: [0.85, 0.20, 0.85],  # magenta— demount
    3: [0.20, 0.85, 0.40],  # green  — return to cradle
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=float, default=4.0)
    args = ap.parse_args()

    cfg = make_env_config(stage=args.stage, phase=args.phase, render=True)
    env = TyroEnv(cfg=cfg, render=True, seed=args.seed)
    env.reset(seed=args.seed)

    hub, _ = env.scene.hub_pose()
    hub = np.asarray(hub, dtype=np.float64)

    # Frame the camera on the hub (the new world origin).
    p.resetDebugVisualizerCamera(
        cameraDistance=2.6, cameraYaw=45, cameraPitch=-22,
        cameraTargetPosition=hub.tolist(), physicsClientId=env.client,
    )

    trajs = env.compute_all_stage_trajectories()
    total = 0
    for stage in sorted(trajs):
        pts = trajs[stage]
        color = STAGE_COLORS.get(stage, [1.0, 1.0, 1.0])
        for i in range(len(pts) - 1):
            p.addUserDebugLine(
                pts[i].tolist(), pts[i + 1].tolist(),
                lineColorRGB=color, lineWidth=args.width,
                lifeTime=0.0, physicsClientId=env.client,
            )
        total += len(pts)
        # Mark the stage end pose with a small cross.
        end = pts[-1]
        for d in (np.array([0.05, 0, 0]), np.array([0, 0.05, 0]),
                  np.array([0, 0, 0.05])):
            p.addUserDebugLine(
                (end - d).tolist(), (end + d).tolist(),
                lineColorRGB=color, lineWidth=args.width,
                lifeTime=0.0, physicsClientId=env.client,
            )

    print(
        "[paths] drew nominal EE trajectories for stages %s (%d pts). "
        "blue=approach orange=carry/mount magenta=demount green=return. "
        "Active trained stage = %d (mount-only)."
        % (sorted(trajs), total, int(env.task_stage))
    )
    active = getattr(env, "_traj_pos", None)
    if active is not None:
        print(f"[paths] active baked nominal (_traj_pos): {active.shape} "
              f"start={np.round(active[0],3)} end={np.round(active[-1],3)}")

    print("[paths] GUI held open — close window or Ctrl-C to exit.")
    try:
        while p.isConnected(env.client):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
