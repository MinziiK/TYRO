#!/usr/bin/env python3
"""Render off-screen snapshots of the carry/mount so we can SEE the tire
orientation exactly as the GUI replay shows it (same overrides as
replay_planner). Saves PNGs to /tmp/snap_*.png at several FSM moments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402

W, H = 900, 700


def _save(rgba, name):
    try:
        from PIL import Image
        Image.fromarray(rgba).save(f"/tmp/{name}.png")
    except Exception:
        import imageio.v2 as imageio
        imageio.imwrite(f"/tmp/{name}.png", rgba)
    print(f"  saved /tmp/{name}.png")


def shot(env, name, target):
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=W / H, nearVal=0.05, farVal=14)
    # eye offsets: top-down, and side along -X (so cargo doesn't occlude)
    eyes = {
        "top": [target[0] + 0.01, target[1] + 0.01, target[2] + 3.2],
        "sideX": [target[0] - 3.0, target[1], target[2] + 0.6],
    }
    for suf, eye in eyes.items():
        view = p.computeViewMatrix(eye, list(target), [0, 0, 1])
        img = p.getCameraImage(W, H, view, proj,
                               renderer=p.ER_TINY_RENDERER,
                               physicsClientId=env.client)
        rgba = np.reshape(np.asarray(img[2], np.uint8), (H, W, 4))[:, :, :3]
        _save(rgba, f"{name}_{suf}")


def main() -> int:
    cfg = make_env_config(
        stage=3, phase=1, scene_layout="fanuc_spacious",
        contact_force_terminate_above=0.0,
        mount_hold_steps=40, pin_tire_on_mount=True,
    )
    env = TyroEnv(cfg=cfg, render=False, seed=42)
    env.set_start_pos_easy_prob(0.999)
    env.reset(seed=42)
    env.set_mount_tol(0.55, np.deg2rad(45.0))
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    hub = np.asarray(cfg.tire_mount_pos, float)

    targets = {30: "snap_carry_early", 90: "snap_carry_mid",
               150: "snap_standoff", 175: "snap_mount"}
    steps = 0
    while True:
        _, _, term, trunc, _ = env.step(zero)
        steps += 1
        if steps in targets:
            tp = np.asarray(env.scene.tire_pose()[0], float)
            shot(env, targets[steps], tp)
            print(f"    step{steps}: tire_pos={tuple(round(v,2) for v in tp)} "
                  f"tire_axis={tuple(round(v,2) for v in env.scene.tire_axis())} "
                  f"hub_axis={tuple(round(v,2) for v in env.scene.hub_axis())}")
        if term or trunc or steps > 220:
            break
    # final held frame centered on hub
    shot(env, "snap_final", hub)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
