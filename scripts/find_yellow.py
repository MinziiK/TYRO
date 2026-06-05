#!/usr/bin/env python3
"""Identify every YELLOW-ish visual in the scene (R>0.7,G>0.6,B<0.35) and
report its body/link, world position relative to the FANUC EE, and render an
EE close-up so we can see what 'the yellow part on the gripper' is.
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


def main() -> int:
    cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious",
                          mount_hold_steps=40, pin_tire_on_mount=True)
    env = TyroEnv(cfg=cfg, render=False, seed=42)
    env.set_start_pos_easy_prob(0.999)
    env.reset(seed=42)
    env.set_mount_tol(0.55, np.deg2rad(45.0))
    cli = env.client
    fanuc = env.robot_A.uid

    # advance a bit so the arm is mid-carry (EE away from base)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(90):
        env.step(zero)
    ee = np.asarray(env.robot_A.ee_pose()[0], float)

    print("=" * 70)
    print(f"  FANUC EE pos = {tuple(round(v,2) for v in ee)}  (uid={fanuc})")
    print("  Yellow-ish visuals (R>0.7,G>0.6,B<0.35):")
    for uid in range(p.getNumBodies(physicsClientId=cli)):
        body = p.getBodyUniqueId(uid, physicsClientId=cli) if False else uid
        try:
            vsd = p.getVisualShapeData(body, physicsClientId=cli)
        except p.error:
            continue
        for v in vsd:
            link = v[1]
            rgba = v[7]
            r, g, b, a = rgba
            if r > 0.7 and g > 0.6 and b < 0.35:
                if link == -1:
                    pos = np.asarray(p.getBasePositionAndOrientation(
                        body, physicsClientId=cli)[0], float)
                else:
                    ls = p.getLinkState(body, link, physicsClientId=cli)
                    pos = np.asarray(ls[0], float)
                d_ee = float(np.linalg.norm(pos - ee))
                name = "?"
                try:
                    if link >= 0:
                        name = p.getJointInfo(body, link,
                                              physicsClientId=cli)[12].decode()
                except Exception:
                    pass
                print(f"    body={body} link={link:>2} ({name:<10}) "
                      f"rgba=({r:.2f},{g:.2f},{b:.2f}) "
                      f"pos={tuple(round(x,2) for x in pos)} "
                      f"dist_to_EE={d_ee:.2f}m "
                      f"{'<-- ON/NEAR GRIPPER' if d_ee < 0.4 else ''}")

    # EE close-up render
    view = p.computeViewMatrix([ee[0]+1.2, ee[1]+1.2, ee[2]+0.8],
                               list(ee), [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(60, 1.3, 0.05, 8)
    img = p.getCameraImage(820, 640, view, proj,
                           renderer=p.ER_TINY_RENDERER, physicsClientId=cli)
    rgba = np.reshape(np.asarray(img[2], np.uint8), (640, 820, 4))[:, :, :3]
    try:
        from PIL import Image
        Image.fromarray(rgba).save("/tmp/ee_closeup.png")
        print("  saved /tmp/ee_closeup.png")
    except Exception as e:
        print(f"  (render save failed: {e})")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
