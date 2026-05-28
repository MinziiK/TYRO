"""Inspect the current UR10 HOME_POSE: EE pose, tool +Z, AABB overlap,
and EE↔grasp_target distance. Headless, prints a single report."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def aabb_intersect(a, b, eps=1e-4):
    a0, a1 = a
    b0, b1 = b
    return all(a0[i] <= b1[i] + eps and b0[i] <= a1[i] + eps for i in range(3))


def main() -> int:
    cfg = make_env_config(stage=3, phase=1, contact_force_terminate_above=0.0)
    env = TyroEnv(cfg=cfg, render=False, seed=42)
    obs, _ = env.reset(seed=42)
    client = env.client
    ur = env.robot_A

    q, _ = ur.joint_state()
    print("HOME q (rad) =", [round(float(v), 4) for v in q])
    print("HOME q (deg) =", [round(math.degrees(float(v)), 2) for v in q])

    link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                          computeForwardKinematics=True,
                          physicsClientId=client)
    ee_pos = np.asarray(link[4])
    ee_orn = link[5]
    rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    z_world = rot[:, 2]
    angle_to_up = math.degrees(math.acos(max(-1.0, min(1.0, float(z_world[2])))))
    print(f"EE world pos  = ({ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f})")
    print(f"tool +Z (world) = ({z_world[0]:+.4f}, {z_world[1]:+.4f}, {z_world[2]:+.4f})")
    print(f"angle to world +Z = {angle_to_up:.2f} deg "
          f"({'OK' if angle_to_up < 5 else 'NOT VERTICAL'})")
    print(f"EE quaternion = ({ee_orn[0]:+.6e}, {ee_orn[1]:+.6e}, "
          f"{ee_orn[2]:+.6e}, {ee_orn[3]:+.6e})")
    x_world = rot[:, 0]
    y_world = rot[:, 1]
    print(f"tool +X (world) = ({x_world[0]:+.4f}, {x_world[1]:+.4f}, {x_world[2]:+.4f})")
    print(f"tool +Y (world) = ({y_world[0]:+.4f}, {y_world[1]:+.4f}, {y_world[2]:+.4f})")

    tire_pos, _ = env.scene.tire_pose()
    R = float(cfg.tire_outer_radius)
    grasp = np.asarray(tire_pos) + np.array([0.0, 0.0, -R])
    d_grasp = float(np.linalg.norm(ee_pos - grasp))
    print(f"tire COM       = {tuple(round(float(v),4) for v in tire_pos)}")
    print(f"grasp_target   = {tuple(round(float(v),4) for v in grasp)}")
    print(f"EE -> grasp    = {d_grasp*100:.2f} cm "
          f"(soft={cfg.approach_tol_soft*100:.0f} cm, "
          f"hard={cfg.approach_radius_tol*100:.0f} cm)")

    blockers = []
    for uid in (env.handles.tire, env.handles.vehicle, env.handles.truck_uid):
        if uid is not None and uid >= 0:
            blockers.append(uid)
    for bi in range(p.getNumBodies(physicsClientId=client)):
        if bi in (env.handles.plane, env.handles.tire, env.handles.vehicle,
                  env.handles.truck_uid, ur.uid, env.robot_B.uid):
            continue
        pos, _ = p.getBasePositionAndOrientation(bi, physicsClientId=client)
        if -2.2 < pos[0] < -1.6 and -0.6 < pos[1] < 0.6 and -0.7 < pos[2] < 0.0:
            blockers.append(bi)

    n_links = p.getNumJoints(ur.uid, physicsClientId=client)
    any_overlap = False
    detail = []
    for li in range(-1, n_links):
        try:
            a = p.getAABB(ur.uid, li, physicsClientId=client)
        except Exception:
            continue
        for bu in blockers:
            bn = p.getNumJoints(bu, physicsClientId=client)
            for bl in range(-1, bn):
                try:
                    b = p.getAABB(bu, bl, physicsClientId=client)
                except Exception:
                    continue
                if aabb_intersect(a, b):
                    any_overlap = True
                    detail.append((li, bu, bl))
    print(f"AABB overlap (UR10 vs blockers): "
          f"{'NONE' if not any_overlap else f'YES ({len(detail)} pairs)'}")
    if any_overlap:
        for d in detail[:10]:
            print(f"   link {d[0]} <-> body {d[1]} link {d[2]}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
