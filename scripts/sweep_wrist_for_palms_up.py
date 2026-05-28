"""Fix shoulder_pan / shoulder_lift / elbow at the user-specified values
and sweep wrist_1, wrist_2, wrist_3 to find a configuration whose
    tool +Z ≈ world +Z   (palm faces straight up)
    tool +X ≈ world −X   (gripper closure axis points at the tire)
Both within ~2°. Picks the EE position that is highest (largest Z),
clear of all blockers, and reasonably far from the grasp target.
"""
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


FIXED_SHOULDER_PAN = math.pi          # +180°
FIXED_SHOULDER_LIFT = math.radians(-65)
FIXED_ELBOW = math.radians(+130)


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

    tire_pos, _ = env.scene.tire_pose()
    R = float(cfg.tire_outer_radius)
    grasp = np.asarray(tire_pos) + np.array([0.0, 0.0, -R])

    n = len(ur.arm.indices)
    indices = ur.arm.indices
    print(f"FIXED: pan={math.degrees(FIXED_SHOULDER_PAN):.2f}, "
          f"lift={math.degrees(FIXED_SHOULDER_LIFT):.2f}, "
          f"elbow={math.degrees(FIXED_ELBOW):.2f}")

    # Two-stage sweep:
    #   1. coarse over wrist_1, wrist_2 (wrist_3=0) to find candidates whose
    #      tool +Z is within 2° of world +Z;
    #   2. for each survivor, solve wrist_3 analytically so tool +X = world −X.
    step = math.radians(2)   # 2° resolution
    w1_range = np.arange(-math.pi, math.pi + 1e-6, step)
    w2_range = np.arange(-math.pi, math.pi + 1e-6, step)
    n_ok = 0
    best = None
    for w1 in w1_range:
        for w2 in w2_range:
            q = [FIXED_SHOULDER_PAN, FIXED_SHOULDER_LIFT, FIXED_ELBOW,
                 float(w1), float(w2), 0.0]
            for idx, qi in zip(indices, q):
                p.resetJointState(ur.uid, idx, qi, 0.0,
                                  physicsClientId=client)
            link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                                  computeForwardKinematics=True,
                                  physicsClientId=client)
            ee = np.asarray(link[4])
            orn = link[5]
            rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            z_world = rot[:, 2]
            if z_world[2] < math.cos(math.radians(2.0)):
                continue   # tool +Z not within 2° of world +Z
            # Solve wrist_3 so that tool +X aligns with world −X (closure
            # axis points at the tire, which sits at world −X from the EE).
            # Currently (w3=0): tool +X = rot[:,0], tool +Y = rot[:,1].
            # Rotating by w3 about tool +Z gives:
            #   new_X = cos(w3)*tool+X + sin(w3)*tool+Y.
            # We want new_X · world(-X) maximized, i.e.
            #   -cos(w3)*rot[0,0] - sin(w3)*rot[0,1] → max.
            #   ⇔ w3 = atan2(-rot[0,1], -rot[0,0]).
            w3 = math.atan2(-rot[0, 1], -rot[0, 0])
            # Wrap into [-pi, pi].
            if w3 > math.pi: w3 -= 2 * math.pi
            if w3 < -math.pi: w3 += 2 * math.pi
            q[5] = float(w3)
            p.resetJointState(ur.uid, indices[5], q[5], 0.0,
                              physicsClientId=client)
            link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                                  computeForwardKinematics=True,
                                  physicsClientId=client)
            ee = np.asarray(link[4])
            orn = link[5]
            rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            z_world = rot[:, 2]
            x_world = rot[:, 0]
            # Re-check alignment after w3 update.
            if z_world[2] < math.cos(math.radians(2.0)):
                continue
            # |tool +X · world (-X)| should be ≈ 1 (allow 2° error)
            if -x_world[0] < math.cos(math.radians(2.0)):
                continue
            n_links = p.getNumJoints(ur.uid, physicsClientId=client)
            any_ov = False
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
                            any_ov = True
                            break
                    if any_ov:
                        break
                if any_ov:
                    break
            if any_ov:
                continue
            d_grasp = float(np.linalg.norm(ee - grasp))
            # Prefer: high Z, close to grasp x/y, not penetrating tire.
            tire_xy = float(np.linalg.norm(ee[:2] - np.asarray(tire_pos)[:2]))
            score = (
                -ee[2] * 2.0                  # higher EE = lower score (better)
                + 1.0 * abs(d_grasp - 0.6)    # prefer ~0.6 m from grasp target
                + 0.5 * tire_xy                # close XY to tire is good
            )
            n_ok += 1
            if best is None or score < best[0]:
                best = (score, q, ee, z_world, d_grasp)

    print(f"viable candidates: {n_ok}")
    if best is None:
        print("NO viable wrist configuration with tool +Z = world +Z AND tool +X = world -X.")
        env.close()
        return 1
    score, q, ee, z_world, d_grasp = best
    print(f"\nBEST score={score:.4f}")
    print(f"  wrist_1 = {q[3]:+.4f} rad ({math.degrees(q[3]):+.2f} deg)")
    print(f"  wrist_2 = {q[4]:+.4f} rad ({math.degrees(q[4]):+.2f} deg)")
    print(f"  wrist_3 = {q[5]:+.4f} rad ({math.degrees(q[5]):+.2f} deg)")
    # Recompute final FK at the winning joint vector for a clean report.
    for idx, qi in zip(indices, q):
        p.resetJointState(ur.uid, idx, qi, 0.0, physicsClientId=client)
    link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                          computeForwardKinematics=True,
                          physicsClientId=client)
    ee_final = np.asarray(link[4])
    orn_final = link[5]
    rot_final = np.array(p.getMatrixFromQuaternion(orn_final)).reshape(3, 3)
    print(f"  EE pos      = ({ee_final[0]:+.4f}, {ee_final[1]:+.4f}, {ee_final[2]:+.4f})")
    print(f"  tool +X     = ({rot_final[0,0]:+.4f}, {rot_final[1,0]:+.4f}, {rot_final[2,0]:+.4f})")
    print(f"  tool +Y     = ({rot_final[0,1]:+.4f}, {rot_final[1,1]:+.4f}, {rot_final[2,1]:+.4f})")
    print(f"  tool +Z     = ({rot_final[0,2]:+.4f}, {rot_final[1,2]:+.4f}, {rot_final[2,2]:+.4f})")
    print(f"  EE quat     = ({orn_final[0]:+.6e}, {orn_final[1]:+.6e}, "
          f"{orn_final[2]:+.6e}, {orn_final[3]:+.6e})")
    print(f"  EE -> grasp = {d_grasp*100:.2f} cm")
    print()
    print("HOME_POSE = (")
    for qi in q:
        print(f"    {qi:+.4f},")
    print(")")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
