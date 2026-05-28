"""Hold the current EE position fixed, rotate the gripper so tool +Z = world +Z.

Solves PyBullet IK at::

    target_pos = current EE position (from HOME_POSE FK)
    target_orn = R_z(π) = (0, 0, 1, 0)   ⇒ tool +Z = world +Z, tool +X = world −X

Reports the joint vector. Use it as the new ``UR10Robot.HOME_POSE`` if
the result satisfies the orientation requirement.
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


def main() -> int:
    cfg = make_env_config(stage=3, phase=1, contact_force_terminate_above=0.0)
    env = TyroEnv(cfg=cfg, render=False, seed=42)
    obs, _ = env.reset(seed=42)
    client = env.client
    ur = env.robot_A

    link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                          computeForwardKinematics=True,
                          physicsClientId=client)
    cur_pos = np.asarray(link[4])
    print(f"Current EE pos: ({cur_pos[0]:+.4f}, {cur_pos[1]:+.4f}, {cur_pos[2]:+.4f})")

    target_pos = cur_pos.tolist()
    candidates = [
        ("R_z(π)  tool+Z=+Z, tool+X=-X", [0.0, 0.0, 1.0, 0.0]),
        ("identity tool+Z=+Z, tool+X=+X", [0.0, 0.0, 0.0, 1.0]),
        ("R_z(+π/2) tool+Z=+Z, tool+X=-Y", [0.0, 0.0, 0.7071, 0.7071]),
        ("R_z(-π/2) tool+Z=+Z, tool+X=+Y", [0.0, 0.0, -0.7071, 0.7071]),
    ]

    for label, target_orn in candidates:
        ik = p.calculateInverseKinematics(
            ur.uid, ur.EE_LINK_INDEX,
            target_pos, target_orn,
            lowerLimits=ur.arm.lower.tolist(),
            upperLimits=ur.arm.upper.tolist(),
            jointRanges=ur.arm.range.tolist(),
            restPoses=ur.arm.rest.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=client,
        )
        ik = np.asarray(ik, dtype=np.float64)
        arm_q = ik[ur._ik_arm_slots]
        # Apply and read achieved pose
        for idx, qi in zip(ur.arm.indices, arm_q):
            p.resetJointState(ur.uid, idx, float(qi), 0.0, physicsClientId=client)
        ls = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                            computeForwardKinematics=True,
                            physicsClientId=client)
        ee = np.asarray(ls[4])
        orn = ls[5]
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        z_world = rot[:, 2]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(z_world[2])))))
        pos_err = float(np.linalg.norm(ee - cur_pos))

        print(f"\n[{label}]")
        print(f"  q (deg) = {[round(math.degrees(float(v)), 2) for v in arm_q]}")
        print(f"  achieved EE pos = ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})  "
              f"err={pos_err*100:.2f} cm")
        print(f"  tool +Z = ({rot[0,2]:+.4f}, {rot[1,2]:+.4f}, {rot[2,2]:+.4f})  "
              f"angle to +Z = {ang:.2f}°")
        print(f"  tool +X = ({rot[0,0]:+.4f}, {rot[1,0]:+.4f}, {rot[2,0]:+.4f})")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
