"""Reset to HOME, then take a handful of zero-Δpos lock-mode steps. The
gripper must not move and wrist_3 must stay at 0 — proving that
``FINAL_LOCK_QUATERNION`` is a fixed point of the IK round-trip.
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


def forearm_axis(env: TyroEnv) -> np.ndarray:
    """Return forearm link unit vector in world frame (elbow → wrist_1)."""
    ur = env.robot_A
    # forearm_link comes right after elbow_joint; wrist_1_link comes after.
    # We probe by joint index — elbow_joint is the 3rd UR10 arm joint
    # (index in arm.indices[2]); the link that follows is forearm.
    elbow_idx = ur.arm.indices[2]   # elbow_joint
    w1_idx = ur.arm.indices[3]      # wrist_1_joint
    ls_elbow = p.getLinkState(ur.uid, elbow_idx, computeForwardKinematics=True,
                              physicsClientId=env.client)
    ls_w1 = p.getLinkState(ur.uid, w1_idx, computeForwardKinematics=True,
                           physicsClientId=env.client)
    p_elbow = np.asarray(ls_elbow[4])
    p_w1 = np.asarray(ls_w1[4])
    v = p_w1 - p_elbow
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


def report(env: TyroEnv, label: str) -> None:
    ur = env.robot_A
    q, _ = ur.joint_state()
    link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                          computeForwardKinematics=True,
                          physicsClientId=env.client)
    ee = np.asarray(link[4])
    orn = link[5]
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    fa = forearm_axis(env)
    fa_tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(abs(fa[2]))))))
    print(f"\n[{label}]")
    print("  q (deg) =", [round(math.degrees(float(v)), 3) for v in q])
    print(f"  shoulder_lift + elbow = {math.degrees(float(q[1] + q[2])):+.2f}° "
          f"(target +90°)")
    print(f"  EE pos       = ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})")
    print(f"  tool +Z      = ({rot[0,2]:+.4f}, {rot[1,2]:+.4f}, {rot[2,2]:+.4f})")
    print(f"  tool +X      = ({rot[0,0]:+.4f}, {rot[1,0]:+.4f}, {rot[2,0]:+.4f})")
    print(f"  forearm axis = ({fa[0]:+.4f}, {fa[1]:+.4f}, {fa[2]:+.4f}) "
          f"tilt from world +Z = {fa_tilt:.2f}°")


def main() -> int:
    cfg = make_env_config(stage=3, phase=1, contact_force_terminate_above=0.0)
    env = TyroEnv(cfg=cfg, render=False, seed=42)
    obs, _ = env.reset(seed=42)
    report(env, "after reset")

    # Zero action — Δpos=0, Δrot ignored. Lock should be a fixed point.
    n = env.action_space.shape[0]
    zero = np.zeros(n, dtype=np.float32)
    for i in range(5):
        env.step(zero)
    report(env, "after 5 zero-action steps")

    # Pure +Z lift of 5 cm via Δpos only.
    cmd = np.zeros(n, dtype=np.float32)
    cmd[2] = +1.0   # +Z scaled by action.pos_scale per step
    for i in range(10):
        env.step(cmd)
    report(env, "after 10 +Z lift steps")

    # Lateral +X then +Y to stress shoulder rotation as well.
    obs, _ = env.reset(seed=42)
    cmd = np.zeros(n, dtype=np.float32)
    cmd[0] = +1.0
    for i in range(10):
        env.step(cmd)
    report(env, "after 10 +X lateral steps")

    obs, _ = env.reset(seed=42)
    cmd = np.zeros(n, dtype=np.float32)
    cmd[1] = +1.0
    for i in range(10):
        env.step(cmd)
    report(env, "after 10 +Y lateral steps")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
