"""Corridor probe v2: strong iterative IK at staging AND hub-face base, then
sweep the joint-space stroke and measure B<->A / B<->tire clearance.

Answers: 'can B's arm fit near A and do the +-Y insert stroke?' per bolt.
"""
import math
import numpy as np
import pybullet as p
from src.config import make_env_config
from src.env import TyroEnv
from src.env.utils import axisangle3_to_quat

cfg = make_env_config(
    3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True,
    USE_DOMAIN_RANDOMIZATION=False, nut_a_hold_jitter_rad=0.0,
)
env = TyroEnv(cfg=cfg, render=False, seed=0)
env.reset(seed=0)
rb = env.robot_B
ua = env.robot_A
tire = env.handles.tire
L = float(cfg.bolt_length)
warm, _ = rb.joint_state()
warm = np.asarray(warm, dtype=np.float64)
rng = np.random.default_rng(0)


def solve(pos, want_z, gate=0.03):
    """Iterative-refine roll/seed search (the validated strong solver).
    Returns (arm_q, err)."""
    base_quat = env._quat_align_tool_z(want_z)
    best_q, best_err = None, 1e9
    for ri in range(16):
        roll = 2.0 * math.pi * ri / 16
        rq = np.asarray(axisangle3_to_quat(want_z * roll))
        _, tq = p.multiplyTransforms([0, 0, 0], rq.tolist(),
                                     [0, 0, 0], base_quat.tolist())
        for si in range(6):
            seed = warm if si == 0 else rng.uniform(rb.arm.lower, rb.arm.upper)
            ik = np.asarray(seed, dtype=np.float64).copy()
            for _ in range(12):
                sol = np.asarray(p.calculateInverseKinematics(
                    rb.uid, rb.EE_LINK_INDEX, pos.tolist(), list(tq),
                    lowerLimits=rb.arm.lower.tolist(),
                    upperLimits=rb.arm.upper.tolist(),
                    jointRanges=rb.arm.range.tolist(), restPoses=ik.tolist(),
                    maxNumIterations=300, residualThreshold=1e-6,
                    physicsClientId=env.client))
                arm = np.clip(sol[rb._ik_arm_slots], rb.arm.lower, rb.arm.upper)
                for s, q in zip(rb.arm.indices, arm):
                    p.resetJointState(rb.uid, int(s), float(q),
                                      physicsClientId=env.client)
                ik = sol
                if np.linalg.norm(np.asarray(rb.ee_pose()[0]) - pos) < 0.01:
                    break
            err = float(np.linalg.norm(np.asarray(rb.ee_pose()[0]) - pos))
            if err < best_err:
                best_err = err
                best_q = np.clip(np.asarray(ik)[rb._ik_arm_slots],
                                 rb.arm.lower, rb.arm.upper).copy()
            for s, q in zip(rb.arm.indices, warm):
                p.resetJointState(rb.uid, int(s), float(q),
                                  physicsClientId=env.client)
        if best_err < 0.005:
            break
    return best_q, best_err


print("bolt | stageErr baseErr | maxLat | minB-A | minB-tire  (stroke=staging->base)")
for idx in range(len(env.handles.bolts)):
    a = env._nut_axis_unit(idx)
    want_z = -a
    p_stage = env._nut_point_on_axis(idx, env._nut_staging_axial())
    p_base = env._nut_point_on_axis(idx, -0.5 * L)
    q_s, e_s = solve(p_stage, want_z)
    q_b, e_b = solve(p_base, want_z)
    if e_s > 0.04 or e_b > 0.04:
        print(f"  {idx}  | {e_s*100:6.1f}cm {e_b*100:6.1f}cm |  UNREACHABLE")
        continue
    min_ba = min_bt = float("inf")
    max_lat = 0.0
    for t in np.linspace(0.0, 1.0, 25):
        q = (1.0 - t) * q_s + t * q_b
        for s, qq in zip(rb.arm.indices, q):
            p.resetJointState(rb.uid, int(s), float(qq),
                              physicsClientId=env.client)
        _, lat, _ = env._nut_axial_lateral(idx)
        max_lat = max(max_lat, float(lat))
        cps = p.getClosestPoints(rb.uid, ua.uid, distance=5.0,
                                 physicsClientId=env.client)
        min_ba = min(min_ba, min((c[8] for c in cps), default=float("inf")))
        cps = p.getClosestPoints(rb.uid, tire, distance=5.0,
                                 physicsClientId=env.client)
        min_bt = min(min_bt, min((c[8] for c in cps), default=float("inf")))
    for s, qq in zip(rb.arm.indices, warm):
        p.resetJointState(rb.uid, int(s), float(qq), physicsClientId=env.client)
    print(f"  {idx}  | {e_s*100:6.1f}cm {e_b*100:6.1f}cm | {max_lat*100:5.1f}cm | "
          f"{min_ba*100:6.1f}cm | {min_bt*100:6.1f}cm")
env.close()
