"""Why are bolts 4-7 'unreachable'? Separate raw distance vs orientation
constraint vs joint limits.

For each bolt: print straight-line distance from B's shoulder to the staging /
base targets, then IK error (a) position-only (orientation free) and
(b) coaxial-constrained. If (a) succeeds but (b) fails -> orientation/joint
limits, not reach radius.
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
L = float(cfg.bolt_length)
warm, _ = rb.joint_state()
warm = np.asarray(warm, dtype=np.float64)
rng = np.random.default_rng(0)

base_pos, _ = p.getBasePositionAndOrientation(rb.uid, physicsClientId=env.client)
# shoulder = link 1 world position (UR10e shoulder_link)
sh = np.asarray(p.getLinkState(rb.uid, 1, physicsClientId=env.client)[4])
print(f"B base = {np.round(base_pos,3)}  shoulder = {np.round(sh,3)}")
print(f"arm joint limits lo={np.round(rb.arm.lower,2)}")
print(f"                 hi={np.round(rb.arm.upper,2)}")


def restore():
    for s, q in zip(rb.arm.indices, warm):
        p.resetJointState(rb.uid, int(s), float(q), physicsClientId=env.client)


def ik_err(pos, quat=None, tries=8):
    best = 1e9
    for si in range(tries):
        seed = warm if si == 0 else rng.uniform(rb.arm.lower, rb.arm.upper)
        ik = np.asarray(seed, dtype=np.float64).copy()
        for _ in range(12):
            kw = dict(lowerLimits=rb.arm.lower.tolist(),
                      upperLimits=rb.arm.upper.tolist(),
                      jointRanges=rb.arm.range.tolist(),
                      restPoses=ik.tolist(),
                      maxNumIterations=300, residualThreshold=1e-6,
                      physicsClientId=env.client)
            if quat is None:
                sol = np.asarray(p.calculateInverseKinematics(
                    rb.uid, rb.EE_LINK_INDEX, pos.tolist(), **kw))
            else:
                sol = np.asarray(p.calculateInverseKinematics(
                    rb.uid, rb.EE_LINK_INDEX, pos.tolist(), list(quat), **kw))
            arm = np.clip(sol[rb._ik_arm_slots], rb.arm.lower, rb.arm.upper)
            for s, q in zip(rb.arm.indices, arm):
                p.resetJointState(rb.uid, int(s), float(q),
                                  physicsClientId=env.client)
            ik = sol
            e = float(np.linalg.norm(np.asarray(rb.ee_pose()[0]) - pos))
            if e < 0.005:
                break
        best = min(best, e)
        restore()
        if best < 0.005:
            break
    return best


def coax_err(pos, want_z):
    base_quat = env._quat_align_tool_z(want_z)
    best = 1e9
    for ri in range(16):
        rq = np.asarray(axisangle3_to_quat(want_z * (2 * math.pi * ri / 16)))
        _, tq = p.multiplyTransforms([0, 0, 0], rq.tolist(),
                                     [0, 0, 0], base_quat.tolist())
        best = min(best, ik_err(pos, quat=tq, tries=4))
        if best < 0.005:
            break
    return best


print("\nbolt |  target(stage)        | dist(sh) | posOnlyIK | coaxIK")
for idx in range(len(env.handles.bolts)):
    a = env._nut_axis_unit(idx)
    p_stage = env._nut_point_on_axis(idx, env._nut_staging_axial())
    p_base = env._nut_point_on_axis(idx, -0.5 * L)
    d_s = float(np.linalg.norm(p_stage - sh))
    d_b = float(np.linalg.norm(p_base - sh))
    e_pos = ik_err(p_stage)
    e_cx = coax_err(p_stage, -a)
    e_pos_b = ik_err(p_base)
    e_cx_b = coax_err(p_base, -a)
    print(f"  {idx}  | {np.round(p_stage,2)} | "
          f"st {d_s:4.2f}m ba {d_b:4.2f}m | "
          f"st {e_pos*100:5.1f}cm ba {e_pos_b*100:5.1f}cm | "
          f"st {e_cx*100:5.1f}cm ba {e_cx_b*100:5.1f}cm")
env.close()
