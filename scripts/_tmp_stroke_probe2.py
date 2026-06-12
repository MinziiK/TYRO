"""Full-corridor probe with ground-truth (scipy) IK for ALL 10 bolts.

For each bolt: solve staging + hub-face base joints with direct joint
optimization, ensuring the two solutions share a branch (base solve is seeded
from the staging solution). Then sweep the joint lerp (the insert stroke) and
measure B<->A and B<->tire min surface distance, plus max lateral deviation.
"""
import numpy as np
import pybullet as p
from scipy.optimize import least_squares
from src.config import make_env_config
from src.env import TyroEnv
from src.env.utils import quat_axis

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
rng = np.random.default_rng(0)
lo, hi = rb.arm.lower, rb.arm.upper


def fk(q):
    for s, qq in zip(rb.arm.indices, q):
        p.resetJointState(rb.uid, int(s), float(qq), physicsClientId=env.client)
    ee, eq = rb.ee_pose()
    return np.asarray(ee), np.asarray(quat_axis(eq, "z"))


def solve(pos, want_z, seed_q=None, n_starts=60):
    best_q, best = None, (1e9, 1e9)
    starts = []
    if seed_q is not None:
        starts.append(np.asarray(seed_q))
    starts.append(np.asarray(warm))
    while len(starts) < n_starts:
        starts.append(rng.uniform(lo, hi))
    for q0 in starts:
        def resid(q):
            ee, gz = fk(q)
            mis = min(np.linalg.norm(gz - want_z), np.linalg.norm(gz + want_z))
            return np.concatenate([(ee - pos) * 10.0, [mis * 2.0]])
        try:
            r = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10,
                              ftol=1e-10, max_nfev=400, diff_step=1e-4)
        except Exception:
            continue
        ee, gz = fk(r.x)
        pe = float(np.linalg.norm(ee - pos))
        ae = float(np.degrees(np.arccos(np.clip(abs(np.dot(gz, want_z)), -1, 1))))
        if (pe, ae) < best:
            best, best_q = (pe, ae), r.x.copy()
        if pe < 0.003 and ae < 2.0:
            break
    return best, best_q


print("bolt | stagePE basePE | maxLat | minB-A | minB-tire | dq(stroke,rad)")
for idx in range(len(env.handles.bolts)):
    a = env._nut_axis_unit(idx)
    p_stage = env._nut_point_on_axis(idx, env._nut_staging_axial())
    p_base = env._nut_point_on_axis(idx, -0.5 * L)
    (pe_s, _), q_s = solve(p_stage, -a)
    (pe_b, _), q_b = solve(p_base, -a, seed_q=q_s, n_starts=20)
    if pe_s > 0.02 or pe_b > 0.02 or q_s is None or q_b is None:
        print(f"  {idx}  | {pe_s*100:5.1f}cm {pe_b*100:5.1f}cm | SOLVE FAIL")
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
    dq = float(np.max(np.abs(q_b - q_s)))
    print(f"  {idx}  | {pe_s*100:5.1f}cm {pe_b*100:5.1f}cm | {max_lat*100:5.1f}cm"
          f" | {min_ba*100:6.1f}cm | {min_bt*100:6.1f}cm | {dq:5.2f}")
env.close()
