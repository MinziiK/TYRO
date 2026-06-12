"""Ground-truth reachability: bypass pybullet IK entirely.

Direct optimization over the 6 arm joints (scipy least_squares, multi-start)
minimizing [tool-tip position error ; coaxial alignment error]. If THIS cannot
reach, the pose is truly outside the workspace; if it can, pybullet IK was the
problem all along.
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
L = float(cfg.bolt_length)
warm, _ = rb.joint_state()
rng = np.random.default_rng(0)
lo, hi = rb.arm.lower, rb.arm.upper


def fk(q):
    for s, qq in zip(rb.arm.indices, q):
        p.resetJointState(rb.uid, int(s), float(qq), physicsClientId=env.client)
    ee, eq = rb.ee_pose()
    return np.asarray(ee), np.asarray(quat_axis(eq, "z"))


def solve(pos, want_z, n_starts=60):
    best_q, best = None, (1e9, 1e9)
    for si in range(n_starts):
        q0 = np.asarray(warm) if si == 0 else rng.uniform(lo, hi)

        def resid(q):
            ee, gz = fk(q)
            # sign-free coaxial: tool +Z may align with +axis or -axis
            mis = min(np.linalg.norm(gz - want_z), np.linalg.norm(gz + want_z))
            return np.concatenate([(ee - pos) * 10.0, [mis * 2.0]])

        try:
            r = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10,
                              ftol=1e-10, max_nfev=400, diff_step=1e-4)
        except Exception:
            continue
        ee, gz = fk(r.x)
        pe = float(np.linalg.norm(ee - pos))
        ae = float(np.degrees(np.arccos(np.clip(abs(np.dot(gz, want_z)),
                                                -1, 1))))
        if (pe, ae) < best:
            best, best_q = (pe, ae), r.x.copy()
        if pe < 0.003 and ae < 2.0:
            break
    return best, best_q


print("bolt | depth   | posErr  angErr  (scipy multi-start, 60 restarts)")
for idx in [3, 4, 5, 6, 7]:
    a = env._nut_axis_unit(idx)
    for name, axial in [("stage", env._nut_staging_axial()),
                        ("base ", -0.5 * L)]:
        tgt = env._nut_point_on_axis(idx, axial)
        (pe, ae), _ = solve(tgt, -a)
        print(f"  {idx}  | {name}  | {pe*100:6.2f}cm {ae:6.1f}deg")
env.close()
