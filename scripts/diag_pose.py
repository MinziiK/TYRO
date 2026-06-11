"""Diagnose WHY specific bolts fail at alpha=1.0 (hot-start right at the bolt).

For each bolt we measure, with the current policy:
  - START pose error right after hot-start teleport  (theta, lateral, d_stage)
    -> tells us if the per-bolt SEED already achieves the coaxial pose.
  - WORST/END pose error while the policy runs the leg, and where it ends
    (subphase, macro_stage) -> tells us if the POLICY breaks alignment.
  - whether a roll-free multi-seed IK can reach a coaxial seat (pose attainable
    by SOME branch at all).

This isolates: "pose unreachable (IK/branch)" vs "policy can't hold the pose".
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from src.config import make_env_config
from src.env import TyroEnv

V20 = dict(nut_b_align_servo=True, nut_a_kinematic_freeze=True, nut_collision_fail=True,
    nut_b_solo_action=True, nut_arrive_lat_tol=0.015, nut_seat_lat_mult=1.0,
    nut_b_axial_insert_servo=True, nut_insert_depth_tol=0.007,
    nut_b_insert_branch_search=True)


def diag_bolt(model, b):
    order = [b] + [k for k in range(10) if k != b]
    cfg = make_env_config(3, 1, render=False, scene_layout='fanuc_spacious',
        nut_fastening_task=True, nut_pure_rl=True, nut_b_hotstart_enable=True,
        nut_b_hotstart_alpha=1.0, nut_b_hotstart_random_bolt=False,
        nut_per_leg_episode=True, max_steps=400, terminate_on='never',
        nut_bolt_order=tuple(order), USE_DOMAIN_RANDOMIZATION=False, **V20)
    env = TyroEnv(cfg=cfg, render=False, seed=11 + b)
    # instrument branch search
    orig = env._nut_switch_to_seat_branch
    cnt = {'n': 0, 'ok': 0}
    def wrap(i, _o=orig, _c=cnt):
        _c['n'] += 1
        r = _o(i)
        if r:
            _c['ok'] += 1
        return r
    env._nut_switch_to_seat_branch = wrap
    obs, _ = env.reset(seed=11 + b)
    idx = int(env._nut_target_idx)
    L = float(getattr(cfg, "bolt_length", 0.10))

    ang_gate = float(getattr(cfg, "nut_arrive_ang_tol_rad", np.deg2rad(35.0)))
    pos_gate = float(getattr(cfg, "nut_arrive_pos_tol", 0.05))
    lat_gate = float(getattr(cfg, "nut_arrive_lat_tol", 0.03))

    def errs():
        ax, lat, th = env._nut_axial_lateral(idx)
        d_stage = float(np.hypot(ax - env._nut_staging_axial(), lat))
        return d_stage, lat, np.degrees(th)

    d0, lat0, th0 = errs()
    # does a roll-free multi-seed IK find a coaxial seat at all?
    qbest = env._nut_best_seat_q(idx)
    seat_attainable = qbest is not None

    ok = False
    end_reason = "ran out"
    reached_macro = False
    best_d, best_lat, best_th = d0, lat0, th0
    deepest_axial = 1e9      # most negative axial reached during INSERT (stage 0)
    max_macro_step = 0
    n_sub0 = 0
    n_sub1 = 0
    t_first_macro = -1
    end_sub, end_stage = 0, 0
    flags = (bool(getattr(cfg, "nut_b_insert_branch_search", False)),
             bool(getattr(cfg, "nut_pure_rl", False)),
             int(getattr(cfg, "nut_insert_reseat_after", 40)))
    for t in range(400):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if int(env._nut_subphase) == 1:
            reached_macro = True
            n_sub1 += 1
            if t_first_macro < 0:
                t_first_macro = t
            max_macro_step = max(max_macro_step, int(getattr(env, "_nut_macro_step", 0)))
            if int(getattr(env, "_nut_macro_stage", 0)) == 0:
                ax, _l, _t = env._nut_axial_lateral(idx)
                deepest_axial = min(deepest_axial, ax)
        else:
            n_sub0 += 1
        end_sub = int(env._nut_subphase)
        end_stage = int(getattr(env, "_nut_macro_stage", 0))
        if env._nut_subphase == 0:  # still approaching: track best alignment
            d, lat, th = errs()
            if (th < best_th):
                best_d, best_lat, best_th = d, lat, th
        if info.get('fastened') or info.get('n_fastened', 0) > 0:
            ok = True
            break
        if term or trunc:
            end_reason = (f"term={term} trunc={trunc} "
                          f"keys={[k for k in info.keys() if any(s in k.lower() for s in ('coll','fail','reason','term'))]} "
                          f"vals={ {k: info[k] for k in info if any(s in k.lower() for s in ('coll','fail','reason'))} }")
            break
    else:
        end_reason = "ran out (no term/trunc)"
    de, late, the = errs()
    # axial shortfall: how far the deepest plunge stayed from seat target (-L/2)
    seat_target = -0.5 * L
    shortfall_cm = (deepest_axial - seat_target) * 100 if deepest_axial < 1e8 else -1.0

    # what is B touching right now (at the terminating step)?
    import pybullet as p
    hitbodies = set()
    cps = p.getContactPoints(bodyB=env.robot_B.uid, physicsClientId=env.client)
    for cp in cps:
        if cp[4] > 1:  # B link index > base
            hitbodies.add(int(cp[1]))
    # name the bodies
    def bname(uid):
        h = env.handles
        if uid == getattr(h, 'plane', -99): return 'floor'
        if uid == env.robot_A.uid: return 'robotA'
        if uid == getattr(h, 'vehicle', -99): return 'truck'
        if uid == getattr(h, 'cargo_back_wall', -99): return 'backwall'
        for r in (getattr(h, 'floor_rim', []) or []):
            if uid == r: return 'rim'
        return f'uid{uid}'
    hit = sorted({bname(u) for u in hitbodies})

    # is the seat branch (best_seat_q) collision-free?
    seat_branch_collfree = None
    if qbest is not None:
        qs, _ = env.robot_B.joint_state()
        for s, qq in zip(env.robot_B.arm.indices, qbest):
            p.resetJointState(env.robot_B.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=env.client)
        seat_branch_collfree = (not env._in_bad_collision())
        for s, qq in zip(env.robot_B.arm.indices, qs):
            p.resetJointState(env.robot_B.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=env.client)
    env.close()
    return dict(b=b, ok=ok, reached_macro=reached_macro, seat_attainable=seat_attainable,
                start=(d0, lat0, th0), best=(best_d, best_lat, best_th),
                end=(de, late, the), end_sub=end_sub, end_stage=end_stage,
                fired=cnt['n'], fired_ok=cnt['ok'], shortfall_cm=shortfall_cm,
                max_macro_step=max_macro_step, flags=flags,
                n_sub0=n_sub0, n_sub1=n_sub1, t_first_macro=t_first_macro,
                end_reason=end_reason, hit=hit, seat_collfree=seat_branch_collfree,
                gates=(pos_gate, lat_gate, np.degrees(ang_gate)))


def seat_scan(b, tries=24, rolls=24, seeds=8):
    """Collision-AWARE exhaustive seat IK search for bolt b.

    Mirrors _nut_best_seat_q's geometry, but among IK candidates that reach the
    seat band (pos<7mm, coax<5deg) it separately tracks the best collision-FREE
    one. Decisive answer to: does a collision-free seat pose exist at all?
    """
    import math
    import pybullet as p
    order = [b] + [k for k in range(10) if k != b]
    cfg = make_env_config(3, 1, render=False, scene_layout='fanuc_spacious',
        nut_fastening_task=True, nut_pure_rl=True, nut_b_hotstart_enable=True,
        nut_b_hotstart_alpha=1.0, nut_b_hotstart_random_bolt=False,
        nut_per_leg_episode=True, max_steps=400, terminate_on='never',
        nut_bolt_order=tuple(order), USE_DOMAIN_RANDOMIZATION=False, **V20)
    env = TyroEnv(cfg=cfg, render=False, seed=11 + b)
    env.reset(seed=11 + b)
    idx = int(env._nut_target_idx)
    rb = env.robot_B
    L = float(getattr(cfg, "bolt_length", 0.10))
    axis = env._nut_axis_unit(idx)
    bolt = np.asarray(env.scene.bolt_pose(idx)[0], dtype=np.float64)
    seat_pt = bolt + axis * (-0.5 * L)
    want_z = -axis / max(float(np.linalg.norm(axis)), 1e-9)
    lo, hi = rb.arm.lower, rb.arm.upper
    n_seated = 0
    n_seated_cf = 0
    best_cf_res = 1e9
    best_any_res = 1e9
    coll_detail = {}   # "bodyname:Blink" -> count among seated colliding configs
    rng = np.random.default_rng(7 + b)
    for tri in range(tries):
        for ri in range(rolls):
            quat = list(env._quat_z_roll(want_z, 2.0 * math.pi * ri / rolls))
            for k in range(seeds):
                seed = (rb.arm.rest if k == 0 else rng.uniform(lo, hi)).tolist()
                ik = p.calculateInverseKinematics(
                    rb.uid, rb.EE_LINK_INDEX, seat_pt.tolist(), quat,
                    lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                    jointRanges=rb.arm.range.tolist(), restPoses=seed,
                    maxNumIterations=300, residualThreshold=1e-6,
                    physicsClientId=env.client)
                ik = np.asarray(ik, dtype=np.float64)
                if not (rb._ik_arm_slots and len(ik) > max(rb._ik_arm_slots)):
                    continue
                q = np.clip(ik[rb._ik_arm_slots], lo, hi)
                for s, qq in zip(rb.arm.indices, q):
                    p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                                      physicsClientId=env.client)
                p.performCollisionDetection(physicsClientId=env.client)
                ax, lat, th = env._nut_axial_lateral(idx)
                seated = abs(ax - (-0.5 * L)) < 0.007 and lat < 0.015
                res = abs(ax - (-0.5 * L)) + lat
                if seated:
                    n_seated += 1
                    best_any_res = min(best_any_res, res)
                    if not env._in_bad_collision():
                        n_seated_cf += 1
                        best_cf_res = min(best_cf_res, res)
                    else:
                        # record which body/link B touches at this seated pose
                        cps = p.getContactPoints(bodyB=rb.uid, physicsClientId=env.client)
                        h = env.handles
                        def nm(uid):
                            if uid == getattr(h, 'plane', -99): return 'floor'
                            if uid == env.robot_A.uid: return 'robotA'
                            if uid == getattr(h, 'vehicle', -99): return 'truck'
                            if uid == getattr(h, 'cargo_back_wall', -99): return 'backwall'
                            for r in (getattr(h, 'floor_rim', []) or []):
                                if uid == r: return 'rim'
                            return f'uid{uid}'
                        for cp in cps:
                            if cp[4] > 1:
                                key = f"{nm(cp[1])}:Blink{cp[4]}"
                                coll_detail[key] = coll_detail.get(key, 0) + 1
                        # also A-vs-B (bodyA=A): catch B-base vs A-arm
                        cpsab = p.getContactPoints(bodyA=env.robot_A.uid, bodyB=rb.uid,
                                                   physicsClientId=env.client)
                        for cp in cpsab:
                            if cp[3] > 2 or cp[4] > 2:
                                key = f"robotA(Al{cp[3]}):Blink{cp[4]}"
                                coll_detail[key] = coll_detail.get(key, 0) + 1
    env.close()
    top = sorted(coll_detail.items(), key=lambda kv: -kv[1])[:4]
    return dict(b=b, n_seated=n_seated, n_seated_cf=n_seated_cf,
                best_cf_res_mm=best_cf_res * 1000 if best_cf_res < 1e8 else -1,
                best_any_res_mm=best_any_res * 1000 if best_any_res < 1e8 else -1,
                coll_top=top)


def clean_seat_opt(b, restarts=80, rolls=12):
    """Collision-AWARE optimization: explicitly minimize tire penetration while
    seating coaxially. scipy least_squares residual = [pos, coax, tire_pen],
    many random restarts + roll sweep. Decisive test for 'does a tire-free
    coaxial seat exist?' (much stronger than random IK sampling)."""
    import math
    import pybullet as p
    from scipy.optimize import least_squares
    order = [b] + [k for k in range(10) if k != b]
    cfg = make_env_config(3, 1, render=False, scene_layout='fanuc_spacious',
        nut_fastening_task=True, nut_pure_rl=True, nut_b_hotstart_enable=True,
        nut_b_hotstart_alpha=1.0, nut_b_hotstart_random_bolt=False,
        nut_per_leg_episode=True, max_steps=400, terminate_on='never',
        nut_bolt_order=tuple(order), USE_DOMAIN_RANDOMIZATION=False, **V20)
    env = TyroEnv(cfg=cfg, render=False, seed=18)
    env.reset(seed=18)
    idx = int(env._nut_target_idx)
    rb = env.robot_B
    cl = env.client
    L = float(getattr(cfg, "bolt_length", 0.10))
    axis = env._nut_axis_unit(idx)
    bolt = np.asarray(env.scene.bolt_pose(idx)[0], dtype=np.float64)
    seat_pt = bolt + axis * (-0.5 * L)
    want_z = -axis / max(float(np.linalg.norm(axis)), 1e-9)
    lo, hi = rb.arm.lower, rb.arm.upper

    def set_q(q):
        for s, qq in zip(rb.arm.indices, q):
            p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=cl)

    def tire_pen():
        worst = 0.0
        cps = p.getClosestPoints(bodyA=env.handles.tire, bodyB=rb.uid,
                                 distance=0.05, physicsClientId=cl)
        for cp in cps:
            if cp[4] > 1 and len(cp) > 8:
                worst = min(worst, float(cp[8]))
        return worst  # <0 = penetration

    def resid(q):
        set_q(q)
        ee, eq = rb.ee_pose()
        ee = np.asarray(ee, dtype=np.float64)
        gz = np.asarray(quat_axis_z(eq), dtype=np.float64)
        mis = min(float(np.linalg.norm(gz - want_z)),
                  float(np.linalg.norm(gz + want_z)))
        # penetration beyond the 5mm tolerance the env uses
        pen = tire_pen()
        viol = max(0.0, -(pen) - 0.005)
        return np.concatenate([(ee - seat_pt) * 10.0, [mis * 2.0, viol * 40.0]])

    rng = np.random.default_rng(123 + b)
    best = None  # (seated, clean, pos_mm, pen_mm, q)
    for r in range(restarts):
        q0 = rb.arm.rest if r == 0 else rng.uniform(lo, hi)
        try:
            sol = least_squares(resid, np.asarray(q0, dtype=np.float64),
                                bounds=(lo, hi), xtol=1e-10, ftol=1e-10,
                                max_nfev=400, diff_step=2e-3)
        except Exception:
            continue
        q = np.clip(sol.x, lo, hi)
        set_q(q)
        ax, lat, th = env._nut_axial_lateral(idx)
        seated = abs(ax - (-0.5 * L)) < 0.007 and lat < 0.015
        pen = tire_pen()
        clean = pen >= -0.005
        pos_mm = (abs(ax - (-0.5 * L)) + lat) * 1000
        cand = (seated and clean, seated, -pen * 1000, pos_mm, q.copy())
        # rank: prefer seated+clean, then seated, then least penetration
        if best is None:
            best = cand
        else:
            if (cand[0], cand[1], -cand[2]) > (best[0], best[1], -best[2]):
                best = cand
        if best[0]:  # found seated+clean
            break
    env.close()
    return dict(b=b, found_clean=bool(best[0]), seated=bool(best[1]),
                pen_mm=best[2], pos_mm=best[3])


def quat_axis_z(q):
    from src.env.tyro_env import quat_axis
    return quat_axis(q, "z")


def _solve_clean_seat_q(env, idx, restarts=80):
    """Collision-aware least_squares seat solve for the LIVE env (returns q)."""
    import pybullet as p
    from scipy.optimize import least_squares
    rb = env.robot_B
    cl = env.client
    L = float(getattr(env.cfg, "bolt_length", 0.10))
    axis = env._nut_axis_unit(idx)
    bolt = np.asarray(env.scene.bolt_pose(idx)[0], dtype=np.float64)
    seat_pt = bolt + axis * (-0.5 * L)
    want_z = -axis / max(float(np.linalg.norm(axis)), 1e-9)
    lo, hi = rb.arm.lower, rb.arm.upper
    q_save, _ = rb.joint_state()

    def set_q(q):
        for s, qq in zip(rb.arm.indices, q):
            p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=cl)

    def tire_pen():
        worst = 0.0
        for cp in p.getClosestPoints(bodyA=env.handles.tire, bodyB=rb.uid,
                                     distance=0.05, physicsClientId=cl):
            if cp[4] > 1 and len(cp) > 8:
                worst = min(worst, float(cp[8]))
        return worst

    def resid(q):
        set_q(q)
        ee, eq = rb.ee_pose()
        ee = np.asarray(ee, dtype=np.float64)
        gz = np.asarray(quat_axis_z(eq), dtype=np.float64)
        mis = min(float(np.linalg.norm(gz - want_z)),
                  float(np.linalg.norm(gz + want_z)))
        viol = max(0.0, -(tire_pen()) - 0.005)
        return np.concatenate([(ee - seat_pt) * 10.0, [mis * 2.0, viol * 40.0]])

    rng = np.random.default_rng(123 + idx)
    best_q, best_key = None, None
    for r in range(restarts):
        q0 = rb.arm.rest if r == 0 else rng.uniform(lo, hi)
        try:
            sol = least_squares(resid, np.asarray(q0, dtype=np.float64),
                                bounds=(lo, hi), xtol=1e-10, ftol=1e-10,
                                max_nfev=400, diff_step=2e-3)
        except Exception:
            continue
        q = np.clip(sol.x, lo, hi)
        set_q(q)
        ax, lat, _t = env._nut_axial_lateral(idx)
        seated = abs(ax - (-0.5 * L)) < 0.007 and lat < 0.015
        pen = tire_pen()
        clean = pen >= -0.005
        key = (seated and clean, seated, pen)
        if best_key is None or key > best_key:
            best_key, best_q = key, q.copy()
        if best_key[0]:
            break
    set_q(q_save)
    rb._cmd_q = None
    return best_q


def _solve_clean_staging_q(env, idx, seat_q, restarts=40):
    """Staging config in the SAME clean branch as seat_q: IK to the staging
    point (outside the tip), seeded from seat_q, with a tire-penetration cost.
    Keeps the approach in the collision-free branch so the axial plunge stays
    clean all the way to the seat."""
    import pybullet as p
    from scipy.optimize import least_squares
    rb = env.robot_B
    cl = env.client
    axis = env._nut_axis_unit(idx)
    stage_ax = env._nut_staging_axial()
    stage_pt = env._nut_point_on_axis(idx, stage_ax)
    want_z = -axis / max(float(np.linalg.norm(axis)), 1e-9)
    lo, hi = rb.arm.lower, rb.arm.upper
    q_save, _ = rb.joint_state()

    def set_q(q):
        for s, qq in zip(rb.arm.indices, q):
            p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                              physicsClientId=cl)

    def tire_pen():
        worst = 0.0
        for cp in p.getClosestPoints(bodyA=env.handles.tire, bodyB=rb.uid,
                                     distance=0.05, physicsClientId=cl):
            if cp[4] > 1 and len(cp) > 8:
                worst = min(worst, float(cp[8]))
        return worst

    def resid(q):
        set_q(q)
        ee, eq = rb.ee_pose()
        ee = np.asarray(ee, dtype=np.float64)
        gz = np.asarray(quat_axis_z(eq), dtype=np.float64)
        mis = min(float(np.linalg.norm(gz - want_z)),
                  float(np.linalg.norm(gz + want_z)))
        viol = max(0.0, -(tire_pen()) - 0.005)
        return np.concatenate([(ee - stage_pt) * 10.0, [mis * 2.0, viol * 40.0]])

    rng = np.random.default_rng(321 + idx)
    best_q, best_key = None, None
    for r in range(restarts):
        q0 = seat_q if r == 0 else (
            seat_q + rng.uniform(-0.3, 0.3, size=len(seat_q)))
        q0 = np.clip(q0, lo, hi)
        try:
            sol = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10,
                                ftol=1e-10, max_nfev=300, diff_step=2e-3)
        except Exception:
            continue
        q = np.clip(sol.x, lo, hi)
        set_q(q)
        ee = np.asarray(rb.ee_pose()[0], dtype=np.float64)
        pe = float(np.linalg.norm(ee - stage_pt))
        pen = tire_pen()
        clean = pen >= -0.005
        key = (pe < 0.01 and clean, clean, -pe)
        if best_key is None or key > best_key:
            best_key, best_q = key, q.copy()
        if best_key[0]:
            break
    set_q(q_save)
    rb._cmd_q = None
    return best_q


def fix_test2(model, bolts):
    """Patch the hot-start seed to the CLEAN-branch staging config, run policy."""
    seated = 0
    fails = []
    for b in bolts:
        order = [b] + [k for k in range(10) if k != b]
        cfg = make_env_config(3, 1, render=False, scene_layout='fanuc_spacious',
            nut_fastening_task=True, nut_pure_rl=True, nut_b_hotstart_enable=True,
            nut_b_hotstart_alpha=1.0, nut_b_hotstart_random_bolt=False,
            nut_per_leg_episode=True, max_steps=400, terminate_on='never',
            nut_bolt_order=tuple(order), USE_DOMAIN_RANDOMIZATION=False, **V20)
        import pybullet as p
        env = TyroEnv(cfg=cfg, render=False, seed=11 + b)
        obs, _ = env.reset(seed=11 + b)
        idx = int(env._nut_target_idx)
        seat_q = _solve_clean_seat_q(env, idx)
        stage_q = _solve_clean_staging_q(env, idx, seat_q) if seat_q is not None else None
        # Teleport B directly into the clean-branch staging config, then
        # recompute obs so the policy sees the new pose.
        if stage_q is not None:
            rb = env.robot_B
            for s, qq in zip(rb.arm.indices, stage_q):
                p.resetJointState(rb.uid, int(s), float(qq), targetVelocity=0.0,
                                  physicsClientId=env.client)
            rb._cmd_q = None
            rb.drive_arm_targets(np.asarray(stage_q, dtype=np.float64))
            rb.last_target_pos = np.asarray(rb.ee_pose()[0], dtype=np.float64).copy()
            obs = env._compute_obs()
        ok = False
        for t in range(400):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if info.get('fastened') or info.get('n_fastened', 0) > 0:
                ok = True
                break
            if term or trunc:
                break
        seated += int(ok)
        if not ok:
            fails.append(b)
        env.close()
    print(f"FIXTEST2 (clean-branch approach) per_bolt_seat = "
          f"{seated}/{len(bolts)}  (failed: {fails})")


def fix_test(model, bolts):
    """For each bolt: reset, OVERRIDE _nut_base_q[idx] with a collision-free
    seat, run the policy, report fastened. Validates the env fix on the
    existing checkpoint without retraining."""
    seated = 0
    fails = []
    for b in bolts:
        order = [b] + [k for k in range(10) if k != b]
        cfg = make_env_config(3, 1, render=False, scene_layout='fanuc_spacious',
            nut_fastening_task=True, nut_pure_rl=True, nut_b_hotstart_enable=True,
            nut_b_hotstart_alpha=1.0, nut_b_hotstart_random_bolt=False,
            nut_per_leg_episode=True, max_steps=400, terminate_on='never',
            nut_bolt_order=tuple(order), USE_DOMAIN_RANDOMIZATION=False, **V20)
        env = TyroEnv(cfg=cfg, render=False, seed=11 + b)
        obs, _ = env.reset(seed=11 + b)
        idx = int(env._nut_target_idx)
        q_clean = _solve_clean_seat_q(env, idx)
        if q_clean is not None and idx < len(env._nut_base_q):
            env._nut_base_q[idx] = np.asarray(q_clean, dtype=np.float64)
        ok = False
        for t in range(400):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if info.get('fastened') or info.get('n_fastened', 0) > 0:
                ok = True
                break
            if term or trunc:
                break
        seated += int(ok)
        if not ok:
            fails.append(b)
        env.close()
    print(f"FIXTEST per_bolt_seat = {seated}/{len(bolts)}  (failed: {fails})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bolts", default="1,7,8,9,0,2")
    ap.add_argument("--seatscan", action="store_true",
                    help="collision-aware exhaustive seat IK search (no policy)")
    ap.add_argument("--cleanseat", action="store_true",
                    help="collision-aware OPTIMIZATION for a tire-free coaxial seat")
    ap.add_argument("--fixtest", action="store_true",
                    help="override _nut_base_q with clean seat + run policy")
    ap.add_argument("--fixtest2", action="store_true",
                    help="patch hot-start to clean-branch staging + run policy")
    args = ap.parse_args()
    if args.fixtest:
        m = PPO.load(args.model, device='cpu')
        bolts = [int(x) for x in args.bolts.split(",")]
        fix_test(m, bolts)
        return
    if args.fixtest2:
        m = PPO.load(args.model, device='cpu')
        bolts = [int(x) for x in args.bolts.split(",")]
        fix_test2(m, bolts)
        return
    if args.cleanseat:
        bolts = [int(x) for x in args.bolts.split(",")]
        print(f"{'b':>2} {'cleanSEAT?':>10} {'seated?':>8} "
              f"{'pen(mm)':>9} {'posErr(mm)':>11}")
        for b in bolts:
            d = clean_seat_opt(b)
            print(f"{d['b']:>2} {str(d['found_clean']):>10} {str(d['seated']):>8} "
                  f"{d['pen_mm']:7.1f}   {d['pos_mm']:8.1f}")
        return
    if args.seatscan:
        bolts = [int(x) for x in args.bolts.split(",")]
        print(f"{'b':>2} {'#seated':>8} {'#seatedCF':>10} "
              f"{'bestCF_res':>11} {'bestAny_res':>12}")
        for b in bolts:
            d = seat_scan(b)
            print(f"{d['b']:>2} {d['n_seated']:>8} {d['n_seated_cf']:>10} "
                  f"{d['best_cf_res_mm']:8.1f}mm {d['best_any_res_mm']:9.1f}mm  "
                  f"| collide: {d['coll_top']}")
        return
    m = PPO.load(args.model, device='cpu')
    bolts = [int(x) for x in args.bolts.split(",")]
    pg, lg, ag = None, None, None
    print(f"model={args.model}")
    rows = []
    for b in bolts:
        d = diag_bolt(m, b)
        rows.append(d)
        pg, lg, ag = d['gates']
    print(f"GATES: d_stage<{pg:.3f}m  lateral<{lg:.3f}m  theta<{ag:.1f}deg")
    f0 = rows[0]['flags']
    print(f"FLAGS: branch_search={f0[0]} pure_rl={f0[1]} reseat_after={f0[2]}  "
          f"(max_macro_step per bolt below)\n")
    print(f"{'b':>2} {'ok':>3} {'seatIK':>6} {'seatCF':>7} {'#macro':>6} "
          f"{'shortfl':>8} {'hits':>14} | reason")
    for d in rows:
        reason = d['end_reason'].split('keys=')[0].strip() if 'keys=' in d['end_reason'] else d['end_reason']
        print(f"{d['b']:>2} {str(d['ok']):>3} "
              f"{str(d['seat_attainable']):>6} {str(d['seat_collfree']):>7} "
              f"{d['n_sub1']:>6} {d['shortfall_cm']:6.1f}cm "
              f"{','.join(d['hit']) if d['hit'] else '-':>14} | {reason}")


if __name__ == "__main__":
    main()
