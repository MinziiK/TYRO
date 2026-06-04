#!/usr/bin/env python3
"""Diagnose why the 6-stage remount cycle stalls at S1 (mount).

Reproduces the exact reset used by ``check_remount_cycle.py`` and then,
instead of stepping the FSM, inspects the S1 baked plan:

  * is the tire actually grasped (constraint / kinematic sync)?
  * what is the S1 end EE target vs the current EE pose?
  * does the baked joint trajectory's FINAL waypoint actually reach the
    target EE pose (FK residual)? where does it stall?
  * where does the tire end up if we drive the arm to the baked final q?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def fk_ee(env):
    pos, quat = env.robot_A.ee_pose()
    return np.asarray(pos, float), np.asarray(quat, float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torque-scale", type=float, default=1.0)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--standoff", type=float, default=None)
    ap.add_argument("--lift", type=float, default=None)
    ap.add_argument("--cutout-back", type=float, default=None,
                    help="cargo_wheel_well_x_range_from_hub min (negative, e.g. -1.1)")
    ap.add_argument("--sweep-only", action="store_true")
    args = ap.parse_args()
    extra = {}
    if args.standoff is not None:
        extra["planner_stage1_approach_standoff"] = float(args.standoff)
    if args.lift is not None:
        extra["planner_stage1_lift"] = float(args.lift)
    if args.cutout_back is not None:
        extra["cargo_wheel_well_x_range_from_hub"] = (float(args.cutout_back), 0.85)
    cfg = make_env_config(
        stage=1, phase=1, scene_layout="fanuc_spacious",
        remount_cycle_enable=True,
        terminate_on="never",
        reverse_curriculum_enable=False,
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        start_pos_easy_prob=1.0,
        attached_spawn_when_easy=True,
        mount_radius_tol=0.18,
        regrip_radius_tol=0.18,
        home_return_radius_tol=0.20,
        rack_return_radius_tol=0.12,
        fanuc_torque_scale=float(args.torque_scale),
        max_steps=1400,
        **extra,
    )
    print(f"[cfg] fanuc_torque_scale = {args.torque_scale}  "
          f"standoff = {getattr(cfg, 'planner_stage1_approach_standoff', None)}  "
          f"lift = {getattr(cfg, 'planner_stage1_lift', None)}")
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    env.set_mount_tol(0.25, np.deg2rad(40.0))

    print("=" * 72)
    print("S1 STALL DIAGNOSTIC")
    print("=" * 72)
    print(f"task_stage              = {int(env.task_stage)}")
    print(f"grasp_constraint        = {env._grasp_constraint}")
    print(f"kinematic_grasp_active  = {getattr(env, '_kinematic_grasp_active', 'n/a')}")
    print(f"T_ee_tire cached        = {env._grasp_t_ee_tire_pos is not None}")
    print(f"planner_traj_steps      = {getattr(cfg, 'planner_traj_steps', None)}")
    print(f"planner_stage1_lift     = {getattr(cfg, 'planner_stage1_lift', None)}")
    print(f"tire_outer_radius (R)   = {cfg.tire_outer_radius}")
    print(f"tire_mount_pos (hub)    = {np.asarray(cfg.tire_mount_pos, float)}")
    print(f"hub_axis_world          = {np.asarray(cfg.hub_axis_world, float)}")

    cur_pos, cur_quat = fk_ee(env)
    print("-" * 72)
    print(f"current EE pos          = {cur_pos}")

    end_pos, end_quat = env._compute_stage_end_ee_pose(1)
    end_pos = np.asarray(end_pos, float)
    end_quat = np.asarray(end_quat, float)
    print(f"S1 target EE pos        = {end_pos}")
    print(f"|target - current| EE   = {np.linalg.norm(end_pos - cur_pos):.4f} m")

    tp0 = np.asarray(env.scene.tire_pose()[0], float)
    hp = np.asarray(env.scene.hub_pose()[0], float)
    print(f"tire pos (reset)        = {tp0}")
    print(f"hub pos                 = {hp}")
    print(f"d(tire,hub) at reset    = {np.linalg.norm(tp0 - hp):.4f} m")

    # Force a replan for the current (S1) stage and inspect the baked plan.
    env._replan_for_current_stage()
    traj_pos = env._traj_pos
    traj_q = env._traj_q
    print("-" * 72)
    print(f"baked traj_pos shape    = {None if traj_pos is None else traj_pos.shape}")
    print(f"baked traj_q   shape    = {None if traj_q is None else traj_q.shape}")
    if traj_pos is not None:
        print(f"nominal final waypoint  = {np.asarray(traj_pos[-1], float)}")
        print(f"|nominal_final - target|= {np.linalg.norm(np.asarray(traj_pos[-1], float) - end_pos):.4f} m")

    if traj_q is None:
        print("No baked joint trajectory (precompute disabled).")
        env.close()
        return 1

    # Drive the arm to each baked waypoint q and measure FK residual to nominal.
    ur = env.robot_A
    if args.sweep_only:
        return _kin_sweep(env, ur, args)
    print("-" * 72)
    print("idx |  FK-EE pos                         | resid-to-nominal(m)")
    residuals = []
    for i in range(traj_q.shape[0]):
        q = traj_q[i]
        for jidx, qv in zip(ur.arm.indices, q):
            p.resetJointState(ur.uid, jidx, targetValue=float(qv),
                              targetVelocity=0.0, physicsClientId=env.client)
        fkp, _ = fk_ee(env)
        nominal = np.asarray(traj_pos[i], float)
        r = float(np.linalg.norm(fkp - nominal))
        residuals.append(r)
        if i % 10 == 0 or i == traj_q.shape[0] - 1:
            print(f"{i:3d} | {np.array2string(fkp, precision=3, floatmode='fixed')} | {r:.4f}")

    residuals = np.asarray(residuals)
    # FK at the final baked q
    qf = traj_q[-1]
    for jidx, qv in zip(ur.arm.indices, qf):
        p.resetJointState(ur.uid, jidx, targetValue=float(qv),
                          targetVelocity=0.0, physicsClientId=env.client)
    fkp_final, _ = fk_ee(env)
    print("-" * 72)
    print(f"FINAL baked FK-EE pos   = {fkp_final}")
    print(f"|final FK - S1 target|  = {np.linalg.norm(fkp_final - end_pos):.4f} m  <== IK reach error at endpoint")
    print(f"max FK-vs-nominal resid = {residuals.max():.4f} m  (mean {residuals.mean():.4f})")
    print(f"argmax residual at idx  = {int(residuals.argmax())} / {len(residuals)-1}")

    # ---- LIVE dynamics probe: step zero action, watch where it freezes ----
    print("=" * 72)
    print("LIVE DYNAMICS PROBE (zero action)")
    print(f"motor forces (scaled)   = {getattr(ur, '_motor_forces', getattr(ur, '_arm_motor_forces', lambda: '?')())}")
    print(f"max_joint_vel           = {getattr(ur, '_max_joint_vel', '?')}")
    print(f"tire mass               = {p.getDynamicsInfo(env.scene.handles.tire, -1, physicsClientId=env.client)[0]}")
    print("-" * 72)
    print("step | idx | EE pos                        | d(tire,hub) | max|qd| | maxTrqMargin")
    env.reset()
    env.set_mount_tol(0.25, np.deg2rad(40.0))
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    N = 140
    for s in range(N):
        env.step(act)
        if s % 20 == 0 or s == N - 1:
            ee = np.asarray(env.robot_A.ee_pose()[0], float)
            tp = np.asarray(env.scene.tire_pose()[0], float)
            hpn = np.asarray(env.scene.hub_pose()[0], float)
            dh = float(np.linalg.norm(tp - hpn))
            js = p.getJointStates(ur.uid, ur.arm.indices, physicsClientId=env.client)
            qd = np.array([j[1] for j in js])
            trq = np.array([abs(j[3]) for j in js])
            forces = np.asarray(getattr(ur, "_motor_forces", [150] * len(trq)), float)
            margin = float(np.min(forces - trq))  # negative => torque-saturated
            print(f"{s:4d} | {int(env.current_traj_step):3d} | "
                  f"{np.array2string(ee, precision=3, floatmode='fixed')} | "
                  f"{dh:9.3f} | {float(np.max(np.abs(qd))):6.3f} | {margin:8.1f}")
    # Contact report at the stall: what is the tire / arm touching?
    print("-" * 72)
    print("CONTACTS at stall (bodyA, linkA, bodyB, linkB, normalForce):")
    tire_uid = env.scene.handles.tire
    names = {ur.uid: "FANUC", tire_uid: "TIRE"}
    for other in {ur.uid: None}:
        pass
    cps = p.getContactPoints(physicsClientId=env.client)
    agg = {}
    for cp in cps:
        a, b = cp[1], cp[2]
        if a not in (ur.uid, tire_uid) and b not in (ur.uid, tire_uid):
            continue
        key = (a, cp[3], b, cp[4])
        agg[key] = agg.get(key, 0.0) + abs(float(cp[9]))
    for (a, la, b, lb), f in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  bodyA={a}({names.get(a,'?')}) lA={la}  bodyB={b}({names.get(b,'?')}) lB={lb}  Fn={f:.0f}")
    print("=" * 72)

    return _kin_sweep(env, ur, args)


def _kin_sweep(env, ur, args) -> int:
    # ---- KINEMATIC SWEEP: force arm through baked q with tire rigidly
    #      attached (reset both each waypoint). Separates "goal infeasible"
    #      from "path collides midway".
    print("KINEMATIC SWEEP of baked S1 path (FAITHFUL: rigid T_ee_tire = JOINT_FIXED carry)")
    env.reset()
    env.set_mount_tol(0.25, np.deg2rad(40.0))
    env._replan_for_current_stage()
    tq = env._traj_q
    tire_uid = env.scene.handles.tire
    t_pos = np.asarray(env._grasp_t_ee_tire_pos, float).tolist()
    t_orn = np.asarray(env._grasp_t_ee_tire_quat, float).tolist()
    truck_bodies = [b for b in range(p.getNumBodies(physicsClientId=env.client))
                    if b not in (ur.uid, tire_uid, 0)]
    first_collide = None
    worst_pen = 0.0
    print("idx | d(tire,hub) | min_pen(tire vs scene) | colliding bodies")
    for i in range(tq.shape[0]):
        for jidx, qv in zip(ur.arm.indices, tq[i]):
            p.resetJointState(ur.uid, jidx, targetValue=float(qv),
                              targetVelocity=0.0, physicsClientId=env.client)
        eep, eeo = env.robot_A.ee_pose()
        ttp_l, tto_l = p.multiplyTransforms(
            np.asarray(eep, float).tolist(), np.asarray(eeo, float).tolist(),
            t_pos, t_orn)
        ttp = np.asarray(ttp_l, float)
        p.resetBasePositionAndOrientation(
            tire_uid, ttp.tolist(), list(tto_l), physicsClientId=env.client)
        hpn = np.asarray(env.scene.hub_pose()[0], float)
        dh = float(np.linalg.norm(ttp - hpn))
        min_pen = 0.0
        hitters = []
        for b in truck_bodies:
            cps = p.getClosestPoints(tire_uid, b, distance=0.0,
                                     physicsClientId=env.client)
            for cp in cps:
                if cp[8] < min_pen:
                    min_pen = cp[8]
                if cp[8] < -0.005 and b not in hitters:
                    hitters.append(b)
        worst_pen = min(worst_pen, min_pen)
        if hitters and first_collide is None:
            first_collide = i
        if i % 10 == 0 or i == tq.shape[0] - 1 or (hitters and i == first_collide):
            print(f"{i:3d} | {dh:9.3f} | {min_pen:9.4f} | {hitters}")
    print(f"first colliding waypoint = {first_collide} / {tq.shape[0]-1}   "
          f"worst penetration = {worst_pen:.4f} m")
    print("=" * 72)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
