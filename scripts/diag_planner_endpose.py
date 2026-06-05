#!/usr/bin/env python3
"""Isolate planner-target error from dynamics stall.

Builds the trainer env (stage3/phase1/fanuc_spacious, attached easy), then
drives the FANUC arm DIRECTLY to the baked stage-1 final joint config
``env._traj_q[-1]`` (kinematic reset + a few settle steps holding that
config). Measures the resulting tire mount gate (d_mount, theta) at the
*intended* end pose, removing torque/collision stalling from the picture.

If theta is large here, the planner end-orientation is wrong (a planning
bug). If theta is small here but large in the dynamic roll, the arm is
stalling before reaching the target (a dynamics/collision problem).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _gate(env, mount_target):
    tp = np.asarray(env.scene.tire_pose()[0], float)
    dm = float(np.linalg.norm(tp - mount_target))
    th = float(np.arccos(np.clip(
        np.dot(env.scene.tire_axis(), env.scene.hub_axis()), -1.0, 1.0)))
    return dm, np.rad2deg(th)


def main() -> int:
    cfg = make_env_config(
        stage=3, phase=1, scene_layout="fanuc_spacious",
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        start_pos_easy_prob=0.9,
        attached_spawn_when_easy=True,
        terminate_on="never",
    )
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    env.set_mount_tol(float(cfg.mount_radius_tol_soft),
                      float(cfg.mount_angle_tol_soft_rad))

    mount_target = np.asarray(cfg.tire_mount_pos, float)
    traj_q = getattr(env, "_traj_q", None)
    traj_quat = getattr(env, "_traj_quat", None)
    if traj_q is None:
        print("NO baked _traj_q (planner produced no joint trajectory).")
        env.close()
        return 1

    arm = env.robot_A.arm
    n = traj_q.shape[0]
    print(f"baked traj: {n} waypoints; q[-1]={np.round(traj_q[-1], 3)}")

    # Planned EE waypoint path vs IK-achieved (FK of baked joints).
    traj_pos = getattr(env, "_traj_pos", None)
    R = float(cfg.tire_outer_radius)
    intended_end_ee = mount_target + np.array([0.0, 0.0, -R])
    print(f"  tire_outer_radius R={R:.3f}  mount_target={np.round(mount_target,3)}")
    print(f"  intended stage-1 end EE pos (mount-(0,0,R))={np.round(intended_end_ee,3)}")
    if traj_pos is not None:
        print(f"  PLANNED traj_pos[-1]={np.round(traj_pos[-1],3)}  "
              f"traj_pos[-3:]=\n{np.round(traj_pos[-3:],3)}")
    # Per-waypoint IK error: FK(baked q) vs planned EE pos.
    if traj_pos is not None:
        errs = []
        for k in range(n):
            for ji, jq in zip(arm.indices, traj_q[k]):
                p.resetJointState(env.robot_A.uid, ji, float(jq), 0.0,
                                  physicsClientId=env.client)
            fk_pos, _ = env.robot_A.ee_pose()
            errs.append(float(np.linalg.norm(np.asarray(fk_pos) - traj_pos[k])))
        errs = np.asarray(errs)
        print(f"  IK pos err along baked traj: max={errs.max():.3f} "
              f"mean={errs.mean():.3f} last={errs[-1]:.3f} "
              f"first_bad_idx(>0.05)={int(np.argmax(errs>0.05)) if (errs>0.05).any() else -1}")

    dm0, th0 = _gate(env, mount_target)
    print(f"at reset (stage1 attached): d_mount={dm0:.3f} theta={th0:.1f}deg")

    cli = env.client
    # Kinematically place arm at the FINAL baked waypoint, then hold it with
    # position control for a short settle so the JOINT_FIXED grasp drags the
    # tire to the intended seated pose.
    qf = np.asarray(traj_q[-1], float)
    for ji, jq in zip(arm.indices, qf):
        p.resetJointState(env.robot_A.uid, ji, float(jq), 0.0,
                          physicsClientId=cli)
    for _ in range(120):
        env.robot_A.drive_arm_targets(qf)
        p.stepSimulation(physicsClientId=cli)
        env._sync_grasped_tire_upright() if hasattr(
            env, "_sync_grasped_tire_upright") else None

    # Contact points after settling: who is the tire/arm jamming against?
    cps = p.getContactPoints(physicsClientId=cli)
    by_pair = {}
    h = env.handles
    name = {h.tire: "tire", env.robot_A.uid: "fanuc"}
    for cp in cps:
        a, b = cp[1], cp[2]
        f = cp[9]
        if f <= 1.0:
            continue
        key = tuple(sorted((a, b)))
        by_pair[key] = by_pair.get(key, 0.0) + f
    print("  contacts (>1N) after settle [bodyA,bodyB -> totalNormalForce]:")
    for (a, b), f in sorted(by_pair.items(), key=lambda kv: -kv[1])[:8]:
        na = name.get(a, f"body{a}")
        nb = name.get(b, f"body{b}")
        print(f"    {na}({a}) <-> {nb}({b}) : {f:.0f} N")

    dm, th = _gate(env, mount_target)
    ee_pos, ee_quat = env.robot_A.ee_pose()
    print(f"AT BAKED FINAL JOINT CONFIG (kinematic, no traj stall):")
    print(f"  tire d_mount={dm:.3f}  theta={th:.1f}deg")
    print(f"  GATE: mtol={cfg.mount_radius_tol_soft:.2f}m "
          f"atol={np.rad2deg(cfg.mount_angle_tol_soft_rad):.0f}deg "
          f"-> {'PASS' if (dm < cfg.mount_radius_tol_soft and np.deg2rad(th) < cfg.mount_angle_tol_soft_rad) else 'FAIL'}")
    print(f"  EE pos={np.round(ee_pos,3)} mount_target={np.round(mount_target,3)}")
    if traj_quat is not None:
        print(f"  baked final EE quat (target) ={np.round(traj_quat[-1],3)}")
        print(f"  achieved EE quat             ={np.round(np.asarray(ee_quat),3)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
