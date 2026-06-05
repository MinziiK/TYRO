#!/usr/bin/env python3
"""Diagnose why Phase-A (stage 3 / phase 1, NO remount) mount training
never fires a success. Reproduces the trainer's env config and rolls a
few episodes with ZERO action to inspect the attached-easy carry path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def main() -> int:
    # Mirror the trainer: stage 3, phase 1, fanuc_spacious, NO remount cycle.
    cfg = make_env_config(
        stage=3, phase=1, scene_layout="fanuc_spacious",
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        start_pos_easy_prob=0.9,
        attached_spawn_when_easy=True,
        terminate_on="mount",
    )
    print("=" * 72)
    print("PHASE-A MOUNT DIAGNOSTIC (stage3/phase1, no remount)")
    print(f"  attached_spawn_when_easy = {cfg.attached_spawn_when_easy}")
    print(f"  start_pos_easy_prob      = {cfg.start_pos_easy_prob}")
    print(f"  mount_radius_tol         = {cfg.mount_radius_tol}")
    print(f"  mount_radius_tol_soft    = {getattr(cfg, 'mount_radius_tol_soft', None)}")
    print(f"  planner_traj_steps       = {getattr(cfg, 'planner_traj_steps', None)}")
    print(f"  planner_stage1_standoff  = {getattr(cfg, 'planner_stage1_approach_standoff', None)}")
    print(f"  max_joint_vel(fanuc)     = {getattr(cfg, 'fanuc_motor_max_velocity_rad_s', None)}")
    print(f"  waypoint_gate_enable     = {getattr(cfg, 'planner_waypoint_gate_enable', None)}")
    print(f"  max_steps                = {cfg.max_steps}")
    print("=" * 72)

    env = TyroEnv(cfg=cfg, render=False, seed=0)
    # Force easy (attached) spawn so we isolate the carry/mount path.
    env.set_start_pos_easy_prob(0.999)
    # Soft mount gate as the curriculum starts (0.30 m / 30 deg).
    for ep in range(1):
        env.reset()
        env.set_mount_tol(
            float(cfg.mount_radius_tol_soft),
            float(cfg.mount_angle_tol_soft_rad),
        )
        act = np.zeros(env.action_space.shape, dtype=np.float32)
        ts0 = int(env.task_stage)
        grasped0 = env._is_tire_grasped()
        tp = np.asarray(env.scene.tire_pose()[0], float)
        hp = np.asarray(env.scene.hub_pose()[0], float)
        d0 = float(np.linalg.norm(tp - hp))
        min_d = d0
        success = False
        last_stage = ts0
        stage_seq = [ts0]
        import pybullet as p
        mount_target = np.asarray(cfg.tire_mount_pos, float)
        mtol = float(getattr(env, "_mount_radius_tol", cfg.mount_radius_tol))
        atol = float(getattr(env, "_mount_angle_tol", cfg.reward.delta_A))

        def _gate():
            tp_ = np.asarray(env.scene.tire_pose()[0], float)
            dm = float(np.linalg.norm(tp_ - mount_target))
            th = float(np.arccos(np.clip(
                np.dot(env.scene.tire_axis(), env.scene.hub_axis()),
                -1.0, 1.0)))
            return dm, th

        dm0, th0 = _gate()
        min_dm = dm0
        min_th = th0
        snap = {}
        traj_n = int(env._traj_pos.shape[0]) if env._traj_pos is not None else -1
        idx_snap = {}
        cap = int(getattr(cfg, "max_steps", 600))
        for i in range(cap):
            _, _, term, trunc, info = env.step(act)
            for ms in (100, 200, 300, 400, 500, 600):
                if i + 1 == ms:
                    idx_snap[ms] = int(env.current_traj_step)
            dm, th = _gate()
            min_dm = min(min_dm, dm)
            min_th = min(min_th, th)
            st = int(info.get("task_stage", env.task_stage))
            if st != last_stage:
                stage_seq.append(st)
                last_stage = st
            if info.get("is_success"):
                success = True
            for ms in (100, 200, 300, 400, 600):
                if i + 1 == ms:
                    snap[ms] = (round(dm, 3), round(np.rad2deg(th), 1))
            if term or trunc:
                break
        print(f"ep{ep}: start_stage={ts0} grasped0={grasped0} GATE mtol={mtol:.2f} "
              f"atol={np.rad2deg(atol):.0f}deg | d_mount0={dm0:.3f} "
              f"min_d_mount={min_dm:.3f} min_theta={np.rad2deg(min_th):.1f}deg | "
              f"(d_mount,thetaDeg)@steps={snap} end_stage={last_stage} "
              f"stages={stage_seq} success={success} steps={i+1}")
        print(f"      TRAJ INDEX: reached {int(env.current_traj_step)}/{traj_n} "
              f"waypoints (gate={getattr(cfg,'planner_waypoint_gate_enable',None)} "
              f"max_stall={getattr(cfg,'planner_waypoint_max_stall',None)}) "
              f"idx@steps={idx_snap}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
