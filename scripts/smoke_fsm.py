"""Phase-1 FSM smoke test: reset + a few zero-action steps + diag.

Used to verify the new FSM env wiring (world pin / vertical lock / Stage 0
approach reward / termination conditions) without launching PPO.
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
    cfg = make_env_config(
        stage=3, phase=1, contact_force_terminate_above=0.0,
    )
    print(f"tire_pickup_pos     = {cfg.tire_pickup_pos}")
    print(f"tire_mount_pos      = {cfg.tire_mount_pos}")
    print(f"hub_axis_world      = {cfg.hub_axis_world}")
    print(f"vertical_tol (deg)  = {math.degrees(cfg.vertical_tol_rad):.2f}")
    print(f"approach_radius_tol = {cfg.approach_radius_tol} m")
    print(f"mount_radius_tol    = {cfg.mount_radius_tol} m")
    print(f"return_radius_tol   = {cfg.return_radius_tol} m")
    print(f"landing_speed_max   = {cfg.landing_speed_max} m/s")

    env = TyroEnv(cfg=cfg, render=False, seed=42)
    obs, info = env.reset(seed=42)
    print(f"obs.shape           = {obs.shape}")
    print(f"reset info keys     = {list(info.keys())}")

    tire_pos, tire_orn = env.scene.tire_pose()
    ee_pos, _ = env.robot_A.ee_pose()
    hub_pos, _ = env.scene.hub_pose()
    print()
    print(f"init tire COM       = {tuple(round(float(v), 4) for v in tire_pos)}")
    print(
        f"init tire RPY (deg) = "
        f"{tuple(round(math.degrees(a), 2) for a in p.getEulerFromQuaternion(tire_orn))}"
    )
    print(f"init UR10 EE        = {tuple(round(float(v), 4) for v in ee_pos)}")
    print(f"init hub            = {tuple(round(float(v), 4) for v in hub_pos)}")
    print(
        f"task_stage          = {env.task_stage}   "
        f"world_pin={env._world_pin}   grasp={env._grasp_constraint}"
    )

    R = float(cfg.tire_outer_radius)
    grasp_target = np.asarray(tire_pos, dtype=np.float64) + np.array([0.0, 0.0, -R])
    d_initial = float(np.linalg.norm(np.asarray(ee_pos) - grasp_target))
    print(f"EE→grasp_target dist = {d_initial * 100:.1f} cm")
    print(f"grasp_target world   = {tuple(round(float(v), 4) for v in grasp_target)}")

    ur10_base = np.asarray(cfg.robot_A_base_pos, dtype=np.float64)
    print(
        f"UR10 base -> grasp_target = "
        f"{float(np.linalg.norm(ur10_base - grasp_target)) * 100:.1f} cm "
        f"(UR10 max reach ~130 cm)"
    )
    print(f"freeze_robot_b      = {cfg.freeze_robot_b}")
    qB_home, _ = env.robot_B.joint_state()
    print(f"Panda joints @ HOME = {[round(float(q), 3) for q in qB_home]}")

    print()
    print("-- body uid map --")
    print(f"  plane       = {env.handles.plane}")
    print(f"  truck       = {env.handles.truck_uid}")
    print(f"  vehicle     = {env.handles.vehicle}")
    print(f"  tire        = {env.handles.tire}")
    print(f"  robot_A     = {env.robot_A.uid}")
    print(f"  robot_B     = {env.robot_B.uid}")
    n_bodies = p.getNumBodies(physicsClientId=env.client)
    print(f"  total bodies in sim = {n_bodies}")
    for bi in range(n_bodies):
        try:
            info = p.getBodyInfo(bi, physicsClientId=env.client)
            base_name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        except Exception:
            base_name = "?"
        try:
            pos, _ = p.getBasePositionAndOrientation(bi, physicsClientId=env.client)
            pos_str = f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
        except Exception:
            pos_str = "?"
        n_links = p.getNumJoints(bi, physicsClientId=env.client)
        print(f"  body {bi}: name='{base_name}'  n_links={n_links}  base={pos_str}")

    print()
    print("-- contact diagnostics on first physics step --")
    cps = p.getContactPoints(physicsClientId=env.client)
    if not cps:
        print("  (no contacts yet)")
    else:
        for cp in cps[:20]:
            bodyA, bodyB, linkA, linkB = cp[1], cp[2], cp[3], cp[4]
            normal_force = float(cp[9]) if len(cp) > 9 else 0.0
            print(
                f"  bodyA={bodyA} link={linkA}  bodyB={bodyB} link={linkB}  "
                f"normal_force={normal_force:+.2f}N"
            )

    print()
    print("-- 5 zero-action steps --")
    total_r = 0.0
    for i in range(5):
        obs, r, term, trunc, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        total_r += float(r)
        tp, _ = env.scene.tire_pose()
        print(
            f"step {i}: r={r:+.4f} stage={info['task_stage']} "
            f"cf_max={info['contact_force_max']:.1f}N "
            f"vertical_err_deg={math.degrees(info['tire_vertical_err_rad']):.2f} "
            f"tire_z={tp[2]:.4f} term={info.get('termination')}"
        )
        if term or trunc:
            break
    print(f"cumulative reward = {total_r:+.4f}")

    print()
    print("-- random-policy rollout over 5 episodes --")
    rng = np.random.default_rng(7)
    successes = 0
    term_counts: dict[str, int] = {}
    stage_steps: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for ep in range(5):
        obs, info = env.reset(seed=42 + ep)
        ep_r = 0.0
        ep_steps = 0
        last_stage = 0
        max_stage = 0
        for _ in range(200):
            action = rng.uniform(
                -0.3, 0.3, size=env.action_space.shape
            ).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)
            ep_r += float(r)
            ep_steps += 1
            stage = int(info["task_stage"])
            stage_steps[stage] = stage_steps.get(stage, 0) + 1
            max_stage = max(max_stage, stage)
            last_stage = stage
            if term or trunc:
                tk = info.get("termination", "unknown")
                term_counts[tk] = term_counts.get(tk, 0) + 1
                if info.get("is_success"):
                    successes += 1
                print(
                    f"  ep {ep} → ended after {ep_steps} steps  "
                    f"max_stage={max_stage}  return={ep_r:+8.2f}  "
                    f"termination={tk}"
                )
                break
        else:
            term_counts["timeout"] = term_counts.get("timeout", 0) + 1
            print(
                f"  ep {ep} → ran full 200 steps  "
                f"max_stage={max_stage}  return={ep_r:+8.2f}  "
                f"final stage={last_stage}"
            )
    print(f"\nsuccess count: {successes}/5")
    print(f"termination histogram: {term_counts}")
    print(f"step time per stage: {stage_steps}")

    qB_after, _ = env.robot_B.joint_state()
    drift = float(np.max(np.abs(qB_after - qB_home)))
    print(
        f"\nPanda joint drift after random rollouts: "
        f"max |dq| = {math.degrees(drift):.3f} deg "
        f"({'FROZEN' if drift < math.radians(0.5) else 'MOVING'})"
    )

    env.close()
    print("\nOK: env smoke test ran to completion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
