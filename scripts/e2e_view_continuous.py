#!/usr/bin/env python3
"""Seamless single-window E2E demo: A mount -> B nut in ONE PyBullet session.

Unlike ``e2e_eval.py --render`` (which opens an A window, closes it, then opens
a separate B window — a jarring window close/reopen at the handoff), this viewer
keeps a SINGLE TyroEnv / PyBullet GUI connection alive the whole time:

  1. Run Robot A (mount cfg) until the tire is mounted on the (offset) hub.
  2. Switch the SAME env to the nut cfg and ``reset()`` — the GL window persists
     (same client, no disconnect). The handoff reset is wrapped in the env's
     render freeze so the scene rebuild does not flicker on screen. The tire is
     re-seated on the SAME hub and A is frozen holding it, so visually the tire
     never leaves the hub.
  3. Run Robot B (nut cfg) until all 10 bolts are fastened.

Then loop. The hub XY offset is shared by A and B within an iteration so the
two robots act on the same scenario.

Requires DISPLAY (e.g. ``DISPLAY=:2``).

Example
-------
    DISPLAY=:2 python scripts/e2e_view_continuous.py \
        --hub-x-cm 2.74 --hub-y-cm -0.61 --loop
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402
from scripts.e2e_eval import _nut_overrides_v24, _resolve_model_path  # noqa: E402

# Fixed ms between interpolated render frames (visual smoothness only).
_FRAME_MS = 8.0


def _ensure_panels_off(env) -> None:
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client)
        p.configureDebugVisualizer(
            p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=env.client)
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=env.client)
        p.configureDebugVisualizer(
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=env.client)
        p.configureDebugVisualizer(
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
            physicsClientId=env.client)
    except p.error:
        pass


def _viewer_frame(env) -> None:
    time.sleep(_FRAME_MS / 1000.0)


def _snapshot_handoff_pose(env):
    """Capture arm + tire pose at the end of the mount phase."""
    qA = np.asarray(env.robot_A.joint_state()[0], dtype=np.float64).copy()
    qB = np.asarray(env.robot_B.joint_state()[0], dtype=np.float64).copy()
    try:
        tp, to = env.scene.tire_pose()
        tire = (np.asarray(tp, dtype=np.float64), np.asarray(to, dtype=np.float64))
    except Exception:
        tire = None
    return qA, qB, tire


def _apply_handoff_pose(env, qA, qB, tire_pose=None) -> None:
    a_idx = env.robot_A.arm.indices
    b_idx = env.robot_B.arm.indices
    for idx, q in zip(a_idx, qA):
        p.resetJointState(env.robot_A.uid, int(idx), float(q), 0.0,
                          physicsClientId=env.client)
    for idx, q in zip(b_idx, qB):
        p.resetJointState(env.robot_B.uid, int(idx), float(q), 0.0,
                          physicsClientId=env.client)
    env.robot_A.drive_arm_targets(np.asarray(qA, dtype=np.float64))
    env.robot_B.drive_arm_targets(np.asarray(qB, dtype=np.float64))
    if tire_pose is not None:
        tp, to = tire_pose
        p.resetBasePositionAndOrientation(
            env.handles.tire, list(tp), list(to), physicsClientId=env.client)


def _handoff_transition(env, qA0, qB0, tire0, *, frames: int) -> None:
    """Visually blend from the mount end-pose into the nut-task reset pose."""
    qA1 = np.asarray(env.robot_A.joint_state()[0], dtype=np.float64)
    qB1 = np.asarray(env.robot_B.joint_state()[0], dtype=np.float64)
    tire1 = None
    try:
        tp, to = env.scene.tire_pose()
        tire1 = (np.asarray(tp, dtype=np.float64), np.asarray(to, dtype=np.float64))
    except Exception:
        pass
    K = max(1, int(frames))
    for k in range(1, K + 1):
        a = k / float(K)
        a = a * a * (3.0 - 2.0 * a)
        qA = (1.0 - a) * qA0 + a * qA1
        qB = (1.0 - a) * qB0 + a * qB1
        tire_pose = None
        if tire0 is not None and tire1 is not None:
            tp = (1.0 - a) * tire0[0] + a * tire1[0]
            to = (1.0 - a) * tire0[1] + a * tire1[1]
            n = float(np.linalg.norm(to))
            if n > 1e-12:
                to = to / n
            tire_pose = (tp, to)
        _apply_handoff_pose(env, qA, qB, tire_pose)
        _viewer_frame(env)
    if hasattr(env, "_nut_frozen_qA"):
        env._nut_frozen_qA = qA1.copy()


def _rebind_gym_spaces(env, cfg) -> None:
    from gymnasium import spaces
    env.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(cfg.action.dim,), dtype=np.float32)
    env.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(cfg.obs.dim,), dtype=np.float32)


def _hold_view(env, seconds: float) -> None:
    n = max(1, int(round(float(seconds) * 1000.0 / _FRAME_MS)))
    for _ in range(n):
        _viewer_frame(env)


def _render_lerp(env, qA0, qA1, qB0, qB1, tire_pose, carry, *, substeps):
    """Visually interpolate both arms from q0->q1 over ``substeps`` frames."""
    a_idx = env.robot_A.arm.indices
    b_idx = env.robot_B.arm.indices
    K = max(1, int(substeps))
    for k in range(1, K + 1):
        a = k / float(K)
        a = a * a * (3.0 - 2.0 * a)
        qA = (1.0 - a) * qA0 + a * qA1
        qB = (1.0 - a) * qB0 + a * qB1
        for idx, q in zip(a_idx, qA):
            p.resetJointState(env.robot_A.uid, int(idx), float(q), 0.0,
                              physicsClientId=env.client)
        for idx, q in zip(b_idx, qB):
            p.resetJointState(env.robot_B.uid, int(idx), float(q), 0.0,
                              physicsClientId=env.client)
        if carry:
            env._replace_grasped_tire_rigid()
        elif tire_pose is not None:
            p.resetBasePositionAndOrientation(
                env.handles.tire, list(tire_pose[0]), list(tire_pose[1]),
                physicsClientId=env.client)
        _viewer_frame(env)


def _run(env, model, obs, *, deterministic=True, substeps=4):
    steps = 0
    terminated = truncated = False
    last_info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        qA0 = np.asarray(env.robot_A.joint_state()[0], dtype=np.float64).copy()
        qB0 = np.asarray(env.robot_B.joint_state()[0], dtype=np.float64).copy()
        ee0 = np.asarray(env.robot_A.ee_pose()[0], dtype=np.float64)
        with env._render_frozen():
            obs, _r, terminated, truncated, last_info = env.step(action)
        qA1 = np.asarray(env.robot_A.joint_state()[0], dtype=np.float64).copy()
        qB1 = np.asarray(env.robot_B.joint_state()[0], dtype=np.float64).copy()
        ee1 = np.asarray(env.robot_A.ee_pose()[0], dtype=np.float64)
        carry = bool(int(env.task_stage) == 1
                     and not env._mount_seat_active
                     and env._is_tire_grasped())
        try:
            tp, to = env.scene.tire_pose()
            tire_pose = (np.asarray(tp), np.asarray(to))
        except Exception:
            tire_pose = None
        # Skip backward EE lerp during early carry — otherwise a planner/policy
        # snap away from the tire reads as the arm "resetting" mid-pickup.
        tire_com = tire_pose[0] if tire_pose is not None else None
        backward = (
            carry
            and tire_com is not None
            and float(np.linalg.norm(ee1 - tire_com))
            > float(np.linalg.norm(ee0 - tire_com)) + 0.012
        )
        if backward:
            _viewer_frame(env)
        else:
            _render_lerp(env, qA0, qA1, qB0, qB1, tire_pose, carry,
                         substeps=substeps)
        steps += 1
    return {
        "success": bool(last_info.get("is_success", False)),
        "termination": str(last_info.get("termination", "unknown")),
        "steps": int(steps),
        "n_fastened": int(last_info.get(
            "n_fastened",
            (last_info.get("reward_terms") or {}).get("n_fastened", 0),
        )),
    }


def main() -> int:
    global _FRAME_MS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", default="runs/phase1_mount_v3_dr/final.zip")
    ap.add_argument(
        "--model-b",
        default="runs/nut_fastening_v24_dr_stageB3/ckpts/ppo_1749440_steps.zip",
    )
    ap.add_argument("--hub-x-cm", type=float, default=2.74)
    ap.add_argument("--hub-y-cm", type=float, default=-0.61)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--a-max-steps", type=int, default=2000)
    ap.add_argument("--b-max-steps", type=int, default=2500)
    ap.add_argument("--mount-radius-tol", type=float, default=0.55)
    ap.add_argument("--mix-easy-prob", type=float, default=0.0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--frame-ms", type=float, default=8.0,
                    help="Ms pause per render frame (smoothness only).")
    ap.add_argument("--smooth-frames", type=int, default=4,
                    help="Interpolation frames per policy step.")
    ap.add_argument("--handoff-frames", type=int, default=36,
                    help="Blend frames for the A→B mount→nut handoff.")
    ap.add_argument("--handoff-hold-s", type=float, default=0.6,
                    help="Pause (seconds) on the mounted pose before handoff.")
    args = ap.parse_args()
    _FRAME_MS = float(args.frame_ms)

    off = np.array([args.hub_x_cm / 100.0, args.hub_y_cm / 100.0], dtype=np.float64)
    det = not args.stochastic

    model_a = PPO.load(_resolve_model_path(args.model_a), device="cpu")
    model_b = PPO.load(_resolve_model_path(args.model_b), device="cpu")
    print(f"[view] A obs={model_a.observation_space.shape[0]} "
          f"act={model_a.action_space.shape[0]}  "
          f"B obs={model_b.observation_space.shape[0]} "
          f"act={model_b.action_space.shape[0]}")
    print(f"[view] hub offset = ({args.hub_x_cm:+.2f}, {args.hub_y_cm:+.2f}) cm "
          f"|{np.linalg.norm(off)*100:.2f}| cm")

    mount_overrides = dict(
        render=True, scene_layout="fanuc_spacious", terminate_on="mount",
        max_steps=int(args.a_max_steps),
        USE_DOMAIN_RANDOMIZATION=True,
        RANDOM_POSITION_RANGE=float(np.linalg.norm(off)) + 1e-6,
        DR_CARGO_ENABLE=False, planner_pos_offset_scale=0.06,
        mount_radius_tol=float(args.mount_radius_tol),
        mount_seat_glide_steps=10, contact_force_terminate_above=0.0,
        start_pos_curriculum_enable=True, include_hub_guide_obs=True,
        carry_tire_rigid_sync=True,
        attached_spawn_when_easy=False,
    )
    nut_overrides = _nut_overrides_v24(
        render=True, dr_range_m=float(np.linalg.norm(off)) + 1e-6,
        max_steps=int(args.b_max_steps),
    )
    cfg_a = make_env_config(stage=3, phase=1, **mount_overrides)
    cfg_b = make_env_config(stage=3, phase=1, **nut_overrides)

    env = TyroEnv(cfg=cfg_a, render=True, seed=args.seed)
    env.set_start_pos_easy_prob(float(args.mix_easy_prob))
    _ensure_panels_off(env)

    print(f"[view] frame={_FRAME_MS:.0f}ms  smooth={args.smooth_frames} frames/step")

    it = 0
    try:
        while True:
            it += 1
            print(f"\n[view] === iteration {it} ===")

            env.cfg = cfg_a
            env.set_dr_hub_xy_offset(off)
            _ensure_panels_off(env)
            env.set_start_pos_easy_prob(float(args.mix_easy_prob))
            obs, _ = env.reset(seed=args.seed)
            env.set_start_pos_easy_prob(float(args.mix_easy_prob))
            env._draw_world_axes()

            a = _run(env, model_a, obs, deterministic=det,
                     substeps=int(args.smooth_frames))
            print(f"[view] A: success={a['success']} steps={a['steps']} "
                  f"term={a['termination']}")

            qA0, qB0, tire0 = _snapshot_handoff_pose(env)
            _hold_view(env, float(args.handoff_hold_s))

            env.cfg = cfg_b
            env.set_dr_hub_xy_offset(off)
            _rebind_gym_spaces(env, cfg_b)
            with env._render_frozen():
                obs, _ = env.reset(seed=args.seed + 1)
            _handoff_transition(
                env, qA0, qB0, tire0, frames=int(args.handoff_frames),
            )
            env._draw_world_axes()
            _ensure_panels_off(env)

            b = _run(env, model_b, obs, deterministic=det,
                     substeps=int(args.smooth_frames))
            print(f"[view] B: success={b['success']} steps={b['steps']} "
                  f"n_fastened={b['n_fastened']}/10 term={b['termination']}")
            print(f"[view] E2E: {'PASS' if (a['success'] and b['success']) else 'FAIL'}")

            if not args.loop:
                break
            _rebind_gym_spaces(env, cfg_a)
            for _ in range(10):
                _viewer_frame(env)

        print("[view] done — holding window (Ctrl-C to exit).")
        while True:
            _viewer_frame(env)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
