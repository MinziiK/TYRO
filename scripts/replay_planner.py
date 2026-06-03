"""Replay the Min-Jerk nominal trajectory only (zero policy residual).

Opens the PyBullet GUI when ``--render`` is set. The env still runs FSM /
physics / kinematic tire lock; only ``action`` is forced to zero so
``final_pos = nominal[idx]`` with no PPO offset.

Examples
--------
    python scripts/replay_planner.py --render --easy-start --episodes 1
    python scripts/replay_planner.py --render --episodes 3 --max-steps 600
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config
from src.env import TyroEnv


def main() -> int:
    ap = argparse.ArgumentParser(description="Planner-only replay (zero action).")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--no-hold", action="store_true",
                    help="Exit immediately after the last episode.")
    ap.add_argument("--easy-start", action="store_true",
                    help="Force easy spawn + attached hot-start every reset.")
    ap.add_argument("--home-start", action="store_true",
                    help="Force HOME spawn (no easy mix).")
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--smooth-alpha", type=float, default=1.0,
        help="Joint-target EMA (1.0=off). Baked planner traj is already smooth.",
    )
    ap.add_argument(
        "--max-step", type=float, default=0.0,
        help="Max per-step joint change rad (0=off). Default 0.",
    )
    ap.add_argument(
        "--pos-gain", type=float, default=1.0,
        help="UR10 PD positionGain. Default 1.0 so replay reaches the hub.",
    )
    ap.add_argument(
        "--vel-gain", type=float, default=1.0,
        help="UR10 PD velocityGain (damping). Default 1.0.",
    )
    ap.add_argument(
        "--no-path", action="store_true",
        help="Don't draw the planner nominal EE paths in the GUI.",
    )
    ap.add_argument(
        "--no-trail", action="store_true",
        help="Don't draw the actual travelled EE trail in the GUI.",
    )
    ap.add_argument(
        "--mount-tol", type=float, default=None,
        help=(
            "Override the mount radius gate (m). Default uses the hard "
            "0.04 m gate, which zero-action replay cannot reach (the "
            "trained policy closes the last ~2 cm). Set e.g. 0.10 to let "
            "the nominal trajectory alone seat the tire and confirm carry "
            "+ insertion smoothness end-to-end."
        ),
    )
    ap.add_argument(
        "--mount-ang-tol-deg", type=float, default=None,
        help="Override the mount axis-angle gate (degrees).",
    )
    args = ap.parse_args()
    if args.easy_start and args.home_start:
        ap.error("Use only one of --easy-start or --home-start.")

    overrides: dict = dict(
        contact_force_terminate_above=0.0,
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        ur10_joint_target_smooth_alpha=float(args.smooth_alpha),
        ur10_joint_max_step_rad=float(args.max_step),
        ur10_position_gain=float(args.pos_gain),
        ur10_velocity_gain=float(args.vel_gain),
        mount_hold_steps=40,
        pin_tire_on_mount=True,
    )
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)

    cfg = make_env_config(stage=args.stage, phase=args.phase, **overrides)
    env = TyroEnv(cfg=cfg, render=args.render, seed=args.seed)
    if args.easy_start:
        env.set_start_pos_easy_prob(0.999)
        print("[planner] start: easy (forced)")
    elif args.home_start:
        env.set_start_pos_easy_prob(0.0)
        print("[planner] start: HOME (forced)")
    else:
        print(
            f"[planner] start: mix easy_prob="
            f"{getattr(cfg, 'start_pos_easy_prob', 0.5)}"
        )

    print(
        f"[planner] use_planner_residual={cfg.use_planner_residual}  "
        f"precompute_joints={cfg.planner_precompute_joint_traj}  "
        f"traj_steps={cfg.planner_traj_steps}  "
        f"max_steps={cfg.max_steps}  "
        f"terminate_on={cfg.terminate_on!r}"
    )
    print(
        f"[planner] smoothing: alpha={cfg.ur10_joint_target_smooth_alpha}  "
        f"max_step={cfg.ur10_joint_max_step_rad} rad  "
        f"PD gains: pos={cfg.ur10_position_gain} vel={cfg.ur10_velocity_gain}"
    )
    print("[planner] action = zeros (nominal EE only)")

    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    step_period = 1.0 / cfg.control_freq_hz
    draw_path = args.render and not args.no_path
    draw_trail = args.render and not args.no_trail
    path_line_ids: list[int] = []
    # Per-stage planner path colours (stage 0..3).
    stage_colors = {
        0: [0.20, 0.60, 1.00],  # blue  — approach / pickup
        1: [1.00, 0.55, 0.00],  # orange — carry to hub
        2: [0.85, 0.20, 0.85],  # magenta — demount
        3: [0.20, 0.85, 0.40],  # green — return to cradle
    }

    def draw_all_stage_paths() -> None:
        """Draw every remaining FSM stage's nominal EE path (colored)."""
        if not draw_path:
            return
        for lid in path_line_ids:
            try:
                p.removeUserDebugItem(lid, physicsClientId=env.client)
            except p.error:
                pass
        path_line_ids.clear()
        trajs = env.compute_all_stage_trajectories()
        total = 0
        for stage in sorted(trajs):
            pts = trajs[stage]
            color = stage_colors.get(stage, [1.0, 1.0, 1.0])
            for i in range(len(pts) - 1):
                lid = p.addUserDebugLine(
                    pts[i].tolist(), pts[i + 1].tolist(),
                    lineColorRGB=color, lineWidth=3.0,
                    physicsClientId=env.client,
                )
                path_line_ids.append(lid)
            total += len(pts)
        print("[planner] nominal paths: stages %s (%d pts)  "
              "blue=approach orange=carry magenta=demount green=return"
              % (sorted(trajs), total))

    def draw_trail_segment(a, b) -> None:
        if not draw_trail:
            return
        p.addUserDebugLine(
            a.tolist(), b.tolist(),
            lineColorRGB=[1.0, 1.0, 0.0], lineWidth=2.0,
            lifeTime=0.0, physicsClientId=env.client,
        )

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        if args.mount_tol is not None or args.mount_ang_tol_deg is not None:
            cur_r, cur_a = env.get_mount_tol()
            new_r = float(args.mount_tol) if args.mount_tol is not None else cur_r
            new_a = (
                np.deg2rad(float(args.mount_ang_tol_deg))
                if args.mount_ang_tol_deg is not None else cur_a
            )
            env.set_mount_tol(new_r, new_a)
            print(
                f"[planner] mount gate override: radius={new_r:.3f} m  "
                f"angle={np.rad2deg(new_a):.1f}°"
            )
        draw_all_stage_paths()
        total_r = 0.0
        steps = 0
        terminated = truncated = False
        info: dict = {}
        prev_ee = np.asarray(env.robot_A.ee_pose()[0], dtype=np.float64)
        ee_jumps: list[float] = []          # per-step EE displacement (m)
        jump_stages: list[int] = []          # FSM stage at each step
        while not (terminated or truncated):
            if args.render and not p.isConnected(env.client):
                print("[planner] GUI closed — stopping replay.")
                break
            t0 = time.time()
            try:
                _, r, terminated, truncated, info = env.step(zero)
            except p.error:
                print("[planner] physics server disconnected — stopping replay.")
                break
            cur_ee = np.asarray(env.robot_A.ee_pose()[0], dtype=np.float64)
            ee_jumps.append(float(np.linalg.norm(cur_ee - prev_ee)))
            jump_stages.append(int(getattr(env, "task_stage", -1)))
            draw_trail_segment(prev_ee, cur_ee)
            prev_ee = cur_ee
            total_r += float(r)
            steps += 1
            if args.render:
                dt = time.time() - t0
                if dt < step_period:
                    time.sleep(step_period - dt)
        print(
            f"  ep {ep:3d}  r={total_r:+8.2f}  len={steps:4d}  "
            f"success={info.get('is_success', False)}  "
            f"termination={info.get('termination', '-')}"
        )
        if ee_jumps:
            j = np.asarray(ee_jumps, dtype=np.float64)
            n_big = int((j > 0.15).sum())
            print(
                f"      EE jump (cm): max={j.max()*100:6.2f}  "
                f"mean={j.mean()*100:5.2f}  >15cm={n_big}"
            )
            stages = np.asarray(jump_stages, dtype=np.int64)
            for st in sorted(set(jump_stages)):
                if st < 0:
                    continue
                m = stages == st
                js = j[m]
                print(
                    f"        stage {st}: steps={int(m.sum()):4d}  "
                    f"max={js.max()*100:6.2f}cm  mean={js.mean()*100:5.2f}cm  "
                    f">15cm={int((js > 0.15).sum())}"
                )
        if args.render and not p.isConnected(env.client):
            break

    if args.render and not args.no_hold:
        print("[planner] GUI held open. Close window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
