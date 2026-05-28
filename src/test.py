"""Smoke test: reset + random rollout to verify env wiring.

Run from repo root:
    python -m src.test                       # headless, fast
    python -m src.test --render              # GUI, real-time playback, holds open
    python -m src.test --render --no-hold    # GUI, exits when rollout ends
    python -m src.test --render --action-scale 0.0   # static scene inspection
    python -m src.test --render --random-max         # full action_space.sample() — motor-control sanity

Checks:
  * Observation shape == config default (89) and dtype == float32
  * Action shape == 13
  * N random steps complete without exceptions
  * env.check_env passes the Gymnasium API contract (with --check)
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pybullet as p

from src.config import EnvConfig, make_env_config
from src.env import TyroEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--steps", type=int, default=None,
                    help="Default: 50 headless / 200 with --render.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Reward curriculum stage (default 1 for Phase-1 smoke).")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                    help="Domain-randomization phase.")
    ap.add_argument("--check", action="store_true",
                    help="Also run gymnasium.utils.env_checker.check_env (slow).")
    ap.add_argument("--action-scale", type=float, default=0.3,
                    help="Magnitude of random actions in [-1,1]. 0 = static scene.")
    ap.add_argument(
        "--random-max",
        action="store_true",
        help=(
            "Motor-control sanity check: ignore --action-scale and inject the "
            "FULL ``env.action_space.sample()`` every step (joints should swing "
            "visibly). If the arms still freeze, the issue is in the PyBullet "
            "motor mode / gains, not the policy."
        ),
    )
    ap.add_argument("--no-hold", action="store_true",
                    help="Don't wait for user to close GUI window (default holds).")
    args = ap.parse_args()

    if args.steps is None:
        args.steps = 200 if args.render else 50

    cfg = make_env_config(stage=args.stage, phase=args.phase, render=args.render)
    env = TyroEnv(cfg=cfg, render=args.render, seed=args.seed)

    print(f"[smoke] action_space={env.action_space}")
    print(f"[smoke] observation_space={env.observation_space}")

    obs, info = env.reset(seed=args.seed)
    assert obs.shape == (cfg.obs.dim,), (
        f"obs shape {obs.shape} != ({cfg.obs.dim},)"
    )
    assert obs.dtype == np.float32, f"obs dtype {obs.dtype} != float32"
    print(f"[smoke] reset OK -- obs[:6]={obs[:6]}, target_bolt={info['target_bolt_idx']}")

    rng = np.random.default_rng(args.seed)
    # Seed the action_space RNG so --random-max is reproducible across runs.
    if args.random_max:
        env.action_space.seed(args.seed)
        print(
            "[smoke] --random-max ON: injecting env.action_space.sample() every "
            "step (joints should swing visibly). If they don't, inspect "
            "`p.setJointMotorControl2` mode/gains in src/env/robots.py."
        )

    # EE pose deltas across the episode prove the motor commands actually
    # propagate to the simulator; "0.000 m" travelled with --random-max means
    # PyBullet is silently ignoring the control commands.
    eeA0, _ = env.robot_A.ee_pose()
    eeB0, _ = env.robot_B.ee_pose()
    eeA_prev = np.asarray(eeA0, dtype=np.float64)
    eeB_prev = np.asarray(eeB0, dtype=np.float64)
    eeA_travel = 0.0
    eeB_travel = 0.0

    total_reward = 0.0
    breakdown_sum: dict = {}
    step_period = 1.0 / cfg.control_freq_hz   # real-time playback when rendering
    t0 = time.time()
    actual_steps = 0
    for t in range(args.steps):
        loop_start = time.time()
        if args.random_max:
            action = env.action_space.sample().astype(np.float32)
        elif args.action_scale > 0:
            action = rng.uniform(-args.action_scale, args.action_scale,
                                 size=(13,)).astype(np.float32)
        else:
            action = np.zeros(13, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        actual_steps = t + 1
        for k, v in info["reward_terms"].items():
            if isinstance(v, (int, float)):
                breakdown_sum[k] = breakdown_sum.get(k, 0.0) + float(v)

        eeA_now = np.asarray(env.robot_A.ee_pose()[0], dtype=np.float64)
        eeB_now = np.asarray(env.robot_B.ee_pose()[0], dtype=np.float64)
        eeA_travel += float(np.linalg.norm(eeA_now - eeA_prev))
        eeB_travel += float(np.linalg.norm(eeB_now - eeB_prev))
        eeA_prev, eeB_prev = eeA_now, eeB_now

        if t % 10 == 0:
            terms = info["reward_terms"]
            print(f"  step {t:3d}  r={reward:+.3f}  d_A={terms['d_A']:.3f}  "
                  f"theta_A={terms['theta_A']:.3f}  d_B={terms['d_B']:.3f}  "
                  f"theta_B={terms['theta_B']:.3f}  "
                  f"|a|={float(np.linalg.norm(action)):.2f}  "
                  f"ΔeeA={eeA_travel*100:.1f}cm  ΔeeB={eeB_travel*100:.1f}cm")
        if terminated or truncated:
            print(f"[smoke] episode ended at step {t}: {info.get('termination')}")
            break
        if args.render:
            elapsed = time.time() - loop_start
            if elapsed < step_period:
                time.sleep(step_period - elapsed)
    dt = time.time() - t0
    print(f"[smoke] {actual_steps} steps in {dt:.2f}s  "
          f"({actual_steps / max(dt, 1e-6):.1f} steps/s)")
    print(f"[smoke] total_reward = {total_reward:+.3f}")
    print(
        f"[smoke] EE path length:  Robot A = {eeA_travel*100:7.2f} cm   "
        f"Robot B = {eeB_travel*100:7.2f} cm   "
        f"(per step ~ {eeA_travel*100/max(actual_steps,1):.2f} / "
        f"{eeB_travel*100/max(actual_steps,1):.2f} cm)"
    )
    if args.random_max and (eeA_travel + eeB_travel) < 0.01:
        print(
            "[smoke] WARNING: --random-max but EE barely moved (< 1 cm total). "
            "Check joint control mode / gains in src/env/robots.py "
            "(p.setJointMotorControlArray / setJointMotorControl2)."
        )
    print("[smoke] reward term sums:")
    for k, v in breakdown_sum.items():
        print(f"  {k:12s} {v:+.3f}")

    if args.check:
        from gymnasium.utils.env_checker import check_env
        print("[smoke] running check_env (this may print warnings)...")
        check_env(TyroEnv(cfg=EnvConfig(render=False), seed=1), warn=True)
        print("[smoke] check_env passed.")

    if args.render and not args.no_hold:
        print("[smoke] rollout finished -- GUI held open. "
              "Close the PyBullet window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                # Keep the GUI responsive without advancing physics; users can
                # drag the camera, inspect contact points, etc.
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
