"""Evaluate a trained PPO checkpoint on TyroEnv.

Examples
--------
    # Run 20 deterministic episodes, print success rate / reward stats:
    python -m src.eval runs/stage3_phase1_*/best/best_model.zip --episodes 20

    # Watch the policy in GUI, hold window after each episode:
    python -m src.eval <ckpt.zip> --render --episodes 3

    # Phase-1 mount policy with easy-start (matches 75%% mix training):
    python -m src.eval runs/.../best/best_model.zip --render --phase 1 \\
        --easy-start --episodes 5

    # Evaluate at a different DR phase than training to check transfer:
    python -m src.eval <ckpt.zip> --phase 3
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from src.config import make_env_config
from src.env import TyroEnv


def _layout_overrides_for_checkpoint(model: PPO) -> dict:
    """Match env action/obs layout to the checkpoint's training era."""
    obs_d = int(model.observation_space.shape[0])
    act_d = int(model.action_space.shape[0])
    if obs_d == 85 and act_d == 6:
        return {"include_hub_guide_obs": True}
    if obs_d == 82 and act_d == 6:
        return {"include_hub_guide_obs": False}
    if obs_d == 83 and act_d == 7:
        return {
            "include_hub_guide_obs": False,
            "legacy_action_dim": 7,
            "legacy_obs_dim": 83,
        }
    raise ValueError(
        f"Unsupported checkpoint layout: obs_dim={obs_d}, action_dim={act_d}. "
        "Known layouts: (85,6) current, (82,6) pre-hub-guide, (83,7) legacy sharp."
    )


def _resolve_model_path(path: str) -> str:
    """SB3 PPO.load appends '.zip'; strip it if the user passed '*.zip'."""
    p = Path(path)
    if not p.exists() and p.suffix.lower() == ".zip":
        bare = p.with_suffix("")
        if bare.with_suffix(".zip").exists():
            return str(bare)
    if p.suffix.lower() == ".zip" and p.exists():
        return str(p.with_suffix(""))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="Path to .zip checkpoint.")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                    help="DR phase (Phase-1 FSM checkpoints use --phase 1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of using deterministic mean.")
    ap.add_argument("--no-hold", action="store_true",
                    help="Don't hold the GUI window after the last episode.")
    ap.add_argument(
        "--easy-start",
        action="store_true",
        help="Force easy spawn every reset (tire near grasp, lift 0.10 m).",
    )
    ap.add_argument(
        "--home-start",
        action="store_true",
        help="Force HOME spawn every reset (production eval, no easy mix).",
    )
    ap.add_argument(
        "--mix-easy-prob",
        type=float,
        default=None,
        help="Bernoulli easy probability per reset (mix mode). "
        "Default: EnvConfig.start_pos_easy_prob (0.75 for v9b).",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode horizon (default: EnvConfig.max_steps).",
    )
    ap.add_argument(
        "--terminate-on",
        type=str,
        default=None,
        choices=("never", "pickup", "mount", "demount"),
        help="FSM early-success gate (default: EnvConfig.terminate_on, usually 'never').",
    )
    ap.add_argument(
        "--scene-layout",
        type=str,
        default="fanuc_spacious",
        choices=("shipping", "fanuc_spacious"),
        help=(
            "Scene layout to render. Must match the checkpoint's training "
            "layout (server runs default to 'fanuc_spacious')."
        ),
    )
    args = ap.parse_args()
    if args.easy_start and args.home_start:
        ap.error("Use only one of --easy-start or --home-start.")

    overrides: dict = dict(
        render=args.render,
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        contact_force_terminate_above=0.0,
        scene_layout=str(args.scene_layout),
    )
    if args.terminate_on is not None:
        overrides["terminate_on"] = str(args.terminate_on)
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.mix_easy_prob is not None:
        overrides["start_pos_easy_prob"] = float(args.mix_easy_prob)

    model_path = _resolve_model_path(args.model)
    print(f"[eval] loading {model_path}")
    model = PPO.load(model_path, device="cpu")
    layout = _layout_overrides_for_checkpoint(model)
    overrides.update(layout)
    print(
        f"[eval] checkpoint layout: obs_dim={model.observation_space.shape[0]} "
        f"action_dim={model.action_space.shape[0]} "
        f"hub_guide={layout.get('include_hub_guide_obs', True)}"
    )

    cfg = make_env_config(stage=args.stage, phase=args.phase, **overrides)
    env = TyroEnv(cfg=cfg, render=args.render, seed=args.seed)
    if args.easy_start:
        # mix mode treats alpha>=1.0 as "use cfg"; 0.999 forces always-easy.
        env.set_start_pos_easy_prob(0.999)
        print("[eval] start pose: easy (forced)")
    elif args.home_start:
        env.set_start_pos_easy_prob(0.0)
        print("[eval] start pose: HOME (forced)")
    elif args.mix_easy_prob is not None:
        env.set_start_pos_easy_prob(float(args.mix_easy_prob))
        print(f"[eval] start pose: mix easy_prob={args.mix_easy_prob}")
    else:
        print(
            f"[eval] start pose: mix easy_prob="
            f"{getattr(cfg, 'start_pos_easy_prob', 0.5)} (EnvConfig default)"
        )
    print(f"[eval] terminate_on={cfg.terminate_on!r}  max_steps={cfg.max_steps}")

    successes: list[bool] = []
    rewards: list[float] = []
    lengths: list[int] = []
    term_counts: dict[str, int] = {}
    step_period = 1.0 / cfg.control_freq_hz

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_r = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            t_start = time.time()
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            steps += 1
            if args.render:
                dt = time.time() - t_start
                if dt < step_period:
                    time.sleep(step_period - dt)
        is_success = bool(info.get("is_success", False))
        term = info.get("termination", "unknown")
        term_counts[term] = term_counts.get(term, 0) + 1
        successes.append(is_success)
        rewards.append(total_r)
        lengths.append(steps)
        print(f"  ep {ep:3d}  r={total_r:+8.2f}  len={steps:4d}  "
              f"success={is_success}  termination={term}")

    n = len(successes)
    sr = sum(successes) / n * 100
    print("\n=== Eval summary ===")
    print(f"  episodes:     {n}")
    print(f"  success rate: {sr:.1f}%  ({sum(successes)}/{n})")
    print(f"  reward:       mean={statistics.mean(rewards):+.2f}  "
          f"std={statistics.pstdev(rewards):.2f}  "
          f"min={min(rewards):+.2f}  max={max(rewards):+.2f}")
    print(f"  length:       mean={statistics.mean(lengths):.1f}  "
          f"min={min(lengths)}  max={max(lengths)}")
    print(f"  terminations: {term_counts}")

    if args.render and not args.no_hold:
        print("[eval] GUI held open. Close window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
