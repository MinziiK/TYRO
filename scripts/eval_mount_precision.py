#!/usr/bin/env python3
"""Measure Robot-A mount precision (radial + angular residual) for a checkpoint.

Rolls out a mount policy deterministically and, at the ``mounted`` FSM event,
records the mount-gate metrics actually used by the env:

  * ``d_mount``  = ‖tire_pos − tire_mount_pos‖           (m)
  * ``theta``    = angle(tire_axis, hub_axis)            (deg)

Prints mean / std / p95 / max over the captured mounts so the finetune
(``planner_pos_offset_scale 0.12 → 0.03``) can be compared before/after.

Usage:
  python -m scripts.eval_mount_precision <ckpt.zip> --episodes 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_baselines3 import PPO  # noqa: E402

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402
from src.eval import _layout_overrides_for_checkpoint, _resolve_model_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="Path to the mount-policy .zip checkpoint.")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--scene-layout", type=str, default="fanuc_spacious")
    ap.add_argument("--mix-easy-prob", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tag", type=str, default="", help="Label printed with the summary.")
    args = ap.parse_args()

    model_path = _resolve_model_path(args.model)
    print(f"[prec] loading {model_path}")
    model = PPO.load(model_path, device="cpu")
    layout = _layout_overrides_for_checkpoint(model)

    overrides = dict(
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        contact_force_terminate_above=0.0,
        scene_layout=str(args.scene_layout),
        terminate_on="mount",
        start_pos_easy_prob=float(args.mix_easy_prob),
        **layout,
    )
    cfg = make_env_config(stage=args.stage, phase=args.phase, **overrides)
    env = TyroEnv(cfg=cfg, render=False, seed=args.seed)
    env.set_start_pos_easy_prob(float(args.mix_easy_prob))

    mount_target = np.asarray(cfg.tire_mount_pos, dtype=np.float64)

    d_list, ang_list = [], []
    attempts = 0
    target = int(args.episodes)
    env.reset(seed=args.seed)
    while len(d_list) < target and attempts < target * 4:
        attempts += 1
        obs, _ = env.reset(seed=args.seed + attempts)
        done = trunc = False
        captured = False
        while not (done or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, done, trunc, info = env.step(act)
            if info.get("mounted") or info.get("termination") == "mount_success":
                tp, _ = env.scene.tire_pose()
                t_axis = env.scene.tire_axis()
                h_axis = env.scene.hub_axis()
                d = float(np.linalg.norm(np.asarray(tp, dtype=np.float64) - mount_target))
                cth = float(np.clip(np.dot(t_axis, h_axis), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cth)))
                d_list.append(d)
                ang_list.append(ang)
                captured = True
                break
        flag = "OK" if captured else "no-mount"
        print(f"  ep {attempts}: {flag}  (captured {len(d_list)}/{target})")
    env.close()

    n = len(d_list)
    sr = 100.0 * n / max(attempts, 1)
    print(f"\n=== Mount precision {args.tag} ===")
    print(f"  model:        {model_path}")
    print(f"  captured:     {n}/{attempts} attempts  (mount rate {sr:.1f}%)")
    if n == 0:
        print("  FAIL: no mounts captured.")
        return 1
    d = np.asarray(d_list)
    a = np.asarray(ang_list)
    print(f"  d_mount (m):  mean={d.mean():.4f}  std={d.std():.4f}  "
          f"p95={np.percentile(d,95):.4f}  max={d.max():.4f}")
    print(f"  theta (deg):  mean={a.mean():.3f}  std={a.std():.3f}  "
          f"p95={np.percentile(a,95):.3f}  max={a.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
