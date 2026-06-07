#!/usr/bin/env python3
"""Extract Robot-A mount-completion poses from a trained mount policy.

Rolls out the Phase-A mount checkpoint and, at the instant the tire is
mounted (FSM ``mounted`` event), records Robot A's arm joint vector and the
seated tire pose. The collected snapshots are saved to a ``.npz`` consumed by
the nut-fastening task (``cfg.nut_mount_endpose_path``) so Robot A is frozen
at the *actual* learned mount-hold configuration (+ optional jitter) while
Robot B trains to fasten the bolts.

Usage:
  python -m scripts.extract_mount_endpose runs/phase1_mount_v2/best/best_model.zip \
      --episodes 40 --out data/nut_mount_endpose.npz
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
    ap.add_argument("--mix-easy-prob", type=float, default=0.9,
                    help="Easy-spawn probability (matches the mount eval).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="data/nut_mount_endpose.npz")
    args = ap.parse_args()

    model_path = _resolve_model_path(args.model)
    print(f"[extract] loading {model_path}")
    model = PPO.load(model_path, device="cpu")
    layout = _layout_overrides_for_checkpoint(model)
    print(f"[extract] checkpoint layout obs={model.observation_space.shape[0]} "
          f"act={model.action_space.shape[0]} -> {layout}")

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

    qA_list, tpos_list, torn_list = [], [], []
    env.reset(seed=args.seed)  # build robots so we can read the arm DOF
    n_dof = int(env.robot_A.arm.n)
    attempts = 0
    target = int(args.episodes)
    while len(qA_list) < target and attempts < target * 4:
        attempts += 1
        obs, _ = env.reset(seed=args.seed + attempts)
        done = trunc = False
        captured = False
        while not (done or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, done, trunc, info = env.step(act)
            if info.get("mounted") or info.get("termination") == "mount_success":
                qA, _ = env.robot_A.joint_state()
                tp, to = env.scene.tire_pose()
                qA_list.append(np.asarray(qA, dtype=np.float64)[:n_dof])
                tpos_list.append(np.asarray(tp, dtype=np.float64))
                torn_list.append(np.asarray(to, dtype=np.float64))
                captured = True
                break
        flag = "OK" if captured else "no-mount"
        print(f"  ep {attempts}: {flag}  (collected {len(qA_list)}/{target})")
    env.close()

    if not qA_list:
        print("[extract] FAIL: no mount completions captured.")
        return 1

    qA = np.stack(qA_list)
    tpos = np.stack(tpos_list)
    torn = np.stack(torn_list)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, qA=qA, tire_pos=tpos, tire_orn=torn)
    print(f"[extract] saved {qA.shape[0]} mount-hold poses -> {out}")
    print(f"[extract] qA mean={np.round(qA.mean(0),3)} "
          f"std={np.round(qA.std(0),3)} (rad)")
    print(f"[extract] tire_pos mean={np.round(tpos.mean(0),4)} "
          f"std={np.round(tpos.std(0),4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
