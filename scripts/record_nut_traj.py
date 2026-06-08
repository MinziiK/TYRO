#!/usr/bin/env python3
"""Record a trained Robot-B nut-fastening policy rollout to an .npz trajectory.

Headless (render=False, torch loaded) — produces /tmp/nut_traj.npz with the
exact per-frame poses the GUI ``replay`` mode reads back (qB/qA/tire + the
fastened-count / target-bolt bookkeeping). Also stores the analytic REFERENCE
("answer") path: for each bolt in the configured fastening order, the on-axis
staging approach point and the seated hub-face base point, so the GUI can draw
the ideal visiting route alongside the policy's actual behaviour.

Usage:
  python -m scripts.record_nut_traj --model runs/nut_fastening_v8/best/best_model.zip \
      --alpha 0.0 --max-steps 600 --out /tmp/nut_traj.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Trained PPO .zip path.")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="hot-start alpha (0 = cold/real deployment).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="/tmp/nut_traj.npz")
    ap.add_argument("--endpose", default="data/nut_mount_endpose.npz")
    ap.add_argument("--no-random-bolt", action="store_true",
                    help="Force sequential start at bolt 0 (no premark) so the "
                         "recording shows the policy fastening from the first "
                         "bolt onward instead of a random mid-sequence start.")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    cfg = make_env_config(
        stage=3, phase=1, nut_fastening_task=True,
        scene_layout="fanuc_spacious", terminate_on="never",
        nut_mount_endpose_path=args.endpose,
        contact_force_terminate_above=0.0,
    )
    if args.no_random_bolt:
        cfg.nut_b_hotstart_random_bolt = False
    env = TyroEnv(cfg=cfg, render=False, seed=int(args.seed))
    env.set_nut_b_hotstart_alpha(float(args.alpha))
    obs, _ = env.reset(seed=int(args.seed))

    rb, ra = env.robot_B, env.robot_A
    bidx = list(rb.arm.indices)
    aidx = list(ra.arm.indices)
    n = len(env.handles.bolts)
    home_ee = np.asarray(rb.ee_pose()[0], dtype=np.float64)  # start (HOME) EE

    qB, qA, tpos, torn, nf, tgt = [], [], [], [], [], []

    def snap() -> None:
        qB.append(np.asarray(rb.joint_state()[0], dtype=np.float32))
        qA.append(np.asarray(ra.joint_state()[0], dtype=np.float32))
        tp, to = env.scene.tire_pose()
        tpos.append(np.asarray(tp, dtype=np.float32))
        torn.append(np.asarray(to, dtype=np.float32))
        nf.append(len(env._nut_fastened))
        tgt.append(int(env._nut_target_idx))

    snap()  # initial frame
    last_f = len(env._nut_fastened)
    for t in range(int(args.max_steps)):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(a)
        snap()
        if len(env._nut_fastened) != last_f:
            print(f"  step {t+1:4d}: fastened -> {len(env._nut_fastened)}/{n} "
                  f"(target bolt {int(env._nut_target_idx)})")
            last_f = len(env._nut_fastened)
        if done or trunc:
            print(f"  ended @ step {t+1}: {info.get('termination','done')}")
            break

    # Reference waypoints (staging / base / retract per bolt, fastening order).
    from scripts.preview_nut_fastening import _compute_ref_waypoints  # noqa
    (_, ref_stage, ref_base, ref_retract, ref_order,
     _ref_center) = _compute_ref_waypoints(env)

    np.savez(
        args.out,
        qB=np.asarray(qB), qA=np.asarray(qA),
        tpos=np.asarray(tpos), torn=np.asarray(torn),
        nf=np.asarray(nf), tgt=np.asarray(tgt),
        bidx=np.asarray(bidx), aidx=np.asarray(aidx),
        alpha=float(args.alpha), seed=int(args.seed),
        ref_stage=np.asarray(ref_stage), ref_base=np.asarray(ref_base),
        ref_retract=np.asarray(ref_retract),
        ref_order=np.asarray(ref_order), home_ee=home_ee,
    )
    print(f"[record] saved {len(qB)} frames -> {args.out}  "
          f"(final {len(env._nut_fastened)}/{n}, max-target bolt {int(env._nut_target_idx)})")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
