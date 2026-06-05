#!/usr/bin/env python3
"""Identify WHICH FANUC link touches the floor pit rim, where, and whether
the contact actually impedes the carry (persistent base contact vs a moving
arm-link collision). Logs per-link peak force + contact Z and the EE
tracking error during Stage 1.
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


def main() -> int:
    cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious")
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    env.set_mount_tol(0.55, np.deg2rad(45.0))
    cli = env.client
    fanuc = env.robot_A.uid
    rim = set(getattr(env.handles, "floor_rim", []) or [])

    njoints = p.getNumJoints(fanuc, physicsClientId=cli)
    link_name = {-1: "base"}
    for j in range(njoints):
        link_name[j] = p.getJointInfo(fanuc, j, physicsClientId=cli)[12].decode()

    print("=" * 72)
    print(f"  fanuc uid={fanuc}  rim_uids={rim}  base_pos="
          f"{tuple(round(v,2) for v in cfg.robot_A_base_pos)}")
    print("=" * 72)

    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    peak = {}      # link_idx -> [peakF, contactZ]
    steps = 0
    while True:
        _, _, term, trunc, _ = env.step(zero)
        steps += 1
        if int(env.task_stage) != 1:
            if steps > 5:
                break
        for cp in p.getContactPoints(bodyA=fanuc, physicsClientId=cli):
            if cp[2] not in rim:
                continue
            li = cp[3]            # link index on fanuc (bodyA)
            f = cp[9]
            cz = cp[6][2]         # contact position Z on B
            d = peak.setdefault(li, [0.0, cz])
            if f > d[0]:
                d[0] = f
                d[1] = cz
        if term or trunc:
            break

    print(f"  ran {steps} steps. FANUC links touching pit rim:")
    if not peak:
        print("    (none)")
    for li in sorted(peak):
        f, cz = peak[li]
        print(f"    link[{li:2d}] {link_name.get(li,'?'):<12} "
              f"peakF={f:7.0f}N  contactZ={cz:+.2f}m")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
