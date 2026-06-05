#!/usr/bin/env python3
"""Verify heavy-tire contact stability at lower physics_num_sub_steps.

Runs a zero-action full carry->mount->demount->return cycle for each substep
value and reports, per stage: peak contact force (any FANUC contact + tire<->
hub), mount approach distance, and any NaN/explosion. Lets us pick the lowest
stable substep (the single biggest collection-speed lever, ~2x at 12->6).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def run(substeps: int) -> dict:
    cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious",
                          physics_num_sub_steps=int(substeps))
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    env.set_mount_tol(0.55, np.deg2rad(45.0))
    cli = env.client
    fanuc = env.robot_A.uid
    hub_uid = env.handles.hub.uid
    tire_uid = env.handles.tire
    mount_target = np.asarray(cfg.tire_mount_pos, float)

    act = np.zeros(env.action_space.shape, dtype=np.float32)
    peak_fanuc = {}
    peak_tire_hub = 0.0
    min_d_mount = 1e9
    nan_seen = False
    stages_reached = set()
    for _ in range(int(cfg.max_steps)):
        _, _, term, trunc, info = env.step(act)
        st = int(info.get("task_stage", env.task_stage))
        stages_reached.add(st)
        tp = np.asarray(env.scene.tire_pose()[0], float)
        if not np.all(np.isfinite(tp)):
            nan_seen = True
            break
        min_d_mount = min(min_d_mount, float(np.linalg.norm(tp - mount_target)))
        for cp in p.getContactPoints(physicsClientId=cli):
            a, b, f = cp[1], cp[2], cp[9]
            pair = {a, b}
            if fanuc in pair:
                peak_fanuc[st] = max(peak_fanuc.get(st, 0.0), f)
            if hub_uid in pair and tire_uid in pair:
                peak_tire_hub = max(peak_tire_hub, f)
        if term or trunc:
            break
    env.close()
    return {
        "substeps": substeps,
        "stages": sorted(stages_reached),
        "min_d_mount": min_d_mount,
        "peak_tire_hub": peak_tire_hub,
        "peak_fanuc": peak_fanuc,
        "nan": nan_seen,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substeps", type=int, nargs="+", default=[12, 8, 6])
    args = ap.parse_args()
    print("=" * 70)
    print("HEAVY-TIRE SUBSTEP STABILITY (zero-action full cycle)")
    print("=" * 70)
    results = []
    for ss in args.substeps:
        r = run(ss)
        results.append(r)
        pf = max(r["peak_fanuc"].values()) if r["peak_fanuc"] else 0.0
        seated = "SEATED" if r["min_d_mount"] < 0.10 else f"d={r['min_d_mount']:.2f}m"
        flag = " <<< NaN/EXPLODE" if r["nan"] else ""
        print(f"  substeps={ss:3d}  stages={r['stages']}  mount={seated}  "
              f"peak_tire_hub={r['peak_tire_hub']:7.0f}N  "
              f"peak_fanuc={pf:7.0f}N{flag}")
    # Compare against the 12 baseline
    base = next((r for r in results if r["substeps"] == 12), results[0])
    print("-" * 70)
    print(f"  baseline substeps=12: mount d={base['min_d_mount']:.3f}m  "
          f"tire_hub={base['peak_tire_hub']:.0f}N")
    for r in results:
        if r["substeps"] == base["substeps"]:
            continue
        ok = (not r["nan"]) and r["min_d_mount"] < max(0.10, base["min_d_mount"] * 1.5)
        verdict = "OK (stable + seats)" if ok else "RISKY"
        print(f"  substeps={r['substeps']}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
