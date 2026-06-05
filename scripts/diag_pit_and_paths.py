#!/usr/bin/env python3
"""Diagnose two GUI-observed issues on the current fanuc_spacious layout:

  1) Does the FANUC (Robot A) collide with the floor pit / rim (or its own
     plinth) now that the hub/cargo set was shifted +Y? Logs the peak
     arm-link contact force against the pit-rim bodies and the plane, per
     FSM stage, during a zero-action carry→mount→demount→return rollout.

  2) Are the Stage-2 (demount) and Stage-3 (return) nominal EE paths shaped
     as expected? Prints the planned EE polyline straightness (per-axis
     variance) and dominant motion direction for every stage.

Run with the live config (no overrides) so it reflects exactly what the
GUI replay shows.
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
    cfg = make_env_config(stage=3, phase=1, scene_layout="fanuc_spacious",
                          terminate_on="never")
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    # Wide gate so the full carry→mount→demount→return cycle actually runs.
    env.set_mount_tol(0.55, np.deg2rad(45.0))
    cli = env.client
    fanuc = env.robot_A.uid
    plane = env.handles.plane
    rim = list(getattr(env.handles, "floor_rim", []) or [])
    veh = env.handles.vehicle
    hub_uid = env.handles.hub.uid

    print("=" * 74)
    print("PIT / PATH DIAGNOSTIC (live fanuc_spacious config)")
    print(f"  robot_A_base   = {tuple(round(v,2) for v in cfg.robot_A_base_pos)}")
    print(f"  hub/mount      = {tuple(round(v,2) for v in cfg.tire_mount_pos)}")
    print(f"  pit center/R   = {cfg.floor_pit_center} / {cfg.floor_pit_radius}")
    print(f"  fanuc uid={fanuc} plane={plane} rim={rim} veh={veh} hub={hub_uid}")
    print("=" * 74)

    # ---- planned EE paths per stage (what the GUI draws) ----
    tr = env.compute_all_stage_trajectories(start_stage=1)
    for st in sorted(tr):
        pts = tr[st]
        d = np.diff(pts, axis=0)
        dom = ["x", "y", "z"][int(np.argmax(np.abs(d).sum(axis=0)))]
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        print(f"  stage{st} EE-path: pts={len(pts)} dom_axis={dom} "
              f"y[{ymin:.2f},{ymax:.2f}] "
              f"var(x,y,z)=({np.var(pts[:,0]):.4f},{np.var(pts[:,1]):.4f},"
              f"{np.var(pts[:,2]):.4f}) "
              f"end={tuple(round(float(v),2) for v in pts[-1])}")

    # ---- zero-action rollout: per-stage pit/plane/cargo contact ----
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    peak = {}  # stage -> {'rim':N,'plane':N,'cargo':N}
    rim_set = set(rim)
    for _ in range(int(cfg.max_steps)):
        _, _, term, trunc, info = env.step(act)
        st = int(info.get("task_stage", env.task_stage))
        d = peak.setdefault(st, {"rim": 0.0, "plane": 0.0, "cargo": 0.0})
        for cp in p.getContactPoints(physicsClientId=cli):
            a, b, f = cp[1], cp[2], cp[9]
            pair = {a, b}
            if fanuc not in pair:
                continue
            other = (pair - {fanuc}).pop() if len(pair) == 2 else fanuc
            if other in rim_set:
                d["rim"] = max(d["rim"], f)
            elif other == plane:
                d["plane"] = max(d["plane"], f)
            elif other == veh:
                d["cargo"] = max(d["cargo"], f)
        if term or trunc:
            break

    print("-" * 74)
    print("  FANUC peak contact force per stage (N):")
    for st in sorted(peak):
        d = peak[st]
        print(f"    stage{st}: rim={d['rim']:7.0f}  plane={d['plane']:7.0f}  "
              f"cargo={d['cargo']:7.0f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
