#!/usr/bin/env python3
"""Measure Robot-B <-> Robot-A clearance along a RECORDED policy trajectory.

Replays /tmp/nut_traj.npz frame-by-frame (the exact poses the user watches in
the GUI), driving B and A joints + the tire from the recording, and reports the
closest B-link <-> A-link distance per frame. Flags frames with penetration
(<0) or tight clearance, grouped by the recorded target bolt, to pinpoint where
and how B contacts A during transit.
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


def _link_names(uid, client):
    names = {-1: "base"}
    for li in range(p.getNumJoints(uid, physicsClientId=client)):
        names[li] = p.getJointInfo(uid, li, physicsClientId=client)[12].decode()
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/tmp/nut_traj.npz")
    args = ap.parse_args()

    d = np.load(args.traj)
    qB, qA, tpos, torn = d["qB"], d["qA"], d["tpos"], d["torn"]
    nf, tgt = d["nf"], d["tgt"]
    bidx = [int(x) for x in d["bidx"]]
    aidx = [int(x) for x in d["aidx"]]
    T = len(qB)

    cfg = make_env_config(
        stage=3, phase=1, nut_fastening_task=True,
        scene_layout="fanuc_spacious", terminate_on="never",
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
        contact_force_terminate_above=0.0,
    )
    env = TyroEnv(cfg=cfg, render=False, seed=int(d["seed"]))
    env.set_nut_b_hotstart_alpha(float(d["alpha"]))
    env.reset(seed=int(d["seed"]))

    a_uid, b_uid = int(env.robot_A.uid), int(env.robot_B.uid)
    a_names, b_names = _link_names(a_uid, env.client), _link_names(b_uid, env.client)
    cid = env.client

    print(f"[traj] {T} frames  seed={int(d['seed'])} alpha={float(d['alpha']):.2f}")
    # Per-target-bolt min clearance.
    per_bolt = {}
    coll_frames = []
    for t in range(T):
        for s, q in zip(bidx, qB[t]):
            p.resetJointState(b_uid, s, float(q), physicsClientId=cid)
        for s, q in zip(aidx, qA[t]):
            p.resetJointState(a_uid, s, float(q), physicsClientId=cid)
        cps = p.getClosestPoints(b_uid, a_uid, distance=0.4, physicsClientId=cid)
        if not cps:
            mind, bl, al = 0.4, "-", "-"
        else:
            cp = min(cps, key=lambda c: float(c[8]))
            mind = float(cp[8])
            bl, al = b_names.get(cp[3], str(cp[3])), a_names.get(cp[4], str(cp[4]))
        bolt = int(tgt[t])
        rec = per_bolt.setdefault(bolt, [9, None, None, None])
        if mind < rec[0]:
            per_bolt[bolt] = [mind, t, bl, al]
        if mind < 0:
            coll_frames.append((t, bolt, mind, bl, al))

    env.close()

    print(f"\n{'tgtBolt':>7} {'minClear(cm)':>12} {'@frame':>7}  culprit B<->A")
    for bolt in sorted(per_bolt):
        mind, t, bl, al = per_bolt[bolt]
        flag = "  <== COLLISION" if mind < 0 else ("  <- tight" if mind < 0.03 else "")
        print(f"{bolt:>7} {mind*100:>12.1f} {t:>7}  {bl} <-> {al}{flag}")

    if coll_frames:
        print(f"\n[traj] {len(coll_frames)} colliding frames; first few:")
        for t, bolt, mind, bl, al in coll_frames[:8]:
            print(f"   frame {t:4d} bolt{bolt} {mind*100:6.1f}cm  {bl} <-> {al}")
    else:
        print("\n[traj] NO B<->A penetration along the recorded trajectory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
