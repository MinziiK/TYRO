#!/usr/bin/env python3
"""Headless end-to-end oracle: custom bolt order + corrected XZ transit path.

Drives Robot B along the analytic reference route (no RL policy):
  HOME → bolt0 staging → insert/hold/retract (FSM macro via teleport gates)
  → XZ hop at fixed retract Y → next bolt staging → … → 10/10

Follows ``env._nut_target_idx`` (custom ``nut_bolt_order``), not 0..9.
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
from scripts.preview_nut_fastening import (  # noqa: E402
    _best_b_ik, _hold_at, _teleport_b,
)


def _want_z(env: TyroEnv, idx: int) -> np.ndarray:
    a = np.asarray(env.scene.bolt_axis(idx), dtype=np.float64)
    return -a / max(float(np.linalg.norm(a)), 1e-9)


def _bolt_poses(env: TyroEnv, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (approach, insert, retract) world positions along bolt axis."""
    bp = np.asarray(env.scene.bolt_pose(idx)[0], dtype=np.float64)
    a = np.asarray(env.scene.bolt_axis(idx), dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    L = float(env.cfg.bolt_length)
    standoff = float(getattr(env.cfg, "nut_insert_standoff", 0.05))
    retract_clear = float(getattr(env.cfg, "nut_retract_clear", 0.03))
    approach = bp + a * (0.5 * L + standoff)
    insert = bp - a * (0.5 * L)
    retract = bp + a * (0.5 * L + retract_clear + 0.03)
    return approach, insert, retract


def _xz_hop(from_retr: np.ndarray, to_approach: np.ndarray) -> np.ndarray:
    return np.array([to_approach[0], from_retr[1], to_approach[2]], dtype=np.float64)


def _goto(env: TyroEnv, pos: np.ndarray, want_z: np.ndarray, seed: int,
          n_hold: int = 4) -> None:
    q = _best_b_ik(env, pos, want_z, seed_key=seed)
    _hold_at(env, q, n_hold, 0.0)


def _transit_to_approach(env: TyroEnv, idx: int, prev_retr: np.ndarray | None,
                         approach: np.ndarray, want_z: np.ndarray) -> str:
    """Move B to the bolt approach pose (HOME or XZ-only hop)."""
    if prev_retr is None:
        _goto(env, approach, want_z, seed=idx * 10)
        return "HOME→approach"
    hop = _xz_hop(prev_retr, approach)
    _goto(env, hop, want_z, seed=idx * 10 + 1, n_hold=3)
    if float(np.linalg.norm(hop - approach)) > 1e-4:
        _goto(env, approach, want_z, seed=idx * 10 + 2, n_hold=3)
    return "XZ→approach"


def _oracle_bolt(env: TyroEnv, idx: int, hold_need: int,
                 prev_retr: np.ndarray | None) -> tuple[bool, np.ndarray]:
    """One bolt: approach → insert/hold → retract. Returns (ok, retract_pos)."""
    want_z = _want_z(env, idx)
    approach, insert, retract = _bolt_poses(env, idx)
    phase = _transit_to_approach(env, idx, prev_retr, approach, want_z)

    _hold_at(env, _best_b_ik(env, approach, want_z, seed_key=idx), 4, 0.0)
    ax, lat, th = env._nut_axial_lateral(idx)

    q_ins = _best_b_ik(env, insert, want_z, seed_key=idx)
    _teleport_b(env, q_ins)
    ax, lat, th = env._nut_axial_lateral(idx)
    done, _, info = _hold_at(env, q_ins, hold_need + 6, 0.0,
                             stop_on_fasten_change=True)
    if done and info.get("all_fastened"):
        return True, retract

    q_ret = _best_b_ik(env, retract, want_z, seed_key=idx)
    _hold_at(env, q_ret, 10, 0.0, stop_on_fasten_change=True)
    ok = idx in env._nut_fastened
    return ok, retract


def run_e2e(seed: int = 0, verbose: bool = True) -> dict:
    cfg = make_env_config(
        stage=3, phase=1, nut_fastening_task=True,
        scene_layout="fanuc_spacious", terminate_on="never",
        nut_mount_endpose_path="data/nut_mount_endpose.npz",
        contact_force_terminate_above=0.0,
    )
    cfg.nut_b_hotstart_random_bolt = False
    cfg.nut_b_hotstart_alpha = 0.0

    env = TyroEnv(cfg=cfg, render=False, seed=seed)
    env.set_nut_b_hotstart_alpha(0.0)
    env.reset(seed=seed)

    order = env._nut_order()
    n = len(order)
    hold_need = int(cfg.nut_hold_steps)
    prev_retr: np.ndarray | None = None
    results = []
    seq = 0
    min_ba = float("inf")

    if verbose:
        print(f"[e2e] bolt order: {order}")
        print(f"{'seq':>3} {'bolt':>4} {'phase':<14} {'lat':>6} {'ba':>6} {'ok':>4}")

    stalled = 0
    while len(env._nut_fastened) < n and stalled < 3:
        idx = int(env._nut_target_idx)
        n_before = len(env._nut_fastened)
        ok, retract = _oracle_bolt(env, idx, hold_need, prev_retr)
        _, lat, _ = env._nut_axial_lateral(idx)
        ba = float(env._nut_ba_clearance())
        min_ba = min(min_ba, ba)
        phase = "HOME→approach" if prev_retr is None else "XZ→approach"
        results.append(dict(seq=seq, bolt=idx, ok=ok, lat=lat, ba=ba, phase=phase,
                            fastened=len(env._nut_fastened)))
        if verbose:
            print(f"{seq:3d} {idx:4d} {phase:<14} {lat:6.3f} {ba:6.3f} "
                  f"{'YES' if ok else 'NO':>4}  (total {len(env._nut_fastened)}/{n})")
        if ok:
            prev_retr = retract
            stalled = 0
        else:
            stalled += 1
        seq += 1
        if env._nut_done:
            break
        if len(env._nut_fastened) == n_before and not ok:
            # IK jitter: keep same transit origin, retry target bolt.
            continue

    summary = dict(
        order=order,
        n_fastened=len(env._nut_fastened),
        n_bolts=n,
        all_fastened=env._nut_done,
        min_ba_clearance=min_ba,
        results=results,
    )
    if verbose:
        print(f"\n[e2e] RESULT: {summary['n_fastened']}/{n} fastened  "
              f"all_fastened={summary['all_fastened']}  "
              f"min_B-A_clear={min_ba:.3f}m")
    env.close()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    s = run_e2e(seed=int(args.seed))
    return 0 if s["all_fastened"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
