#!/usr/bin/env python3
"""Dynamic physical-feasibility audit for the fanuc_spacious layout.

Beyond static IK reach, this DRIVES the env through the full Phase-1 FSM using
the env's OWN baked joint trajectories (the reachable path the sim replays),
watching every physics gate the policy would hit:

  1. Static clearances at reset — robot↔robot, robot↔vehicle/back-wall,
     robot↔floor, self-collision, rack↔base distance.
  2. Per-stage baked replay — for S0..S3 drive the baked joints step by step
     and record: early termination (collision / workspace / contact_force),
     max per-step EE jump (singularity "whip"), peak contact force, and the
     tire's vertical drift (does it stay seated / mounted?).
  3. Tire settle test — 60 steps of zero action at reset; the tire must stay
     on the rails (no fall / explosion).

PASS criteria are printed per row; a final verdict summarises blockers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pybullet as p  # noqa: E402

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def pr(*a):
    print(*a)
    sys.stdout.flush()


def closest(env, a, b, links_a_min=-2):
    cps = p.getClosestPoints(a, b, distance=0.5, physicsClientId=env.client)
    vals = [c[8] for c in cps if c[3] >= links_a_min]
    return min(vals) if vals else 9.9


def static_clearances(env):
    h = env.handles
    A, B = env.robot_A.uid, env.robot_B.uid
    pr("-" * 70)
    pr("1) STATIC CLEARANCES @ reset")
    pr("-" * 70)
    dAB = closest(env, A, B)
    pr(f"  robotA <-> robotB      {dAB*100:+7.1f} cm   {'OK' if dAB > 0.05 else 'TIGHT/HIT'}")
    if h.vehicle is not None:
        dav = closest(env, A, h.vehicle)
        dbv = closest(env, B, h.vehicle)
        pr(f"  robotA <-> vehicle     {dav*100:+7.1f} cm   {'OK' if dav > 0.02 else 'HIT'}")
        pr(f"  robotB <-> vehicle     {dbv*100:+7.1f} cm   {'OK' if dbv > 0.02 else 'HIT'}")
    bw = getattr(h, "cargo_back_wall", None)
    if bw is not None:
        dabw = closest(env, A, bw)
        pr(f"  robotA <-> back-wall   {dabw*100:+7.1f} cm   {'OK' if dabw > 0.02 else 'HIT'}")
    # self-collision (A): contacts between non-adjacent links
    p.stepSimulation(physicsClientId=env.client)
    selfcps = p.getContactPoints(A, A, physicsClientId=env.client)
    nontrivial = [c for c in selfcps if abs(c[3] - c[4]) > 1]
    pr(f"  robotA self-collision  {len(nontrivial)} pairs   {'OK' if not nontrivial else 'SELF-HIT'}")
    # rack <-> A base planar
    ab = np.asarray(env.cfg.robot_A_base_pos, float)
    rk = np.asarray(env.cfg.tire_rack_inner_center, float)
    pr(f"  rack <-> A base planar {np.hypot(*(rk-ab)[:2]):.2f} m   (A reach 2.655 m)")
    # tire seated
    for u in h.tire_rack:
        d = closest(env, h.tire, u)
        pr(f"  tire <-> rail u{u}      {d*100:+7.1f} cm   {'seated' if abs(d) < 0.02 else ('FLOAT' if d>0 else 'overlap-ok')}")


def drive_stage_baked(env, stage, max_steps=160):
    """Drive the baked joint traj for a stage; return diagnostics dict."""
    env.task_stage = stage
    env._replan_for_current_stage()
    q = np.asarray(env._traj_q, float)
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    prev_ee = np.asarray(env.robot_A.ee_pose()[0], float)
    max_jump = 0.0
    peak_cf = 0.0
    term_reason = None
    tire_z0 = float(env.scene.tire_pose()[0][2])
    n = min(len(q), max_steps)
    for i in range(n):
        # command the baked arm joints directly (bypass policy residual)
        env.robot_A.drive_arm_targets(q[i])
        p.stepSimulation(physicsClientId=env.client)
        ee = np.asarray(env.robot_A.ee_pose()[0], float)
        max_jump = max(max_jump, float(np.linalg.norm(ee - prev_ee)))
        prev_ee = ee
        # contact force
        cf = 0.0
        for cp in p.getContactPoints(bodyA=env.robot_A.uid, physicsClientId=env.client):
            cf = max(cf, abs(float(cp[9])))
        peak_cf = max(peak_cf, cf)
        if env._in_bad_collision():
            term_reason = "collision"
            break
    tire_z1 = float(env.scene.tire_pose()[0][2])
    return {
        "steps": n, "max_jump_cm": max_jump * 100.0,
        "peak_cf": peak_cf, "term": term_reason,
        "tire_dz_cm": (tire_z1 - tire_z0) * 100.0,
    }


def main() -> int:
    cfg = make_env_config(stage=1, phase=1, scene_layout="fanuc_spacious")
    pr("=" * 70)
    pr("DYNAMIC FEASIBILITY — fanuc_spacious")
    pr(f"  A={cfg.robot_A_base_pos} B={cfg.robot_B_base_pos} floor={cfg.floor_z}")
    pr(f"  hub={cfg.hub_pos_nominal} pickup={cfg.tire_pickup_pos}")
    pr("=" * 70)

    # --- static + tire settle (home start, no grasp) ---
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.0)
    env.reset()
    static_clearances(env)
    pr("-" * 70)
    pr("2) TIRE SETTLE — 60 steps zero action (tire must stay on rails)")
    pr("-" * 70)
    tz0 = float(env.scene.tire_pose()[0][2])
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    term = None
    for i in range(60):
        _, _, t, tr, info = env.step(act)
        if t or tr:
            term = (i + 1, info.get("termination"))
            break
    tz1 = float(env.scene.tire_pose()[0][2])
    pr(f"  tire z {tz0:.3f} -> {tz1:.3f} (drift {abs(tz1-tz0)*100:.1f} cm)   "
       f"{'STABLE' if abs(tz1-tz0) < 0.03 and term is None else ('term@'+str(term) if term else 'DRIFT')}")
    env.close()

    # --- per-stage baked replay (easy start to cache grasp transform) ---
    pr("-" * 70)
    pr("3) PER-STAGE BAKED REPLAY (drive reachable joints, watch gates)")
    pr("   jump=max per-step EE move (singularity whip), cf=peak contact force")
    pr("-" * 70)
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    env.set_start_pos_easy_prob(0.999)
    env.reset()
    names = {0: "S0 grasp ", 1: "S1 mount ", 2: "S2 demount", 3: "S3 return"}
    blockers = []
    for st in (0, 1, 2, 3):
        d = drive_stage_baked(env, st)
        whip = d["max_jump_cm"] > 15.0
        hit = d["term"] is not None
        if hit:
            blockers.append(f"{names[st].strip()}: {d['term']}")
        if whip:
            blockers.append(f"{names[st].strip()}: whip {d['max_jump_cm']:.0f}cm")
        pr(f"  {names[st]}  steps={d['steps']:3d}  jump={d['max_jump_cm']:5.1f}cm  "
           f"peak_cf={d['peak_cf']:8.0f}  tire_dz={d['tire_dz_cm']:+6.1f}cm  "
           f"term={d['term'] or '-':9s}  {'<<BLOCK' if (hit or whip) else 'ok'}")
    env.close()

    pr("=" * 70)
    if blockers:
        pr("VERDICT: BLOCKERS FOUND")
        for b in blockers:
            pr("  - " + b)
    else:
        pr("VERDICT: no hard blockers — all stages replay without collision/whip")
    pr("=" * 70)
    pr("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
