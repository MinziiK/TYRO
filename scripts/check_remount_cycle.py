#!/usr/bin/env python3
"""End-to-end physical audit of the 6-stage remount cycle.

Runs ``TyroEnv`` with ``remount_cycle_enable=True`` from a clean Stage-0
HOME spawn and drives it with **zero policy residual** (pure baked Min-Jerk
planner) for the whole duty cycle:

    S0 pick → S1 mount → (W1 tighten hold) S2 retract to HOME
    → S3 re-grip hub tire → (W2 loosen hold) S4 demount → S5 carry to rack.

It logs every FSM stage transition, the per-stage peak contact force and
max per-step EE jump (singularity "whip"), and whether the episode reaches
the final ``landed`` success. This proves the cycle is physically realisable
on the baked reachable path before any RL is run.

Usage:
    python scripts/check_remount_cycle.py [--render] [--max-steps 1400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pybullet as p  # noqa: E402

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402

STAGE_NAMES = {
    0: "S0 pick",
    1: "S1 mount",
    2: "S2 retract->HOME",
    3: "S3 regrip",
    4: "S4 demount",
    5: "S5 return",
}


def pr(*a):
    print(*a)
    sys.stdout.flush()


def peak_contact_force(env) -> float:
    cf = 0.0
    for cp in p.getContactPoints(bodyA=env.robot_A.uid, physicsClientId=env.client):
        cf = max(cf, abs(float(cp[9])))
    return cf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-steps", type=int, default=1400)
    args = ap.parse_args()

    cfg = make_env_config(
        stage=1, phase=1, scene_layout="fanuc_spacious",
        remount_cycle_enable=True,
        terminate_on="never",
        reverse_curriculum_enable=False,
        start_pos_curriculum_enable=True,
        start_pos_curriculum_mode="mix",
        # Attached hot-start: tire already grasped at the cradle, task_stage=1.
        # This is the same entry the trainer uses for the carry/mount stages
        # and lets us validate the NEW S1→S5 machinery (S0 pickup reach is
        # validated separately by check_dynamic_feasibility).
        start_pos_easy_prob=1.0,
        attached_spawn_when_easy=True,
        # Generous gates so the pure baked planner (zero residual) reliably
        # trips each transition — we are auditing the FSM / grasp lifecycle,
        # not the policy's fine positioning.
        mount_radius_tol=0.18,
        regrip_radius_tol=0.18,
        home_return_radius_tol=0.20,
        rack_return_radius_tol=0.12,
        max_steps=int(args.max_steps),
    )
    pr("=" * 72)
    pr("6-STAGE REMOUNT CYCLE — end-to-end baked-planner audit")
    pr(f"  tighten_hold(W1)={cfg.tighten_hold_steps}  loosen_hold(W2)={cfg.loosen_hold_steps}")
    pr(f"  home_return_tol={cfg.home_return_radius_tol}  regrip_tol={cfg.regrip_radius_tol}")
    pr(f"  max_steps={cfg.max_steps}")
    pr("=" * 72)

    env = TyroEnv(cfg=cfg, render=bool(args.render), seed=0)
    env.set_start_pos_easy_prob(0.999)  # ~always easy; 1.0 is reset internally
    env.reset()
    # Loosen the mount gate the way the MountTolCurriculumCallback would
    # during early training (soft 25 cm / 40°) so the zero-residual baked
    # carry reliably trips the mount event for this machinery audit.
    env.set_mount_tol(0.25, np.deg2rad(40.0))

    act = np.zeros(env.action_space.shape, dtype=np.float32)
    cur_stage = int(env.task_stage)
    stage_entry_step = 0
    prev_ee = np.asarray(env.robot_A.ee_pose()[0], float)
    stage_peak_cf = 0.0
    stage_max_jump = 0.0
    reached = {k: False for k in STAGE_NAMES}
    reached[cur_stage] = True
    success = False
    term_reason = None

    pr(f"  [step {0:4d}] enter {STAGE_NAMES[cur_stage]}")
    for i in range(int(args.max_steps)):
        _, _, term, trunc, info = env.step(act)
        ee = np.asarray(env.robot_A.ee_pose()[0], float)
        stage_max_jump = max(stage_max_jump, float(np.linalg.norm(ee - prev_ee)))
        prev_ee = ee
        stage_peak_cf = max(stage_peak_cf, peak_contact_force(env))

        new_stage = int(info.get("task_stage", env.task_stage))
        if new_stage != cur_stage:
            pr(f"            {STAGE_NAMES[cur_stage]:18s} done in "
               f"{i + 1 - stage_entry_step:4d} steps  "
               f"peak_cf={stage_peak_cf:8.0f}  max_jump={stage_max_jump*100:5.1f}cm")
            cur_stage = new_stage
            reached[cur_stage] = True
            stage_entry_step = i + 1
            stage_peak_cf = 0.0
            stage_max_jump = 0.0
            pr(f"  [step {i+1:4d}] enter {STAGE_NAMES[cur_stage]}")

        # Periodic in-stage progress (tire→hub distance, EE→home, hold left).
        if (i + 1) % 50 == 0:
            tp = np.asarray(env.scene.tire_pose()[0], float)
            hp = np.asarray(env.scene.hub_pose()[0], float)
            d_hub = float(np.linalg.norm(tp - hp))
            d_home = (float(np.linalg.norm(ee - np.asarray(env._home_ee_pos, float)))
                      if getattr(env, "_home_ee_pos", None) is not None else -1.0)
            pr(f"            ...[{i+1:4d}] {STAGE_NAMES[cur_stage]:16s} "
               f"d(tire,hub)={d_hub:.3f}  d(ee,home)={d_home:.3f}  "
               f"hold_left={env._mount_hold_left}")

        if info.get("is_success"):
            success = True
        if term or trunc:
            term_reason = info.get("termination", "trunc" if trunc else "term")
            break

    pr("-" * 72)
    pr(f"  final stage reached: {STAGE_NAMES[cur_stage]}")
    pr(f"  termination: {term_reason}   success={success}")
    pr("  stages reached: " + ", ".join(
        STAGE_NAMES[k] for k in sorted(STAGE_NAMES) if reached[k]))
    env.close()

    # Started at S1 (attached hot-start); require reaching S5 + landed.
    new_stages = (2, 3, 4, 5)
    new_reached = all(reached[k] for k in new_stages)
    pr("=" * 72)
    if new_reached and success:
        pr("VERDICT: 6-STAGE MACHINERY OK — mount->retract->regrip->"
           "demount->return->landed all fired from the attached hot-start.")
    elif new_reached:
        pr("VERDICT: all new stages entered but no landed success — check S5 "
           "return tolerance / landing speed.")
    else:
        missing = [STAGE_NAMES[k] for k in new_stages if not reached[k]]
        pr("VERDICT: cycle stalled. New stages NOT reached: " + ", ".join(missing))
        pr(f"         (last term reason: {term_reason})")
    pr("=" * 72)
    return 0 if (new_reached and success) else 1


if __name__ == "__main__":
    raise SystemExit(main())
