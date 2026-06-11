"""Policy-quality evaluation for a v21 nut checkpoint.

Two complementary tests, both deterministic (no exploration noise):

1) PER-BOLT SEAT (alpha=1, per-leg): hot-start B at EACH of the 10 bolts in
   turn and run one leg. Counts how many bolts the policy approaches AND seats
   (full envelopment). Isolates the seat/INSERT capability and directly exercises
   the v21 branch-aware fix (edge bolts must seat). Also reports how often the
   branch search fired / succeeded.

2) CHAIN FROM HOME (alpha=0): full 10-bolt chain from the HOME pose, the REAL
   task. Reports n_fastened per episode (mean / distribution).

Prints a machine-readable summary line:  RESULT seat=X/10 chain_mean=Y ...
"""
import argparse
import sys
import numpy as np
from stable_baselines3 import PPO
from src.config import make_env_config
from src.env import TyroEnv

V20 = dict(nut_b_align_servo=True, nut_a_kinematic_freeze=True, nut_collision_fail=True,
    nut_b_solo_action=True, nut_arrive_lat_tol=0.015, nut_seat_lat_mult=1.0,
    nut_b_axial_insert_servo=True, nut_insert_depth_tol=0.007,
    nut_b_insert_branch_search=True)


def per_bolt_seat(model):
    """Hot-start each bolt (alpha=1, per-leg); count approached+seated."""
    seated = 0
    fired = 0
    fired_ok = 0
    fails = []
    for b in range(10):
        order = [b] + [k for k in range(10) if k != b]
        cfg = make_env_config(3, 1, render=False, scene_layout='fanuc_spacious',
            nut_fastening_task=True, nut_pure_rl=True, nut_b_hotstart_enable=True,
            nut_b_hotstart_alpha=1.0, nut_b_hotstart_random_bolt=False,
            nut_per_leg_episode=True, max_steps=400, terminate_on='never',
            nut_bolt_order=tuple(order), USE_DOMAIN_RANDOMIZATION=False, **V20)
        env = TyroEnv(cfg=cfg, render=False, seed=11 + b)
        # instrument branch switch
        orig = env._nut_switch_to_seat_branch
        cnt = {'n': 0, 'ok': 0}
        def wrap(idx, _orig=orig, _c=cnt):
            _c['n'] += 1
            r = _orig(idx)
            if r:
                _c['ok'] += 1
            return r
        env._nut_switch_to_seat_branch = wrap
        obs, _ = env.reset(seed=11 + b)
        idx = int(env._nut_target_idx)
        ok = False
        for t in range(400):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if info.get('fastened') or info.get('n_fastened', 0) > 0:
                ok = True
                break
            if term or trunc:
                break
        seated += int(ok)
        fired += cnt['n']
        fired_ok += cnt['ok']
        if not ok:
            fails.append(b)
        env.close()
    return seated, fails, fired, fired_ok


def chain_from_home(model, seeds=(200, 201, 202, 203, 204)):
    cfg = make_env_config(3, 1, render=False, scene_layout='fanuc_spacious',
        nut_fastening_task=True, nut_pure_rl=True, nut_b_hotstart_enable=True,
        nut_b_hotstart_alpha=0.0, nut_b_hotstart_random_bolt=False,
        nut_per_leg_episode=False, max_steps=2500, terminate_on='never',
        nut_stall_steps=0, USE_DOMAIN_RANDOMIZATION=False, **V20)
    env = TyroEnv(cfg=cfg, render=False, seed=1)
    res = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        nf = 0
        for t in range(2500):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            nf = info.get('n_fastened', nf)
            if term or trunc:
                break
        res.append(nf)
    env.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--skip-chain", action="store_true")
    args = ap.parse_args()
    m = PPO.load(args.model, device='cpu')
    seat, fails, fired, fired_ok = per_bolt_seat(m)
    chain = [] if args.skip_chain else chain_from_home(m)
    chain_mean = float(np.mean(chain)) if chain else -1.0
    print(f"\nRESULT model={args.model}")
    print(f"  per_bolt_seat = {seat}/10   (failed bolts: {fails})")
    print(f"  branch_search fired={fired} ok={fired_ok}")
    if chain:
        print(f"  chain_from_home n_fastened = {chain}  mean={chain_mean:.1f}/10")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
