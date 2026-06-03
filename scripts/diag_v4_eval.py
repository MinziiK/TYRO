"""Run deterministic eval episodes against latest v4 checkpoint and
log per-episode FSM event stats + termination reasons.

Hard-mode (no easy spawn) and easy-mode (start_pos_easy_prob=1.0) both.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.config import make_env_config
from src.env.tyro_env import TyroEnv


def run_eval(ckpt: str, easy_prob: float, n_eps: int, max_steps: int = 600,
             seed: int = 0):
    cfg = make_env_config(stage=3, phase=1)
    cfg.max_steps = max_steps
    cfg.terminate_on = "never"
    cfg.start_pos_easy_prob = easy_prob
    cfg.contact_force_terminate_above = 0.0
    cfg.render = False
    env = TyroEnv(cfg=cfg, seed=seed)
    model = PPO.load(ckpt, device="cpu")
    term_counter: Counter = Counter()
    success_count = 0
    final_stages: Counter = Counter()
    fsm_event_count: Counter = Counter()
    rew_list: list[float] = []
    len_list: list[int] = []

    last_fsm_seen: dict = {}

    for ep in range(n_eps):
        obs, info = env.reset(seed=seed + 100 + ep)
        ep_rew = 0.0
        ep_len = 0
        last_info: dict = info
        last_fsm = {"picked_up": False, "mounted": False,
                    "demounted": False, "landed": False}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_rew += float(r)
            ep_len += 1
            last_info = info
            for k in last_fsm:
                if info.get(k):
                    last_fsm[k] = True
            if term or trunc:
                break
        success_count += 1 if last_info.get("is_success") else 0
        term_counter[last_info.get("termination", "?")] += 1
        final_stages[int(env.task_stage)] += 1
        for k, v in last_fsm.items():
            if v:
                fsm_event_count[k] += 1
        rew_list.append(ep_rew)
        len_list.append(ep_len)
        print(
            f"  ep {ep:3d}: stage={env.task_stage} "
            f"term={last_info.get('termination'):>20s} "
            f"len={ep_len:4d} rew={ep_rew:7.1f} "
            f"picked={int(last_fsm['picked_up'])} "
            f"mounted={int(last_fsm['mounted'])} "
            f"demount={int(last_fsm['demounted'])} "
            f"landed={int(last_fsm['landed'])}"
        )

    print()
    print(f"--- mode: easy_prob={easy_prob:.2f}  n={n_eps} ---")
    print(f"success_rate: {success_count}/{n_eps} "
          f"= {100*success_count/n_eps:.1f}%")
    print(f"mean reward: {np.mean(rew_list):.1f}")
    print(f"mean length: {np.mean(len_list):.1f}")
    print("Termination breakdown:")
    for k, v in term_counter.most_common():
        print(f"  {k:25s} {v:3d} ({100*v/n_eps:.0f}%)")
    print("Final task_stage at termination:")
    for k, v in sorted(final_stages.items()):
        print(f"  stage={k}  {v:3d} ({100*v/n_eps:.0f}%)")
    print("FSM events ever fired during episode:")
    for k in ["picked_up", "mounted", "demounted", "landed"]:
        v = fsm_event_count.get(k, 0)
        print(f"  {k:12s} {v:3d} ({100*v/n_eps:.0f}%)")

    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase1_grad_v4/ckpts/"
                    "ppo_749880_steps.zip")
    ap.add_argument("--n-eps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("HARD MODE (no easy spawn, full pickup pipeline)")
    print("=" * 70)
    run_eval(args.ckpt, easy_prob=0.0, n_eps=args.n_eps, seed=args.seed)

    print()
    print("=" * 70)
    print("EASY MODE (always Stage 1 attached spawn)")
    print("=" * 70)
    run_eval(args.ckpt, easy_prob=1.0, n_eps=args.n_eps, seed=args.seed)
