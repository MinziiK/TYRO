"""Per-bolt diagnosis of the v17 attempt4 policy at hot-start alpha=1.0.

For each episode (random target bolt): record target idx, success, whether
INSERT subphase fired, sum of collision penalties (B contacts), and min
B<->A clearance seen. Buckets by bolt to find which bolts poison training.
"""
from pathlib import Path
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from src.config import make_env_config
from src.env import TyroEnv

ckpt = sorted(Path("runs/nut_fastening_v17_purerl/ckpts").glob("ppo_*_steps.zip"),
              key=lambda pp: int(pp.stem.split("_")[1]))[-1]
model = PPO.load(str(ckpt.with_suffix("")), device="cpu")
print("model:", ckpt.name, flush=True)

cfg = make_env_config(3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True, nut_pure_rl=True,
    nut_b_hotstart_enable=True, nut_b_hotstart_alpha=1.0,
    nut_b_hotstart_random_bolt=True, max_steps=800, terminate_on="never",
    USE_DOMAIN_RANDOMIZATION=False, collision_terminates=False)
env = TyroEnv(cfg=cfg, render=False, seed=0)

stats = {i: dict(n=0, succ=0, ins=0, col=0, minba=[], eplen=[]) for i in range(10)}
ep = 0
while min(s["n"] for s in stats.values()) < 5 and ep < 120:
    obs, _ = env.reset(seed=ep)
    ep += 1
    idx = int(env._nut_target_idx)
    st = stats[idx]
    st["n"] += 1
    sub1 = False
    ncol = 0
    minba = 1e9
    steps = 0
    info = {}
    for t in range(800):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if int(env._nut_subphase) == 1:
            sub1 = True
        if t % 10 == 0:
            cps = p.getClosestPoints(env.robot_B.uid, env.robot_A.uid,
                                     distance=1.0, physicsClientId=env.client)
            d = min((c[8] for c in cps), default=1e9)
            minba = min(minba, d)
            if d < 0.0:
                ncol += 1
        if term or trunc:
            break
    st["succ"] += int(bool(info.get("is_success")))
    st["ins"] += int(sub1)
    st["col"] += int(ncol > 0)
    st["minba"].append(minba)
    st["eplen"].append(steps)
    if ep % 20 == 0:
        print(f"... ep {ep}", flush=True)

print("\nbolt |  n | succ | insert | A-collide | minB-A(med) | eplen(med)")
for i in range(10):
    s = stats[i]
    if s["n"] == 0:
        print(f"  {i}  |  0 |  -")
        continue
    mba = np.median(s["minba"]) * 100
    el = np.median(s["eplen"])
    print(f"  {i}  | {s['n']:2d} | {s['succ']:2d}/{s['n']:2d} | {s['ins']:2d}/{s['n']:2d}"
          f" | {s['col']:2d}/{s['n']:2d} | {mba:7.1f}cm | {el:5.0f}")
env.close()
