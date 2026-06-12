"""Verify the fixed hot-start: with alpha=1.0 + random-bolt, every bolt's
hot-start must spawn B on-axis (lateral < 2cm, |axial - staging| < 2cm).
Resets with many seeds until all 10 bolts are sampled.
"""
import numpy as np
from src.config import make_env_config
from src.env import TyroEnv

cfg = make_env_config(
    3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True, nut_pure_rl=True,
    nut_b_hotstart_enable=True, nut_b_hotstart_alpha=1.0,
    nut_b_hotstart_random_bolt=True,
    USE_DOMAIN_RANDOMIZATION=True,
)
env = TyroEnv(cfg=cfg, render=False, seed=0)
seen = {}
for s in range(120):
    env.reset(seed=s)
    idx = int(env._nut_target_idx)
    axial, lat, theta = env._nut_axial_lateral(idx)
    stage = env._nut_staging_axial()
    ok = lat < 0.02 and abs(axial - stage) < 0.02
    rec = (axial - stage, lat, np.degrees(theta), ok)
    if idx not in seen or (not seen[idx][3] and ok):
        seen[idx] = rec
    if len(seen) == 10 and all(v[3] for v in seen.values()):
        break
print("bolt | dAxial | lateral | theta | OK")
for idx in sorted(seen):
    da, lat, th, ok = seen[idx]
    print(f"  {idx}  | {da*100:+6.2f}cm | {lat*100:5.2f}cm | {th:5.1f}deg | "
          f"{'OK' if ok else 'FAIL'}")
env.close()
