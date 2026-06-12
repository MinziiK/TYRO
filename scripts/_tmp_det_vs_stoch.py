"""Final 4M model: deterministic vs stochastic success at alpha=0.3, plus the
deterministic failure point (axial depth vs seat target when the episode ends).
"""
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from src.config import make_env_config
from src.env import TyroEnv

ckpt = sorted(Path("runs/nut_fastening_v17_purerl/ckpts").glob("ppo_*_steps.zip"),
              key=lambda pp: int(pp.stem.split("_")[1]))[-1]
model = PPO.load(str(ckpt.with_suffix("")), device="cpu")
print("model:", ckpt.name, " log_std:",
      np.round(model.policy.log_std.detach().numpy(), 2), flush=True)

cfg = make_env_config(3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True, nut_pure_rl=True,
    nut_b_hotstart_enable=True, nut_b_hotstart_alpha=0.3,
    nut_b_hotstart_random_bolt=True, max_steps=800, terminate_on="never",
    USE_DOMAIN_RANDOMIZATION=False, collision_terminates=False)
env = TyroEnv(cfg=cfg, render=False, seed=0)

for det in (True, False):
    succ = 0
    n_ep = 20
    fails = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep)
        info = {}
        for t in range(800):
            a, _ = model.predict(obs, deterministic=det)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ok = bool(info.get("is_success"))
        succ += int(ok)
        if not ok:
            idx = int(env._nut_target_idx)
            axial, lat, _ = env._nut_axial_lateral(idx)
            fails.append((idx, env._nut_subphase, axial, lat))
    print(f"deterministic={det}: success {succ}/{n_ep}", flush=True)
    for f in fails[:8]:
        print(f"   fail bolt={f[0]} sub={f[1]} axial={f[2]*100:+.1f}cm "
              f"lat={f[3]*100:.1f}cm", flush=True)
env.close()
