"""Benchmark hotstart IK cache: cold (full search) vs warm (cached refine)."""
import time, numpy as np
from src.config import make_env_config
from src.env import TyroEnv

V20 = dict(
    nut_b_align_servo=True, nut_b_axial_insert_servo=True,
    nut_a_kinematic_freeze=True, nut_collision_fail=True,
    nut_b_solo_action=True, nut_arrive_lat_tol=0.015,
    nut_seat_lat_mult=1.0, nut_insert_depth_tol=0.007, nut_stall_steps=250,
)
cfg = make_env_config(
    3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True, nut_pure_rl=True,
    nut_b_hotstart_enable=True, nut_b_hotstart_alpha=1.0,
    nut_b_hotstart_random_bolt=False, max_steps=800,
    terminate_on="never", USE_DOMAIN_RANDOMIZATION=False,
    nut_a_hold_jitter_rad=0.0, contact_force_terminate_above=0.0,
    collision_terminates=False, **V20,
)
cfg.nut_per_leg_episode = True
cfg.reward.w_nut_path_waste = 5.0
cfg.reward.w_nut_joint_vel = 0.06
adim = int(cfg.action.dim)

env = TyroEnv(cfg=cfg, render=False, seed=0)
# prime scipy disk cache + URDF parse
env.reset(seed=0)
for _ in range(10):
    env.step(np.zeros(adim, dtype=np.float32))

cold, warm = [], []
for bolt in range(10):
    env._nut_hotstart_ik_cache = {}  # force cold for this bolt
    t0 = time.perf_counter()
    env.reset(seed=1000 + bolt)
    cold.append(time.perf_counter() - t0)
    for rep in range(8):
        t0 = time.perf_counter()
        env.reset(seed=2000 + bolt * 10 + rep)
        warm.append(time.perf_counter() - t0)
env.close()

c = float(np.mean(cold))
w = float(np.mean(warm))
step_t = 0.03314  # from prior profile (stable)
print(f"cold reset (full IK search): {c*1000:.0f} ms  (n={len(cold)})")
print(f"warm reset (cached refine):  {w*1000:.0f} ms  (n={len(warm)})")
print(f"reset speedup warm/cold:     {c/w:.2f}x")
for L in (100,):
    old = c + L * step_t
    new = w + L * step_t
    print(f"L={L}: episode {old*1000:.0f}ms -> {new*1000:.0f}ms  "
          f"({old/new:.2f}x)  implied fps {L/old:.1f} -> {L/new:.1f}")
