"""Profile reset() vs step() cost in the v20 nut config (single env, headless).
Tells us the ceiling of any reset optimization at the realistic episode length."""
import time, numpy as np
from src.config import make_env_config
from src.env import TyroEnv

V20 = dict(
    nut_b_align_servo=True, nut_b_axial_insert_servo=True,
    nut_a_kinematic_freeze=True, nut_collision_fail=True,
    nut_b_solo_action=True, nut_arrive_lat_tol=0.015,
    nut_seat_lat_mult=1.0, nut_insert_depth_tol=0.007,
    nut_stall_steps=250,
)
cfg = make_env_config(
    3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True, nut_pure_rl=True,
    nut_b_hotstart_enable=True, nut_b_hotstart_alpha=1.0,
    nut_b_hotstart_random_bolt=True, max_steps=800,
    terminate_on="never", USE_DOMAIN_RANDOMIZATION=False,
    nut_a_hold_jitter_rad=0.0, contact_force_terminate_above=0.0,
    collision_terminates=False, **V20,
)
cfg.nut_per_leg_episode = True
cfg.reward.w_nut_path_waste = 5.0
cfg.reward.w_nut_joint_vel = 0.06

env = TyroEnv(cfg=cfg, render=False, seed=0)
adim = int(cfg.action.dim)

# warm caches (scipy IK seed disk cache, first URDF parse, etc.)
env.reset(seed=0)
for _ in range(20):
    env.step(np.zeros(adim, dtype=np.float32))

# --- time resets ---
N_RESET = 25
t0 = time.perf_counter()
for i in range(N_RESET):
    env.reset(seed=100 + i)
reset_t = (time.perf_counter() - t0) / N_RESET

# --- time steps (no reset; long horizon hold) ---
env.reset(seed=7)
N_STEP = 400
a = np.zeros(adim, dtype=np.float32)
t0 = time.perf_counter()
for _ in range(N_STEP):
    obs, r, term, trunc, info = env.step(a)
    if term or trunc:
        env.reset(seed=7)
step_t = (time.perf_counter() - t0) / N_STEP
env.close()

for L in (50, 100, 150):
    epis_t = reset_t + L * step_t
    frac = reset_t / epis_t
    eff_fps = L / epis_t
    print(f"L={L:3d}: reset={reset_t*1000:6.1f}ms  {L}*step={L*step_t*1000:7.1f}ms "
          f" reset_frac={frac*100:4.1f}%  single-env fps={eff_fps:5.1f}")
print(f"\nreset={reset_t*1000:.1f}ms/ea   step={step_t*1000:.2f}ms/ea")
print(f"theoretical max speedup if reset->0 at L=100: "
      f"{(reset_t+100*step_t)/(100*step_t):.2f}x")
