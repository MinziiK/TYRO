"""v20 smoke: INSERT axial servo reaches hub-face base (-L/2)."""
import numpy as np
from src.config import make_env_config
from src.env import TyroEnv

V20 = dict(
    nut_b_align_servo=True,
    nut_b_axial_insert_servo=True,
    nut_a_kinematic_freeze=True,
    nut_collision_fail=True,
    nut_b_solo_action=True,
    nut_arrive_lat_tol=0.015,
    nut_seat_lat_mult=1.0,
    nut_insert_depth_tol=0.007,
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
L = float(cfg.bolt_length)
target = -0.5 * L
env = TyroEnv(cfg=cfg, render=False, seed=42)
all_ok = True

for bolt in range(10):
    env.reset(seed=100 + bolt)
    env._nut_target_idx = bolt
    # Jump straight into INSERT at staging (hot-start alpha=1 already placed B).
    env._nut_subphase = 1
    env._nut_macro_stage = 0
    env._nut_macro_step = 0
    env._nut_lock_quat = env._coaxial_quat_preserving_roll(bolt)
    deepest = +9.9
    for _ in range(120):
        obs, _, term, trunc, info = env.step(
            np.zeros(cfg.action.dim, dtype=np.float32))
        ax, lat, _ = env._nut_axial_lateral(bolt)
        deepest = min(deepest, ax)
        if term or trunc:
            break
    ok = abs(deepest - target) < 0.007
    all_ok = all_ok and ok
    print(f"bolt={bolt} deepest={deepest*100:+.1f}cm target={target*100:+.1f}cm ok={ok}")

env.close()
print("PASS" if all_ok else "FAIL — see deepest values above")
