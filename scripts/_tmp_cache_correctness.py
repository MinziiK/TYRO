"""Warm-cache hotstart still reaches staging for each bolt."""
import numpy as np
from src.config import make_env_config
from src.env import TyroEnv

V20 = dict(
    nut_b_align_servo=True, nut_b_axial_insert_servo=True,
    nut_a_kinematic_freeze=True, nut_collision_fail=True,
    nut_b_solo_action=True, nut_arrive_lat_tol=0.015,
    nut_seat_lat_mult=1.0, nut_insert_depth_tol=0.007,
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
env = TyroEnv(cfg=cfg, render=False, seed=0)
# cold pass — populate cache
for bolt in range(10):
    env.reset(seed=bolt)
    idx = int(env._nut_target_idx)
    ax, lat, _ = env._nut_axial_lateral(idx)
    print(f"cold bolt={idx} lat={lat*100:.1f}cm")
# warm pass — must match
ok = True
for bolt in range(10):
    env.reset(seed=100 + bolt)
    idx = int(env._nut_target_idx)
    ax, lat, _ = env._nut_axial_lateral(idx)
    good = lat < 0.04
    ok = ok and good
    print(f"warm bolt={idx} lat={lat*100:.1f}cm ok={good}")
env.close()
print("PASS" if ok else "FAIL")
