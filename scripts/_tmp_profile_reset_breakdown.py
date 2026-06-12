"""Break down where reset()'s ~1.6s goes: resetSimulation, Scene.build,
robot URDF loads, settle steps, hotstart IK."""
import time, numpy as np, pybullet as p
from src.config import make_env_config
from src.env import TyroEnv
from src.env.scene import Scene
from src.env.robots import make_robot_a, make_robot_b

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
    nut_b_hotstart_random_bolt=True, max_steps=800,
    terminate_on="never", USE_DOMAIN_RANDOMIZATION=False,
    nut_a_hold_jitter_rad=0.0, contact_force_terminate_above=0.0,
    collision_terminates=False, **V20,
)
cfg.nut_per_leg_episode = True
env = TyroEnv(cfg=cfg, render=False, seed=0)
env.reset(seed=0)  # warm caches
cl = env.client
acc = {}
N = 12
for i in range(N):
    rng = env._np_random
    t = time.perf_counter(); p.resetSimulation(physicsClientId=cl); acc['resetSim']=acc.get('resetSim',0)+time.perf_counter()-t
    p.setGravity(*cfg.gravity, physicsClientId=cl)
    p.setTimeStep(1.0/cfg.sim_freq_hz, physicsClientId=cl)
    t = time.perf_counter()
    sc = Scene(cl, cfg, rng, hub_xy_offset=(0,0), cargo_xy_offset=(0,0))
    h = sc.build()
    acc['scene.build']=acc.get('scene.build',0)+time.perf_counter()-t
    t = time.perf_counter(); ra = make_robot_a(cl, cfg); acc['robot_A']=acc.get('robot_A',0)+time.perf_counter()-t
    t = time.perf_counter(); rb = make_robot_b(cl, cfg); acc['robot_B']=acc.get('robot_B',0)+time.perf_counter()-t
    t = time.perf_counter()
    for _ in range(5): p.stepSimulation(physicsClientId=cl)
    acc['settle5']=acc.get('settle5',0)+time.perf_counter()-t

# full reset (includes hotstart IK + everything) for reference
t = time.perf_counter()
for i in range(N): env.reset(seed=200+i)
full = (time.perf_counter()-t)/N
env.close()

print("per-reset breakdown (ms):")
tot=0
for k,v in acc.items():
    print(f"  {k:14s} {v/N*1000:7.1f}"); tot+=v/N*1000
print(f"  {'(subtotal)':14s} {tot:7.1f}")
print(f"  {'FULL reset()':14s} {full*1000:7.1f}  (remainder = hotstart IK + state setup: {full*1000-tot:.1f})")
