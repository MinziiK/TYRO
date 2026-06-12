"""v19 smoke test: solo 3-d action, rigid A, align servo, stall truncation,
hot-start spawn across bolts, gates."""
import numpy as np
from src.config import make_env_config
from src.env import TyroEnv

V19 = dict(
    nut_b_align_servo=True, nut_a_kinematic_freeze=True,
    nut_collision_fail=True, nut_b_solo_action=True,
    nut_arrive_lat_tol=0.015, nut_seat_lat_mult=1.0,
    nut_stall_steps=250,
)

cfg = make_env_config(3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True, nut_pure_rl=True,
    nut_b_hotstart_enable=True, nut_b_hotstart_alpha=1.0,
    nut_b_hotstart_random_bolt=True, max_steps=800, terminate_on="never",
    USE_DOMAIN_RANDOMIZATION=False, **V19)
cfg.reward.w_nut_path_waste = 5.0
env = TyroEnv(cfg=cfg, render=False, seed=0)
print(f"action_space={env.action_space.shape} obs_space={env.observation_space.shape}")
assert env.action_space.shape == (3,), "solo action dim"

# --- 1. rigid A under stepping -------------------------------------------
obs, _ = env.reset(seed=0)
qA0 = np.asarray(env._nut_frozen_qA, dtype=np.float64).copy()
for _ in range(40):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        break
qA1, _ = env.robot_A.joint_state()
drift = float(np.max(np.abs(np.asarray(qA1) - qA0)))
print(f"A joint drift over steps: {drift:.2e} rad  (rigid={'OK' if drift < 1e-3 else 'FAIL'})")

# --- 2. align servo + full insert/hold/retract with scripted axial action --
ok_servo = None
for seed in range(8):
    obs, _ = env.reset(seed=100 + seed)
    idx = int(env._nut_target_idx)
    a = env._nut_axis_unit(idx)
    lat0 = None
    max_lat = 0.0
    info = {}
    for t in range(300):
        if int(env._nut_subphase) == 1 and int(env._nut_macro_stage) >= 2:
            act = np.asarray(a, dtype=np.float32)    # retract: pull out
        else:
            act = np.asarray(-a, dtype=np.float32)   # approach/insert/hold
        obs, r, term, trunc, info = env.step(act)
        if int(env._nut_subphase) == 1:
            _, lat, _ = env._nut_axial_lateral(idx)
            max_lat = max(max_lat, float(lat))
            if lat0 is None:
                lat0 = float(lat)
        if term or trunc:
            break
    if lat0 is not None:
        ok_servo = (lat0, max_lat, info.get("termination"),
                    bool(info.get("is_success")))
        break
if ok_servo:
    print(f"servo: lat@insert-start={ok_servo[0]*100:.2f}cm maxLat={ok_servo[1]*100:.2f}cm "
          f"term={ok_servo[2]} success={ok_servo[3]}")
else:
    print("servo: insert never fired in probe")

# --- 3. stall truncation (zero action, HOME start) ------------------------
cfg2 = make_env_config(3, 1, render=False, scene_layout="fanuc_spacious",
    nut_fastening_task=True, nut_pure_rl=True,
    nut_b_hotstart_enable=True, nut_b_hotstart_alpha=0.0,
    max_steps=800, terminate_on="never",
    USE_DOMAIN_RANDOMIZATION=False, **V19)
env2 = TyroEnv(cfg=cfg2, render=False, seed=1)
obs, _ = env2.reset(seed=1)
steps = 0
info = {}
for t in range(800):
    obs, r, term, trunc, info = env2.step(np.zeros(3, dtype=np.float32))
    steps += 1
    if term or trunc:
        break
print(f"stall: terminated after {steps} steps  term={info.get('termination')} "
      f"({'OK' if info.get('termination') == 'nut_stall' and steps < 350 else 'CHECK'})")

# --- 4. hot-start spawn coverage (v19 cfg, all bolts) ----------------------
seen = {}
for s in range(80):
    env.reset(seed=200 + s)
    idx = int(env._nut_target_idx)
    axial, lat, th = env._nut_axial_lateral(idx)
    stage = env._nut_staging_axial()
    ok = lat < 0.02 and abs(axial - stage) < 0.02
    if idx not in seen or (not seen[idx][0] and ok):
        seen[idx] = (ok, lat)
    if len(seen) == 10 and all(v[0] for v in seen.values()):
        break
bad = [i for i, v in seen.items() if not v[0]]
print(f"hot-start coverage: {len(seen)}/10 bolts sampled, failures={bad or 'none'}")
env.close()
env2.close()
print("SMOKE_DONE")
