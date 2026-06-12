"""Diagnose viewer behavior: why no insert? why huge joint motion?"""
import numpy as np
from stable_baselines3 import PPO
from src.config import make_env_config
from src.env import TyroEnv

V19 = dict(
    nut_b_align_servo=True, nut_a_kinematic_freeze=True,
    nut_collision_fail=True, nut_b_solo_action=True,
    nut_arrive_lat_tol=0.015, nut_seat_lat_mult=1.0,
    nut_stall_steps=250,
)
ckpt = "runs/nut_fastening_v19_stage1/ckpts/ppo_1649472_steps.zip"
model = PPO.load(ckpt, device="cpu")

for alpha, det in [(0.3, True), (0.3, False), (1.0, True)]:
    cfg = make_env_config(3, 1, render=False, scene_layout="fanuc_spacious",
        nut_fastening_task=True, nut_pure_rl=True,
        nut_b_hotstart_enable=True, nut_b_hotstart_alpha=alpha,
        nut_b_hotstart_random_bolt=True, max_steps=800, terminate_on="never",
        USE_DOMAIN_RANDOMIZATION=False, **V19)
    cfg.reward.w_nut_path_waste = 5.0
    env = TyroEnv(cfg=cfg, render=False, seed=0)
    n_arr = n_ins = n_fast = n_col = 0
    sub1_steps = []
    for ep in range(8):
        obs, _ = env.reset(seed=ep)
        idx = int(env._nut_target_idx)
        saw_sub1 = False
        info = {}
        dq_max = 0.0
        q0, _ = env.robot_B.joint_state()
        for t in range(800):
            a, _ = model.predict(obs, deterministic=det)
            obs, r, term, trunc, info = env.step(a)
            q, dq = env.robot_B.joint_state()
            dq_max = max(dq_max, float(np.max(np.abs(dq))))
            if int(env._nut_subphase) == 1:
                saw_sub1 = True
            if term or trunc:
                break
        n_arr += int(info.get("termination") != "nut_stall" and saw_sub1)
        n_ins += int(saw_sub1)
        n_fast += int(bool(info.get("is_success")))
        n_col += int(info.get("termination") == "nut_collision")
        if saw_sub1:
            sub1_steps.append(t + 1)
        axial, lat, th = env._nut_axial_lateral(idx)
        stage = env._nut_staging_axial()
        print(f"  ep{ep} bolt={idx} alpha={alpha} det={det} "
              f"sub1={saw_sub1} term={info.get('termination')} ok={info.get('is_success')} "
              f"steps={t+1} end_ax={axial*100:+.1f}cm lat={lat*100:.1f}cm "
              f"theta={np.degrees(th):.1f}deg dq_max={dq_max:.2f}")
    print(f"SUM alpha={alpha} det={det}: insert_seen={n_ins}/8 fasten={n_fast}/8 "
          f"collision={n_col}/8 sub1_dur_med={np.median(sub1_steps) if sub1_steps else 0}\n")
    env.close()
