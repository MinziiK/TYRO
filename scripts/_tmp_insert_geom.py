"""(A) How deep does the socket wrap the bolt during INSERT? Track the socket
tool_tip axial depth vs the bolt (base=-L/2, tip=+L/2) at the deepest point.
(B) Reproduce the viewer's set_dr_hub_xy_offset(None) to see if it breaks the
hot-start that headless (without it) gets right."""
import glob, numpy as np
from stable_baselines3 import PPO
from src.config import make_env_config
from src.env import TyroEnv

V19=dict(nut_b_align_servo=True,nut_a_kinematic_freeze=True,nut_collision_fail=True,
         nut_b_solo_action=True,nut_arrive_lat_tol=0.015,nut_seat_lat_mult=1.0)
ck=sorted(glob.glob('runs/nut_fastening_v19_stage1/ckpts/ppo_*_steps.zip'),
          key=lambda p:int(p.split('ppo_')[1].split('_')[0]))[-1]
m=PPO.load(ck,device='cpu')
print('ckpt',ck)
L=None

def run(call_setdr, tag):
    cfg=make_env_config(3,1,render=False,scene_layout='fanuc_spacious',
        nut_fastening_task=True,nut_pure_rl=True,nut_b_hotstart_enable=True,
        nut_b_hotstart_alpha=1.0,nut_b_hotstart_random_bolt=True,max_steps=800,
        terminate_on='never',USE_DOMAIN_RANDOMIZATION=False,
        nut_a_hold_jitter_rad=0.0,contact_force_terminate_above=0.0,
        collision_terminates=False,**V19)
    cfg.nut_per_leg_episode=True
    env=TyroEnv(cfg=cfg,render=False,seed=42)
    global L
    L=float(cfg.bolt_length)
    res=[]
    for ep in range(6):
        if call_setdr:
            env.set_dr_hub_xy_offset(None)
        obs,_=env.reset(seed=42+ep)
        idx=int(env._nut_target_idx)
        deepest=+9.9; sub1=False
        info={}
        for t in range(800):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(a)
            if int(env._nut_subphase)==1:
                sub1=True
                ax,lat,_=env._nut_axial_lateral(idx)
                deepest=min(deepest,ax)   # most negative = deepest into base
            if term or trunc: break
        res.append((idx,info.get('termination'),info.get('is_success'),sub1,deepest))
        print(f"  [{tag}] ep{ep} bolt={idx} term={info.get('termination')} "
              f"ok={info.get('is_success')} insert={sub1} "
              f"deepest_axial={deepest*100:+.1f}cm (base={-0.5*L*100:.1f}cm tip={+0.5*L*100:.1f}cm)")
    env.close()
    nok=sum(1 for r in res if r[2])
    print(f"  [{tag}] success {nok}/6\n")

print("--- WITHOUT set_dr (headless style) ---")
run(False,"plain")
print("--- WITH set_dr_hub_xy_offset(None) each reset (viewer style) ---")
run(True,"viewer")
print(f"bolt_length L={L*100:.1f}cm  -> full wrap = socket tip reaches axial {-0.5*L*100:.1f}cm")
