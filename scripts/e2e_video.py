#!/usr/bin/env python3
"""Render a single continuous E2E video: Robot A mount → Robot B nut-fastening.

Both phases are captured headless (TinyRenderer) with one FIXED camera and the
SAME hub offset, then streamed to a single MP4 via ffmpeg. No GUI window is
opened, so this runs on a headless server and produces one continuous clip.

Note: A and B run in two env instances (different obs/action layouts), so there
is a one-frame scene swap at the hand-off — but the tire is already mounted in
the B scene at the same offset, so it reads as a continuous take.

Example
-------
    python scripts/e2e_video.py --scenario-idx 1 --seed 42 \
        --out runs/e2e_eval/e2e_scenario1.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _resolve_model_path(path: str) -> str:
    pp = Path(path)
    if pp.suffix.lower() == ".zip" and pp.exists():
        return str(pp.with_suffix(""))
    if not pp.exists() and pp.suffix.lower() == ".zip":
        bare = pp.with_suffix("")
        if bare.with_suffix(".zip").exists():
            return str(bare)
    return path


def _capture(env, width, height, view, proj):
    img = p.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER, physicsClientId=env.client,
    )
    rgba = np.reshape(np.asarray(img[2], dtype=np.uint8), (height, width, 4))
    return rgba


def _view_proj(target, width, height, dist, yaw, pitch):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target, distance=dist,
        yaw=yaw, pitch=pitch, roll=0.0, upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=55.0, aspect=float(width) / float(height), nearVal=0.05, farVal=100.0,
    )
    return view, proj


def _run_phase(env, model, *, seed, hub_offset, width, height,
               dist, yaw, pitch, ffmpeg_stdin, deterministic, every,
               banner, hold_frames):
    env.set_dr_hub_xy_offset(hub_offset)
    obs, _ = env.reset(seed=seed)
    try:
        hub = np.asarray(env.scene.hub_pose()[0], dtype=np.float64)
    except Exception:
        hub = np.array([env.cfg.tire_mount_pos[0], env.cfg.tire_mount_pos[1],
                        env.cfg.tire_mount_pos[2]], dtype=np.float64)
    target = [float(hub[0]), float(hub[1]), float(hub[2])]
    view, proj = _view_proj(target, width, height, dist, yaw, pitch)

    terminated = truncated = False
    steps = 0
    last = None
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _r, terminated, truncated, info = env.step(action)
        steps += 1
        if steps % every == 0:
            last = _capture(env, width, height, view, proj)
            ffmpeg_stdin.write(last.tobytes())
    if last is None:
        last = _capture(env, width, height, view, proj)
    for _ in range(hold_frames):
        ffmpeg_stdin.write(last.tobytes())
    print(f"  [{banner}] steps={steps}  success={info.get('is_success')}  "
          f"term={info.get('termination')}")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous E2E A→B MP4 render.")
    ap.add_argument("--model-a", default="runs/phase1_mount_v3_dr/final.zip")
    ap.add_argument("--model-b", default="runs/nut_fastening_v16_dr/final.zip")
    ap.add_argument("--scenario-idx", type=int, default=2,
                    help="1-based scenario index (default 2: headless E2E pass).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dr-range-cm", type=float, default=5.0)
    ap.add_argument("--mix-easy-prob", type=float, default=0.8)
    ap.add_argument("--a-max-steps", type=int, default=2000)
    ap.add_argument("--mount-radius-tol", type=float, default=0.55)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=2,
                    help="Capture 1 frame every N env steps (speeds render).")
    ap.add_argument("--cam-dist", type=float, default=2.6)
    ap.add_argument("--cam-yaw", type=float, default=55.0)
    ap.add_argument("--cam-pitch", type=float, default=-28.0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--out", default="runs/e2e_eval/e2e_continuous.mp4")
    args = ap.parse_args()

    dr_range_m = float(args.dr_range_cm) / 100.0
    idx = int(args.scenario_idx) - 1
    if idx < 0:
        ap.error("--scenario-idx is 1-based and must be >= 1")

    offset_rng = np.random.default_rng(args.seed)
    if dr_range_m <= 0.0:
        off = np.zeros(2, dtype=np.float64)
    else:
        offs = offset_rng.uniform(-dr_range_m, dr_range_m, size=(idx + 1, 2))
        off = offs[idx]
    norm_cm = float(np.linalg.norm(off)) * 100.0
    print(f"[vid] scenario {args.scenario_idx}  "
          f"hub=({off[0]*100:+.2f}, {off[1]*100:+.2f}) cm  |hub|={norm_cm:.2f} cm")

    model_a = PPO.load(_resolve_model_path(args.model_a), device="cpu")
    model_b = PPO.load(_resolve_model_path(args.model_b), device="cpu")

    mount_overrides = dict(
        render=False, scene_layout="fanuc_spacious", terminate_on="mount",
        max_steps=int(args.a_max_steps), USE_DOMAIN_RANDOMIZATION=True,
        RANDOM_POSITION_RANGE=dr_range_m, DR_CARGO_ENABLE=False,
        planner_pos_offset_scale=0.06, mount_radius_tol=float(args.mount_radius_tol),
        contact_force_terminate_above=0.0, start_pos_curriculum_enable=True,
        include_hub_guide_obs=True,
    )
    nut_overrides = dict(
        render=False, scene_layout="fanuc_spacious", nut_fastening_task=True,
        nut_b_planner_residual=True, terminate_on="never", max_steps=2000,
        USE_DOMAIN_RANDOMIZATION=True, RANDOM_POSITION_RANGE=dr_range_m,
        DR_CARGO_ENABLE=False, nut_a_hold_jitter_rad=float(np.deg2rad(6.0)),
        contact_force_terminate_above=0.0, collision_terminates=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", f"{args.width}x{args.height}",
        "-framerate", str(args.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out_path),
    ]
    print(f"[vid] encoding → {out_path}")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    det = not args.stochastic

    cfg_a = make_env_config(stage=3, phase=1, **mount_overrides)
    env_a = TyroEnv(cfg=cfg_a, render=False, seed=args.seed)
    env_a.set_start_pos_easy_prob(float(args.mix_easy_prob))
    print("[vid] Phase A: Robot A mount")
    _run_phase(
        env_a, model_a, seed=args.seed + idx * 2, hub_offset=off,
        width=args.width, height=args.height, dist=args.cam_dist,
        yaw=args.cam_yaw, pitch=args.cam_pitch, ffmpeg_stdin=proc.stdin,
        deterministic=det, every=int(args.every), banner="A",
        hold_frames=int(args.fps),  # ~1s freeze on mounted tire
    )
    env_a.close()

    cfg_b = make_env_config(stage=3, phase=1, **nut_overrides)
    env_b = TyroEnv(cfg=cfg_b, render=False, seed=args.seed + idx * 2 + 1)
    print("[vid] Phase B: Robot B nut-fastening")
    _run_phase(
        env_b, model_b, seed=args.seed + idx * 2 + 1, hub_offset=off,
        width=args.width, height=args.height, dist=args.cam_dist,
        yaw=args.cam_yaw, pitch=args.cam_pitch, ffmpeg_stdin=proc.stdin,
        deterministic=det, every=int(args.every), banner="B",
        hold_frames=int(args.fps),
    )
    env_b.close()

    proc.stdin.close()
    proc.wait()
    print(f"[vid] DONE → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
