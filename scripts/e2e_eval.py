#!/usr/bin/env python3
"""Headless E2E eval: Robot A mount + Robot B nut-fastening under hub DR.

For each scenario a hub XY offset is sampled once (reproducible via ``--seed``),
then:
  1. Robot A mount policy runs with that offset (``terminate_on=mount``).
  2. Robot B nut policy runs with the *same* offset (tire pre-seated on hub).

E2E success = A mount success AND B 10/10 success.

Examples
--------
    python scripts/e2e_eval.py --scenarios 100
    python scripts/e2e_eval.py --scenarios 10 --dr-range-cm 0
    python scripts/e2e_eval.py --scenarios 5 --render
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _resolve_model_path(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".zip" and p.exists():
        return str(p.with_suffix(""))
    if not p.exists() and p.suffix.lower() == ".zip":
        bare = p.with_suffix("")
        if bare.with_suffix(".zip").exists():
            return str(bare)
    return path


def _run_policy(
    env: TyroEnv,
    model: PPO,
    *,
    seed: int,
    hub_offset: np.ndarray,
    deterministic: bool = True,
) -> dict:
    env.set_dr_hub_xy_offset(hub_offset)
    obs, _info = env.reset(seed=seed)
    total_r = 0.0
    steps = 0
    terminated = truncated = False
    last_info: dict = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, last_info = env.step(action)
        total_r += float(r)
        steps += 1
    rt = last_info.get("reward_terms") or {}
    return {
        "success": bool(last_info.get("is_success", False)),
        "termination": str(last_info.get("termination", "unknown")),
        "steps": int(steps),
        "reward": float(total_r),
        "n_fastened": int(last_info.get("n_fastened", rt.get("n_fastened", 0))),
        "n_fastened_policy": int(rt.get("n_fastened_policy", 0)),
    }


@dataclass
class ScenarioResult:
    scenario: int
    seed: int
    hub_offset_x_m: float
    hub_offset_y_m: float
    hub_offset_norm_cm: float
    a_success: bool
    a_steps: int
    a_termination: str
    a_reward: float
    b_success: bool
    b_steps: int
    b_termination: str
    b_reward: float
    b_n_fastened: int
    e2e_success: bool


def _nut_overrides_v16(*, render: bool, dr_range_m: float, max_steps: int) -> dict:
    """Legacy planner+residual nut stack (v16_dr checkpoints)."""
    return dict(
        render=render,
        scene_layout="fanuc_spacious",
        nut_fastening_task=True,
        nut_b_planner_residual=True,
        terminate_on="never",
        max_steps=max_steps,
        USE_DOMAIN_RANDOMIZATION=True,
        RANDOM_POSITION_RANGE=dr_range_m,
        DR_CARGO_ENABLE=False,
        nut_a_hold_jitter_rad=float(np.deg2rad(6.0)),
        contact_force_terminate_above=0.0,
        collision_terminates=False,
    )


def _nut_overrides_v23(*, render: bool, dr_range_m: float, max_steps: int) -> dict:
    """v23 pure-RL clean-branch + approach-seed IK (matches training)."""
    return dict(
        render=render,
        scene_layout="fanuc_spacious",
        nut_fastening_task=True,
        nut_pure_rl=True,
        nut_b_planner_residual=False,
        nut_b_hotstart_enable=True,
        nut_b_hotstart_alpha=0.0,
        nut_b_hotstart_random_bolt=False,
        nut_per_leg_episode=False,
        nut_b_align_servo=True,
        nut_a_kinematic_freeze=True,
        nut_collision_fail=True,
        nut_b_solo_action=True,
        nut_arrive_lat_tol=0.015,
        nut_seat_lat_mult=1.0,
        nut_b_axial_insert_servo=True,
        nut_insert_depth_tol=0.007,
        nut_b_insert_branch_search=True,
        nut_b_clean_branch_insert=True,
        nut_clean_approach_seed=True,
        nut_clean_seat_cache="",
        nut_a_hold_jitter_rad=float(np.deg2rad(6.0)),
        nut_stall_steps=0,
        terminate_on="never",
        max_steps=max_steps,
        USE_DOMAIN_RANDOMIZATION=True,
        RANDOM_POSITION_RANGE=dr_range_m,
        DR_CARGO_ENABLE=False,
        contact_force_terminate_above=0.0,
        collision_terminates=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E eval: A mount + B nut under hub DR.")
    ap.add_argument(
        "--model-a",
        default="runs/phase1_mount_v3_dr/final.zip",
        help="Robot A mount checkpoint (.zip).",
    )
    ap.add_argument(
        "--model-b",
        default=None,
        help="Robot B nut-fastening checkpoint (.zip). Default: v16_dr or v23_dr.",
    )
    ap.add_argument(
        "--v23",
        action="store_true",
        help=(
            "Use v23 pure-RL clean-branch B wiring (implies longer horizon, "
            "solo 3-d action, approach-seed IK). Default model-b: "
            "runs/nut_fastening_v23_dr/final.zip."
        ),
    )
    ap.add_argument(
        "--b-max-steps",
        type=int,
        default=None,
        help="Robot B episode horizon (default: 2000 v16, 6000 v23 chain).",
    )
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42,
                    help="Master seed for scenario offset sampling.")
    ap.add_argument(
        "--dr-range-cm",
        type=float,
        default=5.0,
        help="Half-width of uniform hub XY offset sampling (metres = cm/100).",
    )
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument(
        "--mix-easy-prob",
        type=float,
        default=0.8,
        help="Robot A start-pose easy mix (matches v3_dr end-of-training schedule).",
    )
    ap.add_argument(
        "--a-max-steps",
        type=int,
        default=2000,
        help="Robot A episode horizon (mount from easy start needs >600).",
    )
    ap.add_argument(
        "--mount-radius-tol",
        type=float,
        default=0.55,
        help="Mount gate radius (m); v3_dr training eval used soft 0.55.",
    )
    ap.add_argument(
        "--only",
        choices=("a", "b"),
        default=None,
        help=(
            "View only one robot (for side-by-side GUI: launch one process "
            "with --only a and another with --only b). Skips result saving."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default="runs/e2e_eval",
        help="Directory for JSON/CSV results.",
    )
    args = ap.parse_args()

    dr_range_m = float(args.dr_range_cm) / 100.0
    n = int(args.scenarios)
    if n <= 0:
        ap.error("--scenarios must be positive")

    if args.model_b is None:
        args.model_b = (
            "runs/nut_fastening_v23_dr/final.zip" if args.v23
            else "runs/nut_fastening_v16_dr/final.zip"
        )
    b_max_steps = int(args.b_max_steps) if args.b_max_steps is not None else (
        6000 if args.v23 else 2000
    )

    model_a_path = _resolve_model_path(args.model_a)
    model_b_path = None if args.only == "a" else _resolve_model_path(args.model_b)
    print(f"[e2e] loading A: {model_a_path}")
    model_a = PPO.load(model_a_path, device="cpu")
    if model_b_path is not None:
        print(f"[e2e] loading B: {model_b_path}")
        model_b = PPO.load(model_b_path, device="cpu")
    else:
        model_b = None
    print(
        f"[e2e] A layout obs={model_a.observation_space.shape[0]} "
        f"act={model_a.action_space.shape[0]}"
    )
    if model_b is not None:
        print(
            f"[e2e] B layout obs={model_b.observation_space.shape[0]} "
            f"act={model_b.action_space.shape[0]}"
        )

    print(
        f"[e2e] B stack: {'v23 pure-RL' if args.v23 else 'v16 planner+residual'}  "
        f"max_steps={b_max_steps}"
    )

    mount_overrides = dict(
        render=args.render,
        scene_layout="fanuc_spacious",
        terminate_on="mount",
        max_steps=int(args.a_max_steps),
        USE_DOMAIN_RANDOMIZATION=True,
        RANDOM_POSITION_RANGE=dr_range_m,
        DR_CARGO_ENABLE=False,
        planner_pos_offset_scale=0.06,
        mount_radius_tol=float(args.mount_radius_tol),
        contact_force_terminate_above=0.0,
        start_pos_curriculum_enable=True,
        include_hub_guide_obs=True,
    )
    if args.v23:
        nut_overrides = _nut_overrides_v23(
            render=args.render, dr_range_m=dr_range_m, max_steps=b_max_steps,
        )
    else:
        nut_overrides = _nut_overrides_v16(
            render=args.render, dr_range_m=dr_range_m, max_steps=b_max_steps,
        )

    print(
        f"[e2e] DR hub range: ±{args.dr_range_cm:.1f} cm  "
        f"scenarios={n}  seed={args.seed}"
    )

    offset_rng = np.random.default_rng(args.seed)
    if dr_range_m <= 0.0:
        offsets = np.zeros((n, 2), dtype=np.float64)
    else:
        offsets = offset_rng.uniform(-dr_range_m, dr_range_m, size=(n, 2))

    t0 = time.time()
    det = not args.stochastic

    cfg_a = make_env_config(stage=3, phase=1, **mount_overrides)
    cfg_b = make_env_config(stage=3, phase=1, **nut_overrides)

    # Side-by-side GUI: one process shows only A, another only B. Each process
    # owns a separate PyBullet GUI window, so two windows run in parallel.
    if args.only is not None:
        which = args.only
        print(f"[e2e] VIEW-ONLY mode: Robot {which.upper()} ({n} scenarios)")
        if which == "a":
            env = TyroEnv(cfg=cfg_a, render=args.render, seed=args.seed)
            env.set_start_pos_easy_prob(float(args.mix_easy_prob))
            model = model_a
        else:
            env = TyroEnv(cfg=cfg_b, render=args.render, seed=args.seed + 1)
            model = model_b
        for i in range(n):
            off = offsets[i]
            norm_cm = float(np.linalg.norm(off)) * 100.0
            ep_seed = args.seed + i * 2 + (0 if which == "a" else 1)
            print(
                f"\n[e2e] scenario {i + 1}/{n}  "
                f"hub=({off[0]*100:+.2f}, {off[1]*100:+.2f}) cm  "
                f"|hub|={norm_cm:.2f} cm"
            )
            r = _run_policy(
                env, model, seed=ep_seed, hub_offset=off, deterministic=det,
            )
            if which == "a":
                print(
                    f"  A: success={r['success']}  steps={r['steps']}  "
                    f"term={r['termination']}"
                )
            else:
                print(
                    f"  B: success={r['success']}  steps={r['steps']}  "
                    f"n_fastened={r['n_fastened']}/10  term={r['termination']}"
                )
        env.close()
        print(f"\n[e2e] VIEW-ONLY {which.upper()} done (no results saved).")
        return 0

    results: list[ScenarioResult] = []

    if args.render:
        # PyBullet allows only ONE in-process GUI connection. Per scenario:
        # open A window → mount → close → open B window → fasten → close.
        env_a = env_b = None
        print(f"[e2e] A start pose: mix easy_prob={args.mix_easy_prob}")
        print(f"[e2e] GUI mode: A→B per scenario (one window at a time)")
        for i in range(n):
            off = offsets[i]
            norm_cm = float(np.linalg.norm(off)) * 100.0
            print(
                f"\n[e2e] scenario {i + 1}/{n}  "
                f"hub=({off[0]*100:+.2f}, {off[1]*100:+.2f}) cm  "
                f"|hub|={norm_cm:.2f} cm"
            )
            env_a = TyroEnv(cfg=cfg_a, render=True, seed=args.seed + i * 2)
            env_a.set_start_pos_easy_prob(float(args.mix_easy_prob))
            a = _run_policy(
                env_a, model_a, seed=args.seed + i * 2,
                hub_offset=off, deterministic=det,
            )
            env_a.close()
            env_a = None
            print(
                f"  A: success={a['success']}  steps={a['steps']}  "
                f"term={a['termination']}"
            )

            env_b = TyroEnv(cfg=cfg_b, render=True, seed=args.seed + i * 2 + 1)
            b = _run_policy(
                env_b, model_b, seed=args.seed + i * 2 + 1,
                hub_offset=off, deterministic=det,
            )
            env_b.close()
            env_b = None
            e2e_ok = bool(a["success"] and b["success"])
            print(
                f"  B: success={b['success']}  steps={b['steps']}  "
                f"n_fastened={b['n_fastened']}/10  term={b['termination']}"
            )
            print(f"  E2E: {'PASS' if e2e_ok else 'FAIL'}")

            results.append(ScenarioResult(
                scenario=i,
                seed=args.seed + i * 2,
                hub_offset_x_m=float(off[0]),
                hub_offset_y_m=float(off[1]),
                hub_offset_norm_cm=norm_cm,
                a_success=bool(a["success"]),
                a_steps=int(a["steps"]),
                a_termination=str(a["termination"]),
                a_reward=float(a["reward"]),
                b_success=bool(b["success"]),
                b_steps=int(b["steps"]),
                b_termination=str(b["termination"]),
                b_reward=float(b["reward"]),
                b_n_fastened=int(b["n_fastened"]),
                e2e_success=e2e_ok,
            ))
    else:
        # Headless: keep both envs open; A→B per scenario (true E2E order).
        env_a = TyroEnv(cfg=cfg_a, render=False, seed=args.seed)
        env_b = TyroEnv(cfg=cfg_b, render=False, seed=args.seed + 1)
        env_a.set_start_pos_easy_prob(float(args.mix_easy_prob))
        print(f"[e2e] A start pose: mix easy_prob={args.mix_easy_prob}")
        for i in range(n):
            off = offsets[i]
            norm_cm = float(np.linalg.norm(off)) * 100.0
            print(
                f"\n[e2e] scenario {i + 1}/{n}  "
                f"hub=({off[0]*100:+.2f}, {off[1]*100:+.2f}) cm  "
                f"|hub|={norm_cm:.2f} cm"
            )
            a = _run_policy(
                env_a, model_a, seed=args.seed + i * 2,
                hub_offset=off, deterministic=det,
            )
            print(
                f"  A: success={a['success']}  steps={a['steps']}  "
                f"term={a['termination']}"
            )
            b = _run_policy(
                env_b, model_b, seed=args.seed + i * 2 + 1,
                hub_offset=off, deterministic=det,
            )
            e2e_ok = bool(a["success"] and b["success"])
            print(
                f"  B: success={b['success']}  steps={b['steps']}  "
                f"n_fastened={b['n_fastened']}/10  term={b['termination']}"
            )
            print(f"  E2E: {'PASS' if e2e_ok else 'FAIL'}")

            results.append(ScenarioResult(
                scenario=i,
                seed=args.seed + i * 2,
                hub_offset_x_m=float(off[0]),
                hub_offset_y_m=float(off[1]),
                hub_offset_norm_cm=norm_cm,
                a_success=bool(a["success"]),
                a_steps=int(a["steps"]),
                a_termination=str(a["termination"]),
                a_reward=float(a["reward"]),
                b_success=bool(b["success"]),
                b_steps=int(b["steps"]),
                b_termination=str(b["termination"]),
                b_reward=float(b["reward"]),
                b_n_fastened=int(b["n_fastened"]),
                e2e_success=e2e_ok,
            ))
        env_a.close()
        env_b.close()

    elapsed = time.time() - t0
    n_a = sum(r.a_success for r in results)
    n_b = sum(r.b_success for r in results)
    n_e2e = sum(r.e2e_success for r in results)
    b_nf = [r.b_n_fastened for r in results]

    print("\n=== E2E eval summary ===")
    print(f"  scenarios:     {n}")
    print(f"  dr range:      ±{args.dr_range_cm:.1f} cm")
    print(f"  A success:     {100.0 * n_a / n:.1f}%  ({n_a}/{n})")
    print(f"  B success:     {100.0 * n_b / n:.1f}%  ({n_b}/{n})")
    print(f"  E2E success:   {100.0 * n_e2e / n:.1f}%  ({n_e2e}/{n})")
    print(
        f"  B n_fastened:  mean={statistics.mean(b_nf):.2f}  "
        f"min={min(b_nf)}  max={max(b_nf)}"
    )
    print(f"  wall time:     {elapsed / 60.0:.1f} min")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"e2e_{n}sc_{int(args.dr_range_cm)}cm_{stamp}"
    json_path = out_dir / f"{tag}.json"
    csv_path = out_dir / f"{tag}.csv"

    payload = {
        "meta": {
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),
            "b_stack": "v23" if args.v23 else "v16",
            "b_max_steps": b_max_steps,
            "scenarios": n,
            "seed": args.seed,
            "dr_range_cm": args.dr_range_cm,
            "deterministic": det,
            "elapsed_s": elapsed,
            "a_success_rate": n_a / n,
            "b_success_rate": n_b / n,
            "e2e_success_rate": n_e2e / n,
        },
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(asdict(results[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\n[e2e] saved {json_path}")
    print(f"[e2e] saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
