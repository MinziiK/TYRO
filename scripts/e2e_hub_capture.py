#!/usr/bin/env python3
"""Capture hub front-view PNGs (before/after) for E2E / DR-sweep scenarios.

Modes (``--mode``):

  * **dr-sweep** (default) — representative spread across the full ±DR box
  * **far-offset** — max|axis| ≥ 4 cm (+ E2E passes ≥ 3 cm), sorted farthest first
  * **e2e-success** — before+after for E2E-passing rows only

Rendering uses PyBullet TinyRenderer (headless). Camera sits on the
outboard side (bore axis / robot-bank side, −Y in fanuc_spacious), not
behind the cargo (+Y).

Examples
--------
    # All 100 rows, DR coverage (default):
    python scripts/e2e_hub_capture.py \\
        --from-json runs/e2e_eval/e2e_100sc_5cm_20260615_184724.json

    # Far-offset set → new folder:
    python scripts/e2e_hub_capture.py --from-json ... --mode far-offset \\
        --out-dir runs/e2e_eval/hub_captures_far

    # E2E passes only:
    python scripts/e2e_hub_capture.py --from-json ... --mode e2e-success

    # Single scenario smoke test:
    python scripts/e2e_hub_capture.py --from-json ... --scenario-ids 0 --v24
"""
from __future__ import annotations

import argparse
import json
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
from scripts.e2e_eval import (  # noqa: E402
    _nut_overrides_v24,
    _resolve_model_path,
)


def _load_all_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("results", data))


def _load_success_rows(path: Path) -> list[dict]:
    return [r for r in _load_all_rows(path) if bool(r.get("e2e_success", False))]


def _max_axis_cm(row: dict) -> float:
    return max(
        abs(float(row["hub_offset_x_m"])),
        abs(float(row["hub_offset_y_m"])),
    ) * 100.0


def _select_dr_sweep_rows(rows: list[dict]) -> list[dict]:
    """Pick scenarios spanning the full ±DR box (incl. near ±5 cm corners).

    E2E-success rows alone cluster around 2–4 cm because corner cases fail
    more often — this set is for visualising hub DR coverage.
    """
    if not rows:
        return []
    targets = (0.5, 2.0, 3.0, 4.0, 4.5, 4.8)
    picked: list[dict] = []
    used: set[int] = set()

    def pick_nearest(target_cm: float) -> None:
        best = min(rows, key=lambda r: abs(_max_axis_cm(r) - target_cm))
        sc = int(best["scenario"])
        if sc not in used:
            picked.append(best)
            used.add(sc)

    for t in targets:
        pick_nearest(t)

    # Global max-|axis| row (typically ~4.9 cm in a ±5 cm box sample).
    hardest = max(rows, key=_max_axis_cm)
    sc = int(hardest["scenario"])
    if sc not in used:
        picked.append(hardest)
        used.add(sc)

    # Four quadrant corners (largest x+y in each sign combo).
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        corner = max(
            rows,
            key=lambda r: sx * float(r["hub_offset_x_m"]) + sy * float(r["hub_offset_y_m"]),
        )
        sc = int(corner["scenario"])
        if sc not in used:
            picked.append(corner)
            used.add(sc)

    return sorted(picked, key=lambda r: int(r["scenario"]))


def _select_far_offset_rows(
    rows: list[dict],
    *,
    min_cm: float = 4.0,
    e2e_min_cm: float = 3.0,
) -> list[dict]:
    """Far DR scenarios: max|axis| ≥ ``min_cm``, plus E2E passes ≥ ``e2e_min_cm``."""
    picked: list[dict] = []
    seen: set[int] = set()
    for row in sorted(rows, key=lambda r: -_max_axis_cm(r)):
        sc = int(row["scenario"])
        if sc in seen:
            continue
        far = _max_axis_cm(row) >= float(min_cm)
        e2e_far = bool(row.get("e2e_success")) and _max_axis_cm(row) >= float(e2e_min_cm)
        if far or e2e_far:
            picked.append(row)
            seen.add(sc)
    return picked


def _set_all_bolt_colors(env: TyroEnv, rgba: tuple[float, float, float, float]) -> None:
    if env.handles is None:
        return
    cl = env.client
    for bref in env.handles.bolts:
        try:
            p.changeVisualShape(
                bref.uid, bref.link_index,
                rgbaColor=list(rgba), physicsClientId=cl,
            )
        except p.error:
            pass


def _hub_focus(env: TyroEnv) -> np.ndarray:
    try:
        hub, _ = env.scene.hub_pose()
        return np.asarray(hub, dtype=np.float64).reshape(3)
    except Exception:
        return np.asarray(env.cfg.tire_mount_pos, dtype=np.float64).reshape(3)


def _rotation_quat(from_vec: np.ndarray, to_vec: np.ndarray) -> list[float]:
    """Quaternion (x,y,z,w) rotating ``from_vec`` onto ``to_vec``."""
    a = np.asarray(from_vec, dtype=np.float64).reshape(3)
    b = np.asarray(to_vec, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    a = a / na
    b = b / nb
    cross = np.cross(a, b)
    w = 1.0 + float(np.dot(a, b))
    if w < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    q = np.array([cross[0], cross[1], cross[2], w], dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _add_axis_cylinder(
    env: TyroEnv,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    rgba: tuple[float, float, float, float],
    *,
    radius: float,
) -> int:
    """Visual-only cylinder segment along ``direction`` (headless-safe)."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        d = d / dn
    half = float(length) * 0.5
    pos = np.asarray(origin, dtype=np.float64).reshape(3) + d * half
    orn = _rotation_quat(np.array([0.0, 0.0, 1.0], dtype=np.float64), d)
    vis = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(length),
        rgbaColor=list(rgba),
        physicsClientId=env.client,
    )
    return int(p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=vis,
        basePosition=[float(pos[0]), float(pos[1]), float(pos[2])],
        baseOrientation=orn,
        physicsClientId=env.client,
    ))



def _hub_bore_axis(env: TyroEnv) -> np.ndarray:
    bore = np.asarray(env.cfg.hub_axis_world, dtype=np.float64).reshape(3)
    try:
        live = np.asarray(env.scene.hub_axis(), dtype=np.float64).reshape(3)
        if float(np.dot(live, bore)) < 0.0:
            live = -live
        if float(np.linalg.norm(live)) > 1e-9:
            bore = live
    except Exception:
        pass
    bn = float(np.linalg.norm(bore))
    if bn < 1e-9:
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return bore / bn


def _world_origin() -> np.ndarray:
    return np.zeros(3, dtype=np.float64)


def _draw_capture_axes(
    env: TyroEnv,
    *,
    axis_length: float,
) -> None:
    """World-origin RGB triad (+X red, +Y green, +Z blue)."""
    if getattr(env, "_capture_axes_drawn", False):
        return
    axes = (
        ([1.0, 0.0, 0.0, 1.0], np.array([1.0, 0.0, 0.0])),
        ([0.0, 1.0, 0.0, 1.0], np.array([0.0, 1.0, 0.0])),
        ([0.0, 0.0, 1.0, 1.0], np.array([0.0, 0.0, 1.0])),
    )
    uids: list[int] = []
    origin = _world_origin()
    for rgba, direc in axes:
        uids.append(_add_axis_cylinder(
            env, origin, direc, float(axis_length), rgba, radius=0.010,
        ))
    env._capture_axis_uids = uids
    env._capture_axes_drawn = True


def _hide_robot_visuals(env: TyroEnv) -> None:
    """Make Robot A/B meshes invisible for a clean hub-only frame."""
    cl = env.client
    for robot in (env.robot_A, env.robot_B):
        if robot is None:
            continue
        uid = int(robot.uid)
        n = int(p.getNumJoints(uid, physicsClientId=cl))
        for link_idx in range(-1, n):
            try:
                p.changeVisualShape(
                    uid, link_idx,
                    rgbaColor=[0.0, 0.0, 0.0, 0.0],
                    physicsClientId=cl,
                )
            except p.error:
                pass


def _annotate_png(rgb: np.ndarray, *, hub_pos: np.ndarray) -> np.ndarray:
    """Burn hub world XY (metres) into the frame — large text only."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    hub = np.asarray(hub_pos, dtype=np.float64).reshape(3)
    text = f"x={hub[0]*100:+.1f}cm  y={hub[1]*100:+.1f}cm"
    font = None
    for path, size in (
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 48),
    ):
        try:
            font = ImageFont.truetype(path, size)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=48)
        except TypeError:
            font = ImageFont.load_default()
    draw.text((24, 24), text, fill=(255, 220, 0), font=font)
    return np.asarray(im)


def _hub_front_view(
    env: TyroEnv,
    *,
    width: int,
    height: int,
    dist: float,
    fov: float,
    yaw_offset: float,
    pitch: float,
) -> tuple[list[float], list[float]]:
    """Outboard (−Y) view; camera target = hub centre (axes stay at world origin)."""
    hub = np.asarray(_hub_focus(env), dtype=np.float64).reshape(3)
    bore = _hub_bore_axis(env)
    eye = hub + bore * float(dist)
    eye[2] = float(hub[2])
    target = hub.copy()
    if abs(yaw_offset) > 1e-6 or abs(pitch) > 1e-6:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[float(target[0]), float(target[1]), float(target[2])],
            distance=float(dist),
            yaw=float(yaw_offset),
            pitch=float(pitch),
            roll=0.0,
            upAxisIndex=2,
        )
    else:
        view = p.computeViewMatrix(
            cameraEyePosition=[float(eye[0]), float(eye[1]), float(eye[2])],
            cameraTargetPosition=[float(target[0]), float(target[1]), float(target[2])],
            cameraUpVector=[0.0, 0.0, 1.0],
        )
    proj = p.computeProjectionMatrixFOV(
        fov=float(fov),
        aspect=float(width) / float(height),
        nearVal=0.05,
        farVal=100.0,
    )
    return view, proj


def _capture_png(
    env: TyroEnv,
    out_path: Path,
    *,
    width: int,
    height: int,
    dist: float,
    fov: float,
    yaw: float,
    pitch: float,
    draw_axes: bool,
    axis_length: float,
    hide_robots: bool = True,
) -> None:
    if draw_axes:
        _draw_capture_axes(env, axis_length=float(axis_length))
    if hide_robots:
        _hide_robot_visuals(env)
    view, proj = _hub_front_view(
        env, width=width, height=height, dist=dist, fov=fov,
        yaw_offset=yaw, pitch=pitch,
    )
    img = p.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER, physicsClientId=env.client,
    )
    rgba = np.reshape(np.asarray(img[2], dtype=np.uint8), (height, width, 4))
    rgb = rgba[..., :3]
    rgb = _annotate_png(rgb, hub_pos=_hub_focus(env))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(out_path, rgb)
    except ImportError:
        from PIL import Image
        Image.fromarray(rgb).save(out_path)


def _run_policy(
    env: TyroEnv,
    model: PPO,
    *,
    seed: int,
    hub_offset: np.ndarray,
    deterministic: bool,
) -> dict:
    env.set_dr_hub_xy_offset(hub_offset)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    last_info: dict = {}
    steps = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _r, terminated, truncated, last_info = env.step(action)
        steps += 1
    return {
        "success": bool(last_info.get("is_success", False)),
        "termination": str(last_info.get("termination", "unknown")),
        "steps": int(steps),
        "n_fastened": int(last_info.get("n_fastened", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hub front before/after PNGs for successful E2E scenarios.",
    )
    ap.add_argument(
        "--mode",
        choices=("dr-sweep", "far-offset", "e2e-success"),
        default="dr-sweep",
        help="dr-sweep | far-offset (≥4 cm) | e2e-success.",
    )
    ap.add_argument(
        "--from-json", required=True,
        help="e2e_eval JSON.",
    )
    ap.add_argument(
        "--out-dir", default="runs/e2e_eval/hub_captures_far",
        help="Output root (scenario subfolders created here).",
    )
    ap.add_argument(
        "--scenario-ids",
        default=None,
        help="Comma-separated scenario indices to capture (default: all passes).",
    )
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--v24", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Master seed (must match the eval JSON meta.seed).")
    ap.add_argument("--dr-range-cm", type=float, default=5.0)
    ap.add_argument("--b-max-steps", type=int, default=2500)
    ap.add_argument("--mount-radius-tol", type=float, default=0.55)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--cam-dist", type=float, default=1.35,
                    help="Camera distance from hub centre (m); smaller = tighter crop.")
    ap.add_argument("--cam-fov", type=float, default=37.0,
                    help="Vertical FOV (deg); smaller = more zoom.")
    ap.add_argument("--cam-yaw", type=float, default=0.0,
                    help="Yaw offset (deg); 0 = hub-face normal, level at hub z.")
    ap.add_argument("--cam-pitch", type=float, default=0.0,
                    help="Pitch offset (deg); 0 = horizontal (eye z = hub z).")
    ap.add_argument("--axis-length", type=float, default=0.28,
                    help="World-origin RGB axis length (m).")
    ap.add_argument("--no-axes", action="store_true",
                    help="Skip 3-D axis geometry (overlay text remains).")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    json_path = Path(args.from_json)
    if not json_path.is_file():
        ap.error(f"JSON not found: {json_path}")

    rows_all = _load_all_rows(json_path)
    if args.mode == "e2e-success":
        rows = _load_success_rows(json_path)
    elif args.mode == "far-offset":
        rows = _select_far_offset_rows(rows_all)
    else:
        rows = _select_dr_sweep_rows(rows_all)
    if args.scenario_ids:
        want = {int(x.strip()) for x in args.scenario_ids.split(",") if x.strip()}
        rows = [r for r in rows if int(r["scenario"]) in want]
    if not rows:
        print("[capture] no matching scenarios")
        return 1

    print(
        f"[capture] mode={args.mode}  selected={len(rows)}  "
        f"max|axis| cm: "
        f"{min(_max_axis_cm(r) for r in rows):.1f}–"
        f"{max(_max_axis_cm(r) for r in rows):.1f}"
    )

    meta = json.loads(json_path.read_text(encoding="utf-8")).get("meta", {})
    if args.model_b is None:
        args.model_b = meta.get(
            "model_b",
            "runs/nut_fastening_v24_dr_stageB3/ckpts/ppo_1749440_steps.zip"
            if args.v24 else "runs/nut_fastening_v16_dr/final.zip",
        )

    dr_range_m = float(args.dr_range_cm) / 100.0
    det = not args.stochastic
    model_b = PPO.load(_resolve_model_path(args.model_b), device="cpu")

    mount_overrides = dict(
        render=False, scene_layout="fanuc_spacious", terminate_on="mount",
        max_steps=2000, USE_DOMAIN_RANDOMIZATION=True,
        RANDOM_POSITION_RANGE=dr_range_m, DR_CARGO_ENABLE=False,
        planner_pos_offset_scale=0.06,
        mount_radius_tol=float(args.mount_radius_tol),
        mount_seat_glide_steps=10, contact_force_terminate_above=0.0,
        start_pos_curriculum_enable=True, include_hub_guide_obs=True,
        carry_tire_rigid_sync=True, attached_spawn_when_easy=False,
    )
    nut_overrides = _nut_overrides_v24(
        render=False, dr_range_m=dr_range_m, max_steps=int(args.b_max_steps),
    )

    out_root = Path(args.out_dir)
    print(f"[capture] {len(rows)} scenario(s) → {out_root}")

    for row in rows:
        sc = int(row["scenario"])
        a_seed = int(row.get("seed", args.seed + sc * 2))
        b_seed = a_seed + 1
        off = np.array([
            float(row["hub_offset_x_m"]),
            float(row["hub_offset_y_m"]),
        ], dtype=np.float64)
        sc_dir = out_root / f"scenario_{sc:03d}"
        before_path = sc_dir / "before_mount.png"
        after_path = sc_dir / "after_fastened.png"
        meta_path = sc_dir / "meta.json"

        print(
            f"\n[capture] scenario {sc}  hub=({off[0]*100:+.2f}, {off[1]*100:+.2f}) cm"
        )

        cfg_a = make_env_config(stage=3, phase=1, **mount_overrides)
        env_a = TyroEnv(cfg=cfg_a, render=False, seed=a_seed)
        env_a.set_start_pos_easy_prob(0.0)
        env_a.set_dr_hub_xy_offset(off)
        env_a.reset(seed=a_seed)
        _set_all_bolt_colors(env_a, (0.7, 0.7, 0.7, 1.0))
        _capture_png(
            env_a, before_path,
            width=int(args.width), height=int(args.height),
            dist=float(args.cam_dist), fov=float(args.cam_fov),
            yaw=float(args.cam_yaw), pitch=float(args.cam_pitch),
            draw_axes=not args.no_axes,
            axis_length=float(args.axis_length),
        )
        env_a.close()
        print(f"  before → {before_path}")

        e2e_ok = bool(row.get("e2e_success", False))
        if args.mode in ("dr-sweep", "far-offset") and not e2e_ok:
            print("  after  → skipped (E2E fail at this offset)")
            meta_path.write_text(json.dumps({
                "scenario": sc,
                "a_seed": a_seed,
                "hub_offset_m": [float(off[0]), float(off[1])],
                "hub_offset_norm_cm": float(row.get("hub_offset_norm_cm", 0)),
                "max_axis_cm": _max_axis_cm(row),
                "e2e_success": False,
                "source_json": str(json_path),
            }, indent=2), encoding="utf-8")
            continue

        cfg_b = make_env_config(stage=3, phase=1, **nut_overrides)
        env_b = TyroEnv(cfg=cfg_b, render=False, seed=b_seed)
        b = _run_policy(
            env_b, model_b, seed=b_seed, hub_offset=off, deterministic=det,
        )
        if not b["success"]:
            print(
                f"  WARN: B re-run failed ({b['termination']}, "
                f"{b['n_fastened']}/10), after capture skipped"
            )
            env_b.close()
            continue
        _set_all_bolt_colors(env_b, (0.10, 0.80, 0.25, 1.0))
        _capture_png(
            env_b, after_path,
            width=int(args.width), height=int(args.height),
            dist=float(args.cam_dist), fov=float(args.cam_fov),
            yaw=float(args.cam_yaw), pitch=float(args.cam_pitch),
            draw_axes=not args.no_axes,
            axis_length=float(args.axis_length),
        )
        env_b.close()
        print(f"  after  → {after_path}")

        meta_path.write_text(json.dumps({
            "scenario": sc,
            "a_seed": a_seed,
            "b_seed": b_seed,
            "hub_offset_m": [float(off[0]), float(off[1])],
            "hub_offset_norm_cm": float(row.get("hub_offset_norm_cm", 0)),
            "max_axis_cm": _max_axis_cm(row),
            "e2e_success": e2e_ok,
            "b_steps": b["steps"],
            "source_json": str(json_path),
        }, indent=2), encoding="utf-8")

    print(f"\n[capture] done — {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
