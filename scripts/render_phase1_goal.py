"""Render the Phase-1 ideal goal pose as a static snapshot.

Phase 1 (stage=1) trains **Robot A (UR10) only**: carry the tire onto the hub
flange so it is coaxial and seated. Robot B (Panda) is inactive — kept at home.

This script sets the *visual answer key* without RL:

  1. Reset env (both robots spawn; tire grasped by UR10).
  2. Remove the grasp constraint; Panda → ``reset_to_home()``.
  3. Teleport tire centre to hub centre, axis = hub axis (mounted on flange).
  4. Snap UR10 joints via IK so the EE sits on the tire (static pose).
  5. Save PNG and/or hold the GUI open.

Usage (repo root, ``conda activate tyro``)::

    python scripts/render_phase1_goal.py --out runs/phase1_goal.png
    python scripts/render_phase1_goal.py --render
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s])
    if m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return np.array([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s])
    if m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return np.array([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s])
    s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return np.array([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s])


def _orientation_with_axis(z_axis: np.ndarray, x_hint: np.ndarray | None = None) -> np.ndarray:
    """Quaternion (xyzw): local +Z along ``z_axis``, +X near ``x_hint``."""
    z = np.asarray(z_axis, dtype=np.float64)
    z /= np.linalg.norm(z) + 1e-12
    hint = np.array([0.0, 0.0, 1.0]) if x_hint is None else np.asarray(x_hint, dtype=np.float64)
    x = hint - z * np.dot(hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return _rotmat_to_quat_xyzw(np.column_stack([x, y, z]))


def _remove_grasp(env: TyroEnv) -> None:
    if env._grasp_constraint is not None:
        try:
            p.removeConstraint(env._grasp_constraint, physicsClientId=env.client)
        except p.error:
            pass
        env._grasp_constraint = None


def _snap_robot_ee(robot, target_pos: np.ndarray,
                   target_orn: np.ndarray | None, client: int) -> float:
    """IK + ``resetJointState`` — static EE pose. Returns IK residual (m).

    ``target_orn`` may be ``None`` to let IK pick any wrist orientation.
    """
    ik_kwargs = dict(
        lowerLimits=robot.arm.lower.tolist(),
        upperLimits=robot.arm.upper.tolist(),
        jointRanges=robot.arm.range.tolist(),
        restPoses=robot.arm.rest.tolist(),
        maxNumIterations=200,
        residualThreshold=1e-4,
        physicsClientId=client,
    )
    if target_orn is None:
        ik = p.calculateInverseKinematics(
            robot.uid, robot.EE_LINK_INDEX, list(target_pos), **ik_kwargs,
        )
    else:
        ik = p.calculateInverseKinematics(
            robot.uid, robot.EE_LINK_INDEX,
            list(target_pos), list(target_orn), **ik_kwargs,
        )
    ik = np.asarray(ik, dtype=np.float64)
    max_slot = max(robot._ik_arm_slots) if robot._ik_arm_slots else -1
    if len(ik) <= max_slot:
        targets = robot.arm.rest.copy()
    else:
        targets = np.clip(ik[robot._ik_arm_slots], robot.arm.lower, robot.arm.upper)
    for jidx, q in zip(robot.arm.indices, targets):
        p.resetJointState(
            robot.uid, jidx, float(q), targetVelocity=0.0, physicsClientId=client,
        )
    achieved, _ = robot.ee_pose()
    return float(np.linalg.norm(achieved - target_pos))


def _mount_tire_on_hub(env: TyroEnv) -> tuple[np.ndarray, np.ndarray]:
    hub_pos, hub_orn = env.scene.hub_pose()
    hub_axis = env.scene.hub_axis()
    # Same flange normal as hub; tire spawn uses rpy (0, -pi/2, 0) ≡ hub_base_rpy.
    tire_orn = _orientation_with_axis(hub_axis)
    p.resetBasePositionAndOrientation(
        env.handles.tire,
        list(hub_pos),
        list(tire_orn),
        physicsClientId=env.client,
    )
    return np.asarray(hub_pos), tire_orn


def _lift_tire_off_rack(env: TyroEnv, lift_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Pickup-instant pose — gripper has just descended into the rack gap,
    fixed-joint attached, and lifted the tire ``lift_height`` straight up.

    Reproduces the exact spawn orientation defined by ``cfg.tire_spawn_rpy``
    (default (0, π/2, 0) → bore axis = world +X, facing robot A).
    """
    pickup_xy = np.asarray(env.cfg.tire_pickup_pos, dtype=np.float64)
    tire_pos = pickup_xy.copy()
    tire_pos[2] += float(lift_height)
    tire_orn = np.asarray(
        p.getQuaternionFromEuler(list(env.cfg.tire_spawn_rpy)),
        dtype=np.float64,
    )
    p.resetBasePositionAndOrientation(
        env.handles.tire,
        list(tire_pos),
        list(tire_orn),
        physicsClientId=env.client,
    )
    return tire_pos, tire_orn


def _render(env: TyroEnv, view, proj, w: int, h: int) -> np.ndarray:
    _, _, rgba, _, _ = p.getCameraImage(
        w, h, view, proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=env.client,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[..., :3]


def _save(rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(out_path, rgb)
    except ImportError:
        from PIL import Image
        Image.fromarray(rgb).save(out_path)


def _snapshot(
    env: TyroEnv,
    focus: np.ndarray,
    out_path: Path,
    w: int,
    h: int,
    *,
    eye_offset: tuple[float, float, float] = (0.20, -2.6, 0.70),
    fov: float = 62.0,
    top_eye_z: float = 4.5,
    top_fov: float = 68.0,
) -> None:
    """Render side + top snapshots framed on ``focus`` (world XYZ).

    ``focus`` is typically the hub centre for the mounted goal, or the
    tire-pickup point for the rack/start/lift views. Camera framing is
    relative to ``focus`` so the rack scene at X = −1.50 is fully in
    frame instead of being lost off to the left of a hub-centric view.
    """
    eye = [focus[0] + eye_offset[0], focus[1] + eye_offset[1], focus[2] + eye_offset[2]]
    view = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=[focus[0], focus[1] + 0.10, focus[2] + 0.05],
        cameraUpVector=[0, 0, 1],
    )
    proj = p.computeProjectionMatrixFOV(fov, w / h, 0.05, 12.0)
    _save(_render(env, view, proj, w, h), out_path)

    # Top-down: cameraUpVector = world +Y so the rotated cargo reads as a
    # long horizontal block at the top of the frame with the robots lined
    # up underneath. ``focus`` re-centres the view on whatever scene we
    # care about (pickup zone vs hub-mount zone).
    top_out = out_path.with_name(out_path.stem + "_top" + out_path.suffix)
    top_view = p.computeViewMatrix(
        cameraEyePosition=[focus[0], focus[1] - 0.20, focus[2] + top_eye_z],
        cameraTargetPosition=[focus[0], focus[1] - 0.20, focus[2]],
        cameraUpVector=[0.0, 1.0, 0.0],
    )
    top_proj = p.computeProjectionMatrixFOV(top_fov, w / h, 0.05, 12.0)
    _save(_render(env, top_view, top_proj, w, h), top_out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/phase1_goal.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument(
        "--render",
        action="store_true",
        help="Open PyBullet GUI and hold until the window is closed.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = make_env_config(stage=1, phase=1, render=args.render)
    env = TyroEnv(cfg=cfg, render=args.render, seed=args.seed)
    env.reset(seed=args.seed)

    out_path = (_REPO_ROOT / args.out).resolve()
    start_out = out_path.with_name(out_path.stem + "_start" + out_path.suffix)
    mounted_out = out_path.with_name(out_path.stem + "_mounted" + out_path.suffix)

    R = float(cfg.tire_outer_radius)
    pickup_pos = np.asarray(cfg.tire_pickup_pos, dtype=np.float64)
    rack_center = pickup_pos.copy()
    rack_center[2] = float(cfg.tire_rack_inner_center[2]) + 0.4  # mid frame

    # Camera tuned for the Y-split rack + bore=+X tire (Robot B-centric
    # world): the two rails run along X (30 cm each), flanking the
    # tire in Y with a 30 cm gap on the Y=0 centreline (Panda base
    # line). Viewing from +X with a slight −Y bias keeps the bore
    # opening visible (the tire faces robot A in +X) and frames both
    # rails as parallel rectangles passing underneath the tire on
    # either side of the Y-gap.
    pickup_eye_offset = (1.55, -0.60, 0.55)
    pickup_fov = 56.0

    # ------------------------------------------------------------------
    # Phase 1 FSM start snapshot — tire seated on the dual-block rack,
    # UR10 at HOME, no grasp yet. Camera framed on the pickup zone.
    # ------------------------------------------------------------------
    tire_pos0, _ = env.scene.tire_pose()
    ee_pos0, _ = env.robot_A.ee_pose()
    grasp_target0 = np.asarray(tire_pos0) + np.array([0.0, 0.0, -R])
    print(
        f"[phase1-fsm-start] tire pickup pos = "
        f"{tuple(round(float(v), 4) for v in tire_pos0)}  "
        f"UR10 EE @ HOME = {tuple(round(float(v), 4) for v in ee_pos0)}  "
        f"d_EE_to_grasp_target = "
        f"{float(np.linalg.norm(np.asarray(ee_pos0) - grasp_target0)) * 100:.1f} cm"
    )
    rail_top_z = float(cfg.tire_rack_inner_center[2]) + float(
        cfg.tire_rack_half_extents[2]
    )
    print(
        f"[phase1-fsm-start] rack: inner rail (truck side) center "
        f"{cfg.tire_rack_inner_center}, outer rail (robot side) center "
        f"{cfg.tire_rack_outer_center}, gap Y = [-0.15, +0.15] (30cm), "
        f"rail top plane Z = {rail_top_z:.3f}"
    )
    _snapshot(
        env, rack_center, start_out, args.width, args.height,
        eye_offset=pickup_eye_offset, fov=pickup_fov,
        top_eye_z=3.2, top_fov=58.0,
    )
    print(f"[phase1-fsm-start] saved: {start_out}")

    # ------------------------------------------------------------------
    # Phase 1 FSM GOAL snapshot — pickup / lift instant.
    # UR10 gripper has plunged down through the 30 cm-wide Y-gap between
    # the two short parallel rails (tire bore points at robot A in +X),
    # hooked under the tire's 6 o'clock outer point with a JOINT_FIXED,
    # and lifted the tire ``lift_height`` straight up. The rails remain
    # in frame just below the lifted tire so the "wide corridor entry"
    # intent reads clearly.
    # ------------------------------------------------------------------
    LIFT_H = 0.10  # 10 cm above the rack-rest pose
    _, ur10_grasp_orn = env.robot_A.ee_pose()
    env._release_world_pin()
    _remove_grasp(env)
    tire_pos, _tire_orn = _lift_tire_off_rack(env, lift_height=LIFT_H)
    ur10_target_pos = tire_pos + np.array([0.0, 0.0, -R])
    ik_res = _snap_robot_ee(
        env.robot_A, ur10_target_pos, ur10_grasp_orn, env.client,
    )
    env.robot_B.reset_to_home()
    achieved_ee, _ = env.robot_A.ee_pose()
    ee_above_rack_top = float(achieved_ee[2] - rail_top_z)
    print(
        f"[phase1-goal-lift] tire COM = "
        f"{tuple(round(float(v), 4) for v in tire_pos)}  "
        f"(lift = {LIFT_H * 100:.0f} cm above rack rest)"
    )
    print(
        f"[phase1-goal-lift] UR10 EE   = "
        f"{tuple(round(float(v), 4) for v in achieved_ee)}  "
        f"IK residual = {ik_res * 100:.2f} cm  "
        f"EE Z {ee_above_rack_top * 100:+.1f} cm above rail-top plane"
    )
    _snapshot(
        env, tire_pos, out_path, args.width, args.height,
        eye_offset=pickup_eye_offset, fov=pickup_fov,
        top_eye_z=3.2, top_fov=58.0,
    )
    print(f"[phase1-goal-lift] saved: {out_path}")
    print(
        f"[phase1-goal-lift] saved: "
        f"{out_path.with_name(out_path.stem + '_top' + out_path.suffix)}"
    )

    # ------------------------------------------------------------------
    # Auxiliary: mid-cycle mounted goal (Stage 1 → 2 transition) saved
    # next to the main pickup goal so we still have a visual reference
    # for the carry / mount sub-task.
    # ------------------------------------------------------------------
    tire_pos_m, _ = _mount_tire_on_hub(env)
    hub_pos, _ = env.scene.hub_pose()
    ur10_target_m = tire_pos_m + np.array([0.0, 0.0, -R])
    ik_res_m = _snap_robot_ee(
        env.robot_A, ur10_target_m, ur10_grasp_orn, env.client,
    )
    env.robot_B.reset_to_home()
    panda_res = 0.0
    d_A = float(np.linalg.norm(tire_pos_m - hub_pos))
    achieved_ee_m, _ = env.robot_A.ee_pose()
    ee_below_center = float(tire_pos_m[2] - achieved_ee_m[2])
    print(
        f"[phase1-mounted]   tire-hub d_A = {d_A * 100:.3f} cm  "
        f"UR10 IK residual = {ik_res_m * 100:.2f} cm  "
        f"EE below tire centre = {ee_below_center * 100:.2f} cm "
        f"(expect ~{R * 100:.1f} cm)"
    )
    panda_ee, _ = env.robot_B.ee_pose()
    ur10_base = np.asarray(cfg.robot_A_base_pos, dtype=np.float64)
    panda_base = np.asarray(cfg.robot_B_base_pos, dtype=np.float64)
    vehicle_center = np.asarray(cfg.vehicle_center_world, dtype=np.float64)
    vehicle_he = np.asarray(cfg.vehicle_half_extents, dtype=np.float64)
    cargo_rpy = np.asarray(
        getattr(cfg, "vehicle_base_rpy", (0.0, 0.0, 0.0)), dtype=np.float64,
    )
    Rc = np.array(
        p.getMatrixFromQuaternion(list(p.getQuaternionFromEuler(cargo_rpy.tolist()))),
        dtype=np.float64,
    ).reshape(3, 3)
    world_he = np.abs(Rc) @ vehicle_he
    cargo_min = vehicle_center - world_he
    cargo_max = vehicle_center + world_he
    print(
        f"[phase1-mounted]   Panda EE parked at = "
        f"{np.round(panda_ee, 3).tolist()}  (IK residual {panda_res * 100:.2f} cm)  "
        f"cargo X [{cargo_min[0]:.2f}, {cargo_max[0]:.2f}], "
        f"Y [{cargo_min[1]:.2f}, {cargo_max[1]:.2f}], "
        f"Z [{cargo_min[2]:.2f}, {cargo_max[2]:.2f}]"
    )
    print(
        f"[phase1-mounted]   layout: UR10 base X={ur10_base[0]:.2f} Y={ur10_base[1]:.2f}, "
        f"Panda X={panda_base[0]:.2f} Y={panda_base[1]:.2f}, "
        f"hub Y={hub_pos[1]:.2f}, hub-robot Y margin "
        f"{(hub_pos[1] - ur10_base[1]) * 100:.0f} cm"
    )
    _snapshot(env, hub_pos, mounted_out, args.width, args.height)
    print(f"[phase1-mounted]   saved: {mounted_out}")

    if args.render:
        print("[phase1-goal] GUI open — close the PyBullet window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
