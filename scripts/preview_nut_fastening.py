#!/usr/bin/env python3
"""GUI preview of the Robot-B nut-fastening training setup.

Shows what B will learn against:
  * Robot A frozen at the mount-hold pose (tire bonded on hub)
  * Robot B (UR10e + nut-runner) seating on bolts 0..9 sequentially

Modes:
  setup   — static initial scene (A holding, bolt 0 highlighted yellow)
  oracle  — roll-free IK demo: B visits each bolt, holds 12 steps, bolts turn green
  zero    — untrained zero-policy rollout (shows the learning starting point)

Headless server (noVNC on display :2):
  DISPLAY=:2 python -m scripts.preview_nut_fastening --mode oracle

Local desktop:
  python -m scripts.preview_nut_fastening --mode setup
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


def _quat_axis(quat, axis: str = "z") -> np.ndarray:
    R = np.asarray(p.getMatrixFromQuaternion(list(quat)), float).reshape(3, 3)
    return R[:, {"x": 0, "y": 1, "z": 2}[axis]]


def _quat_from_z_roll(z: np.ndarray, roll: float) -> list[float]:
    z = np.asarray(z, float) / max(np.linalg.norm(z), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = np.cross(ref, z)
    x0 /= max(np.linalg.norm(x0), 1e-9)
    y0 = np.cross(z, x0)
    cr, sr = np.cos(roll), np.sin(roll)
    xr, yr = cr * x0 + sr * y0, -sr * x0 + cr * y0
    m = np.column_stack([xr, yr, z])
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        zz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        zz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        zz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        zz = 0.25 * s
    return [x, y, zz, w]


def _best_b_ik(env: TyroEnv, target: np.ndarray, want_z: np.ndarray,
               seed_key: int = 0) -> np.ndarray:
    """Roll-free IK: place tool_tip at ``target`` with tool +Z = ``want_z``."""
    robot = env.robot_B
    want_z = np.asarray(want_z, float) / max(float(np.linalg.norm(want_z)), 1e-9)
    target = np.asarray(target, dtype=np.float64)
    lo, hi = robot.arm.lower, robot.arm.upper
    rng = np.random.default_rng(11 + seed_key)
    best_q, best_cost = None, 1e9
    for ri in range(16):
        quat = _quat_from_z_roll(want_z, 2 * np.pi * ri / 16)
        for k in range(4):
            seed = (robot.arm.rest if k == 0 else rng.uniform(lo, hi)).tolist()
            ik = p.calculateInverseKinematics(
                robot.uid, robot.EE_LINK_INDEX,
                target.tolist(), quat,
                lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
                jointRanges=robot.arm.range.tolist(), restPoses=seed,
                maxNumIterations=400, residualThreshold=1e-6,
                physicsClientId=env.client,
            )
            q = np.clip(np.asarray(ik, float)[robot._ik_arm_slots], lo, hi)
            st = p.saveState(physicsClientId=env.client)
            for s, qq in zip(robot.arm.indices, q):
                p.resetJointState(robot.uid, int(s), float(qq),
                                  physicsClientId=env.client)
            ee, eq = robot.ee_pose()
            dp = float(np.linalg.norm(np.asarray(ee, float) - target))
            got_z = _quat_axis(eq, "z")
            ang = float(np.degrees(np.arccos(np.clip(np.dot(got_z, want_z), -1, 1))))
            p.restoreState(st, physicsClientId=env.client)
            p.removeState(st, physicsClientId=env.client)
            cost = dp + 0.02 * ang
            if cost < best_cost:
                best_cost, best_q = cost, q
        if best_cost < 0.01:
            break
    assert best_q is not None
    return best_q


def _teleport_b(env: TyroEnv, q: np.ndarray) -> None:
    robot = env.robot_B
    for s, qq in zip(robot.arm.indices, q):
        p.resetJointState(robot.uid, int(s), float(qq), targetVelocity=0.0,
                          physicsClientId=env.client)
    # ``apply_delta_ee`` accumulates Δpos onto ``last_target_pos`` (a pure
    # math accumulator seeded at HOME-EE). After teleporting the joints we
    # must re-seed it to the new EE pos, else the next zero-Δ step yanks the
    # arm back toward HOME and the gate hold never accumulates.
    robot.last_target_pos = robot.ee_pose()[0].copy()


def _frame_camera(env: TyroEnv) -> None:
    hub, _ = env.scene.hub_pose()
    hub = np.asarray(hub, dtype=np.float64)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2, cameraYaw=55, cameraPitch=-18,
        cameraTargetPosition=hub.tolist(), physicsClientId=env.client,
    )


def _zero_action(env: TyroEnv) -> np.ndarray:
    return np.zeros(int(env.cfg.action.dim), dtype=np.float32)


def _hold_at(env: TyroEnv, q: np.ndarray, n_steps: int, step_sleep: float,
             stop_on_fasten_change: bool = False):
    """Hold B teleported at ``q`` and advance the nut FSM ``n_steps`` times.

    Demo-only: we re-assert the seated joints and read the gate metrics
    *immediately after the teleport* (so a hub-face contact can't drift the
    tool before the gate sees it), advance the FSM directly, then run a
    single light physics tick purely so the GUI redraws. Returns
    ``(done, trunc, info)`` analogous to ``step`` (done set on all_fastened).
    """
    n_before_f = len(env._nut_fastened)
    sub_before = env._nut_subphase
    info = {}
    for _ in range(int(n_steps)):
        _teleport_b(env, q)
        # Keep the bonded tire clamped (step() normally does this).
        if env._mount_seated_pos is not None:
            p.resetBasePositionAndOrientation(
                env.handles.tire, env._mount_seated_pos.tolist(),
                env._mount_seated_orn.tolist(), physicsClientId=env.client,
            )
        events = env._advance_nut_fastening()
        info = dict(events)
        p.stepSimulation(physicsClientId=env.client)
        if step_sleep > 0:
            time.sleep(step_sleep)
        if events.get("all_fastened"):
            info["termination"] = "all_fastened"
            return True, False, info
        if stop_on_fasten_change and (
            len(env._nut_fastened) != n_before_f
            or env._nut_subphase != sub_before
        ):
            break
    return False, False, info


def _run_oracle(env: TyroEnv, hold_steps: int, step_sleep: float) -> None:
    n = len(env.handles.bolts)
    hold_need = int(hold_steps)
    L = float(env.cfg.bolt_length)
    standoff = float(getattr(env.cfg, "nut_insert_standoff", 0.05))
    retract_clear = float(getattr(env.cfg, "nut_retract_clear", 0.03))
    print(f"[preview] oracle insertion-retract: {n} bolts, "
          f"hold={hold_need} steps ({hold_need * 0.05:.1f}s), "
          f"bolt_len={L*100:.0f}cm")
    for i in range(n):
        env.robot_B.reset_to_home()
        bp = np.asarray(env.scene.bolt_pose(i)[0], dtype=np.float64)
        a = np.asarray(env.scene.bolt_axis(i), dtype=np.float64)
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        want_z = -a  # tool +Z points INTO the bolt (the +Y entry direction)

        # Stage poses along the bolt axis:
        approach_pos = bp + a * (0.5 * L + standoff)   # outside the tip
        insert_pos = bp - a * (0.5 * L)                # hub-face base (seated)
        # Aim the retract target a margin BEYOND the gate threshold
        # (L/2 + clear) so IK jitter can't leave it just short of clearing.
        retract_pos = bp + a * (0.5 * L + retract_clear + 0.03)

        # APPROACH — stage on-axis just outside the stud tip.
        q_app = _best_b_ik(env, approach_pos, want_z, seed_key=i)
        _hold_at(env, q_app, 4, step_sleep)
        ax, lat, th = env._nut_axial_lateral(i)
        print(f"  bolt {i:2d} APPROACH  axial={ax*100:+5.1f}cm "
              f"lat={lat*100:.1f}cm ang={np.degrees(th):.1f}deg")

        # INSERT&HOLD — drive +Y down the axis to the base and dwell.
        q_ins = _best_b_ik(env, insert_pos, want_z, seed_key=i)
        ax, lat, th = (lambda: (_teleport_b(env, q_ins),
                                env._nut_axial_lateral(i))[1])()
        print(f"  bolt {i:2d} INSERT    axial={ax*100:+5.1f}cm "
              f"lat={lat*100:.1f}cm ang={np.degrees(th):.1f}deg "
              f"(base target={-50*L:+.1f}cm)")
        done, trunc, info = _hold_at(env, q_ins, hold_need + 6, step_sleep,
                                     stop_on_fasten_change=True)
        if done or trunc:
            print(f"  [preview] ended at bolt {i} (insert): "
                  f"{info.get('termination', 'done')}")
            return
        seated = env._nut_subphase == 1
        print(f"  bolt {i:2d} {'SEATED' if seated else 'MISS-insert'} "
              f"(subphase={env._nut_subphase})")

        # RETRACT — back out −Y past the tip + clearance.
        q_ret = _best_b_ik(env, retract_pos, want_z, seed_key=i)
        done, trunc, info = _hold_at(env, q_ret, 10, step_sleep,
                                     stop_on_fasten_change=True)
        ax, lat, th = env._nut_axial_lateral(i)
        cleared = i in env._nut_fastened
        print(f"  bolt {i:2d} {'RETRACTED' if cleared else 'MISS-retract'} "
              f"axial={ax*100:+5.1f}cm  (fastened {len(env._nut_fastened)}/{n})")
        if done or trunc:
            print(f"  [preview] ended at bolt {i} (retract): "
                  f"{info.get('termination', 'done')}")
            return
    print(f"[preview] complete — fastened {len(env._nut_fastened)}/{n}")


def _run_zero(env: TyroEnv, max_steps: int, step_sleep: float) -> None:
    act = _zero_action(env)
    print(f"[preview] zero-policy rollout ({max_steps} steps) — "
          "B stays near spawn; A holds tire.")
    obs, _ = env.reset()
    for t in range(max_steps):
        obs, r, done, trunc, info = env.step(act)
        if step_sleep > 0:
            time.sleep(step_sleep)
        if (t + 1) % 50 == 0:
            idx = int(env._nut_target_idx)
            d_B, theta_B = env._nut_gate_metrics(idx)
            print(f"  step {t+1:4d}: target=bolt{idx}  d_B={d_B*100:.1f}cm  "
                  f"fastened={len(env._nut_fastened)}")
        if done or trunc:
            print(f"  ended @ step {t+1}: {info.get('termination', 'done')}")
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", type=str, default="setup",
        choices=("setup", "oracle", "zero"),
        help="setup=static scene; oracle=IK bolt demo; zero=untrained rollout",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-steps", type=int, default=None,
                    help="Override nut_hold_steps for oracle (default: cfg value).")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="Horizon for zero-policy mode.")
    ap.add_argument("--step-sleep", type=float, default=0.02,
                    help="Real-time pause per sim step (0=fast).")
    ap.add_argument("--endpose", type=str, default=None,
                    help="Optional nut_mount_endpose.npz for A hold pose.")
    args = ap.parse_args()

    overrides = dict(
        nut_fastening_task=True,
        contact_force_terminate_above=0.0,
        scene_layout="fanuc_spacious",
        terminate_on="never",
    )
    if args.endpose:
        overrides["nut_mount_endpose_path"] = str(args.endpose)
    cfg = make_env_config(stage=3, phase=1, **overrides)
    hold = int(args.hold_steps if args.hold_steps is not None else cfg.nut_hold_steps)

    env = TyroEnv(cfg=cfg, render=True, seed=args.seed)
    env.reset(seed=args.seed)
    _frame_camera(env)

    print("[preview] Robot-B nut-fastening GUI")
    print(f"  mode={args.mode}  action_dim={cfg.action.dim}  obs_dim={cfg.obs.dim}")
    print(f"  A: frozen mount-hold  |  B: active Δpose [6:12]")
    print(f"  bolts={len(env.handles.bolts)}  reach_tol={cfg.nut_reach_tol}m  "
          f"align_tol={np.degrees(cfg.nut_align_tol_rad):.0f}deg  "
          f"hold_steps={hold} ({hold*0.05:.1f}s)")
    print("  colors: grey=pending  yellow=target  green=fastened")

    if args.mode == "setup":
        print("[preview] static setup — close GUI or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    elif args.mode == "oracle":
        _run_oracle(env, hold, float(args.step_sleep))
        print("[preview] GUI held open — close window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    else:
        _run_zero(env, int(args.max_steps), float(args.step_sleep))
        print("[preview] GUI held open — close window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
