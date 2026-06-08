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
from stable_baselines3 import PPO  # noqa: E402  (match src.eval import order)

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


def _compute_ref_waypoints(env: TyroEnv) -> tuple:
    """Analytic per-bolt waypoints along each bolt axis (staging/base/retract).

    All bolts on this hub share the same world Y and a common −Y axis, so
    inter-bolt transit is purely in XZ at a fixed staging/retract altitude.
    """
    order = env._nut_order()
    L = float(getattr(env.cfg, "bolt_length", 0.10))
    retract_clear = float(getattr(env.cfg, "nut_retract_clear", 0.03))
    margin = float(getattr(env.cfg, "nut_insert_margin", 0.0))
    stage_ax = float(env._nut_staging_axial())
    retr_ax = 0.5 * L + retract_clear + 0.03 + margin
    ref_stage, ref_base, ref_retract, ref_order = [], [], [], []
    for bi in order:
        ref_stage.append(env._nut_point_on_axis(bi, stage_ax))
        ref_base.append(env._nut_point_on_axis(bi, -0.5 * L))
        ref_retract.append(env._nut_point_on_axis(bi, retr_ax))
        ref_order.append(int(bi))
    home_ee = np.asarray(env.robot_B.ee_pose()[0], dtype=np.float64)
    # Hub-and-spoke center: bolt-ring centroid at the staging depth (the
    # hot-start start pose). All bolt staging points share this Y, so every
    # spoke (center→bolt, bolt→center) is a pure-XZ radial move at fixed Y.
    n_b = len(env.handles.bolts)
    centroid = np.mean(
        [np.asarray(env.scene.bolt_pose(i)[0], dtype=np.float64)
         for i in range(n_b)], axis=0,
    )
    a = np.asarray(env.scene.bolt_axis(int(order[0])), dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    ref_center = centroid + a * stage_ax
    return home_ee, ref_stage, ref_base, ref_retract, ref_order, ref_center


def _draw_ref_path(
    cid: int,
    home_ee: np.ndarray,
    ref_stage: list,
    ref_base: list,
    ref_retract: list,
    ref_ord: list,
    ref_center: np.ndarray = None,
) -> None:
    """Draw the ideal nut-fastening route in the debug visualizer.

    Route: HOME → hub CENTER (one-time) → bolt0 → bolt5 → … (the center is
    visited only once, at the start; thereafter bolt-to-bolt is direct).

    * grey-blue  HOME → hub center → first staging (one-time approach)
    * orange     insert: staging → hub-face base (−Y along bolt axis)
    * yellow     retract: base → past stud tip (+Y along bolt axis)
    * cyan       inter-bolt transit: XZ only at fixed retract Y (no Y travel
                 between bolts — all studs share the same hub Y / −Y axis)
    """
    if not ref_stage:
        return
    first_stage = np.asarray(ref_stage[0]).tolist()
    if ref_center is not None:
        # HOME → hub center (one-time), then center → first bolt staging.
        c = np.asarray(ref_center, dtype=np.float64)
        p.addUserDebugLine(
            home_ee.tolist(), c.tolist(),
            lineColorRGB=[0.5, 0.5, 1.0], lineWidth=1.5, physicsClientId=cid,
        )
        p.addUserDebugLine(
            c.tolist(), first_stage,
            lineColorRGB=[0.5, 0.5, 1.0], lineWidth=1.5, physicsClientId=cid,
        )
        p.addUserDebugText(
            "center", c.tolist(),
            textColorRGB=[0.6, 0.6, 1.0], textSize=1.2, physicsClientId=cid,
        )
    else:
        p.addUserDebugLine(
            home_ee.tolist(), first_stage,
            lineColorRGB=[0.5, 0.5, 1.0], lineWidth=1.5, physicsClientId=cid,
        )
    for k in range(len(ref_stage)):
        stage = np.asarray(ref_stage[k], dtype=np.float64)
        base = np.asarray(ref_base[k], dtype=np.float64)
        retr = np.asarray(ref_retract[k], dtype=np.float64)
        p.addUserDebugLine(
            stage.tolist(), base.tolist(),
            lineColorRGB=[1.0, 0.55, 0.0], lineWidth=3.0, physicsClientId=cid,
        )
        p.addUserDebugLine(
            base.tolist(), retr.tolist(),
            lineColorRGB=[1.0, 1.0, 0.0], lineWidth=2.0, physicsClientId=cid,
        )
        p.addUserDebugText(
            f"{k}:b{ref_ord[k]}", stage.tolist(),
            textColorRGB=[1.0, 1.0, 0.2], textSize=1.2, physicsClientId=cid,
        )
        if k + 1 < len(ref_stage):
            ns = np.asarray(ref_stage[k + 1], dtype=np.float64)
            # Circumferential hop: hold Y, move only in XZ to above the next bolt.
            hop = np.array([ns[0], retr[1], ns[2]], dtype=np.float64)
            p.addUserDebugLine(
                retr.tolist(), hop.tolist(),
                lineColorRGB=[0.0, 0.9, 1.0], lineWidth=2.0, physicsClientId=cid,
            )
            if float(np.linalg.norm(ns - hop)) > 1e-4:
                # Tiny coaxial slide onto the next staging point (≈1 cm ΔY here).
                p.addUserDebugLine(
                    hop.tolist(), ns.tolist(),
                    lineColorRGB=[0.0, 0.9, 1.0], lineWidth=2.0,
                    physicsClientId=cid,
                )
    # Closing leg: last bolt retract → HOME (return to spawn pose).
    last_retr = np.asarray(ref_retract[-1], dtype=np.float64)
    p.addUserDebugLine(
        last_retr.tolist(), home_ee.tolist(),
        lineColorRGB=[0.5, 0.5, 1.0], lineWidth=1.5, physicsClientId=cid,
    )
    print(f"  [ref] answer path: HOME→center→{ref_ord}→HOME  "
          f"cyan=XZ transit (Y fixed)  orange=insert  yellow=retract  "
          f"grey-blue=HOME/center legs")


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
    from scripts.e2e_nut_oracle import _bolt_poses, _want_z

    n = len(env.handles.bolts)
    hold_need = int(hold_steps)
    order = env._nut_order()
    L = float(env.cfg.bolt_length)
    print(f"[preview] oracle insertion-retract: {n} bolts, order={order}, "
          f"hold={hold_need} steps ({hold_need * 0.05:.1f}s), "
          f"bolt_len={L*100:.0f}cm  (XZ transit between bolts)")
    prev_retr = None
    seq = 0
    while len(env._nut_fastened) < n and seq < n + 2:
        i = int(env._nut_target_idx)
        want_z = _want_z(env, i)
        approach, insert, retract = _bolt_poses(env, i)
        if prev_retr is None:
            q_app = _best_b_ik(env, approach, want_z, seed_key=i)
            _hold_at(env, q_app, 4, step_sleep)
        else:
            hop = np.array([approach[0], prev_retr[1], approach[2]], dtype=np.float64)
            for k, pos in enumerate((hop, approach) if np.linalg.norm(hop - approach) > 1e-4 else (hop,)):
                q = _best_b_ik(env, pos, want_z, seed_key=i * 10 + k + 1)
                _hold_at(env, q, 3, step_sleep)
        ax, lat, th = env._nut_axial_lateral(i)
        print(f"  bolt {i:2d} APPROACH  axial={ax*100:+5.1f}cm "
              f"lat={lat*100:.1f}cm ang={np.degrees(th):.1f}deg")

        q_ins = _best_b_ik(env, insert, want_z, seed_key=i)
        _teleport_b(env, q_ins)
        ax, lat, th = env._nut_axial_lateral(i)
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

        q_ret = _best_b_ik(env, retract, want_z, seed_key=i)
        done, trunc, info = _hold_at(env, q_ret, 10, step_sleep,
                                     stop_on_fasten_change=True)
        ax, lat, th = env._nut_axial_lateral(i)
        cleared = i in env._nut_fastened
        print(f"  bolt {i:2d} {'RETRACTED' if cleared else 'MISS-retract'} "
              f"axial={ax*100:+5.1f}cm  (fastened {len(env._nut_fastened)}/{n})")
        prev_retr = retract
        seq += 1
        if done or trunc:
            print(f"  [preview] ended at bolt {i} (retract): "
                  f"{info.get('termination', 'done')}")
            return
    print(f"[preview] complete — fastened {len(env._nut_fastened)}/{n}")


def _run_policy(env: TyroEnv, model, obs, alpha: float,
                max_steps: int, step_sleep: float) -> None:
    """Roll out a trained PPO policy and watch where it stalls.

    NOTE: the env must already be reset by the caller (with the desired
    hot-start alpha set *before* that reset) and the resulting ``obs`` passed
    in. Calling env.reset() a second time here while the GUI render thread is
    live re-creates PyBullet bodies and segfaults llvmpipe.
    """
    n = len(env.handles.bolts)
    print(f"[preview] policy rollout")
    print(f"  hotstart_alpha={alpha:.2f}  (0=cold/real, 1=at-bolt)")
    last_tgt = int(env._nut_target_idx)
    last_f = len(env._nut_fastened)
    for t in range(max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(a)
        if step_sleep > 0:
            time.sleep(step_sleep)
        idx = int(env._nut_target_idx)
        nf = len(env._nut_fastened)
        if nf != last_f:
            print(f"  step {t+1:4d}: FASTENED bolt -> {nf}/{n} "
                  f"(now targeting bolt {idx})")
            last_f = nf
        elif idx != last_tgt:
            print(f"  step {t+1:4d}: target bolt {last_tgt} -> {idx}")
        last_tgt = idx
        if (t + 1) % 50 == 0:
            ax, lat, th = env._nut_axial_lateral(idx)
            print(f"  step {t+1:4d}: target=bolt{idx}  "
                  f"axial={ax*100:+5.1f}cm lat={lat*100:.1f}cm "
                  f"ang={np.degrees(th):.1f}deg  fastened={nf}/{n}")
        if done or trunc:
            print(f"  ended @ step {t+1}: {info.get('termination', 'done')}  "
                  f"fastened={nf}/{n}")
            break
    print(f"[preview] policy done — fastened {len(env._nut_fastened)}/{n}, "
          f"stalled at bolt {int(env._nut_target_idx)}")


def _run_replay(env: TyroEnv, traj_path: str, step_sleep: float,
                loop: bool = True) -> None:
    """Replay a recorded policy trajectory in the LIVE interactive GUI.

    The pybullet window stays fully interactive (mouse rotate/zoom) — only the
    robot/tire poses are driven from the recording, so no torch lives in this
    process and the GL render thread never races/segfaults. Bolt colors track
    the recorded fastened-count.
    """
    d = np.load(traj_path)
    qB, qA = d["qB"], d["qA"]
    tpos, torn = d["tpos"], d["torn"]
    nf, tgt = d["nf"], d["tgt"]
    bidx = [int(x) for x in d["bidx"]]
    aidx = [int(x) for x in d["aidx"]]
    T = len(qB)
    n = len(env.handles.bolts)
    print(f"[preview] REPLAY {T} frames  (recorded final {int(nf[-1])}/{n}, "
          f"max {int(nf.max())}/{n})  alpha={float(d['alpha']):.2f}")
    print("  GUI is LIVE — drag mouse to rotate, scroll to zoom.")

    rb, ra = env.robot_B, env.robot_A
    cid = env.client

    # Draw the analytic reference path from the live scene geometry (all bolts
    # share hub Y / −Y axis → inter-bolt hops are XZ-only at fixed altitude).
    (home_ee, ref_stage, ref_base, ref_retract, ref_ord,
     ref_center) = _compute_ref_waypoints(env)
    _draw_ref_path(cid, home_ee, ref_stage, ref_base, ref_retract, ref_ord,
                   ref_center)

    def set_frame(t: int) -> None:
        for s, q in zip(bidx, qB[t]):
            p.resetJointState(rb.uid, s, float(q), targetVelocity=0.0,
                              physicsClientId=cid)
        for s, q in zip(aidx, qA[t]):
            p.resetJointState(ra.uid, s, float(q), targetVelocity=0.0,
                              physicsClientId=cid)
        p.resetBasePositionAndOrientation(
            env.handles.tire, tpos[t].tolist(), torn[t].tolist(),
            physicsClientId=cid)
        k = int(nf[t])
        cur = int(tgt[t])
        for i in range(n):
            if i < k:
                env._set_bolt_color(i, env._NUT_COLOR_FASTENED)
            elif i == cur:
                env._set_bolt_color(i, (1.0, 1.0, 0.0, 1.0))
            else:
                env._set_bolt_color(i, (0.6, 0.6, 0.6, 1.0))

    last_f = -1
    while True:
        for t in range(T):
            set_frame(t)
            if int(nf[t]) != last_f:
                print(f"  frame {t:4d}: fastened {int(nf[t])}/{n}  "
                      f"target=bolt{int(tgt[t])}")
                last_f = int(nf[t])
            p.stepSimulation(physicsClientId=cid)
            if step_sleep > 0:
                time.sleep(step_sleep)
        if not loop:
            break
        print(f"  [replay] looping (recorded stalls at bolt {int(tgt[-1])}, "
              f"{int(nf[-1])}/{n} fastened) — Ctrl-C to stop")
        time.sleep(0.8)


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
        choices=("setup", "oracle", "zero", "policy", "replay"),
        help="setup=static scene; oracle=IK bolt demo; "
             "zero=untrained rollout; policy=trained PPO rollout (live torch); "
             "replay=interactive replay of a recorded policy trajectory",
    )
    ap.add_argument("--model", type=str, default=None,
                    help="Path to trained PPO .zip (required for --mode policy).")
    ap.add_argument("--traj", type=str, default=None,
                    help="Recorded trajectory .npz (required for --mode replay).")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="hot-start alpha for policy mode (0=cold/real).")
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

    # Load the PPO model BEFORE connecting the GUI/OpenGL context. Initializing
    # torch after PyBullet's GUI renderer is live segfaults on the VNC display.
    policy_model = None
    if args.mode == "policy":
        if not args.model:
            print("[preview] --mode policy requires --model PATH")
            return 2
        policy_model = PPO.load(args.model, device="cpu")
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

    # Replay must reproduce the recorded scene (A pose, tire, bolts), so reset
    # with the recording's seed/alpha.
    reset_seed = args.seed
    reset_alpha = args.alpha
    if args.mode == "replay":
        if not args.traj:
            print("[preview] --mode replay requires --traj PATH")
            return 2
        _d = np.load(args.traj)
        reset_seed = int(_d["seed"])
        reset_alpha = float(_d["alpha"])

    env = TyroEnv(cfg=cfg, render=True, seed=reset_seed)
    # The nut reset re-creates bodies and runs ~640 IK solves with
    # saveState/restoreState. With torch loaded, doing that while PyBullet's
    # async GL render thread is actively drawing races and segfaults the
    # render thread (intermittently). Freeze rendering across the heavy reset,
    # then re-enable it for the live rollout.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                               physicsClientId=env.client)
    # Set the hot-start alpha BEFORE the (single) reset so policy mode never
    # needs a second reset.
    if args.mode in ("policy", "replay"):
        env.set_nut_b_hotstart_alpha(float(reset_alpha))
    reset_obs, _ = env.reset(seed=reset_seed)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                               physicsClientId=env.client)
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
    elif args.mode == "policy":
        _run_policy(env, policy_model, reset_obs, float(args.alpha),
                    int(args.max_steps), float(args.step_sleep))
        print("[preview] GUI held open — close window or Ctrl-C to exit.")
        try:
            while p.isConnected(env.client):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    elif args.mode == "replay":
        try:
            _run_replay(env, args.traj, float(args.step_sleep), loop=True)
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
