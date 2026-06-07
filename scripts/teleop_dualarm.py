"""Interactive GUI teleop to diagnose dual-arm tire mounting.

Opens the full Phase-1 scene (truck hub + cargo + UR10 + Panda) with the
**UR10-feasible URDF tire** and exposes debug sliders so you can drive each
arm's end-effector in Cartesian space and reposition the tire by hand. Use it
to *see* what is reachable / where collisions happen, instead of guessing from
headless sweeps.

What the on-screen readout tells you (top-left text, refreshed every frame):
  * each EE's achieved world position and its distance from that arm's base
    (UR10 reach ~1.30 m, Panda reach ~0.85 m) + the **IK residual**
    ``|target - achieved|`` — a large residual means that pose is NOT
    reachable / outside the arm's dexterous workspace.
  * tire COM and its distance to the hub centre + bore-vs-hub-axis angle
    (mount needs ~0 angle and < 0.04 m distance).
  * collision flags and the max contact-normal force.

Sliders:
  * ``urA_x/y/z``  — UR10 EE Cartesian target.
  * ``urA_palm_up``— 1 = lock UR10 tool +Z to world +Z (the grasp pose used in
                     training); 0 = position-only IK (orientation free).
  * ``panB_x/y/z`` — Panda EE Cartesian target (position-only IK).
  * ``tire_x/y/z`` — tire COM (only while NOT grasped; static/posable).
  * ``tire_bore``  — 0:+X  1:-Y(hub axis)  2:+Z  (rounded).
  * ``grasp_UR_A`` — 1 = rigid-grasp tire to UR10 EE at the live pose.
  * ``grasp_pan_B``— 1 = rigid-grasp tire to Panda EE at the live pose.
  * ``reset``      — nudge to re-pin the tire to the slider pose & drop grasps.

Run (needs a display):
    python -m scripts.teleop_dualarm
    python -m scripts.teleop_dualarm --proc-tire   # original big procedural tire
    python -m scripts.teleop_dualarm --rack        # also show the cradle rails
    python -m scripts.teleop_dualarm --check       # headless self-test (no GUI)
"""
from __future__ import annotations

import argparse
import numpy as np
import pybullet as p

from src.config import make_env_config
from src.env.tyro_env import TyroEnv


def _quat_align_z(target):
    z = np.array([0.0, 0.0, 1.0])
    t = np.asarray(target, float)
    t = t / max(np.linalg.norm(t), 1e-9)
    v = np.cross(z, t); c = float(np.dot(z, t)); s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.array([0., 0., 0., 1.]) if c > 0 else np.array([1., 0., 0., 0.])
    ax = v / s; h = 0.5 * np.arctan2(s, c); sh = np.sin(h)
    return np.array([ax[0]*sh, ax[1]*sh, ax[2]*sh, np.cos(h)])


BORE_DIRS = {0: [1., 0., 0.], 1: [0., -1., 0.], 2: [0., 0., 1.]}


def _ik(env, rb, pos, quat=None, forces=None, reset=False):
    cl = env.client
    cur, _ = rb.joint_state()
    kw = dict(maxNumIterations=300, residualThreshold=1e-5, physicsClientId=cl)
    if quat is not None:
        sol = p.calculateInverseKinematics(rb.uid, rb.EE_LINK_INDEX, list(pos),
                                            list(quat), **kw)
    else:
        sol = p.calculateInverseKinematics(rb.uid, rb.EE_LINK_INDEX, list(pos), **kw)
    sol = np.asarray(sol)
    arm = np.clip(sol[rb._ik_arm_slots], rb.arm.lower, rb.arm.upper)
    if reset:
        for idx, q in zip(rb.arm.indices, arm):
            p.resetJointState(rb.uid, idx, float(q), 0, physicsClientId=cl)
    else:
        f = forces if forces is not None else [200.0] * rb.arm.n
        p.setJointMotorControlArray(rb.uid, rb.arm.indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=arm.tolist(), forces=f,
                                    positionGains=[1.0] * rb.arm.n,
                                    velocityGains=[1.0] * rb.arm.n,
                                    physicsClientId=cl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-tire", action="store_true",
                    help="Use the original big procedural tire instead of the URDF one.")
    ap.add_argument("--rack", action="store_true", help="Show the cradle rails.")
    ap.add_argument("--check", action="store_true",
                    help="Headless self-test: build + 50 steps in DIRECT, no GUI.")
    args = ap.parse_args()

    overrides = dict(
        freeze_robot_b=False,            # Panda controllable
        spawn_tire_rack=bool(args.rack),
        start_pos_curriculum_enable=False,
        attached_spawn_when_easy=False,
        reverse_curriculum_enable=False,
        use_planner_residual=False,      # we drive arms manually
        contact_force_terminate_above=0.0,
    )
    if not args.proc_tire:
        overrides.update(use_tire_urdf=True, tire_outer_radius=0.30,
                         tire_inner_radius=0.23, tire_thickness=0.16, tire_mass=1.5)
    cfg = make_env_config(stage=3, phase=1, **overrides)
    R = float(cfg.tire_outer_radius)

    env = TyroEnv(cfg=cfg, render=not args.check)
    env.reset()
    cl = env.client
    A, B, tire = env.robot_A, env.robot_B, env.handles.tire
    ur_base = np.asarray(cfg.robot_A_base_pos)
    pa_base = np.asarray(cfg.robot_B_base_pos)
    hub = np.asarray(cfg.tire_mount_pos)

    # Start the tire in the cooperative zone in front of the hub.
    tire0 = np.array([-0.20, 0.50, 0.20])
    eeA0, _ = A.ee_pose(); eeB0, _ = B.ee_pose()

    if args.check:
        # Headless sanity: pin tire, IK both arms a few steps, no crash.
        p.changeDynamics(tire, -1, mass=0.0, physicsClientId=cl)
        p.resetBasePositionAndOrientation(tire, tire0.tolist(),
                                          _quat_align_z([0, -1, 0]).tolist(),
                                          physicsClientId=cl)
        for _ in range(50):
            _ik(env, A, np.array(eeA0), quat=A.FINAL_LOCK_QUATERNION)
            _ik(env, B, np.array(eeB0))
            p.stepSimulation(physicsClientId=cl)
        print("[teleop] --check OK: scene + dual-arm IK ran 50 steps headless.")
        env.close()
        return 0

    sl = {}
    def add(name, lo, hi, start):
        sl[name] = p.addUserDebugParameter(name, lo, hi, start, physicsClientId=cl)
    # Robot BASE position sliders — move the arms closer to the work zone so
    # their workspace actually covers the hub. Applied via resetBase each frame.
    add("urA_base_x", -1.4, 0.2, float(ur_base[0]))
    add("urA_base_y", -0.4, 0.9, float(ur_base[1]))
    add("urA_base_z", -0.5, 0.1, float(ur_base[2]))
    add("panB_base_x", -0.6, 0.6, float(pa_base[0]))
    add("panB_base_y", -0.6, 0.6, float(pa_base[1]))
    add("panB_base_z", -0.3, 0.3, float(pa_base[2]))
    add("urA_x", -2.2, 0.6, float(eeA0[0])); add("urA_y", -0.7, 1.3, float(eeA0[1]))
    add("urA_z", -0.6, 1.0, float(eeA0[2])); add("urA_palm_up", 0, 1, 1)
    add("panB_x", -0.8, 0.9, float(eeB0[0])); add("panB_y", -0.7, 1.2, float(eeB0[1]))
    add("panB_z", -0.6, 1.0, float(eeB0[2]))
    add("tire_x", -2.1, 0.5, float(tire0[0])); add("tire_y", -0.6, 1.2, float(tire0[1]))
    add("tire_z", -0.5, 1.0, float(tire0[2])); add("tire_bore", 0, 2, 1)
    add("grasp_UR_A", 0, 1, 0); add("grasp_pan_B", 0, 1, 0)
    add("reset", 0, 1, 0)

    txt = p.addUserDebugText("", [0, 0, 0], physicsClientId=cl)
    grasp = {"A": None, "B": None}
    last_reset = 0.0
    frame = 0
    print("[teleop] GUI up. Drag sliders (top-right panel). Ctrl-C to quit.")
    print("[teleop] Readout also printed here every ~0.5 s — watch 'IKres'.")

    def read(n):
        return float(p.readUserDebugParameter(sl[n], physicsClientId=cl))

    def do_grasp(rb, key):
        ee_p, ee_o = rb.ee_pose()
        tp, to = p.getBasePositionAndOrientation(tire, physicsClientId=cl)
        ip, io = p.invertTransform(list(tp), list(to))
        cpos, corn = p.multiplyTransforms(ip, io, list(ee_p), list(ee_o))
        cid = p.createConstraint(rb.uid, rb.EE_LINK_INDEX, tire, -1, p.JOINT_FIXED,
                                 [0, 0, 0], [0, 0, 0], list(cpos), [0, 0, 0, 1],
                                 list(corn), physicsClientId=cl)
        p.changeConstraint(cid, maxForce=1e6, physicsClientId=cl)
        grasp[key] = cid

    def drop(key):
        if grasp[key] is not None:
            try: p.removeConstraint(grasp[key], physicsClientId=cl)
            except p.error: pass
            grasp[key] = None

    while True:
        gA = read("grasp_UR_A") > 0.5
        gB = read("grasp_pan_B") > 0.5
        rv = read("reset")
        if rv > 0.5 and last_reset <= 0.5:
            drop("A"); drop("B")
        last_reset = rv
        grasped_any = (grasp["A"] is not None) or (grasp["B"] is not None)

        # Reposition robot bases live (fixed-base bodies can be teleported).
        ur_base = np.array([read("urA_base_x"), read("urA_base_y"), read("urA_base_z")])
        pa_base = np.array([read("panB_base_x"), read("panB_base_y"), read("panB_base_z")])
        p.resetBasePositionAndOrientation(A.uid, ur_base.tolist(),
                                          list(A.base_orn), physicsClientId=cl)
        p.resetBasePositionAndOrientation(B.uid, pa_base.tolist(),
                                          list(B.base_orn), physicsClientId=cl)

        if not grasped_any:
            # tire is static and posable from sliders
            p.changeDynamics(tire, -1, mass=0.0, physicsClientId=cl)
            bore = int(round(read("tire_bore")))
            q = _quat_align_z(BORE_DIRS.get(bore, [0, -1, 0]))
            tpos = np.array([read("tire_x"), read("tire_y"), read("tire_z")])
            p.resetBasePositionAndOrientation(tire, tpos.tolist(), q.tolist(),
                                              physicsClientId=cl)
            p.resetBaseVelocity(tire, [0, 0, 0], [0, 0, 0], physicsClientId=cl)
        else:
            p.changeDynamics(tire, -1, mass=float(cfg.tire_mass), physicsClientId=cl)

        if gA and grasp["A"] is None:
            do_grasp(A, "A")
        if not gA:
            drop("A")
        if gB and grasp["B"] is None:
            do_grasp(B, "B")
        if not gB:
            drop("B")

        urt = np.array([read("urA_x"), read("urA_y"), read("urA_z")])
        pat = np.array([read("panB_x"), read("panB_y"), read("panB_z")])
        urq = A.FINAL_LOCK_QUATERNION if read("urA_palm_up") > 0.5 else None
        _ik(env, A, urt, quat=urq, forces=[400, 400, 300, 60, 60, 60])
        _ik(env, B, pat, forces=[87, 87, 87, 87, 12, 12, 12])
        for _ in range(cfg.decimation):
            p.stepSimulation(physicsClientId=cl)

        eeA, _ = A.ee_pose(); eeB, _ = B.ee_pose()
        tp, _ = p.getBasePositionAndOrientation(tire, physicsClientId=cl)
        tp = np.asarray(tp)
        ax = env.scene.tire_axis(); hubax = env.scene.hub_axis()
        bore_ang = np.degrees(np.arccos(np.clip(np.dot(ax, hubax), -1, 1)))
        coll = env._in_bad_collision(); cf = env._max_contact_normal_force()
        msg = (
            f"UR10 base {np.round(ur_base,2).tolist()}  Panda base {np.round(pa_base,2).tolist()}\n"
            f"UR10 EE {np.round(eeA,3).tolist()} reach {np.linalg.norm(eeA-ur_base):.2f} "
            f"IKres {np.linalg.norm(eeA-urt):.3f}\n"
            f"Panda EE {np.round(eeB,3).tolist()} reach {np.linalg.norm(eeB-pa_base):.2f} "
            f"IKres {np.linalg.norm(eeB-pat):.3f}\n"
            f"tire {np.round(tp,3).tolist()}  d_hub {np.linalg.norm(tp-hub):.3f}  "
            f"bore_vs_hub {bore_ang:.1f}deg\n"
            f"grasp A={'Y' if grasp['A'] else '-'} B={'Y' if grasp['B'] else '-'}  "
            f"collision={coll}  max_cf={cf:.0f}"
        )
        txt = p.addUserDebugText(msg, [-1.6, -0.6, 1.3], textColorRGB=[0, 0, 0],
                                 textSize=1.1, replaceItemUniqueId=txt,
                                 physicsClientId=cl)
        frame += 1
        if frame % 30 == 0:
            print("\n" + msg, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
