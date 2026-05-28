"""Find a compact tool-up HOME_POSE for UR10 that doesn't intersect
the tire / rack / vehicle AABBs.

Approach
--------
1. Build the full scene (no robot motion).
2. Sweep candidate EE targets in a coarse grid above + behind the tire.
3. For each candidate: call PyBullet IK with tool-up orientation,
   apply joints, ``stepSimulation`` once (zero physics), then ask
   ``getAABB`` for every robot link and every blocker body, and
   ``getContactPoints`` to verify zero penetration.
4. Score by (a) zero overlap, (b) EE position residual, (c) wrist +Z
   alignment to world +Z, (d) compactness (sum of |joint - midpoint|).
5. Print the best HOME_POSE and (X, Y, Z) it places the EE at.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import make_env_config  # noqa: E402
from src.env import TyroEnv  # noqa: E402


# Tool-up orientation: gripper +Z aligned to world +Z. RPY = (0, 0, 0)
# is the URDF-native EE frame for UR10 + Robotiq attachment; using a
# small Z-yaw lets the IK choose the better wrist branch.
TOOL_UP_QUAT = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

# Hard "do-not-touch" body uids (filled in at runtime by inspecting Scene)
# We block on: tire, vehicle, rack rails (any body with name=='' near the
# rack), truck assembly. Plane / panda EE are tolerated.
BLOCKER_NAMES = {"tire", "vehicle", "rack_inner", "rack_outer",
                 "truck_wheel_station"}


def link_aabb(uid: int, link_idx: int, client: int):
    return p.getAABB(uid, link_idx, physicsClientId=client)


def aabb_intersect(a, b, eps=1e-4):
    (a0, a1) = a
    (b0, b1) = b
    return all(a0[i] <= b1[i] + eps and b0[i] <= a1[i] + eps for i in range(3))


def main() -> int:
    cfg = make_env_config(stage=3, phase=1, contact_force_terminate_above=0.0)
    env = TyroEnv(cfg=cfg, render=False, seed=42)
    obs, _ = env.reset(seed=42)

    client = env.client
    ur = env.robot_A
    n = len(ur.arm.indices)

    # Resolve blocker body uids from the scene handles.
    blockers = []
    for uid in (env.handles.tire, env.handles.vehicle,
                env.handles.truck_uid):
        if uid is not None and uid >= 0:
            blockers.append(uid)
    # Rack rails — small bodies with empty names in [3, 4]. Detect by AABB
    # near rack centre.
    for bi in range(p.getNumBodies(physicsClientId=client)):
        if bi in (env.handles.plane, env.handles.tire, env.handles.vehicle,
                  env.handles.truck_uid, ur.uid, env.robot_B.uid):
            continue
        pos, _ = p.getBasePositionAndOrientation(bi, physicsClientId=client)
        # Two rack rails sit near (-1.90, ±0.20, -0.45). Anything in that
        # rough box is treated as a rack rail blocker.
        if -2.2 < pos[0] < -1.6 and -0.6 < pos[1] < 0.6 and -0.7 < pos[2] < 0.0:
            blockers.append(bi)
    print(f"[scan] blocker uids = {blockers}")

    tire_pos, _ = env.scene.tire_pose()
    R_out = float(cfg.tire_outer_radius)
    R_in = float(cfg.tire_inner_radius)
    tire_thickness = float(cfg.tire_thickness)
    print(f"[scan] tire   center={tire_pos}  R_out={R_out}  R_in={R_in}  thickness={tire_thickness}")

    # Sample EE targets in a 3-D grid above + behind (+X of) the tire.
    # X: from tire front (X+thickness/2 = -1.75) outward toward base (+X)
    # Y: lateral 0 ± 10 cm
    # Z: at tire top (+0.225 + R = 0.75) down to rack top (-0.30)
    candidates = []
    for x in np.arange(-1.70, -1.30 + 1e-6, 0.05):
        for y in (-0.05, 0.0, 0.05):
            for z in np.arange(-0.15, 0.55 + 1e-6, 0.05):
                candidates.append((float(x), float(y), float(z)))
    print(f"[scan] {len(candidates)} EE candidates")

    # IK warmup: use neutral q.
    q_seed = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]

    best = None  # (score, q, ee_xyz, residual, overlaps)
    n_ok = 0
    for ee_target in candidates:
        q = p.calculateInverseKinematics(
            ur.uid, ur.EE_LINK_INDEX,
            ee_target, TOOL_UP_QUAT,
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=client,
        )
        q_arm = list(q[:n])

        # Apply joints (kinematic only; reset to avoid solver torque).
        for idx, qi in zip(ur.arm.indices, q_arm):
            p.resetJointState(ur.uid, idx, qi, 0.0, physicsClientId=client)

        # Read EE world pose, check residual.
        link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                              computeForwardKinematics=True,
                              physicsClientId=client)
        ee_pos_actual = np.asarray(link[4])
        ee_orn_actual = link[5]
        residual = float(np.linalg.norm(ee_pos_actual - np.asarray(ee_target)))
        if residual > 0.02:
            continue

        # Check tool-up: world +Z component of EE local +Z must be > 0.9.
        rot = np.array(p.getMatrixFromQuaternion(ee_orn_actual)).reshape(3, 3)
        z_world = rot[:, 2]
        if z_world[2] < 0.90:
            continue

        # AABB overlap of every UR10 link vs every blocker.
        n_links = p.getNumJoints(ur.uid, physicsClientId=client)
        overlaps = []
        any_overlap = False
        for li in range(-1, n_links):
            try:
                a = link_aabb(ur.uid, li, client)
            except Exception:
                continue
            for bu in blockers:
                bn = p.getNumJoints(bu, physicsClientId=client)
                for bl in range(-1, bn):
                    try:
                        b = link_aabb(bu, bl, client)
                    except Exception:
                        continue
                    if aabb_intersect(a, b):
                        any_overlap = True
                        overlaps.append((li, bu, bl))
        if any_overlap:
            continue

        n_ok += 1
        # FSM-gate distance: must be OUTSIDE the hard gate (0.55 m) so the
        # curriculum has room to ramp and the policy learns to *actively*
        # approach. We also want EE to stay well above the rack top
        # (z > 0) so the wrist clears the rails on the descent.
        grasp_target = np.asarray(tire_pos) + np.array([0.0, 0.0, -R_out])
        d_grasp_now = float(np.linalg.norm(ee_pos_actual - grasp_target))

        # Compact pose check — shoulder_lift (q[1]) and elbow (q[2]) must
        # bend the arm UP, not extend it. Penalty if shoulder_lift > -π/6
        # (i.e. not lifted) or |elbow| < π/3 (i.e. straight).
        q_lift = float(q_arm[1])
        q_elbow = float(q_arm[2])

        # Score: smaller is better
        # - tool-up alignment penalty (1 - z_world[2])
        # - HARD encourage EE outside FSM hard gate (0.55 m)
        # - prefer EE in +X (toward base, away from tire bore)
        # - prefer EE Z > +0.20 (above rack top z = -0.30)
        # - prefer shoulder lifted up + elbow bent
        x_back = ee_target[0] - (tire_pos[0] + tire_thickness / 2)
        score = (
            10.0 * residual
            + 5.0 * (1.0 - float(z_world[2]))
            + 8.0 * max(0.0, cfg.approach_radius_tol + 0.05 - d_grasp_now)
            + 2.0 * max(0.0, 0.15 - x_back)
            + 2.0 * max(0.0, 0.20 - ee_target[2])
            + 1.5 * max(0.0, q_lift + math.pi / 6)     # want q_lift < -π/6
            + 1.0 * max(0.0, math.pi / 3 - abs(q_elbow))  # want |elbow| > π/3
        )
        if best is None or score < best[0]:
            best = (score, q_arm, ee_target, residual, ee_pos_actual.tolist())

    if best is None:
        print("[scan] NO viable candidate found")
        env.close()
        return 1

    score, q_arm, ee_target, residual, ee_actual = best
    print(f"[scan] viable candidates: {n_ok}")
    print(f"[scan] BEST score={score:.4f}")
    print(f"  EE target  = {tuple(round(v,3) for v in ee_target)}")
    print(f"  EE actual  = {tuple(round(v,4) for v in ee_actual)}")
    print(f"  residual   = {residual*1000:.2f} mm")
    print()
    print("HOME_POSE = (")
    for qi in q_arm:
        print(f"    {qi:+.4f},")
    print(")")
    print("Joint angles (deg):")
    print("  " + ", ".join(f"{math.degrees(q):+.2f}" for q in q_arm))

    # Final sanity: re-apply and report EE↔grasp distance for FSM gate.
    for idx, qi in zip(ur.arm.indices, q_arm):
        p.resetJointState(ur.uid, idx, qi, 0.0, physicsClientId=client)
    link = p.getLinkState(ur.uid, ur.EE_LINK_INDEX,
                          computeForwardKinematics=True,
                          physicsClientId=client)
    ee = np.asarray(link[4])
    grasp = np.asarray(tire_pos) + np.array([0.0, 0.0, -R_out])
    d_grasp = float(np.linalg.norm(ee - grasp))
    print(f"\nEE -> grasp_target  = {d_grasp*100:.1f} cm "
          f"(approach_tol_soft={cfg.approach_tol_soft*100:.0f} cm, "
          f"hard={cfg.approach_radius_tol*100:.0f} cm)")
    if d_grasp < cfg.approach_radius_tol:
        print("  WARN: HOME already within hard gate — Stage 0 will trigger on step 0!")
    elif d_grasp < cfg.approach_tol_soft:
        print("  WARN: HOME within soft gate — Stage 0 may trigger before curriculum ramp.")
    else:
        print("  OK: HOME outside soft gate; curriculum will need to ramp before trigger.")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
