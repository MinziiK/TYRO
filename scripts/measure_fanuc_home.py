#!/usr/bin/env python3
"""Reach / manipulability sweep for FANUC spacious scene candidates.

Usage (tyro env)::
    python scripts/measure_fanuc_home.py
    python scripts/measure_fanuc_home.py --layout fanuc_spacious
    python scripts/measure_fanuc_home.py --sweep
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import EnvConfig, apply_fanuc_spacious_layout
from src.env.robots import FanucR2000icRobot, make_robot_a
import pybullet as p

FANUC_REACH = 2.65


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _manip_at_hub(robot: FanucR2000icRobot, hub_pos) -> float:
    """Manipulability w = sqrt(det(JJ^T)) at hub-facing pose (approx)."""
    hub = np.asarray(hub_pos, dtype=np.float64)
    target = hub + np.array([0.0, -0.15, 0.0], dtype=np.float64)
    q, _ = robot.joint_state()
    ik = p.calculateInverseKinematics(
        robot.uid, robot.EE_LINK_INDEX,
        target.tolist(), list(robot.FINAL_LOCK_QUATERNION),
        lowerLimits=robot.arm.lower.tolist(),
        upperLimits=robot.arm.upper.tolist(),
        jointRanges=robot.arm.range.tolist(),
        restPoses=q.tolist(),
        maxNumIterations=200,
        physicsClientId=robot.client,
    )
    movable = robot._movable_joint_indices()
    q_full = [float(ik[i]) if i < len(ik) else 0.0 for i in range(max(movable) + 1)]
    zeros = [0.0] * len(movable)
    jac_lin, jac_ang = p.calculateJacobian(
        robot.uid, robot.EE_LINK_INDEX, [0, 0, 0],
        q_full, zeros, zeros, physicsClientId=robot.client,
    )
    J = np.vstack([np.asarray(jac_lin), np.asarray(jac_ang)])
    JJt = J @ J.T
    return math.sqrt(max(float(np.linalg.det(JJt)), 0.0))


def measure_layout(cfg: EnvConfig, label: str) -> Dict[str, float]:
    cid = p.connect(p.DIRECT)
    robot = make_robot_a(cid, cfg)
    if not isinstance(robot, FanucR2000icRobot):
        p.disconnect(cid)
        raise TypeError("robot_a_kind must be fanuc_r2000ic")
    ee, _ = robot.ee_pose()
    base = np.asarray(cfg.robot_A_base_pos, dtype=np.float64)
    pickup = np.asarray(cfg.tire_pickup_pos, dtype=np.float64)
    hub = np.asarray(cfg.hub_pos_nominal, dtype=np.float64)
    d_pick = _dist(base, pickup)
    d_hub = _dist(base, hub)
    d_ee_hub = _dist(ee, hub)
    try:
        w = _manip_at_hub(robot, hub)
    except Exception:
        w = 0.0
    p.disconnect(cid)
    out = {
        "label": label,
        "base_x": base[0],
        "hub_y": hub[1],
        "pickup_x": pickup[0],
        "reach_pick_pct": 100.0 * d_pick / FANUC_REACH,
        "reach_hub_pct": 100.0 * d_hub / FANUC_REACH,
        "ee_hub_m": d_ee_hub,
        "manip_hub": w,
    }
    print(
        f"[{label}] base=({base[0]:.2f},0) hub_y={hub[1]:.2f} pickup_x={pickup[0]:.2f}  "
        f"reach pick={out['reach_pick_pct']:.0f}% hub={out['reach_hub_pct']:.0f}%  "
        f"HOME→hub={d_ee_hub:.2f}m  w@hub={w:.4f}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("shipping", "fanuc_spacious"), default="fanuc_spacious")
    ap.add_argument("--sweep", action="store_true", help="Print candidate grid")
    args = ap.parse_args()

    if args.sweep:
        print(f"FANUC nominal reach = {FANUC_REACH} m\n")
        candidates: list[Tuple[str, EnvConfig]] = []
        for bx in (-1.50, -1.80, -2.00):
            for hy in (1.00, 1.10, 1.20):
                for px in (-2.20, -2.40):
                    cfg = EnvConfig(robot_a_kind="fanuc_r2000ic")
                    apply_fanuc_spacious_layout(cfg)
                    cfg.robot_A_base_pos = (bx, 0.0, -0.30)
                    cfg.hub_pos_nominal = (0.0, hy, 0.22)
                    cfg.tire_mount_pos = cfg.hub_pos_nominal
                    cfg.tire_pickup_pos = (px, 0.0, 0.3913)
                    cfg.vehicle_center_world = (0.0, hy + 0.25, 0.78)
                    cfg.tire_rack_inner_center = (px, 0.40, -0.30)
                    cfg.tire_rack_outer_center = (px, -0.40, -0.30)
                    candidates.append((f"bx={bx} hy={hy} px={px}", cfg))
        best = None
        for label, cfg in candidates:
            m = measure_layout(cfg, label)
            if m["reach_hub_pct"] <= 75 and m["reach_pick_pct"] <= 65:
                if best is None or m["manip_hub"] > best["manip_hub"]:
                    best = m
        if best:
            print(f"\n[recommended] {best['label']}  hub reach={best['reach_hub_pct']:.0f}%")
        else:
            print("\n[warn] no candidate met reach<=75%; using fanuc_spacious defaults")
        return 0

    cfg = EnvConfig(robot_a_kind="fanuc_r2000ic")
    if args.layout == "fanuc_spacious":
        apply_fanuc_spacious_layout(cfg)
    measure_layout(cfg, args.layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
