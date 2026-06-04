#!/usr/bin/env python3
"""Print spacious vs shipping layout distances and FANUC reach / gripper EE check."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import EnvConfig, apply_fanuc_spacious_layout

FANUC_REACH = 2.65


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def report_layout(label: str, cfg: EnvConfig) -> None:
    a = cfg.robot_A_base_pos
    b = cfg.robot_B_base_pos
    hub = cfg.hub_pos_nominal
    pick = cfg.tire_pickup_pos
    veh = cfg.vehicle_center_world
    print(f"\n=== {label} ===")
    print(f"  robot_A base     {tuple(round(x, 3) for x in a)}")
    print(f"  robot_B base     {tuple(round(x, 3) for x in b)}")
    print(f"  hub              {tuple(round(x, 3) for x in hub)}")
    print(f"  tire pickup      {tuple(round(x, 3) for x in pick)}")
    print(f"  vehicle center   {tuple(round(x, 3) for x in veh)}")
    d_ab = _dist(a, b)
    d_ah = _dist(a, hub)
    d_ap = _dist(a, pick)
    d_hp = _dist(hub, pick)
    print(f"  A <-> B          {d_ab:.2f} m")
    print(f"  A -> hub         {d_ah:.2f} m  ({100 * d_ah / FANUC_REACH:.0f}% of {FANUC_REACH}m reach)")
    print(f"  A -> pickup      {d_ap:.2f} m  ({100 * d_ap / FANUC_REACH:.0f}% reach)")
    print(f"  hub <-> pickup   {d_hp:.2f} m")
    wall_y = float(hub[1]) + float(cfg.cargo_back_wall_y_offset)
    print(f"  cargo back wall  Y ≈ {wall_y:.2f}")


def report_gripper() -> None:
    import pybullet as p
    from src.env.robots import FanucR2000icRobot, make_robot_a

    cfg = EnvConfig(robot_a_kind="fanuc_r2000ic")
    apply_fanuc_spacious_layout(cfg)
    cid = p.connect(p.DIRECT)
    robot = make_robot_a(cid, cfg)
    assert isinstance(robot, FanucR2000icRobot)
    ee_name = str(getattr(cfg, "fanuc_ee_link_name", "wheel_tool_tip"))
    ee_idx = robot.EE_LINK_INDEX
    ee_pos, _ = robot.ee_pose()
    # tool0 link index
    tool0_idx = robot._link_index_for_child_link("tool0")
    t0_pos = np.array(
        p.getLinkState(robot.uid, tool0_idx, physicsClientId=cid)[4], dtype=np.float64,
    )
    tip_offset = np.linalg.norm(ee_pos - t0_pos)
    print(f"\n=== gripper / EE ===")
    print(f"  fanuc_urdf       {cfg.fanuc_urdf}")
    print(f"  fanuc_ee_link    {ee_name}  (link index {ee_idx})")
    print(f"  tool0 link index {tool0_idx}")
    print(f"  HOME wheel_tool_tip  {tuple(round(v, 3) for v in ee_pos)}")
    print(f"  HOME tool0             {tuple(round(v, 3) for v in t0_pos)}")
    print(f"  tip offset from tool0  {tip_offset * 100:.1f} cm")
    print(f"  grasp_com_offset_world {cfg.grasp_com_offset_world} (× R={cfg.tire_outer_radius}m)")
    p.disconnect(cid)


def main() -> int:
    ship = EnvConfig()
    sp = EnvConfig()
    apply_fanuc_spacious_layout(sp)
    report_layout("shipping (default UR10 layout)", ship)
    report_layout("fanuc_spacious", sp)
    print("\n--- spacious vs shipping deltas ---")
    print(f"  hub Y:     {ship.hub_pos_nominal[1]:.2f} -> {sp.hub_pos_nominal[1]:.2f}  (+{sp.hub_pos_nominal[1]-ship.hub_pos_nominal[1]:.2f} m)")
    print(f"  pickup X:  {ship.tire_pickup_pos[0]:.2f} -> {sp.tire_pickup_pos[0]:.2f}  ({sp.tire_pickup_pos[0]-ship.tire_pickup_pos[0]:+.2f} m)")
    print(f"  A base X:  {ship.robot_A_base_pos[0]:.2f} -> {sp.robot_A_base_pos[0]:.2f}  ({sp.robot_A_base_pos[0]-ship.robot_A_base_pos[0]:+.2f} m)")
    report_gripper()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
