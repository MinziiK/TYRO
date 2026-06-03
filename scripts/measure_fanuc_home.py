"""Print FANUC HOME FK and reach distances for the shipping layout.

Usage (tyro env):
    python scripts/measure_fanuc_home.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import EnvConfig, make_env_config
from src.env.robots import FanucR2000icRobot, make_robot_a
import pybullet as p


def main() -> int:
    cfg = make_env_config(robot_a_kind="fanuc_r2000ic")
    cid = p.connect(p.DIRECT)
    robot = make_robot_a(cid, cfg)
    assert isinstance(robot, FanucR2000icRobot)

    ee, quat = robot.ee_pose()
    base = np.asarray(cfg.robot_A_base_pos, dtype=np.float64)
    pickup = np.asarray(cfg.tire_pickup_pos, dtype=np.float64)
    hub = np.asarray(cfg.hub_pos_nominal, dtype=np.float64)

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    print(f"HOME joints (rad): {tuple(round(q, 4) for q in robot.HOME_POSE)}")
    print(f"tool0 pos:  {tuple(round(v, 4) for v in ee)}")
    print(f"tool0 quat: {tuple(round(v, 6) for v in quat)}")
    print(f"FINAL_LOCK_QUATERNION: {robot.FINAL_LOCK_QUATERNION}")
    print(f"dist base→tool0:     {dist(base, ee):.3f} m")
    print(f"dist tool0→pickup:   {dist(ee, pickup):.3f} m")
    print(f"dist tool0→hub:      {dist(ee, hub):.3f} m")
    print(f"dist pickup→hub:     {dist(pickup, hub):.3f} m")
    print(f"reach ratio pickup:  {dist(ee, pickup) / 2.65 * 100:.1f}% of 2.65 m")
    print(f"reach ratio hub:     {dist(ee, hub) / 2.65 * 100:.1f}% of 2.65 m")

    p.disconnect(cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
