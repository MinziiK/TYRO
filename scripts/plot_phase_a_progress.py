#!/usr/bin/env python3
"""Plot Robot-A Phase-A mount learning (initial run only).

Plots the initial mount training run ``phase1_mount_v2`` on its own:
  - success_rate (left axis, raw + 7-window moving average)
  - mount_radius_tol curriculum (right axis, m; tightening 0.55 -> 0.12)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_RUNS = _REPO / "runs"
_OUT = _REPO / "docs" / "phase_a_progress.png"

# Single initial mount run.
RUN = "phase1_mount_v2"


def parse(logf: Path) -> list[dict]:
    """Parse rollout blocks only (skip eval/ to avoid post-eval dips)."""
    pts, cur = [], {}
    section = None
    pats = {
        "rew": r"ep_rew_mean\s*\|\s*([-\d.]+)",
        "tol": r"mount_radius_tol\s*\|\s*([\d.]+)",
        "ang": r"mount_angle_tol_deg\s*\|\s*([\d.]+)",
        "dr": r"dr_range_cm\s*\|\s*([\d.]+)",
        "t": r"total_timesteps\s*\|\s*([\d.e+]+)",
    }
    for line in logf.open(errors="ignore"):
        if "rollout/" in line:
            section = "rollout"
        elif "eval/" in line:
            section = "eval"
        m = re.search(r"success_rate\s*\|\s*([\d.]+)", line)
        if m and section == "rollout":
            cur["succ"] = float(m.group(1))
        for k, p in pats.items():
            m = re.search(p, line)
            if m:
                cur[k] = float(m.group(1))
        if "t" in cur and "succ" in cur:
            pts.append(dict(cur))
            cur = {}
    return pts


def smooth(ys: list[float], w: int = 7) -> list[float]:
    out, n = [], len(ys)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        vals = [v for v in ys[lo:hi] if v == v]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def main() -> None:
    f = _RUNS / f"{RUN}.log"
    if not f.exists():
        print(f"no Phase-A log found: {f}")
        return
    pts = parse(f)
    if not pts:
        print(f"no data points in {f}")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax2 = ax1.twinx()

    all_t = [d["t"] / 1e6 for d in pts]
    all_s = [d.get("succ", float("nan")) for d in pts]
    all_tol = [d.get("tol", float("nan")) for d in pts]

    ax1.plot(all_t, all_s, color="#a5d6a7", linewidth=1.0, alpha=0.55,
             zorder=3)
    ax1.plot(all_t, smooth(all_s, 7), color="#2e7d32", linewidth=2.6,
             label="success_rate (smoothed)", zorder=5)
    ax2.plot(all_t, all_tol, color="#1565c0", linewidth=1.8,
             linestyle="--", label="mount_radius_tol (m)")

    ax1.set_xlabel("training steps (x1e6)")
    ax1.set_ylabel("success_rate", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylabel("mount_radius_tol (m)  [tightening]", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")
    ax2.set_ylim(0.0, 0.60)

    ax1.axhline(1.0, color="#c62828", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_title("Robot A mount - success vs tightening gate")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower left", fontsize=9)

    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130)
    print(f"saved {_OUT} ({len(pts)} pts from {RUN})")


if __name__ == "__main__":
    main()
