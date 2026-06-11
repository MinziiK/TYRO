#!/usr/bin/env python3
"""Plot the Robot-B nut-fastening learning curve (nominal run only, MEASURED).

Mirrors the Phase-A plot: success vs the tightening difficulty curriculum.
Plots the nominal (no-DR) learning run ``nut_fastening_v15``:
  - rollout success_rate (left axis, raw + 7-window moving average; all-10)
  - arrive-angle gate (right axis, deg; tightening 35 -> 12 as difficulty rises)

Pass a single log path to plot a different run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_RUNS = _REPO / "runs"
_OUT = _REPO / "docs" / "nut_fastening_progress.png"

# Nominal (no-DR) learning run.
RUN = "nut_fastening_v15"


def parse(logf: Path) -> list[dict]:
    """Parse rollout blocks: success_rate + arrive-angle gate + step."""
    pts, cur = [], {}
    section = None
    for line in logf.open(errors="ignore"):
        if "rollout/" in line:
            section = "rollout"
        elif "eval/" in line:
            section = "eval"
        m = re.search(r"success_rate\s*\|\s*([\d.]+)", line)
        if m and section == "rollout":
            cur["sr"] = float(m.group(1))
        m = re.search(r"nut_arrive_ang_deg\s*\|\s*([\d.]+)", line)
        if m:
            cur["ang"] = float(m.group(1))
        m = re.search(r"total_timesteps\s*\|\s*([\d.e+]+)", line)
        if m:
            cur["t"] = int(float(m.group(1)))
            if "sr" in cur:
                pts.append(dict(cur))
                cur = {}
    return pts


def smooth(xs: list[float], ys: list[float], w: int = 7):
    out = []
    n = len(ys)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        vals = [v for v in ys[lo:hi] if v == v]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def main() -> None:
    logf = Path(sys.argv[1]) if len(sys.argv) > 1 else _RUNS / f"{RUN}.log"
    if not logf.exists():
        print(f"no nut-fastening log found: {logf}")
        return
    p = parse(logf)
    if not p:
        print(f"no data in {logf}")
        return
    all_t = [d["t"] / 1e6 for d in p]
    all_sr = [d.get("sr", float("nan")) for d in p]
    all_ang = [d.get("ang", float("nan")) for d in p]

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax2 = ax1.twinx()

    # success_rate (left): raw + smoothed.
    ax1.plot(all_t, all_sr, color="#a5d6a7", linewidth=1.0, alpha=0.55,
             zorder=3)
    ax1.plot(all_t, smooth(all_t, all_sr), color="#2e7d32", linewidth=2.6,
             zorder=5, label="success_rate (smoothed)")
    # arrive-angle difficulty gate (right): tightens 35 -> 12 deg.
    ax2.plot(all_t, all_ang, color="#1565c0", linewidth=1.8,
             linestyle="--", label="arrive-angle gate (deg)")

    ax1.axhline(1.0, color="#c62828", linestyle="--", linewidth=0.8,
                alpha=0.5)
    ax1.set_xlabel("training steps (x1e6)")
    ax1.set_ylabel("success_rate (all-10)", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylabel("arrive-angle gate (deg)  [tightening]", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")
    ax2.set_ylim(0.0, 40.0)
    ax1.set_title("Robot B nut fastening - success vs tightening gate")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left",
               fontsize=9)

    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130)
    print(f"saved {_OUT} ({len(p)} pts from {logf.name})")


if __name__ == "__main__":
    main()
