#!/usr/bin/env python3
"""Plot the Robot-B nut-fastening learning curve (latest model).

Parses the latest ``runs/nut_fastening_*.log`` SB3 stdout and plots
``n_fastened_policy`` (the count of bolts the policy itself fastens,
excluding any pre-marked ones) vs training steps, with a moving-average
overlay and the 10/10 target line.
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


def latest_log() -> Path | None:
    logs = sorted(
        _RUNS.glob("nut_fastening_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def parse(logf: Path) -> list[tuple[int, float]]:
    pts, cur = [], {}
    for line in logf.open(errors="ignore"):
        m = re.search(r"n_fastened_policy\s*\|\s*([\d.]+)", line)
        if m:
            cur["nf"] = float(m.group(1))
        m = re.search(r"total_timesteps\s*\|\s*([\d.e+]+)", line)
        if m:
            cur["t"] = int(float(m.group(1)))
            if "nf" in cur:
                pts.append((cur["t"], cur["nf"]))
                cur = {}
    return pts


def smooth(ys: list[float], w: int = 7) -> list[float]:
    out, n = [], len(ys)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def main() -> None:
    logf = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_log()
    if logf is None or not logf.exists():
        print("no nut-fastening log found")
        return
    pts = parse(logf)
    if not pts:
        print(f"no data points in {logf}")
        return

    xs = [t / 1e6 for t, _ in pts]
    ys = [nf for _, nf in pts]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(xs, ys, color="#a5d6a7", linewidth=1.0, alpha=0.6,
            zorder=3, label="n_fastened_policy (raw)")
    ax.plot(xs, smooth(ys), color="#2e7d32", linewidth=2.6,
            zorder=5, label="n_fastened_policy (smoothed)")
    ax.axhline(10, color="#c62828", linestyle="--", linewidth=1.0,
               alpha=0.7, label="target (10/10)")

    ax.set_xlabel("training steps (x1e6)")
    ax.set_ylabel("policy-fastened bolts  (n_fastened_policy)")
    ax.set_title("Robot B nut fastening - learning curve")
    ax.set_ylim(-0.3, 10.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130)
    print(f"saved {_OUT} (from {logf.name})")


if __name__ == "__main__":
    main()
