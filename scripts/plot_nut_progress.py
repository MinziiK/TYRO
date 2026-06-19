#!/usr/bin/env python3
"""Plot Robot-B nut-fastening learning — v24 chain (pre-DR) only.

Mirrors ``plot_phase_a_progress.py``:
  - rollout success_rate (left axis, raw + 7-window moving average)
  - hot-start alpha curriculum (right axis; 0.3 -> 0, approach-from-HOME)

Default run: ``nut_fastening_v24_chain`` (4M, nominal hub, spin-free recovery).

Pass ``--full-pipeline`` to plot chain + DR stages B / B2 / B3 (legacy view).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_RUNS = _REPO / "runs"
_OUT = _REPO / "docs" / "nut_fastening_progress.png"

# Pre-DR chain recovery (default).
RUN = "nut_fastening_v24_chain"

V24_PIPELINE: list[tuple[str, str, float]] = [
    ("nut_fastening_v24_chain", "Stage A\n(chain)", 4.0),
    ("nut_fastening_v24_dr_stageB", "Stage B\n(DR 0→5cm)", 3.0),
    ("nut_fastening_v24_dr_stageB2", "Stage B2\n(DR 3.5→5cm)", 2.0),
    ("nut_fastening_v24_dr_stageB3", "Stage B3\n(5cm corner)", 2.0),
]


def _init_hotstart_from_log(logf: Path) -> float | None:
    for line in logf.open(errors="ignore"):
        m = re.search(r"\[curriculum\] nut_b_hotstart_alpha init = ([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None


def parse_chain(logf: Path) -> list[dict]:
    """Parse rollout blocks: success_rate + hot-start alpha + step."""
    init_hs = _init_hotstart_from_log(logf)
    pts: list[dict] = []
    cur: dict = {}
    section: str | None = None
    last_hs = init_hs if init_hs is not None else 0.3

    for line in logf.open(errors="ignore"):
        if "rollout/" in line:
            section = "rollout"
        elif "eval/" in line:
            section = "eval"
        if section != "rollout":
            continue

        m = re.search(r"success_rate\s*\|\s*([\d.]+)", line)
        if m:
            cur["succ"] = float(m.group(1))

        m = re.search(r"nut_b_hotstart_alpha\s*\|\s*([\d.]+)", line)
        if m:
            last_hs = float(m.group(1))
            cur["hs"] = last_hs
        elif "succ" in cur:
            cur["hs"] = last_hs

        m = re.search(r"total_timesteps\s*\|\s*([\d.e+]+)", line)
        if m:
            cur["t"] = float(m.group(1))
            if "succ" in cur:
                pts.append(dict(cur))
                cur = {}

    return pts


def parse_dr(logf: Path) -> list[dict]:
    """Parse rollout blocks for DR pipeline view."""
    init_dr = None
    for line in logf.open(errors="ignore"):
        m = re.search(r"\[curriculum\] dr_range init = ([\d.]+)cm", line)
        if m:
            init_dr = float(m.group(1))
    pts: list[dict] = []
    cur: dict = {}
    section: str | None = None
    last_dr = init_dr if init_dr is not None else 0.0

    for line in logf.open(errors="ignore"):
        if "rollout/" in line:
            section = "rollout"
        elif "eval/" in line:
            section = "eval"
        if section != "rollout":
            continue

        m = re.search(r"success_rate\s*\|\s*([\d.]+)", line)
        if m:
            cur["succ"] = float(m.group(1))

        m = re.search(r"dr_range_cm\s*\|\s*([\d.]+)", line)
        if m:
            last_dr = float(m.group(1))
            cur["dr"] = last_dr
        elif "succ" in cur:
            cur["dr"] = last_dr

        m = re.search(r"total_timesteps\s*\|\s*([\d.e+]+)", line)
        if m:
            cur["t"] = float(m.group(1))
            if "succ" in cur:
                pts.append(dict(cur))
                cur = {}

    return pts


def smooth(ys: list[float], w: int = 7) -> list[float]:
    out: list[float] = []
    n = len(ys)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        vals = [v for v in ys[lo:hi] if v == v]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def plot_chain(logf: Path) -> None:
    pts = parse_chain(logf)
    if not pts:
        raise SystemExit(f"no rollout data in {logf}")

    all_t = [d["t"] / 1e6 for d in pts]
    all_s = [d.get("succ", float("nan")) for d in pts]
    all_hs = [d.get("hs", float("nan")) for d in pts]

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax2 = ax1.twinx()

    ax1.plot(all_t, all_s, color="#a5d6a7", linewidth=1.0, alpha=0.55, zorder=3)
    ax1.plot(
        all_t,
        smooth(all_s, 7),
        color="#2e7d32",
        linewidth=2.6,
        label="success_rate (smoothed)",
        zorder=5,
    )
    ax2.plot(
        all_t,
        all_hs,
        color="#1565c0",
        linewidth=1.8,
        linestyle="--",
        label="hot-start alpha",
    )

    ax1.axhline(1.0, color="#c62828", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("training steps (x1e6)")
    ax1.set_ylabel("success_rate (10/10)", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylabel("hot-start alpha  [decay → full approach]", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")
    ax2.set_ylim(0.0, 0.35)
    ax1.set_title("Robot B nut fastening (v24 chain, pre-DR) — success vs hot-start decay")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130)
    print(f"saved {_OUT} ({len(pts)} pts from {logf.name})")


def plot_full_pipeline() -> None:
    all_t: list[float] = []
    all_s: list[float] = []
    all_dr: list[float] = []
    boundaries: list[tuple[float, str]] = []
    offset_m = 0.0

    for run_name, label, span_m in V24_PIPELINE:
        logf = _RUNS / f"{run_name}.log"
        if not logf.exists():
            continue
        pts = parse_dr(logf)
        if not pts:
            continue
        boundaries.append((offset_m, label))
        for d in pts:
            all_t.append(offset_m + d["t"] / 1e6)
            all_s.append(d.get("succ", float("nan")))
            all_dr.append(d.get("dr", 0.0))
        offset_m += span_m

    if not all_t:
        raise SystemExit("no v24 pipeline logs found under runs/")

    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    ax2 = ax1.twinx()
    ax1.plot(all_t, all_s, color="#a5d6a7", linewidth=1.0, alpha=0.55, zorder=3)
    ax1.plot(all_t, smooth(all_s, 7), color="#2e7d32", linewidth=2.6,
             label="success_rate (smoothed)", zorder=5)
    ax2.plot(all_t, all_dr, color="#1565c0", linewidth=1.8, linestyle="--",
             label="dr_range_cm (hub offset)", zorder=4)
    ax1.axhline(1.0, color="#c62828", linestyle="--", linewidth=0.8, alpha=0.5)
    for x0, label in boundaries[1:]:
        ax1.axvline(x0, color="#9e9e9e", linestyle=":", linewidth=0.9, alpha=0.7)
        ax1.text(x0 + 0.06, 0.97, label, fontsize=7.5, color="#616161",
                 va="top", ha="left")
    ax1.set_xlabel("training steps (x1e6)  —  chain → DR B → B2 → B3")
    ax1.set_ylabel("success_rate (10/10)", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylabel("dr_range_cm  [hub DR difficulty]", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")
    ax2.set_ylim(0.0, 5.5)
    ax1.set_title("Robot B nut fastening (v24) — success vs hub DR curriculum")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130)
    print(f"saved {_OUT} ({len(all_t)} pts, full pipeline)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Plot chain + DR stages (legacy combined view).",
    )
    ap.add_argument(
        "log",
        nargs="?",
        default=None,
        help=f"Single log path (default: runs/{RUN}.log).",
    )
    args = ap.parse_args()

    if args.full_pipeline:
        plot_full_pipeline()
        return

    logf = Path(args.log) if args.log else _RUNS / f"{RUN}.log"
    if not logf.exists():
        raise SystemExit(f"no nut-fastening log found: {logf}")
    plot_chain(logf)


if __name__ == "__main__":
    main()
