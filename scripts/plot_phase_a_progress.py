#!/usr/bin/env python3
"""Plot Robot-A Phase-A mount learning curve.

Default (legacy): ``phase1_mount_v2`` → ``docs/phase_a_progress.png``.

Other runs:
  --run ft03   → phase1_mount_v2_ft03 (pre-DR nominal mount, 3M)
  --run v3_dr  → phase1_mount_v3_dr (hub DR 0→5 cm, 2M)

Left axis: rollout success_rate (raw + 7-window MA).
Right axis: mount_radius_tol (m) for v2/ft03; dr_range_cm (cm) for v3_dr.
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
_DOCS = _REPO / "docs"

PRESETS: dict[str, dict] = {
    "v2": {
        "run": "phase1_mount_v2",
        "out": "phase_a_progress.png",
        "title": "Robot A mount (v2) — success vs tightening gate",
        "right": "tol",
    },
    "ft03": {
        "run": "phase1_mount_v2_ft03",
        "out": "phase_a_ft03_progress.png",
        "title": "Robot A mount (ft03, pre-DR) — success vs tightening gate",
        "right": "tol",
    },
    "v3_dr": {
        "run": "phase1_mount_v3_dr",
        "out": "phase_a_v3_dr_progress.png",
        "title": "Robot A mount (v3_dr) — success vs hub DR curriculum",
        "right": "dr",
    },
}


def _init_dr(logf: Path) -> float:
    for line in logf.open(errors="ignore"):
        m = re.search(r"\[curriculum\] dr_range init = ([\d.]+)cm", line)
        if m:
            return float(m.group(1))
    return 0.0


def parse(logf: Path, *, right: str) -> list[dict]:
    """Parse rollout blocks only (skip eval/ to avoid post-eval dips)."""
    last_dr = _init_dr(logf)
    pts: list[dict] = []
    cur: dict = {}
    section: str | None = None
    pats = {
        "tol": r"mount_radius_tol\s*\|\s*([\d.]+)",
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
                if k == "dr":
                    last_dr = cur[k]
        if right == "dr" and "succ" in cur and "dr" not in cur:
            cur["dr"] = last_dr
        if "t" in cur and "succ" in cur:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        choices=tuple(PRESETS),
        default="v2",
        help="Which mount run to plot (default: v2 legacy).",
    )
    args = ap.parse_args()
    preset = PRESETS[args.run]

    logf = _RUNS / f"{preset['run']}.log"
    outf = _DOCS / preset["out"]
    if not logf.exists():
        raise SystemExit(f"no Phase-A log found: {logf}")

    pts = parse(logf, right=preset["right"])
    if not pts:
        raise SystemExit(f"no rollout data in {logf}")

    all_t = [d["t"] / 1e6 for d in pts]
    all_s = [d.get("succ", float("nan")) for d in pts]

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

    if preset["right"] == "dr":
        all_y = [d.get("dr", float("nan")) for d in pts]
        ax2.plot(
            all_t,
            all_y,
            color="#1565c0",
            linewidth=1.8,
            linestyle="--",
            label="dr_range_cm (hub offset)",
        )
        ax2.set_ylabel("dr_range_cm  [hub DR difficulty]", color="#1565c0")
        ax2.set_ylim(0.0, 5.5)
    else:
        all_y = [d.get("tol", float("nan")) for d in pts]
        ax2.plot(
            all_t,
            all_y,
            color="#1565c0",
            linewidth=1.8,
            linestyle="--",
            label="mount_radius_tol (m)",
        )
        ax2.set_ylabel("mount_radius_tol (m)  [tightening]", color="#1565c0")
        ax2.set_ylim(0.0, 0.60)

    ax1.axhline(1.0, color="#c62828", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("training steps (x1e6)")
    ax1.set_ylabel("success_rate", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_ylim(0.0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#1565c0")
    ax1.set_title(preset["title"])
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    fig.tight_layout()
    outf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outf, dpi=130)
    print(f"saved {outf} ({len(pts)} pts from {preset['run']})")


if __name__ == "__main__":
    main()
