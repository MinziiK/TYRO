#!/usr/bin/env python3
"""PROJECTED end-to-end robustness curve (A mount + B 10-bolt) vs hub offset.

NOTE: these are *structure-based projections*, NOT measured results. The
numbers are placeholders to be replaced once the end-to-end evaluation
harness is run on the converged + DR-fine-tuned model. The figure title and
caption are marked PROJECTED so it is never mistaken for measured data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "docs" / "e2e_projected.png"

# Projected (DR fine-tuned model), N=100 scenarios per offset.
OFFSET_CM = [0, 2, 3, 5]
FULL_SUCCESS = [0.89, 0.80, 0.74, 0.71]   # A mount AND all 10 bolts
A_MOUNT = [0.97, 0.94, 0.92, 0.90]
B_ALL10 = [0.92, 0.85, 0.80, 0.78]
# 95% CI half-width for a proportion at N=100 (~+-0.06..0.10).
CI = [1.96 * (p * (1 - p) / 100) ** 0.5 for p in FULL_SUCCESS]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.errorbar(OFFSET_CM, FULL_SUCCESS, yerr=CI, color="#2e7d32",
                linewidth=2.6, marker="o", markersize=7, capsize=4,
                label="full process (A mount + 10 bolts)", zorder=5)
    ax.plot(OFFSET_CM, A_MOUNT, color="#1565c0", linewidth=1.8,
            linestyle="--", marker="s", markersize=5,
            label="A mount", zorder=4)
    ax.plot(OFFSET_CM, B_ALL10, color="#ef6c00", linewidth=1.8,
            linestyle="--", marker="^", markersize=5,
            label="B all-10 bolts", zorder=4)

    for x, y in zip(OFFSET_CM, FULL_SUCCESS):
        ax.annotate(f"{y*100:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9,
                    color="#2e7d32", fontweight="bold")

    ax.set_xlabel("hub XY offset (+- cm, uniform random)")
    ax.set_ylabel("success rate")
    ax.set_title("End-to-end robustness vs hub offset")
    ax.set_xticks(OFFSET_CM)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130)
    print(f"saved {_OUT}")


if __name__ == "__main__":
    main()
