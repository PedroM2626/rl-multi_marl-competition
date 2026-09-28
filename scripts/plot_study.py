"""Render the replicated study's held-out learning curves into results/study/learning_curves.png.

Each point comes from a greedy, fixed-variant, no-domain-randomisation evaluation run at that step
count during training, so the y axis is held-out win rate rather than the training-time running average.
The dashed line is the 1/3 three-way chance level.

Usage:  python scripts/plot_study.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "results" / "study"

PALETTE = ["#4c72b0", "#55a868", "#c44e52"]


def load() -> dict[str, dict[str, dict[float, list[float]]]]:
    """series[experiment][paradigm][fraction] = one held-out win rate per replicate.

    Curve points fire at a match boundary, so each replicate's absolute step count differs slightly;
    grouping by the nominal fraction is what makes the mean across seeds meaningful.
    """
    series: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for summary in sorted(RUNS.glob("*/evaluation/run_summary.json")):
        experiment = summary.parents[1].name.split("_seed")[0]
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for point in payload["curve"]:
            for team, stats in point["per_team"].items():
                series[experiment][payload["paradigms"][team]][point["fraction"]].append(stats["win_rate"])
    return series


def main() -> None:
    series = load()
    experiments = sorted(series)
    fig, axes = plt.subplots(1, len(experiments), figsize=(7.2 * len(experiments), 5.2), squeeze=False)

    for axis, experiment in zip(axes[0], experiments):
        for index, (paradigm, by_fraction) in enumerate(sorted(series[experiment].items())):
            fractions = sorted(by_fraction)
            seeds = max(len(by_fraction[f]) for f in fractions)
            colour = PALETTE[index % len(PALETTE)]
            for fraction in fractions:
                for value in by_fraction[fraction]:
                    axis.plot([fraction], [value], marker="o", markersize=3.5, linewidth=0,
                              color=colour, alpha=0.35)
            means = [sum(by_fraction[f]) / len(by_fraction[f]) for f in fractions]
            axis.plot(fractions, means, marker="o", markersize=5, linewidth=2.4, color=colour,
                      label=f"{paradigm} (mean of {seeds} seeds)")

        axis.axhline(1 / 3, linestyle="--", linewidth=1, color="grey", alpha=0.8)
        axis.set_xlabel("fraction of the training budget")
        axis.set_xlim(0.15, 1.1)
        axis.set_xticks([0.25, 0.5, 0.75, 1.0])
        axis.set_xticklabels(["25 %", "50 %", "75 %", "100 %"])
        axis.set_ylabel("held-out greedy win rate")
        axis.set_title(f"{experiment}: {' vs '.join(sorted(series[experiment]))}")
        axis.set_ylim(-0.03, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=9, loc="upper left")

    fig.suptitle("Held-out win rate during training, one point per replicate", y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "learning_curves.png"
    fig.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
