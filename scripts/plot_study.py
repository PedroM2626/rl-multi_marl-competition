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


def load() -> dict[str, dict[str, list[tuple[int, float]]]]:
    series: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for summary in sorted(RUNS.glob("*/evaluation/run_summary.json")):
        experiment = summary.parents[1].name.split("_seed")[0]
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for point in payload["curve"]:
            for team, stats in point["per_team"].items():
                series[experiment][payload["paradigms"][team]].append((point["env_steps"], stats["win_rate"]))
    return series


def main() -> None:
    series = load()
    experiments = sorted(series)
    fig, axes = plt.subplots(1, len(experiments), figsize=(7.2 * len(experiments), 5.2), squeeze=False)

    for axis, experiment in zip(axes[0], experiments):
        for index, (paradigm, points) in enumerate(sorted(series[experiment].items())):
            by_seed: dict[int, list] = defaultdict(list)
            for steps, value in points:
                by_seed[steps].append(value)
            xs = sorted(by_seed)
            means = [sum(by_seed[x]) / len(by_seed[x]) for x in xs]
            colour = PALETTE[index % len(PALETTE)]
            for steps, value in sorted(points):
                axis.plot([steps], [value], marker="o", markersize=3.5, linewidth=0, color=colour, alpha=0.35)
            axis.plot(xs, means, marker="o", markersize=5, linewidth=2.4, color=colour,
                      label=f"{paradigm} (mean of {len(by_seed[xs[0]])} seeds)")

        axis.axhline(1 / 3, linestyle="--", linewidth=1, color="grey", alpha=0.8)
        axis.text(axis.get_xlim()[1], 1 / 3, "  chance", va="center", fontsize=8, color="grey")
        axis.set_xlabel("environment steps")
        axis.set_ylabel("held-out greedy win rate")
        axis.set_title(f"{experiment}: {' vs '.join(sorted(series[experiment]))}")
        axis.set_ylim(-0.03, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=9)

    fig.suptitle("Held-out win rate during training, one point per replicate", y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "learning_curves.png"
    fig.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
