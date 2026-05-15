from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_team_metric_series(team_metrics_csv: Path) -> dict[str, list[dict[str, float | int | str]]]:
    if not team_metrics_csv.exists():
        return {}

    grouped_rows: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    with team_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return {}

    rows.sort(key=lambda item: int(item["match_index"]))
    cumulative_wins: dict[str, int] = defaultdict(int)
    cumulative_eliminations: dict[str, int] = defaultdict(int)
    match_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        team_name = str(row["team_name"])
        paradigm = str(row["paradigm"])
        match_index = int(row["match_index"])
        winner = int(row["winner"])
        eliminations = int(row["eliminations"])
        mean_survival_time = float(row["mean_survival_time"])
        shot_accuracy = float(row["shot_accuracy"])

        match_counts[team_name] += 1
        cumulative_wins[team_name] += winner
        cumulative_eliminations[team_name] += eliminations

        grouped_rows[team_name].append(
            {
                "team_name": team_name,
                "paradigm": paradigm,
                "match_index": match_index,
                "win_rate": cumulative_wins[team_name] / match_counts[team_name],
                "cum_eliminations": cumulative_eliminations[team_name],
                "mean_survival_time": mean_survival_time,
                "shot_accuracy": shot_accuracy,
            }
        )

    return dict(grouped_rows)
