from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _load_team_metric_series(team_metrics_csv: Path) -> dict[str, list[dict[str, float | int | str]]]:
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


def export_metric_dashboard(team_metrics_csv: Path, export_dir: Path) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    series_by_team = _load_team_metric_series(team_metrics_csv)
    if not series_by_team:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    metrics = [
        ("win_rate", "Taxa de Vitorias", axes[0, 0]),
        ("cum_eliminations", "Eliminacoes Acumuladas", axes[0, 1]),
        ("mean_survival_time", "Sobrevivencia Media", axes[1, 0]),
        ("shot_accuracy", "Precisao de Disparos", axes[1, 1]),
    ]

    for team_name, points in series_by_team.items():
        label = f"{team_name} ({points[0]['paradigm']})"
        for metric_name, _title, axis in metrics:
            axis.plot(
                [float(point["match_index"]) for point in points],
                [float(point[metric_name]) for point in points],
                marker="o",
                linewidth=2,
                label=label,
            )
            axis.set_xlabel("Partida")
            axis.grid(True, alpha=0.3)

    for metric_name, title, axis in metrics:
        axis.set_title(title)

    for axis in axes.ravel():
        axis.legend(fontsize=8)

    fig.tight_layout()
    dashboard_path = export_dir / "comparative_dashboard.png"
    fig.savefig(dashboard_path)
    plt.close(fig)
    return [dashboard_path]
