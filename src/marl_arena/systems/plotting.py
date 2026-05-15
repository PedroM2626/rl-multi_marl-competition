from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from marl_arena.systems.analytics import load_team_metric_series


def export_metric_dashboard(team_metrics_csv: Path, export_dir: Path) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    series_by_team = load_team_metric_series(team_metrics_csv)
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
        for metric_name, title, axis in metrics:
            axis.plot(
                [float(point["match_index"]) for point in points],
                [float(point[metric_name]) for point in points],
                marker="o",
                linewidth=2,
                label=label,
            )
            axis.set_title(title)
            axis.set_xlabel("Partida")
            axis.grid(True, alpha=0.3)

    for axis in axes.ravel():
        axis.legend(fontsize=8)

    fig.tight_layout()
    dashboard_path = export_dir / "comparative_dashboard.png"
    fig.savefig(dashboard_path)
    plt.close(fig)
    return [dashboard_path]
