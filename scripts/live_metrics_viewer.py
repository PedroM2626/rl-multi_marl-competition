from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import EXPORTS_DIR, METRICS_DIR
from marl_arena.systems.analytics import load_team_metric_series


TEAM_METRICS_CSV = METRICS_DIR / "team_match_metrics.csv"
EXPORT_PATH = EXPORTS_DIR / "live_metrics_snapshot.png"


def draw_on_axes(fig: plt.Figure, axes_grid) -> None:
    axes = axes_grid.ravel()
    for axis in axes:
        axis.clear()
    series_by_team = load_team_metric_series(TEAM_METRICS_CSV)
    if not series_by_team:
        for axis in axes:
            axis.set_title("Aguardando dados")
        fig.tight_layout()
        fig.savefig(EXPORT_PATH)
        return

    metrics = [
        ("win_rate", "Taxa de Vitorias", axes_grid[0, 0]),
        ("cum_eliminations", "Eliminacoes Acumuladas", axes_grid[0, 1]),
        ("mean_survival_time", "Sobrevivencia Media", axes_grid[1, 0]),
        ("shot_accuracy", "Precisao de Disparos", axes_grid[1, 1]),
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

    for axis in axes:
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(EXPORT_PATH)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    draw_on_axes(fig, axes)

    def update(_: int) -> None:
        draw_on_axes(fig, axes)

    FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
