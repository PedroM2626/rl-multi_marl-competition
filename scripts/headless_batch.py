from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG
from marl_arena.systems.metrics import MetricsStore
from marl_arena.systems.simulation import ArenaSimulation


def main() -> None:
    metrics = MetricsStore()
    simulation = ArenaSimulation()
    total_matches = CONFIG.headless_matches
    for _ in range(total_matches):
        while True:
            finished = simulation.step(0.1)
            if finished:
                break
        result = simulation.finish_match()
        metrics.record_match(result, simulation.cumulative_metrics)
        print(f"Partida {result.team_rows[0]['match_index']} concluida | vencedora: {result.winner_team}")
        simulation.reset_match()
    print("Execucao headless finalizada.")


if __name__ == "__main__":
    main()
